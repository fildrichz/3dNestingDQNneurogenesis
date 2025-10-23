
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def to_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.as_tensor(x, device=device)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, max_actions: int, action_feat_dim: int, n_step:int=1, gamma:float=0.99):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_actions = max_actions
        self.action_feat_dim = action_feat_dim
        self.n_step = max(1, n_step)
        self.gamma = gamma

        self.ptr = 0
        self.size = 0

        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a_idx = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s_next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        A, D = max_actions, action_feat_dim
        self.curr_feats = np.zeros((capacity, A, D), dtype=np.float32)
        self.curr_mask  = np.zeros((capacity, A), dtype=np.float32)
        self.next_feats = np.zeros((capacity, A, D), dtype=np.float32)
        self.next_mask  = np.zeros((capacity, A), dtype=np.float32)

        self.n_step_buffer = deque(maxlen=self.n_step)

    def _n_step_push(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return None
        R, s, a_idx, currF, currM = 0.0, None, None, None, None
        for i, (si, ai, ri, sni, di, cF, cM, nF, nM) in enumerate(self.n_step_buffer):
            if i == 0:
                s = si; a_idx = ai; currF = cF; currM = cM
            R = R + (self.gamma ** i) * ri
            if di:
                s_next, done, nextF, nextM = sni, di, nF, nM
                break
        else:
            s_next, done, nextF, nextM = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4], self.n_step_buffer[-1][7], self.n_step_buffer[-1][8]
        self.n_step_buffer.clear()
        return (s, a_idx, R, s_next, float(done), currF, currM, nextF, nextM)

    def push(self, s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask):
        if self.n_step > 1:
            agg = self._n_step_push((s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask))
            if agg is None:
                return
            s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask = agg

        i = self.ptr
        self.s[i] = s
        self.a_idx[i] = -1 if a_idx is None else int(a_idx)
        self.r[i] = r
        self.s_next[i] = s_next
        self.done[i] = float(done)
        self.curr_feats[i] = curr_action_feats
        self.curr_mask[i]  = curr_mask
        self.next_feats[i] = next_action_feats
        self.next_mask[i]  = next_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size:int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s = self.s[idxs],
            a_idx = self.a_idx[idxs],
            r = self.r[idxs],
            s_next = self.s_next[idxs],
            done = self.done[idxs],
            curr_feats = self.curr_feats[idxs],
            curr_mask = self.curr_mask[idxs],
            next_feats = self.next_feats[idxs],
            next_mask = self.next_mask[idxs],
        )

class MLP(nn.Module):
    def __init__(self, in_dim:int, hidden:int=256, out_dim:int=256, layers:int=2, activation=nn.ReLU):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(max(0,layers-1)):
            mods += [nn.Linear(d, hidden), activation()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

class SimpleHead(nn.Module):
    def __init__(self, in_dim:int, hidden:int=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, z):
        return self.net(z)

class QNetwork(nn.Module):
    def __init__(self, obs_dim:int, action_feat_dim:int, hidden:int=256, enc_layers:int=2, head_hidden:int=256):
        super().__init__()
        self.state_enc = MLP(obs_dim, hidden=hidden, out_dim=hidden, layers=enc_layers)
        self.action_enc = MLP(action_feat_dim, hidden=hidden, out_dim=hidden, layers=enc_layers)
        self.head = SimpleHead(2*hidden, hidden=head_hidden)

    def forward(self, s:torch.Tensor, action_feats:torch.Tensor) -> torch.Tensor:
        B, A, D = action_feats.shape
        zs = self.state_enc(s)                               # (B, H)
        za = self.action_enc(action_feats.view(B*A, D))      # (B*A, H)
        zs_rep = zs.unsqueeze(1).repeat(1, A, 1).view(B*A, -1)
        z = torch.cat([zs_rep, za], dim=-1)                  # (B*A, 2H)
        q = self.head(z).view(B, A)                          # (B, A)
        return q

from dataclasses import dataclass

@dataclass
class DQNConfig:
    obs_dim: int
    action_feat_dim: int
    max_actions: int
    gamma: float = 0.985
    lr: float = 2e-4
    batch_size: int = 64
    grad_clip: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    buffer_size: int = 200_000
    n_step: int = 1
    target_update_interval: int = 1_000
    tau: float = 0.0
    hidden: int = 256
    enc_layers: int = 2
    head_hidden: int = 256
    device: str = "cpu"
    double_dqn: bool = True

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = QNetwork(cfg.obs_dim, cfg.action_feat_dim, hidden=cfg.hidden, enc_layers=cfg.enc_layers, head_hidden=cfg.head_hidden).to(self.device)
        self.q_target = QNetwork(cfg.obs_dim, cfg.action_feat_dim, hidden=cfg.hidden, enc_layers=cfg.enc_layers, head_hidden=cfg.head_hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.obs_dim, cfg.max_actions, cfg.action_feat_dim, n_step=cfg.n_step, gamma=cfg.gamma)
        self.step_count = 0
        self._eps = cfg.eps_start

    def epsilon(self):
        frac = min(1.0, self.step_count / max(1, self.cfg.eps_decay_steps))
        self._eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac
        return self._eps

    def select_action(self, obs: np.ndarray, action_feats: np.ndarray, mask: np.ndarray) -> int | None:
        valid = np.where(mask > 0.5)[0]
        if valid.size == 0:
            return None
        eps = self.epsilon()
        self.step_count += 1
        if np.random.rand() < eps:
            return int(np.random.choice(valid))
        obs_t = to_torch(obs, self.device).float().unsqueeze(0)
        feats_t = to_torch(action_feats, self.device).float().unsqueeze(0)
        with torch.no_grad():
            q = self.q(obs_t, feats_t)[0].cpu().numpy()
        q[mask < 0.5] = -1e9
        return int(np.argmax(q))

    def store(self, s, a_idx, r, s_next, done, *, curr_action_feats, curr_mask, next_action_feats, next_mask):
        self.buffer.push(s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask)

    def train_step(self) -> float | None:
        if self.buffer.size < self.cfg.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        s = to_torch(batch['s'], self.device).float()
        a_idx = to_torch(batch['a_idx'], self.device).long()
        r = to_torch(batch['r'], self.device).float()
        s_next = to_torch(batch['s_next'], self.device).float()
        done = to_torch(batch['done'], self.device).float()
        curr_feats = to_torch(batch['curr_feats'], self.device).float()
        curr_mask  = to_torch(batch['curr_mask'], self.device).float()
        next_feats = to_torch(batch['next_feats'], self.device).float()
        next_mask  = to_torch(batch['next_mask'], self.device).float()

        q_all = self.q(s, curr_feats)  # (B, A)
        a_idx_clamped = torch.clamp(a_idx, 0, self.cfg.max_actions-1)
        q_sa = q_all.gather(1, a_idx_clamped.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                q_next_online = self.q(s_next, next_feats)
                q_next_online[next_mask < 0.5] = -1e9
                next_a = torch.argmax(q_next_online, dim=1, keepdim=True)
                q_next_target = self.q_target(s_next, next_feats)
                q_next_target[next_mask < 0.5] = -1e9
                max_next = q_next_target.gather(1, next_a).squeeze(1)
            else:
                q_next = self.q_target(s_next, next_feats)
                q_next[next_mask < 0.5] = -1e9
                max_next = torch.max(q_next, dim=1)[0]
            target = r + (1.0 - done) * self.cfg.gamma * max_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(); loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        # Target updates
        if self.cfg.tau and self.cfg.tau > 0.0:
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                    pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
        elif self.step_count % self.cfg.target_update_interval == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.detach().cpu().item())
