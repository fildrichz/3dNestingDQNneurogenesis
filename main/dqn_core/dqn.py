from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# -------- utils --------
def to_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.as_tensor(x, device=device)

# -------- replay (with correct n-step sliding window) --------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, max_actions: int, action_feat_dim: int,
                 n_step:int=1, gamma:float=0.99, min_buffer:int=10_000):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_actions = max_actions
        self.action_feat_dim = action_feat_dim
        self.n_step = max(1, n_step)
        self.gamma = gamma
        self.min_buffer = min_buffer

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

        self._deque = deque(maxlen=self.n_step)

    def __len__(self):
        return self.size

    def _commit(self, transition):
        s, a_idx, R, s_next, done, cF, cM, nF, nM = transition
        i = self.ptr
        self.s[i] = s
        self.a_idx[i] = -1 if a_idx is None else int(a_idx)
        self.r[i] = R
        self.s_next[i] = s_next
        self.done[i] = float(done)
        self.curr_feats[i] = cF; self.curr_mask[i]  = cM
        self.next_feats[i] = nF; self.next_mask[i]  = nM
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _aggregate(self):
        R = 0.0
        done = 0.0
        for i, (s, a_idx, r, s_next, d, cF, cM, nF, nM) in enumerate(self._deque):
            R += (self.gamma ** i) * r
            if d:
                done = 1.0
                s_next_final, nF_final, nM_final = s_next, nF, nM
                break
        else:
            s_next_final, nF_final, nM_final = self._deque[-1][3], self._deque[-1][7], self._deque[-1][8]
        s0, a0, _, _, _, cF0, cM0, _, _ = self._deque[0]
        return (s0, a0, R, s_next_final, done, cF0, cM0, nF_final, nM_final)

    def push(self, s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask):
        t = (np.array(s, np.float32), a_idx, float(r),
             np.array(s_next, np.float32), bool(done),
             np.array(curr_action_feats, np.float32), np.array(curr_mask, np.float32),
             np.array(next_action_feats, np.float32), np.array(next_mask, np.float32))
        if self.n_step == 1:
            self._commit((t[0], t[1], t[2], t[3], float(t[4]), t[5], t[6], t[7], t[8]))
            return

        self._deque.append(t)
        if len(self._deque) == self.n_step:
            self._commit(self._aggregate())
            self._deque.popleft()

        if done:
            while len(self._deque) > 0:
                self._commit(self._aggregate())
                self._deque.popleft()

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

# -------- networks --------
class MLP(nn.Module):
    def __init__(self, in_dim:int, hidden:int=256, out_dim:int=256, layers:int=2):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(max(0, layers-1)):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x): 
        return self.net(x)

class QNetwork(nn.Module):
    def __init__(self, obs_dim:int, action_feat_dim:int, hidden:int=256, enc_layers:int=2, head_hidden:int=256):
        super().__init__()
        self.state_enc = MLP(obs_dim, hidden=hidden, out_dim=hidden, layers=enc_layers)
        self.action_enc = MLP(action_feat_dim, hidden=hidden, out_dim=hidden, layers=enc_layers)
        self.head = nn.Sequential(nn.Linear(2*hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, s:torch.Tensor, action_feats:torch.Tensor) -> torch.Tensor:
        B, A, D = action_feats.shape
        zs = self.state_enc(s)
        za = self.action_enc(action_feats.view(B*A, D))
        zs_rep = zs.unsqueeze(1).repeat(1, A, 1).view(B*A, -1)
        z = torch.cat([zs_rep, za], dim=-1)
        q = self.head(z).view(B, A)
        return q

@dataclass
class DQNConfig:
    obs_dim: int
    action_feat_dim: int
    max_actions: int
    gamma: float = 0.992
    lr: float = 1e-3
    batch_size: int = 128
    grad_clip: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    buffer_size: int = 300_000
    n_step: int = 3
    min_buffer: int = 8_000
    # target updates per Double DQN paper
    target_update_interval: int = 30_000   # hard copy every 30k steps
    tau: float = 0.0                       # τ=0 ⇒ hard update only
    # nets
    hidden: int = 256
    enc_layers: int = 2
    head_hidden: int = 256
    device: str = "cpu"
    double_dqn: bool = True

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = QNetwork(cfg.obs_dim, cfg.action_feat_dim, hidden=cfg.hidden,
                          enc_layers=cfg.enc_layers, head_hidden=cfg.head_hidden).to(self.device)
        self.q_target = QNetwork(cfg.obs_dim, cfg.action_feat_dim, hidden=cfg.hidden,
                                 enc_layers=cfg.enc_layers, head_hidden=cfg.head_hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.obs_dim, cfg.max_actions,
                                   cfg.action_feat_dim, n_step=cfg.n_step, gamma=cfg.gamma,
                                   min_buffer=cfg.min_buffer)
        self.step_count = 0

    def epsilon(self):
        frac = min(1.0, self.step_count / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, action_feats: np.ndarray, mask: np.ndarray) -> int | None:
        valid = np.where(mask > 0.5)[0]
        if valid.size == 0:
            return None
        self.step_count += 1
        if np.random.rand() < self.epsilon():
            return int(np.random.choice(valid))
        obs_t = to_torch(obs, self.device).float().unsqueeze(0)
        feats_t = to_torch(action_feats, self.device).float().unsqueeze(0)
        q = self.q(obs_t, feats_t)[0]
        q = q.masked_fill(to_torch(mask, self.device) <= 0.5, -1e9)
        return int(torch.argmax(q).item())

    def store(self, s, a_idx, r, s_next, done, *, curr_action_feats, curr_mask, next_action_feats, next_mask):
        self.buffer.push(s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask)

    def _update_target(self):
        # Double DQN paper ⇒ hard copy every target_update_interval steps
        if self.cfg.tau <= 0.0:
            if self.step_count % self.cfg.target_update_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())
        else:
            # optional Polyak averaging
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                    pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def train_step(self) -> Optional[float]:
        if self.buffer.size < max(self.cfg.batch_size, self.cfg.min_buffer):
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

        q_all = self.q(s, curr_feats)
        q_all = q_all.masked_fill(curr_mask < 0.5, -1e9)
        valid = (a_idx >= 0)
        if not valid.any():
            return None
        q_sa = q_all[valid].gather(1, a_idx[valid].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            has_next_valid = (next_mask.max(dim=1).values > 0.5).float()
            if self.cfg.double_dqn:
                q_next_online = self.q(s_next, next_feats).masked_fill(next_mask < 0.5, -1e9)
                next_a = torch.argmax(q_next_online, dim=1, keepdim=True)
                q_next_target = self.q_target(s_next, next_feats).masked_fill(next_mask < 0.5, -1e9)
                max_next = q_next_target.gather(1, next_a).squeeze(1)
            else:
                q_next = self.q_target(s_next, next_feats).masked_fill(next_mask < 0.5, -1e9)
                max_next = torch.max(q_next, dim=1)[0]
            bootstrap = (1.0 - done) * has_next_valid
            target_all = r + bootstrap * self.cfg.gamma * max_next
            target = target_all[valid]

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self._update_target()
        return float(loss.item())
