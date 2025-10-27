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

        # ✅ FIX: Don't use maxlen - we manually control the size with popleft()
        self.n_step_buffer = deque()

    def _n_step_push(self, transition):
        """
        ✅ FIXED: Properly implements sliding window n-step learning.
        
        Key changes:
        1. Uses popleft() instead of clear() to maintain sliding window
        2. Computes n-step return from OLDEST transition
        3. Generates one sample per timestep (after warmup)
        
        Returns:
            n-step transition tuple or None if buffer not full
        """
        self.n_step_buffer.append(transition)
        
        # Wait until we have n transitions
        if len(self.n_step_buffer) < self.n_step:
            return None
        
        # ✅ FIX: Extract data from OLDEST transition (index 0)
        oldest = self.n_step_buffer[0]
        s = oldest[0]
        a_idx = oldest[1]
        currF = oldest[5]
        currM = oldest[6]
        
        # Compute n-step return
        R = 0.0
        gamma_power = 1.0
        
        # Check for terminal states within n-step window
        for i, (si, ai, ri, sni, di, cF, cM, nF, nM) in enumerate(self.n_step_buffer):
            R += gamma_power * ri
            gamma_power *= self.gamma
            
            if di:
                # Episode terminated - use this state as final
                # ✅ FIX: Use popleft() to maintain sliding window
                self.n_step_buffer.popleft()
                return (s, a_idx, R, sni, 1.0, currF, currM, nF, nM)
        
        # No termination in window - bootstrap from last state
        last = self.n_step_buffer[-1]
        s_next = last[3]
        done = last[4]
        nextF = last[7]
        nextM = last[8]
        
        # ✅ CRITICAL FIX: Remove only oldest, maintain sliding window
        self.n_step_buffer.popleft()  # Not clear()!
        
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
    tau: float = 0.0  # 0 = hard update, >0 = soft update
    hidden: int = 256
    enc_layers: int = 2
    head_hidden: int = 256
    device: str = "cpu"
    double_dqn: bool = True
    warmup_steps: int = 1000  # Steps before training starts

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Initialize networks
        self.q = QNetwork(
            cfg.obs_dim, cfg.action_feat_dim, 
            hidden=cfg.hidden, enc_layers=cfg.enc_layers, 
            head_hidden=cfg.head_hidden
        ).to(self.device)
        
        self.q_target = QNetwork(
            cfg.obs_dim, cfg.action_feat_dim, 
            hidden=cfg.hidden, enc_layers=cfg.enc_layers, 
            head_hidden=cfg.head_hidden
        ).to(self.device)
        
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()  # Target network always in eval mode
        
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        
        self.buffer = ReplayBuffer(
            cfg.buffer_size, cfg.obs_dim, cfg.max_actions, 
            cfg.action_feat_dim, n_step=cfg.n_step, gamma=cfg.gamma
        )
        
        # Step counters
        self.env_steps = 0  # Total environment interactions
        self.training_steps = 0  # Total training updates
        self._eps = cfg.eps_start

    def epsilon(self) -> float:
        """Epsilon for exploration, decays based on environment steps (not training steps)"""
        # ✅ IMPROVEMENT: Use env_steps for more consistent exploration schedule
        frac = min(1.0, self.env_steps / max(1, self.cfg.eps_decay_steps))
        self._eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac
        return self._eps

    def select_action(self, obs: np.ndarray, action_feats: np.ndarray, mask: np.ndarray) -> int | None:
        """Select action using epsilon-greedy policy"""
        valid = np.where(mask > 0.5)[0]
        if valid.size == 0:
            return None
        
        eps = self.epsilon()
        self.env_steps += 1
        
        # Epsilon-greedy
        if np.random.rand() < eps:
            return int(np.random.choice(valid))
        
        # Greedy action
        obs_t = to_torch(obs, self.device).float().unsqueeze(0)
        feats_t = to_torch(action_feats, self.device).float().unsqueeze(0)
        
        with torch.no_grad():
            q = self.q(obs_t, feats_t)[0].cpu().numpy()
        
        # Mask invalid actions
        q[mask < 0.5] = -np.inf  # Use -inf for safety
        return int(np.argmax(q))

    def store(self, s, a_idx, r, s_next, done, *, curr_action_feats, curr_mask, next_action_feats, next_mask):
        """Store transition in replay buffer"""
        self.buffer.push(s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask)

    def train_step(self) -> float | None:
        """Perform one training step. Returns loss or None if not enough samples."""
        # Wait for warmup
        if self.buffer.size < self.cfg.warmup_steps:
            return None
        
        if self.buffer.size < self.cfg.batch_size:
            return None
        
        # Sample batch
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

        # Compute Q(s,a) for taken actions
        q_all = self.q(s, curr_feats)
        
        # Only compute loss for valid actions (where a_idx >= 0)
        valid = (a_idx >= 0)
        if not valid.any():
            return None
        
        q_sa = q_all[valid].gather(1, a_idx[valid].unsqueeze(1)).squeeze(1)

        # Compute target values
        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double DQN implementation (matches van Hasselt et al. 2016)
                # Use online network for action selection
                q_next_online = self.q(s_next, next_feats)
                
                # Better masking strategy to avoid gradient issues
                # Mask before argmax to select best valid action
                q_next_online_masked = torch.where(
                    next_mask > 0.5,
                    q_next_online,
                    torch.tensor(float('-inf'), device=self.device)
                )
                next_a = torch.argmax(q_next_online_masked, dim=1, keepdim=True)
                
                # Use target network for evaluation
                q_next_target = self.q_target(s_next, next_feats)
                max_next = q_next_target.gather(1, next_a).squeeze(1)
                
                # Zero out if selected action was invalid (shouldn't happen, but safe)
                action_was_valid = next_mask.gather(1, next_a).squeeze(1) > 0.5
                max_next = torch.where(action_was_valid, max_next, torch.zeros_like(max_next))
                
            else:
                # Standard DQN
                q_next = self.q_target(s_next, next_feats)
                q_next_masked = torch.where(
                    next_mask > 0.5,
                    q_next,
                    torch.tensor(float('-inf'), device=self.device)
                )
                max_next = torch.max(q_next_masked, dim=1)[0]

            target_all = r + (1.0 - done) * self.cfg.gamma * max_next

        target = target_all[valid]

        # Compute loss
        loss = F.smooth_l1_loss(q_sa, target)
        
        # Optimize
        self.opt.zero_grad()
        loss.backward()
        
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        
        self.opt.step()
        
        # Increment training steps
        self.training_steps += 1

        # Update target network
        if self.training_steps % self.cfg.target_update_interval == 0:
            self._update_target_network()
        
        return loss.item()

    def _update_target_network(self):
        """Update target network (hard or soft update)"""
        if self.cfg.tau > 0:
            # Soft update: θ_target = τ*θ + (1-τ)*θ_target
            for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
                target_param.data.copy_(
                    self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
                )
        else:
            # Hard update: θ_target = θ
            self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_state_dict': self.q.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'env_steps': self.env_steps,
            'training_steps': self.training_steps,
            'epsilon': self._eps,
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q.load_state_dict(checkpoint['q_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.env_steps = checkpoint['env_steps']
        self.training_steps = checkpoint['training_steps']
        self._eps = checkpoint['epsilon']