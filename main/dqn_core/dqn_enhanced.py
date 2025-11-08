"""
Enhanced DQN with:
1. HeightmapCNN - processes 7x7 patches around each placement position
2. Transformer attention - allows actions to reason about each other
"""
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
    """Enhanced replay buffer that stores heightmap patches"""
    def __init__(self, capacity: int, obs_dim: int, max_actions: int, action_feat_dim: int, 
                 patch_size: int = 7, n_step:int=1, gamma:float=0.99):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_actions = max_actions
        self.action_feat_dim = action_feat_dim
        self.patch_size = patch_size
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
        
        # NEW: Store heightmap patches
        self.curr_patches = np.zeros((capacity, A, patch_size, patch_size), dtype=np.float32)
        self.next_patches = np.zeros((capacity, A, patch_size, patch_size), dtype=np.float32)

        self.n_step_buffer = deque()

    def _n_step_push(self, transition):
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_step:
            return None
        
        oldest = self.n_step_buffer[0]
        s = oldest[0]
        a_idx = oldest[1]
        currF = oldest[5]
        currM = oldest[6]
        currP = oldest[7]  # Current patches
        
        R = 0.0
        gamma_power = 1.0
        
        for i, (si, ai, ri, sni, di, cF, cM, cP, nF, nM, nP) in enumerate(self.n_step_buffer):
            R += gamma_power * ri
            gamma_power *= self.gamma
            
            if di:
                self.n_step_buffer.popleft()
                return (s, a_idx, R, sni, 1.0, currF, currM, currP, nF, nM, nP)
        
        last = self.n_step_buffer[-1]
        s_next = last[3]
        done = last[4]
        nextF = last[8]
        nextM = last[9]
        nextP = last[10]
        
        self.n_step_buffer.popleft()
        
        return (s, a_idx, R, s_next, float(done), currF, currM, currP, nextF, nextM, nextP)

    def push(self, s, a_idx, r, s_next, done, curr_action_feats, curr_mask, curr_patches,
             next_action_feats, next_mask, next_patches):
        if self.n_step > 1:
            agg = self._n_step_push((s, a_idx, r, s_next, done, 
                                    curr_action_feats, curr_mask, curr_patches,
                                    next_action_feats, next_mask, next_patches))
            if agg is None:
                return
            s, a_idx, r, s_next, done, curr_action_feats, curr_mask, curr_patches, next_action_feats, next_mask, next_patches = agg

        i = self.ptr
        self.s[i] = s
        self.a_idx[i] = -1 if a_idx is None else int(a_idx)
        self.r[i] = r
        self.s_next[i] = s_next
        self.done[i] = float(done)
        self.curr_feats[i] = curr_action_feats
        self.curr_mask[i]  = curr_mask
        self.curr_patches[i] = curr_patches
        self.next_feats[i] = next_action_feats
        self.next_mask[i]  = next_mask
        self.next_patches[i] = next_patches

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
            curr_patches = self.curr_patches[idxs],
            next_feats = self.next_feats[idxs],
            next_mask = self.next_mask[idxs],
            next_patches = self.next_patches[idxs],
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

class HeightmapCNN(nn.Module):
    """CNN encoder for heightmap patches around placement positions"""
    def __init__(self, patch_size:int=7, out_dim:int=64):
        super().__init__()
        self.patch_size = patch_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU()
        )
    
    def forward(self, patches):
        # patches: (B, 1, patch_size, patch_size)
        return self.cnn(patches)

class SimpleHead(nn.Module):
    def __init__(self, in_dim:int, hidden:int=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, z):
        return self.net(z)

class QNetworkEnhanced(nn.Module):
    """Enhanced Q-Network with heightmap CNN and Transformer attention"""
    def __init__(self, obs_dim:int, action_feat_dim:int, hidden:int=256, enc_layers:int=2, 
                 head_hidden:int=256, heightmap_patch_size:int=7, use_attention:bool=True):
        super().__init__()
        self.patch_size = heightmap_patch_size
        self.use_attention = use_attention
        
        # State encoder
        self.state_enc = MLP(obs_dim, hidden=hidden, out_dim=hidden, layers=enc_layers)
        
        # Heightmap CNN
        self.heightmap_cnn = HeightmapCNN(patch_size=heightmap_patch_size, out_dim=64)
        
        # Action encoder (now takes action_feats + heightmap embedding)
        self.action_enc = MLP(action_feat_dim + 64, hidden=hidden, out_dim=hidden, layers=enc_layers)
        
        # Transformer for action relationships
        if use_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, 
                nhead=4, 
                dim_feedforward=hidden*2, 
                dropout=0.1,
                batch_first=True
            )
            self.action_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.action_transformer = None
        
        self.head = SimpleHead(2*hidden, hidden=head_hidden)

    def forward(self, s:torch.Tensor, action_feats:torch.Tensor, 
                heightmap_patches:torch.Tensor, action_mask:torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B, obs_dim) - state observations
            action_feats: (B, A, action_feat_dim) - action features
            heightmap_patches: (B, A, patch_size, patch_size) - heightmap patches for each action
            action_mask: (B, A) - 1 for valid actions, 0 for padding
        
        Returns:
            q: (B, A) - Q-values for each action
        """
        B, A = action_feats.shape[:2]
        
        # Encode state
        zs = self.state_enc(s)  # (B, hidden)
        
        # Process heightmap patches through CNN
        h_patches = heightmap_patches.view(B*A, 1, self.patch_size, self.patch_size)
        zh = self.heightmap_cnn(h_patches)  # (B*A, 64)
        
        # Combine action features with heightmap embeddings
        action_feats_flat = action_feats.view(B*A, -1)
        combined = torch.cat([action_feats_flat, zh], dim=-1)  # (B*A, action_feat_dim + 64)
        
        # Encode actions
        za = self.action_enc(combined)  # (B*A, hidden)
        za = za.view(B, A, -1)  # (B, A, hidden)
        
        # Apply transformer attention if enabled
        if self.action_transformer is not None:
            # Create attention mask (True = ignore)
            attn_mask = (action_mask < 0.5)  # (B, A)
            # Apply transformer - actions attend to each other
            za = self.action_transformer(za, src_key_padding_mask=attn_mask)  # (B, A, hidden)
        
        # Score each action
        za_flat = za.view(B*A, -1)
        zs_rep = zs.unsqueeze(1).repeat(1, A, 1).view(B*A, -1)
        z = torch.cat([zs_rep, za_flat], dim=-1)  # (B*A, 2*hidden)
        q = self.head(z).view(B, A)  # (B, A)
        
        return q

@dataclass
class DQNConfigEnhanced:
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
    heightmap_patch_size: int = 7
    use_attention: bool = True
    device: str = "cpu"
    double_dqn: bool = True
    warmup_steps: int = 1000

class DQNAgentEnhanced:
    def __init__(self, cfg: DQNConfigEnhanced):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Initialize networks
        self.q = QNetworkEnhanced(
            cfg.obs_dim, cfg.action_feat_dim, 
            hidden=cfg.hidden, enc_layers=cfg.enc_layers, 
            head_hidden=cfg.head_hidden,
            heightmap_patch_size=cfg.heightmap_patch_size,
            use_attention=cfg.use_attention
        ).to(self.device)
        
        self.q_target = QNetworkEnhanced(
            cfg.obs_dim, cfg.action_feat_dim, 
            hidden=cfg.hidden, enc_layers=cfg.enc_layers, 
            head_hidden=cfg.head_hidden,
            heightmap_patch_size=cfg.heightmap_patch_size,
            use_attention=cfg.use_attention
        ).to(self.device)
        
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        
        self.buffer = ReplayBuffer(
            cfg.buffer_size, cfg.obs_dim, cfg.max_actions, 
            cfg.action_feat_dim, patch_size=cfg.heightmap_patch_size,
            n_step=cfg.n_step, gamma=cfg.gamma
        )
        
        self.env_steps = 0
        self.training_steps = 0
        self._eps = cfg.eps_start

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.cfg.eps_decay_steps))
        self._eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac
        return self._eps

    def select_action(self, obs: np.ndarray, action_feats: np.ndarray, 
                     heightmap_patches: np.ndarray, mask: np.ndarray) -> int | None:
        """Select action using epsilon-greedy policy"""
        valid = np.where(mask > 0.5)[0]
        if valid.size == 0:
            return None
        
        eps = self.epsilon()
        self.env_steps += 1
        
        if np.random.rand() < eps:
            return int(np.random.choice(valid))
        
        # Greedy action
        obs_t = to_torch(obs, self.device).float().unsqueeze(0)
        feats_t = to_torch(action_feats, self.device).float().unsqueeze(0)
        patches_t = to_torch(heightmap_patches, self.device).float().unsqueeze(0)
        mask_t = to_torch(mask, self.device).float().unsqueeze(0)
        
        with torch.no_grad():
            q = self.q(obs_t, feats_t, patches_t, mask_t)[0].cpu().numpy()
        
        q[mask < 0.5] = -np.inf
        return int(np.argmax(q))

    def store(self, s, a_idx, r, s_next, done, *, curr_action_feats, curr_mask, curr_patches,
              next_action_feats, next_mask, next_patches):
        """Store transition in replay buffer"""
        self.buffer.push(s, a_idx, r, s_next, done, 
                        curr_action_feats, curr_mask, curr_patches,
                        next_action_feats, next_mask, next_patches)

    def train_step(self) -> float | None:
        if self.buffer.size < self.cfg.warmup_steps:
            return None
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
        curr_patches = to_torch(batch['curr_patches'], self.device).float()
        next_feats = to_torch(batch['next_feats'], self.device).float()
        next_mask  = to_torch(batch['next_mask'], self.device).float()
        next_patches = to_torch(batch['next_patches'], self.device).float()

        # Compute Q(s,a)
        q_all = self.q(s, curr_feats, curr_patches, curr_mask)
        
        valid = (a_idx >= 0)
        if not valid.any():
            return None
        
        q_sa = q_all[valid].gather(1, a_idx[valid].unsqueeze(1)).squeeze(1)

        # Compute target
        with torch.no_grad():
            if self.cfg.double_dqn:
                q_next_online = self.q(s_next, next_feats, next_patches, next_mask)
                q_next_online_masked = torch.where(
                    next_mask > 0.5,
                    q_next_online,
                    torch.tensor(float('-inf'), device=self.device)
                )
                next_a = torch.argmax(q_next_online_masked, dim=1, keepdim=True)
                
                q_next_target = self.q_target(s_next, next_feats, next_patches, next_mask)
                max_next = q_next_target.gather(1, next_a).squeeze(1)
                
                action_was_valid = next_mask.gather(1, next_a).squeeze(1) > 0.5
                max_next = torch.where(action_was_valid, max_next, torch.zeros_like(max_next))
            else:
                q_next = self.q_target(s_next, next_feats, next_patches, next_mask)
                q_next_masked = torch.where(
                    next_mask > 0.5,
                    q_next,
                    torch.tensor(float('-inf'), device=self.device)
                )
                max_next = torch.max(q_next_masked, dim=1)[0]

            target_all = r + (1.0 - done) * self.cfg.gamma * max_next

        target = target_all[valid]

        # Optimize
        loss = F.smooth_l1_loss(q_sa, target)
        
        self.opt.zero_grad()
        loss.backward()
        
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        
        self.opt.step()
        
        self.training_steps += 1

        if self.training_steps % self.cfg.target_update_interval == 0:
            self._update_target_network()
        
        return loss.item()

    def _update_target_network(self):
        if self.cfg.tau > 0:
            for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
                target_param.data.copy_(
                    self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
                )
        else:
            self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save({
            'q_state_dict': self.q.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'env_steps': self.env_steps,
            'training_steps': self.training_steps,
            'epsilon': self._eps,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q.load_state_dict(checkpoint['q_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.env_steps = checkpoint['env_steps']
        self.training_steps = checkpoint['training_steps']
        self._eps = checkpoint['epsilon']
