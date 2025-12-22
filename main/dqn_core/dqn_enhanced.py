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


def calculate_dynamic_epsilon_decay(total_episodes: int,
                                     plateau_at_ratio: float = 0.8,
                                     verbose: bool = False) -> tuple[int, int]:
    """
    Calculate epsilon decay parameters based on episode count (NOT steps).

    Strategy:
    - Epsilon decays from eps_start to eps_end over the first 80% of episodes
    - Remains at eps_end for the final 20% of episodes (plateau for pure exploitation)

    Args:
        total_episodes: Total number of training episodes
        plateau_at_ratio: Fraction of episodes where epsilon should reach minimum (default: 0.8)
        verbose: Print epsilon decay schedule (default: False)

    Returns:
        tuple: (eps_decay_episodes, total_episodes)
               - eps_decay_episodes: Number of episodes to decay epsilon
               - total_episodes: Total episodes (for config)

    Example:
        - 100 total episodes
        - Decay over first 80 episodes (80%)
        - Plateau for last 20 episodes (20%)

    This is EPISODE-based, so it's robust to variable episode lengths!
    """
    eps_decay_episodes = int(total_episodes * plateau_at_ratio)
    plateau_episodes = total_episodes - eps_decay_episodes

    if verbose:
        print(f"Epsilon decay: {total_episodes} episodes total, "
              f"decay over first {eps_decay_episodes} episodes (80%), "
              f"plateau at 1% for final {plateau_episodes} episodes (20%)")

    return max(1, eps_decay_episodes), total_episodes

def to_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.as_tensor(x, device=device)

def get_activation(name: str):
    """Get activation class from string name"""
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
    }
    return activations.get(name.lower(), nn.ReLU)


class MAB(nn.Module):
    """Multihead Attention Block for Set Transformer"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln = ln

    def forward(self, Q, K, key_padding_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Compute attention scores
        scores = Q_.bmm(K_.transpose(1,2)) / np.sqrt(self.dim_V)

        # Apply mask BEFORE softmax (critical fix: mask with -inf, not after softmax)
        if key_padding_mask is not None:
            # Replicate mask for multi-head
            key_padding_mask = torch.cat([key_padding_mask for _ in range(self.num_heads)], 0)
            # Mask invalid positions with -inf so they get zero probability after softmax
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float('-inf'))

        # Now softmax produces properly normalized attention weights (sum to 1)
        A = torch.softmax(scores, 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    """Induced Set Attention Block for Set Transformer"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask)
        return self.mab1(X, H)


class SetTransformer(nn.Module):
    """Set Transformer encoder with induced attention for permutation-invariant processing"""
    def __init__(self, dim_input, dim_output, num_heads=4, num_inds=32, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_output, num_heads, num_inds, ln=ln),
            ISAB(dim_output, dim_output, num_heads, num_inds, ln=ln)
        )

    def forward(self, X, key_padding_mask=None):
        # X: (B, A, dim_input)
        # key_padding_mask: (B, A) - True for padding
        # Properly chain the ISAB blocks (not parallel addition)
        out = self.enc[0](X, key_padding_mask=key_padding_mask)
        out = self.enc[1](out, key_padding_mask=key_padding_mask)
        return out

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
    def __init__(self, in_dim:int, hidden:int=256, out_dim:int=256, layers:int=2,
                 activation=nn.ReLU, dropout:float=0.0):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(max(0,layers-1)):
            mods += [nn.Linear(d, hidden), activation()]
            if dropout > 0:
                mods += [nn.Dropout(dropout)]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

class HeightmapCNN(nn.Module):
    """CNN encoder for heightmap patches around placement positions"""
    def __init__(self, patch_size:int=7, out_dim:int=64, channels:list=None, activation=nn.ReLU):
        super().__init__()
        self.patch_size = patch_size
        if channels is None:
            channels = [16, 32]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            activation(),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            activation(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[1], out_dim),
            activation()
        )

    def forward(self, patches):
        # patches: (B, 1, patch_size, patch_size)
        return self.cnn(patches)

class MultiScaleHeightmapCNN(nn.Module):
    """Multi-scale CNN for heightmap patches - sees local, medium, and global context"""
    def __init__(self, base_patch_size:int=7, out_dim:int=64, channels:list=None, activation=nn.ReLU):
        super().__init__()
        self.base_patch_size = base_patch_size

        # Three scales: half, base, double (e.g., 3, 7, 15 for base=7)
        self.small_size = max(3, base_patch_size // 2)
        self.medium_size = base_patch_size
        self.large_size = min(15, base_patch_size * 2)

        # Each scale gets its own CNN (shared architecture, different parameters)
        out_per_scale = out_dim // 3  # Split output evenly across scales
        self.small_cnn = HeightmapCNN(self.small_size, out_per_scale, channels, activation)
        self.medium_cnn = HeightmapCNN(self.medium_size, out_per_scale, channels, activation)
        self.large_cnn = HeightmapCNN(self.large_size, out_per_scale, channels, activation)

    def forward(self, patches_small, patches_medium, patches_large):
        """
        Args:
            patches_small: (B, 1, small_size, small_size)
            patches_medium: (B, 1, medium_size, medium_size)
            patches_large: (B, 1, large_size, large_size)
        Returns:
            (B, out_dim) concatenated embeddings
        """
        z_small = self.small_cnn(patches_small)
        z_medium = self.medium_cnn(patches_medium)
        z_large = self.large_cnn(patches_large)
        return torch.cat([z_small, z_medium, z_large], dim=-1)

class SimpleHead(nn.Module):
    def __init__(self, in_dim:int, hidden:int=256, dropout:float=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class QNetworkEnhanced(nn.Module):
    """Enhanced Q-Network with multi-scale heightmap CNN, item embeddings, and Transformer attention"""
    def __init__(self, obs_dim:int, action_feat_dim:int, hidden:int=256, enc_layers:int=2,
                 head_hidden:int=256, heightmap_patch_size:int=7, use_attention:bool=True,
                 attention_type:str="standard", attention_heads:int=4, num_inducing_points:int=32,
                 cnn_channels:list=None, dropout:float=0.0, activation:str="relu",
                 item_embed_dim:int=16, num_item_types:int=100):
        super().__init__()
        self.patch_size = heightmap_patch_size
        self.use_attention = use_attention
        self.attention_type = attention_type

        # Get activation function
        act_fn = get_activation(activation)

        # State encoder
        self.state_enc = MLP(obs_dim, hidden=hidden, out_dim=hidden, layers=enc_layers,
                           activation=act_fn, dropout=dropout)

        # Multi-scale Heightmap CNN
        self.heightmap_cnn = MultiScaleHeightmapCNN(base_patch_size=heightmap_patch_size,
                                                   out_dim=64, channels=cnn_channels,
                                                   activation=act_fn)

        # Item type embedding
        self.item_embedding = nn.Embedding(num_item_types, item_embed_dim)

        # Action encoder (now takes action_feats + multi-scale heightmap + item embedding)
        self.action_enc = MLP(action_feat_dim + 64 + item_embed_dim, hidden=hidden, out_dim=hidden,
                            layers=enc_layers, activation=act_fn, dropout=dropout)

        # Attention mechanism for action relationships
        if use_attention:
            if attention_type == "set_transformer":
                # Set Transformer with induced attention (permutation invariant)
                self.action_attention = SetTransformer(
                    dim_input=hidden,
                    dim_output=hidden,
                    num_heads=attention_heads,  # ✅ Now uses evolved parameter
                    num_inds=num_inducing_points,
                    ln=True
                )
            else:
                # Standard Transformer (default)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=attention_heads,  # ✅ Now uses evolved parameter
                    dim_feedforward=hidden*2,
                    dropout=dropout,  # Use evolved dropout value directly
                    batch_first=True
                )
                self.action_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.action_attention = None

        self.head = SimpleHead(2*hidden, hidden=head_hidden, dropout=dropout)

    def forward(self, s:torch.Tensor, action_feats:torch.Tensor,
                heightmap_patches_small:torch.Tensor, heightmap_patches_medium:torch.Tensor,
                heightmap_patches_large:torch.Tensor, item_ids:torch.Tensor,
                action_mask:torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B, obs_dim) - state observations
            action_feats: (B, A, action_feat_dim) - action features
            heightmap_patches_small: (B, A, small_size, small_size) - small scale patches
            heightmap_patches_medium: (B, A, medium_size, medium_size) - medium scale patches
            heightmap_patches_large: (B, A, large_size, large_size) - large scale patches
            item_ids: (B, A) - item type IDs for each action
            action_mask: (B, A) - 1 for valid actions, 0 for padding

        Returns:
            q: (B, A) - Q-values for each action
        """
        B, A = action_feats.shape[:2]

        # Encode state
        zs = self.state_enc(s)  # (B, hidden)

        # Process multi-scale heightmap patches through CNN
        small_size = self.heightmap_cnn.small_size
        medium_size = self.heightmap_cnn.medium_size
        large_size = self.heightmap_cnn.large_size

        h_patches_s = heightmap_patches_small.view(B*A, 1, small_size, small_size)
        h_patches_m = heightmap_patches_medium.view(B*A, 1, medium_size, medium_size)
        h_patches_l = heightmap_patches_large.view(B*A, 1, large_size, large_size)
        zh = self.heightmap_cnn(h_patches_s, h_patches_m, h_patches_l)  # (B*A, 64)

        # Embed item IDs
        item_ids_flat = item_ids.view(B*A).long()
        zi = self.item_embedding(item_ids_flat)  # (B*A, item_embed_dim)

        # Combine action features with heightmap embeddings and item embeddings
        action_feats_flat = action_feats.view(B*A, -1)
        combined = torch.cat([action_feats_flat, zh, zi], dim=-1)  # (B*A, action_feat_dim + 64 + item_embed_dim)
        
        # Encode actions
        za = self.action_enc(combined)  # (B*A, hidden)
        za = za.view(B, A, -1)  # (B, A, hidden)

        # Apply attention if enabled
        if self.action_attention is not None:
            # Create attention mask (True = ignore padding)
            attn_mask = (action_mask < 0.5)  # (B, A)

            if self.attention_type == "set_transformer":
                # Set Transformer: permutation-invariant attention
                za = self.action_attention(za, key_padding_mask=attn_mask)  # (B, A, hidden)
            else:
                # Standard Transformer: sequential attention
                za = self.action_attention(za, src_key_padding_mask=attn_mask)  # (B, A, hidden)
        
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
    eps_decay_episodes: int = None  # CHANGED: Episode-based decay instead of step-based
    total_episodes: int = None  # Total number of episodes for training
    buffer_size: int = 200_000
    n_step: int = 1
    target_update_interval: int = 1_000
    tau: float = 0.0
    hidden: int = 256
    enc_layers: int = 2
    head_hidden: int = 256
    heightmap_patch_size: int = 7
    use_attention: bool = True
    attention_type: str = "standard"  # "standard", "set_transformer", or "none"
    attention_heads: int = 4  # Number of attention heads (NEW: now configurable)
    num_inducing_points: int = 32  # For set_transformer only
    cnn_channels: list = None  # CNN channel progression, e.g., [16, 32]
    dropout: float = 0.0  # Dropout rate
    activation: str = "relu"  # Activation function: "relu", "gelu", or "silu"
    item_embed_dim: int = 16  # Item embedding dimension
    num_item_types: int = 100  # Number of unique item types
    device: str = "cpu"
    double_dqn: bool = True
    warmup_steps: int = 1000

    def __post_init__(self):
        """Set default values for mutable defaults"""
        if self.cnn_channels is None:
            self.cnn_channels = [16, 32]

        # Set default eps_decay_episodes if None
        if self.eps_decay_episodes is None and self.total_episodes is not None:
            # Default: decay over 80% of episodes
            self.eps_decay_episodes = int(self.total_episodes * 0.8)
        elif self.eps_decay_episodes is None:
            # Fallback if total_episodes not set (will be set by training function)
            self.eps_decay_episodes = 100
            self.total_episodes = 100

        # Validate attention_heads divisibility
        if self.use_attention and self.hidden % self.attention_heads != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by attention_heads ({self.attention_heads}). "
                f"Got remainder: {self.hidden % self.attention_heads}"
            )

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
            use_attention=cfg.use_attention,
            attention_type=cfg.attention_type,
            attention_heads=cfg.attention_heads,
            num_inducing_points=cfg.num_inducing_points,
            cnn_channels=cfg.cnn_channels,
            dropout=cfg.dropout,
            activation=cfg.activation,
            item_embed_dim=cfg.item_embed_dim,
            num_item_types=cfg.num_item_types
        ).to(self.device)

        self.q_target = QNetworkEnhanced(
            cfg.obs_dim, cfg.action_feat_dim,
            hidden=cfg.hidden, enc_layers=cfg.enc_layers,
            head_hidden=cfg.head_hidden,
            heightmap_patch_size=cfg.heightmap_patch_size,
            use_attention=cfg.use_attention,
            attention_type=cfg.attention_type,
            attention_heads=cfg.attention_heads,
            num_inducing_points=cfg.num_inducing_points,
            cnn_channels=cfg.cnn_channels,
            dropout=cfg.dropout,
            activation=cfg.activation,
            item_embed_dim=cfg.item_embed_dim,
            num_item_types=cfg.num_item_types
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
        self.episodes_completed = 0  # NEW: Track episode count for episode-based epsilon
        self._eps = cfg.eps_start

    def on_episode_end(self):
        """Call this at the end of each episode to update episode-based epsilon decay."""
        self.episodes_completed += 1

    def epsilon(self) -> float:
        """Calculate epsilon based on episode count (not step count)."""
        # Episode-based decay (robust to variable episode lengths)
        frac = min(1.0, self.episodes_completed / max(1, self.cfg.eps_decay_episodes))
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
                # Double DQN: use online network to select action, target network to evaluate
                q_next_online = self.q(s_next, next_feats, next_patches, next_mask)
                q_next_online_masked = torch.where(
                    next_mask > 0.5,
                    q_next_online,
                    torch.tensor(float('-inf'), device=self.device)
                )
                next_a = torch.argmax(q_next_online_masked, dim=1, keepdim=True)

                q_next_target = self.q_target(s_next, next_feats, next_patches, next_mask)
                max_next = q_next_target.gather(1, next_a).squeeze(1)

                # Check if ANY valid actions exist (not just whether selected action is valid)
                # If no valid actions, this is effectively a terminal state
                has_valid_actions = (next_mask.sum(dim=1) > 0.5)
                max_next = torch.where(has_valid_actions, max_next, torch.zeros_like(max_next))
            else:
                q_next = self.q_target(s_next, next_feats, next_patches, next_mask)
                q_next_masked = torch.where(
                    next_mask > 0.5,
                    q_next,
                    torch.tensor(float('-inf'), device=self.device)
                )
                max_next = torch.max(q_next_masked, dim=1)[0]

                # Check if ANY valid actions exist
                has_valid_actions = (next_mask.sum(dim=1) > 0.5)
                max_next = torch.where(has_valid_actions, max_next, torch.zeros_like(max_next))

            target_all = r + (1.0 - done) * self.cfg.gamma * max_next

        target = target_all[valid]

        # Optimize
        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()

        if self.cfg.grad_clip > 0:
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