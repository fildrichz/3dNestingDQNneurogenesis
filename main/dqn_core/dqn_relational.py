"""
Relational DQN for constraint-aware 3D bin packing.

Key ideas (vs dqn_enhanced.py):

1. The network receives the RAW problem structure instead of hand-crafted
   per-action features:
     - item-type tokens        (dims, weight, remaining quantity)
     - placed-item tokens      (dims, weight, position, bin)
     - EMS tokens              (raw geometry of every empty maximal space)
     - bin tokens              (full heightmap through a CNN + raw scalars)
     - a global token
   Constraints from the problem file are passed as TYPED EDGES between item
   tokens (incompatibility / affinity / must-not-be-above). The network is
   given no semantics for the edge types - it has to discover what each
   relation implies from reward, which makes the learned knowledge transfer
   to any other problem using the same constraint vocabulary (item IDs never
   enter the network).

2. The action space is the full combinatorial product
       (item type) x (EMS) x (rotation)
   scored compositionally from the contextual token embeddings
   (pointer-network style dot composition), so the agent "produces the
   combination" instead of ranking a pre-enumerated candidate list.

3. Only physics is masked (does not geometrically fit / would exceed the
   container after gravity). Problem-file constraints are NOT masked: the
   environment refuses violating placements with a penalty, and the agent
   must learn the constraint semantics itself.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Edge type vocabulary (global across all problems - this is what transfers)
EDGE_NONE = 0
EDGE_INCOMPATIBLE = 1        # items may not share a bin (symmetric)
EDGE_AFFINITY = 2            # items must share a bin (symmetric)
EDGE_NOT_ABOVE = 3           # source item may not be placed above target (source=heavy)
EDGE_NOT_BELOW = 4           # source item may not be placed below target (source=light)
EDGE_SAME_TYPE = 5           # tokens are instances of the same item type
NUM_EDGE_TYPES = 6


def to_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.as_tensor(x, device=device)


@dataclass
class RelationalDQNConfig:
    # --- state geometry caps (define padded tensor shapes) ---
    max_types: int = 16          # action-scorable remaining item-type slots
    max_placed: int = 64         # placed-item token slots
    max_ems: int = 96            # EMS token slots (across all bins, after pruning)
    max_bins: int = 4            # bin token slots
    num_rotations: int = 6
    grid: int = 24               # bin heightmap is resampled to grid x grid
    patch_size: int = 7          # local heightmap patch around each EMS corner

    # --- raw feature dims (must match the environment's state builder) ---
    item_feat_dim: int = 10
    ems_feat_dim: int = 7
    bin_feat_dim: int = 12
    global_feat_dim: int = 4

    # --- architecture ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    comp_dim: int = 64           # dimension of the Q composition space
    ffn_mult: int = 2
    dropout: float = 0.0
    cnn_channels: tuple = (16, 32)

    # --- RL / optimisation ---
    gamma: float = 0.992
    lr: float = 2e-4
    batch_size: int = 64
    grad_clip: float = 1.0
    buffer_size: int = 20_000
    n_step: int = 3
    warmup_steps: int = 1000
    target_update_interval: int = 200
    tau: float = 0.0
    double_dqn: bool = True

    # episode-based epsilon (same scheme as dqn_enhanced)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_episodes: int = 100
    device: str = "cpu"

    @property
    def num_item_tokens(self) -> int:
        return self.max_types + self.max_placed

    @property
    def num_actions(self) -> int:
        return self.max_types * self.max_ems * self.num_rotations


# ---------------------------------------------------------------------------
# Replay buffer over structured states
# ---------------------------------------------------------------------------

STATE_KEYS = ("item_feats", "item_mask", "edge_types", "ems_feats", "ems_patches",
              "ems_mask", "bin_feats", "bin_hm", "bin_mask", "global_feats", "valid")


class RelationalReplayBuffer:
    """Stores raw structured states; embeddings are recomputed at train time."""

    def __init__(self, cfg: RelationalDQNConfig):
        self.cfg = cfg
        c = cfg.buffer_size
        TI, TE, TB, G = cfg.num_item_tokens, cfg.max_ems, cfg.max_bins, cfg.grid
        I, R = cfg.max_types, cfg.num_rotations

        def state_arrays():
            return {
                "item_feats": np.zeros((c, TI, cfg.item_feat_dim), np.float32),
                "item_mask": np.zeros((c, TI), np.float32),
                "edge_types": np.zeros((c, TI, TI), np.int8),
                "ems_feats": np.zeros((c, TE, cfg.ems_feat_dim), np.float32),
                "ems_patches": np.zeros((c, TE, cfg.patch_size, cfg.patch_size), np.float16),
                "ems_mask": np.zeros((c, TE), np.float32),
                "bin_feats": np.zeros((c, TB, cfg.bin_feat_dim), np.float32),
                "bin_hm": np.zeros((c, TB, G, G), np.float16),
                "bin_mask": np.zeros((c, TB), np.float32),
                "global_feats": np.zeros((c, cfg.global_feat_dim), np.float32),
                "valid": np.zeros((c, I, TE, R), np.bool_),
            }

        self.s = state_arrays()
        self.s_next = state_arrays()
        self.a = np.zeros((c,), np.int64)
        self.r = np.zeros((c,), np.float32)
        self.done = np.zeros((c,), np.float32)

        self.ptr = 0
        self.size = 0
        self.n_step = max(1, cfg.n_step)
        self.gamma = cfg.gamma
        self._n_buf = deque()

    def _write(self, s, a, r, s_next, done):
        i = self.ptr
        for k in STATE_KEYS:
            self.s[k][i] = s[k]
            self.s_next[k][i] = s_next[k]
        self.a[i] = a
        self.r[i] = r
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.cfg.buffer_size
        self.size = min(self.size + 1, self.cfg.buffer_size)

    def push(self, s, a, r, s_next, done, isolated: bool = False):
        # isolated transitions (constraint refusals: self-loops that do not
        # advance the packing) are written directly as 1-step transitions and
        # NEVER enter the n-step chain, so their penalty cannot contaminate
        # the returns of the real placement sequence
        if isolated or self.n_step == 1:
            self._write(s, a, r, s_next, done)
            return
        self._n_buf.append((s, a, r, s_next, done))
        if len(self._n_buf) < self.n_step and not done:
            return
        # flush aggregated transitions; on `done` flush everything remaining
        while self._n_buf and (len(self._n_buf) >= self.n_step or self._n_buf[-1][4]):
            s0, a0 = self._n_buf[0][0], self._n_buf[0][1]
            R, g = 0.0, 1.0
            sn, dn = self._n_buf[-1][3], self._n_buf[-1][4]
            for (_, _, ri, sni, di) in self._n_buf:
                R += g * ri
                g *= self.gamma
                if di:
                    sn, dn = sni, True
                    break
            self._write(s0, a0, R, sn, dn)
            self._n_buf.popleft()
            if not dn and len(self._n_buf) < self.n_step:
                break
        if self._n_buf and self._n_buf[-1][4]:
            self._n_buf.clear()

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        out = {"a": self.a[idx], "r": self.r[idx], "done": self.done[idx]}
        out["s"] = {k: self.s[k][idx] for k in STATE_KEYS}
        out["s_next"] = {k: self.s_next[k][idx] for k in STATE_KEYS}
        return out


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class EdgeBiasedSelfAttention(nn.Module):
    """Multi-head self-attention with an additive learned bias per edge type
    (Graphormer-style). The bias table is the only place constraint relations
    enter the network - their meaning is learned from reward."""

    def __init__(self, d_model: int, n_heads: int, num_edge_types: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Embedding(num_edge_types, n_heads)
        nn.init.zeros_(self.edge_bias.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_types, pad_mask):
        # x: (B,T,d)   edge_types: (B,T,T) long   pad_mask: (B,T) True=pad
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)   # (B,h,T,dk)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.dk ** 0.5)  # (B,h,T,T)
        scores = scores + self.edge_bias(edge_types).permute(0, 3, 1, 2)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(y)


class RelationalEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, num_edge_types, ffn_mult=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = EdgeBiasedSelfAttention(d_model, n_heads, num_edge_types, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, edge_types, pad_mask):
        x = x + self.attn(self.ln1(x), edge_types, pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class HeightmapCNN(nn.Module):
    def __init__(self, out_dim: int, channels=(16, 32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1), nn.GELU(),
            nn.Conv2d(channels[0], channels[1], 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(channels[1], out_dim),
        )

    def forward(self, hm):  # (N,1,G,G)
        return self.net(hm)


class RelationalQNetwork(nn.Module):
    """Encodes the heterogeneous token set with edge-biased attention and
    scores every (item type, EMS, rotation) combination compositionally."""

    # token type ids for the type embedding table
    TOK_GLOBAL, TOK_ITEM, TOK_EMS, TOK_BIN = 0, 1, 2, 3

    def __init__(self, cfg: RelationalDQNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.item_in = nn.Linear(cfg.item_feat_dim, d)
        self.ems_in = nn.Linear(cfg.ems_feat_dim, d)
        self.ems_patch_cnn = HeightmapCNN(d, cfg.cnn_channels)  # local terrain at EMS
        self.bin_in = nn.Linear(cfg.bin_feat_dim, d)
        self.bin_cnn = HeightmapCNN(d, cfg.cnn_channels)        # whole-bin heightmap
        self.glob_in = nn.Linear(cfg.global_feat_dim, d)
        self.tok_type = nn.Embedding(4, d)

        self.layers = nn.ModuleList([
            RelationalEncoderLayer(d, cfg.n_heads, NUM_EDGE_TYPES,
                                   cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(d)

        # compositional Q head:  Q(i,e,r) = <u_i, v_e + rho_r>/sqrt(dc) + b_i + c_er
        dc = cfg.comp_dim
        self.u_proj = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, dc))
        self.v_proj = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, dc))
        self.rot_emb = nn.Parameter(torch.randn(cfg.num_rotations, dc) * 0.02)
        self.item_bias = nn.Linear(dc, 1)
        self.ems_bias = nn.Linear(dc, 1)

    def forward(self, s: Dict[str, torch.Tensor]) -> torch.Tensor:
        """s: batch of state tensors (see STATE_KEYS). Returns q: (B, I, E, R)."""
        cfg = self.cfg
        B = s["item_feats"].shape[0]
        TI, TE, TB = cfg.num_item_tokens, cfg.max_ems, cfg.max_bins

        z_item = self.item_in(s["item_feats"]) + self.tok_type.weight[self.TOK_ITEM]
        P = cfg.patch_size
        patches = s["ems_patches"].float().view(B * TE, 1, P, P)
        z_ems = (self.ems_in(s["ems_feats"])
                 + self.ems_patch_cnn(patches).view(B, TE, -1)
                 + self.tok_type.weight[self.TOK_EMS])
        hm = s["bin_hm"].float().view(B * TB, 1, cfg.grid, cfg.grid)
        z_bin = (self.bin_in(s["bin_feats"]) + self.bin_cnn(hm).view(B, TB, -1)
                 + self.tok_type.weight[self.TOK_BIN])
        z_glob = (self.glob_in(s["global_feats"])
                  + self.tok_type.weight[self.TOK_GLOBAL]).unsqueeze(1)

        x = torch.cat([z_glob, z_item, z_ems, z_bin], dim=1)   # (B,T,d)
        T = x.shape[1]

        # pad mask: True = padding (global token always valid)
        ones = torch.ones(B, 1, device=x.device)
        tok_mask = torch.cat([ones, s["item_mask"], s["ems_mask"], s["bin_mask"]], dim=1)
        pad_mask = tok_mask < 0.5

        # full edge-type matrix: constraint edges only within the item block
        edges = torch.zeros(B, T, T, dtype=torch.long, device=x.device)
        edges[:, 1:1 + TI, 1:1 + TI] = s["edge_types"].long()

        for layer in self.layers:
            x = layer(x, edges, pad_mask)
        x = self.ln_out(x)

        z_glob_o = x[:, 0]                              # (B,d)
        z_items_o = x[:, 1:1 + cfg.max_types]           # (B,I,d)  action-scorable slots
        z_ems_o = x[:, 1 + TI:1 + TI + TE]              # (B,E,d)

        g_i = z_glob_o.unsqueeze(1).expand(-1, cfg.max_types, -1)
        g_e = z_glob_o.unsqueeze(1).expand(-1, TE, -1)
        u = self.u_proj(torch.cat([z_items_o, g_i], dim=-1))       # (B,I,dc)
        v = self.v_proj(torch.cat([z_ems_o, g_e], dim=-1))         # (B,E,dc)
        v_er = v.unsqueeze(2) + self.rot_emb[None, None, :, :]     # (B,E,R,dc)

        q = torch.einsum('bic,berc->bier', u, v_er) / (self.cfg.comp_dim ** 0.5)
        q = q + self.item_bias(u).unsqueeze(-1)                    # (B,I,1,1) -> broadcast
        q = q + self.ems_bias(v_er).squeeze(-1).unsqueeze(1)       # (B,1,E,R)
        return q                                                    # (B,I,E,R)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RelationalDQNAgent:
    def __init__(self, cfg: RelationalDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = RelationalQNetwork(cfg).to(self.device)
        self.q_target = RelationalQNetwork(cfg).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = RelationalReplayBuffer(cfg)
        self.env_steps = 0
        self.training_steps = 0
        self.episodes_completed = 0

    # --- epsilon (episode based, same scheme as dqn_enhanced) ---
    def on_episode_end(self):
        self.episodes_completed += 1

    def epsilon(self) -> float:
        frac = min(1.0, self.episodes_completed / max(1, self.cfg.eps_decay_episodes))
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac

    def _state_to_torch(self, s: Dict[str, np.ndarray], batched=False):
        out = {}
        for k in STATE_KEYS:
            t = to_torch(s[k], self.device)
            if not batched:
                t = t.unsqueeze(0)
            if k == "edge_types":
                t = t.long()
            elif k == "valid":
                t = t.bool()
            elif t.dtype in (torch.float16, torch.float64):
                t = t.float()
            out[k] = t
        return out

    def select_action(self, s: Dict[str, np.ndarray], greedy: bool = False) -> Optional[int]:
        """Returns a flat action index into (max_types, max_ems, R), or None."""
        valid = s["valid"]
        flat_valid = np.flatnonzero(valid.reshape(-1))
        if flat_valid.size == 0:
            return None
        self.env_steps += 1
        if not greedy and np.random.rand() < self.epsilon():
            return int(np.random.choice(flat_valid))
        with torch.no_grad():
            st = self._state_to_torch(s)
            q = self.q(st)[0].cpu().numpy().reshape(-1)
        q[~valid.reshape(-1)] = -np.inf
        return int(np.argmax(q))

    def store(self, s, a, r, s_next, done, isolated: bool = False):
        self.buffer.push(s, a, r, s_next, done, isolated=isolated)

    def train_step(self) -> Optional[float]:
        if self.buffer.size < max(self.cfg.warmup_steps, self.cfg.batch_size):
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        s = self._state_to_torch(batch["s"], batched=True)
        s_next = self._state_to_torch(batch["s_next"], batched=True)
        a = to_torch(batch["a"], self.device).long()
        r = to_torch(batch["r"], self.device).float()
        done = to_torch(batch["done"], self.device).float()

        B = a.shape[0]
        # a = -1 marks dead-end transitions (no action taken): they carry
        # terminal reward into n-step returns but take no TD loss themselves
        row_valid = a >= 0
        if not row_valid.any():
            return None
        q_all = self.q(s).view(B, -1)
        q_sa = q_all.gather(1, a.clamp(min=0).unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_valid = s_next["valid"].view(B, -1)
            has_valid = next_valid.any(dim=1)
            neg_inf = torch.tensor(float('-inf'), device=self.device)
            if self.cfg.double_dqn:
                q_next_online = self.q(s_next).view(B, -1)
                q_next_online = torch.where(next_valid, q_next_online, neg_inf)
                # rows without any valid action would argmax over -inf; give
                # them index 0 (their value is zeroed below anyway)
                safe = torch.where(has_valid.unsqueeze(1), q_next_online,
                                   torch.zeros_like(q_next_online))
                next_a = torch.argmax(safe, dim=1, keepdim=True)
                q_next_t = self.q_target(s_next).view(B, -1)
                max_next = q_next_t.gather(1, next_a).squeeze(1)
            else:
                q_next_t = self.q_target(s_next).view(B, -1)
                q_next_t = torch.where(next_valid, q_next_t, neg_inf)
                max_next = q_next_t.max(dim=1)[0]
            max_next = torch.where(has_valid, max_next, torch.zeros_like(max_next))
            # n-step: gamma^n applied with the aggregated reward's own gamma
            gamma_n = self.cfg.gamma ** self.cfg.n_step
            target = r + (1.0 - done) * gamma_n * max_next

        loss = F.smooth_l1_loss(q_sa[row_valid], target[row_valid])
        self.opt.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.training_steps += 1
        if self.training_steps % self.cfg.target_update_interval == 0:
            self._update_target()
        return loss.item()

    def _update_target(self):
        if self.cfg.tau > 0:
            for tp, p in zip(self.q_target.parameters(), self.q.parameters()):
                tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)
        else:
            self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save({
            'q_state_dict': self.q.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'config': self.cfg.__dict__,
            'env_steps': self.env_steps,
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt['q_state_dict'])
        self.q_target.load_state_dict(ckpt['q_target_state_dict'])
        self.opt.load_state_dict(ckpt['optimizer_state_dict'])
        self.env_steps = ckpt['env_steps']
        self.training_steps = ckpt['training_steps']
        self.episodes_completed = ckpt.get('episodes_completed', 0)
