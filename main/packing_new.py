import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# Import your core modules (flexible imports)
try:
    from nesting.packing_core import Container
    from dqn_core.dqn import DQNAgent, DQNConfig
except ImportError:
    # If not in subdirectories, try direct import
    from packing_core import Container
    from dqn import DQNAgent, DQNConfig

# ---------------------
# Lightweight RL Env around Container
# ---------------------
def volume(size: Tuple[int,int,int]) -> int:
    w,d,h = size; return int(w)*int(d)*int(h)

class PackingEnv:
    def __init__(self, W=40, D=40, H=40, items: List[Tuple[int,int,int]] = None, 
                 max_actions: int = 128, topk_eps: int = 32, seed: int = 0, gamma: float = 0.992,
                 multi_bin: bool = False):
        """
        PackingEnv for 3D bin packing.
        
        Args:
            W, D, H: Bin dimensions
            items: List of items to pack
            max_actions: Maximum actions per step
            topk_eps: Maximum extreme points to consider
            seed: Random seed
            gamma: Discount factor
            multi_bin: If True, uses multiple bins (thpack9 mode: minimize bins)
                      If False, uses single bin (maximize utilization)
        """
        self.rng = np.random.default_rng(seed)
        self.bin_size = (int(W), int(D), int(H))
        self.bin_volume = int(W*D*H)
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.topk_eps = int(topk_eps)
        self.initial_items = list(items) if items is not None else []
        self.multi_bin = multi_bin
        self.reset()

    def reset(self, items: List[Tuple[int,int,int]] = None):
        self.items = list(items if items is not None else self.initial_items)
        self.n_items = len(self.items)
        
        if self.multi_bin:
            # Multi-bin mode: start with one bin, can add more
            self.bins = [Container(*self.bin_size)]
            self.current_bin_idx = 0
            self.C = self.bins[0]  # Current active bin
        else:
            # Single-bin mode: maximize utilization of one bin
            self.C = Container(*self.bin_size)
            self.bins = [self.C]
        
        # Initialize EPs
        if not getattr(self.C, "eps", None):
            self.C.eps = [(0,0,0)]
        else:
            self.C.eps = [(0,0,0)]
        self.C.placed = []
        
        self.total_placed_volume = 0
        self.done = False
        return self._obs()

    # Feasibility wrapper using your container checks
    def _feasible(self, ep, size) -> bool:
        return (self.C._fits_caps(ep, size) and 
                self.C._fits_container(ep, size) and 
                self.C._fits_collision_free(ep, size))

    def enumerate_actions(self):
        actions = []
        eps_sorted = sorted(set(self.C.eps), key=lambda p:(p[2], p[1], p[0]))[:self.topk_eps]
        for ep_idx, ep in enumerate(eps_sorted):
            for item_idx, item in enumerate(self.items):
                # 6 axis-aligned rotations
                rots = ((item[0],item[1],item[2]), (item[0],item[2],item[1]),
                        (item[1],item[0],item[2]), (item[1],item[2],item[0]),
                        (item[2],item[0],item[1]), (item[2],item[1],item[0]))
                for rot_idx, size in enumerate(rots):
                    if self._feasible(ep, size):
                        actions.append((item_idx, ep_idx, rot_idx, ep, size))
        return actions, eps_sorted

    def action_space(self):
        all_actions, eps_subset = self.enumerate_actions()
        A = len(all_actions)
        if A == 0:
            return [], np.zeros((0,), dtype=np.float32), eps_subset
        if A > self.max_actions:
            idxs = self.rng.choice(A, size=self.max_actions, replace=False)
            actions = [all_actions[i] for i in idxs]
            mask = np.ones((self.max_actions,), dtype=np.float32)
        else:
            actions = list(all_actions)
            pad = self.max_actions - A
            actions.extend([None]*pad)
            mask = np.zeros((self.max_actions,), dtype=np.float32); mask[:A] = 1.0
        return actions, mask, eps_subset

    def _obs(self) -> np.ndarray:
        """Reduced observation for faster learning"""
        W,D,H = self.bin_size
        
        if self.multi_bin:
            # Multi-bin mode: include bin count info
            bins_used = len(self.bins)
            bins_normalized = bins_used / max(1, self.n_items)  # Normalized by items
            packed = self.total_placed_volume / (self.bin_volume * bins_used)  # Average utilization
            items_left = len(self.items) / max(1, self.n_items)
            
            # Current bin geometry
            if self.C.eps:
                caps = [self.C.ep_rs.get(ep, (W-ep[0], D-ep[1], H-ep[2])) for ep in self.C.eps]
                cx = max(c[0] for c in caps)/W
                cy = max(c[1] for c in caps)/D
                cz = max(c[2] for c in caps)/H
            else:
                cx=cy=cz=0.0
            
            return np.array([packed, items_left, cx, cy, cz, bins_normalized], dtype=np.float32)
        else:
            # Single-bin mode: original observation
            packed = self.total_placed_volume / self.bin_volume
            items_left = len(self.items) / max(1, self.n_items)
            
            if self.C.eps:
                caps = [self.C.ep_rs.get(ep, (W-ep[0], D-ep[1], H-ep[2])) for ep in self.C.eps]
                cx = max(c[0] for c in caps)/W
                cy = max(c[1] for c in caps)/D
                cz = max(c[2] for c in caps)/H
            else:
                cx=cy=cz=0.0
            
            return np.array([packed, items_left, cx, cy, cz], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode done, reset required.")
        info = {}

        if self.multi_bin:
            # Multi-bin mode: minimize bins used
            bins_before = len(self.bins)
            
            # Agent decides to stop or no action
            if action is None:
                self.done = True
                # Penalize if items remain
                items_remaining = len(self.items)
                if items_remaining > 0:
                    # Heavy penalty for not packing all items
                    reward = -10.0 * (items_remaining / max(1, self.n_items))
                else:
                    # Reward based on bins used (fewer is better)
                    reward = 1.0 - (len(self.bins) / max(1, self.n_items))
                
                info["bins_used"] = len(self.bins)
                info["items_remaining"] = items_remaining
                info["all_packed"] = (items_remaining == 0)
                return self._obs(), reward, self.done, info

            # Try to place in current bin
            item_idx, ep_idx, rot_idx, pos, size = action
            ok = self.C.place_at(pos, size)
            
            if not ok:
                # Should be rare - action was checked as feasible
                self.done = True
                return self._obs(), -1.0, True, {"invalid": True}

            # Successfully placed
            v = int(size[0] * size[1] * size[2])
            self.total_placed_volume += v
            del self.items[item_idx]
            
            # Reward: small positive for placing, penalty for opening new bins
            bins_after = len(self.bins)
            reward = 0.01  # Small reward for each placement
            
            if bins_after > bins_before:
                # Opened a new bin - penalty
                reward -= 0.5
            
            # Check if all items packed
            if len(self.items) == 0:
                self.done = True
                # Bonus for packing everything, scaled by efficiency
                reward += 2.0 - (len(self.bins) / max(1, self.n_items))
                info["bins_used"] = len(self.bins)
                info["items_remaining"] = 0
                info["all_packed"] = True
            else:
                # Check if current bin has feasible actions
                actions, _ = self.enumerate_actions()
                if len(actions) == 0:
                    # Current bin full, try to open new bin
                    self._open_new_bin()
                    bins_after = len(self.bins)
                    if bins_after > bins_before:
                        reward -= 0.5  # Penalty for opening bin
                    
                    # Check if new bin has feasible actions
                    actions, _ = self.enumerate_actions()
                    if len(actions) == 0:
                        # No feasible actions even in new bin - terminate
                        self.done = True
                        items_remaining = len(self.items)
                        reward -= 5.0 * (items_remaining / max(1, self.n_items))
                        info["bins_used"] = len(self.bins)
                        info["items_remaining"] = items_remaining
                        info["all_packed"] = False

            return self._obs(), reward, self.done, info
            
        else:
            # Single-bin mode: maximize utilization (original behavior)
            util_prev = self.total_placed_volume / self.bin_volume

            if action is None:
                self.done = True
                reward = (self.gamma * util_prev) - util_prev
                reward += util_prev  # Terminal bonus
                info["utilization"] = util_prev
                return self._obs(), reward, self.done, info

            # Try to place
            item_idx, ep_idx, rot_idx, pos, size = action
            ok = self.C.place_at(pos, size)
            if not ok:
                self.done = True
                return self._obs(), -1.0, True, {"invalid": True}

            # Update volume and remove item
            v = int(size[0] * size[1] * size[2])
            self.total_placed_volume += v
            del self.items[item_idx]

            # New utilization
            util_next = self.total_placed_volume / self.bin_volume

            # Potential-based shaping
            reward = (self.gamma * util_next) - util_prev

            # Check termination
            actions, eps_subset = self.enumerate_actions()
            if len(actions) == 0 or len(self.items) == 0:
                self.done = True
                reward += util_next  # Terminal bonus
                info["utilization"] = util_next

            return self._obs(), reward, self.done, info
    
    def _open_new_bin(self):
        """Open a new bin for multi-bin mode"""
        new_bin = Container(*self.bin_size)
        new_bin.eps = [(0, 0, 0)]
        new_bin.placed = []
        self.bins.append(new_bin)
        self.current_bin_idx = len(self.bins) - 1
        self.C = new_bin


# ---------------------
# Feature utils for DQN
# ---------------------
ACTION_FEAT_DIM = 19

def _safe_caps(C, ep):
    """Get residual capacities with fallback to container bounds"""
    x,y,z = ep
    return C.ep_rs.get(ep, (C.w - x, C.d - y, C.h - z))

def build_action_features(env: PackingEnv, actions):
    """Build 19-dimensional action features"""
    W,D,H = env.bin_size
    binV = float(env.bin_volume)
    rows = []
    for a in actions:
        if a is None:
            rows.append([0.0]*ACTION_FEAT_DIM); continue
        item_idx, ep_idx, rot_idx, ep, size = a
        iw,id_,ih = env.items[item_idx]
        rw,rd,rh  = size
        ex,ey,ez  = ep
        cx,cy,cz  = _safe_caps(env.C, ep)
        # normalized sizes/pos/caps
        rw_n, rd_n, rh_n = rw/W, rd/D, rh/H
        ex_n, ey_n, ez_n = ex/W, ey/D, ez/H
        cx_n, cy_n, cz_n = cx/W, cy/D, cz/H
        # slack (cap - size), normalized
        sx, sy, sz = max(0,cx-rw), max(0,cy-rd), max(0,cz-rh)
        sx_n, sy_n, sz_n = sx/W, sy/D, sz/H
        # cheap immediate gain + tightness
        vol     = float(rw*rd*rh)
        delta_u = vol / binV
        tight   = float((sx==0) + (sy==0) + (sz==0))
        rows.append([
            iw/W, id_/D, ih/H,         # item original dims (normed)
            rw_n, rd_n, rh_n,          # chosen rotation (normed)
            ex_n, ey_n, ez_n,          # EP position (normed)
            cx_n, cy_n, cz_n,          # residual caps (normed)
            sx_n, sy_n, sz_n,          # slack (normed)
            delta_u, tight,            # reward hints
            W/D, D/H                   # bin aspect ratios
        ])
    return np.asarray(rows, dtype=np.float32)


def pad_feats_mask(feats: np.ndarray, mask_short: np.ndarray, maxA:int):
    """Pad action features and mask to max_actions length"""
    A = feats.shape[0]
    F = np.zeros((maxA, ACTION_FEAT_DIM), np.float32)
    M = np.zeros((maxA,), np.float32)
    F[:A] = feats
    M[:A] = mask_short[:A] if mask_short.shape[0] >= A else 1.0
    return F, M

# ---------------------
# Utility functions
# ---------------------
def make_items(n=12, lo=4, hi=10, seed=0):
    """Generate random items"""
    rng=np.random.default_rng(seed)
    return [(int(rng.integers(lo,hi)), int(rng.integers(lo,hi)), int(rng.integers(lo,hi))) for _ in range(n)]


# ---------------------
# Training functions
# ---------------------
def train_pack_dqn(episodes=100, W=40, D=40, H=40, n_items=50, seed=42, 
                   max_actions=128, topk_eps=48, train_freq=1, 
                   num_train_steps=1, log_interval=10, save_path=None):
    """
    Train DQN agent on 3D bin packing with varying item sets.
    
    Args:
        episodes: Number of training episodes
        W, D, H: Container dimensions
        n_items: Number of items per episode
        seed: Random seed
        max_actions: Maximum actions to consider per step
        topk_eps: Maximum extreme points to consider
        train_freq: Train every N environment steps
        num_train_steps: Number of training updates per training call
        log_interval: Print stats every N episodes
        save_path: Path to save final model (optional)
    """
    # Create output_data directory if saving
    import os
    if save_path:
        os.makedirs("output_data", exist_ok=True)
        if not save_path.startswith("output_data/"):
            save_path = os.path.join("output_data", os.path.basename(save_path))
    print(f"\n{'='*60}")
    print(f"Starting DQN Training for 3D Bin Packing")
    print(f"{'='*60}")
    print(f"Container: {W}x{D}x{H} (volume={W*D*H})")
    print(f"Items per episode: {n_items}")
    print(f"Episodes: {episodes}")
    print(f"Max actions: {max_actions}, Top EPs: {topk_eps}")
    print(f"Training frequency: every {train_freq} step(s), {num_train_steps} update(s) per call")
    print(f"{'='*60}\n")
    
    # Create environment
    env = PackingEnv(W, D, H, items=make_items(n=n_items, seed=seed), 
                     max_actions=max_actions, topk_eps=topk_eps, seed=seed, gamma=0.992)
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    
    # Create DQN agent
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cpu",
        gamma=0.992,
        lr=2e-4,
        batch_size=64,
        buffer_size=200_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=episodes * 20,  # Decay over most of training
        target_update_interval=500,
        n_step=3,
        double_dqn=True,
        warmup_steps=500,  # Start training after 500 samples
    )
    agent = DQNAgent(cfg)
    
    # Tracking
    import collections
    util_hist = collections.deque(maxlen=50)
    returns_hist = collections.deque(maxlen=50)
    steps_hist = collections.deque(maxlen=50)
    best_util = 0.0
    
    print(f"Agent initialized. Starting training...\n")
    
    for ep in range(episodes):
        # Generate new random items each episode
        obs = env.reset(items=make_items(n=n_items, seed=None))
        ep_ret = 0.0
        steps = 0
        losses = []
        
        while True:
            # Get valid actions
            actions, mask_short, _ = env.action_space()
            feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            
            # Select action
            act_idx = agent.select_action(
                obs, 
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
            )
            act = None if (act_idx is None or actions==[] or actions[act_idx] is None) else actions[act_idx]
            
            # Pad current action features/mask for storage
            currF, currM = pad_feats_mask(
                feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            
            # Environment step
            nobs, rew, done, info = env.step(act)
            
            # Get next action features/mask
            n_actions, n_mask_short, _ = env.action_space()
            n_feats = build_action_features(env, n_actions) if len(n_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            nextF, nextM = pad_feats_mask(
                n_feats, 
                n_mask_short if n_mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            
            # Store transition
            agent.store(obs, act_idx, rew, nobs, done,
                       curr_action_feats=currF, curr_mask=currM,
                       next_action_feats=nextF, next_mask=nextM)
            
            # Train agent
            if steps % train_freq == 0:
                for _ in range(num_train_steps):
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)
            
            obs = nobs
            ep_ret += rew
            steps += 1
            
            if done:
                util = info.get("utilization", 0.0)
                util_hist.append(util)
                returns_hist.append(ep_ret)
                steps_hist.append(steps)
                best_util = max(best_util, util)
                
                # Logging
                if (ep + 1) % log_interval == 0 or ep == 0:
                    ma_util = np.mean(util_hist) if len(util_hist) > 0 else util
                    ma_ret = np.mean(returns_hist) if len(returns_hist) > 0 else ep_ret
                    ma_steps = np.mean(steps_hist) if len(steps_hist) > 0 else steps
                    avg_loss = np.mean(losses) if losses else 0.0
                    
                    print(f"Episode {ep+1:4d}/{episodes} | "
                          f"Steps: {steps:3d} | "
                          f"Return: {ep_ret:7.3f} | "
                          f"Util: {util:.3f} | "
                          f"MA50: {ma_util:.3f} | "
                          f"Best: {best_util:.3f} | "
                          f"ε: {agent.epsilon():.3f} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Buffer: {agent.buffer.size}/{agent.buffer.capacity}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best utilization achieved: {best_util:.3f}")
    print(f"Final MA50 utilization: {np.mean(util_hist):.3f}")
    print(f"Total training steps: {agent.training_steps}")
    print(f"{'='*60}\n")
    
    # Save model
    if save_path:
        agent.save(save_path)
        print(f"Model saved to: {save_path}\n")
    
    # Visualize final episode
    env.C.plot3d(title=f"Final Episode Packing (Util: {util:.3f})")
    
    return agent, env


def train_pack_dqn_fixed_items(episodes=100, W=40, D=40, H=40, n_items=50, seed=42, 
                                max_actions=128, topk_eps=48, train_freq=1, 
                                num_train_steps=1, log_interval=10, save_path=None):
    """
    Train DQN agent on 3D bin packing with FIXED item set (for debugging/analysis).
    
    Same arguments as train_pack_dqn, but uses the same item set every episode.
    """
    # Create output_data directory if saving
    import os
    if save_path:
        os.makedirs("output_data", exist_ok=True)
        if not save_path.startswith("output_data/"):
            save_path = os.path.join("output_data", os.path.basename(save_path))
    print(f"\n{'='*60}")
    print(f"Starting DQN Training (FIXED ITEMS) for 3D Bin Packing")
    print(f"{'='*60}")
    print(f"Container: {W}x{D}x{H} (volume={W*D*H})")
    print(f"Items per episode: {n_items} (SAME EVERY EPISODE)")
    print(f"Episodes: {episodes}")
    print(f"Max actions: {max_actions}, Top EPs: {topk_eps}")
    print(f"{'='*60}\n")
    
    # Create environment with fixed items
    fixed_items = make_items(n=n_items, seed=42)
    env = PackingEnv(W, D, H, items=fixed_items, 
                     max_actions=max_actions, topk_eps=topk_eps, seed=seed, gamma=0.992)
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    
    # Create DQN agent
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cpu",
        gamma=0.992,
        lr=2e-4,
        batch_size=64,
        buffer_size=200_000,
        eps_start=1.0,
        eps_end=0.01,  # Lower final epsilon for fixed items
        eps_decay_steps=episodes * 15,
        target_update_interval=500,
        n_step=3,
        double_dqn=True,
        warmup_steps=500,
    )
    agent = DQNAgent(cfg)
    
    # Tracking
    import collections
    util_hist = collections.deque(maxlen=50)
    returns_hist = collections.deque(maxlen=50)
    steps_hist = collections.deque(maxlen=50)
    best_util = 0.0
    
    print(f"Agent initialized. Starting training...\n")
    
    for ep in range(episodes):
        # Use SAME items every episode
        obs = env.reset(items=fixed_items.copy())
        ep_ret = 0.0
        steps = 0
        losses = []
        
        while True:
            # Get valid actions
            actions, mask_short, _ = env.action_space()
            feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            
            # Select action
            act_idx = agent.select_action(
                obs, 
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
            )
            act = None if (act_idx is None or actions==[] or actions[act_idx] is None) else actions[act_idx]
            
            # Pad current action features/mask for storage
            currF, currM = pad_feats_mask(
                feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            
            # Environment step
            nobs, rew, done, info = env.step(act)
            
            # Get next action features/mask
            n_actions, n_mask_short, _ = env.action_space()
            n_feats = build_action_features(env, n_actions) if len(n_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            nextF, nextM = pad_feats_mask(
                n_feats, 
                n_mask_short if n_mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            
            # Store transition
            agent.store(obs, act_idx, rew, nobs, done,
                       curr_action_feats=currF, curr_mask=currM,
                       next_action_feats=nextF, next_mask=nextM)
            
            # Train agent
            if steps % train_freq == 0:
                for _ in range(num_train_steps):
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)
            
            obs = nobs
            ep_ret += rew
            steps += 1
            
            if done:
                util = info.get("utilization", 0.0)
                util_hist.append(util)
                returns_hist.append(ep_ret)
                steps_hist.append(steps)
                best_util = max(best_util, util)
                
                # Logging
                if (ep + 1) % log_interval == 0 or ep == 0:
                    ma_util = np.mean(util_hist) if len(util_hist) > 0 else util
                    ma_ret = np.mean(returns_hist) if len(returns_hist) > 0 else ep_ret
                    ma_steps = np.mean(steps_hist) if len(steps_hist) > 0 else steps
                    avg_loss = np.mean(losses) if losses else 0.0
                    
                    print(f"Episode {ep+1:4d}/{episodes} | "
                          f"Steps: {steps:3d} | "
                          f"Return: {ep_ret:7.3f} | "
                          f"Util: {util:.3f} | "
                          f"MA50: {ma_util:.3f} | "
                          f"Best: {best_util:.3f} | "
                          f"ε: {agent.epsilon():.3f} | "
                          f"Loss: {avg_loss:.4f}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training Complete (FIXED ITEMS)!")
    print(f"Best utilization achieved: {best_util:.3f}")
    print(f"Final MA50 utilization: {np.mean(util_hist):.3f}")
    print(f"{'='*60}\n")
    
    # Save model
    if save_path:
        agent.save(save_path)
        print(f"Model saved to: {save_path}\n")
    
    # Visualize final episode
    env.C.plot3d(title=f"Final Episode Packing (Fixed Items, Util: {util:.3f})")
    
    return agent, env


# ---------------------
# THPACK9 Support Functions
# ---------------------

def train_thpack_instance(thpack_instance, episodes=200, max_actions=128, topk_eps=48,
                         train_freq=1, num_train_steps=1, log_interval=10, 
                         save_path=None, shuffle_items=True, seed=42, multi_bin=True):
    """
    Train DQN agent on a thpack9 instance.
    
    Args:
        thpack_instance: ThpackInstance object from thpack_loader
        episodes: Number of training episodes
        max_actions: Maximum actions to consider per step
        topk_eps: Maximum extreme points to consider
        shuffle_items: Shuffle item order each episode for generalization
        save_path: Path to save trained model (optional)
        multi_bin: If True, minimize bins (thpack9 objective)
                   If False, maximize single bin utilization
    
    Returns:
        agent, env, history: Trained agent, final env state, training history
    """
    # Extract container dimensions and items
    W, D, H = thpack_instance.container_w, thpack_instance.container_d, thpack_instance.container_h
    base_items = thpack_instance.expand_to_items()
    
    # Create output_data directory if saving
    import os
    if save_path:
        os.makedirs("output_data", exist_ok=True)
        if not save_path.startswith("output_data/"):
            save_path = os.path.join("output_data", os.path.basename(save_path))
    
    mode_str = "Multi-Bin (Minimize Bins)" if multi_bin else "Single-Bin (Maximize Util)"
    print(f"\n{'='*60}")
    print(f"Training DQN on Thpack9 Instance - {mode_str}")
    print(f"{'='*60}")
    print(f"Problem ID: {thpack_instance.problem_id}")
    print(f"Container: {W}x{D}x{H} (volume={W*D*H})")
    print(f"Box types: {len(thpack_instance.box_types)}")
    print(f"Total items: {len(base_items)}")
    print(f"Total item volume: {thpack_instance.total_volume()}")
    print(f"Density: {thpack_instance.total_volume() / thpack_instance.container_volume():.2f}x")
    print(f"Theoretical min bins: {int(np.ceil(thpack_instance.total_volume() / thpack_instance.container_volume()))}")
    print(f"Episodes: {episodes}")
    print(f"Shuffle items: {shuffle_items}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = PackingEnv(W, D, H, items=base_items, 
                     max_actions=max_actions, topk_eps=topk_eps, seed=seed, 
                     gamma=0.992, multi_bin=multi_bin)
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    
    # Create DQN agent
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cpu",
        gamma=0.992,
        lr=2e-4,
        batch_size=64,
        buffer_size=200_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=episodes * 15,
        target_update_interval=1000,
        n_step=3,
        double_dqn=True,
        warmup_steps=1000,
    )
    agent = DQNAgent(cfg)
    
    # Tracking
    import collections
    if multi_bin:
        bins_hist = collections.deque(maxlen=50)
        packed_all_hist = collections.deque(maxlen=50)
    else:
        util_hist = collections.deque(maxlen=50)
    
    returns_hist = collections.deque(maxlen=50)
    steps_hist = collections.deque(maxlen=50)
    
    history = {'bins_used': [], 'returns': [], 'steps': [], 'losses': [], 
               'all_packed': [], 'items_remaining': []}
    
    best_bins = float('inf')
    best_episode_all_packed = None
    rng = np.random.default_rng(seed)
    
    print(f"Agent initialized. Starting training...\n")
    
    for ep in range(episodes):
        # Shuffle items each episode if requested
        if shuffle_items:
            items_this_ep = base_items.copy()
            rng.shuffle(items_this_ep)
        else:
            items_this_ep = base_items.copy()
        
        obs = env.reset(items=items_this_ep)
        ep_ret = 0.0
        steps = 0
        losses = []
        
        while True:
            actions, mask_short, _ = env.action_space()
            feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            
            act_idx = agent.select_action(
                obs, 
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
            )
            act = None if (act_idx is None or actions==[] or actions[act_idx] is None) else actions[act_idx]
            
            currF, currM = pad_feats_mask(
                feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            
            nobs, rew, done, info = env.step(act)
            
            n_actions, n_mask_short, _ = env.action_space()
            n_feats = build_action_features(env, n_actions) if len(n_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            nextF, nextM = pad_feats_mask(
                n_feats, 
                n_mask_short if n_mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            
            agent.store(obs, act_idx, rew, nobs, done,
                       curr_action_feats=currF, curr_mask=currM,
                       next_action_feats=nextF, next_mask=nextM)
            
            if steps % train_freq == 0:
                for _ in range(num_train_steps):
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)
            
            obs = nobs
            ep_ret += rew
            steps += 1
            
            if done:
                if multi_bin:
                    bins_used = info.get("bins_used", len(env.bins))
                    items_rem = info.get("items_remaining", len(env.items))
                    all_packed = info.get("all_packed", items_rem == 0)
                    
                    bins_hist.append(bins_used)
                    packed_all_hist.append(1.0 if all_packed else 0.0)
                    returns_hist.append(ep_ret)
                    steps_hist.append(steps)
                    
                    history['bins_used'].append(bins_used)
                    history['all_packed'].append(all_packed)
                    history['items_remaining'].append(items_rem)
                else:
                    util = info.get("utilization", 0.0)
                    util_hist.append(util)
                    returns_hist.append(ep_ret)
                    steps_hist.append(steps)
                    history['bins_used'].append(1)
                    history['all_packed'].append(True)
                    history['items_remaining'].append(0)
                
                history['returns'].append(ep_ret)
                history['steps'].append(steps)
                if losses:
                    history['losses'].append(np.mean(losses))
                
                if multi_bin and all_packed and bins_used < best_bins:
                    best_bins = bins_used
                    best_episode_all_packed = ep
                
                if (ep + 1) % log_interval == 0 or ep == 0:
                    if multi_bin:
                        ma_bins = np.mean(bins_hist) if len(bins_hist) > 0 else bins_used
                        ma_packed = np.mean(packed_all_hist) if len(packed_all_hist) > 0 else (1.0 if all_packed else 0.0)
                        avg_loss = np.mean(losses) if losses else 0.0
                        
                        print(f"Episode {ep+1:4d}/{episodes} | "
                              f"Bins: {bins_used:2d} | "
                              f"AllPacked: {int(all_packed)} | "
                              f"ItemsRem: {items_rem:3d} | "
                              f"MA50_bins: {ma_bins:.1f} | "
                              f"Best: {best_bins if best_bins < float('inf') else '-'} | "
                              f"ε: {agent.epsilon():.3f}")
                    else:
                        ma_util = np.mean(util_hist) if len(util_hist) > 0 else util
                        avg_loss = np.mean(losses) if losses else 0.0
                        
                        print(f"Episode {ep+1:4d}/{episodes} | "
                              f"Steps: {steps:3d} | "
                              f"Util: {util:.3f} | "
                              f"MA50: {ma_util:.3f} | "
                              f"ε: {agent.epsilon():.3f} | "
                              f"Loss: {avg_loss:.4f}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    if multi_bin:
        print(f"Best bins used (all packed): {best_bins if best_bins < float('inf') else 'Never packed all'}")
        if best_episode_all_packed is not None:
            print(f"Best achieved at episode: {best_episode_all_packed + 1}")
        success_rate = np.mean([1 if x else 0 for x in history['all_packed']])
        print(f"Success rate (all packed): {success_rate:.2%}")
    else:
        print(f"Best utilization: {max(util_hist):.3f}")
        print(f"Final MA50 utilization: {np.mean(util_hist):.3f}")
    print(f"{'='*60}\n")
    
    if save_path:
        agent.save(save_path)
        print(f"Model saved to: {save_path}\n")
    
    # Visualize
    if multi_bin and len(env.bins) > 0:
        print(f"Final packing used {len(env.bins)} bins")
        for i, bin in enumerate(env.bins[:3]):  # Show first 3 bins
            bin.plot3d(title=f"Bin {i+1}/{len(env.bins)} - Items: {len(bin.placed)}")
    else:
        env.C.plot3d(title=f"Thpack Instance {thpack_instance.problem_id}")
    
    return agent, env, history


def evaluate_thpack_instance(agent, thpack_instance, n_episodes=10, 
                             max_actions=128, topk_eps=48, visualize_best=True, 
                             seed=42, multi_bin=True):
    """
    Evaluate a trained agent on a thpack9 instance.
    
    Args:
        agent: Trained DQN agent
        thpack_instance: ThpackInstance object
        n_episodes: Number of evaluation episodes
        visualize_best: Show 3D plot of best packing
        multi_bin: If True, use multi-bin mode (minimize bins)
    
    Returns:
        dict with evaluation metrics
    """
    W, D, H = thpack_instance.container_w, thpack_instance.container_d, thpack_instance.container_h
    base_items = thpack_instance.expand_to_items()
    
    mode_str = "Multi-Bin" if multi_bin else "Single-Bin"
    print(f"\n{'='*60}")
    print(f"Evaluating on Thpack Instance {thpack_instance.problem_id} ({mode_str})")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}\n")
    
    env = PackingEnv(W, D, H, items=base_items, 
                     max_actions=max_actions, topk_eps=topk_eps, 
                     seed=seed, gamma=0.992, multi_bin=multi_bin)
    
    bins_used_list = []
    all_packed_list = []
    items_remaining_list = []
    returns = []
    steps_list = []
    best_bins = float('inf')
    best_env = None
    
    rng = np.random.default_rng(seed)
    original_eps = agent._eps
    agent._eps = 0.0  # Greedy evaluation
    
    for ep in range(n_episodes):
        items_this_ep = base_items.copy()
        rng.shuffle(items_this_ep)
        
        obs = env.reset(items=items_this_ep)
        ep_ret = 0.0
        steps = 0
        
        while True:
            actions, mask_short, _ = env.action_space()
            feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            
            act_idx = agent.select_action(
                obs, 
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
            )
            act = None if (act_idx is None or actions==[] or actions[act_idx] is None) else actions[act_idx]
            
            nobs, rew, done, info = env.step(act)
            obs = nobs
            ep_ret += rew
            steps += 1
            
            if done:
                if multi_bin:
                    bins_used = info.get("bins_used", len(env.bins))
                    items_rem = info.get("items_remaining", len(env.items))
                    all_packed = info.get("all_packed", items_rem == 0)
                    
                    bins_used_list.append(bins_used)
                    all_packed_list.append(all_packed)
                    items_remaining_list.append(items_rem)
                    
                    if all_packed and bins_used < best_bins:
                        best_bins = bins_used
                        import copy
                        best_env = copy.deepcopy(env)
                    
                    print(f"Ep {ep+1:2d}/{n_episodes} | "
                          f"Bins: {bins_used:2d} | "
                          f"AllPacked: {int(all_packed)} | "
                          f"ItemsRem: {items_rem:3d}/{len(base_items)} | "
                          f"Steps: {steps:3d}")
                else:
                    util = info.get("utilization", 0.0)
                    bins_used_list.append(1)
                    all_packed_list.append(True)
                    items_remaining_list.append(0)
                    
                    print(f"Ep {ep+1:2d}/{n_episodes} | Util: {util:.3f} | Steps: {steps:3d}")
                
                returns.append(ep_ret)
                steps_list.append(steps)
                break
    
    agent._eps = original_eps
    
    success_rate = np.mean([1 if x else 0 for x in all_packed_list])
    
    results = {
        'mean_bins': np.mean(bins_used_list),
        'std_bins': np.std(bins_used_list),
        'best_bins': best_bins if best_bins < float('inf') else None,
        'worst_bins': max(bins_used_list) if bins_used_list else None,
        'success_rate': success_rate,
        'mean_items_remaining': np.mean(items_remaining_list),
        'mean_return': np.mean(returns),
        'mean_steps': np.mean(steps_list),
        'all_results': {
            'bins_used': bins_used_list,
            'all_packed': all_packed_list,
            'items_remaining': items_remaining_list
        }
    }
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    if multi_bin:
        print(f"Success Rate (all packed): {results['success_rate']:.1%}")
        print(f"Mean Bins Used: {results['mean_bins']:.2f} ± {results['std_bins']:.2f}")
        if results['best_bins']:
            print(f"Best Bins (all packed): {results['best_bins']}")
        else:
            print(f"Best Bins (all packed): Never achieved")
        print(f"Worst Bins: {results['worst_bins']}")
        print(f"Mean Items Remaining: {results['mean_items_remaining']:.1f}")
    else:
        print(f"Mean Utilization: {results['mean_bins']:.3f}")
    print(f"Mean Steps: {results['mean_steps']:.1f}")
    print(f"{'='*60}\n")
    
    # Visualize best packing
    if visualize_best and best_env is not None and multi_bin:
        print(f"Visualizing best packing ({best_bins} bins, all items packed)")
        for i, bin in enumerate(best_env.bins[:3]):  # Show first 3 bins
            bin.plot3d(title=f"Best - Bin {i+1}/{len(best_env.bins)} - Items: {len(bin.placed)}")
    elif visualize_best and not multi_bin:
        env.C.plot3d(title=f"Instance {thpack_instance.problem_id}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check if user wants to train on thpack9
    if len(sys.argv) > 1 and sys.argv[1] == "thpack":
        from thpack_loader import load_thpack9, print_instance_info
        
        thpack_file = sys.argv[2] if len(sys.argv) > 2 else "thpack9.txt"
        instance_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 200
        
        instances = load_thpack9(thpack_file)
        instance = instances[instance_id - 1]  # 1-indexed
        
        print_instance_info(instance)
        
        # Train
        agent, env, history = train_thpack_instance(
            thpack_instance=instance,
            episodes=episodes,
            max_actions=128,
            topk_eps=48,
            save_path=f"thpack_inst{instance_id}_model.pt",
            shuffle_items=True
        )
        
        # Evaluate
        results = evaluate_thpack_instance(
            agent=agent,
            thpack_instance=instance,
            n_episodes=10,
            visualize_best=True
        )
        
    else:
        # Original test with varying items
        print("Training with VARYING items (original behavior):")
        print("="*60)
        agent, env = train_pack_dqn(
            episodes=500, 
            n_items=50, 
            W=20, D=20, H=20, 
            seed=42, 
            max_actions=50, 
            topk_eps=1000,
            train_freq=1,
            num_train_steps=1,
            log_interval=10,
            save_path="dqn_packing_model.pt"
        )
        
        print("\n" + "="*60)
        print("To train on thpack9 instances:")
        print("  python packing_with_dqncore_potential.py thpack thpack9.txt 1 200")
        print("  Arguments: thpack <file> <instance_id> <episodes>")
        print("="*60)