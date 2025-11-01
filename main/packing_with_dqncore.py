import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# Import your core modules
from nesting.packing_core import Container
from dqn_core.dqn import DQNAgent, DQNConfig  # <-- uses fixed dqn.py

# ---------------------
# Lightweight RL Env around Container
# ---------------------
def volume(size: Tuple[int,int,int]) -> int:
    w,d,h = size; return int(w)*int(d)*int(h)

class PackingEnv:
    def __init__(self, W=40, D=40, H=40, items: List[Tuple[int,int,int]] = None, 
                 max_actions: int = 128, topk_eps: int = 32, seed: int = 0, gamma: float = 0.992):
        self.rng = np.random.default_rng(seed)
        self.bin_size = (int(W), int(D), int(H))
        self.bin_volume = int(W*D*H)
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.topk_eps = int(topk_eps)
        self.initial_items = list(items) if items is not None else []
        self.reset()

    def reset(self, items: List[Tuple[int,int,int]] = None):
        self.items = list(items if items is not None else self.initial_items)
        self.n_items = len(self.items)
        self.C = Container(*self.bin_size)
        # initialize EPs the way your core does; ensure at least origin exists
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
        packed = self.total_placed_volume / self.bin_volume
        items_left = len(self.items) / max(1, self.n_items)
        # quick geometry proxy: max residual caps among EPs
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

        # Potential Phi(s) = utilization(s)
        util_prev = self.total_placed_volume / self.bin_volume

        # Agent decides to stop (or no feasible actions chosen)
        if action is None:
            self.done = True
            # Potential-only shaping for a no-op (s' == s): r = gamma*util - util
            reward = (self.gamma * util_prev) - util_prev
            # Terminal bonus reinforces the end signal
            reward += util_prev
            info["utilization"] = util_prev
            return self._obs(), reward, self.done, info

        # Try to place
        item_idx, ep_idx, rot_idx, pos, size = action
        ok = self.C.place_at(pos, size)
        if not ok:
            # Should be rare (enumerate_actions already checked feasibility)
            self.done = True
            return self._obs(), -1.0, True, {"invalid": True}

        # Update packed volume and remove the item
        v = int(size[0] * size[1] * size[2])
        self.total_placed_volume += v
        del self.items[item_idx]

        # New potential after placement
        util_next = self.total_placed_volume / self.bin_volume

        # Potential-based shaping (Ng et al., 1999)
        reward = (self.gamma * util_next) - util_prev

        # Check termination: no feasible actions or no items left
        actions, eps_subset = self.enumerate_actions()
        if len(actions) == 0 or len(self.items) == 0:
            self.done = True
            reward += util_next  # terminal bonus
            info["utilization"] = util_next

        return self._obs(), reward, self.done, info


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
        device="cuda",
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
        device="cuda",
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


if __name__ == "__main__":
    # Quick test with varying items
    print("Training with VARYING items (more realistic):")
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
        save_path="dqn_packing_model.pt"  # Will be saved to ./output_data/
    )
    
    # Uncomment to test with fixed items (useful for debugging)
    # print("\n\nTraining with FIXED items (for analysis):")
    # agent_fixed, env_fixed = train_pack_dqn_fixed_items(
    #     episodes=100, 
    #     n_items=50, 
    #     W=20, D=20, H=20, 
    #     seed=42, 
    #     max_actions=50, 
    #     topk_eps=1000,
    #     log_interval=10,
    #     save_path="dqn_packing_fixed_model.pt"  # Will be saved to ./output_data/
    # )