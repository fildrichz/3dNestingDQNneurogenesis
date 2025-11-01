import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Import enhanced modules
from nesting.packing_core_enhanced import Container, box3d
from dqn_core.dqn import DQNAgent, DQNConfig
from nesting.dataset_loader import load_problem, BinPackingProblem, Item

# ---------------------
# Lightweight RL Env around Container with enhanced features
# ---------------------
def volume(size: Tuple[int,int,int]) -> int:
    w,d,h = size; return int(w)*int(d)*int(h)

class PackingEnv:
    def __init__(self, W=40, D=40, H=40, items: List[Tuple[int,int,int,int,int]] = None, 
                 max_actions: int = 128, topk_eps: int = 32, seed: int = 0, gamma: float = 0.992,
                 max_weight: Optional[int] = None, problem: Optional[BinPackingProblem] = None):
        """
        Enhanced PackingEnv with constraint support.
        
        Args:
            W, D, H: Container dimensions
            items: List of (length, width, height, weight, item_id) tuples
            max_actions: Maximum number of actions in action space
            topk_eps: Maximum extreme points to consider
            seed: Random seed
            gamma: Discount factor for reward shaping
            max_weight: Maximum weight constraint (optional)
            problem: BinPackingProblem instance with full constraints (optional)
        """
        self.rng = np.random.default_rng(seed)
        self.bin_size = (int(W), int(D), int(H))
        self.bin_volume = int(W*D*H)
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.topk_eps = int(topk_eps)
        
        # Handle problem constraints
        self.max_weight = max_weight
        self.problem = problem
        if problem is not None:
            self.max_weight = problem.max_weight
            self.incompatibilities = problem.incompatibilities
            self.positive_affinities = problem.positive_affinities
            self.center_of_mass_constraint = problem.center_of_mass
        else:
            self.incompatibilities = []
            self.positive_affinities = []
            self.center_of_mass_constraint = None
        
        # Parse initial items (now with weight and item_id)
        self.initial_items = []
        if items is not None:
            for item in items:
                if len(item) == 3:
                    # Old format: (w, d, h) -> add weight=0, item_id=-1
                    self.initial_items.append((*item, 0, -1))
                elif len(item) == 5:
                    # New format: (w, d, h, weight, item_id)
                    self.initial_items.append(item)
                else:
                    raise ValueError(f"Item must be (w,d,h) or (w,d,h,weight,item_id), got {item}")
        
        self.reset()

    def reset(self, items: List[Tuple[int,int,int,int,int]] = None):
        """Reset environment with new items."""
        if items is not None:
            self.items = list(items)
        else:
            self.items = list(self.initial_items)
        
        self.n_items = len(self.items)
        
        # Create container with constraints
        self.C = Container(*self.bin_size, max_weight=self.max_weight, resolution=10)
        if self.problem is not None:
            self.C.set_constraints(
                incompatibilities=self.incompatibilities,
                positive_affinities=self.positive_affinities,
                center_of_mass=self.center_of_mass_constraint
            )
        
        # Initialize EPs
        self.C.eps = [(0,0,0)]
        self.C.placed = []
        self.total_placed_volume = 0
        self.done = False
        return self._obs()

    def _feasible(self, ep, size, weight, item_id) -> bool:
        """Check if placement is feasible considering all constraints."""
        return (self.C._fits_caps(ep, size) and 
                self.C._fits_container(ep, size) and 
                self.C._fits_collision_free(ep, size) and
                self.C.check_weight_constraint(weight) and
                self.C.check_incompatibility(item_id))

    def enumerate_actions(self):
        """Enumerate all feasible actions with constraint checking."""
        actions = []
        eps_sorted = sorted(set(self.C.eps), key=lambda p:(p[2], p[1], p[0]))[:self.topk_eps]
        for ep_idx, ep in enumerate(eps_sorted):
            for item_idx, item in enumerate(self.items):
                w, d, h, weight, item_id = item
                # 6 axis-aligned rotations
                rots = ((w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w))
                for rot_idx, size in enumerate(rots):
                    if self._feasible(ep, size, weight, item_id):
                        actions.append((item_idx, ep_idx, rot_idx, ep, size, weight, item_id))
        return actions, eps_sorted

    def action_space(self):
        """Get current action space with random subsampling if needed."""
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
        """Enhanced observation including heightmap statistics."""
        W,D,H = self.bin_size
        packed = self.total_placed_volume / self.bin_volume
        items_left = len(self.items) / max(1, self.n_items)
        
        # Residual caps among EPs
        if self.C.eps:
            caps = [self.C.ep_rs.get(ep, (W-ep[0], D-ep[1], H-ep[2])) for ep in self.C.eps]
            cx = max(c[0] for c in caps)/W
            cy = max(c[1] for c in caps)/D
            cz = max(c[2] for c in caps)/H
        else:
            cx=cy=cz=0.0
        
        # Heightmap statistics
        hm_stats = self.C.get_heightmap_stats()
        max_h_norm = hm_stats['max_height'] / H
        avg_h_norm = hm_stats['avg_height'] / H
        std_h_norm = hm_stats['std_height'] / H
        
        # Weight utilization
        weight_util = 0.0
        if self.max_weight is not None and self.max_weight > 0:
            weight_util = self.C.current_weight / self.max_weight
        
        return np.array([
            packed, items_left, cx, cy, cz,
            max_h_norm, avg_h_norm, std_h_norm,
            weight_util
        ], dtype=np.float32)

    def step(self, action):
        """Execute action in environment."""
        if self.done:
            raise RuntimeError("Episode done, reset required.")
        info = {}

        # Potential Phi(s) = utilization(s)
        util_prev = self.total_placed_volume / self.bin_volume

        # Agent decides to stop (or no feasible actions chosen)
        if action is None:
            self.done = True
            reward = (self.gamma * util_prev) - util_prev
            reward += util_prev  # terminal bonus
            info["utilization"] = util_prev
            info["weight_utilization"] = self.C.current_weight / self.max_weight if self.max_weight else 0.0
            info["items_placed"] = len(self.C.placed)
            return self._obs(), reward, self.done, info

        # Try to place
        item_idx, ep_idx, rot_idx, pos, size, weight, item_id = action
        ok = self.C.place_at(pos, size, weight=weight, item_id=item_id)
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
            info["weight_utilization"] = self.C.current_weight / self.max_weight if self.max_weight else 0.0
            info["items_placed"] = len(self.C.placed)

        return self._obs(), reward, self.done, info


# ---------------------
# Enhanced Feature utils for DQN with heightmap and constraints
# ---------------------
ACTION_FEAT_DIM = 32  # Increased from 19 to include heightmap and constraint features

def _safe_caps(C, ep):
    """Get residual capacities with fallback to container bounds"""
    x,y,z = ep
    return C.ep_rs.get(ep, (C.w - x, C.d - y, C.h - z))

def build_action_features(env: PackingEnv, actions):
    """
    Build enhanced action features including heightmap and constraint information.
    
    Features (32 dimensions):
    - Item original dims (3)
    - Chosen rotation (3)
    - EP position (3)
    - Residual caps (3)
    - Slack (3)
    - Reward hints (2)
    - Bin aspect ratios (2)
    - Heightmap at EP (4): height, local avg, local max, local std
    - Weight features (2): item weight ratio, remaining capacity ratio
    - Constraint features (7): has_affinity, num_affinities, has_incompatibility, 
                                num_incompatibilities, weight_fraction, 
                                is_same_type_in_bin, num_same_type_in_bin
    """
    W,D,H = env.bin_size
    binV = float(env.bin_volume)
    rows = []
    
    for a in actions:
        if a is None:
            rows.append([0.0]*ACTION_FEAT_DIM)
            continue
        
        item_idx, ep_idx, rot_idx, ep, size, weight, item_id = a
        iw, id_, ih, _, _ = env.items[item_idx]
        rw, rd, rh = size
        ex, ey, ez = ep
        cx, cy, cz = _safe_caps(env.C, ep)
        
        # Original features (normalized sizes/pos/caps)
        rw_n, rd_n, rh_n = rw/W, rd/D, rh/H
        ex_n, ey_n, ez_n = ex/W, ey/D, ez/H
        cx_n, cy_n, cz_n = cx/W, cy/D, cz/H
        
        # Slack (cap - size), normalized
        sx, sy, sz = max(0,cx-rw), max(0,cy-rd), max(0,cz-rh)
        sx_n, sy_n, sz_n = sx/W, sy/D, sz/H
        
        # Reward hints
        vol = float(rw*rd*rh)
        delta_u = vol / binV
        tight = float((sx==0) + (sy==0) + (sz==0))
        
        # --- NEW: Heightmap features at EP ---
        try:
            # Get heightmap value at EP
            h_at_ep = env.C.get_heightmap_at(ex, ey)
            h_at_ep_norm = h_at_ep / H
            
            # Get heightmap statistics in local region around EP
            resolution = env.C.resolution
            gx = min(ex // resolution, env.C.heightmap.shape[0] - 1)
            gy = min(ey // resolution, env.C.heightmap.shape[1] - 1)
            
            # Local 3x3 region around EP (if available)
            local_heights = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    ngx = gx + dx
                    ngy = gy + dy
                    if 0 <= ngx < env.C.heightmap.shape[0] and 0 <= ngy < env.C.heightmap.shape[1]:
                        local_heights.append(env.C.heightmap[ngx, ngy])
            
            if local_heights:
                local_avg = np.mean(local_heights) / H
                local_max = np.max(local_heights) / H
                local_std = np.std(local_heights) / H
            else:
                local_avg = local_max = local_std = 0.0
        except:
            h_at_ep_norm = local_avg = local_max = local_std = 0.0
        
        # --- NEW: Weight features ---
        item_weight_ratio = weight / env.C.w if env.C.w > 0 else 0.0  # Normalize by container width as proxy
        remaining_weight_capacity = 0.0
        if env.max_weight is not None and env.max_weight > 0:
            remaining_weight_capacity = (env.max_weight - env.C.current_weight) / env.max_weight
        
        # --- NEW: Constraint features ---
        # Check if item has affinity with any items already in bin
        has_affinity = 0.0
        num_affinities = 0
        for aff_a, aff_b in env.positive_affinities:
            if item_id == aff_a and aff_b in env.C.item_ids_in_bin:
                has_affinity = 1.0
                num_affinities += 1
            elif item_id == aff_b and aff_a in env.C.item_ids_in_bin:
                has_affinity = 1.0
                num_affinities += 1
        
        # Check if item has incompatibility (should be 0 if action is feasible)
        has_incompatibility = 0.0
        num_incompatibilities = 0
        for inc_a, inc_b in env.incompatibilities:
            if item_id == inc_a and inc_b in env.C.item_ids_in_bin:
                has_incompatibility = 1.0
                num_incompatibilities += 1
            elif item_id == inc_b and inc_a in env.C.item_ids_in_bin:
                has_incompatibility = 1.0
                num_incompatibilities += 1
        
        # What fraction of total weight is this item?
        weight_fraction = 0.0
        if env.max_weight is not None and env.max_weight > 0:
            weight_fraction = weight / env.max_weight
        
        # Is same item type already in bin?
        is_same_type_in_bin = 1.0 if item_id in env.C.item_ids_in_bin else 0.0
        num_same_type_in_bin = sum(1 for b in env.C.placed if b.item_id == item_id)
        num_same_type_in_bin_norm = num_same_type_in_bin / max(1, len(env.C.placed))
        
        row = [
            # Original features (19)
            iw/W, id_/D, ih/H,           # 0-2: item original dims
            rw_n, rd_n, rh_n,            # 3-5: chosen rotation
            ex_n, ey_n, ez_n,            # 6-8: EP position
            cx_n, cy_n, cz_n,            # 9-11: residual caps
            sx_n, sy_n, sz_n,            # 12-14: slack
            delta_u, tight,              # 15-16: reward hints
            W/D, D/H,                    # 17-18: bin aspect ratios
            
            # New heightmap features (4)
            h_at_ep_norm,                # 19: height at EP
            local_avg,                   # 20: local avg height
            local_max,                   # 21: local max height  
            local_std,                   # 22: local height std
            
            # New weight features (2)
            item_weight_ratio,           # 23: item weight (normalized)
            remaining_weight_capacity,   # 24: remaining weight capacity
            
            # New constraint features (7)
            has_affinity,                # 25: has affinity with items in bin
            num_affinities / 10.0,       # 26: number of affinities (normalized)
            has_incompatibility,         # 27: has incompatibility (should be 0)
            num_incompatibilities / 10.0, # 28: number of incompatibilities (normalized)
            weight_fraction,             # 29: item weight as fraction of max
            is_same_type_in_bin,         # 30: same type already in bin
            num_same_type_in_bin_norm,   # 31: fraction of placed items of same type
        ]
        
        rows.append(row)
    
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
def make_items(n=12, lo=4, hi=10, seed=0, weight_range=(10, 50)):
    """Generate random items with weights and item IDs."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        w = int(rng.integers(lo, hi))
        d = int(rng.integers(lo, hi))
        h = int(rng.integers(lo, hi))
        weight = int(rng.integers(weight_range[0], weight_range[1]))
        item_id = i % 10  # Reuse item IDs to create some repeats
        items.append((w, d, h, weight, item_id))
    return items


def load_problem_as_items(problem: BinPackingProblem) -> List[Tuple[int,int,int,int,int]]:
    """
    Convert BinPackingProblem to list of items for PackingEnv.
    
    Returns:
        List of (length, width, height, weight, item_id) tuples
    """
    items = []
    for item in problem.items:
        for _ in range(item.quantity):
            items.append((item.length, item.width, item.height, item.weight, item.id))
    return items


# ---------------------
# Training functions
# ---------------------
def train_pack_dqn_from_problem(problem_path: str, episodes=100, seed=42,
                                 max_actions=128, topk_eps=48, train_freq=1,
                                 num_train_steps=1, log_interval=10, save_path=None):
    """
    Train DQN agent on a specific problem from the dataset.
    
    Args:
        problem_path: Path to .txt problem file
        episodes: Number of training episodes
        seed: Random seed
        max_actions: Maximum actions in action space
        topk_eps: Maximum extreme points to consider
        train_freq: Training frequency (steps)
        num_train_steps: Number of training updates per step
        log_interval: Logging interval (episodes)
        save_path: Path to save trained model
    """
    import os
    
    # Load problem
    print(f"\n{'='*60}")
    print(f"Loading problem from: {problem_path}")
    print(f"{'='*60}")
    
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)
    
    W, D, H = problem.bin_dimensions
    
    print(f"Container: {W}x{D}x{H} (volume={W*D*H})")
    print(f"Max weight: {problem.max_weight}")
    print(f"Items: {len(items)} total items from {len(problem.items)} types")
    print(f"Incompatibilities: {problem.incompatibilities}")
    print(f"Affinities: {problem.positive_affinities}")
    print(f"Center of mass: {problem.center_of_mass}")
    print(f"Episodes: {episodes}")
    print(f"{'='*60}\n")
    
    # Create environment with problem constraints
    env = PackingEnv(
        W, D, H, 
        items=items, 
        max_actions=max_actions, 
        topk_eps=topk_eps, 
        seed=seed, 
        gamma=0.992,
        problem=problem
    )
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    
    print(f"Observation dimension: {OBS_DIM}")
    print(f"Action feature dimension: {ACTION_FEAT_DIM}")
    
    # Create DQN agent
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        gamma=0.992,
        lr=2e-4,
        batch_size=64,
        buffer_size=200_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=episodes * 20,
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
    
    print(f"Agent initialized on {cfg.device}. Starting training...\n")
    
    for ep in range(episodes):
        # Reset with same items each time (single-bin problem)
        obs = env.reset(items=items.copy())
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
                weight_util = info.get("weight_utilization", 0.0)
                items_placed = info.get("items_placed", 0)
                
                util_hist.append(util)
                returns_hist.append(ep_ret)
                steps_hist.append(steps)
                best_util = max(best_util, util)
                
                # Logging
                if (ep + 1) % log_interval == 0 or ep == 0:
                    ma_util = np.mean(util_hist) if len(util_hist) > 0 else util
                    ma_ret = np.mean(returns_hist) if len(returns_hist) > 0 else ep_ret
                    avg_loss = np.mean(losses) if losses else 0.0
                    
                    print(f"Ep {ep+1:4d}/{episodes} | "
                          f"Steps: {steps:3d} | "
                          f"Util: {util:.3f} | "
                          f"WUtil: {weight_util:.3f} | "
                          f"Items: {items_placed}/{len(items)} | "
                          f"MA50: {ma_util:.3f} | "
                          f"Best: {best_util:.3f} | "
                          f"ε: {agent.epsilon():.3f} | "
                          f"Loss: {avg_loss:.4f}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best utilization: {best_util:.3f}")
    print(f"Final MA50 utilization: {np.mean(util_hist):.3f}")
    print(f"{'='*60}\n")
    
    # Save model
    if save_path:
        os.makedirs("output_data", exist_ok=True)
        if not save_path.startswith("output_data/"):
            save_path = os.path.join("output_data", os.path.basename(save_path))
        agent.save(save_path)
        print(f"Model saved to: {save_path}\n")
    
    # Visualize final episode
    env.C.plot3d(
        title=f"Final Packing (Util: {util:.3f}, Weight: {env.C.current_weight}/{env.max_weight})",
        save_path="output_data/final_packing.png" if save_path else None,
        show=False
    )
    env.C.visualize_heightmap(save_path="output_data/final_heightmap.png" if save_path else None)
    
    return agent, env


def full_datapath(filename: str) -> str:
    """Helper to get full path to dataset files."""
    # will get file name, need to add this C:\Users\Filip\Desktop\diplomovaPraceJanku\sdilenygit\dp-filip-spidla-spidlfil\main\nesting\inputData\Benchmark dataset and instance generator for Real-World 3dBPP\Input
    
    tofile = "nesting\inputData\Benchmark dataset and instance generator for Real-World 3dBPP\Input\\"

    return tofile + filename

if __name__ == "__main__":
    # Example 1: Train on a specific problem from the dataset
    print("Training DQN on 3dBPP_1.txt:")
    agent, env = train_pack_dqn_from_problem(
        problem_path= full_datapath("3dBPP_1.txt"),
        episodes=500,
        seed=42,
        max_actions=128,
        topk_eps=48,
        train_freq=1,
        num_train_steps=1,
        log_interval=10,
        save_path="dqn_3dbpp1_model.pt"
    )
    
    print("\n" + "="*60)
    print("Training on 3dBPP_11.txt with constraints:")
    agent2, env2 = train_pack_dqn_from_problem(
        problem_path= full_datapath("3dBPP_11.txt"),
        episodes=500,
        seed=42,
        max_actions=128,
        topk_eps=48,
        train_freq=1,
        num_train_steps=1,
        log_interval=10,
        save_path="dqn_3dbpp11_model.pt"
    )