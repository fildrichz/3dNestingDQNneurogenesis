"""
Multi-Bin Packing Environment s

- Creates all bins upfront based on problem.max_bins
- Agent must decide WHICH bin to pack into
- Reward penalizes using more bins (minimize bins used)
- Tracks per-bin and overall statistics
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from nesting.packing_core_enhanced import Container, box3d
from dqn_core.dqn import DQNAgent, DQNConfig
from nesting.dataset_loader import load_problem, BinPackingProblem

def volume(size: Tuple[int,int,int]) -> int:
    w,d,h = size; return int(w)*int(d)*int(h)


class MultiBinPackingEnv:
    """
    Multi-bin packing environment with fixed number of bins from dataset.
    
    Key features:
    - Respects max_bins from problem definition
    - All bins created upfront
    - Agent chooses which bin to pack into
    - Minimizes bins used while maximizing utilization
    """
    def __init__(self, W=40, D=40, H=40, items: List[Tuple[int,int,int,int,int]] = None, 
                 max_actions: int = 128, topk_eps: int = 32, seed: int = 0, gamma: float = 0.992,
                 max_weight: Optional[int] = None, problem: Optional[BinPackingProblem] = None):
        """
        Multi-bin packing environment.
        
        Args:
            problem: BinPackingProblem with max_bins specification
        """
        self.rng = np.random.default_rng(seed)
        self.bin_size = (int(W), int(D), int(H))
        self.bin_volume = int(W*D*H)
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.topk_eps = int(topk_eps)
        
        # Get max_bins from problem
        if problem is not None:
            self.max_bins = problem.max_bins
            self.max_weight = problem.max_weight
            self.incompatibilities = problem.incompatibilities
            self.positive_affinities = problem.positive_affinities
            self.center_of_mass_constraint = problem.center_of_mass
        else:
            self.max_bins = 1
            self.max_weight = max_weight
            self.incompatibilities = []
            self.positive_affinities = []
            self.center_of_mass_constraint = None
        
        self.problem = problem
        
        # Parse items
        self.initial_items = []
        if items is not None:
            for item in items:
                if len(item) == 3:
                    self.initial_items.append((*item, 0, -1))
                elif len(item) == 5:
                    self.initial_items.append(item)
                else:
                    raise ValueError(f"Item must be (w,d,h) or (w,d,h,weight,item_id)")
        
        self.reset()

    def _create_bin(self) -> Container:
        """Create a new empty bin."""
        W, D, H = self.bin_size
        adaptive_resolution = max(1, min(W, D) // 20)
        
        C = Container(*self.bin_size, max_weight=self.max_weight, 
                     resolution=adaptive_resolution)
        if self.problem is not None:
            C.set_constraints(
                incompatibilities=self.incompatibilities,
                positive_affinities=self.positive_affinities,
                center_of_mass=self.center_of_mass_constraint
            )
        C.eps = [(0,0,0)]
        C.placed = []
        return C

    def reset(self, items: List[Tuple[int,int,int,int,int]] = None):
        """Reset environment with new items."""
        if items is not None:
            self.items = list(items)
        else:
            self.items = list(self.initial_items)
        
        self.n_items = len(self.items)
        
        # Create ALL bins upfront based on max_bins
        self.bins: List[Container] = [self._create_bin() for _ in range(self.max_bins)]
        
        # Track which bins have been used (have items in them)
        self.bins_used_mask = np.zeros(self.max_bins, dtype=bool)
        
        self.total_placed_volume = 0
        self.done = False
        return self._obs()

    def _get_bins_used(self) -> int:
        """Get number of bins that have items in them."""
        return sum(1 for bin in self.bins if len(bin.placed) > 0)

    def _obs(self) -> np.ndarray:
        """
        Observation includes:
        - Overall utilization across all available bins
        - Items remaining
        - Stats from the bin with most remaining capacity
        - Number of bins already used
        """
        W, D, H = self.bin_size
        
        # Overall stats
        total_available_volume = self.max_bins * self.bin_volume
        packed = self.total_placed_volume / total_available_volume
        items_left = len(self.items) / max(1, self.n_items)
        
        # Find bin with most remaining capacity (best candidate for next placement)
        best_bin = None
        best_capacity = -1
        for bin in self.bins:
            remaining_vol = self.bin_volume - sum(b.w * b.d * b.h for b in bin.placed)
            if remaining_vol > best_capacity:
                best_capacity = remaining_vol
                best_bin = bin
        
        # Stats from best candidate bin
        if best_bin and best_bin.eps:
            caps = [best_bin.ep_rs.get(ep, (W-ep[0], D-ep[1], H-ep[2])) for ep in best_bin.eps]
            cx = max(c[0] for c in caps) / W
            cy = max(c[1] for c in caps) / D
            cz = max(c[2] for c in caps) / H
            
            hm_stats = best_bin.get_heightmap_stats()
            avg_h_norm = hm_stats['avg_height'] / H
            
            weight_util = 0.0
            if self.max_weight is not None and self.max_weight > 0:
                weight_util = best_bin.current_weight / self.max_weight
        else:
            cx = cy = cz = 0.0
            avg_h_norm = 0.0
            weight_util = 0.0
        
        # Number of bins used
        bins_used = self._get_bins_used()
        bins_used_norm = bins_used / self.max_bins
        
        return np.array([
            packed,           # Overall utilization
            items_left,       # Fraction of items remaining
            cx, cy, cz,       # Best bin residual caps
            avg_h_norm,       # Best bin height
            weight_util,      # Best bin weight util
            bins_used_norm,   # Fraction of bins used
        ], dtype=np.float32)

    def enumerate_actions(self):
        """
        Enumerate all feasible actions across ALL bins.
        Each action includes which bin it's for.
        """
        actions = []
        
        for bin_idx, bin in enumerate(self.bins):
            eps_sorted = sorted(set(bin.eps), key=lambda p:(p[2], p[1], p[0]))[:self.topk_eps]
            
            for ep_idx, ep in enumerate(eps_sorted):
                for item_idx, item in enumerate(self.items):
                    w, d, h, weight, item_id = item
                    rots = ((w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w))
                    
                    for rot_idx, size in enumerate(rots):
                        if (bin._fits_caps(ep, size) and 
                            bin._fits_container(ep, size) and 
                            bin._fits_collision_free(ep, size) and
                            bin.check_weight_constraint(weight) and
                            bin.check_incompatibility(item_id)):
                            
                            actions.append((bin_idx, item_idx, ep_idx, rot_idx, ep, size, weight, item_id))
        
        return actions

    def action_space(self):
        """Get current action space across all bins."""
        all_actions = self.enumerate_actions()
        A = len(all_actions)
        
        if A == 0:
            return [], np.zeros((0,), dtype=np.float32)
        
        if A > self.max_actions:
            idxs = self.rng.choice(A, size=self.max_actions, replace=False)
            actions = [all_actions[i] for i in idxs]
            mask = np.ones((self.max_actions,), dtype=np.float32)
        else:
            actions = list(all_actions)
            pad = self.max_actions - A
            actions.extend([None]*pad)
            mask = np.zeros((self.max_actions,), dtype=np.float32)
            mask[:A] = 1.0
        
        return actions, mask

    def step(self, action):
        """Execute action in specified bin."""
        if self.done:
            raise RuntimeError("Episode done, reset required.")
        
        info = {}
        
        # Calculate utilization (across all available bins)
        total_available_volume = self.max_bins * self.bin_volume
        util_prev = self.total_placed_volume / total_available_volume
        
        # If no action (no feasible placements anywhere)
        if action is None:
            self.done = True
            reward = (self.gamma * util_prev) - util_prev
            
            bins_used = self._get_bins_used()
            items_placed = self.n_items - len(self.items)
            
            # Reward based on:
            # 1. Utilization of space used
            # 2. Penalty for using more bins
            # 3. Penalty for unplaced items
            
            if len(self.items) == 0:
                # All items placed - bonus based on efficiency
                bins_efficiency = 1.0 - (bins_used / self.max_bins)
                reward += 0.3 + bins_efficiency * 0.2
            else:
                # Items remaining - penalty
                reward -= 0.2 * (len(self.items) / self.n_items)
            
            info["utilization"] = util_prev
            info["bins_used"] = bins_used
            info["items_placed"] = items_placed
            info["items_remaining"] = len(self.items)
            
            # Per-bin stats
            info["per_bin_utils"] = []
            info["per_bin_weights"] = []
            info["per_bin_items"] = []
            for bin in self.bins:
                if len(bin.placed) > 0:
                    bin_util = sum(b.w * b.d * b.h for b in bin.placed) / self.bin_volume
                    info["per_bin_utils"].append(bin_util)
                    info["per_bin_weights"].append(bin.current_weight)
                    info["per_bin_items"].append(len(bin.placed))
            
            return self._obs(), reward, self.done, info
        
        # Place item in specified bin
        bin_idx, item_idx, ep_idx, rot_idx, pos, size, weight, item_id = action
        target_bin = self.bins[bin_idx]
        
        ok = target_bin.place_at(pos, size, weight=weight, item_id=item_id)
        
        if not ok:
            self.done = True
            return self._obs(), -1.0, True, {"invalid": True}
        
        # Update tracking
        v = int(size[0] * size[1] * size[2])
        self.total_placed_volume += v
        del self.items[item_idx]
        
        # Calculate new utilization
        util_next = self.total_placed_volume / total_available_volume
        
        # Potential-based shaping
        reward = (self.gamma * util_next) - util_prev
        
        # Small bonus for placing in a bin with fewer items (encourage balancing)
        current_bin_items = len(target_bin.placed)
        avg_items_per_used_bin = sum(len(b.placed) for b in self.bins) / max(1, self._get_bins_used())
        if current_bin_items < avg_items_per_used_bin:
            reward += 0.01  # Small bonus for balancing
        
        # Check if done
        if len(self.items) == 0:
            self.done = True
            bins_used = self._get_bins_used()
            
            # Completion bonus with efficiency multiplier
            bins_efficiency = 1.0 - (bins_used / self.max_bins)
            reward += 0.3 + bins_efficiency * 0.2
            
            info["utilization"] = util_next
            info["bins_used"] = bins_used
            info["items_placed"] = self.n_items
            info["items_remaining"] = 0
            
            # Per-bin stats
            info["per_bin_utils"] = []
            info["per_bin_weights"] = []
            info["per_bin_items"] = []
            for bin in self.bins:
                if len(bin.placed) > 0:
                    bin_util = sum(b.w * b.d * b.h for b in bin.placed) / self.bin_volume
                    info["per_bin_utils"].append(bin_util)
                    info["per_bin_weights"].append(bin.current_weight)
                    info["per_bin_items"].append(len(bin.placed))
        
        return self._obs(), reward, self.done, info


# Action features now include bin_idx
ACTION_FEAT_DIM = 25  # +1 for bin_idx

def _safe_caps(C, ep):
    """Get residual capacities with fallback."""
    x, y, z = ep
    return C.ep_rs.get(ep, (C.w - x, C.d - y, C.h - z))


def build_action_features(env: MultiBinPackingEnv, actions):
    """Build action features for multi-bin environment."""
    W, D, H = env.bin_size
    binV = float(env.bin_volume)
    rows = []
    
    for a in actions:
        if a is None:
            rows.append([0.0] * ACTION_FEAT_DIM)
            continue
        
        bin_idx, item_idx, ep_idx, rot_idx, ep, size, weight, item_id = a
        target_bin = env.bins[bin_idx]
        
        iw, id_, ih, _, _ = env.items[item_idx]
        rw, rd, rh = size
        ex, ey, ez = ep
        cx, cy, cz = _safe_caps(target_bin, ep)
        
        # Normalized features
        rw_n, rd_n, rh_n = rw/W, rd/D, rh/H
        ex_n, ey_n, ez_n = ex/W, ey/D, ez/H
        cx_n, cy_n, cz_n = cx/W, cy/D, cz/H
        
        sx, sy, sz = max(0, cx-rw), max(0, cy-rd), max(0, cz-rh)
        sx_n, sy_n, sz_n = sx/W, sy/D, sz/H
        
        vol = float(rw * rd * rh)
        delta_u = vol / binV
        tight = float((sx==0) + (sy==0) + (sz==0))
        
        # Heightmap features
        h_at_ep_norm = 0.0
        local_avg = 0.0
        try:
            h_at_ep = target_bin.get_heightmap_at(ex, ey)
            h_at_ep_norm = h_at_ep / H
            
            resolution = target_bin.resolution
            gx = min(ex // resolution, target_bin.heightmap.shape[0] - 1)
            gy = min(ey // resolution, target_bin.heightmap.shape[1] - 1)
            
            local_heights = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    ngx, ngy = gx + dx, gy + dy
                    if 0 <= ngx < target_bin.heightmap.shape[0] and 0 <= ngy < target_bin.heightmap.shape[1]:
                        local_heights.append(target_bin.heightmap[ngx, ngy])
            
            if local_heights:
                local_avg = np.mean(local_heights) / H
        except:
            pass
        
        # Weight features
        item_weight_ratio = weight / env.max_weight if (env.max_weight and env.max_weight > 0) else 0.0
        remaining_weight_capacity = 0.0
        if env.max_weight is not None and env.max_weight > 0:
            remaining_weight_capacity = (env.max_weight - target_bin.current_weight) / env.max_weight
        
        # Bin features
        bin_idx_norm = bin_idx / env.max_bins
        bin_current_util = sum(b.w * b.d * b.h for b in target_bin.placed) / binV
        
        row = [
            iw/W, id_/D, ih/H,        # 0-2
            rw_n, rd_n, rh_n,          # 3-5
            ex_n, ey_n, ez_n,          # 6-8
            cx_n, cy_n, cz_n,          # 9-11
            sx_n, sy_n, sz_n,          # 12-14
            delta_u, tight,            # 15-16
            W/D, D/H,                  # 17-18
            h_at_ep_norm,              # 19
            local_avg,                 # 20
            item_weight_ratio,         # 21
            remaining_weight_capacity, # 22
            bin_idx_norm,              # 23: which bin (0 to max_bins-1, normalized)
            bin_current_util,          # 24: how full is this bin already
        ]
        
        rows.append(row)
    
    return np.asarray(rows, dtype=np.float32)


def pad_feats_mask(feats: np.ndarray, mask_short: np.ndarray, maxA: int):
    """Pad action features and mask."""
    A = feats.shape[0]
    F = np.zeros((maxA, ACTION_FEAT_DIM), np.float32)
    M = np.zeros((maxA,), np.float32)
    F[:A] = feats
    M[:A] = mask_short[:A] if mask_short.shape[0] >= A else 1.0
    return F, M


def load_problem_as_items(problem):
    """Convert problem items to env format."""
    items = []
    for type_idx, item in enumerate(problem.items):
        item_id = getattr(item, 'item_id', getattr(item, 'id', type_idx))
        for _ in range(item.quantity):
            items.append((item.length, item.width, item.height, item.weight, item_id))
    return items


def train_multibin_pack_dqn(
    problem_path: str,
    episodes: int = 1000,
    seed: int = 42,
    max_actions: int = 128,
    topk_eps: int = 1000,
    train_freq: int = 1,
    num_train_steps: int = 1,
    log_interval: int = 10,
    save_path: str = None
):
    """
    Train DQN for multi-bin packing with proper bin limit handling.
    """
    import os
    
    print(f"\n{'='*80}")
    print(f"MULTI-BIN PACKING DQN TRAINING")
    print(f"{'='*80}")
    print(f"Loading: {problem_path}")
    
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)
    
    W, D, H = problem.bin_dimensions
    
    print(f"\n📦 PROBLEM SPECIFICATION:")
    print(f"   Container: {W}×{D}×{H} (volume: {W*D*H:,})")
    print(f"   Max weight per bin: {problem.max_weight}")
    print(f"   Max bins available: {problem.max_bins}")
    print(f"   Items to pack: {len(items)} items from {len(problem.items)} types")
    print(f"   Total volume needed: {sum(i[0]*i[1]*i[2] for i in items):,}")
    print(f"   Total weight needed: {sum(i[3] for i in items):,}")
    print(f"   Episodes: {episodes}")
    print(f"{'='*80}\n")
    
    # Create multi-bin environment
    env = MultiBinPackingEnv(
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
    
    print(f"Observation dim: {OBS_DIM}")
    print(f"Action feature dim: {ACTION_FEAT_DIM}")
    
    # DQN config
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        gamma=0.992,
        lr=1e-4,
        batch_size=128,
        buffer_size=400_000,
        eps_start=1.0,
        eps_end=0.15,
        eps_decay_steps=episodes * 30,
        target_update_interval=500,
        n_step=3,
        double_dqn=True,
        warmup_steps=1500,
    )
    agent = DQNAgent(cfg)
    
    # Tracking
    import collections
    import copy
    util_hist = collections.deque(maxlen=50)
    bins_hist = collections.deque(maxlen=50)
    items_hist = collections.deque(maxlen=50)
    returns_hist = collections.deque(maxlen=50)
    best_bins = float('inf')
    best_items = 0
    best_util = 0.0
    best_solution = None  # Will store the best bins configuration
    
    print(f"Agent initialized on {cfg.device}")
    print(f"Starting training...\n")
    
    for ep in range(episodes):
        obs = env.reset(items=items.copy())
        ep_ret = 0.0
        steps = 0
        losses = []
        
        while True:
            actions, mask_short = env.action_space()
            feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            
            act_idx = agent.select_action(
                obs,
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
            )
            act = None if (act_idx is None or actions == [] or actions[act_idx] is None) else actions[act_idx]
            
            currF, currM = pad_feats_mask(
                feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            
            nobs, rew, done, info = env.step(act)
            
            n_actions, n_mask_short = env.action_space()
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
                util = info.get("utilization", 0.0)
                bins_used = info.get("bins_used", 0)
                items_placed = info.get("items_placed", 0)
                items_remaining = info.get("items_remaining", 0)
                
                util_hist.append(util)
                bins_hist.append(bins_used)
                items_hist.append(items_placed)
                returns_hist.append(ep_ret)
                
                # Track best solution based on utilization (greatest utilization)
                if util > best_util:
                    best_util = util
                    best_items = items_placed
                    best_bins = bins_used
                    # Deep copy the bins to save best solution
                    best_solution = copy.deepcopy(env.bins)
                
                if (ep + 1) % log_interval == 0 or ep == 0:
                    ma_util = np.mean(util_hist) if len(util_hist) > 0 else util
                    ma_bins = np.mean(bins_hist) if len(bins_hist) > 0 else bins_used
                    ma_items = np.mean(items_hist) if len(items_hist) > 0 else items_placed
                    avg_loss = np.mean(losses) if losses else 0.0
                    
                    # Main log line
                    print(f"Ep {ep+1:4d}/{episodes} | "
                          f"Bins: {bins_used:2d}/{env.max_bins} (MA:{ma_bins:.1f}) | "
                          f"Items: {items_placed:2d}/{len(items)} ({items_placed/len(items)*100:.0f}%) | "
                          f"Util: {util:.3f} (MA:{ma_util:.3f}) | "
                          f"Best: {best_bins}bins/{best_items}items | "
                          f"ε:{agent.epsilon():.3f} | L:{avg_loss:.4f}")
                    
                    # Per-bin breakdown every 50 episodes
                    if (ep + 1) % 50 == 0 and "per_bin_utils" in info:
                        print(f"  └─ Bins breakdown:")
                        for i, (u, w, n) in enumerate(zip(info["per_bin_utils"], 
                                                          info["per_bin_weights"], 
                                                          info["per_bin_items"])):
                            print(f"     Bin {i+1}: {n:2d}items | Vol:{u:.3f} | Wgt:{w:4d}/{env.max_weight}")
                
                break
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best result: {best_bins} bins used, {best_items}/{len(items)} items placed, util: {best_util:.3f}")
    print(f"Final MA50: {np.mean(bins_hist):.1f} bins, {np.mean(items_hist):.1f} items, {np.mean(util_hist):.3f} util")
    print(f"{'='*80}\n")
    
    if save_path:
        os.makedirs("output_data", exist_ok=True)
        if not save_path.startswith("output_data/"):
            save_path = os.path.join("output_data", os.path.basename(save_path))
        agent.save(save_path)
        print(f"Model saved to: {save_path}\n")
    
    # Visualize best solution (use last solution as fallback if no best was found)
    bins_to_visualize = best_solution if best_solution is not None else env.bins
    print(f"Visualizing {'best' if best_solution is not None else 'final'} packing...")
    for i, bin in enumerate(bins_to_visualize):
        if len(bin.placed) > 0:
            bin_util = sum(b.w * b.d * b.h for b in bin.placed) / env.bin_volume
            
            # Save version WITH extreme points (for debugging/analysis)
            bin.plot3d(
                title=f"Bin {i+1}/{env.max_bins} ({len(bin.placed)} items, util:{bin_util:.3f}) [with EPs]",
                save_path=f"output_data/best_bin_{i+1}_with_eps.png",
                show=False
            )
            
            # Save version WITHOUT extreme points (clean, filled boxes only)
            # Temporarily store and clear eps for clean visualization
            


            bin.plot3d_filled(
                title=f"Bin {i+1}/{env.max_bins} ({len(bin.placed)} items, util:{bin_util:.3f})",
                save_path=f"output_data/best_bin_{i+1}_filled.png",
                show=False
            )
            # Restore eps

    
    return agent, env, best_solution


def full_datapath(filename: str) -> str:
    """Helper for dataset path."""
    return "nesting\\inputData\\Benchmark dataset and instance generator for Real-World 3dBPP\\Input\\" + filename


if __name__ == "__main__":
    print("MULTI-BIN PACKING DQN")
    
    problem = "3dBPP_3.txt"

    agent, env, best_solution = train_multibin_pack_dqn(
        problem_path=full_datapath(problem),
        episodes=50,
        seed=42,
        max_actions=128,
        topk_eps=1000,
        train_freq=1,
        num_train_steps=1,
        log_interval=10,
        save_path=f"multibin_dqn_{problem.replace('.txt','')}.pth"
    )
    
    # Use best solution for final statistics if available
    bins_for_stats = best_solution if best_solution is not None else env.bins
    
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS (BEST SOLUTION):")
    print(f"{'='*80}")
    print(f"Max bins allowed: {env.max_bins}")
    
    # Calculate stats from best solution
    bins_used = sum(1 for bin in bins_for_stats if len(bin.placed) > 0)
    items_placed = sum(len(bin.placed) for bin in bins_for_stats)
    total_volume = sum(sum(b.w * b.d * b.h for b in bin.placed) for bin in bins_for_stats)
    overall_util = total_volume / (env.max_bins * env.bin_volume)
    
    print(f"Bins actually used: {bins_used}")
    print(f"Items placed: {items_placed}/{env.n_items}")
    print(f"Overall utilization: {overall_util:.3f}")
    print(f"\nPer-bin breakdown:")
    for i, bin in enumerate(bins_for_stats):
        if len(bin.placed) > 0:
            util = sum(b.w * b.d * b.h for b in bin.placed) / env.bin_volume
            print(f"  Bin {i+1}: {len(bin.placed)} items | Vol:{util:.3f} | Wgt:{bin.current_weight}/{env.max_weight}")
    print(f"{'='*80}")