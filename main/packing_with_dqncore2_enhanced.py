"""
Multi-Bin Packing Environment - ENHANCED VERSION

âœ… ENHANCEMENTS:
   1. Heightmap CNN: Processes 7x7 patches around each placement position
      - Learns spatial patterns (corners, walls, valleys)
      - Adds 64-dim spatial embedding to action features
   
   2. Transformer Attention: Actions reason about each other
      - Self-attention between all available actions
      - Learns action relationships (competition, alternatives)
      - Better strategic planning

Original features:
- Creates all bins upfront based on problem.max_bins
- Agent must decide WHICH bin to pack into
- Reward penalizes using more bins (minimize bins used)
- Tracks per-bin and overall statistics
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from nesting.packing_core_enhanced import Container, box3d
#from dqn_core.dqn_base import DQNAgent, DQNConfig  # Keep for reference
from dqn_core.dqn_enhanced import DQNAgentEnhanced, DQNConfigEnhanced  # NEW: Enhanced architecture
from nesting.heightmap_utils import extract_patches_for_actions, pad_patches  # NEW: Heightmap processing
from nesting.dataset_loader import load_problem, BinPackingProblem

def volume(size: Tuple[int,int,int]) -> int:
    w,d,h = size; return int(w)*int(d)*int(h)


class MultiBinPackingEnv:
    """
    Multi-bin packing environment with COMPLETE constraint enforcement.
    """
    def __init__(self, W=40, D=40, H=40, items: List[Tuple[int,int,int,int,int]] = None, 
                 max_actions: int = 128, topk_eps: int = 32, seed: int = 0, gamma: float = 0.992,
                 max_weight: Optional[int] = None, problem: Optional[BinPackingProblem] = None):
        self.rng = np.random.default_rng(seed)
        self.bin_size = (int(W), int(D), int(H))
        self.bin_volume = int(W*D*H)
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.topk_eps = int(topk_eps)
        
        # Store problem constraints
        if problem is not None:
            self.max_bins = problem.max_bins
            self.max_weight = problem.max_weight
            self.incompatibilities = problem.incompatibilities
            self.positive_affinities = problem.positive_affinities
            self.center_of_mass_constraint = problem.center_of_mass
            self.relative_pos = problem.relative_pos
        else:
            self.max_bins = 1
            self.max_weight = max_weight
            self.incompatibilities = []
            self.positive_affinities = []
            self.center_of_mass_constraint = None
            self.relative_pos = {}
        
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
        """Create a new empty bin with all constraints."""
        W, D, H = self.bin_size
        adaptive_resolution = max(1, min(W, D) // 20)

        C = Container(*self.bin_size, max_weight=self.max_weight,
                     resolution=adaptive_resolution, max_ems=50)
        if self.problem is not None:
            C.set_constraints(
                incompatibilities=self.incompatibilities,
                positive_affinities=self.positive_affinities,
                center_of_mass=self.center_of_mass_constraint,
                relative_pos=self.relative_pos
            )
        # EMS is initialized automatically in Container.__init__
        C.placed = []
        return C

    def reset(self, items: List[Tuple[int,int,int,int,int]] = None):
        """Reset environment with new items."""
        if items is not None:
            self.items = list(items)
        else:
            self.items = list(self.initial_items)
        
        self.n_items = len(self.items)
        
        # Create ALL bins upfront
        self.bins: List[Container] = [self._create_bin() for _ in range(self.max_bins)]
        
        self.bins_used_mask = np.zeros(self.max_bins, dtype=bool)
        self.total_placed_volume = 0
        self.done = False
        return self._obs()

    def _get_bins_used(self) -> int:
        """Get number of bins that have items in them."""
        return sum(1 for bin in self.bins if len(bin.placed) > 0)

    def _compute_ems_quality(self, bin) -> float:
        """
        Compute EMS quality for potential-based reward shaping.

        Uses cube root of AVERAGE of top-3 largest EMS volumes.
        This provides intermediate feedback about packing quality:
        - High quality: Large usable spaces remain (good for future placements)
        - Low quality: Fragmented space (hard to fit remaining items)

        Returns:
            float: Cube root of (average of top-3 EMS volumes / bin volume)
                   Range: 0.0 to ~1.0 (naturally bounded, comparable to utilization)
        """
        if not bin.ems_list:
            return 0.0

        # Get top-3 largest EMS by volume
        volumes = sorted([ems.volume() for ems in bin.ems_list], reverse=True)
        top_3 = volumes[:min(3, len(volumes))]

        # Average and normalize by bin volume (keeps it in 0-1 range like utilization)
        avg_top_3 = sum(top_3) / len(top_3)
        normalized_avg = avg_top_3 / self.bin_volume

        # Cube root to get diminishing returns for larger spaces
        # This makes the metric less sensitive to exactly which EMS split occurred
        ems_quality = normalized_avg ** (1.0/3.0)

        return ems_quality

    def _obs(self) -> np.ndarray:
        """Observation state."""
        W, D, H = self.bin_size
        
        total_available_volume = self.max_bins * self.bin_volume
        packed = self.total_placed_volume / total_available_volume
        items_left = len(self.items) / max(1, self.n_items)
        
        # Find bin with most remaining capacity
        best_bin = None
        best_capacity = -1
        for bin in self.bins:
            remaining_vol = self.bin_volume - sum(b.w * b.d * b.h for b in bin.placed)
            if remaining_vol > best_capacity:
                best_capacity = remaining_vol
                best_bin = bin
        
        if best_bin and best_bin.ems_list:
            # Get max capacity from largest EMS
            cx = max(ems.w for ems in best_bin.ems_list) / W
            cy = max(ems.d for ems in best_bin.ems_list) / D
            cz = max(ems.h for ems in best_bin.ems_list) / H
            
            hm_stats = best_bin.get_heightmap_stats()
            avg_h_norm = hm_stats['avg_height'] / H
            
            weight_util = 0.0
            if self.max_weight is not None and self.max_weight > 0:
                weight_util = best_bin.current_weight / self.max_weight
        else:
            cx = cy = cz = 0.0
            avg_h_norm = 0.0
            weight_util = 0.0
        
        bins_used = self._get_bins_used()
        bins_used_norm = bins_used / self.max_bins
        
        return np.array([
            packed,
            items_left,
            cx, cy, cz,
            avg_h_norm,
            weight_util,
            bins_used_norm,
        ], dtype=np.float32)

    def check_affinity_placement(self, item_id: int, target_bin_idx: int) -> bool:
        """
        PROACTIVE: Check if placing item_id in target_bin would violate affinity.
        
        This prevents splitting affinity pairs across bins by checking BEFORE placement.
        
        Args:
            item_id: ID of item to place
            target_bin_idx: Index of bin where we want to place it
            
        Returns:
            True if placement allowed, False if would violate affinity
        """
        if not self.positive_affinities:
            return True
        
        # Find all items that have affinity with this item
        affinity_partners = set()
        for a, b in self.positive_affinities:
            if item_id == a:
                affinity_partners.add(b)
            if item_id == b:
                affinity_partners.add(a)
        
        if not affinity_partners:
            return True  # No affinity constraints for this item
        
        # Check: are any affinity partners already placed in OTHER bins?
        for bin_idx, bin in enumerate(self.bins):
            if bin_idx == target_bin_idx:
                continue  # Placing in same bin is OK
            
            # Check if this bin contains any affinity partners
            for partner_id in affinity_partners:
                if partner_id in bin.item_ids_in_bin:
                    # VIOLATION: Partner already in different bin!
                    return False
        
        return True

    def enumerate_actions(self):
        """
        Enumerate all feasible actions using EMS with ALL constraint checks including:
        - Fits within EMS dimensions
        - Basic placement (fits, collision-free)
        - Weight constraint
        - Incompatibility
        - Relative positioning (heavy not on light)
        - PROACTIVE affinity (prevents splitting pairs)
        """
        actions = []

        for bin_idx, bin in enumerate(self.bins):
            # Sort EMS by volume (largest first) and take top-k
            ems_sorted = sorted(bin.ems_list, key=lambda e: (-e.volume(), e.z, e.y, e.x))[:self.topk_eps]

            for ems_idx, ems in enumerate(ems_sorted):
                for item_idx, item in enumerate(self.items):
                    w, d, h, weight, item_id = item
                    rots = ((w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w))

                    for rot_idx, size in enumerate(rots):
                        # Get EMS corner as placement position
                        ep = (ems.x, ems.y, ems.z)
                        w_rot, d_rot, h_rot = size

                        # CONSTRAINT CHECKS AT EMS POSITION (Z-independent checks)
                        # Note: Collision check OMITTED - EMS is empty by definition
                        # Relative positioning only checks XY footprint overlap (Z irrelevant)
                        if not (bin._fits_ems(ems, size) and
                                bin._fits_container(ep, size) and
                                bin.check_weight_constraint(weight) and
                                bin.check_incompatibility(item_id) and
                                bin.check_relative_positioning(item_id, ep, size) and
                                self.check_affinity_placement(item_id, bin_idx)):
                            continue

                        # CRITICAL: Check if box will fit after gravity is applied
                        # Gravity can drop the box onto tall boxes, potentially exceeding container height
                        final_z = bin.apply_gravity(ems.x, ems.y, ems.z, w_rot, d_rot, h_rot)
                        if final_z + h_rot > bin.h:
                            # Box would exceed container height after gravity - SKIP this action
                            continue

                        # All checks passed - add to valid actions
                        actions.append((bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id))

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
        
        total_available_volume = self.max_bins * self.bin_volume
        util_prev = self.total_placed_volume / total_available_volume
        
        # If no action (no feasible placements anywhere)
        if action is None:
            self.done = True
            reward = (self.gamma * util_prev) - util_prev
            
            bins_used = self._get_bins_used()
            items_placed = self.n_items - len(self.items)
            
            if len(self.items) == 0:
                bins_efficiency = 1.0 - (bins_used / self.max_bins)
                reward += 0.3 + bins_efficiency * 0.2
                
                # SAFETY NET: Check affinities (should rarely trigger with proactive check)
                affinity_violations = 0
                for bin in self.bins:
                    if len(bin.placed) > 0:
                        if not bin.check_positive_affinity_before_completion():
                            affinity_violations += 1
                
                if affinity_violations > 0:
                    reward -= 0.1 * affinity_violations  # Small penalty
                    info["affinity_violations"] = affinity_violations
                    print(f"âš ï¸  WARNING: Affinity violation despite proactive blocking!")
            else:
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
        bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id = action
        target_bin = self.bins[bin_idx]

        # === COMPUTE POTENTIAL BEFORE PLACEMENT ===
        # Potential-based shaping: Φ(s) = bin_utilization + β * EMS_quality
        # Use TARGET BIN's utilization (not global) for meaningful per-step rewards
        bin_vol_prev = sum(b.w * b.d * b.h for b in target_bin.placed)
        bin_util_prev = bin_vol_prev / self.bin_volume
        ems_quality_prev = self._compute_ems_quality(target_bin)
        potential_prev = bin_util_prev + 0.3 * ems_quality_prev

        # NOTE: place_at_ems applies gravity automatically and updates EMS!
        ok = target_bin.place_at_ems(ems, size, weight=weight, item_id=item_id)

        if not ok:
            self.done = True
            return self._obs(), -1.0, True, {"invalid": True}

        # Update tracking
        v = int(size[0] * size[1] * size[2])
        self.total_placed_volume += v
        del self.items[item_idx]

        util_next = self.total_placed_volume / total_available_volume

        # === COMPUTE POTENTIAL AFTER PLACEMENT ===
        bin_vol_next = sum(b.w * b.d * b.h for b in target_bin.placed)
        bin_util_next = bin_vol_next / self.bin_volume
        ems_quality_next = self._compute_ems_quality(target_bin)
        potential_next = bin_util_next + 0.3 * ems_quality_next

        # Potential-based shaped reward (theoretically sound - doesn't change optimal policy)
        # F(s,a,s') = r + γ*Φ(s') - Φ(s)
        # Using per-bin utilization gives ~0.01-0.05 per step instead of ~0.001
        reward = (self.gamma * potential_next) - potential_prev

        # Small bonus for balancing bins
        current_bin_items = len(target_bin.placed)
        avg_items_per_used_bin = sum(len(b.placed) for b in self.bins) / max(1, self._get_bins_used())
        if current_bin_items < avg_items_per_used_bin:
            reward += 0.01
        
        # Check if done
        if len(self.items) == 0:
            self.done = True
            bins_used = self._get_bins_used()
            
            bins_efficiency = 1.0 - (bins_used / self.max_bins)
            reward += 0.3 + bins_efficiency * 0.2
            
            # SAFETY NET: Check affinities
            affinity_violations = 0
            for bin in self.bins:
                if len(bin.placed) > 0:
                    if not bin.check_positive_affinity_before_completion():
                        affinity_violations += 1
            
            if affinity_violations > 0:
                reward -= 0.1 * affinity_violations
                info["affinity_violations"] = affinity_violations
                print(f"âš ï¸  WARNING: Affinity violation despite proactive blocking!")
            
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
ACTION_FEAT_DIM = 25  # Same dimension, but different semantics


def build_action_features(env: MultiBinPackingEnv, actions):
    """
    Build action features for multi-bin environment using EMS.

    NEW (EMS-based):
    - Features 6-8: EMS corner position (sx, sy, sz)
    - Features 9-11: EMS dimensions (ew, ed, eh) - BETTER than residual capacity!
    - Features 12-14: Slack space after placement (ew-rw, ed-rd, eh-rh)
    """
    W, D, H = env.bin_size
    binV = float(env.bin_volume)
    rows = []

    for a in actions:
        if a is None:
            rows.append([0.0] * ACTION_FEAT_DIM)
            continue

        bin_idx, item_idx, ems_idx, rot_idx, ems, size, weight, item_id = a
        target_bin = env.bins[bin_idx]

        iw, id_, ih, _, _ = env.items[item_idx]
        rw, rd, rh = size

        # EMS features
        sx, sy, sz = ems.x, ems.y, ems.z  # EMS corner position
        ew, ed, eh = ems.w, ems.d, ems.h  # EMS dimensions

        # Normalized features
        rw_n, rd_n, rh_n = rw/W, rd/D, rh/H
        sx_n, sy_n, sz_n = sx/W, sy/D, sz/H  # EMS corner
        ew_n, ed_n, eh_n = ew/W, ed/D, eh/H  # EMS dimensions

        # Slack space after placing item in EMS
        slack_x, slack_y, slack_z = max(0, ew-rw), max(0, ed-rd), max(0, eh-rh)
        slack_x_n, slack_y_n, slack_z_n = slack_x/W, slack_y/D, slack_z/H

        vol = float(rw * rd * rh)
        delta_u = vol / binV
        tight = float((slack_x==0) + (slack_y==0) + (slack_z==0))  # How many dimensions are tight

        # Heightmap features
        h_at_ep_norm = 0.0
        local_avg = 0.0
        try:
            h_at_ep = target_bin.get_heightmap_at(sx, sy)
            h_at_ep_norm = h_at_ep / H

            resolution = target_bin.resolution
            gx = min(sx // resolution, target_bin.heightmap.shape[0] - 1)
            gy = min(sy // resolution, target_bin.heightmap.shape[1] - 1)

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
            iw/W, id_/D, ih/H,          # 0-2: Original item size
            rw_n, rd_n, rh_n,            # 3-5: Rotated item size
            sx_n, sy_n, sz_n,            # 6-8: EMS corner position
            ew_n, ed_n, eh_n,            # 9-11: EMS dimensions (available space)
            slack_x_n, slack_y_n, slack_z_n,  # 12-14: Slack after placement
            delta_u, tight,              # 15-16: Volume utilization, tightness
            W/D, D/H,                    # 17-18: Container aspect ratios
            h_at_ep_norm,                # 19: Height at placement position
            local_avg,                   # 20: Local average height
            item_weight_ratio,           # 21: Item weight / max weight
            remaining_weight_capacity,   # 22: Available weight capacity
            bin_idx_norm,                # 23: Which bin
            bin_current_util,            # 24: Bin utilization
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


def evaluate_agent_on_problem(agent, env, items, episodes=20, patch_size=7, 
                               train_freq=1, num_train_steps=1, verbose=False):
    """
    Reusable training/evaluation function for GA architecture evolution.
    
    This function trains an agent for a specified number of episodes and returns
    performance metrics. Used by GA to evaluate different architectures.
    
    Args:
        agent: Pre-initialized DQNAgentEnhanced instance
        env: MultiBinPackingEnv instance
        items: List of items to pack (w, d, h, weight, item_id)
        episodes: Number of episodes to train
        patch_size: Heightmap patch size (should match agent's config)
        train_freq: Training frequency
        num_train_steps: Training steps per frequency
        verbose: Print episode-by-episode progress
        
    Returns:
        dict: {
            'avg_utilization': Average utilization across episodes,
            'avg_bins_used': Average bins used,
            'best_utilization': Best utilization achieved,
            'total_items_placed': Total items successfully placed,
            'episode_returns': List of episode returns
        }
    """
    import time
    
    start_time = time.time()
    
    utils = []
    bins_used_list = []
    items_placed_list = []
    returns = []
    
    for ep in range(episodes):
        obs = env.reset(items=items.copy())
        ep_ret = 0.0
        steps = 0
        
        while True:
            actions, mask_short = env.action_space()
            feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            patches = extract_patches_for_actions(env, actions, patch_size=patch_size) if len(actions) > 0 else np.zeros((0, patch_size, patch_size), np.float32)
            
            act_idx = agent.select_action(
                obs,
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                patches if patches.shape[0] > 0 else np.zeros((1, patch_size, patch_size), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
            )
            act = None if (act_idx is None or actions == [] or actions[act_idx] is None) else actions[act_idx]
            
            currF, currM = pad_feats_mask(
                feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            currP = pad_patches(
                patches if patches.shape[0] > 0 else np.zeros((0, patch_size, patch_size), np.float32),
                env.max_actions,
                patch_size
            )
            
            nobs, rew, done, info = env.step(act)
            
            n_actions, n_mask_short = env.action_space()
            n_feats = build_action_features(env, n_actions) if len(n_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            n_patches = extract_patches_for_actions(env, n_actions, patch_size=patch_size) if len(n_actions) > 0 else np.zeros((0, patch_size, patch_size), np.float32)
            
            nextF, nextM = pad_feats_mask(
                n_feats,
                n_mask_short if n_mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            nextP = pad_patches(
                n_patches if n_patches.shape[0] > 0 else np.zeros((0, patch_size, patch_size), np.float32),
                env.max_actions,
                patch_size
            )
            
            agent.store(obs, act_idx, rew, nobs, done,
                       curr_action_feats=currF, curr_mask=currM, curr_patches=currP,
                       next_action_feats=nextF, next_mask=nextM, next_patches=nextP)
            
            if steps % train_freq == 0:
                for _ in range(num_train_steps):
                    agent.train_step()
            
            obs = nobs
            ep_ret += rew
            steps += 1
            
            if done:
                util = info.get('utilization', 0.0)
                bins = info.get('bins_used', 0)
                items_placed = info.get('items_placed', 0)

                utils.append(util)
                bins_used_list.append(bins)
                items_placed_list.append(items_placed)
                returns.append(ep_ret)

                # Update episode count for episode-based epsilon decay
                agent.on_episode_end()

                if verbose:
                    print(f"  Episode {ep+1}/{episodes}: Util={util:.3f}, Bins={bins}, Items={items_placed}/{len(items)}, eps={agent.epsilon():.3f}")

                break
    
    training_time = time.time() - start_time
    
    return {
        'avg_utilization': np.mean(utils),
        'avg_bins_used': np.mean(bins_used_list),
        'best_utilization': np.max(utils),
        'total_items_placed': np.mean(items_placed_list),
        'episode_returns': returns,
        'training_time': training_time
    }


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
    Train ENHANCED DQN for multi-bin packing with:
    - Heightmap CNN: Processes 7x7 patches around placement positions
    - Transformer Attention: Actions reason about each other
    """
    import os
    
    print(f"\n{'='*80}")
    print(f"MULTI-BIN PACKING DQN TRAINING (ENHANCED VERSION)")
    print(f" Heightmap CNN for spatial reasoning")
    print(f" Transformer attention for action relationships")
    print(f"{'='*80}")
    print(f"Loading: {problem_path}")
    
    problem = load_problem(problem_path)
    items = load_problem_as_items(problem)
    
    W, D, H = problem.bin_dimensions
    
    print(f"\n“ PROBLEM SPECIFICATION:")
    print(f"   Container: {W}—{D}—{H} (volume: {W*D*H:,})")
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

    # DYNAMIC EPSILON DECAY: Episode-based (robust to variable episode lengths!)
    from dqn_core.dqn_enhanced import calculate_dynamic_epsilon_decay

    eps_decay_episodes, total_episodes = calculate_dynamic_epsilon_decay(
        total_episodes=episodes,
        plateau_at_ratio=0.8
    )

    # DQN config - ENHANCED with heightmap CNN + Transformer attention
    cfg = DQNConfigEnhanced(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        gamma=0.992,
        lr=1e-4,
        batch_size=128,
        buffer_size=400_000,
        eps_start=1.0,
        eps_end=0.01,  # Reduced from 0.05 to 0.01 (1% exploration)
        eps_decay_episodes=eps_decay_episodes,  # EPISODE-BASED: robust to variable lengths
        total_episodes=total_episodes,
        target_update_interval=200,  # Reduced from 500
        n_step=3,  # REDUCED from 15: Better for learning during training
        double_dqn=True,
        warmup_steps=1500,  # Reduced from 1500
        heightmap_patch_size=7,  # NEW: Size of heightmap patches
        use_attention=True,      # NEW: Enable Transformer attention
    )
    agent = DQNAgentEnhanced(cfg)  # Use enhanced agent
    
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
    
    print(f"Enhanced agent initialized on {cfg.device}")
    print(f"   Heightmap patches: {cfg.heightmap_patch_size}x{cfg.heightmap_patch_size}")
    print(f"   Transformer attention: {'Enabled' if cfg.use_attention else 'Disabled'}")
    print(f"Starting training...\n")
    
    for ep in range(episodes):
        obs = env.reset(items=items.copy())
        ep_ret = 0.0
        steps = 0
        losses = []
        
        while True:
            actions, mask_short = env.action_space()
            feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            
            # NEW: Extract heightmap patches for each action
            patches = extract_patches_for_actions(env, actions, patch_size=7) if len(actions) > 0 else np.zeros((0, 7, 7), np.float32)
            
            act_idx = agent.select_action(
                obs,
                feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                patches if patches.shape[0] > 0 else np.zeros((1, 7, 7), np.float32),  # NEW: Pass patches
                mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
            )
            act = None if (act_idx is None or actions == [] or actions[act_idx] is None) else actions[act_idx]
            
            currF, currM = pad_feats_mask(
                feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            # NEW: Pad patches
            currP = pad_patches(
                patches if patches.shape[0] > 0 else np.zeros((0, 7, 7), np.float32),
                env.max_actions,
                7
            )
            
            nobs, rew, done, info = env.step(act)
            
            n_actions, n_mask_short = env.action_space()
            n_feats = build_action_features(env, n_actions) if len(n_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            # NEW: Extract next patches
            n_patches = extract_patches_for_actions(env, n_actions, patch_size=7) if len(n_actions) > 0 else np.zeros((0, 7, 7), np.float32)
            
            nextF, nextM = pad_feats_mask(
                n_feats,
                n_mask_short if n_mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                env.max_actions
            )
            # NEW: Pad next patches
            nextP = pad_patches(
                n_patches if n_patches.shape[0] > 0 else np.zeros((0, 7, 7), np.float32),
                env.max_actions,
                7
            )
            
            agent.store(obs, act_idx, rew, nobs, done,
                       curr_action_feats=currF, curr_mask=currM, curr_patches=currP,  # NEW: Add curr_patches
                       next_action_feats=nextF, next_mask=nextM, next_patches=nextP)  # NEW: Add next_patches
            
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
                          f"mu:{agent.epsilon():.3f} | L:{avg_loss:.4f}")
                    
                    # Per-bin breakdown every 50 episodes
                    if (ep + 1) % 50 == 0 and "per_bin_utils" in info:
                        print(f" Bins breakdown:")
                        for i, (u, w, n) in enumerate(zip(info["per_bin_utils"],
                                                          info["per_bin_weights"],
                                                          info["per_bin_items"])):
                            print(f"     Bin {i+1}: {n:2d}items | Vol:{u:.3f} | Wgt:{w:4d}/{env.max_weight}")

                # Update episode count for episode-based epsilon decay
                agent.on_episode_end()
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

        # Save with enhanced metadata
        problem_info = {
            'bin_dimensions': (W, D, H),
            'num_items': len(items),
            'max_bins': problem.max_bins,
            'max_weight': problem.max_weight
        }
        training_stats = {
            'best_bins': best_bins,
            'best_items': best_items,
            'best_util': best_util,
            'final_ma50_bins': np.mean(bins_hist),
            'final_ma50_items': np.mean(items_hist),
            'final_ma50_util': np.mean(util_hist)
        }
        save_model(agent, save_path, problem_info, training_stats)
        print()
    
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


def save_model(agent, save_path: str, problem_info: dict = None, training_stats: dict = None):
    """
    Save a trained DQN agent to disk.

    Args:
        agent: DQNAgentEnhanced instance to save
        save_path: Path where to save the model
        problem_info: Optional dict with problem metadata (bin dimensions, num items, etc.)
        training_stats: Optional dict with training statistics (best_bins, best_util, etc.)
    """
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': agent.q.state_dict(),
        'target_state_dict': agent.q_target.state_dict(),
        'optimizer_state_dict': agent.opt.state_dict(),
        'config': agent.cfg.__dict__,
        'env_steps': agent.env_steps,
        'training_steps': agent.training_steps,
        'epsilon': agent.epsilon,
    }

    if problem_info:
        checkpoint['problem_info'] = problem_info

    if training_stats:
        checkpoint['training_stats'] = training_stats

    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def load_model(load_path: str, problem_path: str = None, device: str = None):
    """
    Load a previously trained DQN agent from disk.

    Args:
        load_path: Path to saved model checkpoint
        problem_path: Optional path to problem file (needed if continuing training)
        device: Device to load on ('cuda' or 'cpu', auto-detect if None)

    Returns:
        agent: Loaded DQNAgentEnhanced
        checkpoint: Full checkpoint dict with metadata
        env: Environment (if problem_path provided, else None)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)

    # Reconstruct config
    from dqn_core.dqn_enhanced import DQNConfigEnhanced
    cfg_dict = checkpoint['config']
    cfg = DQNConfigEnhanced(**cfg_dict)
    cfg.device = device

    # Create agent
    from dqn_core.dqn_enhanced import DQNAgentEnhanced
    agent = DQNAgentEnhanced(cfg)

    # Load weights
    agent.q.load_state_dict(checkpoint['model_state_dict'])
    agent.q_target.load_state_dict(checkpoint['target_state_dict'])
    agent.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.env_steps = checkpoint['env_steps']
    agent.training_steps = checkpoint['training_steps']
    agent._eps = checkpoint['epsilon']

    print(f"Model loaded on {device}")
    print(f"  Training steps: {agent.training_steps}")
    print(f"  Epsilon: {agent.epsilon:.4f}")

    # Optionally reconstruct environment
    env = None
    if problem_path:
        from nesting.dataset_loader import load_problem
        problem = load_problem(problem_path)
        items = load_problem_as_items(problem)
        W, D, H = problem.bin_dimensions

        env = MultiBinPackingEnv(
            W, D, H,
            items=items,
            max_actions=cfg.max_actions,
            topk_eps=1000,
            seed=42,
            gamma=cfg.gamma,
            problem=problem
        )
        print(f"  Environment created: {W}×{D}×{H}")

    if 'training_stats' in checkpoint:
        stats = checkpoint['training_stats']
        print(f"\nBest Training Performance:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    return agent, checkpoint, env


def full_datapath(filename: str) -> str:
    """Helper for dataset path."""
    return "nesting\\inputData\\Benchmark dataset and instance generator for Real-World 3dBPP\\Input\\" + filename


if __name__ == "__main__":
    print("MULTI-BIN PACKING DQN")
    
    problem = "3dBPP_4.txt"

    agent, env, best_solution = train_multibin_pack_dqn(
        problem_path=full_datapath(problem),
        episodes=400,
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