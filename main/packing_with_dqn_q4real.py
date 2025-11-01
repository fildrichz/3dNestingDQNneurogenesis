# packing_with_dqn_q4real.py
# Sequential-bins DQN trainer aligned with your core semantics.
# - Packs bin-by-bin: for each episode, iterate over bins; items packed in earlier bins are removed before next bin.
# - Same action model: (item × EP × 6 rots) enumerated, sampled/padded to max_actions, mask marks feasible.
# - Same reward shaping: potential-based (gamma*U_next - U_prev) with terminal bonus = final utilization.
# - Same logging: MA50, Best, ε, Buffer, Return, Steps.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import collections

from dqn_core.dqn import DQNAgent, DQNConfig
from nesting.packing_core_q4real import Q4RealBPPContainer
from nesting.q4realbpp_loader import Item, Q4RealBPPInstance, Q4RealBPPLoader


# ---------------------
# Utilities
# ---------------------

def item_rotations(i: Item) -> Tuple[Tuple[int, int, int], ...]:
    """Six axis-aligned rotations for an item."""
    w, d, h = int(i.width), int(i.depth), int(i.height)
    return (
        (w, d, h), (w, h, d),
        (d, w, h), (d, h, w),
        (h, w, d), (h, d, w),
    )


# ---------------------
# Q4Real Packing Environment (single-bin env)
# ---------------------

class Q4RealPackingEnv:
    def __init__(
        self,
        instance: Q4RealBPPInstance,
        bin_id: int = 0,
        max_actions: int = 128,
        topk_eps: int = 48,
        gamma: float = 0.992,
        seed: Optional[int] = 0,
    ):
        self.instance = instance
        self.bin_spec = instance.bins[bin_id]
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.topk_eps = int(topk_eps)
        self.rng = np.random.default_rng(seed)

        self.W = int(self.bin_spec.width)
        self.D = int(self.bin_spec.depth)
        self.H = int(self.bin_spec.height)
        self.bin_volume = float(self.W * self.D * self.H)

        self.reset()

    def _new_container(self) -> Q4RealBPPContainer:
        C = Q4RealBPPContainer(
            self.W, self.D, self.H,
            self.bin_spec.max_weight, self.bin_spec.id
        )
        # apply dataset constraints
        C.set_constraints(
            self.instance.affinities,
            self.instance.load_bearing_ratio,
            self.instance.load_balancing
        )
        if not getattr(C, "eps", None) or len(C.eps) == 0:
            C.eps = [(0, 0, 0)]
        return C

    def reset(self, items: Optional[List[Item]] = None):
        """Start a new episode for THIS bin. If items is None, the env shuffles a fresh copy."""
        self.C = self._new_container()
        self.items: List[Item] = list(items) if items is not None else list(self.instance.items)
        self.n_items = len(self.items)
        self.rng.shuffle(self.items)
        self.done = False
        return self._obs()

    # ------- feasibility and action enumeration -------

    def _feasible(self, ep: Tuple[int, int, int], rot: Tuple[int, int, int], it: Item) -> bool:
        """Feasibility via Q4Real constraints (geometry + weight + affinities/load)."""
        return self.C.can_place_at(ep, rot, it.weight, it.category)

    def enumerate_actions(self):
        """Return list of feasible actions and the EP subset. Each action is (item_idx, ep_idx, rot_idx, ep, rot)."""
        actions = []
        eps_sorted = sorted(set(self.C.eps), key=lambda p: (p[2], p[1], p[0]))[: self.topk_eps]

        for ep_idx, ep in enumerate(eps_sorted):
            ex, ey, ez = ep
            if ex >= self.W or ey >= self.D or ez >= self.H:
                continue
            for item_idx, it in enumerate(self.items):
                rots = item_rotations(it)
                for rot_idx, rot in enumerate(rots):
                    if self._feasible(ep, rot, it):
                        actions.append((item_idx, ep_idx, rot_idx, ep, rot))
        return actions, eps_sorted

    def action_space(self):
        """Sample/pad actions to max_actions and return (actions, mask, eps_subset)."""
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
            actions.extend([None] * pad)
            mask = np.zeros((self.max_actions,), dtype=np.float32)
            mask[:A] = 1.0
        return actions, mask, eps_subset

    # ------- observation / features -------

    def _utilization(self) -> float:
        """Volume utilization of this bin."""
        if not getattr(self.C, "placed", None):
            return 0.0
        used = sum(b.w * b.d * b.h for b in self.C.placed)
        return float(used) / self.bin_volume if self.bin_volume > 0 else 0.0

    def _obs(self) -> np.ndarray:
        """Compact state vector."""
        packed = self._utilization()
        items_left = len(self.items) / max(1, self.n_items)

        if getattr(self.C, "eps", None):
            caps = [(self.W - x, self.D - y, self.H - z) for (x, y, z) in self.C.eps]
            cx = max(c[0] for c in caps) / self.W
            cy = max(c[1] for c in caps) / self.D
            cz = max(c[2] for c in caps) / self.H
        else:
            cx = cy = cz = 0.0
        return np.array([packed, items_left, cx, cy, cz], dtype=np.float32)

    # ------- stepping -------

    def step(self, action):
        """
        action is either:
          - None (agent decides to stop / no feasible actions), or
          - (item_idx, ep_idx, rot_idx, ep, rot)
        """
        if self.done:
            raise RuntimeError("Episode done, call reset().")

        info: Dict = {}
        util_prev = self._utilization()

        # No-op / stop action
        if action is None:
            self.done = True
            reward = (self.gamma * util_prev) - util_prev  # potential-only step
            reward += util_prev                            # terminal bonus
            info["utilization"] = util_prev
            return self._obs(), reward, self.done, info

        # Try placement
        item_idx, ep_idx, rot_idx, ep, rot = action
        it = self.items[item_idx]
        ok = self.C.place_at(ep, rot, it.weight, it.category, it.id)
        if not ok:
            # Should be rare (we enumerate only feasible)
            self.done = True
            return self._obs(), -1.0, True, {"invalid": True, "utilization": util_prev}

        # Remove the placed item and compute new potential
        del self.items[item_idx]
        util_next = self._utilization()

        # Potential-based shaping
        reward = (self.gamma * util_next) - util_prev

        # Termination: no actions or no items left
        actions, _ = self.enumerate_actions()
        if (len(actions) == 0) or (len(self.items) == 0):
            self.done = True
            reward += util_next  # terminal bonus
            info["utilization"] = util_next

        return self._obs(), reward, self.done, info


# ---------------------
# Action features & padding
# ---------------------

ACTION_FEAT_DIM = 19

def _safe_caps(W: int, D: int, H: int, ep: Tuple[int, int, int]) -> Tuple[int, int, int]:
    x, y, z = ep
    return (W - x, D - y, H - z)

def build_action_features(env: Q4RealPackingEnv, actions):
    """19-D action features per action, padded later."""
    W, D, H = env.W, env.D, env.H
    binV = float(env.bin_volume)

    rows: List[List[float]] = []
    for a in actions:
        if a is None:
            rows.append([0.0] * ACTION_FEAT_DIM)
            continue

        item_idx, ep_idx, rot_idx, ep, rot = a
        it = env.items[item_idx]
        iw, id_, ih = int(it.width), int(it.depth), int(it.height)
        rw, rd, rh = rot
        ex, ey, ez = ep
        cx, cy, cz = _safe_caps(W, D, H, ep)

        # normalized dims/pos/caps
        rw_n, rd_n, rh_n = rw / W, rd / D, rh / H
        ex_n, ey_n, ez_n = ex / W, ey / D, ez / H
        cx_n, cy_n, cz_n = cx / W, cy / D, cz / H

        # slack (cap - rot) normalized
        sx, sy, sz = max(0, cx - rw), max(0, cy - rd), max(0, cz - rh)
        sx_n, sy_n, sz_n = sx / W, sy / D, sz / H

        vol = float(rw * rd * rh)
        delta_u = vol / binV if binV > 0 else 0.0
        tight = float((sx == 0) + (sy == 0) + (sz == 0))

        rows.append([
            iw / W, id_ / D, ih / H,  # original item dims
            rw_n, rd_n, rh_n,         # chosen rotation
            ex_n, ey_n, ez_n,         # EP position
            cx_n, cy_n, cz_n,         # residual caps
            sx_n, sy_n, sz_n,         # slack
            delta_u, tight,           # hints
            W / D, D / H,             # bin aspect
        ])

    return np.asarray(rows, dtype=np.float32)

def pad_feats_mask(feats: np.ndarray, mask_short: np.ndarray, maxA: int):
    """Pad features/mask to fixed maxA length."""
    A = feats.shape[0]
    F = np.zeros((maxA, ACTION_FEAT_DIM), np.float32)
    M = np.zeros((maxA,), np.float32)
    F[:A] = feats
    if mask_short.shape[0] >= A:
        M[:A] = mask_short[:A]
    else:
        M[:A] = 1.0 if A > 0 else 0.0
    return F, M


# ---------------------
# Training (sequential bins)
# ---------------------

def train_q4real_dqn_sequential(
    instance: Q4RealBPPInstance,
    episodes: int = 300,
    max_actions: int = 128,
    topk_eps: int = 48,
    device: str = "cuda",
    eps_decay_factor: int = 20,   # episodes * 20 (aligned decay)
    log_interval: int = 10,
    save_path: Optional[str] = None,
):
    """
    Train DQN on Q4RealBPP by packing bins sequentially in each episode.
    For each episode:
      - Shuffle all items.
      - For bin in instance.bins:
           create env for that bin, pass remaining items, run till done.
           remove packed items (env already removed them).
      - Stats/logs aggregate across bins for the episode.
    """
    # Create a "template" env to get shapes/config
    tmpl_env = Q4RealPackingEnv(instance, bin_id=0, max_actions=max_actions, topk_eps=topk_eps, gamma=0.992, seed=42)
    obs = tmpl_env.reset()
    OBS_DIM = obs.shape[0]

    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device=device,
        gamma=0.992,
        lr=2e-4,
        batch_size=64,
        buffer_size=200_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=episodes * eps_decay_factor,
        target_update_interval=500,
        n_step=3,
        double_dqn=True,
        warmup_steps=500,
    )
    agent = DQNAgent(cfg)

    util_hist = collections.deque(maxlen=50)
    returns_hist = collections.deque(maxlen=50)
    steps_hist = collections.deque(maxlen=50)
    best_overall_util = 0.0

    print("\n" + "=" * 60)
    print("Starting DQN Training for Q4Real 3D Bin Packing (Sequential Bins)")
    print("=" * 60)
    print(f"#Bins: {len(instance.bins)} | Episodes: {episodes}")
    print(f"Max actions: {max_actions}, Top EPs: {topk_eps}\n")

    for ep in range(episodes):
        # Episode item pool (shuffled once)
        all_items = list(instance.items)
        np.random.default_rng(1000 + ep).shuffle(all_items)

        ep_ret = 0.0
        steps = 0
        losses = []

        # Aggregate utilization across bins (volume-weighted)
        used_vol_sum = 0.0
        total_vol_sum = 0.0

        # Iterate bins sequentially
        for b_id, b in enumerate(instance.bins):
            # Build env for this bin; feed remaining items
            env = Q4RealPackingEnv(
                instance, bin_id=b_id,
                max_actions=max_actions, topk_eps=topk_eps,
                gamma=0.992, seed=43 + ep + b_id
            )
            obs = env.reset(items=all_items)

            # Bin loop
            while True:
                actions, mask_short, _ = env.action_space()
                feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)

                act_idx = agent.select_action(
                    obs,
                    feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                    mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32),
                )
                act = None if (act_idx is None or actions == [] or actions[act_idx] is None) else actions[act_idx]

                currF, currM = pad_feats_mask(
                    feats if feats.shape[0] > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                    mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                    env.max_actions,
                )

                nobs, rew, done, info = env.step(act)
                ep_ret += rew

                n_actions, n_mask_short, _ = env.action_space()
                n_feats = build_action_features(env, n_actions) if len(n_actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
                nextF, nextM = pad_feats_mask(
                    n_feats,
                    n_mask_short if n_mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                    env.max_actions,
                )

                agent.store(
                    obs, act_idx, rew, nobs, done,
                    curr_action_feats=currF, curr_mask=currM,
                    next_action_feats=nextF, next_mask=nextM,
                )
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                obs = nobs
                steps += 1

                if done:
                    # Bin finished – accumulate volume-weighted utilization
                    util_bin = info.get("utilization", env._utilization())
                    used_vol_sum += util_bin * env.bin_volume
                    total_vol_sum += env.bin_volume
                    # Remaining items go to next bin
                    all_items = env.items
                    break

            # Early exit if no items remain
            if len(all_items) == 0:
                break

        # Episode overall utilization across bins
        overall_util = (used_vol_sum / total_vol_sum) if total_vol_sum > 0 else 0.0
        best_overall_util = max(best_overall_util, overall_util)
        util_hist.append(overall_util)
        returns_hist.append(ep_ret)
        steps_hist.append(steps)

        if (ep + 1) % log_interval == 0 or ep == 0:
            ma_util = float(np.mean(util_hist)) if len(util_hist) else overall_util
            ma_ret = float(np.mean(returns_hist)) if len(returns_hist) else ep_ret
            ma_steps = float(np.mean(steps_hist)) if len(steps_hist) else steps
            avg_loss = float(np.mean(losses)) if losses else 0.0
            print(
                f"Episode {ep+1:4d}/{episodes} | "
                f"Steps: {steps:3d} | "
                f"Return: {ep_ret:7.3f} | "
                f"Util: {overall_util:.3f} | "
                f"MA50: {ma_util:.3f} | "
                f"Best: {best_overall_util:.3f} | "
                f"ε: {agent.epsilon():.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Buffer: {agent.buffer.size}/{agent.buffer.capacity}"
            )

    print("\n" + "=" * 60)
    print("Training Complete (Sequential Bins)!")
    print(f"Best overall utilization: {best_overall_util:.3f}")
    print(f"Final MA50 utilization: {np.mean(util_hist) if len(util_hist) else 0.0:.3f}")
    print("=" * 60 + "\n")

    if save_path:
        agent.save(save_path)
        print(f"Model saved to: {save_path}\n")

    return agent


# ---------------------
# Evaluation (greedy, sequential bins)
# ---------------------

def evaluate_q4real_dqn_sequential(
    agent: DQNAgent,
    instance: Q4RealBPPInstance,
    episodes: int = 10,
    max_actions: int = 128,
    topk_eps: int = 48,
):
    agent._eps = 0.0  # greedy
    utils = []

    for ep in range(episodes):
        all_items = list(instance.items)
        np.random.default_rng(9000 + ep).shuffle(all_items)

        used_vol_sum = 0.0
        total_vol_sum = 0.0

        for b_id, b in enumerate(instance.bins):
            env = Q4RealPackingEnv(instance, bin_id=b_id, max_actions=max_actions, topk_eps=topk_eps, gamma=0.992, seed=777 + ep + b_id)
            obs = env.reset(items=all_items)

            while True:
                actions, mask_short, _ = env.action_space()
                feats = build_action_features(env, actions) if len(actions) > 0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
                act_idx = agent.select_action(
                    obs,
                    feats if feats.shape[0] > 0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                    mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32),
                )
                act = None if (act_idx is None or actions == [] or actions[act_idx] is None) else actions[act_idx]
                obs, rew, done, info = env.step(act)
                if done:
                    util_bin = info.get("utilization", env._utilization())
                    used_vol_sum += util_bin * env.bin_volume
                    total_vol_sum += env.bin_volume
                    all_items = env.items
                    break

            if len(all_items) == 0:
                break

        overall_util = (used_vol_sum / total_vol_sum) if total_vol_sum > 0 else 0.0
        utils.append(overall_util)

    avg_util = float(np.mean(utils)) if utils else 0.0
    print(f"Evaluation (greedy, sequential bins) over {episodes} eps: avg utilization = {avg_util:.3f}")
    return avg_util


# ---------------------
# Example main
# ---------------------

def main():
    print("=" * 60)
    print("Q4Real DQN Packing — Sequential Bins")
    print("=" * 60)

    # Replace with your real loader if needed:
    # instance = Q4RealBPPLoader.load_from_txt("path/to/q4real_instance.txt")
    instance = Q4RealBPPLoader.generate_sample_instance(num_items=40, num_bins=2)

    # Train (sequential bins)
    agent = train_q4real_dqn_sequential(
        instance=instance,
        episodes=300,
        max_actions=128,
        topk_eps=64,
        device="cuda",
        log_interval=10,
        save_path=None,
    )

    # Evaluate (sequential bins)
    _ = evaluate_q4real_dqn_sequential(agent, instance, episodes=10, max_actions=128, topk_eps=64)


if __name__ == "__main__":
    main()
