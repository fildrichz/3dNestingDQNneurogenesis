"""
Relational Multi-Bin Packing - environment + training loop for the
relational (constraint-learning) DQN in dqn_core/dqn_relational.py.

Differences from packing_with_dqncore2_enhanced.py:

- NO hand-crafted per-action features and NO candidate-action subsampling.
  The state is the raw problem structure, tokenised:
      * one token per remaining item TYPE (dims, weight, remaining qty)
      * one token per already-placed box (dims, weight, position, bin)
      * one token per EMS (raw geometry) across all bins
      * one token per bin (full heightmap + raw scalars incl. CoM target)
  Problem-file constraints are passed as TYPED EDGES between item tokens
  (incompatibility / affinity / must-not-be-above). Item IDs never enter
  the network, so what it learns about an edge type transfers to any
  problem that uses the same constraint vocabulary.

- The action space is the full product (item type) x (EMS) x (rotation).

- Only PHYSICS is masked (item does not fit the EMS / container, or would
  exceed the container height after gravity). Problem constraints
  (weight cap, incompatibility, relative positioning, affinity) are NOT
  masked: choosing a violating placement is refused by the environment
  with a negative reward, and the agent has to learn the constraint
  semantics itself. A refused combination is masked for the remainder of
  the current state (until the next successful placement) so the agent
  cannot loop on the same refusal.
"""
from __future__ import annotations
import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from nesting.packing_core_enhanced import Container
from nesting.dataset_loader import load_problem, BinPackingProblem
from dqn_core.dqn_relational import (
    RelationalDQNAgent, RelationalDQNConfig,
    EDGE_NONE, EDGE_INCOMPATIBLE, EDGE_AFFINITY,
    EDGE_NOT_ABOVE, EDGE_NOT_BELOW, EDGE_SAME_TYPE,
)

# rotation r permutes (l, w, h) by ROTATIONS[r]
ROTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def resample_heightmap(hm: np.ndarray, grid: int) -> np.ndarray:
    """Nearest-neighbour resample of a heightmap to (grid, grid)."""
    ix = np.round(np.linspace(0, hm.shape[0] - 1, grid)).astype(int)
    iy = np.round(np.linspace(0, hm.shape[1] - 1, grid)).astype(int)
    return hm[np.ix_(ix, iy)]


class RelationalPackingEnv:
    """Multi-bin packing environment exposing a tokenised relational state."""

    def __init__(self, problem: BinPackingProblem, cfg: RelationalDQNConfig,
                 seed: int = 0, gamma: float = 0.992,
                 violation_penalty: float = 0.25, max_env_steps: int = 2000):
        self.problem = problem
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.gamma = float(gamma)
        self.violation_penalty = float(violation_penalty)
        self.max_env_steps = int(max_env_steps)

        W, D, H = problem.bin_dimensions
        self.bin_size = (int(W), int(D), int(H))
        self.bin_volume = int(W) * int(D) * int(H)
        self.n_bins = min(problem.max_bins, cfg.max_bins)
        if problem.max_bins > cfg.max_bins:
            print(f"WARNING: problem.max_bins={problem.max_bins} capped to {cfg.max_bins}")
        self.max_weight = problem.max_weight

        if len(problem.items) > cfg.max_types:
            raise ValueError(f"problem has {len(problem.items)} item types, "
                             f"cfg.max_types={cfg.max_types} is too small")

        # weight normaliser: max bin weight if present else heaviest item
        self.wnorm = float(self.max_weight if self.max_weight
                           else max((it.weight for it in problem.items), default=1) or 1)

        # constraint lookups by item id (only used inside the env referee /
        # edge builder - never exposed to the network as IDs)
        self.incompat = {frozenset(p) for p in problem.incompatibilities}
        self.affinity = {frozenset(p) for p in problem.positive_affinities}
        self.relpos_pairs = set()   # (light_id, heavy_id)
        for _, tuples in (problem.relative_pos or {}).items():
            for light_id, heavy_id in tuples:
                self.relpos_pairs.add((int(light_id), int(heavy_id)))

        self.reset()

    # ------------------------------------------------------------------ setup

    def _create_bin(self) -> Container:
        W, D, H = self.bin_size
        adaptive_resolution = max(1, min(W, D) // 20)
        C = Container(W, D, H, max_weight=self.max_weight,
                      resolution=adaptive_resolution, max_ems=50)
        C.set_constraints(
            incompatibilities=self.problem.incompatibilities,
            positive_affinities=self.problem.positive_affinities,
            center_of_mass=self.problem.center_of_mass,
            relative_pos=self.problem.relative_pos,
        )
        return C

    def reset(self) -> Dict[str, np.ndarray]:
        # fixed type slots for the whole episode (stable action indices)
        self.types = []
        for it in self.problem.items:
            self.types.append({
                "id": int(it.id),
                "dims": (int(it.length), int(it.width), int(it.height)),
                "weight": int(it.weight),
                "qty": int(it.quantity),
                "qty0": int(it.quantity),
            })
        self.total_items0 = sum(t["qty0"] for t in self.types)
        self.bins: List[Container] = [self._create_bin() for _ in range(self.n_bins)]
        self.placed_tokens: List[dict] = []   # {'id','dims','weight','pos','bin'}
        self.total_placed_volume = 0
        self.done = False
        self.env_steps = 0
        self.refused: set = set()
        self._edge_type_matrix = self._build_type_edge_matrix()
        self._rebuild_ems_refs()
        return self.state()

    def _build_type_edge_matrix(self) -> np.ndarray:
        """Directed edge type between item TYPE ids a->b (n_types x n_types)."""
        n = len(self.types)
        M = np.zeros((n, n), np.int8)
        for a in range(n):
            for b in range(n):
                ida, idb = self.types[a]["id"], self.types[b]["id"]
                pair = frozenset((ida, idb))
                if a != b and pair in self.incompat:
                    M[a, b] = EDGE_INCOMPATIBLE
                elif a != b and pair in self.affinity:
                    M[a, b] = EDGE_AFFINITY
                elif (idb, ida) in self.relpos_pairs:   # a heavy, b light
                    M[a, b] = EDGE_NOT_ABOVE
                elif (ida, idb) in self.relpos_pairs:   # a light, b heavy
                    M[a, b] = EDGE_NOT_BELOW
                elif ida == idb:
                    # diagonal: two distinct TOKENS of the same type (a type
                    # slot and its placed instances) look this entry up;
                    # true self-edges are zeroed in state()
                    M[a, b] = EDGE_SAME_TYPE
        return M

    def _rebuild_ems_refs(self):
        """Flat, volume-sorted list of (bin_idx, EMS) capped at cfg.max_ems."""
        refs = []
        for b_idx, b in enumerate(self.bins):
            for ems in b.ems_list:
                refs.append((b_idx, ems))
        refs.sort(key=lambda r: -r[1].volume())
        self.ems_refs = refs[:self.cfg.max_ems]

    # ------------------------------------------------------------------ state

    def _physics_valid(self) -> np.ndarray:
        """(max_types, max_ems, R) bool - geometric feasibility only."""
        cfg = self.cfg
        W, D, H = self.bin_size
        valid = np.zeros((cfg.max_types, cfg.max_ems, cfg.num_rotations), np.bool_)
        nT, nE = len(self.types), len(self.ems_refs)
        if nE == 0:
            return valid

        dims = np.array([t["dims"] for t in self.types], np.int64)      # (nT,3)
        qty = np.array([t["qty"] for t in self.types], np.int64)
        sizes = np.stack([dims[:, list(p)] for p in ROTATIONS], axis=1)  # (nT,6,3)
        rw, rd, rh = sizes[..., 0], sizes[..., 1], sizes[..., 2]         # (nT,6)

        ex = np.array([r[1].x for r in self.ems_refs])
        ey = np.array([r[1].y for r in self.ems_refs])
        ew = np.array([r[1].w for r in self.ems_refs])
        ed = np.array([r[1].d for r in self.ems_refs])
        eh = np.array([r[1].h for r in self.ems_refs])
        ebin = np.array([r[0] for r in self.ems_refs])

        # fits EMS + container XY  -> (nT, nE, 6)
        ok = ((rw[:, None, :] <= ew[None, :, None]) &
              (rd[:, None, :] <= ed[None, :, None]) &
              (rh[:, None, :] <= eh[None, :, None]) &
              (ex[None, :, None] + rw[:, None, :] <= W) &
              (ey[None, :, None] + rd[:, None, :] <= D))

        # gravity: final z from the placed boxes of the EMS's bin
        for b_idx, b in enumerate(self.bins):
            e_sel = np.flatnonzero(ebin == b_idx)
            if e_sel.size == 0:
                continue
            if b.placed:
                bx = np.array([k.x for k in b.placed])
                by = np.array([k.y for k in b.placed])
                bw = np.array([k.w for k in b.placed])
                bd = np.array([k.d for k in b.placed])
                tops = np.array([k.z + k.h for k in b.placed])
                # (nEb, nT, 6, P)
                ox = ((ex[e_sel][:, None, None, None] < (bx + bw)[None, None, None, :]) &
                      (bx[None, None, None, :] < ex[e_sel][:, None, None, None] + rw[None, :, :, None]))
                oy = ((ey[e_sel][:, None, None, None] < (by + bd)[None, None, None, :]) &
                      (by[None, None, None, :] < ey[e_sel][:, None, None, None] + rd[None, :, :, None]))
                fz = np.max(np.where(ox & oy, tops[None, None, None, :], 0),
                            axis=-1, initial=0)                       # (nEb,nT,6)
            else:
                fz = np.zeros((e_sel.size, nT, 6), np.int64)
            fits_h = fz + rh[None, :, :] <= H                          # (nEb,nT,6)
            ok[:, e_sel, :] &= np.transpose(fits_h, (1, 0, 2))

        ok &= (qty > 0)[:, None, None]
        valid[:nT, :nE, :] = ok
        return valid

    def state(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        W, D, H = self.bin_size

        # ---- item tokens: [type slots | placed boxes] --------------------
        TI = cfg.num_item_tokens
        item_feats = np.zeros((TI, cfg.item_feat_dim), np.float32)
        item_mask = np.zeros((TI,), np.float32)
        token_type_idx = np.full((TI,), -1, np.int64)  # type-slot index per token

        for t_idx, t in enumerate(self.types):
            l, w, h = t["dims"]
            item_feats[t_idx] = [l / W, w / D, h / H, t["weight"] / self.wnorm,
                                 t["qty"] / max(1, self.total_items0),
                                 0.0, 0.0, 0.0, 0.0, 0.0]
            item_mask[t_idx] = 1.0 if t["qty"] > 0 else 0.0
            token_type_idx[t_idx] = t_idx

        placed = self.placed_tokens[-cfg.max_placed:]   # cap: keep most recent
        for j, p in enumerate(placed):
            row = cfg.max_types + j
            pw, pd, ph = p["dims"]
            px, py, pz = p["pos"]
            item_feats[row] = [pw / W, pd / D, ph / H, p["weight"] / self.wnorm,
                               0.0, 1.0, px / W, py / D, pz / H,
                               (p["bin"] + 1) / cfg.max_bins]
            item_mask[row] = 1.0
            token_type_idx[row] = p["type_idx"]

        # ---- constraint edges between item tokens ------------------------
        edge_types = np.zeros((TI, TI), np.int8)
        act = np.flatnonzero(item_mask > 0.5)
        if act.size:
            tt = token_type_idx[act]
            edge_types[np.ix_(act, act)] = self._edge_type_matrix[np.ix_(tt, tt)]
            np.fill_diagonal(edge_types, EDGE_NONE)

        # ---- EMS tokens ---------------------------------------------------
        ems_feats = np.zeros((cfg.max_ems, cfg.ems_feat_dim), np.float32)
        ems_mask = np.zeros((cfg.max_ems,), np.float32)
        for e_idx, (b_idx, ems) in enumerate(self.ems_refs):
            ems_feats[e_idx] = [ems.x / W, ems.y / D, ems.z / H,
                                ems.w / W, ems.d / D, ems.h / H,
                                (b_idx + 1) / cfg.max_bins]
            ems_mask[e_idx] = 1.0

        # ---- bin tokens -----------------------------------------------------
        bin_feats = np.zeros((cfg.max_bins, cfg.bin_feat_dim), np.float32)
        bin_hm = np.zeros((cfg.max_bins, cfg.grid, cfg.grid), np.float16)
        bin_mask = np.zeros((cfg.max_bins,), np.float32)
        com_t = self.problem.center_of_mass
        for b_idx, b in enumerate(self.bins):
            vol_util = sum(k.w * k.d * k.h for k in b.placed) / self.bin_volume
            cx, cy = b.get_center_of_mass()
            bin_feats[b_idx] = [
                (b.current_weight / self.wnorm), 1.0 if self.max_weight else 0.0,
                vol_util, cx / W, cy / D,
                (com_t[0] / W) if com_t else 0.0,
                (com_t[1] / D) if com_t else 0.0,
                1.0 if com_t else 0.0,
            ]
            bin_hm[b_idx] = (resample_heightmap(b.heightmap, cfg.grid) / H).astype(np.float16)
            bin_mask[b_idx] = 1.0

        # ---- global token ---------------------------------------------------
        items_left = sum(t["qty"] for t in self.types)
        global_feats = np.array([
            items_left / max(1, self.total_items0),
            self.total_placed_volume / (self.n_bins * self.bin_volume),
            self._get_bins_used() / max(1, self.n_bins),
            self.n_bins / cfg.max_bins,
        ], np.float32)

        # ---- action validity: physics minus already-refused combos ---------
        valid = self._physics_valid()
        for (t, e, r) in self.refused:
            valid[t, e, r] = False

        return {
            "item_feats": item_feats, "item_mask": item_mask,
            "edge_types": edge_types,
            "ems_feats": ems_feats, "ems_mask": ems_mask,
            "bin_feats": bin_feats, "bin_hm": bin_hm, "bin_mask": bin_mask,
            "global_feats": global_feats, "valid": valid,
        }

    # ------------------------------------------------------------------ misc

    def _get_bins_used(self) -> int:
        return sum(1 for b in self.bins if len(b.placed) > 0)

    def _compute_ems_quality(self, b: Container) -> float:
        if not b.ems_list:
            return 0.0
        vols = sorted((e.volume() for e in b.ems_list), reverse=True)[:3]
        return (sum(vols) / len(vols) / self.bin_volume) ** (1.0 / 3.0)

    def _potential(self, b: Container) -> float:
        util = sum(k.w * k.d * k.h for k in b.placed) / self.bin_volume
        return util + 0.3 * self._compute_ems_quality(b)

    def _check_affinity_placement(self, item_id: int, target_bin_idx: int) -> bool:
        """Placing item_id into target bin must not strand an affinity partner
        (same proactive rule as MultiBinPackingEnv.check_affinity_placement)."""
        if not self.affinity:
            return True
        partners = set()
        for pair in self.affinity:
            if item_id in pair:
                partners |= (pair - {item_id})
        if not partners and not any(item_id in p for p in self.affinity):
            return True
        for b_idx, b in enumerate(self.bins):
            if b_idx == target_bin_idx:
                continue
            if item_id in b.item_ids_in_bin:
                return False
            for pid in partners:
                if pid in b.item_ids_in_bin:
                    return False
        return True

    def flat_to_action(self, flat: int) -> Tuple[int, int, int]:
        R, E = self.cfg.num_rotations, self.cfg.max_ems
        return flat // (E * R), (flat // R) % E, flat % R

    # ------------------------------------------------------------------ step

    def _terminal_info(self, util: float) -> dict:
        info = {
            "utilization": util,
            "bins_used": self._get_bins_used(),
            "items_placed": self.total_items0 - sum(t["qty"] for t in self.types),
            "items_remaining": sum(t["qty"] for t in self.types),
            "per_bin_utils": [], "per_bin_weights": [], "per_bin_items": [],
        }
        for b in self.bins:
            if b.placed:
                info["per_bin_utils"].append(
                    sum(k.w * k.d * k.h for k in b.placed) / self.bin_volume)
                info["per_bin_weights"].append(b.current_weight)
                info["per_bin_items"].append(len(b.placed))
        return info

    def _completion_reward(self) -> Tuple[float, int]:
        bins_used = self._get_bins_used()
        bins_eff = 1.0 - (bins_used / max(1, self.n_bins))
        reward = 0.3 + bins_eff * 0.2
        violations = sum(1 for b in self.bins if b.placed
                         and not b.check_positive_affinity_before_completion())
        reward -= 0.1 * violations
        return reward, violations

    def step(self, flat_action: Optional[int]):
        if self.done:
            raise RuntimeError("Episode done, reset required.")
        self.env_steps += 1
        info: dict = {}
        total_capacity = self.n_bins * self.bin_volume
        util_prev = self.total_placed_volume / total_capacity
        items_left = sum(t["qty"] for t in self.types)

        # -- no feasible action anywhere: terminal ------------------------
        if flat_action is None:
            self.done = True
            reward = (self.gamma * util_prev) - util_prev
            if items_left == 0:
                bonus, av = self._completion_reward()
                reward += bonus
                if av:
                    info["affinity_violations"] = av
            else:
                reward -= 0.2 * (items_left / max(1, self.total_items0))
            info.update(self._terminal_info(util_prev))
            return self.state(), reward, True, info

        t_idx, e_idx, r_idx = self.flat_to_action(int(flat_action))
        t = self.types[t_idx]
        b_idx, ems = self.ems_refs[e_idx]
        b = self.bins[b_idx]
        item_id = t["id"]
        size = tuple(t["dims"][p] for p in ROTATIONS[r_idx])
        weight = t["weight"]

        # -- constraint referee (NOT masked - the agent must learn these) --
        violation = None
        if not b.check_weight_constraint(weight):
            violation = "weight"
        elif not b.check_incompatibility(item_id):
            violation = "incompatibility"
        else:
            fz = b.apply_gravity(ems.x, ems.y, ems.z, *size)
            if not b.check_relative_positioning(item_id, (ems.x, ems.y, fz), size):
                violation = "relative_pos"
            elif not self._check_affinity_placement(item_id, b_idx):
                violation = "affinity"

        pot_prev = self._potential(b)
        if violation is None:
            ok = b.place_at_ems(ems, size, weight=weight, item_id=item_id)
            if not ok:
                violation = "placement_failed"

        if violation is not None:
            # refusal: penalty, state unchanged except this combo is masked
            self.refused.add((t_idx, e_idx, r_idx))
            info["violation"] = violation
            reward = -self.violation_penalty
            s = self.state()
            if not s["valid"].any() or self.env_steps >= self.max_env_steps:
                self.done = True
                reward += (self.gamma * util_prev) - util_prev
                reward -= 0.2 * (items_left / max(1, self.total_items0))
                info.update(self._terminal_info(util_prev))
            return s, reward, self.done, info

        # -- successful placement ------------------------------------------
        placed_box = b.placed[-1]
        v = int(size[0] * size[1] * size[2])
        pot_next = self._potential(b)

        self.total_placed_volume += v
        t["qty"] -= 1
        self.placed_tokens.append({
            "id": item_id, "type_idx": t_idx, "dims": size, "weight": weight,
            "pos": (placed_box.x, placed_box.y, placed_box.z), "bin": b_idx,
        })
        self.refused.clear()
        self._rebuild_ems_refs()

        reward = (self.gamma * pot_next) - pot_prev
        # small bin-balance bonus (as in the enhanced env)
        avg_items = sum(len(x.placed) for x in self.bins) / max(1, self._get_bins_used())
        if len(b.placed) < avg_items:
            reward += 0.01

        items_left = sum(t2["qty"] for t2 in self.types)
        if items_left == 0:
            self.done = True
            bonus, av = self._completion_reward()
            reward += bonus
            if av:
                info["affinity_violations"] = av
            info.update(self._terminal_info(self.total_placed_volume / total_capacity))
            return self.state(), reward, True, info

        if self.env_steps >= self.max_env_steps:
            self.done = True
            util = self.total_placed_volume / total_capacity
            reward -= 0.2 * (items_left / max(1, self.total_items0))
            info.update(self._terminal_info(util))
            return self.state(), reward, True, info

        return self.state(), reward, False, info


# ===========================================================================
# Training
# ===========================================================================

def train_relational_dqn(
    problem_path: str,
    episodes: int = 400,
    seed: int = 42,
    device: str = None,
    violation_penalty: float = 0.25,
    log_interval: int = 10,
    train_freq: int = 1,
    save_path: str = None,
    cfg_overrides: dict = None,
):
    import os
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    problem = load_problem(problem_path)
    W, D, H = problem.bin_dimensions
    total_items = sum(it.quantity for it in problem.items)

    cfg = RelationalDQNConfig(device=device)
    cfg.eps_decay_episodes = max(1, int(episodes * 0.8))
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    print(f"\n{'='*80}")
    print("RELATIONAL MULTI-BIN PACKING DQN (constraints learned, not masked)")
    print(f"{'='*80}")
    print(f"Problem: {problem_path}")
    print(f"  Container: {W}x{D}x{H} | max bins: {problem.max_bins} | "
          f"max weight: {problem.max_weight}")
    print(f"  Item types: {len(problem.items)} | total items: {total_items}")
    print(f"  Constraint edges: incompat={len(problem.incompatibilities)} "
          f"affinity={len(problem.positive_affinities)} "
          f"relpos={sum(len(v) for v in (problem.relative_pos or {}).values())} "
          f"CoM={'yes' if problem.center_of_mass else 'no'}")
    print(f"  Action space: {cfg.max_types} types x {cfg.max_ems} EMS x "
          f"{cfg.num_rotations} rotations = {cfg.num_actions}")
    print(f"  Device: {device} | episodes: {episodes} | "
          f"violation penalty: {violation_penalty}")
    print(f"{'='*80}\n")

    env = RelationalPackingEnv(problem, cfg, seed=seed, gamma=cfg.gamma,
                               violation_penalty=violation_penalty)
    agent = RelationalDQNAgent(cfg)

    from collections import deque as _dq
    util_hist, bins_hist, items_hist, viol_hist = _dq(maxlen=50), _dq(maxlen=50), _dq(maxlen=50), _dq(maxlen=50)
    best_util, best_bins, best_items, best_solution = 0.0, float('inf'), 0, None
    t0 = time.time()

    global_step = 0
    for ep in range(episodes):
        s = env.reset()
        ep_ret, ep_violations, losses = 0.0, 0, []

        while True:
            a = agent.select_action(s)
            s_next, r, done, info = env.step(a)
            # dead-end steps (a=None) are stored with action -1: excluded from
            # the TD loss but their terminal reward still propagates backwards
            # through the n-step aggregation
            agent.store(s, -1 if a is None else a, r, s_next, done)
            global_step += 1
            if global_step % train_freq == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
            if "violation" in info:
                ep_violations += 1
            s = s_next
            ep_ret += r
            if done:
                break

        util = info.get("utilization", 0.0)
        bins_used = info.get("bins_used", 0)
        items_placed = info.get("items_placed", 0)
        util_hist.append(util); bins_hist.append(bins_used)
        items_hist.append(items_placed); viol_hist.append(ep_violations)

        if util > best_util:
            best_util, best_bins, best_items = util, bins_used, items_placed
            best_solution = copy.deepcopy(env.bins)

        agent.on_episode_end()

        if (ep + 1) % log_interval == 0 or ep == 0:
            print(f"Ep {ep+1:4d}/{episodes} | "
                  f"Bins {bins_used}/{env.n_bins} (MA {np.mean(bins_hist):.1f}) | "
                  f"Items {items_placed:2d}/{total_items} (MA {np.mean(items_hist):.1f}) | "
                  f"Util {util:.3f} (MA {np.mean(util_hist):.3f}) | "
                  f"Viol {ep_violations:3d} (MA {np.mean(viol_hist):.1f}) | "
                  f"eps {agent.epsilon():.3f} | "
                  f"L {np.mean(losses) if losses else 0.0:.4f} | "
                  f"{time.time()-t0:.0f}s")

    print(f"\n{'='*80}")
    print(f"DONE. Best: util={best_util:.3f}, bins={best_bins}, "
          f"items={best_items}/{total_items} | "
          f"final MA50: util={np.mean(util_hist):.3f}, "
          f"violations/ep={np.mean(viol_hist):.1f}")
    print(f"{'='*80}\n")

    if save_path:
        import os
        os.makedirs("output_data", exist_ok=True)
        if not save_path.startswith("output_data/"):
            save_path = os.path.join("output_data", os.path.basename(save_path))
        agent.save(save_path)
        print(f"Model saved to: {save_path}")

    return agent, env, best_solution


def full_datapath(filename: str) -> str:
    import os
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nesting", "inputData",
        "Benchmark dataset and instance generator for Real-World 3dBPP",
        "Input", filename)


if __name__ == "__main__":
    train_relational_dqn(
        problem_path=full_datapath("3dBPP_11.txt"),
        episodes=400,
        seed=42,
        log_interval=10,
        save_path="relational_dqn_3dBPP_11.pth",
    )
