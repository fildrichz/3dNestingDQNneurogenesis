
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# Import your core modules
from nesting.packing_core import Container
from dqn_core.dqn import DQNAgent, DQNConfig  # <-- uses your dqn.py

# ---------------------
# Lightweight RL Env around Container
# ---------------------
def volume(size: Tuple[int,int,int]) -> int:
    w,d,h = size; return int(w)*int(d)*int(h)

class PackingEnv:
    def __init__(self, W=40, D=40, H=40, items: List[Tuple[int,int,int]] = None, max_actions: int = 128, topk_eps: int = 32, seed: int = 0, gamma: float = 0.992):
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
        return (self.C._fits_caps(ep, size) and self.C._fits_container(ep, size) and self.C._fits_collision_free(ep, size))

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

        """def _obs(self) -> np.ndarray:
        W,D,H = self.bin_size
        items_arr = np.array(self.items, dtype=np.float32) if self.items else np.zeros((0,3), dtype=np.float32)
        eps_arr = np.array(self.C.eps, dtype=np.float32) if self.C.eps else np.zeros((0,3), dtype=np.float32)
        if len(items_arr) > 0:
            mean_items = items_arr.mean(axis=0); std_items = items_arr.std(axis=0)
        else:
            mean_items = np.zeros(3, dtype=np.float32); std_items = np.zeros(3, dtype=np.float32)
        if len(eps_arr) > 0:
            mean_eps = eps_arr.mean(axis=0)
        else:
            mean_eps = np.zeros(3, dtype=np.float32)
        item_frac = np.array([len(self.items)/max(1,self.n_items)], dtype=np.float32)
        ep_frac = np.array([len(self.C.eps)/100.0], dtype=np.float32)
        packed_frac = np.array([self.total_placed_volume/self.bin_volume], dtype=np.float32)
        return np.concatenate([np.array([W,D,H], dtype=np.float32), mean_items, std_items, mean_eps, item_frac, ep_frac, packed_frac], axis=0)"""

    #reduced obs for
    def _obs(self) -> np.ndarray:
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
            reward += util_next                # terminal bonus
            info["utilization"] = util_next

        return self._obs(), reward, self.done, info


# ---------------------
# Feature utils for DQN from dqn.py
# ---------------------
# in packing_with_dqncore_potential.py
ACTION_FEAT_DIM = 19

def _safe_caps(C, ep):
    # residual caps you already maintain; fallback to container bounds
    x,y,z = ep
    return C.ep_rs.get(ep, (C.w - x, C.d - y, C.h - z))

def build_action_features(env: PackingEnv, actions):
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
    A = feats.shape[0]
    F = np.zeros((maxA, ACTION_FEAT_DIM), np.float32)
    M = np.zeros((maxA,), np.float32)
    F[:A] = feats
    M[:A] = mask_short[:A] if mask_short.shape[0] >= A else 1.0
    return F, M

# ---------------------
# Tiny training driver
# ---------------------
def make_items(n=12, lo=4, hi=10, seed=0):
    rng=np.random.default_rng(seed)
    return [(int(rng.integers(lo,hi)), int(rng.integers(lo,hi)), int(rng.integers(lo,hi))) for _ in range(n)]


#i want to keep items same for the whole training, lets make alternate copy of train_pack_dqn function with fixed items
def train_pack_dqn_fixed_items(episodes=50, W=40, D=40, H=40, n_items=50, seed=42, max_actions=128, topk_eps=48):
    env = PackingEnv(W,D,H, items=make_items(n=n_items, seed=seed), max_actions=max_actions, topk_eps=topk_eps, seed=seed, gamma=0.992)
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cpu",
        gamma=0.992, lr=2e-4,
        batch_size=64, buffer_size=200_000,
        eps_decay_steps=1000,
        target_update_interval=1_000,
        n_step=3,
    )
    agent = DQNAgent(cfg)

    import collections
    util_hist = collections.deque(maxlen=50)

    items_consistent=make_items(n=n_items, seed=42)

    for ep in range(episodes):
        obs = env.reset(items = items_consistent)  # vary instance each episode
        ep_ret=0.0; steps=0
        while True:
            actions, mask_short, _ = env.action_space()
            feats = build_action_features(env, actions) if len(actions)>0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            act_idx = agent.select_action(obs, feats if feats.shape[0]>0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                                          mask_short if mask_short.shape[0]>0 else np.zeros((1,), np.float32))
            act = None if (act_idx is None or actions==[] or actions[act_idx] is None) else actions[act_idx]
            # CURRENT (padded) for storage
            currF, currM = pad_feats_mask(feats if feats.shape[0]>0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                                          mask_short if mask_short.shape[0]>0 else np.zeros((0,), np.float32),
                                          env.max_actions)
            nobs, rew, done, info = env.step(act)
            n_actions, n_mask_short, _ = env.action_space()
            n_feats = build_action_features(env, n_actions) if len(n_actions)>0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            nextF, nextM = pad_feats_mask(n_feats, n_mask_short if n_mask_short.shape[0]>0 else np.zeros((0,), np.float32),
                                          env.max_actions)

            agent.store(obs, act_idx, rew, nobs, done,
                        curr_action_feats=currF, curr_mask=currM,
                        next_action_feats=nextF, next_mask=nextM)

            for _ in range(2):
                agent.train_step()

            obs=nobs; ep_ret+=rew; steps+=1
            if done:
                util=info.get("utilization", 0.0); util_hist.append(util)
                ma=np.mean(util_hist) if len(util_hist)>0 else util
                print(f"ep={ep+1:03d} steps={steps:02d} return={ep_ret:.3f} util={util:.3f} util_ma50={ma:.3f} eps~{agent.epsilon():.3f}")
                break
    
    env.C.plot3d(title="last episode packing result")
 

def train_pack_dqn(episodes=50, W=40, D=40, H=40, n_items=50, seed=42, max_actions=128, topk_eps=48):
    env = PackingEnv(W,D,H, items=make_items(n=n_items, seed=seed), max_actions=max_actions, topk_eps=topk_eps, seed=seed, gamma=0.992)
    obs = env.reset()
    OBS_DIM = obs.shape[0]
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        action_feat_dim=ACTION_FEAT_DIM,
        max_actions=max_actions,
        device="cpu",
        gamma=0.992, lr=2e-4,
        batch_size=64, buffer_size=200_000,
        eps_decay_steps=1000,
        target_update_interval=1_000,
        n_step=3,
    )
    agent = DQNAgent(cfg)

    import collections
    util_hist = collections.deque(maxlen=50)

    for ep in range(episodes):
        obs = env.reset(items=make_items(n=n_items, seed=None))  # vary instance each episode
        ep_ret=0.0; steps=0
        while True:
            actions, mask_short, _ = env.action_space()
            feats = build_action_features(env, actions) if len(actions)>0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            act_idx = agent.select_action(obs, feats if feats.shape[0]>0 else np.zeros((1, ACTION_FEAT_DIM), np.float32),
                                          mask_short if mask_short.shape[0]>0 else np.zeros((1,), np.float32))
            act = None if (act_idx is None or actions==[] or actions[act_idx] is None) else actions[act_idx]
            # CURRENT (padded) for storage
            currF, currM = pad_feats_mask(feats if feats.shape[0]>0 else np.zeros((0, ACTION_FEAT_DIM), np.float32),
                                          mask_short if mask_short.shape[0]>0 else np.zeros((0,), np.float32),
                                          env.max_actions)
            nobs, rew, done, info = env.step(act)
            n_actions, n_mask_short, _ = env.action_space()
            n_feats = build_action_features(env, n_actions) if len(n_actions)>0 else np.zeros((0, ACTION_FEAT_DIM), np.float32)
            nextF, nextM = pad_feats_mask(n_feats, n_mask_short if n_mask_short.shape[0]>0 else np.zeros((0,), np.float32),
                                          env.max_actions)

            agent.store(obs, act_idx, rew, nobs, done,
                        curr_action_feats=currF, curr_mask=currM,
                        next_action_feats=nextF, next_mask=nextM)

            for _ in range(2):
                agent.train_step()

            obs=nobs; ep_ret+=rew; steps+=1
            if done:
                util=info.get("utilization", 0.0); util_hist.append(util)
                ma=np.mean(util_hist) if len(util_hist)>0 else util
                print(f"ep={ep+1:03d} steps={steps:02d} return={ep_ret:.3f} util={util:.3f} util_ma50={ma:.3f} eps~{agent.epsilon():.3f}")
                break
    
    env.C.plot3d(title="last episode packing result")

if __name__ == "__main__":
    # quick smoke test (reduce episodes if needed)
    train_pack_dqn(episodes=100, n_items=50, W=20, D=20, H=20, seed=42, max_actions=50, topk_eps=1000)

    #train_pack_dqn_fixed_items(episodes=100, n_items=50, W=20, D=20, H=20, seed=42, max_actions=50, topk_eps=1000)
