
# Sanity-check the DQN module with a mock variable-action setting.
# Reward is correlated with one feature so the agent can learn to pick it.

import numpy as np
from dqn import DQNAgent, DQNConfig

OBS_DIM = 8
ACTION_FEAT_DIM = 5
MAX_ACTIONS = 32

def make_obs():
    return np.random.randn(OBS_DIM).astype(np.float32)

def make_action_set():
    A = np.random.randint(5, MAX_ACTIONS+1)
    feats = np.random.randn(A, ACTION_FEAT_DIM).astype(np.float32)
    mask = np.zeros((MAX_ACTIONS,), dtype=np.float32)
    feats_pad = np.zeros((MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32)
    feats_pad[:A] = feats
    mask[:A] = 1.0
    return feats, mask[:A], feats_pad, mask

def reward_from_action(feats, idx):
    return float((feats[idx, -1] + 2.0) / 4.0)  # in ~[0,1]

cfg = DQNConfig(
    obs_dim=OBS_DIM,
    action_feat_dim=ACTION_FEAT_DIM,
    max_actions=MAX_ACTIONS,
    device="cpu",
    buffer_size=20000,
    batch_size=64,
    eps_decay_steps=5000,
    target_update_interval=500,
    n_step=1,
    gamma=0.99,
)

agent = DQNAgent(cfg)

losses = []
for ep in range(50):
    s = make_obs()
    ep_ret = 0.0
    for t in range(50):
        feats, mask_short, feats_pad, mask_pad = make_action_set()
        a = agent.select_action(s, feats, mask_short)
        if a is None: break
        r = reward_from_action(feats, a)
        s_next = make_obs()

        next_feats, next_mask_short, next_feats_pad, next_mask_pad = make_action_set()

        agent.store(s, a, r, s_next, False,
                    curr_action_feats=feats_pad, curr_mask=mask_pad,
                    next_action_feats=next_feats_pad, next_mask=next_mask_pad)

        for _ in range(2):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        s = s_next
        ep_ret += r
    if (ep+1) % 10 == 0 and len(losses) >= 10:
        print(f'Ep {ep+1:>3}  mean_loss(last 100)={np.mean(losses[-100:]):.4f}  eps~{agent.epsilon():.3f}  ep_ret~{ep_ret:.2f}')

print('If mean_loss decreases and eps decreases, the DQN is training.')
