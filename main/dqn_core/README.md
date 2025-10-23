
# dqn_core — A clean Deep Q-Network module (parametric-action ready)

**What this is:** a compact, well-documented DQN you can drop into your 3D bin-packing project.

## Highlights
- **Q(s, a)** with separate encoders for **state** and **action** (parametric actions).
- **Action masking**: invalid actions are excluded during selection and targets.
- **Double DQN** and **Dueling head** (toggle in config).
- **Target network**: hard or soft updates.
- **N-step returns** (n_step ≥ 1).
- **Prioritized Replay** (optional, off by default) — omitted in code for brevity; can be added later.
- PyTorch, type-annotated, minimal dependencies.

## Files
- `dqn.py` — the library (Agent, Networks, Replay).
- `example_usage.py` — tiny mock to sanity-check the API with variable action sets.

## Integration with your env
At each step:
- Build `action_feats` (A × action_feat_dim) for feasible actions and a 1/0 `mask` (A).
- Pad both to fixed sizes (`max_actions`).
- After `step`, also build the *next* ones.
- Call `agent.select_action`, `agent.store(...)`, and `agent.train_step()` as shown in example.
