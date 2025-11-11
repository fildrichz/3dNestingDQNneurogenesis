# Credit Assignment for Large Episodes

## The Problem with n_step=15 for 200-item episodes

### Propagation Chain

For a 200-step episode with n_step=15:

```
Step 1   → sees steps 1-15,   relies on Q(s_16)
Step 16  → sees steps 16-30,  relies on Q(s_31)
Step 31  → sees steps 31-45,  relies on Q(s_46)
...
Step 186 → sees steps 186-200, gets FINAL REWARD (+0.5)
```

**Propagation chain:** ~13 TD updates needed for signal to reach step 1

### Time to Propagate

With experience replay:
- Each batch samples 128 random transitions
- For signal to reach step 1, need Q(s_16), Q(s_31), ... Q(s_186) all learned
- Requires MANY episodes and lucky sampling

**Estimate:** 50-100 episodes before step 1 sees proper value from final reward

### Value Decay

Even if it propagates instantly:
```python
gamma = 0.992
gamma^200 = 0.992^200 = 0.202
```

Final +0.5 bonus is worth only **+0.10** to step 1 action!

## Real Solutions

### Solution 1: Monte Carlo Returns ⭐ (BEST)

Store full episodes, compute exact returns after completion:

```python
# After episode completes
returns = []
G = 0
for r in reversed(episode_rewards):
    G = r + gamma * G
    returns.insert(0, G)

# Each transition gets EXACT return from that point
for i, (s, a, _, s_next, _) in enumerate(episode):
    buffer.push(s, a, returns[i], s_next, done)
```

**Pros:**
- ✅ Perfect credit assignment
- ✅ Every action sees final outcome
- ✅ No propagation delay

**Cons:**
- ❌ Can't learn during episode (must wait for completion)
- ❌ Higher variance (no bootstrapping)
- ❌ Requires storing full episodes

### Solution 2: Adaptive N-Step

Set n_step based on episode length:

```python
# Use 30-50% of episode length
n_step = max(15, episode_length // 3)

# For 200-item episode: n_step = 66
# For 50-item episode: n_step = 16
```

**Pros:**
- ✅ Scales with problem size
- ✅ Better propagation for long episodes

**Cons:**
- ❌ Still not perfect for very long episodes
- ❌ Variable n_step might affect learning stability

### Solution 3: TD(λ) - Eligibility Traces

Use eligibility traces (weighted average of all n-step returns):

```python
# λ = 0.9 means exponentially weighted combination
# Effectively sees ALL future rewards with decay
```

**Pros:**
- ✅ Excellent credit assignment
- ✅ Handles episodes of any length
- ✅ Can learn online

**Cons:**
- ❌ More complex implementation
- ❌ Requires forward or backward view algorithm

### Solution 4: Episode-Proportional N-Step

After each episode, compute n-step returns with large n:

```python
# During episode: use n_step=15 for fast learning
# After episode: recompute with n_step=episode_length for full transitions
# Add both to replay buffer
```

**Pros:**
- ✅ Gets benefits of both TD and MC
- ✅ Fast online learning + accurate offline updates

**Cons:**
- ❌ More complex
- ❌ Requires storing episodes temporarily

## Recommendation

For variable-length episodes (50-200+ items), use **Monte Carlo Returns**:

1. **Store full episodes** in a separate buffer
2. **Compute exact returns** when episode completes
3. **Add to replay buffer** with correct returns
4. **Sample from replay buffer** as normal

This gives perfect credit assignment regardless of episode length.

## Current State

With n_step=15:
- ✅ Good for 50-item episodes (sees 30% of episode)
- ⚠️  Mediocre for 100-item episodes (sees 15%)
- ❌ Poor for 200-item episodes (sees 7.5%)

The signal WILL eventually propagate through experience replay, but it takes many episodes and the value decay makes it weak.

## Next Steps

Should we implement Monte Carlo returns? It's the cleanest solution for this problem.
