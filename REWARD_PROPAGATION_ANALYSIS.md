# Reward Propagation Analysis - The Real Problem

## Your Concern is Correct ✅

You identified the core issue: **The final reward signal isn't reaching early decisions.**

## Current Setup

### N-Step Returns (line 734)
```python
n_step=3  # Only looks 3 steps ahead
```

### Reward Structure
```python
# Per-step reward (line 348)
reward = (gamma * util_next) - util_prev  # Small incremental reward

# Final reward (line 362)
reward += 0.3 + bins_efficiency * 0.2  # BIG bonus only at the end
```

### The Problem

**For a 52-item episode:**

1. **Step 1** (place first item):
   - Immediate reward: ~0.001 (tiny util increase)
   - 3-step return: r₁ + γ·r₂ + γ²·r₃ + γ³·V(s₄)
   - **Never sees the final +0.5 bonus at step 52!**

2. **Step 49** (place 49th item):
   - Can see 3 steps ahead (50, 51, 52)
   - **Might** see final bonus if episode ends at step 52

3. **Step 52** (place last item):
   - Gets the big final bonus: +0.5
   - But this signal doesn't propagate back to step 1-48!

## Why Value Doesn't Propagate Back

### Bellman Equation with N-Step Returns

For n=3:
```
Q(s₁, a₁) = r₁ + γ·r₂ + γ²·r₃ + γ³·max Q(s₄, a)
```

**This means:**
- Action at step 1 only "sees" rewards from steps 1, 2, 3
- It relies on Q(s₄, a) to represent future value
- But Q(s₄) must be learned separately from its own 3-step experience
- The final bonus at step 52 must "percolate" backwards through ~17 updates!

### The Propagation Chain

```
Step 52: Q(s₅₂) learns from final bonus (+0.5)
Step 49: Q(s₄₉) learns from Q(s₅₂) via 3-step return
Step 46: Q(s₄₆) learns from Q(s₄₉) via 3-step return
Step 43: Q(s₄₃) learns from Q(s₄₆) via 3-step return
...
Step 1:  Q(s₁) learns from Q(s₄) via 3-step return (after ~17 episodes!)
```

**Problem:** This requires ~17 episodes for the signal to reach step 1, and assumes:
- States are revisited frequently
- No distribution shift
- Gradual learning rate
- Target network doesn't change too fast

## Evidence This Is Happening

### Symptoms of Poor Value Propagation:

1. **Sparse, inefficient packing** ✓
   - Early actions don't optimize for final outcome
   - Only late-game actions see the completion bonus

2. **High variance in results** (likely)
   - Some episodes get lucky with state overlap
   - Others don't learn at all

3. **Slow learning** (likely)
   - Need many episodes for value to propagate
   - Requires repeated state visits

## Current Gamma = 0.992

With γ=0.992 and n_step=3:

**3-step discount:**
```
γ³ = 0.992³ = 0.976
```

**49-step discount (to reach final reward from step 1):**
```
γ⁴⁹ = 0.992⁴⁹ = 0.646
```

So even if the value DID propagate instantly, the final +0.5 bonus would only be worth +0.32 to the first action.

## Why Potential-Based Shaping Helps (But Not Enough)

The current per-step reward:
```python
reward = (gamma * util_next) - util_prev
```

Is "potential-based shaping" which theoretically doesn't change optimal policy. But:

**It only helps if the potential correlates with final outcome!**

Current potential = utilization

But utilization at step 25 doesn't tell you:
- Will you fit all remaining items?
- Are you creating fragmentation that prevents future placements?
- Are you using the right orientations?

**The intermediate utilization is not a good proxy for final success!**

## Solutions (In Order of Correctness)

### Solution 1: Increase N-Step ⭐ (RECOMMENDED)

```python
# Old
n_step=3

# New (see further into future)
n_step=10  # or even n_step=20
```

**Pros:**
- Theoretically sound
- Propagates final reward better
- No bias introduced

**Cons:**
- Requires more memory (stores longer n-step buffer)
- Slightly slower updates
- Still doesn't solve full credit assignment

### Solution 2: Monte Carlo Returns

Instead of n-step TD, use full episode returns:

```python
# After episode completes, compute returns for each step
returns = []
G = 0
for r in reversed(rewards_from_episode):
    G = r + gamma * G
    returns.insert(0, G)

# Store these full returns in replay buffer
```

**Pros:**
- Perfect credit assignment
- Every action sees final outcome
- Theoretically optimal

**Cons:**
- Need to wait until episode ends
- Can't learn during episode
- High variance (no bootstrapping)
- Requires off-policy corrections

### Solution 3: Use Hindsight/Backward View

After episode, compute what the Q-values SHOULD have been:

```python
# After episode with return G_total
# Update Q(s₁,a₁) toward G_total - (sum of rewards after step 1)
```

This is like "hindsight experience replay" but for rewards.

### Solution 4: Better Intermediate Rewards (What I Did)

Add rewards that correlate with final success:
- Height penalty → lower packing is better
- Slack penalty → tight fits prevent fragmentation
- Tightness bonus → good orientations

**Pros:**
- Immediate feedback
- Guides exploration
- Reduces variance

**Cons:**
- Can bias policy if rewards are wrong!
- Might find local optimum
- Not guaranteed to optimize true objective

## Diagnosis Tools

Let me create a diagnostic to check if Q-values are propagating:

```python
# After training, check Q-values for early vs late actions
# If working correctly:
# - Q(s₁) for good first actions should be ~same as final episode return
# - Q(s₅₀) should also reflect final bonus

# If broken:
# - Q(s₁) will be tiny (only sees immediate util increase)
# - Q(s₅₀) will be much higher (can see final bonus)
```

## Recommendation

### Immediate Action: Increase N-Step

Change line 734:
```python
n_step=3   # OLD
n_step=15  # NEW - sees ~1/3 of episode
```

This gives better credit assignment without changing the MDP.

### Keep Compactness Rewards BUT as Shaping

The compactness rewards I added CAN help, but frame them as **potential-based shaping** to maintain optimality:

```python
# Define potential function
def potential(state):
    return avg_height_penalty + slack_penalty + ...

# Reward shaping
reward = original_reward + gamma * potential(s_next) - potential(s)
```

This is theoretically sound and helps learning.

### Best Solution: Both

1. Increase n_step to 15-20
2. Keep compactness rewards as guidance
3. Monitor if final completion rate improves

The compactness rewards help exploration and reduce variance, while larger n_step ensures the final objective is learned.

## Test: Check Q-Value Propagation

I'll create a diagnostic that:
1. Runs an episode
2. Records Q(s,a) for each action taken
3. Compares Q-values to actual returns
4. Shows if early actions underestimate final outcome
