# Curriculum Learning Review - experiment_multi_problem.py

## Summary
The curriculum learning implementation in `experiment_multi_problem.py` has **one critical bug** and **one design concern**.

---

## ✅ What's Working Correctly

### 1. Multi-Problem Implementation (experiment_multi_problem.py)
The curriculum learning IS being used in `experiment_multi_problem.py`:

**Curriculum Schedule Definition** (lines 700-717):
```python
curriculum_schedule = {
    'generations': [0, generations // 3, 2 * generations // 3],
    'item_fractions': [0.3, 0.6, 1.0],
    'episodes': [
        max(10, episodes_per_eval // 2),
        max(15, int(episodes_per_eval * 0.75)),
        episodes_per_eval
    ]
}
```

**Curriculum Application** (lines 444-460):
```python
# Determine curriculum parameters
current_item_fraction = 1.0
current_episodes = episodes_per_eval

if curriculum_schedule:
    gen_thresholds = curriculum_schedule.get('generations', [])
    item_fractions = curriculum_schedule.get('item_fractions', [1.0])
    episode_counts = curriculum_schedule.get('episodes', [episodes_per_eval])

    for i, threshold in enumerate(gen_thresholds):
        if gen >= threshold:
            current_item_fraction = item_fractions[i]
            current_episodes = episode_counts[i]
```

**Item Selection** (lines 111-114):
```python
# Apply curriculum learning
if item_fraction < 1.0:
    num_items = max(1, int(len(items) * item_fraction))
    items = items[:num_items]
```

This logic works correctly - it progressively increases difficulty by:
- Generation 0-16: 30% items, 50 episodes
- Generation 17-33: 60% items, 75 episodes
- Generation 34+: 100% items, 100 episodes

---

## ❌ Issues Found

### Issue 1: Curriculum Schedule Loop Logic (experiment_multi_problem.py:453-456)

**Problem**: Inefficient loop that overwrites values unnecessarily.

**Current Code**:
```python
for i, threshold in enumerate(gen_thresholds):
    if gen >= threshold:
        current_item_fraction = item_fractions[i]
        current_episodes = episode_counts[i]
```

**What happens**: For generation 35 with thresholds [0, 16, 33]:
1. Checks gen >= 0: TRUE → sets item_fraction=0.3, episodes=50
2. Checks gen >= 16: TRUE → sets item_fraction=0.6, episodes=75
3. Checks gen >= 33: TRUE → sets item_fraction=1.0, episodes=100

**Status**: This works correctly but is inefficient. It should break after finding the right threshold.

**Recommended Fix**:
```python
# Find the highest threshold that has been reached
for i in range(len(gen_thresholds) - 1, -1, -1):
    if gen >= gen_thresholds[i]:
        current_item_fraction = item_fractions[i]
        current_episodes = episode_counts[i]
        break
```

### Issue 2: Item Selection Assumes Sorted Difficulty (experiment_multi_problem.py:111-114)

**Problem**: Curriculum learning assumes items are pre-sorted by difficulty.

**Current Code**:
```python
items = items[:num_items]  # Takes FIRST N items
```

**Risk**: If items are NOT sorted by difficulty (e.g., randomized or sorted by size), this won't provide a proper curriculum. The agent would just train on an arbitrary subset, not progressively harder problems.

**Questions to Answer**:
1. Are items in the dataset files sorted by difficulty?
2. If not, should they be sorted before applying curriculum?
3. What metric defines "difficulty" (volume, aspect ratio, quantity)?

**Recommended Fix**:
Consider sorting items by difficulty first:
```python
# Apply curriculum learning
if item_fraction < 1.0:
    # Sort items by difficulty metric (e.g., volume)
    items_sorted = sorted(items, key=lambda x: x.w * x.d * x.h)
    num_items = max(1, int(len(items_sorted) * item_fraction))
    items = items_sorted[:num_items]
```

---

## ⚠️ Critical Bug in ga_evolution.py

### Issue 3: Curriculum Schedule NOT Used in ga_evolution.py

**Problem**: The `curriculum_schedule` parameter is accepted but **NEVER USED** in the evolution loop!

**File**: `ga_evolution.py` (used by `experiment_problem_specific.py`)

**Evidence**:
- Line 200: `curriculum_schedule: Optional[Dict] = None` (parameter accepted)
- Lines 355-361: `evaluate_genome_fitness()` called WITHOUT curriculum parameters
- The `item_fraction` parameter is never passed, so it defaults to 1.0

**Impact**: When running `experiment_problem_specific.py` with `--use-curriculum`, the curriculum schedule is created but completely ignored during evolution!

**Recommended Fix**:
```python
# In ga_evolution.py, inside the main loop (after line 336)
# Determine curriculum parameters based on current generation
current_item_fraction = 1.0
current_episodes = episodes_per_eval

if curriculum_schedule:
    gen_thresholds = curriculum_schedule.get('generations', [])
    item_fractions = curriculum_schedule.get('item_fractions', [1.0])
    episode_counts = curriculum_schedule.get('episodes', [episodes_per_eval])

    for i in range(len(gen_thresholds) - 1, -1, -1):
        if gen >= gen_thresholds[i]:
            current_item_fraction = item_fractions[i]
            current_episodes = episode_counts[i]
            break

    if verbose:
        print(f"Curriculum: {current_item_fraction*100:.0f}% items, "
              f"{current_episodes} episodes")

# Then update the evaluation call (line 355):
fitness, metrics = evaluate_genome_fitness(
    genome=genome,
    env=env,
    items=items.copy(),
    episodes=current_episodes,  # Use curriculum episodes
    verbose=False,
    item_fraction=current_item_fraction  # Use curriculum item fraction
)
```

---

## Recommendations

### High Priority
1. **Fix ga_evolution.py**: Add curriculum application logic (Issue #3)
2. **Verify item ordering**: Check if items are sorted by difficulty (Issue #2)

### Medium Priority
3. **Optimize loop**: Use reverse iteration with break (Issue #1)
4. **Add validation**: Warn if curriculum is enabled but items aren't sorted
5. **Add tests**: Unit tests for curriculum schedule application

### Low Priority
6. **Document assumptions**: Clarify item ordering requirements in docstrings
7. **Make configurable**: Add option to specify difficulty metric for sorting

---

## Testing Recommendations

To verify curriculum is working:

1. **Print curriculum status per generation**:
   - Current item fraction
   - Current episode count
   - Number of items being used

2. **Track learning curves separately**:
   - Early generations (30% items)
   - Mid generations (60% items)
   - Late generations (100% items)

3. **Compare with/without curriculum**:
   - Run same experiment with `--no-curriculum`
   - Compare convergence speed and final performance
