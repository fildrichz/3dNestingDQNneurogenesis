# Curriculum Learning Review - experiment_multi_problem.py

## Summary
The curriculum learning implementation in `experiment_multi_problem.py` has **FUNDAMENTAL CONCEPTUAL ERRORS** that make it incorrect.

---

## ❌ CRITICAL: Curriculum Applied at Wrong Level

### The Fundamental Problem

**Current Implementation (WRONG):**
Curriculum is applied **across GA generations** in `experiment_multi_problem.py`:

**Curriculum Schedule Definition** (lines 700-717):
```python
curriculum_schedule = {
    'generations': [0, generations // 3, 2 * generations // 3],  # ← WRONG LEVEL
    'item_fractions': [0.3, 0.6, 1.0],
    'episodes': [
        max(10, episodes_per_eval // 2),
        max(15, int(episodes_per_eval * 0.75)),
        episodes_per_eval
    ]
}
```

**What currently happens:**
- Generation 0-16: All genomes evaluated on 30% items, 50 episodes
- Generation 17-33: All genomes evaluated on 60% items, 75 episodes
- Generation 34+: All genomes evaluated on 100% items, 100 episodes

**Why this is fundamentally wrong:**

1. **Incomparable Fitness Scores**:
   - Genome A (gen 5) trained on 30% items → fitness 0.85
   - Genome B (gen 40) trained on 100% items → fitness 0.80
   - Which is better? You can't tell! They solved different problems.

2. **Defeats GA Purpose**:
   - GA should select architectures that excel at the FULL problem
   - Instead, it's selecting based on progressively different problem difficulties
   - Early generations get artificially inflated fitness scores

3. **Items Not Sorted by Difficulty**:
   - `items[:num_items]` assumes items are ordered by difficulty
   - They're not! So you're just getting a random subset
   - Not a true curriculum progression

## ✅ Correct Curriculum Learning Approach

**Curriculum should apply WITHIN each genome's training, not across GA generations.**

### Correct Implementation:

```python
def evaluate_genome_with_curriculum(genome, training_problems, total_episodes=100):
    """
    Train a single genome using curriculum learning.

    Curriculum applies episode-by-episode WITHIN this evaluation:
    - Episodes 0-33: Train on 30% of items (easiest)
    - Episodes 34-66: Train on 60% of items (medium)
    - Episodes 67-100: Train on 100% of items (full problem)

    Then evaluate at eps=0 on FULL PROBLEM for fitness.
    """

    # Define curriculum stages within this evaluation
    curriculum_stages = [
        {'episodes': (0, 33), 'item_fraction': 0.3},
        {'episodes': (34, 66), 'item_fraction': 0.6},
        {'episodes': (67, 100), 'item_fraction': 1.0}
    ]

    for episode in range(total_episodes):
        # Determine current curriculum stage
        current_fraction = 1.0
        for stage in curriculum_stages:
            if stage['episodes'][0] <= episode < stage['episodes'][1]:
                current_fraction = stage['item_fraction']
                break

        # Select items for this episode
        if current_fraction < 1.0:
            # Sort items by difficulty (e.g., volume)
            items_sorted = sorted(items, key=lambda x: x[0] * x[1] * x[2])
            num_items = max(1, int(len(items_sorted) * current_fraction))
            training_items = items_sorted[:num_items]
        else:
            training_items = items

        # Train on current difficulty level
        train_episode(agent, env, training_items)

    # CRITICAL: Evaluate fitness on FULL problem (100% items)
    fitness = evaluate_at_greedy(agent, env, items)  # All items, eps=0

    return fitness
```

### Key Principles:

1. **Same problem for all genomes**: Every genome in the GA is evaluated on the same FULL problem
2. **Curriculum within training**: Difficulty increases episode-by-episode during each genome's training
3. **Fair comparison**: All fitness scores are from the same problem difficulty
4. **Items sorted by difficulty**: Start with smaller/easier items, progress to larger/harder ones

---

## ❌ Issues in Current Implementation

### Issue 1: Curriculum at Generation Level (experiment_multi_problem.py)

**Location**: Lines 444-476, 700-717

**Problem**: Curriculum applied across GA generations instead of within genome training.

**Impact**:
- Genomes trained in different generations face different problem difficulties
- Fitness scores are incomparable
- GA selection is biased toward early easy-problem performers

**Fix**: Remove generation-level curriculum entirely, or apply it per-episode within each evaluation.

### Issue 2: Unsorted Items (everywhere)

**Location**: Lines 111-114 (experiment_multi_problem.py), similar in ga_evolution.py

**Current Code**:
```python
items = items[:num_items]  # Takes arbitrary first N items
```

**Problem**: Items are NOT sorted by difficulty, so this doesn't create a progression.

**Fix**: Sort items before taking subset:
```python
# Sort by volume (smaller items = easier)
items_sorted = sorted(items, key=lambda x: x[0] * x[1] * x[2])
num_items = max(1, int(len(items_sorted) * item_fraction))
items = items_sorted[:num_items]
```

### Issue 3: ga_evolution.py Ignores Curriculum Schedule

**Location**: ga_evolution.py, line 355-361

**Problem**: The `curriculum_schedule` parameter is accepted but never used.

**Status**: Actually this might be CORRECT behavior if curriculum shouldn't be at generation level!

**Decision needed**: Should ga_evolution.py use:
- Option A: No curriculum (current behavior, might be correct)
- Option B: Episode-level curriculum within each genome's evaluation

---

## Recommendations

### Critical (Must Fix)

1. **Redesign Curriculum Application**:
   - Move curriculum from generation-level to episode-level
   - Apply curriculum within each genome's training, not across GA generations
   - Ensure all genomes are evaluated on the same full problem for fair fitness comparison

2. **Sort Items by Difficulty**:
   - Implement proper item sorting before taking subsets
   - Use volume (w × d × h) as a simple difficulty metric
   - Smaller items first → larger items later

3. **Fix evaluate_genome_multi_problem()**:
   - Currently takes `item_fraction` as a parameter for the entire evaluation
   - Should instead vary `item_fraction` per episode within the training loop
   - Always evaluate final fitness on 100% items

### Proposed Changes

#### Option A: Simple Fix - Remove Curriculum from GA Level
If curriculum doesn't help, just remove it:
- Set `use_curriculum=False` by default
- All genomes train on full problem from the start
- Simpler and guarantees fair comparison

#### Option B: Proper Episode-Level Curriculum
Implement curriculum correctly:

```python
def evaluate_genome_multi_problem(
    genome: NetworkGenome,
    training_problems: List[Tuple[BinPackingProblem, Path]],
    episodes: int = 100,
    use_curriculum: bool = True,  # Changed parameter
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Evaluate genome with optional curriculum learning.

    If use_curriculum=True:
        - Episodes 0-33: 30% items (smallest by volume)
        - Episodes 34-66: 60% items
        - Episodes 67-100: 100% items
    Fitness always evaluated on 100% items at eps=0.
    """

    # Load all environments
    envs_and_items = []
    for problem, problem_path in training_problems:
        items = load_problem_as_items(problem)

        # Sort items by volume for proper curriculum
        if use_curriculum:
            items = sorted(items, key=lambda x: x[0] * x[1] * x[2])

        W, D, H = problem.bin_dimensions
        env = MultiBinPackingEnv(...)
        envs_and_items.append((env, items, problem_path))

    # Build agent...
    agent = DQNAgentEnhanced(cfg)

    # Define curriculum stages
    if use_curriculum:
        curriculum_stages = [
            (0, episodes // 3, 0.3),
            (episodes // 3, 2 * episodes // 3, 0.6),
            (2 * episodes // 3, episodes, 1.0)
        ]
    else:
        curriculum_stages = [(0, episodes, 1.0)]

    # Round-robin training with curriculum
    for episode in range(episodes):
        # Determine curriculum fraction for THIS episode
        item_fraction = 1.0
        for start_ep, end_ep, fraction in curriculum_stages:
            if start_ep <= episode < end_ep:
                item_fraction = fraction
                break

        # Select problem (round-robin)
        problem_idx = episode % len(envs_and_items)
        env, items_full, problem_path = envs_and_items[problem_idx]

        # Apply curriculum to items for this episode
        if item_fraction < 1.0:
            num_items = max(1, int(len(items_full) * item_fraction))
            items_episode = items_full[:num_items]  # Already sorted above
        else:
            items_episode = items_full

        # Train on items_episode for this episode
        obs = env.reset(items=items_episode.copy())
        # ... training loop ...

    # Evaluate at eps=0 on FULL problems (100% items)
    for env, items_full, problem_path in envs_and_items:
        # Evaluate using ALL items
        # ... evaluation loop ...

    return fitness, metrics
```

### Testing Plan

1. **Verify Curriculum Application**:
   - Log `item_fraction` and `num_items` for each episode
   - Confirm progression: 30% → 60% → 100%

2. **Confirm Item Sorting**:
   - Print first/last item volumes before curriculum
   - Verify smallest items come first

3. **Compare Approaches**:
   - Run with curriculum: `--use-curriculum`
   - Run without: `--no-curriculum`
   - Compare final fitness and training curves

4. **Validate Fair Comparison**:
   - All genomes should show same evaluation metrics
   - Fitness scores should be comparable across generations

---

## Questions to Answer

1. **Does curriculum help?**
   - Compare convergence speed with/without curriculum
   - Does starting with easier problems improve final performance?

2. **What difficulty metric?**
   - Volume (w × d × h)?
   - Aspect ratio (max/min dimension)?
   - Some combination?

3. **What curriculum schedule?**
   - Current: 30% → 60% → 100% at 1/3 intervals
   - Alternative: More gradual (50% → 75% → 100%)
   - Alternative: Faster (50% → 100% at halfway point)
