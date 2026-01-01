# Model Testing Guide

This guide explains how to test trained 3D bin packing models on specific problems.

## Overview

The testing scripts allow you to:
- Load a trained model and its evolved architecture
- Evaluate the model on specific bin packing problems
- Generate 3D visualizations of packing solutions
- Run batch tests on multiple problems
- Export detailed performance metrics

## Files

- **`test_model.py`**: Test a single problem
- **`test_model_batch.py`**: Test multiple problems in batch mode
- **`TEST_README.md`**: This documentation

## Prerequisites

Make sure you have:
1. A trained model (`.pth` file) - e.g., `results/experiment_multi_problem/trained_model.pth`
2. The evolved genome (`.json` file) - e.g., `results/experiment_multi_problem/evolved_genome.json`
3. Problem files to test - e.g., `nesting/inputData/Thpack/Input/3dBPP_*.txt`

## Usage

### Single Problem Test

Test a model on a single problem:

```bash
python test_model.py \
    --model-path results/experiment_multi_problem/trained_model.pth \
    --genome-path results/experiment_multi_problem/evolved_genome.json \
    --problem-file nesting/inputData/Thpack/Input/3dBPP_1.txt \
    --num-episodes 10 \
    --visualize \
    --save-results
```

**Arguments:**
- `--model-path`: Path to trained model weights (required)
- `--genome-path`: Path to evolved genome configuration (required)
- `--problem-file`: Path to problem file to solve (required)
- `--num-episodes`: Number of evaluation episodes (default: 10)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--visualize`: Generate 3D visualizations of the solution
- `--output-dir`: Directory to save results (default: `test_results`)
- `--device`: Device to use - `cpu` or `cuda` (auto-detected by default)
- `--verbose`: Print detailed step-by-step information
- `--save-results`: Save results to JSON file

**Example Output:**
```
================================================================================
Evaluating on problem: nesting/inputData/Thpack/Input/3dBPP_1.txt
================================================================================

Problem details:
  Bin dimensions: 1000 x 1000 x 1000
  Max bins: 10
  Max weight: None
  Number of items: 12
  Total items to pack: 50
  Incompatibilities: 2
  Positive affinities: 3
  Relative positioning constraints: 1

--- Episode 1/10 ---
  Results:
    Bins used: 3
    Items packed: 50/50
    Utilization: 0.6234
    Total reward: 12.4567
    All items packed: ✓

================================================================================
SUMMARY RESULTS
================================================================================
Average utilization: 0.6150
Average bins used: 3.20
Average items packed: 49.8/50
Success rate: 90.00% (9/10)
================================================================================
```

### Batch Test (Multiple Problems)

Test a model on multiple problems:

```bash
python test_model_batch.py \
    --model-path results/experiment_multi_problem/trained_model.pth \
    --genome-path results/experiment_multi_problem/evolved_genome.json \
    --dataset-dir nesting/inputData/Thpack/Input \
    --problem-ids 1 2 3 4 5 \
    --num-episodes 10 \
    --visualize
```

**Arguments:**
- `--model-path`: Path to trained model weights (required)
- `--genome-path`: Path to evolved genome configuration (required)
- `--dataset-dir`: Directory containing problem files (required)
- `--problem-ids`: List of problem IDs to test, e.g., `1 2 3 4 5` (required)
- `--num-episodes`: Number of evaluation episodes per problem (default: 10)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--visualize`: Generate 3D visualizations (only for first problem)
- `--output-dir`: Directory to save results (default: `test_results_batch`)
- `--device`: Device to use - `cpu` or `cuda` (auto-detected by default)
- `--verbose`: Print detailed information

**Example Output:**
```
================================================================================
BATCH TEST SUMMARY
================================================================================

Problem          Avg Utilization    Avg Bins Used    Avg Items Packed    Success Rate
3dBPP_1          0.6234             3.20             49.8/50             90.00%
3dBPP_2          0.5891             4.10             48.5/50             80.00%
3dBPP_3          0.6456             2.80             50.0/50             100.00%
3dBPP_4          0.6012             3.50             49.2/50             85.00%
3dBPP_5          0.6123             3.30             49.9/50             95.00%

================================================================================
Overall Average Utilization: 0.6143
Overall Average Bins Used: 3.38
Overall Success Rate: 90.00%
================================================================================
```

## Output Files

### Single Problem Test

When running `test_model.py` with `--save-results`, you get:

```
test_results/
├── test_results_3dBPP_1.json          # Detailed results
└── visualizations_3dBPP_1/            # 3D visualizations (if --visualize)
    ├── bin_1_filled.png               # Bin with packed items
    ├── bin_1_with_ems.png             # Bin with empty maximal spaces
    ├── bin_2_filled.png
    └── bin_2_with_ems.png
```

**JSON Structure:**
```json
{
  "problem_path": "nesting/inputData/Thpack/Input/3dBPP_1.txt",
  "problem_name": "3dBPP_1",
  "num_episodes": 10,
  "avg_utilization": 0.6150,
  "avg_bins_used": 3.20,
  "avg_items_packed": 49.8,
  "total_items": 50,
  "success_rate": 0.90,
  "episodes": [
    {
      "episode": 1,
      "bins_used": 3,
      "items_packed": 50,
      "items_left": 0,
      "utilization": 0.6234,
      "total_reward": 12.4567,
      "steps": 50,
      "all_packed": true
    }
  ]
}
```

### Batch Test

When running `test_model_batch.py`, you get:

```
test_results_batch/
├── batch_test_results.json            # Detailed results for all problems
├── batch_test_summary.csv             # Summary table
└── visualizations_3dBPP_1/            # Visualizations (if --visualize)
    └── ...
```

## How It Works

The test scripts follow the same nesting process as the multi-problem experiment:

### 1. Model Loading
```python
# Load genome configuration
genome = Genome.from_json('evolved_genome.json')

# Create DQN agent with evolved architecture
agent = DQNAgentEnhanced(genome.to_dqn_config(...))

# Load trained weights
agent.load_state_dict(torch.load('trained_model.pth'))

# Disable exploration (greedy policy)
agent.epsilon = 0.0
```

### 2. Problem Loading
```python
# Parse problem file
problem = load_problem('3dBPP_1.txt')

# Extract: bin dimensions, constraints, items
# - Incompatibilities: items that can't be in same bin
# - Positive affinities: items that should be together
# - Relative positioning: heavy items can't overlap light items
# - Center of mass: balance constraints
```

### 3. Nesting Process (Episode Loop)
```python
for episode in range(num_episodes):
    obs = env.reset(items=items.copy())

    while not done:
        # 1. Enumerate feasible actions (placement options)
        #    - For each bin, EMS, item, rotation
        #    - Check: fits, constraints, gravity
        actions, mask = env.action_space()

        # 2. Build action features (25-dimensional)
        #    - Bin utilization, space metrics
        #    - Item dimensions, weight
        #    - EMS properties
        action_features = build_action_features(env, actions)

        # 3. Extract heightmap patches (CNN input)
        #    - 7x7 patch around placement position
        #    - Shows local packing density
        patches = extract_patches_for_actions(env, actions, patch_size=7)

        # 4. Agent selects action (neural network)
        #    - State encoder: 8-dim obs -> hidden
        #    - Action encoder: 25-dim features + 64-dim CNN -> hidden
        #    - Attention: actions reason about each other
        #    - Q-values -> argmax action
        action_idx = agent.select_action(obs, action_features, patches, mask)

        # 5. Execute action in environment
        #    - Place item at EMS position
        #    - Apply gravity (drop until hits support)
        #    - Update heightmap, EMS, weight tracking
        #    - Calculate reward
        next_obs, reward, done, info = env.step(actions[action_idx])

        obs = next_obs
```

### 4. Metrics Calculation
- **Utilization**: `packed_volume / (bins_used × bin_volume)`
- **Bins used**: Number of bins with at least one item
- **Items packed**: Total items successfully placed
- **Success rate**: Fraction of episodes where all items packed

## Visualization

When `--visualize` is enabled, the script generates 3D plots for each bin:

- **Filled bins**: Shows all packed items as colored 3D boxes
- **Bins with EMS**: Shows empty maximal spaces (available packing locations)

Colors:
- Different items have different colors
- EMS shown as semi-transparent boxes

## Common Use Cases

### Test on training problems
```bash
python test_model_batch.py \
    --model-path results/experiment_multi_problem/trained_model.pth \
    --genome-path results/experiment_multi_problem/evolved_genome.json \
    --dataset-dir nesting/inputData/Thpack/Input \
    --problem-ids 2 3 5 6 7 8 9 10 11 \
    --num-episodes 10
```

### Test on held-out test problems
```bash
python test_model_batch.py \
    --model-path results/experiment_multi_problem/trained_model.pth \
    --genome-path results/experiment_multi_problem/evolved_genome.json \
    --dataset-dir nesting/inputData/Thpack/Input \
    --problem-ids 1 4 12 \
    --num-episodes 10 \
    --visualize
```

### Quick single problem test with visualization
```bash
python test_model.py \
    --model-path results/experiment_multi_problem/trained_model.pth \
    --genome-path results/experiment_multi_problem/evolved_genome.json \
    --problem-file nesting/inputData/Thpack/Input/3dBPP_1.txt \
    --num-episodes 5 \
    --visualize \
    --verbose
```

### CPU-only testing (no GPU)
```bash
python test_model.py \
    --model-path results/experiment_multi_problem/trained_model.pth \
    --genome-path results/experiment_multi_problem/evolved_genome.json \
    --problem-file nesting/inputData/Thpack/Input/3dBPP_1.txt \
    --device cpu \
    --num-episodes 10
```

## Performance Tips

1. **GPU acceleration**: Use `--device cuda` if available (much faster)
2. **Reduce episodes**: Use fewer episodes (e.g., 5) for quick tests
3. **Disable visualization**: Skip `--visualize` for faster batch tests
4. **Quiet mode**: Don't use `--verbose` for cleaner output

## Troubleshooting

### "Model file not found"
- Check the path to your trained model
- Make sure you ran the training experiment first
- Default path: `results/experiment_multi_problem/trained_model.pth`

### "Genome file not found"
- Check the path to your evolved genome
- Default path: `results/experiment_multi_problem/evolved_genome.json`

### "Problem file not found"
- Verify the problem file exists
- Check the dataset directory path
- Problem files should be named like `3dBPP_1.txt`

### "CUDA out of memory"
- Use `--device cpu` instead
- Or test fewer episodes at once

### "No module named 'genome'"
- Run from the `main/` directory
- Or add `main/` to your PYTHONPATH

## Integration with Experiments

These test scripts are designed to work with models trained using:
- `experiment_multi_problem.py`: Multi-problem neural architecture search
- Any custom training that produces compatible `.pth` and genome `.json` files

The models use the same nesting algorithm, constraints, and environment as during training, ensuring fair and accurate evaluation.
