# Running Experiments

Quick guide for running the 3D bin packing experiments.

## Local Experiments

### Problem-Specific Training

Trains a specialized neural architecture for one specific problem using genetic algorithm (GA).

```bash
cd main
python experiment_problem_specific.py --problem-id 7 --population-size 30 --generations 50
```

Key options:
- `--problem-id` - which problem to train on (e.g., 7 loads 3dBPP_7.txt)
- `--population-size` - GA population size (default: 30)
- `--generations` - GA generations (default: 50)
- `--episodes-per-eval` - episodes for fitness evaluation (default: 100)
- `--training-episodes` - final training episodes (default: 2000)
- `--no-curriculum` - disable curriculum learning
- `--no-resume` - start fresh (ignore checkpoints)

Results saved to `results/experiment_problem_specific/`

### Multi-Problem Training

Evolves one architecture that generalizes across multiple problems.

```bash
python experiment_multi_problem.py \
  --training-problem-ids 2 3 5 6 7 \
  --test-problem-ids 1 4 12 \
  --population-size 30 \
  --generations 50
```

Results saved to `results/experiment_multi_problem/`

### GA Training Example

Simple example demonstrating GA-based architecture evolution.

```bash
python example_ga_training.py
```

This runs a complete workflow: evolves architecture → trains model → compares with baseline

## MetaCentrum Jobs

### Single Problem (GPU)

```bash
qsub metacentrum_jobs/run_problem_specific.sh
```

Edit the script to change problem ID or GA parameters.

### Multi-Problem (GPU)

```bash
qsub metacentrum_jobs/run_multi_problem_experiment.sh
```

Trains on problems 2,3,5,6,7,8,9,10,11 and tests on 1,4,12 by default.

### Array Job (Parallel)

Runs multiple problems in parallel (one job per problem).

```bash
qsub metacentrum_jobs/run_array_experiments.sh
```

Processes problems 3dBPP_1 through 3dBPP_12 in parallel using PBS array jobs.

## Dataset

### Loading Problems

```python
from nesting.dataset_loader import load_problem

# Load a problem file
problem = load_problem("nesting/inputData/.../3dBPP_7.txt")

print(f"Bin dimensions: {problem.bin_dimensions}")
print(f"Items: {len(problem.items)}")
print(f"Max bins: {problem.max_bins}")
```

### Dataset Structure

Problems are stored in:
```
nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input/
```

Files: `3dBPP_1.txt`, `3dBPP_2.txt`, ..., `3dBPP_12.txt`

Each file contains:
- Bin dimensions and constraints
- Item specifications (id, quantity, dimensions, weight)
- Packing constraints (incompatibilities, affinities, etc.)

## Quick Start

Want to just run something?

```bash
# Local quick test (small GA)
python experiment_problem_specific.py --problem-id 7 --population-size 2 --generations 2

# MetaCentrum full run
qsub metacentrum_jobs/run_problem_specific.sh
```

## Tips

- Use `--no-resume` to start fresh experiments
- Checkpoints are automatically saved for resume capability
- Results include trained models, evolution history, and visualizations
- GPU is auto-detected for local runs; MetaCentrum jobs request GPU nodes
