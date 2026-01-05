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

This runs a complete workflow: evolves architecture -> trains model -> compares with baseline

## MetaCentrum Jobs

**Note:** All experiments can be run locally using the Python scripts above. MetaCentrum is used for long-running experiments (full population sizes and generations take hours/days).

### Environment Setup

Before running jobs on MetaCentrum, you need:

1. **Create conda environment** (one-time setup):
   ```bash
   module add mambaforge
   mamba create -p ~/dp_env python=3.10 numpy torch torchvision matplotlib -c pytorch
   ```

2. **Upload project** to your MetaCentrum home:
   ```bash
   scp -r dp-filip-spidla-spidlfil /storage/praha1/home/$USER/
   ```

The job scripts automatically:
- Load the mambaforge module
- Use the pre-configured environment at `~/dp_env/bin/python`
- Copy project to scratch for faster I/O
- Copy results back to storage when done

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

**Note:** This job was used for the generalized training experiment in the thesis experiment section.

### Array Job (Parallel)

Runs multiple problems in parallel (one job per problem).

```bash
qsub metacentrum_jobs/run_array_experiments.sh
```

Processes problems 3dBPP_1 through 3dBPP_12 in parallel using PBS array jobs.

**Note:** This job was used to generate results for the problem-specific experiment section in the thesis.

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

### Generating New Datasets

The dataset comes with a generator script:

```
nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Q4RealBPP-DataGen.py
```

To create new problem instances:
1. Open `Q4RealBPP-DataGen.py`
2. Modify the internal parameters (num_categories, dimensions, weights, constraints)
3. Run the script to generate new dataset files

The generator allows customization of:
- Number of item categories
- Item dimension ranges (width, depth, height, weight)
- Positive affinities and incompatibilities
- Number of items per instance

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
