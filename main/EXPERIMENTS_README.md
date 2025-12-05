# Experimental Framework for Neural Architecture Evolution

This directory contains two experimental setups for neural architecture evolution using genetic algorithms on 3D bin packing problems.

## Experiments

### Experiment 1: Problem-Specific Architecture Evolution

**File:** `experiment_problem_specific.py`

**Description:** Evolves specialized neural architectures for each individual problem instance. Tests the hypothesis that different problem characteristics (constraints, sizes) benefit from different architectures.

**Usage:**
```bash
python experiment_problem_specific.py \
    --dataset-dir "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input" \
    --results-dir results/experiment_problem_specific \
    --population-size 20 \
    --generations 15 \
    --episodes-per-eval 20 \
    --training-episodes 300
```

**Key Features:**
- Evolves architecture for each problem independently
- Full training with best discovered architecture
- Automatic checkpoint/resume support
- Curriculum learning (progressive difficulty)
- Comprehensive results tracking

**Output Structure:**
```
results/experiment_problem_specific/
├── experiment_state.json          # Resume state
├── experiment_summary.json        # Final summary
├── 3dBPP_1/
│   ├── evolution/                 # GA evolution data
│   │   ├── checkpoint_latest.json # Resume checkpoint
│   │   ├── generation_001.json
│   │   └── ...
│   ├── evolved_genome.json        # Best architecture
│   ├── trained_model.pth          # Trained weights
│   └── results.json               # Problem results
├── 3dBPP_2/
└── ...
```

### Experiment 2: Leave-One-Out Generalization Test

**File:** `experiment_leave_one_out.py`

**Description:** Tests architecture generalization by evolving on N-1 problems and evaluating on a held-out test problem. Determines whether a general architecture can work across different instances.

**Usage:**
```bash
python experiment_leave_one_out.py \
    --dataset-dir "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input" \
    --target-problem 3dBPP_12 \
    --population-size 25 \
    --generations 20 \
    --episodes-per-problem 10 \
    --training-episodes 300
```

**To test all problems in leave-one-out fashion:**
```bash
for i in {1..12}; do
    python experiment_leave_one_out.py \
        --dataset-dir "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input" \
        --target-problem 3dBPP_$i \
        --results-dir results/experiment_loo_3dBPP_$i
done
```

**Key Features:**
- Multi-problem fitness aggregation during evolution
- Trains on all problems except target
- Tests generalization to unseen problem
- Checkpoint/resume support
- Curriculum learning

**Output Structure:**
```
results/experiment_leave_one_out_3dBPP_12/
├── evolution/
│   ├── checkpoint_latest.json
│   ├── best_genome.json
│   └── evolution_history.json
├── evolved_genome.json
├── trained_model.pth
└── results.json
```

## Common Parameters

### Evolution Parameters
- `--population-size`: Number of genomes in population (default: 20)
- `--generations`: Number of evolution generations (default: 15)
- `--episodes-per-eval`: Training episodes for fitness evaluation (default: 20 for single-problem, 10 for multi-problem)
- `--elite-size`: Number of top genomes preserved each generation (default: 2)
- `--mutation-rate`: Initial mutation probability (default: 0.2)

### Training Parameters
- `--training-episodes`: Episodes for final model training (default: 300)
- `--no-curriculum`: Disable curriculum learning (use full problems from start)

### System Parameters
- `--seed`: Random seed for reproducibility
- `--no-resume`: Start from scratch, ignoring checkpoints
- `--quiet`: Minimal output (for batch jobs)

## Curriculum Learning

Both experiments support curriculum learning, which speeds up evolution by:
1. **Early generations (0-33%):** Train on 30% of items, fewer episodes
2. **Middle generations (33-66%):** Train on 60% of items, more episodes
3. **Late generations (66-100%):** Train on full problem, full episodes

This allows quick filtering of poor architectures early, with accurate evaluation later.

**Speedup:** Typically 2-3x faster convergence

**Disable:** Use `--no-curriculum` flag

## Checkpoint and Resume

Both experiments automatically save checkpoints after each generation:
- **Checkpoint file:** `{save_dir}/evolution/checkpoint_latest.json`
- **Resume:** Automatically detects and resumes from checkpoint
- **Force restart:** Use `--no-resume` flag

**Manual resume:**
The checkpoint contains full state (population, history, best genome), allowing seamless continuation after interruption.

## Running on Metacentrum

### Setup

1. **Load modules:**
```bash
module load python/3.9.0-gcc
module load cuda/11.7
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Batch Script Example

**File:** `run_experiment1.sh`
```bash
#!/bin/bash
#PBS -N neurogenesis_exp1
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=10gb
#PBS -l walltime=168:00:00
#PBS -m ae

# Load modules
module load python/3.9.0-gcc
module load cuda/11.7

# Activate environment
cd $PBS_O_WORKDIR
source venv/bin/activate

# Run experiment
python experiment_problem_specific.py \
    --dataset-dir "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input" \
    --results-dir results/experiment_problem_specific \
    --population-size 25 \
    --generations 20 \
    --episodes-per-eval 25 \
    --training-episodes 500 \
    --seed 42

# Copy results back
cp -r results/ $PBS_O_WORKDIR/
```

**Submit:**
```bash
qsub run_experiment1.sh
```

### Array Job for Leave-One-Out

**File:** `run_experiment2_array.sh`
```bash
#!/bin/bash
#PBS -N neurogenesis_exp2
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=72:00:00
#PBS -J 1-12
#PBS -m ae

module load python/3.9.0-gcc
module load cuda/11.7

cd $PBS_O_WORKDIR
source venv/bin/activate

# Run for specific problem based on array index
python experiment_leave_one_out.py \
    --dataset-dir "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input" \
    --target-problem 3dBPP_${PBS_ARRAY_INDEX} \
    --population-size 25 \
    --generations 20 \
    --episodes-per-problem 12 \
    --training-episodes 400 \
    --seed ${PBS_ARRAY_INDEX}

cp -r results/experiment_leave_one_out_3dBPP_${PBS_ARRAY_INDEX}/ $PBS_O_WORKDIR/results/
```

**Submit:**
```bash
qsub run_experiment2_array.sh
```

This submits 12 parallel jobs, one for each target problem.

## Results Analysis

### Experiment 1 Summary

After completion, `experiment_summary.json` contains:
- Per-problem results
- Average utilization across all problems
- Average architecture complexity
- Runtime statistics

### Experiment 2 Comparison

Compare leave-one-out results with problem-specific:
```python
import json

# Load LOO result
with open('results/experiment_loo_3dBPP_12/results.json') as f:
    loo_results = json.load(f)

# Load problem-specific result
with open('results/experiment_problem_specific/3dBPP_12/results.json') as f:
    ps_results = json.load(f)

print(f"LOO utilization: {loo_results['target_evaluation']['best_utilization']:.3f}")
print(f"Problem-specific utilization: {ps_results['training']['best_utilization']:.3f}")
```

## Troubleshooting

### Out of Memory
- Reduce `--population-size`
- Reduce `--episodes-per-eval`
- Reduce `--training-episodes`
- Check CUDA memory: `nvidia-smi`

### Slow Convergence
- Enable curriculum learning (default)
- Increase `--population-size` for better exploration
- Increase `--generations`

### Resume Not Working
- Check `checkpoint_latest.json` exists
- Verify JSON is not corrupted
- Use `--no-resume` to force restart

## Implementation Details

### Architecture Search Space

The genome encodes:
- **Network topology:** hidden_dim (64-1024), enc_layers (1-8), head_hidden (64-1024)
- **Attention mechanism:** standard/set_transformer/none, heads (2-16), inducing points (8-128)
- **CNN features:** patch_size (3-15), channels ([8,16] to [32,64])
- **Regularization:** dropout (0.0-0.2), activation (relu/gelu/silu)

### Fitness Function

Multi-objective weighted sum:
```
fitness = 0.70 * utilization + 0.20 * (1 - bins_penalty) + 0.10 * (1 - complexity_penalty)
```

Balances performance, efficiency, and parsimony.

### Multi-Problem Fitness (Experiment 2)

For leave-one-out, fitness is averaged across all training problems:
```
fitness = mean([fitness_problem_1, fitness_problem_2, ..., fitness_problem_N-1])
```

## Citation

If using this framework, please cite:
- Ding et al. (2010): "Using Genetic Algorithms to Optimize Artificial Neural Networks"
- Your thesis work on neurogenesis for 3D bin packing
