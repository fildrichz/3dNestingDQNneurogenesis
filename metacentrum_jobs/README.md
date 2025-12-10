# MetaCentrum Job Scripts

Ready-to-use GPU job scripts for running experiments on MetaCentrum.

## Available Scripts

### 1. `run_gpu_experiment.sh` - Single Leave-One-Out Experiment
Runs a single leave-one-out experiment with GPU acceleration.

**Configuration:**
```bash
TARGET_PROBLEM="3dBPP_12"    # Problem to hold out for testing
POPULATION_SIZE=100          # GA population size
GENERATIONS=100              # GA generations
EPISODES_PER_PROBLEM=200     # Episodes per problem for fitness eval
TRAINING_EPISODES=2000       # Episodes for final training on target
ELITE_SIZE=10                # Number of elite genomes
MUTATION_RATE=0.2            # Initial mutation rate
SEED=42                      # Random seed
```

**Resources:**
- 4 CPUs, 128GB RAM, 40GB scratch, 1 GPU
- 48 hours walltime
- Queue: `gpu`

**Usage:**
```bash
# Edit configuration
nano run_gpu_experiment.sh

# Submit
qsub run_gpu_experiment.sh
```

---

### 2. `run_array_experiments.sh` - Leave-One-Out Array Job (ALL 12 datasets)
Runs leave-one-out experiments in parallel on all 12 datasets, each on its own GPU.

**Configuration:**
```bash
# Automatically runs on 3dBPP_1 through 3dBPP_12
POPULATION_SIZE=100          # GA population size
GENERATIONS=100              # GA generations
EPISODES_PER_PROBLEM=200     # Episodes per problem for fitness eval
TRAINING_EPISODES=2000       # Episodes for final training on target
ELITE_SIZE=10                # Number of elite genomes
MUTATION_RATE=0.2            # Initial mutation rate
SEED=42                      # Random seed
```

**Resources:**
- Each job: 4 CPUs, 128GB RAM, 40GB scratch, 1 GPU
- 48 hours walltime
- Queue: `gpu`
- Submits 12 jobs (indices 1-12)

**Usage:**
```bash
# Edit configuration
nano run_array_experiments.sh

# Submit all 12 jobs
qsub run_array_experiments.sh
```

---

### 3. `run_problem_specific_all.sh` - Problem-Specific for All Datasets (Sequential)
Runs problem_specific experiment on ALL 12 datasets SEQUENTIALLY in a single job.

**How it works:**
- ONE job processes all 12 datasets one by one
- Has built-in checkpoint/resume - tracks completed datasets
- If job times out at 48h, just resubmit → auto-resumes from where it stopped
- Each dataset gets its own evolved architecture and trained model

**Configuration:**
```bash
POPULATION_SIZE=100          # GA population size
GENERATIONS=100              # GA generations
EPISODES_PER_EVAL=200        # Episodes for fitness evaluation
TRAINING_EPISODES=2000       # Episodes for final training
ELITE_SIZE=10                # Number of elite genomes
MUTATION_RATE=0.2            # Initial mutation rate
SEED=42                      # Random seed
```

**Resources:**
- 4 CPUs, 128GB RAM, 40GB scratch, 1 GPU
- 48 hours walltime (resubmit if needed - auto-resumes)
- Queue: `gpu`
- Submits 1 job (not an array)

**Usage:**
```bash
# Just submit - no configuration needed!
qsub run_problem_specific_all.sh

# If it times out, resubmit the same command:
qsub run_problem_specific_all.sh  # Will auto-resume

# Results saved to:
# ~/results/problem_specific/3dBPP_1/
# ~/results/problem_specific/3dBPP_2/
# ...
# ~/results/problem_specific/3dBPP_12/
# ~/results/problem_specific/experiment_state.json  # Checkpoint file
```

---

## What These Scripts Do

### Setup Phase (automatic)
1. Load Python module
2. Create fresh virtual environment in scratch
3. Install numpy and PyTorch 2.x with CUDA 11.8
4. Verify CUDA is available (exits if not)

### Execution Phase
5. Copy your project to scratch (fast local SSD)
6. Run your experiment with GPU acceleration
7. Copy results back to home directory

### Results Location
```
~/results/
├── leave_one_out/
│   ├── 3dBPP_1/
│   ├── 3dBPP_2/
│   ├── ...
│   └── 3dBPP_12/
│       ├── evolved_genome.json
│       ├── trained_model.pth
│       ├── results.json
│       ├── visualizations/
│       └── evolution/
└── problem_specific/
    ├── experiment_state.json      # Checkpoint for resume
    ├── experiment_summary.json    # Overall results
    ├── 3dBPP_1/
    ├── 3dBPP_2/
    ├── ...
    └── 3dBPP_12/
        ├── evolved_genome.json
        ├── trained_model.pth
        ├── results.json
        ├── visualizations/
        └── evolution/
```

---

## First Time Setup

### 1. Run Your First Experiment
```bash
# Submit the job
qsub run_gpu_experiment.sh

# Check status
qstat -u $USER

# Monitor output
tail -f dqn_gpu_experiment.o*
```

---

## Monitoring Jobs

### Check job status
```bash
qstat -u $USER
```

Status codes:
- `Q` - Queued (waiting for GPU)
- `R` - Running
- `C` - Completed

### View output
```bash
# Real-time monitoring
tail -f dqn_gpu_experiment.o*

# Check errors
cat dqn_gpu_experiment.e*

# Job history
qstat -x -u $USER | tail -10
```

### Cancel a job
```bash
qdel JOBID
```

---

## Resource Usage

After your first job completes, check actual usage:
```bash
qstat -f JOBID | grep resources_used
```

This shows:
- `cpupercent` - CPU usage
- `mem` - RAM used
- `walltime` - Time taken

Adjust resources in the script if needed.

---

## Common Issues

### Job exits immediately
```bash
# Check error log
cat dqn_gpu_experiment.e*

# Common causes:
# - CUDA not available: Script will detect and exit
# - Module load failed: Check error message
# - Dependency install failed: Usually network issue, retry
```

### Out of memory
Increase RAM:
```bash
#PBS -l select=1:ncpus=4:mem=192gb:ngpus=1:scratch_local=40gb
```

### Out of time
**GPU queue max: 48 hours**

For `run_problem_specific_all.sh`:
- Just resubmit the same job
- It will automatically resume from checkpoint

For leave-one-out experiments:
- Reduce TRAINING_EPISODES or GENERATIONS
- Or split into smaller jobs

### Results not copied back
```bash
# Check if scratch cleanup was prevented
grep "CLEAN_SCRATCH" dqn_gpu_experiment.o*

# If yes, contact support to retrieve from scratch
```

---

## Tips

1. **Start with single experiment** - Test with `run_gpu_experiment.sh` first
2. **Monitor GPU usage** - Check `nvidia-smi` output in job logs
3. **Use array jobs for efficiency** - Run multiple problems in parallel
4. **Check queue wait times** - GPU queue may have longer waits
5. **Save checkpoints** - Long runs should checkpoint progress

---

## Script Structure

Both scripts follow the same pattern:
```bash
1. Load Python → 2. Create venv → 3. Install deps → 4. Verify CUDA
        ↓
5. Copy project → 6. Run experiment → 7. Copy results back
```

All steps have error checking and exit immediately on failure.

---

## Getting Help

- **MetaCentrum docs**: https://wiki.metacentrum.cz/
- **Support email**: meta@cesnet.cz
- **Main guide**: See `../METACENTRUM_GUIDE.md`

Good luck with your experiments! 🚀
