# MetaCentrum Job Scripts

Ready-to-use GPU job scripts for running experiments on MetaCentrum.

## Available Scripts

### 1. `run_gpu_experiment.sh` - Single GPU Experiment
Runs a single experiment with GPU acceleration.

**Configuration:**
```bash
PROBLEM="3dBPP_12"           # Which problem to run
GENERATIONS=100              # Evolution generations (matches Python default)
POPULATION=100               # Population size (matches Python default)
EXPERIMENT_TYPE="leave_one_out"  # or "problem_specific"
```

**Resources:**
- 4 CPUs, 128GB RAM, 40GB scratch, 1 GPU
- 24 hours walltime
- Queue: `gpu`

**Usage:**
```bash
# Edit configuration
nano run_gpu_experiment.sh

# Submit
qsub run_gpu_experiment.sh
```

---

### 2. `run_array_experiments.sh` - GPU Array Job (5 parallel experiments)
Runs multiple experiments in parallel, each on its own GPU.

**Configuration:**
```bash
PROBLEMS=("3dBPP_12" "3dBPP_15" "3dBPP_18" "3dBPP_20" "3dBPP_25")
GENERATIONS=100              # Matches Python default
POPULATION=100               # Matches Python default
EXPERIMENT_TYPE="leave_one_out"  # or "problem_specific"
```

**Resources:**
- Each job: 4 CPUs, 128GB RAM, 40GB scratch, 1 GPU
- 24 hours walltime
- Queue: `gpu`
- Submits 5 jobs (indices 0-4)

**Usage:**
```bash
# Edit configuration
nano run_array_experiments.sh

# Submit all 5 jobs
qsub run_array_experiments.sh
```

---

### 3. `run_problem_specific_all.sh` - All Datasets (12 parallel jobs)
Runs problem_specific experiment on ALL 12 datasets simultaneously.

**Configuration:**
```bash
# Automatically runs on 3dBPP_1 through 3dBPP_12
GENERATIONS=100
POPULATION=100
```

**Resources:**
- Each job: 4 CPUs, 128GB RAM, 40GB scratch, 1 GPU
- 24 hours walltime
- Queue: `gpu`
- Submits 12 jobs (indices 1-12)

**Usage:**
```bash
# Just submit - no configuration needed!
qsub run_problem_specific_all.sh

# Results saved to:
# ~/results/problem_specific_gpu/3dBPP_1/
# ~/results/problem_specific_gpu/3dBPP_2/
# ...
# ~/results/problem_specific_gpu/3dBPP_12/
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
├── leave_one_out_gpu/
│   └── 3dBPP_12/
│       ├── results files...
│       └── *.log
└── problem_specific_gpu/
    └── 3dBPP_12/
        └── ...
```

---

## First Time Setup

### 1. Update Your Email
```bash
cd ~/dp-filip-spidla-spidlfil/metacentrum_jobs
nano run_gpu_experiment.sh

# Change this line:
#PBS -M your.email@cvut.cz
```

### 2. Run Your First Experiment
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
Increase walltime (max 168 hours):
```bash
#PBS -l walltime=168:00:00
```

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
