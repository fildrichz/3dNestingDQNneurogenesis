# MetaCentrum Job Scripts

This directory contains ready-to-use PBS job scripts for running experiments on MetaCentrum.

## Available Scripts

### 1. `run_leave_one_out.sh`
Runs a single leave-one-out experiment on one problem.

**Usage:**
```bash
# Edit the script to set PROBLEM, GENERATIONS, POPULATION
nano run_leave_one_out.sh

# Submit
qsub run_leave_one_out.sh
```

**Configuration variables:**
- `PROBLEM`: Which problem to test (e.g., "3dBPP_12")
- `GENERATIONS`: Number of generations for evolution
- `POPULATION`: Population size

### 2. `run_problem_specific.sh`
Runs a single problem-specific experiment.

**Usage:**
```bash
# Edit configuration in script
nano run_problem_specific.sh

# Submit
qsub run_problem_specific.sh
```

### 3. `run_array_experiments.sh`
Runs multiple experiments in parallel using PBS array jobs.

**Usage:**
```bash
# Edit PROBLEMS array in script to select which problems to run
nano run_array_experiments.sh

# Submit all at once
qsub run_array_experiments.sh
```

This will submit one job per problem and run them in parallel (resource permitting).

### 4. `run_gpu_experiment.sh`
Runs GPU-accelerated experiments (requires CUDA-enabled PyTorch).

**Usage:**
```bash
# Edit configuration
nano run_gpu_experiment.sh

# Submit to GPU queue
qsub run_gpu_experiment.sh
```

**Note:** GPU jobs may wait longer in queue but run much faster.

## Before First Use

### 1. Update Email Address
In each script, change:
```bash
#PBS -M your.email@cvut.cz
```
to your actual email address.

### 2. Check Python Module Version
On MetaCentrum, check available Python modules:
```bash
module avail python
```

Update the module load line if needed:
```bash
module load python/X.Y.Z-gcc-W.V.U-hash
```

### 3. Virtual Environment (Optional)
If you created a virtual environment, uncomment the activation line in each script:
```bash
source /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis/venv/bin/activate
```

## Monitoring Jobs

### Check job status
```bash
qstat -u $USER
```

### View output in real-time
```bash
# Find the output file (named like: dqn_leave_one_out.oJOBID)
tail -f dqn_leave_one_out.o12345678
```

### Check for errors
```bash
cat dqn_leave_one_out.e12345678
```

## Results Location

Results are copied to your home directory:
```
~/results/
├── leave_one_out/
│   ├── 3dBPP_12/
│   ├── 3dBPP_15/
│   └── ...
├── problem_specific/
│   └── ...
└── leave_one_out_gpu/
    └── ...
```

## Resource Guidelines

### CPU Jobs
- **Short tests** (< 1 hour):
  - `ncpus=2`, `mem=8gb`, `walltime=1:00:00`

- **Medium runs** (few hours):
  - `ncpus=4`, `mem=16gb`, `walltime=12:00:00`

- **Long experiments** (days):
  - `ncpus=8`, `mem=32gb`, `walltime=48:00:00`

### GPU Jobs
- Always use `gpu` queue
- Specify `ngpus=1` (rarely need more)
- More memory: `mem=64gb` (GPUs work with large batches)
- Longer walltime: GPU queue may allow up to 168 hours

## Common Issues

### Job fails immediately
- Check error log: `cat JOBNAME.eJOBID`
- Common causes:
  - Module not found → Update module name
  - Import error → Missing dependencies
  - Path error → Check file paths

### Job runs out of time
- Increase `walltime` in PBS directive
- Or reduce `GENERATIONS` to fit time limit

### Job runs out of memory
- Increase `mem` parameter
- Or reduce `POPULATION` size

### Results not copied back
- Check: `export CLEAN_SCRATCH=false` prevents deletion
- Results may still be in scratch if job didn't complete
- Contact support to retrieve: meta@cesnet.cz

## Tips

1. **Start small**: Test with small GENERATIONS and POPULATION first
2. **Monitor resources**: After first job, check actual usage with `qstat -f JOBID`
3. **Array jobs**: Efficient for running same experiment on multiple problems
4. **GPU**: Only use if your code explicitly supports CUDA
5. **Checkpoints**: Modify Python scripts to save progress every N generations

## Need Help?

- MetaCentrum docs: https://wiki.metacentrum.cz/
- Support email: meta@cesnet.cz
- Main guide: See `../METACENTRUM_GUIDE.md`
