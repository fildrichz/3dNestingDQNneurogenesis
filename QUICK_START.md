# MetaCentrum Quick Start Guide

**First time on MetaCentrum? Start here!**

## Step 1: Login
```bash
ssh your_username@skirit.ics.muni.cz
# Enter your CESNET password when prompted
```

## Step 2: Upload Your Project

### Option A: Using Git (Recommended)
```bash
# On MetaCentrum after login:
cd ~
git clone YOUR_REPOSITORY_URL
cd 3dNestingDQNneurogenesis
```

### Option B: Using SCP
```bash
# On your local machine:
scp -r /path/to/3dNestingDQNneurogenesis your_username@skirit.ics.muni.cz:~/
```

## Step 3: Test Your Setup
```bash
# On MetaCentrum:
cd ~/3dNestingDQNneurogenesis

# Submit test job (runs for ~5-10 minutes)
qsub metacentrum_jobs/test_setup.sh

# Check status
qstat -u $USER

# When finished (status shows 'C'), check results
cat test_metacentrum_setup.o*
```

**If test passes (all ✓), continue to Step 4!**

## Step 4: Run Your First Real Experiment

### Edit the job script
```bash
cd ~/3dNestingDQNneurogenesis
nano metacentrum_jobs/run_leave_one_out.sh

# Update these lines:
#PBS -M your.email@cvut.cz  # <- Your email
PROBLEM="3dBPP_12"           # <- Which problem to test
GENERATIONS=50               # <- Reduce for first test
POPULATION=20                # <- Reduce for first test
```

### Submit the job
```bash
qsub metacentrum_jobs/run_leave_one_out.sh
```

### Monitor progress
```bash
# Check if job is running
qstat -u $USER

# Watch the output in real-time
tail -f dqn_leave_one_out.o*
# Press Ctrl+C to stop watching

# Check for errors
cat dqn_leave_one_out.e*
```

## Step 5: Get Your Results
```bash
# On MetaCentrum, check results
ls -lh ~/results/leave_one_out/

# Download to your local machine:
# On your LOCAL machine:
scp -r your_username@skirit.ics.muni.cz:~/results ./
```

## Common Commands Cheat Sheet

### Job Management
```bash
qsub script.sh              # Submit job
qstat -u $USER              # Check your jobs
qstat -f JOBID              # Detailed job info
qdel JOBID                  # Cancel job
qstat -x -u $USER           # Job history
```

### File Transfer
```bash
# Upload to MetaCentrum (from local machine)
scp file.txt user@skirit.ics.muni.cz:~/path/

# Download from MetaCentrum (from local machine)
scp user@skirit.ics.muni.cz:~/results/data.json ./
```

### Monitoring
```bash
tail -f jobname.o*          # Watch stdout in real-time
cat jobname.e*              # Check for errors
quota -s                    # Check disk quota
module list                 # See loaded modules
```

## Troubleshooting

### Job fails immediately
```bash
# Check error log
cat jobname.e*

# Common fixes:
# - Module not found: Update module version in script
# - Import error: Need to install dependencies
# - Permission denied: Run chmod +x script.sh
```

### Job takes too long to start
- Normal! Queue wait times vary (minutes to hours)
- Check position: `qstat -u $USER`
- GPU jobs wait longer than CPU jobs

### Out of disk space
```bash
# Check quota
quota -s

# Clean up old logs
rm *.o* *.e*

# Move results to local machine and delete from MetaCentrum
```

## Available Job Scripts

Located in `metacentrum_jobs/`:

1. **test_setup.sh** - Quick environment test (run this first!)
2. **run_leave_one_out.sh** - Single leave-one-out experiment
3. **run_problem_specific.sh** - Single problem-specific experiment
4. **run_array_experiments.sh** - Multiple experiments in parallel
5. **run_gpu_experiment.sh** - GPU-accelerated experiments

## Next Steps

- **Small test first**: Start with GENERATIONS=20, POPULATION=10
- **Check resource usage**: After first job, see actual usage with `qstat -f JOBID | grep resources_used`
- **Scale up**: Increase parameters for production runs
- **Use arrays**: Run multiple problems at once with `run_array_experiments.sh`
- **Try GPU**: If experiments are slow, try GPU version

## Getting Help

- **Full guide**: See `METACENTRUM_GUIDE.md` for detailed instructions
- **MetaCentrum docs**: https://wiki.metacentrum.cz/
- **Support**: meta@cesnet.cz

---

**Pro Tip**: Keep this file open in a second terminal while working! 📚
