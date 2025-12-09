# MetaCentrum Setup Guide for CVUT-FEL Students

This guide will walk you through running your 3D bin packing DQN experiments on MetaCentrum.

## Prerequisites
- ✓ CESNET account (you have this)
- ✓ SSH client (Terminal on Linux/Mac, PuTTY on Windows)

## Table of Contents
1. [Logging In](#1-logging-in)
2. [Understanding the Environment](#2-understanding-the-environment)
3. [Uploading Your Project](#3-uploading-your-project)
4. [Python Libraries & Virtual Environment](#4-python-libraries--virtual-environment)
5. [Creating Job Scripts](#5-creating-job-scripts)
6. [Submitting and Monitoring Jobs](#6-submitting-and-monitoring-jobs)
7. [Retrieving Results](#7-retrieving-results)
8. [Tips & Best Practices](#8-tips--best-practices)

---

## 1. Logging In

### First Login
```bash
# Replace 'username' with your CESNET username
ssh username@skirit.ics.muni.cz
```

When prompted, enter your CESNET password.

### Setting Up SSH Keys (Recommended)
To avoid typing password every time:

```bash
# On your local machine:
ssh-keygen -t rsa -b 4096
ssh-copy-id username@skirit.ics.muni.cz
```

### Frontend Servers
MetaCentrum has multiple frontend servers. Common ones:
- `skirit.ics.muni.cz` (recommended for first login)
- `nympha.zcu.cz`
- `tarkil.grid.cesnet.cz`

---

## 2. Understanding the Environment

### Home Directory
- Your home: `/storage/brno2/home/username`
- Quota: Usually 100GB
- **Backed up** - good for code and small data

### Scratch Storage
```bash
# After login, check available scratch spaces:
echo $SCRATCHDIR

# Common scratch locations:
# /scratch/username - temporary, faster, larger quota (500GB-2TB)
# NOT backed up - for temporary computation data
```

### Module System
MetaCentrum uses modules for software:

```bash
# See available modules
module avail

# Search for Python
module avail python

# Search for PyTorch
module avail pytorch

# Load a module
module load python/3.11.4-gcc-10.2.1-mjh74tn

# See what's loaded
module list

# Unload all
module purge
```

---

## 3. Uploading Your Project

### Option A: Using SCP (Simple Copy)
From your local machine:

```bash
# Upload entire project directory
scp -r /path/to/3dNestingDQNneurogenesis username@skirit.ics.muni.cz:~/

# Upload specific files
scp experiment_*.py username@skirit.ics.muni.cz:~/3dNestingDQNneurogenesis/main/
```

### Option B: Using rsync (Recommended for updates)
```bash
# Initial upload
rsync -avz --progress /path/to/3dNestingDQNneurogenesis/ \
    username@skirit.ics.muni.cz:~/3dNestingDQNneurogenesis/

# Later updates (only syncs changed files)
rsync -avz --progress /path/to/3dNestingDQNneurogenesis/ \
    username@skirit.ics.muni.cz:~/3dNestingDQNneurogenesis/
```

### Option C: Using Git (Best Practice)
```bash
# On MetaCentrum (after SSH login):
cd ~
git clone https://github.com/yourusername/3dNestingDQNneurogenesis.git

# Later updates:
cd ~/3dNestingDQNneurogenesis
git pull
```

---

## 4. Python Libraries & Virtual Environment

### Check Available PyTorch Modules
```bash
module avail pytorch
# You'll likely see: pytorch/2.0.1, pytorch/1.13.1, etc.
```

### Option A: Use Pre-installed Modules (Easiest)
```bash
# Load Python and PyTorch
module load python/3.11.4-gcc-10.2.1-mjh74tn
module load pytorch/2.0.1-gpu

# Check if numpy is available
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import torch; print(torch.__version__)"
```

### Option B: Create Virtual Environment (More Control)
```bash
# Load base Python
module load python/3.11.4-gcc-10.2.1-mjh74tn

# Create virtual environment in your home
cd ~/3dNestingDQNneurogenesis
python3 -m venv venv

# Activate it
source venv/bin/activate

# Create requirements.txt if you don't have one
cat > requirements.txt << EOF
numpy>=1.24.0
torch>=2.0.0
EOF

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For GPU support, you may need:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Important**: Save your module commands - you'll need them in job scripts!

---

## 5. Creating Job Scripts

MetaCentrum uses PBS (Portable Batch System) for job scheduling.

### Basic Job Script Template

Create `run_experiment.sh`:

```bash
#!/bin/bash
#PBS -N dqn_nesting_experiment
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=10gb
#PBS -l walltime=24:00:00
#PBS -q default
#PBS -m ae
# PBS -M your.email@cvut.cz

# Set up environment
module load python/3.11.4-gcc-10.2.1-mjh74tn

# If using virtual environment:
source /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis/venv/bin/activate

# Change to scratch directory (faster I/O)
cd $SCRATCHDIR || exit 1

# Copy your code to scratch
cp -r /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis .
cd 3dNestingDQNneurogenesis/main

# Run your experiment
python3 experiment_leave_one_out.py --problem 3dBPP_12 --generations 50 --population 20

# Copy results back to home
mkdir -p /storage/brno2/home/$PBS_O_LOGNAME/results
cp -r results/* /storage/brno2/home/$PBS_O_LOGNAME/results/ || export CLEAN_SCRATCH=false

# Cleanup is automatic unless CLEAN_SCRATCH=false
```

### GPU Job Script

For GPU-accelerated training:

```bash
#!/bin/bash
#PBS -N dqn_nesting_gpu
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:scratch_local=20gb
#PBS -l walltime=48:00:00
#PBS -q gpu

# Load modules for GPU
module load python/3.11.4-gcc-10.2.1-mjh74tn
module load cuda/11.8.0

# Activate virtual environment if needed
source /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis/venv/bin/activate

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

cd $SCRATCHDIR || exit 1
cp -r /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis .
cd 3dNestingDQNneurogenesis/main

# Run experiment
python3 experiment_leave_one_out.py --problem 3dBPP_12 --generations 100 --device cuda

# Copy results back
mkdir -p /storage/brno2/home/$PBS_O_LOGNAME/results
cp -r results/* /storage/brno2/home/$PBS_O_LOGNAME/results/ || export CLEAN_SCRATCH=false
```

### Job Script Parameters Explained

- `#PBS -N`: Job name (shown in queue)
- `#PBS -l select=1`: Number of nodes (usually 1)
  - `ncpus=4`: Number of CPU cores
  - `mem=16gb`: RAM memory
  - `ngpus=1`: Number of GPUs (for GPU jobs)
  - `scratch_local=10gb`: Local scratch space
- `#PBS -l walltime=24:00:00`: Max runtime (HH:MM:SS)
- `#PBS -q`: Queue name (`default`, `gpu`, `long`, etc.)
- `#PBS -m ae`: Email notifications (a=abort, e=end)
- `#PBS -M`: Your email address

---

## 6. Submitting and Monitoring Jobs

### Submit a Job
```bash
cd ~/3dNestingDQNneurogenesis
qsub run_experiment.sh

# Output: Job ID like "12345678.meta-pbs.metacentrum.cz"
```

### Check Job Status
```bash
# Your jobs
qstat -u $USER

# Detailed info about specific job
qstat -f 12345678

# Monitor in real-time
watch -n 10 'qstat -u $USER'
```

Status codes:
- `Q`: Queued (waiting)
- `R`: Running
- `E`: Exiting
- `C`: Completed

### Check Job Output
```bash
# Standard output and error files are created in submission directory:
# dqn_nesting_experiment.o12345678 (stdout)
# dqn_nesting_experiment.e12345678 (stderr)

# View while running
tail -f dqn_nesting_experiment.o12345678

# Check for errors
cat dqn_nesting_experiment.e12345678
```

### Cancel a Job
```bash
qdel 12345678
```

### Interactive Job (for Testing)
```bash
# Request interactive session
qsub -I -l select=1:ncpus=2:mem=8gb -l walltime=2:00:00

# When allocated, you get shell on compute node
# Test your code here
cd ~/3dNestingDQNneurogenesis/main
python3 experiment_leave_one_out.py --test-mode

# Exit when done
exit
```

---

## 7. Retrieving Results

### Download Results to Local Machine
```bash
# On your local machine:

# Download entire results directory
scp -r username@skirit.ics.muni.cz:~/results ./

# Download specific experiment
scp -r username@skirit.ics.muni.cz:~/results/leave_one_out_3dBPP_12_* ./

# Using rsync (better for large data)
rsync -avz --progress username@skirit.ics.muni.cz:~/results/ ./results/
```

### View Results on MetaCentrum
```bash
# SSH into MetaCentrum
ssh username@skirit.ics.muni.cz

# Navigate to results
cd ~/results

# View JSON results
cat results.json | python3 -m json.tool | less

# Search for specific metrics
grep "fitness" results.json
```

---

## 8. Tips & Best Practices

### Resource Estimation
Start conservative and adjust:
```bash
# Test run first with small resources
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=2:mem=8gb

# Check actual usage after job completes
qstat -f JOBID | grep resources_used

# Then scale up for production runs
```

### Monitoring Resource Usage
```bash
# While job is running, SSH to compute node
# First, find the node:
qstat -n JOBID

# SSH to that node (e.g., node123)
ssh node123

# Check processes
top
htop  # if available

# Check GPU usage (if using GPU)
nvidia-smi
watch -n 1 nvidia-smi
```

### Array Jobs (Run Multiple Experiments)
For running multiple problem instances:

```bash
#!/bin/bash
#PBS -N dqn_array
#PBS -J 1-5
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=24:00:00

# Define problem list
PROBLEMS=("3dBPP_12" "3dBPP_15" "3dBPP_18" "3dBPP_20" "3dBPP_25")

# Get problem for this job array element
PROBLEM=${PROBLEMS[$PBS_ARRAY_INDEX-1]}

# Rest of script...
python3 experiment_leave_one_out.py --problem $PROBLEM --generations 100
```

### Checkpoint & Resume
Modify your Python scripts to save checkpoints:

```python
# In your experiment script
checkpoint_file = f"checkpoint_{problem_name}_{generation}.pkl"

# Save periodically
if generation % 10 == 0:
    save_checkpoint(checkpoint_file, {
        'generation': generation,
        'population': population,
        'best_genome': best_genome
    })

# Resume capability
if args.resume and os.path.exists(args.checkpoint):
    checkpoint = load_checkpoint(args.checkpoint)
    start_generation = checkpoint['generation']
```

### Debugging Failed Jobs
```bash
# Check error log
cat dqn_nesting_experiment.e12345678

# Common issues:
# 1. Module not loaded → Add module load to script
# 2. File not found → Check paths (absolute paths safer)
# 3. Out of memory → Increase mem parameter
# 4. Timeout → Increase walltime
# 5. Permission denied → Check file permissions

# Check job epilogue for system messages
grep -i error dqn_nesting_experiment.e12345678
grep -i warning dqn_nesting_experiment.e12345678
```

### Storage Cleanup
```bash
# Check your quota
quota -s

# Clean up old job logs
find ~/3dNestingDQNneurogenesis -name "*.o*" -mtime +30 -delete
find ~/3dNestingDQNneurogenesis -name "*.e*" -mtime +30 -delete

# Archive old results
tar -czf results_backup_$(date +%Y%m%d).tar.gz ~/results
# Move archive to long-term storage or download locally
```

### Useful Commands Summary
```bash
# Submit job
qsub script.sh

# Check status
qstat -u $USER

# Job details
qstat -f JOBID

# Cancel job
qdel JOBID

# Check quota
quota -s

# Available resources
pbsnodes -a | grep -E "(free|state)"

# Your job history
qstat -x -u $USER

# Module management
module avail       # List available
module load NAME   # Load module
module list        # Show loaded
module purge       # Unload all
```

---

## Quick Start Checklist

- [ ] SSH into MetaCentrum: `ssh username@skirit.ics.muni.cz`
- [ ] Upload project files (git/scp/rsync)
- [ ] Test Python imports interactively
- [ ] Create job script based on template above
- [ ] Submit test job with small resources
- [ ] Monitor job: `qstat -u $USER`
- [ ] Check output logs
- [ ] Adjust resources and submit production jobs
- [ ] Download results when complete

---

## Getting Help

- **MetaCentrum Documentation**: https://wiki.metacentrum.cz/
- **User Support**: meta@cesnet.cz
- **CVUT FEL specific**: Your department may have MetaCentrum contacts
- **Check module versions**: `module avail` on the cluster

Good luck with your experiments! 🚀
