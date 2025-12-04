# Metacentrum Setup Guide for 3D Nesting DQN Project

## Overview
This guide will help you run your 3D bin packing DQN training on Metacentrum computing infrastructure.

## Prerequisites

### 1. Metacentrum Account Setup
As a CVUT student, you should have access to Metacentrum. Here's how to get started:

1. **Register for Metacentrum account:**
   - Go to https://metavo.metacentrum.cz/
   - Login with your CVUT credentials (via eduID)
   - Fill out the registration form
   - Wait for approval (usually 1-2 days)

2. **Set up SSH access:**
   ```bash
   # Generate SSH key if you don't have one
   ssh-keygen -t rsa -b 4096

   # Add your public key to Metacentrum
   # Go to https://metavo.metacentrum.cz/ -> SSH keys
   # Upload your ~/.ssh/id_rsa.pub
   ```

3. **Test connection:**
   ```bash
   # Connect to Metacentrum frontend
   ssh your_username@skirit.ics.muni.cz
   # or
   ssh your_username@tarkil.grid.cesnet.cz
   ```

### 2. First Time Setup on Metacentrum

Once logged in to a Metacentrum frontend:

```bash
# Navigate to your home directory
cd ~

# Clone your repository
git clone https://github.com/yourusername/3dNestingDQNneurogenesis.git
cd 3dNestingDQNneurogenesis

# Check available modules
module avail
```

## Running Jobs on Metacentrum

### Option 1: Quick DQN Training (24 hours)

This runs a standard DQN training session on a single problem instance.

```bash
# Submit the job
qsub metacentrum_dqn_training.sh

# Check job status
qstat -u $USER

# View job output (after completion)
cat dqn_packing_training.o*  # Standard output
cat dqn_packing_training.e*  # Error output
```

**Resources requested:**
- 1 GPU with CUDA support
- 32 GB RAM
- 4 CPU cores
- 24-hour time limit
- 10 GB scratch storage

### Option 2: GA Evolution (48 hours)

This runs the full genetic algorithm to evolve DQN architectures. **Warning: This is computationally intensive!**

```bash
# Submit the job
qsub metacentrum_ga_evolution.sh

# Monitor job
qstat -u $USER
```

**Resources requested:**
- 1 GPU with CUDA support
- 64 GB RAM
- 8 CPU cores
- 48-hour time limit
- 20 GB scratch storage

### Customizing Training Parameters

#### Modify DQN Training Script

Edit `main/packing_with_dqncore2_enhanced.py` and change the parameters at the bottom:

```python
if __name__ == "__main__":
    train_multibin_pack_dqn(
        problem_file="3dBPP_4.txt",  # Change problem instance
        episodes=1000,                # Number of training episodes
        log_interval=10,              # How often to log
        # ... other parameters
    )
```

#### Modify GA Evolution Script

Edit `main/example_ga_training.py`:

```python
population_size = 20   # Number of architectures per generation
generations = 10       # Number of evolution generations
episodes_per_eval = 50 # Episodes to evaluate each architecture
```

### PBS Script Customization

You can modify the PBS scripts to request different resources:

```bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=10gb
#PBS -l walltime=24:00:00
```

**Common modifications:**
- `ncpus=X` - Number of CPU cores (1-32)
- `ngpus=X` - Number of GPUs (0-4, depending on node)
- `mem=Xgb` - RAM in gigabytes
- `walltime=HH:MM:SS` - Maximum runtime
- `scratch_local=Xgb` - Local scratch storage

**GPU-specific requests:**
```bash
# Request specific GPU type (if needed)
#PBS -l select=1:ncpus=4:ngpus=1:gpu_cap=cuda80:mem=32gb
# cuda80 = Pascal (P100)
# cuda75 = Maxwell (M40, M60)
# cuda61 = Pascal (P40, GTX 1080)
```

## Monitoring and Managing Jobs

### Check Job Status
```bash
# List your jobs
qstat -u $USER

# Detailed job info
qstat -f JOB_ID

# Show job history
qstat -x JOB_ID
```

### Cancel a Job
```bash
qdel JOB_ID
```

### Interactive GPU Session (for debugging)
```bash
# Request interactive session
qsub -I -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=4:00:00

# Once allocated, you can run commands directly
module add mambaforge
module add cuda/12.2
cd 3dNestingDQNneurogenesis/main
python packing_with_dqncore2_enhanced.py
```

## Output Files

After job completion, results will be saved to:
```
output_data/
├── metacentrum_runs/        # DQN training outputs
│   ├── *.png               # Training curves
│   ├── *.txt               # Logs
│   └── packing_results/    # Packing solutions
└── ga_evolution/           # GA evolution outputs
    ├── best_genome.json    # Best architecture found
    ├── generation_*.json   # Evolution history
    └── *.png               # Performance plots
```

## Troubleshooting

### Module Loading Issues
```bash
# List available modules
module avail python
module avail cuda

# If mambaforge is not available, try:
module add conda-modules
module add python/3.11.0
```

### GPU Not Available
```bash
# Check if GPU is accessible
nvidia-smi

# If not working, check CUDA module
module list
module add cuda/12.2
```

### Out of Memory Errors

If you run out of RAM or GPU memory:

1. **Reduce batch size:** Edit `main/packing_with_dqncore2_enhanced.py`
   ```python
   batch_size = 64  # Default is 128
   ```

2. **Reduce buffer size:**
   ```python
   buffer_size = 200_000  # Default is 400,000
   ```

3. **Request more memory in PBS script:**
   ```bash
   #PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
   ```

### Job Time Limit Exceeded

If jobs are killed due to time limits:

1. **Increase walltime:**
   ```bash
   #PBS -l walltime=48:00:00
   ```

2. **Reduce training episodes:**
   ```python
   episodes = 500  # Instead of 1000
   ```

3. **For GA evolution, reduce population or generations:**
   ```python
   population_size = 10  # Instead of 20
   generations = 5       # Instead of 10
   ```

## Best Practices

1. **Start small:** Run a short test job first (100 episodes) to verify everything works
2. **Monitor resources:** Check `qstat -f JOB_ID` to see memory/CPU usage
3. **Save checkpoints:** The code automatically saves models, but check output files are being created
4. **Use array jobs:** For running multiple experiments (see PBS array jobs documentation)
5. **Email notifications:** The `-m ae` flag sends emails when jobs abort or end

## Useful Metacentrum Resources

- **Main portal:** https://metavo.metacentrum.cz/
- **Documentation:** https://wiki.metacentrum.cz/
- **Support:** meta@cesnet.cz
- **PBS guide:** https://wiki.metacentrum.cz/wiki/About_scheduling_system
- **Available resources:** https://metavo.metacentrum.cz/pbsmon2/

## Example Workflow

```bash
# 1. Connect to Metacentrum
ssh your_username@skirit.ics.muni.cz

# 2. Navigate to project
cd ~/3dNestingDQNneurogenesis

# 3. Pull latest changes (if needed)
git pull

# 4. Submit a test job (edit script to use episodes=100 first)
qsub metacentrum_dqn_training.sh

# 5. Monitor the job
watch -n 30 qstat -u $USER

# 6. Once complete, check results
ls -lh output_data/metacentrum_runs/

# 7. If successful, submit longer jobs or GA evolution
qsub metacentrum_ga_evolution.sh
```

## Getting Help

- **Metacentrum support:** Email meta@cesnet.cz with your job ID
- **CVUT computing support:** Check with your faculty's IT department
- **This project:** Check the code documentation in `main/` directory
