#!/bin/bash
#PBS -N dqn_gpu_experiment
#PBS -l select=1:ncpus=4:mem=128gb:ngpus=1:scratch_local=40gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -m ae
#PBS -M spidlfil@fit.cvut.cz

# ==============================================================================
# MetaCentrum GPU Job Script
# ==============================================================================
# This script runs experiments with GPU acceleration
# Requires: PyTorch with CUDA support
# ==============================================================================

# Configuration
TARGET_PROBLEM="3dBPP_6"
POPULATION_SIZE=100
GENERATIONS=100
EPISODES_PER_PROBLEM=200
TRAINING_EPISODES=2000
ELITE_SIZE=10
MUTATION_RATE=0.2
SEED=42

echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# ------------------------------------------------------------------------------
# 1) Load mambaforge and activate persistent environment
# ------------------------------------------------------------------------------

echo "Loading mambaforge module..."
module purge
module add mambaforge || { echo "ERROR: Failed to load mambaforge module"; exit 1; }

# Initialize mamba shell integration and activate your /storage env
eval "$(mamba shell hook --shell bash)"
mamba activate /storage/praha1/home/$PBS_O_LOGNAME/dp_env || {
    echo "ERROR: Failed to activate /storage/praha1/home/$PBS_O_LOGNAME/dp_env"
    exit 1
}

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Quick dependency check
echo "Verifying Python dependencies (torch, numpy)..."
python << 'EOF'
import sys
import numpy as np
import torch

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
EOF

# IMPORTANT:
# Do NOT override CUDA_VISIBLE_DEVICES; PBS / scheduler already sets GPU visibility.
# export CUDA_VISIBLE_DEVICES=0  # <- leave this commented out

# ------------------------------------------------------------------------------
# 2) Work in scratch and copy project
# ------------------------------------------------------------------------------

echo "Setting up scratch directory..."
cd "$SCRATCHDIR" || { echo "ERROR: Failed to access SCRATCHDIR"; exit 1; }

echo "Copying project files to scratch..."
cp -r /storage/praha1/home/$PBS_O_LOGNAME/dp-filip-spidla-spidlfil . \
  || { echo "ERROR: Failed to copy project to scratch"; exit 1; }

cd dp-filip-spidla-spidlfil/main || { echo "ERROR: Project main directory missing"; exit 1; }

echo "Working directory: $(pwd)"

echo "=========================================="
echo "Target problem: $TARGET_PROBLEM"
echo "Experiment: leave_one_out (GPU)"
echo "Generations: $GENERATIONS"
echo "Population size: $POPULATION_SIZE"
echo "Episodes per problem: $EPISODES_PER_PROBLEM"
echo "Training episodes: $TRAINING_EPISODES"
echo "=========================================="

# ------------------------------------------------------------------------------
# 3) Run experiment
# ------------------------------------------------------------------------------

# Monitor GPU usage in background
nvidia-smi &
NVIDIA_SMI_PID=$!

echo "Starting leave-one-out experiment..."
python experiment_leave_one_out.py \
    --target-problem "$TARGET_PROBLEM" \
    --population-size "$POPULATION_SIZE" \
    --generations "$GENERATIONS" \
    --episodes-per-problem "$EPISODES_PER_PROBLEM" \
    --training-episodes "$TRAINING_EPISODES" \
    --elite-size "$ELITE_SIZE" \
    --mutation-rate "$MUTATION_RATE" \
    --seed "$SEED"

EXIT_CODE=$?

# Stop GPU monitoring
kill $NVIDIA_SMI_PID 2>/dev/null || true

# ------------------------------------------------------------------------------
# 4) Copy results back to /storage
# ------------------------------------------------------------------------------

echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/leave_one_out/${TARGET_PROBLEM}"
mkdir -p "$RESULTS_DIR"

if [ -d "results" ]; then
    cp -r results/* "$RESULTS_DIR/" || echo "WARNING: Failed to copy some results"
    echo "Results copied to: $RESULTS_DIR"
else
    echo "WARNING: No results directory found!"
fi

# Copy any log files if present
cp *.log "$RESULTS_DIR/" 2>/dev/null || true

# Final GPU status
echo "Final GPU status:"
nvidia-smi

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE