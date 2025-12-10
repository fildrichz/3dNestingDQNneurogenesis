#!/bin/bash
#PBS -N dqn_leave_one_out_cpu
#PBS -l select=1:ncpus=2:mem=32gb:scratch_local=20gb
#PBS -l walltime=4:00:00
#PBS -q default
#PBS -m ae
#PBS -M spidlfil@fit.cvut.cz

# ==============================================================================
# MetaCentrum Job Script: Leave-One-Out Experiment
# ==============================================================================
# This script runs the leave-one-out generalization experiment
# Modify the PROBLEM variable below to change which problem to test
# ==============================================================================

# Configuration (FOR TESTING ONLY - use GPU script for real experiments)
PROBLEM="3dBPP_12"
GENERATIONS=10
POPULATION=5

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
echo "Experiment: leave_one_out (CPU - testing)"
echo "Problem: $PROBLEM"
echo "Generations: $GENERATIONS"
echo "Population: $POPULATION"
echo "=========================================="

# ------------------------------------------------------------------------------
# 3) Run experiment
# ------------------------------------------------------------------------------

echo "Starting experiment..."
python experiment_leave_one_out.py \
    --problem "$PROBLEM" \
    --generations "$GENERATIONS" \
    --population "$POPULATION" \
    --verbose

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# 4) Copy results back to /storage
# ------------------------------------------------------------------------------

echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/leave_one_out"
mkdir -p "$RESULTS_DIR"

if [ -d "results" ]; then
    cp -r results/* "$RESULTS_DIR/" || echo "WARNING: Failed to copy some results"
    echo "Results copied to: $RESULTS_DIR"
else
    echo "WARNING: No results directory found!"
fi

# Copy any log files if present
cp *.log "$RESULTS_DIR/" 2>/dev/null || true

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
