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

# Set up environment
echo "Loading modules..."
module load python || { echo "ERROR: Failed to load python module"; exit 1; }

# Change to scratch directory
echo "Setting up scratch directory..."
cd $SCRATCHDIR || { echo "ERROR: Failed to access scratch"; exit 1; }

# Create virtual environment in scratch
echo "Creating virtual environment..."
python3 -m venv venv || { echo "ERROR: Failed to create venv"; exit 1; }
source venv/bin/activate || { echo "ERROR: Failed to activate venv"; exit 1; }

# Set pip cache
export PIP_CACHE_DIR=/storage/praha1/home/$PBS_O_LOGNAME/.pip-cache

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip --quiet || { echo "ERROR: Failed to upgrade pip"; exit 1; }
pip install numpy --quiet || { echo "ERROR: Failed to install numpy"; exit 1; }
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet || { echo "ERROR: Failed to install PyTorch"; exit 1; }

# Verify dependencies
echo "Verifying dependencies..."
python3 << 'EOF' || { echo "ERROR: Dependency check failed"; exit 1; }
import numpy as np
import torch
print(f"✓ NumPy {np.__version__}")
print(f"✓ PyTorch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
EOF

# Copy project to scratch
echo "Copying project files to scratch..."
cp -r /storage/praha1/home/$PBS_O_LOGNAME/dp-filip-spidla-spidlfil .
cd dp-filip-spidla-spidlfil/main

# Display environment info
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Problem: $PROBLEM"
echo "Generations: $GENERATIONS"
echo "Population: $POPULATION"
echo "=========================================="

# Run the experiment
echo "Starting experiment..."
python3 experiment_leave_one_out.py \
    --problem "$PROBLEM" \
    --generations "$GENERATIONS" \
    --population "$POPULATION" \
    --verbose

EXIT_CODE=$?

# Copy results back to home directory
echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/leave_one_out"
mkdir -p "$RESULTS_DIR"

# Copy all result files
if [ -d "results" ]; then
    cp -r results/* "$RESULTS_DIR/" || export CLEAN_SCRATCH=false
    echo "Results copied to: $RESULTS_DIR"
else
    echo "WARNING: No results directory found!"
    export CLEAN_SCRATCH=false
fi

# Also copy logs
cp *.log "$RESULTS_DIR/" 2>/dev/null || true

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Exit with the same code as the Python script
exit $EXIT_CODE
