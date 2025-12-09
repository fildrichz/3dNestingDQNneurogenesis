#!/bin/bash
#PBS -N dqn_leave_one_out
#PBS -l select=1:ncpus=8:mem=32gb:scratch_local=20gb
#PBS -l walltime=48:00:00
#PBS -q default
#PBS -m ae
#PBS -M your.email@cvut.cz

# ==============================================================================
# MetaCentrum Job Script: Leave-One-Out Experiment
# ==============================================================================
# This script runs the leave-one-out generalization experiment
# Modify the PROBLEM variable below to change which problem to test
# ==============================================================================

# Configuration
PROBLEM="3dBPP_12"
GENERATIONS=100
POPULATION=30

# Set up environment
echo "Loading modules..."
module load python || { echo "ERROR: Failed to load python module"; exit 1; }

# Install Python dependencies to user directory (cached after first run)
echo "Installing Python dependencies..."
python3 -m pip install --user --quiet numpy || { echo "ERROR: Failed to install numpy"; exit 1; }
python3 -m pip install --user --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118 || { echo "ERROR: Failed to install PyTorch"; exit 1; }

# Verify dependencies are working
echo "Verifying dependencies..."
python3 << 'EOF' || { echo "ERROR: Dependency check failed"; exit 1; }
import sys
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

print("✓ All dependencies OK")
EOF

# Change to scratch directory for faster I/O
echo "Setting up scratch directory..."
cd $SCRATCHDIR || exit 1

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
