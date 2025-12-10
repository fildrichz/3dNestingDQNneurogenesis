#!/bin/bash
#PBS -N dqn_gpu_array
#PBS -J 0-4
#PBS -l select=1:ncpus=4:mem=128gb:ngpus=1:scratch_local=40gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -m ae
#PBS -M spidlfil@fit.cvut.cz

# ==============================================================================
# MetaCentrum Array Job Script: Run Multiple Experiments in Parallel
# ==============================================================================
# This array job runs experiments on multiple problems simultaneously
# Each array element processes one problem
# Modify PROBLEMS array below to change which problems to test
# ==============================================================================

# Define array of problems to test
PROBLEMS=("3dBPP_12" "3dBPP_15" "3dBPP_18" "3dBPP_20" "3dBPP_25")

# Get the problem for this array index
PROBLEM=${PROBLEMS[$PBS_ARRAY_INDEX]}

# Configuration
GENERATIONS=100
POPULATION=100
EXPERIMENT_TYPE="leave_one_out"  # or "problem_specific"

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

# Set pip cache to scratch (avoids home directory quota issues)
export PIP_CACHE_DIR=$SCRATCHDIR/.pip-cache
mkdir -p $PIP_CACHE_DIR

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip --quiet || { echo "ERROR: Failed to upgrade pip"; exit 1; }
pip install numpy --quiet || { echo "ERROR: Failed to install numpy"; exit 1; }
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet || { echo "ERROR: Failed to install PyTorch"; exit 1; }

# Verify dependencies
echo "Verifying dependencies..."
python3 << 'EOF' || { echo "ERROR: Dependency check failed"; exit 1; }
import sys
import numpy as np
import torch

print(f"✓ NumPy {np.__version__}")
print(f"✓ PyTorch {torch.__version__}")

if not torch.cuda.is_available():
    print("✗ ERROR: GPU job but CUDA not available!")
    sys.exit(1)

print(f"✓ CUDA available")
print(f"  Device: {torch.cuda.get_device_name(0)}")
EOF

export CUDA_VISIBLE_DEVICES=0

# Copy project to scratch
echo "Copying project files to scratch..."
cp -r /storage/praha1/home/$PBS_O_LOGNAME/dp-filip-spidla-spidlfil .
cd dp-filip-spidla-spidlfil/main

# Display environment info
echo "=========================================="
echo "Array Job Index: $PBS_ARRAY_INDEX"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Problem: $PROBLEM"
echo "Generations: $GENERATIONS"
echo "Population: $POPULATION"
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "=========================================="

# Select and run experiment
echo "Starting GPU-accelerated experiment..."
if [ "$EXPERIMENT_TYPE" = "leave_one_out" ]; then
    python3 experiment_leave_one_out.py \
        --problem "$PROBLEM" \
        --generations "$GENERATIONS" \
        --population "$POPULATION" \
        --device cuda \
        --verbose
else
    python3 experiment_problem_specific.py \
        --problem "$PROBLEM" \
        --generations "$GENERATIONS" \
        --population "$POPULATION" \
        --device cuda \
        --verbose
fi

EXIT_CODE=$?

# Copy results back to home directory with problem-specific naming
echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/${EXPERIMENT_TYPE}_gpu/${PROBLEM}"
mkdir -p "$RESULTS_DIR"

# Copy all result files
if [ -d "results" ]; then
    cp -r results/* "$RESULTS_DIR/" || export CLEAN_SCRATCH=false
    echo "Results copied to: $RESULTS_DIR"
else
    echo "WARNING: No results directory found!"
    export CLEAN_SCRATCH=false
fi

# Also copy logs with unique names
cp *.log "$RESULTS_DIR/" 2>/dev/null || true

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE