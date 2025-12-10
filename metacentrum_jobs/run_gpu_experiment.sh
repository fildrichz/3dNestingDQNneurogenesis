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
print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"  CUDA version: {torch.version.cuda}")
EOF

export CUDA_VISIBLE_DEVICES=0

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
echo "Target problem: $TARGET_PROBLEM"
echo "Generations: $GENERATIONS"
echo "Population size: $POPULATION_SIZE"
echo "Episodes per problem: $EPISODES_PER_PROBLEM"
echo "Training episodes: $TRAINING_EPISODES"
echo "=========================================="

# Monitor GPU usage in background
nvidia-smi &
NVIDIA_SMI_PID=$!

# Run the experiment with GPU
echo "Starting leave-one-out experiment..."
python3 experiment_leave_one_out.py \
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

# Copy results back to home directory
echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/leave_one_out/${TARGET_PROBLEM}"
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

# Final GPU status
echo "Final GPU status:"
nvidia-smi

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE