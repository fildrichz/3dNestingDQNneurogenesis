#!/bin/bash
#PBS -N dqn_gpu_experiment
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:scratch_local=30gb:gpu_cap=cuda80
#PBS -l walltime=72:00:00
#PBS -q gpu
#PBS -m ae
#PBS -M your.email@cvut.cz

# ==============================================================================
# MetaCentrum GPU Job Script
# ==============================================================================
# This script runs experiments with GPU acceleration
# Requires: PyTorch with CUDA support
# ==============================================================================

# Configuration
PROBLEM="3dBPP_12"
GENERATIONS=200
POPULATION=50
EXPERIMENT_TYPE="leave_one_out"  # or "problem_specific"

# Set up environment
echo "Loading modules..."
module load python || { echo "ERROR: Failed to load python module"; exit 1; }

# Try to load CUDA module (may not be available on all GPU nodes)
module load cuda 2>/dev/null || echo "CUDA module not found, using system CUDA"

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
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if not cuda_available:
        print("✗ ERROR: GPU job but CUDA not available!")
        sys.exit(1)
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

print("✓ All dependencies OK")
EOF

# Set CUDA device (usually only one GPU allocated)
export CUDA_VISIBLE_DEVICES=0

# Change to scratch directory
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

# Monitor GPU usage in background
nvidia-smi &
NVIDIA_SMI_PID=$!

# Run the experiment with GPU
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

# Stop GPU monitoring
kill $NVIDIA_SMI_PID 2>/dev/null || true

# Copy results back to home directory
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
