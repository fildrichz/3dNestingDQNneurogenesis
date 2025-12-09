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
module load python/3.11.4-gcc-10.2.1-mjh74tn
module load cuda/11.8.0

# Optional: Activate virtual environment if created
# Make sure PyTorch in venv is built with CUDA support!
# source /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis/venv/bin/activate

# Set CUDA device (usually only one GPU allocated)
export CUDA_VISIBLE_DEVICES=0

# Change to scratch directory
echo "Setting up scratch directory..."
cd $SCRATCHDIR || exit 1

# Copy project to scratch
echo "Copying project files to scratch..."
cp -r /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis .
cd 3dNestingDQNneurogenesis/main

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

# Check CUDA availability
echo "Checking CUDA setup..."
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available! Running on CPU.")
EOF

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
RESULTS_DIR="/storage/brno2/home/$PBS_O_LOGNAME/results/${EXPERIMENT_TYPE}_gpu/${PROBLEM}"
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
