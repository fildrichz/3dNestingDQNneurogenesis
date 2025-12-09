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
module load python/3.11.4-gcc-10.2.1-mjh74tn

# Optional: If you created a virtual environment, activate it
# Uncomment and adjust path if needed:
# source /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis/venv/bin/activate

# Change to scratch directory for faster I/O
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

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import numpy; import torch; print(f'NumPy: {numpy.__version__}'); print(f'PyTorch: {torch.__version__}')"

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
RESULTS_DIR="/storage/brno2/home/$PBS_O_LOGNAME/results/leave_one_out"
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
