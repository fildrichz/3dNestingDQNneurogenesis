#!/bin/bash
#PBS -N dqn_problem_specific
#PBS -l select=1:ncpus=8:mem=32gb:scratch_local=20gb
#PBS -l walltime=48:00:00
#PBS -q default
#PBS -m ae
#PBS -M your.email@cvut.cz

# ==============================================================================
# MetaCentrum Job Script: Problem-Specific Experiment
# ==============================================================================
# This script runs the problem-specific architecture evolution experiment
# ==============================================================================

# Configuration
PROBLEM="3dBPP_12"
GENERATIONS=100
POPULATION=30

# Set up environment
echo "Loading modules..."
module load python

# Install Python dependencies to user directory (cached after first run)
echo "Installing Python dependencies..."
python3 -m pip install --user --quiet numpy
python3 -m pip install --user --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118

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

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import numpy; import torch; print(f'NumPy: {numpy.__version__}'); print(f'PyTorch: {torch.__version__}')"

# Run the experiment
echo "Starting experiment..."
python3 experiment_problem_specific.py \
    --problem "$PROBLEM" \
    --generations "$GENERATIONS" \
    --population "$POPULATION" \
    --verbose

EXIT_CODE=$?

# Copy results back to home directory
echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/problem_specific"
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

exit $EXIT_CODE
