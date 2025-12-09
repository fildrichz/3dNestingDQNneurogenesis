#!/bin/bash
#PBS -N dqn_array_experiments
#PBS -J 0-4
#PBS -l select=1:ncpus=8:mem=32gb:scratch_local=20gb
#PBS -l walltime=48:00:00
#PBS -q default
#PBS -m ae
#PBS -M your.email@cvut.cz

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
POPULATION=30
EXPERIMENT_TYPE="leave_one_out"  # or "problem_specific"

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

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import numpy; import torch; print(f'NumPy: {numpy.__version__}'); print(f'PyTorch: {torch.__version__}')"

# Select and run experiment
echo "Starting experiment..."
if [ "$EXPERIMENT_TYPE" = "leave_one_out" ]; then
    python3 experiment_leave_one_out.py \
        --problem "$PROBLEM" \
        --generations "$GENERATIONS" \
        --population "$POPULATION" \
        --verbose
else
    python3 experiment_problem_specific.py \
        --problem "$PROBLEM" \
        --generations "$GENERATIONS" \
        --population "$POPULATION" \
        --verbose
fi

EXIT_CODE=$?

# Copy results back to home directory with problem-specific naming
echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/${EXPERIMENT_TYPE}/${PROBLEM}"
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
