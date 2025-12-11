#!/bin/bash
#PBS -N dqn_problem_specific_7
#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1:scratch_local=40gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -m ae
#PBS -M spidlfil@fit.cvut.cz

# ==============================================================================
# MetaCentrum Job: Problem-Specific for test 7
# ==============================================================================

# Configuration
POPULATION_SIZE=30
GENERATIONS=50
EPISODES_PER_EVAL=100
TRAINING_EPISODES=2000
ELITE_SIZE=3
MUTATION_RATE=0.2
SEED=42

echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# ------------------------------------------------------------------------------
# 1) Load mambaforge and point to the environment's Python
# ------------------------------------------------------------------------------

echo "Loading mambaforge module..."
module purge
module add mambaforge || { echo "ERROR: Failed to load mambaforge module"; exit 1; }

# Use Python from your dp_env directly (no 'activate' script needed)
PYTHON="/storage/praha1/home/$PBS_O_LOGNAME/dp_env/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python executable not found at $PYTHON"
    exit 1
fi

echo "Python executable: $PYTHON"
echo "Python version:"
$PYTHON --version

# Quick dependency check
echo "Verifying Python dependencies (torch, numpy, matplotlib)..."
$PYTHON << 'EOF'
import numpy as np
import torch
import matplotlib

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Matplotlib: {matplotlib.__version__}")
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
echo "Experiment: problem_specific (test 7)"
echo "Generations: $GENERATIONS"
echo "Population size: $POPULATION_SIZE"
echo "Episodes per eval: $EPISODES_PER_EVAL"
echo "Training episodes: $TRAINING_EPISODES"
echo "=========================================="

# ------------------------------------------------------------------------------
# 3) Run experiment
# ------------------------------------------------------------------------------

echo "Starting problem-specific experiment..."
$PYTHON -u experiment_problem_specific.py \
    --population-size "$POPULATION_SIZE" \
    --generations "$GENERATIONS" \
    --episodes-per-eval "$EPISODES_PER_EVAL" \
    --training-episodes "$TRAINING_EPISODES" \
    --elite-size "$ELITE_SIZE" \
    --mutation-rate "$MUTATION_RATE" \
    --seed "$SEED"

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# 4) Copy results back to /storage
# ------------------------------------------------------------------------------

echo "Copying results back to home..."
RESULTS_DIR="/storage/praha1/home/$PBS_O_LOGNAME/results/problem_specific"
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
