#!/bin/bash
#PBS -N dqn_problem_specific
#PBS -l select=1:ncpus=2:mem=128gb:ngpus=1:scratch_local=40gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -m ae
#PBS -M spidlfil@fit.cvut.cz

# ==============================================================================
# MetaCentrum Job: Problem-Specific (argument / array driven)
# ==============================================================================

# ------------------------------------------------------------------------------
# Resolve PROBLEM_ID
# ------------------------------------------------------------------------------
if [ $# -ge 1 ]; then
    PROBLEM_ID="$1"
elif [ -n "${PBS_ARRAY_INDEX:-}" ]; then
    PROBLEM_ID="$PBS_ARRAY_INDEX"
else
    echo "ERROR: No problem ID provided"
    exit 1
fi

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
POPULATION_SIZE=30
GENERATIONS=30
EPISODES_PER_EVAL=100
TRAINING_EPISODES=2000
ELITE_SIZE=2
MUTATION_RATE=0.2
SEED=42

echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Problem ID: $PROBLEM_ID"
echo "=========================================="

# ------------------------------------------------------------------------------
# 1) Load mambaforge and Python (UNCHANGED)
# ------------------------------------------------------------------------------
echo "Loading mambaforge module..."
module purge
module add mambaforge || { echo "ERROR: Failed to load mambaforge module"; exit 1; }

PYTHON="/storage/praha1/home/$PBS_O_LOGNAME/dp_env/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python executable not found at $PYTHON"
    exit 1
fi

echo "Python executable: $PYTHON"
$PYTHON --version

# Dependency check (UNCHANGED)
$PYTHON << 'EOF'
import numpy as np, torch, matplotlib
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Matplotlib: {matplotlib.__version__}")
EOF

# ------------------------------------------------------------------------------
# 2) Scratch setup (UNCHANGED)
# ------------------------------------------------------------------------------
cd "$SCRATCHDIR" || exit 1
cp -r /storage/praha1/home/$PBS_O_LOGNAME/dp-filip-spidla-spidlfil .
cd dp-filip-spidla-spidlfil/main || exit 1

echo "Working directory: $(pwd)"

# ------------------------------------------------------------------------------
# 3) Dataset + results paths (NEW, minimal)
# ------------------------------------------------------------------------------
DATASET_BASE="nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"
DATASET_DIR="${DATASET_BASE}/${PROBLEM_ID}"

RESULTS_LOCAL="results/problem_specific/problem_${PROBLEM_ID}"
RESULTS_STORAGE="/storage/praha1/home/$PBS_O_LOGNAME/results/problem_specific/problem_${PROBLEM_ID}"
mkdir -p "$RESULTS_STORAGE"

echo "Dataset dir: $DATASET_DIR"
echo "Results dir: $RESULTS_LOCAL"

# ------------------------------------------------------------------------------
# 4) Run experiment (ALL FLAGS PRESERVED)
# ------------------------------------------------------------------------------
$PYTHON -u experiment_problem_specific.py \
    --dataset-dir "$DATASET_DIR" \
    --results-dir "$RESULTS_LOCAL" \
    --population-size "$POPULATION_SIZE" \
    --generations "$GENERATIONS" \
    --episodes-per-eval "$EPISODES_PER_EVAL" \
    --training-episodes "$TRAINING_EPISODES" \
    --elite-size "$ELITE_SIZE" \
    --mutation-rate "$MUTATION_RATE" \
    --seed "$SEED"

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# 5) Copy results back (preserved + scoped)
# ------------------------------------------------------------------------------
if [ -d "$RESULTS_LOCAL" ]; then
    cp -r "$RESULTS_LOCAL/"* "$RESULTS_STORAGE/" || true
    echo "Results copied to: $RESULTS_STORAGE"
else
    echo "WARNING: No results directory found!"
fi

cp *.log "$RESULTS_STORAGE/" 2>/dev/null || true

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
