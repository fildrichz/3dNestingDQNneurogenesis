#!/bin/bash
#PBS -N dqn_problem_specific
#PBS -l select=1:ncpus=2:mem=128gb:ngpus=1:scratch_local=40gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -m ae
#PBS -M spidlfil@fit.cvut.cz

set -euo pipefail

# ------------------------------------------------------------------------------
# Resolve INDEX (argument or PBS array index)
# ------------------------------------------------------------------------------
if [ $# -ge 1 ]; then
    PROBLEM_INDEX="$1"
elif [ -n "${PBS_ARRAY_INDEX:-}" ]; then
    PROBLEM_INDEX="$PBS_ARRAY_INDEX"
else
    echo "ERROR: No problem index provided"
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
echo "JobID: ${PBS_JOBID:-N/A}"
echo "Array index: ${PBS_ARRAY_INDEX:-N/A}"
echo "Problem index: $PROBLEM_INDEX"
echo "=========================================="

# ------------------------------------------------------------------------------
# 1) Load environment
# ------------------------------------------------------------------------------
module purge
module add mambaforge || { echo "ERROR: Failed to load mambaforge"; exit 1; }

PYTHON="/storage/praha1/home/$PBS_O_LOGNAME/dp_env/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

$PYTHON --version

# ------------------------------------------------------------------------------
# 2) Scratch setup
# ------------------------------------------------------------------------------
cd "$SCRATCHDIR" || exit 1

cp -r /storage/praha1/home/$PBS_O_LOGNAME/dp-filip-spidla-spidlfil .
cd dp-filip-spidla-spidlfil/main || exit 1

echo "Working directory: $(pwd)"

# ------------------------------------------------------------------------------
# 3) Dataset + results paths
# ------------------------------------------------------------------------------
DATASET_DIR="nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"

RESULTS_LOCAL="results/problem_specific/problem_${PROBLEM_INDEX}"
RESULTS_STORAGE="/storage/praha1/home/$PBS_O_LOGNAME/results/problem_specific/problem_${PROBLEM_INDEX}"
mkdir -p "$RESULTS_STORAGE"

echo "Dataset dir: $DATASET_DIR"
echo "File pattern: 3dBPP_${PROBLEM_INDEX}*.txt"
echo "Results local: $RESULTS_LOCAL"
echo "Results storage: $RESULTS_STORAGE"

# ------------------------------------------------------------------------------
# 4) Run experiment (ONLY change vs single-job version)
# ------------------------------------------------------------------------------
$PYTHON -u experiment_problem_specific.py \
    --dataset-dir "$DATASET_DIR" \
    --problem-id "$PROBLEM_INDEX" \
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
# 5) Copy results back
# ------------------------------------------------------------------------------
if [ -d "$RESULTS_LOCAL" ]; then
    cp -r "$RESULTS_LOCAL/"* "$RESULTS_STORAGE/" || true
fi

cp *.log "$RESULTS_STORAGE/" 2>/dev/null || true

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
