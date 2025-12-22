#!/bin/bash
#PBS -N dqn_multi_problem
#PBS -l select=1:ncpus=2:mem=128gb:ngpus=1:scratch_local=40gb
#PBS -l walltime=47:50:00
#PBS -q gpu
#PBS -m ae
#PBS -M spidlfil@fit.cvut.cz

set -euo pipefail

echo "=========================================="
echo "Multi-problem job started at: $(date)"
echo "Node: $(hostname)"
echo "JobID: ${PBS_JOBID:-N/A}"
echo "=========================================="

# ------------------------------------------------------------------------------
# Configuration (match your Python CLI)
# ------------------------------------------------------------------------------
POPULATION_SIZE=30
GENERATIONS=50
EPISODES_PER_EVAL=100
TRAINING_EPISODES=2000
ELITE_SIZE=3
MUTATION_RATE=0.2
SEED=42

TRAINING_IDS="2 3 5 6 7 8 9 10 11"
TEST_IDS="1 4 12"

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

RESULTS_LOCAL="results/experiment_multi_problem"
RESULTS_STORAGE="/storage/praha1/home/$PBS_O_LOGNAME/results/experiment_multi_problem"

mkdir -p "$RESULTS_STORAGE"
mkdir -p "$RESULTS_LOCAL"

echo "Dataset dir: $DATASET_DIR"
echo "Training IDs: $TRAINING_IDS"
echo "Test IDs: $TEST_IDS"
echo "Results local: $RESULTS_LOCAL"
echo "Results storage: $RESULTS_STORAGE"

# ------------------------------------------------------------------------------
# 4) Restore checkpoint from STORAGE -> SCRATCH (so resume can work)
# ------------------------------------------------------------------------------
CHECKPOINT_STORAGE="$RESULTS_STORAGE/evolution/checkpoint_latest.json"
if [ -f "$CHECKPOINT_STORAGE" ]; then
    echo "Found checkpoint in storage: $CHECKPOINT_STORAGE"
    echo "Restoring results directory from storage to scratch..."
    cp -r "$RESULTS_STORAGE/"* "$RESULTS_LOCAL/" || true
else
    echo "No checkpoint found in storage (starting fresh)."
fi

# ------------------------------------------------------------------------------
# 5) Run multi-problem experiment (resume is ON by default; DO NOT pass --no-resume)
# ------------------------------------------------------------------------------
$PYTHON -u experiment_multi_problem.py \
    --dataset-dir "$DATASET_DIR" \
    --results-dir "$RESULTS_LOCAL" \
    --training-problem-ids $TRAINING_IDS \
    --test-problem-ids $TEST_IDS \
    --population-size "$POPULATION_SIZE" \
    --generations "$GENERATIONS" \
    --episodes-per-eval "$EPISODES_PER_EVAL" \
    --training-episodes "$TRAINING_EPISODES" \
    --elite-size "$ELITE_SIZE" \
    --mutation-rate "$MUTATION_RATE" \
    --seed "$SEED"

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# 6) Copy results back: SCRATCH -> STORAGE (persist checkpoint + outputs)
# ------------------------------------------------------------------------------
echo "Copying results back to storage..."
mkdir -p "$RESULTS_STORAGE"
cp -r "$RESULTS_LOCAL/"* "$RESULTS_STORAGE/" || true

# If you create any *.log files in cwd, copy them too
cp *.log "$RESULTS_STORAGE/" 2>/dev/null || true

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results persisted to: $RESULTS_STORAGE"
echo "=========================================="

exit $EXIT_CODE
