#!/bin/bash
#PBS -N neurogenesis_exp2
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=10gb
#PBS -l walltime=96:00:00
#PBS -J 1-12
#PBS -m ae
#PBS -j oe

# Experiment 2: Leave-One-Out Generalization Test (Array Job)
# This script runs 12 parallel jobs, each holding out one problem.
# Expected runtime: ~4 days per job
# Array index corresponds to problem number (1-12)

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Array Index: $PBS_ARRAY_INDEX"
echo "Node: $(hostname)"
echo "========================================="

# Load required modules
module load python/3.9.0-gcc
module load cuda/11.7

# Set working directory
cd $PBS_O_WORKDIR || exit 1

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found"
    exit 1
fi

# Configuration
DATASET_DIR="nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"
TARGET_PROBLEM="3dBPP_${PBS_ARRAY_INDEX}"
RESULTS_DIR="results/experiment_leave_one_out_${TARGET_PROBLEM}_$(date +%Y%m%d_%H%M%S)"
POPULATION_SIZE=25
GENERATIONS=20
EPISODES_PER_PROBLEM=12
TRAINING_EPISODES=400
SEED=$PBS_ARRAY_INDEX

echo ""
echo "Configuration:"
echo "  Target problem: $TARGET_PROBLEM"
echo "  Training on: All problems except $TARGET_PROBLEM"
echo "  Results: $RESULTS_DIR"
echo "  Population: $POPULATION_SIZE"
echo "  Generations: $GENERATIONS"
echo "  Episodes per problem: $EPISODES_PER_PROBLEM"
echo "  Training episodes: $TRAINING_EPISODES"
echo "  Random seed: $SEED"
echo "========================================="
echo ""

# Check GPU
nvidia-smi

echo ""
echo "Starting experiment for $TARGET_PROBLEM..."
echo ""

# Run experiment with resume support
python experiment_leave_one_out.py \
    --dataset-dir "$DATASET_DIR" \
    --target-problem "$TARGET_PROBLEM" \
    --results-dir "$RESULTS_DIR" \
    --population-size $POPULATION_SIZE \
    --generations $GENERATIONS \
    --episodes-per-problem $EPISODES_PER_PROBLEM \
    --training-episodes $TRAINING_EPISODES \
    --elite-size 2 \
    --mutation-rate 0.2 \
    --seed $SEED

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

# Copy results back to home directory
if [ -d "$RESULTS_DIR" ]; then
    echo "Copying results to permanent storage..."
    mkdir -p "$PBS_O_WORKDIR/results/"
    cp -r "$RESULTS_DIR" "$PBS_O_WORKDIR/results/"
    echo "Results saved to: $PBS_O_WORKDIR/results/$(basename $RESULTS_DIR)"
fi

# Clean up scratch if used
if [ -n "$SCRATCHDIR" ]; then
    rm -rf "$SCRATCHDIR"
fi

exit $EXIT_CODE
