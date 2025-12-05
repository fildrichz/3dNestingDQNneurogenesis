#!/bin/bash
#PBS -N neurogenesis_exp1
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=10gb
#PBS -l walltime=168:00:00
#PBS -m ae
#PBS -j oe

# Experiment 1: Problem-Specific Architecture Evolution
# This script runs neural architecture evolution for each problem independently.
# Expected runtime: ~7 days for 12 problems with full configuration

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
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
    echo "ERROR: Virtual environment not found at venv/bin/activate"
    echo "Please create it first with:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Configuration
DATASET_DIR="nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"
RESULTS_DIR="results/experiment_problem_specific_$(date +%Y%m%d_%H%M%S)"
POPULATION_SIZE=25
GENERATIONS=20
EPISODES_PER_EVAL=25
TRAINING_EPISODES=500
SEED=42

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Population: $POPULATION_SIZE"
echo "  Generations: $GENERATIONS"
echo "  Episodes per eval: $EPISODES_PER_EVAL"
echo "  Training episodes: $TRAINING_EPISODES"
echo "  Random seed: $SEED"
echo "========================================="
echo ""

# Check GPU
nvidia-smi

echo ""
echo "Starting experiment..."
echo ""

# Run experiment with resume support
python experiment_problem_specific.py \
    --dataset-dir "$DATASET_DIR" \
    --results-dir "$RESULTS_DIR" \
    --population-size $POPULATION_SIZE \
    --generations $GENERATIONS \
    --episodes-per-eval $EPISODES_PER_EVAL \
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

# Copy results back to home directory (in case of scratch cleanup)
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
