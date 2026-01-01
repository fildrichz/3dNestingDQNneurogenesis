#!/bin/bash
# Example usage of model testing scripts
#
# This script demonstrates how to test trained models on specific problems.
# Adjust the paths below to match your actual model and problem locations.

# Set paths (ADJUST THESE TO YOUR ACTUAL PATHS)
MODEL_PATH="results/experiment_multi_problem/trained_model.pth"
GENOME_PATH="results/experiment_multi_problem/evolved_genome.json"
DATASET_DIR="nesting/inputData/Thpack/Input"

echo "========================================"
echo "Model Testing Examples"
echo "========================================"
echo ""

# Check if model files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo "Please train a model first using experiment_multi_problem.py"
    exit 1
fi

if [ ! -f "$GENOME_PATH" ]; then
    echo "ERROR: Genome file not found: $GENOME_PATH"
    echo "Please train a model first using experiment_multi_problem.py"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "Please check your dataset path"
    exit 1
fi

echo "Found model files:"
echo "  Model: $MODEL_PATH"
echo "  Genome: $GENOME_PATH"
echo "  Dataset: $DATASET_DIR"
echo ""

# Example 1: Test on a single problem with visualization
echo "========================================"
echo "Example 1: Single Problem Test"
echo "========================================"
echo ""
echo "Testing on problem 3dBPP_1.txt with visualization..."
echo ""

python test_model.py \
    --model-path "$MODEL_PATH" \
    --genome-path "$GENOME_PATH" \
    --problem-file "$DATASET_DIR/3dBPP_1.txt" \
    --num-episodes 5 \
    --visualize \
    --save-results \
    --output-dir "test_results_example1"

echo ""
echo "Results saved to: test_results_example1/"
echo ""

# Example 2: Batch test on multiple problems
echo "========================================"
echo "Example 2: Batch Test (Multiple Problems)"
echo "========================================"
echo ""
echo "Testing on problems 1, 2, 3, 4, 5..."
echo ""

python test_model_batch.py \
    --model-path "$MODEL_PATH" \
    --genome-path "$GENOME_PATH" \
    --dataset-dir "$DATASET_DIR" \
    --problem-ids 1 2 3 4 5 \
    --num-episodes 5 \
    --output-dir "test_results_batch_example"

echo ""
echo "Results saved to: test_results_batch_example/"
echo ""

# Example 3: Quick test without visualization
echo "========================================"
echo "Example 3: Quick Test (No Visualization)"
echo "========================================"
echo ""
echo "Quick test on problem 3dBPP_2.txt..."
echo ""

python test_model.py \
    --model-path "$MODEL_PATH" \
    --genome-path "$GENOME_PATH" \
    --problem-file "$DATASET_DIR/3dBPP_2.txt" \
    --num-episodes 3 \
    --output-dir "test_results_quick"

echo ""
echo "All examples completed!"
echo ""
echo "To run these examples individually:"
echo "  1. Edit this script to adjust paths if needed"
echo "  2. Comment out examples you don't want to run"
echo "  3. Run: ./example_test_usage.sh"
echo ""
