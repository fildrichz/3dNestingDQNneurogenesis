#!/bin/bash
# Example usage of model testing scripts
#
# This script demonstrates how to test trained models on specific problems.
# The scripts now use predetermined paths by default:
#   - Model/Genome: trainedModels/trained_model.pth and evolved_genome.json
#   - Dataset: nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input

echo "========================================"
echo "Model Testing Examples"
echo "========================================"
echo ""
echo "Note: Scripts use default paths from trainedModels/ folder"
echo "You can override with --model-path, --genome-path, --dataset-dir"
echo ""

# Example 1: Test on a single problem with visualization (using defaults)
echo "========================================"
echo "Example 1: Single Problem Test (Default: Problem 1)"
echo "========================================"
echo ""
echo "Testing on problem 3dBPP_1.txt with visualization (using default paths)..."
echo ""

python test_model.py \
    --num-episodes 5 \
    --visualize \
    --save-results \
    --output-dir "test_results_example1"

echo ""
echo "Results saved to: test_results_example1/"
echo ""

# Example 2: Test a specific problem by ID
echo "========================================"
echo "Example 2: Test Specific Problem by ID"
echo "========================================"
echo ""
echo "Testing problem 3dBPP_4.txt (using --problem-id 4)..."
echo ""

python test_model.py \
    --problem-id 4 \
    --num-episodes 5 \
    --visualize \
    --output-dir "test_results_example2"

echo ""
echo "Results saved to: test_results_example2/"
echo ""

# Example 3: Batch test on multiple problems (using defaults)
echo "========================================"
echo "Example 3: Batch Test (Default: Problems 1-5)"
echo "========================================"
echo ""
echo "Testing on problems 1, 2, 3, 4, 5 (using default paths)..."
echo ""

python test_model_batch.py \
    --num-episodes 5 \
    --output-dir "test_results_batch_example"

echo ""
echo "Results saved to: test_results_batch_example/"
echo ""

# Example 4: Batch test with custom problem IDs
echo "========================================"
echo "Example 4: Batch Test (Custom Problems)"
echo "========================================"
echo ""
echo "Testing on problems 6, 7, 8, 9..."
echo ""

python test_model_batch.py \
    --problem-ids 6 7 8 9 \
    --num-episodes 3 \
    --output-dir "test_results_custom"

echo ""
echo "All examples completed!"
echo ""
echo "Quick usage:"
echo "  Test default problem:     python test_model.py"
echo "  Test specific problem:    python test_model.py --problem-id 4"
echo "  Batch test defaults:      python test_model_batch.py"
echo "  Batch test custom:        python test_model_batch.py --problem-ids 1 4 12"
echo ""
