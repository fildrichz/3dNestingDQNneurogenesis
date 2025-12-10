#!/bin/bash
#PBS -N test_metacentrum_setup
#PBS -l select=1:ncpus=2:mem=4gb:scratch_local=2gb
#PBS -l walltime=0:10:00
#PBS -q default

# ==============================================================================
# MetaCentrum Setup Test Script
# ==============================================================================
# Run this first to verify your environment is working correctly
# This is a quick 10-minute test job
# ==============================================================================

# Set up environment
echo "Loading modules..."
module load python/3.11.4-gcc-10.2.1-mjh74tn

# Optional: Activate virtual environment if created
# source /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis/venv/bin/activate

# Change to scratch directory
cd $SCRATCHDIR || exit 1

# Copy project to scratch
echo "Copying project files..."
cp -r /storage/brno2/home/$PBS_O_LOGNAME/3dNestingDQNneurogenesis .
cd 3dNestingDQNneurogenesis

echo "=========================================="
echo "MetaCentrum Setup Test"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Test 1: Python version
echo "Test 1: Python version"
python3 --version
echo ""

# Test 2: Import numpy
echo "Test 2: Testing NumPy"
python3 << 'EOF'
try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
    # Quick array test
    arr = np.random.rand(100, 100)
    print(f"✓ NumPy array creation: OK (shape: {arr.shape})")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    exit(1)
EOF
echo ""

# Test 3: Import PyTorch
echo "Test 3: Testing PyTorch"
python3 << 'EOF'
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
    # Quick tensor test
    x = torch.randn(10, 10)
    print(f"✓ PyTorch tensor creation: OK (shape: {x.shape})")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    exit(1)
EOF
echo ""

# Test 4: Check project structure
echo "Test 4: Checking project structure"
if [ -d "main" ]; then
    echo "✓ main/ directory found"
else
    echo "✗ main/ directory NOT found"
fi

if [ -f "main/experiment_leave_one_out.py" ]; then
    echo "✓ experiment_leave_one_out.py found"
else
    echo "✗ experiment_leave_one_out.py NOT found"
fi

if [ -f "main/experiment_problem_specific.py" ]; then
    echo "✓ experiment_problem_specific.py found"
else
    echo "✗ experiment_problem_specific.py NOT found"
fi
echo ""

# Test 5: Try importing project modules
echo "Test 5: Testing project imports"
cd main
python3 << 'EOF'
import sys
try:
    from nesting.dataset_loader import load_problem
    print("✓ dataset_loader import: OK")
except Exception as e:
    print(f"✗ dataset_loader import failed: {e}")

try:
    from genome import NetworkGenome
    print("✓ genome import: OK")
except Exception as e:
    print(f"✗ genome import failed: {e}")

try:
    from ga_evolution import evaluate_genome_fitness
    print("✓ ga_evolution import: OK")
except Exception as e:
    print(f"✗ ga_evolution import failed: {e}")

try:
    from dqn_core.dqn_enhanced import DQNAgentEnhanced
    print("✓ dqn_enhanced import: OK")
except Exception as e:
    print(f"✗ dqn_enhanced import failed: {e}")
EOF
cd ..
echo ""

# Test 6: Check dataset files
echo "Test 6: Checking dataset availability"
DATASET_DIR="main/nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP"
if [ -d "$DATASET_DIR" ]; then
    echo "✓ Dataset directory found"
    NUM_FILES=$(find "$DATASET_DIR" -name "*.txt" 2>/dev/null | wc -l)
    echo "  - Found $NUM_FILES .txt files"
else
    echo "⚠ Dataset directory NOT found at expected location"
    echo "  You may need to upload dataset files"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Test Complete!"
echo "=========================================="
echo "Job finished at: $(date)"
echo ""
echo "Next steps:"
echo "1. Check this output log for any ✗ errors"
echo "2. If all tests passed (✓), you're ready to submit real jobs!"
echo "3. If tests failed (✗), check the error messages above"
echo "4. Submit your first real experiment with run_leave_one_out.sh"
echo ""
echo "Results will be in: ~/results/"
echo "=========================================="

# No results to copy for this test, but let's test that too
mkdir -p /storage/praha1/home/$PBS_O_LOGNAME/metacentrum_test_results
echo "Test completed successfully at $(date)" > test_output.txt
cp test_output.txt /storage/praha1/home/$PBS_O_LOGNAME/metacentrum_test_results/

exit 0
