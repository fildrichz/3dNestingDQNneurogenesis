#!/bin/bash

# Setup script for running experiments on Metacentrum
# Run this once before submitting jobs

set -e

echo "========================================="
echo "Setting up environment for Metacentrum"
echo "========================================="

# Load modules
echo "Loading modules..."
module load python/3.9.0-gcc
module load cuda/11.7

# Check Python version
echo ""
echo "Python version:"
python --version

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo ""
    echo "Virtual environment already exists"
fi

# Activate environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo ""
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo ""
    echo "WARNING: requirements.txt not found"
    echo "Installing common packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    pip install numpy scipy matplotlib
fi

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check dataset
DATASET_DIR="nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"
if [ -d "$DATASET_DIR" ]; then
    echo ""
    echo "Dataset directory found:"
    ls -1 "$DATASET_DIR"/3dBPP_*.txt | grep -v "_sol.txt" | wc -l
    echo "problem files detected"
else
    echo ""
    echo "WARNING: Dataset directory not found at $DATASET_DIR"
    echo "Please ensure dataset is in the correct location"
fi

# Make scripts executable
echo ""
echo "Making batch scripts executable..."
chmod +x run_experiment1_metacentrum.sh
chmod +x run_experiment2_metacentrum.sh

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify dataset is in correct location"
echo "2. Submit jobs:"
echo "   qsub run_experiment1_metacentrum.sh  # Problem-specific"
echo "   qsub run_experiment2_metacentrum.sh  # Leave-one-out"
echo ""
echo "3. Monitor jobs:"
echo "   qstat -u \$USER"
echo ""
echo "4. Check job output:"
echo "   cat neurogenesis_exp1.o*"
echo "   cat neurogenesis_exp2.o*"
echo ""
echo "========================================="
