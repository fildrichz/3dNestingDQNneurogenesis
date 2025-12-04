#!/bin/bash
#PBS -N dqn_packing_training
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=10gb
#PBS -l walltime=24:00:00
#PBS -m ae

# Metacentrum PBS job script for DQN 3D Bin Packing Training
# This script runs the enhanced DQN training on a single problem instance

# Load required modules
module add mambaforge
module add cuda/12.2

# Set up environment variables
export OMP_NUM_THREADS=$PBS_NUM_PPN
export CUDA_VISIBLE_DEVICES=0

# Go to scratch directory
cd $SCRATCHDIR || exit 1

# Copy project files to scratch
cp -r $PBS_O_WORKDIR/main .
cp -r $PBS_O_WORKDIR/requirements.txt .

# Create virtual environment and install dependencies
mamba create -p ./env python=3.11 -y
source activate ./env
pip install --no-cache-dir -r requirements.txt

# Run the training
cd main
python packing_with_dqncore2_enhanced.py

# Copy results back to home directory
mkdir -p $PBS_O_WORKDIR/output_data/metacentrum_runs
cp -r output_data/* $PBS_O_WORKDIR/output_data/metacentrum_runs/ 2>/dev/null || true
cp *.png $PBS_O_WORKDIR/output_data/metacentrum_runs/ 2>/dev/null || true
cp *.txt $PBS_O_WORKDIR/output_data/metacentrum_runs/ 2>/dev/null || true

# Clean up scratch
clean_scratch
