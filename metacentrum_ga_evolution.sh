#!/bin/bash
#PBS -N ga_evolution_packing
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:scratch_local=20gb
#PBS -l walltime=48:00:00
#PBS -m ae

# Metacentrum PBS job script for GA-based DQN Architecture Evolution
# This script runs the genetic algorithm to evolve DQN architectures
# WARNING: This can take 24-48 hours depending on population size and generations

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

# Run the GA evolution
cd main
python example_ga_training.py

# Copy results back to home directory
mkdir -p $PBS_O_WORKDIR/output_data/ga_evolution
cp -r output_data/ga_evolution/* $PBS_O_WORKDIR/output_data/ga_evolution/ 2>/dev/null || true
cp *.png $PBS_O_WORKDIR/output_data/ga_evolution/ 2>/dev/null || true
cp *.json $PBS_O_WORKDIR/output_data/ga_evolution/ 2>/dev/null || true

# Clean up scratch
clean_scratch
