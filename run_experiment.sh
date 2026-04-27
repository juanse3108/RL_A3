#!/bin/bash
#SBATCH --job-name=rl_a3
#SBATCH --partition=cpu-skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/rl_a3_%j.out
#SBATCH --error=logs/rl_a3_%j.err

set -e

cd /home/s4567846/projects/rl_projects/a3/RL_A3
mkdir -p logs

source /zfsstore/user/s4567846/projects/rl_projects/a3/rl_env/bin/activate

echo "Job started on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Virtual env: $VIRTUAL_ENV"

python -u src/train.py

echo "Job finished successfully"