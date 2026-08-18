#!/bin/bash
#SBATCH --job-name=alignn-coordinate-gpu-primary
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpu-coordinate-primary-%j.out
#SBATCH --error=logs/gpu-coordinate-primary-%j.err
set -euo pipefail
bash scripts/run_gpu_coordinate_cell.sh 0 0
