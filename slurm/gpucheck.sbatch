#!/bin/bash
#SBATCH --job-name=veriflow-gpucheck
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
source "$HOME/VERIFLOW/slurm/_common.sh"

cd "$VERIFLOW_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > gpu_inventory.txt || true

# --models also verifies the pinned revisions against the snapshot cache refs.
uv run python -m scripts.gpucheck --models --must-pass
