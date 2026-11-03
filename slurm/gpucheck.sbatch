#!/bin/bash
#SBATCH --job-name=veriflow-gpucheck
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
source "$HOME/VERIFLOW/slurm/_common.sh"

cd "$VERIFLOW_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv \
    > "logs/gpu_inventory_${SLURM_JOB_ID}.txt" || true
nvidia-smi || true

# --models verifies the pinned snapshot; --generate loads it on the GPU (bf16),
# runs a 16-token greedy generation and records peak VRAM.
uv run --frozen --group gpu python -m scripts.gpucheck --models --generate --must-pass \
    | tee "logs/gpucheck_${SLURM_JOB_ID}.json"
