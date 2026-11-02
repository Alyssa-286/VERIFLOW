#!/bin/bash
#SBATCH --job-name=veriflow-census
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_n%CENSUS_N%_%A.out
#SBATCH --error=%x_n%CENSUS_N%_%A.err
# Usage: sbatch --export=CENSUS_N=7 slurm/census.sbatch
: "${CENSUS_N:?pass --export=CENSUS_N=<digits>}"
source "$HOME/VERIFLOW/slurm/_common.sh"

cd "$VERIFLOW_ROOT"

# Model must already be at the pinned revision in $HF_HOME (see gpucheck --models;
# download once on a node with internet: huggingface-cli download <repo> --revision <sha>).
uv run python -m scripts.gpucheck --models --must-pass

RUN_DIR="$VERIFLOW_ROOT/runs/census"
uv run python -m experiments.base_mode_census \
    --n "$CENSUS_N" --model qwen0p5b --samples 4 \
    --seed 20261102 --n-boot 2000 --out "$RUN_DIR"
STATUS=$?

exit "$STATUS"
