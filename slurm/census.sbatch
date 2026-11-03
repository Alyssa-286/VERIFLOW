#!/bin/bash
#SBATCH --job-name=veriflow-census
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# Usage (from ~/VERIFLOW): sbatch --job-name=veriflow-census-n7 --export=ALL,CENSUS_N=7 slurm/census.sbatch
: "${CENSUS_N:?pass --export=ALL,CENSUS_N=<digits>}"
source "$HOME/VERIFLOW/slurm/_common.sh"

cd "$VERIFLOW_ROOT"

# Model must already be at the pinned revision in $HF_HOME (see gpucheck --models;
# download once on the login node: hf download <repo> --revision <sha>).
uv run --frozen --group gpu python -m scripts.gpucheck --models --must-pass

# Distinct run id per job: a rerun never overwrites an earlier census.
RUN_DIR="$VERIFLOW_ROOT/runs/census"
STATUS=0
uv run --frozen --group gpu python -m experiments.base_mode_census \
    --n "$CENSUS_N" --model qwen0p5b --samples 4 \
    --seed 20261102 --n-boot 2000 --out "$RUN_DIR" \
    --run-id "n${CENSUS_N}_p2_s20261102_j${SLURM_JOB_ID}" || STATUS=$?

# Exit 1 = the census completed but the §9.3 automated gate failed for this n
# (expected for some n); any other nonzero is a run failure.
exit "$STATUS"
