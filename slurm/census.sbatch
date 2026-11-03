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
# Optional: CENSUS_TASK=add|sub (D13), CENSUS_TAXONOMY=mech|v3 (D15),
#           CENSUS_SEED=<split+generation seed> (Amendment 1: 20261104), CENSUS_MODEL=qwen0p5b|qwen1p5b
: "${CENSUS_N:?pass --export=ALL,CENSUS_N=<digits>}"
source "$HOME/VERIFLOW/slurm/_common.sh"

cd "$VERIFLOW_ROOT"

CENSUS_TASK="${CENSUS_TASK:-add}"
CENSUS_TAXONOMY="${CENSUS_TAXONOMY:-mech}"
CENSUS_SEED="${CENSUS_SEED:-20261102}"
CENSUS_MODEL="${CENSUS_MODEL:-qwen0p5b}"

# Model must already be at the pinned revision in $HF_HOME (download once on the
# login node: hf download <repo> --revision <sha>).
uv run --frozen --group gpu python -m scripts.gpucheck --models --model "$CENSUS_MODEL" --must-pass

# Distinct run id per job: a rerun never overwrites an earlier census.
TAG="p2t2"; [ "$CENSUS_TASK" = sub ] && TAG="sub_p2t2"
[ "$CENSUS_TAXONOMY" = v3 ] && TAG="v3"
[ "$CENSUS_MODEL" = qwen1p5b ] && TAG="${TAG}q15"
RUN_DIR="$VERIFLOW_ROOT/runs/census"
STATUS=0
uv run --frozen --group gpu python -m experiments.base_mode_census \
    --n "$CENSUS_N" --task "$CENSUS_TASK" --taxonomy "$CENSUS_TAXONOMY" --model "$CENSUS_MODEL" \
    --samples 4 --seed "$CENSUS_SEED" --base-seed "$CENSUS_SEED" --n-boot 2000 --out "$RUN_DIR" \
    --run-id "n${CENSUS_N}_${TAG}_s${CENSUS_SEED}_j${SLURM_JOB_ID}" || STATUS=$?

# Exit 1 = the census completed but the §9.3 automated gate failed for this n
# (expected for some n); any other nonzero is a run failure.
exit "$STATUS"
