#!/bin/bash
#SBATCH --job-name=veriflow-smoke
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# Gate 4 smoke (master plan §16.3 job 3): 1-step run; 10-step run saving at 5 and 10;
# resume of a copy truncated to checkpoint-5 back to step 10; verifier-log replay check.
# ENGINEERING ONLY: provisional homogeneous verifier from the unfrozen v3 census.
source "$HOME/VERIFLOW/slurm/_common.sh"
cd "$VERIFLOW_ROOT"
: "${SMOKE_CENSUS:=runs/census/n5_v3_s20261104_j3640}"
OUT="runs/smoke/j${SLURM_JOB_ID}"
mkdir -p "$OUT"
PY="uv run --frozen --group gpu python"
COMMON="--run-kind smoke --provisional-census $SMOKE_CENSUS --lr 1e-6 --seed 1 --eval-size 100 --eval-mode-size 100"

echo "== 1-step"
$PY -m training.train_grpo --run-dir "$OUT/one_step" $COMMON --max-steps 1 --eval-every 1
echo "== 10-step, save every 5"
$PY -m training.train_grpo --run-dir "$OUT/ten_step" $COMMON --max-steps 10 --save-every 5 --eval-every 5
echo "== resume: copy truncated to checkpoint-5, continue to 10"
mkdir -p "$OUT/resumed/checkpoints"
cp -r "$OUT/ten_step/checkpoints/checkpoint-5" "$OUT/resumed/checkpoints/"
$PY -m training.train_grpo --run-dir "$OUT/resumed" $COMMON --max-steps 10 --save-every 5 --eval-every 5
echo "== checks"
$PY -m training.smoke_check "$OUT"
