#!/bin/bash
#SBATCH --job-name=veriflow-bench
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# Gate 4 throughput (master plan §16.3 job 5): HF-generate backend, 32 prompts/step,
# G in {8, 16}, 10 steps each; full-size eval (1,000 greedy + 1,000 sampled) timed at steps 0 and 10
# for G = 8. ENGINEERING ONLY: provisional homogeneous verifier.
source "$HOME/VERIFLOW/slurm/_common.sh"
cd "$VERIFLOW_ROOT"
: "${SMOKE_CENSUS:=runs/census/n5_v3_s20261104_j3640}"
OUT="runs/bench/j${SLURM_JOB_ID}"
PY="uv run --frozen --group gpu python"
for G in 8 16; do
  EV=0; [ "$G" = 8 ] && EV=10
  $PY -m training.train_grpo --run-dir "$OUT/G$G" --run-kind bench --provisional-census "$SMOKE_CENSUS" \
      --lr 1e-6 --seed 1 --max-steps 10 --save-every 1000 --eval-every "$EV" --num-generations "$G"
done
$PY -m training.bench_report "$OUT"
