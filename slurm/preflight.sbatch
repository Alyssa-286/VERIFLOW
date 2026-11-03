#!/bin/bash
#SBATCH --job-name=veriflow-preflight
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
source "$HOME/VERIFLOW/slurm/_common.sh"

cd "$VERIFLOW_ROOT"
{
  python --version
  uv --version || true
} > "$VERIFLOW_JOB_META" 2>&1 || true

# 1. environment build/verify (frozen lockfile: uv.lock is committed). The gpu
#    group must be named, or uv sync would uninstall torch/transformers.
uv sync --frozen --group gpu

# 2. full preflight incl. the CPU test suite
STATUS=0
uv run --frozen --group gpu python -m scripts.preflight --tests --require-clean || STATUS=$?

python -m scripts.finalize_manifest --run-dir "$VERIFLOW_ROOT/runs/preflight" \
    --exit-code "$STATUS" 2>/dev/null || true
exit "$STATUS"
