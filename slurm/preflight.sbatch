#!/bin/bash
#SBATCH --job-name=veriflow-preflight
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
source "$HOME/VERIFLOW/slurm/_common.sh"

cd "$VERIFLOW_ROOT"
{
  python --version
  uv --version || true
} > "$VERIFLOW_JOB_META" 2>&1 || true

# 1. environment build/verify (frozen lockfile: uv.lock is committed)
test -d .venv || uv sync --frozen
uv sync --frozen

# 2. full preflight incl. the CPU test suite
uv run python -m scripts.preflight --tests --require-clean
STATUS=$?

python -m scripts.finalize_manifest --run-dir "$VERIFLOW_ROOT/runs/preflight" \
    --exit-code "$STATUS" 2>/dev/null || true
exit "$STATUS"
