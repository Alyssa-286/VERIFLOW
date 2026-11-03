#!/bin/bash
#SBATCH --job-name=veriflow-gate3-report
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# CPU only. Verifies a family of census runs, applies the §9.1 selection rule and,
# for the selected run only, writes gate3.json and the blinded audit export
# (never freezes: the freeze needs the researcher's judged audit).
# Usage: sbatch --export=ALL,REPORT_TAG=v3,REPORT_SEED=20261104 slurm/gate3_report.sbatch
: "${REPORT_TAG:?}" "${REPORT_SEED:?}"
source "$HOME/VERIFLOW/slurm/_common.sh"
cd "$VERIFLOW_ROOT"

OUT="runs/census/gate3_${REPORT_TAG}_s${REPORT_SEED}"
RUNS=$(ls -d runs/census/n*_"${REPORT_TAG}"_s"${REPORT_SEED}"_j* | sort -V)
STATUS=0
uv run --frozen --group gpu python -m experiments.compare_census $RUNS \
    --json "${OUT}_comparison.json" --md "${OUT}_comparison.md" || STATUS=$?
RID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['selection']['run_id'] or '')" "${OUT}_comparison.json")
if [ -n "$RID" ]; then
  uv run --frozen --group gpu python -m experiments.check_construct_gate --census "runs/census/$RID" \
      --json "runs/census/$RID/gate3.json" --audit-sample 200 || STATUS=$?
  echo "selected=$RID"
else
  echo "selected=none"
fi
exit "$STATUS"
