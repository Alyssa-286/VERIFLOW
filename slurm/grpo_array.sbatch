#!/bin/bash
#SBATCH --job-name=veriflow-pilot
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
# One GRPO run per array task, from a run table (master plan §16.4). Always one at a time:
#   sbatch --array=0-3%1 --export=ALL,RUN_TABLE=configs/pilot/runs_lr.csv,RUN_KIND=pilot,RUNS_ROOT=runs/pilot slurm/grpo_array.sbatch
# A failed task is resubmitted with --array=<idx>; the trainer resumes from its latest checkpoint.
: "${RUN_TABLE:?}" "${RUN_KIND:?}" "${RUNS_ROOT:?}"
source "$HOME/VERIFLOW/slurm/_common.sh"
cd "$VERIFLOW_ROOT"
ROW=$(python3 -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1])))[int(sys.argv[2])]; print(r['run_name'], r['config'], r['lr'], r['seed'], r['max_steps'])" "$RUN_TABLE" "$SLURM_ARRAY_TASK_ID")
read -r NAME CONFIG LR SEED STEPS <<< "$ROW"
echo "task $SLURM_ARRAY_TASK_ID: $NAME config=$CONFIG lr=$LR seed=$SEED steps=$STEPS"
uv run --frozen --group gpu python -m training.train_grpo --run-dir "$RUNS_ROOT/$NAME" --run-kind "$RUN_KIND" \
    --verifier-config "$CONFIG" --lr "$LR" --seed "$SEED" --max-steps "$STEPS" &
PID=$!
trap 'kill -TERM $PID 2>/dev/null' USR1
wait $PID
