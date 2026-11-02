#!/bin/bash
# slurm/_common.sh — isolation block (master plan §16.1). Source from every job
# script immediately after the #SBATCH block. Never modify ErrorEcho paths.
set -euo pipefail

export VERIFLOW_ROOT="$HOME/VERIFLOW"
case "$(pwd -P)" in
  "$VERIFLOW_ROOT"*) ;;
  *) echo "refusing: not in VERIFLOW (cwd=$(pwd -P))" >&2; exit 2 ;;
esac

export HF_HOME="$VERIFLOW_ROOT/.cache/hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UV_CACHE_DIR="$VERIFLOW_ROOT/.cache/uv"
export PIP_CACHE_DIR="$VERIFLOW_ROOT/.cache/pip"
export TRITON_CACHE_DIR="$VERIFLOW_ROOT/.cache/triton"
export TORCH_HOME="$VERIFLOW_ROOT/.cache/torch"
export VLLM_CACHE_ROOT="$VERIFLOW_ROOT/.cache/vllm"
export TMPDIR="${SLURM_TMPDIR:-$VERIFLOW_ROOT/.tmp}"
mkdir -p "$TMPDIR"

source "$VERIFLOW_ROOT/.venv/bin/activate"

# Record the absolute identity of every job once.
export VERIFLOW_JOB_META="$VERIFLOW_ROOT/logs/job_meta_${SLURM_JOB_ID:-interactive}.json"
