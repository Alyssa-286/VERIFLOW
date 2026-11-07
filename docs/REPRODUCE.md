# Reproducing VERIFLOW

Every number in the paper traces to a file in this repository, a git commit, a Slurm job id and SHA-256 hashes. This guide goes from a clean checkout to the paper PDF.

## 0. What is where

| Item | Location |
|---|---|
| Pre-registration (frozen) | `docs/PREREGISTRATION.md` at tag `prereg-v1`; OSF registration https://osf.io/bs3wq/ (recorded 2026-11-07T06:02:21Z) |
| Post-freeze disclosure | `docs/PREREG_ADDENDUM_1.md` (also in the OSF "Other" field) |
| Every decision, in order | `docs/DECISIONS.md` (D0–D31+) |
| Model | Qwen/Qwen2.5-0.5B-Instruct @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0), `configs/model/` |
| Task data | generated deterministically from seeds (`vf_datasets/`); no external dataset |
| Locks | `configs/locked/` (`q_ref.lock`, prompt, taxonomy) |
| Pilot outputs | `configs/pilot/{selection,gate5,power}.json`, `results/pilot/validate_het.json` |
| Grid configs and run table | `configs/grid/` (digest `39815e7b…52a7`, committed on OSF) |
| Grid run summaries | `results/grid/` (`analysis.json`, `runs.csv`, `validate_grid.json`, per-run manifests and eval logs) |
| Figures | `figures/out/` (each with a `.sources.json` of input hashes) |
| Paper | `paper/main.tex`, `paper/main.pdf`; numbers in `paper/generated/numbers.tex` |

## 1. Environment

- Python 3.11–3.12; `uv sync` installs the exact versions in `uv.lock`.
- CPU-only steps (tests, theory, simulation, analysis, paper) need no GPU.
- GPU steps used TRL 1.17.1 and torch from the `gpu` dependency group, on one NVIDIA L40 (48 GB) per run, with the HF-generate backend (no vLLM). See `docs/GATE4.md`.

```bash
uv sync
uv run pytest            # full suite; all tests pass at the release commit
```

## 2. CPU reproduction (minutes)

```bash
uv run python -m simulations.sim_replicator && uv run python -m figures.make_fig3       # Gate 1
uv run python -m simulations.sim_shared_policy && uv run python -m figures.make_gate2   # Gate 2
uv run python -m experiments.gate5                    # Gate 5 from the pilot runs (needs runs/pilot)
uv run python -m vf_stats.power_sim --acc0 0.408      # power analysis (D23)
uv run python -m experiments.validate_runs --table configs/grid/runs_grid.csv --runs-root runs/grid --eval-every 10 --out results/grid/validate_grid.json
uv run python -m experiments.analyse_grid             # registered analysis -> results/grid/analysis.json
uv run python -m figures.make_grid_figs               # final figures
uv run python paper/make_numbers.py && (cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main)
```

## 3. GPU reproduction (Slurm)

`slurm/grpo_array.sbatch` runs one GRPO run per array task from a run table. The exact submissions:

```bash
# heterogeneity pilot (array 4020): 12 runs
sbatch --array=0-11%2 --cpus-per-task=2 --export=ALL,RUN_TABLE=configs/pilot/runs_het.csv,RUN_KIND=pilot,RUNS_ROOT=runs/pilot,EVAL_EVERY=10 slurm/grpo_array.sbatch
# confirmatory grid (array 4036): 96 runs
sbatch --array=0-95%2 --cpus-per-task=2 --export=ALL,RUN_TABLE=configs/grid/runs_grid.csv,RUN_KIND=grid,RUNS_ROOT=runs/grid,EVAL_EVERY=10 slurm/grpo_array.sbatch
```

Each run takes about 2.5–3 minutes on an L40, so the grid takes about 4.5 GPU-hours. Every run writes `manifest.json`, `eval.jsonl`, `metrics.jsonl`, `verifier_log/*.parquet` and checkpoints. A failed task is resubmitted with `--array=<idx>` and resumes from its checkpoint.

**Determinism.** The verifier is a SHA-256 PRF of (verifier seed, run seed, step, prompt, sample), so its decisions replay exactly. Training is a deterministic function of (config, seed, eval cadence) up to GPU non-determinism. The trainer re-seeds the torch RNG at each evaluation (D21), so the eval cadence must stay at 10.

## 4. What cannot be reproduced bit-for-bit

- The census and training generations depend on GPU kernels; re-runs agree statistically, not bitwise.
- The construct audit labels (Gate 3′) were made by an AI assistant and reviewed by the researcher (D20, D22). A human re-audit could differ.
- Cluster-specific paths and hostnames in `docs/HPC_RUNBOOK.md` and `docs/CLUSTER_POLICY.md` describe the original SIT cluster.
