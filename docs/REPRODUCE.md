# Reproducing VERIFLOW

Every number in the paper traces to a file in this repository, a git commit, a Slurm job id and SHA-256 hashes. This guide goes from a clean checkout to the paper PDF. Tested on Windows 11 (Python 3.12, MiKTeX) and on the Linux cluster used for the study; the registered analysis gives the same results on both (≤ 1e-9).

## 0. What is where

| Item | Location |
|---|---|
| Pre-registration (frozen) | `docs/PREREGISTRATION.md` at tag `prereg-v1`; OSF https://osf.io/bs3wq/, DOI 10.17605/OSF.IO/BS3WQ (registered 2026-11-07T06:02:21Z, public) |
| Post-freeze disclosure | `docs/PREREG_ADDENDUM_1.md` (also in the OSF "Other" field) |
| Phase 4 closure | `docs/PHASE4_CLOSURE.md`, `docs/gate3_tv_diagnosis.json`, `experiments/diagnose_split_half_tv.py` |
| Every decision, in order | `docs/DECISIONS.md` (D0–D34); phase-by-phase evidence in `docs/PHASE_LEDGER.md` |
| OSF hash commitments and commit map | `docs/PROVENANCE.md` |
| Model | Qwen/Qwen2.5-0.5B-Instruct @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0), `configs/model/`; not redistributed |
| Task data | generated deterministically from seeds (`vf_datasets/`); no external dataset |
| Locks | `configs/locked/` (`q_ref.lock` sha256 `737aab73…`, lock hash `2c34f6d2…d5cb`; prompt; taxonomy) |
| Pilot outputs | `configs/pilot/{selection,gate5,power}.json`, `results/pilot/validate_het.json` |
| Grid configs and run table | `configs/grid/` (directory digest `39815e7b…52a7`, committed on OSF) |
| Per-run records (96 runs) | `results/grid/run_records/<run>/{manifest.json, eval.jsonl, metrics.jsonl, generation_config.json}`; `INDEX.json` gives the SHA-256 of every original and exported file |
| Verifier decision logs (every accept/reject, 96 runs) | release asset `veriflow_grid_verifier_logs.tar.gz` (GitHub release v1.0.0 and the OSF project; SHA-256 in the release notes) |
| Grid validation, registered analysis, supplement | `results/grid/{validate_grid.json, analysis.json, runs.csv, supplement.json, runs_realised.csv}` |
| Figures | `figures/out/` (each with a `.sources.json` of input hashes) |
| Paper | `paper/main.tex`, `paper/sections/`, `paper/main.pdf`; numbers in `paper/generated/` |

**Not included:** model checkpoints (≈ 1.1 TB, kept on the original cluster) and the downloaded model weights. Neither is needed for the analysis; both can be regenerated (§3).

**Redactions:** run manifests have the cluster home path replaced by `<REPO>` and the node hostname by `<cluster-node>`; cluster documents have the login host, its IP address, the NFS server address and the account name replaced by placeholders. Nothing used by the validator or the analysis is affected. See `docs/PROVENANCE.md`.

## 1. Environment

- Python 3.11–3.12; `uv sync` installs the exact versions in `uv.lock`.
- CPU-only steps (tests, theory, simulation, analysis, paper) need no GPU. LaTeX: any TeX Live or MiKTeX with `natbib`, `booktabs`, `microtype`, `lmodern`.
- GPU steps used TRL 1.17.1 and torch from the `gpu` dependency group (`uv sync --group gpu --frozen`), one NVIDIA L40 (48 GB) per run, HF-generate backend (no vLLM). See `docs/GATE4.md`.

```bash
uv sync
uv run pytest            # full suite (184 tests); all pass at the release commit
```

## 2. CPU reproduction of the analysis and paper (minutes)

```bash
# the registered analysis needs only the committed run records
uv run python -m experiments.analyse_grid --runs-root results/grid/run_records --out results/grid
# validation and the supplement also need the verifier logs: unpack the release archive into the records tree
tar xzf veriflow_grid_verifier_logs.tar.gz -C results/grid/run_records
uv run python -m experiments.validate_runs --table configs/grid/runs_grid.csv --runs-root results/grid/run_records --eval-every 10 --out results/grid/validate_grid.json
uv run python -m experiments.analyse_grid_supplement --runs-root results/grid/run_records
# figures and paper
VERIFLOW_GRID_RUNS=results/grid/run_records uv run python -m figures.make_grid_figs
VERIFLOW_GRID_RUNS=results/grid/run_records uv run python -m figures.make_grid_supp_figs
uv run python paper/make_numbers.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Earlier stages:

```bash
uv run python -m simulations.sim_replicator && uv run python -m figures.make_fig3       # Gate 1
uv run python -m simulations.sim_shared_policy && uv run python -m figures.make_gate2   # Gate 2
uv run python -m vf_stats.power_sim --acc0 0.408      # power analysis (D23)
uv run python -m experiments.gate5                    # Gate 5 (needs the pilot run directories, cluster only)
```

## 3. GPU reproduction (Slurm)

`slurm/grpo_array.sbatch` runs one GRPO run per array task from a run table. Edit the `#SBATCH` partition and node lines for your cluster. The exact submissions:

```bash
# heterogeneity pilot (array 4020): 12 runs
sbatch --array=0-11%2 --cpus-per-task=2 --export=ALL,RUN_TABLE=configs/pilot/runs_het.csv,RUN_KIND=pilot,RUNS_ROOT=runs/pilot,EVAL_EVERY=10 slurm/grpo_array.sbatch
# confirmatory grid (array 4036): 96 runs, two at a time, one L40 each
sbatch --array=0-95%2 --cpus-per-task=2 --export=ALL,RUN_TABLE=configs/grid/runs_grid.csv,RUN_KIND=grid,RUNS_ROOT=runs/grid,EVAL_EVERY=10 slurm/grpo_array.sbatch
```

Each run takes about 2.5–3 minutes on an L40 (median 1.04 s per step); the grid used 3.4 GPU-hours of training. Every run writes `manifest.json`, `eval.jsonl`, `metrics.jsonl`, `verifier_log/*.parquet` and checkpoints. A failed task is resubmitted with `--array=<idx>` and resumes from its checkpoint. Validate with `experiments/validate_runs.py` (§2) before analysing.

**Determinism.** The verifier is a SHA-256 PRF of (verifier seed, run seed, step, prompt, sample), so its decisions replay exactly. Training is a deterministic function of (config, seed, eval cadence) up to GPU non-determinism; on the original hardware, duplicated configurations reproduced bit-identical outcomes (D33). The trainer re-seeds the torch RNG at each evaluation (D21), so the eval cadence must stay at 10.

**Code transfer to an offline cluster.** Use a git bundle: `git bundle create veriflow.bundle <branch>`, copy it, then `git pull --ff-only <bundle> <branch>` (see `docs/HPC_RUNBOOK.md`).

## 4. What cannot be reproduced bit-for-bit

- The census and training generations depend on GPU kernels; re-runs on other hardware agree statistically, not bitwise.
- The construct audit labels (Gate 3′) were made by an AI assistant and reviewed by the researcher (D20, D22). A human re-audit could differ.
- `docs/HPC_RUNBOOK.md` and `docs/CLUSTER_POLICY.md` describe the original cluster, with private identifiers replaced by placeholders.
