# VERIFLOW

**Verifier Error Heterogeneity and Learning Dynamics**: does the *shape* of a verifier's false-positive distribution across error modes predict reinforcement-learning-with-verifiable-rewards (RLVR) outcomes when aggregate error rates are matched exactly?

- Plan: [docs/VERIFLOW_MASTER_PLAN.md](docs/VERIFLOW_MASTER_PLAN.md)
- Roadmap and status: [docs/ROADMAP.md](docs/ROADMAP.md)
- Decisions: [docs/DECISIONS.md](docs/DECISIONS.md)
- Gate reports: [docs/GATE1.md](docs/GATE1.md), [docs/GATE2.md](docs/GATE2.md)

## Status (2026-11-01)

| Phase | Name | Status |
|---|---|---|
| 0 | Planning, novelty | Done |
| 1 | Theory | Gate 1 pass |
| 2 | Simulation | Gate 2 pass |
| 3 | Construct tooling | Done |
| 4 | Cluster + census | Tooling done; census pending |

## Setup (CPU)

```bash
uv sync
uv run pytest
```

## Reproduce

```bash
uv run python -m simulations.sim_replicator      # Gate 1 sweep, a few minutes
uv run python -m figures.make_fig3               # Figure 3 (a, b, regime map)
uv run python -m simulations.sim_shared_policy   # Gate 2, about 2 minutes
uv run python -m figures.make_gate2              # Gate 2 figure
```

## Phase 4 (census, first GPU use)

Docs: [docs/HPC_RUNBOOK.md](docs/HPC_RUNBOOK.md) (exact commands),
[docs/CLUSTER_POLICY.md](docs/CLUSTER_POLICY.md) (verified scheduler facts),
[docs/GATE3.md](docs/GATE3.md) (criteria). Smoke the census on CPU:

```bash
uv run python -m experiments.base_mode_census --n 6 --backend mock --size 400 \
    --samples 8 --seed 42 --out .tmp/e2e --n-boot 100
uv run python -m experiments.check_construct_gate --census .tmp/e2e/n6_s42
uv run python -m scripts.preflight                # layout, locks, isolation
```

## Layout

| Path | Contents |
|---|---|
| `verifiers/` | `stats.py` (every verifier statistic), `matched_verifiers.py` (exact matched configs), `noisy_verifier.py` (stochastic verifier + TRL reward fn) |
| `error_modes/` | answer parser and deterministic taxonomy v1 |
| `vf_datasets/` | addition prompts with disjoint, hashed splits |
| `simulations/` | idealized replicator model; finite-G GRPO engine; Gate 2 driver |
| `vf_stats/` | leave-one-config-out predictor comparison |
| `experiments/` | Phase 4: base-policy error census (`base_mode_census.py`, `census_core.py`), §9.3 gate checker, mock backend |
| `models/` | pinned model/tokenizer loading (`configs/model/*.yaml` revisions) |
| `scripts/` | run manifests (§21), `preflight`, `gpucheck`, run-table specs |
| `slurm/`, `bin/` | SIT HPC job scripts + `veriflow-preflight` / `veriflow-gpucheck` wrappers |
| `configs/locked/` | frozen inputs: prompt, taxonomy v1, (later: `q_ref.lock`) |
| `figures/` | figure scripts; outputs in `figures/out/` |
| `tests/` | full test suite |

Packages are named `vf_stats` and `vf_datasets` so they do not shadow Python's `statistics` module or Hugging Face `datasets`.
