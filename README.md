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
| 4 | Cluster + census | Next; first GPU use |

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

## Layout

| Path | Contents |
|---|---|
| `verifiers/` | `stats.py` (every verifier statistic), `matched_verifiers.py` (exact matched configs), `noisy_verifier.py` (stochastic verifier + TRL reward fn) |
| `error_modes/` | answer parser and deterministic taxonomy v1 |
| `vf_datasets/` | addition prompts with disjoint, hashed splits |
| `simulations/` | idealized replicator model; finite-G GRPO engine; Gate 2 driver |
| `vf_stats/` | leave-one-config-out predictor comparison |
| `figures/` | figure scripts; outputs in `figures/out/` |
| `tests/` | full test suite |

Packages are named `vf_stats` and `vf_datasets` so they do not shadow Python's `statistics` module or Hugging Face `datasets`.
