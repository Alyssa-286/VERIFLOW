# VERIFLOW Roadmap

The working index of the project. The detailed plan is in [VERIFLOW_MASTER_PLAN.md](VERIFLOW_MASTER_PLAN.md); section numbers (§) below refer to it.

**Rules:**
- A phase starts only when the previous phase's exit gate is passed and recorded in [DECISIONS.md](DECISIONS.md).
- No GPU time is used before Phase 4.
- ErrorEcho always has priority on the cluster.

| Phase | Name | Compute | Target weeks (2026–27) | Exit gate | Status |
|---|---|---|---|---|---|
| 0 | Planning and literature audit | — | Oct 31 | Gate 0: novelty (modified question survives) | ✅ done |
| 1 | Foundations and idealized theory | CPU | Nov 2–15 | **Gate 1: mathematics** | ✅ PASS 2026-11-01 ([GATE1.md](GATE1.md)) |
| 2 | Matched verifiers and realistic-dynamics simulation | CPU | Nov 9–22 | **Gate 2: simulation** | ✅ PASS 2026-11-01 ([GATE2.md](GATE2.md)) |
| 3 | Construct tooling (task, taxonomy, verifier) | CPU | Nov 23–29 | Unit tests and synthetic recovery ≥ 99 % | ✅ done 2026-11-01 (D6; CARRY/DIGIT overlap flagged) |
| 4 | Cluster setup and base-policy census | CPU + first GPU | Nov 30–Dec 6 | **Gate 3: construct** | ✅ done 2026-11-03; ✗ **Gate 3: STOP** (§9.3 kill rule after the full ladder incl. subtraction); q_ref not frozen; Phase 5 blocked ([GATE3.md](GATE3.md), D14) |
| 5 | Training stack and throughput | GPU | Dec 7–13 | **Gate 4: feasibility** | ☐ |
| 6 | Pilot | GPU | Dec 14–27 | **Gate 5: empirical engagement** | ☐ |
| 7 | Pre-registration and confirmatory grid | GPU | Dec 28–Jan 31 | Grid complete, manifests clean | ☐ |
| 8 | Analysis, paper and release | CPU | Feb 1–21 | Draft + reproducibility bundle | ☐ |
| A | Applications track (parallel) | — | continuous | Summer@EPFL by **Nov 29 2026**; others as verified | ☐ |

---

## Phase 1: Foundations and idealized theory (CPU)

**Goal:** a correct, tested implementation of the idealized replicator model (§7 Props 1–6), plus a first look at which verifier statistic leads in which regime.

**Deliverables:**
- repo skeleton, `pyproject.toml`, `uv.lock`, git repo
- `verifiers/stats.py`: aggregate FPR, weighted and unweighted variance, κ3, CGF and its derivatives, J, J_worst, f_max, trigger frequency, V_q2, all predictors
- `simulations/sim_replicator.py`: closed form, ODE integrator, Gibbs/KL optimum, peak time, matched-family generator (simple version; the constrained generator comes in Phase 2), sweep and forecast evaluation
- `tests/test_stats.py`, `tests/test_sim_replicator.py`: the 8 tests from §19.1
- `figures/make_fig3.py` → `figures/out/fig3a.pdf`, `fig3b.pdf`, `fig3c_regime_map.pdf`
- `docs/GATE1.md`

**Exit (Gate 1):** all tests green; numerical agreement ≤ 1e-10; regime map produced; GATE1.md states what is proved and what is only simulated.

## Phase 2: Matched verifiers and realistic-dynamics simulation (CPU)

**Goal:** decide whether the GPU study is worth running.

**Deliverables:**
- `verifiers/matched_verifiers.py`: exact constrained generator (fixed F̄, V, f_max, with U and I modes), de-confounding (V, f_max) grid, mode-identity permutations; `tests/test_matching.py`
- `simulations/sim_grpo_tabular.py`: finite-G GRPO with a Bernoulli verifier, std normalisation on and off, prompt heterogeneity (B1–B3, B7)
- `simulations/sim_shared_policy.py`: linear-softmax and MLP shared-parameter policies; SGD vs Adam; mode-coupling strength; L2-to-init (B6, B8)
- `vf_stats/predictor_comparison.py`: leave-one-config-out predictor comparison
- `figures/make_gate2.py` → `figures/out/fig_gate2.pdf`
- `docs/GATE2.md` with a regime map per geometry

**Exit (Gate 2):** in Adam-trained shared-parameter sims, V, or a pre-declared competitor such as V_q2 or the CGF, adds held-out predictive information beyond f_max. If nothing does, apply the §27 kill rule and write up a simulation note.

**Primary statistic locked here:** V stays primary unless Gate 2 shows it is dominated. Any switch is logged in DECISIONS.md before any GPU data exist.

## Phase 3: Construct tooling (CPU)

- `error_modes/parse_answer.py`, `error_modes/classify_errors.py`: taxonomy v1, multi-hot output
- `vf_datasets/arithmetic.py`: disjoint splits, hashing and leakage checks (merged; there is no separate splits.py)
- `verifiers/noisy_verifier.py`: Philox stream, decision log
- synthetic recovery test ≥ 99 % per mode; realised-rate, determinism and stream-independence tests

**Exit:** all tests green.

## Phase 4: Cluster setup and census (first GPU use)

- ✅ `docs/CLUSTER_POLICY.md` — §16.2 read-only survey done 2026-11-02 (2 GPUs, MaxTime infinite, no accounting plugin; ErrorEcho holds 1 GPU → `%1` throttle)
- ✅ `experiments/census_core.py` + `experiments/base_mode_census.py` (hf + deterministic mock backends) and `experiments/check_construct_gate.py` (§9.3 evaluator; freezes `q_ref.lock` only for real, gate-passing censuses)
- ✅ manifest machinery (`scripts/manifest_lib.py`, run_id/run_spec/write/finalize), `veriflow-preflight` / `veriflow-gpucheck` (`scripts/preflight.py`, `scripts/gpucheck.py`, `bin/` wrappers, `slurm/*.sbatch`)
- ✅ pinned models `configs/model/qwen0p5b.yaml`, `qwen1p5b.yaml`; `configs/locked/prompt.lock`, `taxonomy.lock`; isolated `gpu` dependency group (torch/transformers/accelerate)
- ✅ on the cluster: `~/VERIFLOW` + `uv sync`, preflight (3590), gpucheck (3593), census n=5..10 (3594–3597, 3600, 3602)
- ✅ prompt v2 census n=5..10 (3610–3615): also fails (D10)
- ✅ taxonomy v2 census n=5..10 (3620–3625): fails (D11, D12)
- ✅ subtraction census n=5..10 (3627–3632, D13): fails
- ✗ Gate 3 verdict: STOP (§9.3 kill rule, D14). Continuing requires a new pre-registered design

**Exit (Gate 3):** construct criteria §9.3 (coverage ≥ 70 %, ≥ 3 modes at ≥ 5 % mass, ambiguity ≤ 10 %, split-half TV ≤ 0.03, manual audit). Automation ready; the census run itself decides.

## Phase 5: Training stack and feasibility

- `training/train_grpo.py` (TRL 1.17.x, all GRPOConfig fields pinned), callbacks, SIGUSR1 checkpointing, resume
- `veriflow-smoke` (1 step and 10 steps, resume test), `veriflow-bench` (vLLM colocate vs HF generate)

**Exit (Gate 4):** measured cost per run; the design tier (minimum, standard or extended) is chosen per §17.

## Phase 6: Pilot

- LR pilot (homogeneous verifier), which fixes LR and horizon T
- heterogeneity pilot: 4 configs × 3 seeds (exploratory)

**Exit (Gate 5):** modes are differentially amplified, i.e. the manipulation reaches the policy (§27).

## Phase 7: Pre-registration and confirmatory grid

- `vf_stats/power_sim.py`, then `PREREGISTRATION.md` registered on OSF
- generate configs and split them into training and held-out sets
- run the grid (Slurm array, `%1`)

**Exit:** all runs complete, or failures handled by the pre-registered rules.

## Phase 8: Analysis, paper and release

- tests of H1–H3, Figures 4–7, paper draft, Zenodo bundle
- optional ICLR 2027 workshop note (theory + simulation, around Feb 1); full paper aimed at COLM 2027 or TMLR

## Applications track (parallel)

- **Summer@EPFL deadline: 29 Nov 2026** [verified]. Attach the Gate 1/2 report.
- Confirm on official pages: DAAD WISE (may be closed), ETH SSRF, MBZUAI UGRIP, MPI-IS. Record them in `docs/APPLICATIONS.md`.
