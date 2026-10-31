# Gate 2: Simulation. Verdict: **PASS** (2026-11-01)

**Question:** under finite-group GRPO with stochastic verifiers, heterogeneous prompts, shared parameters and Adam, does the base-weighted variance V add held-out predictive information beyond the worst-mode rate f_max, at exactly matched F̄ and τ?

**Reproduce:**
- `uv run python -m simulations.sim_shared_policy`, about 2 minutes on a laptop CPU
- `uv run python -m figures.make_gate2`

**Outputs:**
- `figures/out/sim_shared_policy/{summary.csv, predictor_comparison.csv.gz, outcomes.csv.gz, params.json}`
- `figures/out/fig_gate2.pdf`

## Design

| Element | Setting |
|---|---|
| Classes | C + 5 designed modes (ECHO, CARRY, TRUNC, TRANS, DIGIT) + U + I |
| Base model ("family") | 96 prompts. Per-prompt accuracy ~ Beta(mean 0.4). Per-prompt error-mode mix ~ Dirichlet(20·q). 6 families, each with its own q. |
| Census | q_ref = pooled error-mode distribution of the population |
| Verifiers | exact matched grid from `verifiers/matched_verifiers.py`: F̄ = 0.3, τ = 0.9 (matched to ≤1e-12); V levels = {0, .25, .5, .75, 1} of the feasible range; f_max ∈ {free, 0.6, 0.95}; f_max mode drawn at random; 2 configs per cell + homogeneous control ≈ 31 configs per family |
| Training | GRPO, G = 8, 16 prompts/step, 300 steps, std-normalised advantages (except the Dr.GRPO setting). Common prompt batches across configs. 2 seeds per config, seed-averaged (configuration = unit). |
| Settings | tabular {SGD, Adam}; linear shared {SGD, Adam, Adam + mode coupling κ = 0.6, Adam + L2-to-init (KL proxy), Adam without std scaling}; MLP shared {Adam, Adam + coupling} |
| Calibration | learning rates chosen so the homogeneous verifier goes 0.40 → about 0.87–0.97 by step 300 |
| Outcome | logit of mean eval accuracy over [T/2, T], at T ∈ {75, 150, 300} |
| Metric | leave-one-config-out MSE ÷ Var(y); paired family-level differences with a 2,000-sample bootstrap CI over families |

## Results

**Fig. (a): Δ = LOCO-MSE(f_max + V) − LOCO-MSE(f_max), in units of Var(y).**
- Δ < 0 with the 95 % CI excluding 0 in **27/27** setting × horizon cells.
- Range −0.085 (MLP-Adam, T = 300) to −0.685 (tabular-Adam, T = 75).
- The standardised coefficient of V given f_max is **negative in 100 % of families in every cell**: more dispersion means lower accuracy at matched rates.

**Fig. (b): the CGF plug-in vs f_max + V.**
- At T = 75, CGF ≈ f_max + V (differences in [−0.06, +0.03]).
- At T = 150 and T = 300, the CGF is better in 17/18 cells. Typical improvement is 0.04–0.10 of Var(y), with CIs mostly excluding 0.

Median relative LOCO MSE (selected):

| Setting | T | V | f_max | f_max+V | CGF |
|---|---|---|---|---|---|
| linear-adam | 75 | 0.115 | 0.610 | 0.119 | 0.124 |
| linear-adam | 300 | 0.220 | 0.324 | 0.153 | 0.060 |
| mlp-adam | 75 | 0.107 | 0.583 | 0.105 | 0.099 |
| mlp-adam | 300 | 0.389 | **0.171** | 0.132 | 0.066 |
| mlp-adam-coupled | 300 | 0.221 | 0.651 | 0.236 | 0.181 |
| linear-adam-kl | 300 | 0.249 | 0.647 | 0.256 | 0.177 |
| tabular-sgd | 75 | 0.159 | 0.760 | 0.145 | 0.163 |

## Interpretation

1. **Regime structure survives shared parameters and Adam.** V dominates early. f_max gains value late, most strongly for the flexible MLP policy, where f_max alone reaches 0.17 at T = 300. The CGF combines the two. Mode coupling and the KL proxy keep the policy in the "V regime" for longer.
2. **The vanilla-PG pessimism of Gate 1 does not carry over to GRPO.** With exact gradients, vanilla softmax PG made V uninformative at short horizon. Finite-group GRPO with per-group normalisation and stochastic rewards does not show this: tabular-SGD still gets Δ ≈ −0.6. Group normalisation rescales per-prompt updates toward the replicator clock (Prop. 5); this is plausible but not proved.
3. **Seed noise is small relative to between-config spread** (seed SD 0.01–0.15 vs config SD 0.04–0.96 logit units). In these simulations the configuration is clearly the right unit.

## Limits (why this is not evidence about LLMs)

- **Classes are atomic actions.** There are no tokens, no lengths, and no new error modes appear (B9, B10 are absent).
- **The census is exact here** (q_ref = the true pooled distribution). With a real census, estimation error enters (§11).
- **Effect sizes depend on the design.** The grid spans the full feasible V range at F̄ = 0.3. A real experiment with a narrower grid or noisier outcomes will have less power. The pilot (Gate 5) must re-estimate effect size and noise.
- **Short horizons only.** 300 simulated steps with calibrated learning rates; the mapping to LLM steps is unknown.

## Decisions

- **Gate 2 passes.** The GPU study is worth running.
- **V stays the primary statistic (H1).** No switch to V_q2: it was never better than V in any cell.
- **The CGF predictor stays as H3.** It is expected to matter at longer horizons, so the confirmatory horizon T must be set from the pilot *before* the grid is run (rule §13).
- **Power planning input:** the target analysis contrast is f_max + V vs f_max. In simulation it reduced error by 9–69 % of Var(y).
