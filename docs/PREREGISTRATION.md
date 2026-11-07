# VERIFLOW pre-registration (final, ready for OSF)

> **Status:** finalised 2026-11-07 after the pilot (exploratory; D21–D25) and before
> any confirmatory run. Every pilot-derived value below was set by a rule logged
> before the pilot data it uses. The configurations, held-out split and run order
> are already generated and hashed (D26). No confirmatory run has been started.
> Repository: `github.com/Alyssa-286/VERIFLOW`, branch `phase4/hpc-ready`; the frozen
> commit of this document is given on the OSF form.

## 1. Title and question

*Same Error Rate, Different Fate? Error-Mode Structure of Verifier False Positives
and the Dynamics of RLVR.*

Among programmatic verifiers matched exactly on the aggregate false-acceptance
rate (weighted by the base policy's error-mode distribution) and on the
correct-answer acceptance rate, does the base-weighted variance V of the mode-specific
false-acceptance probabilities predict the true held-out accuracy reached by
GRPO-trained Qwen2.5-0.5B-Instruct, after controlling for the worst-mode rate f_max?

## 2. What exists before data collection (with provenance)

| Item | Value | Source |
|---|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct @ `7ae557604adf67be50417f59c2c2f167def9a775` | `configs/model/qwen0p5b.yaml` |
| Task | two-operand addition, n = 5 digits | §9.1, Gate 3′ (D16) |
| Prompt | v2 answer-only, `prompt.lock` `918a83d3…` | D9 |
| Parser | taxonomy-v2 parser (bare integer accepted) | D11 |
| Error modes | taxonomy v3 partition: LEN (incl. merged NEG), MULTI, ONE_HI, ONE_LO; I = unparseable | Amendment 1 (D15) |
| q_ref | census `n5_v3_s20261104_j3640`: LEN 0.162, MULTI 0.420, ONE_HI 0.181, ONE_LO 0.206, I 0.030 | `q_ref.lock` sha256 `2c34f6d2…d5cb` (frozen 2026-11-05, D20, D22) |
| Splits | seed 20261104; train 50,000 / eval 1,000 / eval-mode 1,000, disjoint | Amendment 1 |
| Matched rates | F̄* = 0.3, τ* = 0.9 | D18 |
| Training | TRL 1.17.1 GRPO: dapo loss, β = 0, group reward scaling, G = 8, 32 prompts/step, neutral sampling (T 1, top_p 1), full fine-tune with AdamW, constant LR, fp32 + bf16 autocast | `configs/base.json`, D17 |
| Learning rate | **5e-7**, by the D19 rule on the LR pilot (array 3707; window means 0.883 / 0.869 / 0.770 for 5e-7 / 1e-6 / 3e-6) | `configs/pilot/selection.json`, D21 |
| Horizon T | **100 steps** by the D19 rule (§13; the 100-step floor binds); outcome window [50, 100] | same, D21 |
| Verifier | per-mode Bernoulli acceptance; SHA-256 PRF keyed on (verifier seed 20261105, run seed, step, prompt, sample); every decision logged | §10.1, D4 |

**Disclosed protocol changes before this registration:**
- the answer-only prompt (D9);
- bare-integer parsing (D11);
- the outcome-partition taxonomy and a fresh split (Amendment 1).

All three were made after exploratory censuses, and are reported as such.

**Other disclosures:**
- The 200-row construct audit (Gate 3′) was labelled by an AI assistant (Claude)
  from the written rules and reviewed by the researcher (D20, D22). Disagreement
  with the classifier was 1/200.
- Evaluation cadence is every 10 steps (the plan said 50), fixed before the
  heterogeneity pilot (D21).
- The trainer re-seeds the global torch RNG at each evaluation
  (seed = f(run seed, step)). A run is therefore a deterministic function of
  (config, seed, eval cadence). The cadence is identical across all runs (D21).
- Confirmatory runs use 1 L40 and 2 CPU cores per run (D24). The CPU count
  affects speed only. The
mechanism taxonomy reached its pre-registered stop rule (D14); that negative
result is reported too.

## 3. Design

- **Matched arm.** **R = 36** configurations at F̄ = 0.3 and τ = 0.9 (power rule,
  §7 below). They were generated once from `q_ref.lock` (generator seed 20261108)
  on a de-confounding grid of V fraction {0, .25, .5, .75, 1} × f_max {free, 0.6,
  0.95}, with the mode carrying f_max randomised (§10.2). They comprise the
  homogeneous control plus a balanced cycle over the 14 feasible cells
  (`configs/grid/matched/`).
- **Auxiliary arm.** **12** configurations with F̄ ∈ {0.15, 0.45}, 6 each, V
  fraction 0–0.9, f_max free (extended design, §14; `configs/grid/auxiliary/`).
- **Seeds.** 2 per configuration. The configuration is the unit of analysis.
- **Held-out configurations.** 13 of the 36 matched configurations, by stratified
  random assignment over grid cells (seed 20261109); the homogeneous control is
  always in training. They are used only for H2/H3 (`configs/grid/split.json`).
- **Design size.** Extended: 48 configurations × 2 seeds = **96 runs**, executed in a
  seeded random order (seed 20261110, `configs/grid/runs_grid.csv`), at most 2 at
  a time. Measured cost ≈ 3 min per run, ≈ 4.5 GPU-h in total. The digest of all
  files in `configs/grid/` is `39815e7b…52a7` (D26).
- **Replication.** **No confirmatory Qwen2.5-1.5B arm.** Its census failed the
  split-half TV criterion inside the accuracy window (D16: n = 8 TV 0.039, n = 9
  TV 0.050, threshold 0.03), so no valid q_ref exists for it. No new evidence
  resolves this. Any 1.5B result would be exploratory and labelled as such.

## 4. Outcome

**Primary outcome, one scalar per run:**

    Y = logit(mean greedy held-out accuracy over the eval steps in [T/2, T])

Evaluation uses the 1,000 eval prompts, every **10** steps (D21), so the window [50, 100] has 6 evaluation points. Accuracy is clipped to
[1/2000, 1 − 1/2000] before the logit. The configuration outcome Ȳ is the mean
over its seeds.

**Secondary outcomes (§14):**
- final accuracy at T;
- per-mode trajectories q_k(t) from the eval-mode census;
- growth of the most-accepted mode;
- the hacking indicator H, with threshold **0.029** = 2.5 × the pilot's single-eval
  binomial SE at accuracy 0.838 (D25);
- KL(q(T) ‖ q(0));
- the I/LEN fraction;
- training instability.

## 5. Hypotheses and tests

**Claim type: estimation with confidence intervals, declared in advance.** The power
rule (§7) found < 50 % power for H1 even at R = 36, so this study does not claim a
confirmatory test of H1–H3. The pre-specified estimates and 95 % CIs are the primary
results: b_V and b_f with HC3 CIs, and the H2/H3 paired MSE differences with bootstrap
CIs. The statistics below, with two-sided Holm-adjusted p-values (α = 0.05), are
computed and reported in full, but a p < α is not presented as a confirmatory
finding. Code: `vf_stats/confirmatory.py`, tested on synthetic data
(`tests/test_confirmatory.py`).

- **H1.** In the matched arm, Ȳ_r = b0 + bV·V_r + bf·f_max,r + ε. Test bV with
  HC3 standard errors (t-test, df = R − 3).
- **H2.** On held-out configurations, compare {f_max, V} with {f_max} alone,
  each fitted on training configurations. The statistic is the paired
  difference in squared error, with a 10,000-resample configuration-level
  bootstrap CI and two-sided p. Leave-one-configuration-out MSE on all matched
  configurations is also reported.
- **H3.** The CGF plug-in Z(s) = τs − K(s), with s chosen by
  leave-one-configuration-out on the training configurations from a fixed
  25-point log grid over [0.25, 64], against {f_max, V}. Same paired test.

**Secondary analysis (all arms).** A hierarchical model with predictors F̄, J,
J_worst, f_max, trigger frequency, V, κ3, V_q2 and conditional advantage, plus
a random configuration intercept. Exploratory analyses are labelled as such.
The ICC between configuration and seed is reported.

## 6. Data handling

- **Infrastructure failure:** resume from the last checkpoint with the same seed;
  at most 2 retries.
- **Divergence** (accuracy < 1 % for 3 consecutive evals): this is an outcome,
  not missing data; Y is computed from the evaluations that exist.
- **No exclusion** for "looking wrong". **No checkpoint selection.** No interim
  efficacy looks.

## 7. Pilot results and power analysis

**Pilot (exploratory, excluded from all confirmatory analyses).**
- LR pilot: array 3707, 4 runs × 800 steps.
- Heterogeneity pilot: array 4020, 4 configs × 3 seeds at lr 5e-7, T 100; 12/12 runs
  validated (`results/pilot/validate_het.json`).
- **Gate 5 (rule D21) passed** (`configs/pilot/gate5.json`). The high-V verifier
  amplified its most-accepted mode beyond the homogeneous control (D = +0.23, SE 0.09)
  and raised the realised false-acceptance rate on wrong rollouts (+0.25, SE 0.05).
  The mechanism reaches the policy within T.

**Power (method D23, `vf_stats/power_sim.py`; results `configs/pilot/power.json`).**
- Simulated H1 OLS with R configs × 2 seeds at α = 0.05/3.
- Noise: seed SD 0.246 (pooled from the pilot); config SD 0.246 (conservative floor).
- Effect from the Gate 2 shared-policy simulation, expressed as a share of the
  homogeneous logit gain (ρ = −0.41, median over 54 setting × family cells) and
  rescaled to the pilot's gain (2.02) and V range (0.172): b_V = −4.78.
- Power at R = 12 / 24 / 36: **0.08 / 0.25 / 0.43** (pessimistic ρ = −0.25: 0.04 /
  0.09 / 0.17).
- Rule: no R reaches 80 %, and power(36) < 50 %, so **R = 36 with the claim
  downgraded to estimation with CIs**, stated here before any confirmatory data
  (§5). Seeds stay at 2 per configuration as fixed in §14.
