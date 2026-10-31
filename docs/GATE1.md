# Gate 1: Mathematics. Verdict: **PASS** (2026-11-01)

**Scope:** the idealized model (master plan §7.2) and its numerical checks.

**Reproduce:**
- `uv run pytest`
- `uv run python -m simulations.sim_replicator`
- `uv run python -m figures.make_fig3`

## 1. What is proved (analytical, under assumptions A1–A5)

| Result | Statement | Numerical check (`tests/test_sim_replicator.py`) |
|---|---|---|
| Prop. 1 | The error-mode distribution evolves as the exponential tilt $q_k(t)\propto q_k(0)e^{f_kt}$, independent of τ and A(0) | tilt vs ODE, ≤1e-10, two values of τ |
| Prop. 2 | $\operatorname{logit}A(t)=\operatorname{logit}A(0)+\tau t-\log\sum_k q_k(0)e^{f_kt}$ | 100 random cases, max error ≤1e-10 |
| Prop. 3 | $dF/dt=\operatorname{Var}_{q(t)}(f)$; at $t=0$ this equals $V$ | finite difference, ≤1e-6 |
| Cor. 3a | At matched F̄ and τ, $\Delta\operatorname{logit}A=-\Delta V\,t^2/2+O(t^3)$ | ratio → 1 as t → 0 (error <1e-3 at t=1e-3) |
| Prop. 4 | $f_{\max}>\tau$ gives collapse with a peak at $K'(t^\*)=\tau$; $f_{\max}<\tau$ gives $A\to1$ with logit delay $-\log q_{\arg\max}$ | peak location, both limits, delay to 1e-9 |
| Prop. 5 | A positive common time change preserves orbits | by construction (λ absorbed in $\tilde t$) |
| Prop. 6 | The KL-regularised optimum is the closed form at $t=1/\beta$ (Gibbs) | long-run KL flow vs Gibbs, ≤1e-8 |
| Prop. 7 | Vanilla softmax PG: $dF/dt=P_E\sum_kq_k^2(f_k-F)(f_k-\bar\rho)$ | one-step Euler, rel. 1e-4 |
| — | Permutation invariance (joint relabelling of modes) | property-based, 60 cases |

All 21 Phase-1 tests pass. These are standard replicator-dynamics and Gibbs results applied to verifier error. They are cited as such and are not claimed as new mathematics.

## 2. What is only simulated (tabular, exact gradients, no sampling noise)

Sweep (`figures/out/sim_replicator/`):
- 180 matched families: M ∈ {3,5,8} × Dirichlet α ∈ {0.3,1,5} × F̄ ∈ {0.2,0.4} × τ ∈ {0.8,1.0} × 5 draws, each with 40 configurations;
- 3 update geometries × 4 horizons.

Outcome: logit of mean accuracy over [T/2, T]. Metric: leave-one-config-out MSE ÷ Var(y), median over families. Lower is better; 1 means no information.

| Geometry | T | V | f_max | f_max+V | V_q2 | CGF | best |
|---|---|---|---|---|---|---|---|
| natural PG | 0.5 | **0.000** | 0.381 | 0.000 | 0.045 | 0.000 | CGF ≈ V |
| natural PG | 2 | 0.007 | 0.353 | 0.006 | 0.066 | **0.000** | CGF |
| natural PG | 8 | 0.076 | 0.248 | 0.040 | 0.176 | **0.001** | CGF |
| natural PG | 32 | 0.227 | 0.076 | 0.041 | 0.367 | **0.003** | CGF |
| vanilla PG | 0.5 | 0.893 | 0.808 | **0.783** | 1.010 | 0.790 | none good |
| vanilla PG | 2 | 0.980 | 0.890 | **0.787** | 0.988 | 0.848 | none good |
| vanilla PG | 8 | 0.408 | 0.859 | **0.282** | 0.290 | 0.419 | f_max+V ≈ V_q2 |
| vanilla PG | 32 | 0.090 | 0.449 | 0.098 | 0.126 | **0.092** | V ≈ CGF |
| Adam | 0.5 | 0.434 | 0.603 | 0.399 | 0.486 | **0.396** | CGF ≈ f_max+V |
| Adam | 2 | 0.390 | 0.598 | 0.354 | 0.459 | **0.305** | CGF |
| Adam | 8 | 0.527 | 0.697 | 0.524 | 0.690 | **0.366** | CGF |
| Adam | 32 | 0.854 | 0.854 | 0.813 | 0.960 | 0.673 | κ3 (0.626) |

Figures:
- `figures/out/fig3ab.pdf`: matched verifiers fan out in logit accuracy. The initial FPR drift equals V. The highest-V verifier (f_max = τ) plateaus.
- `figures/out/fig3c_regime_map.pdf`: the table above as lines.
- `figures/out/fig3c_table.csv` and `fig3c_best.csv`: the full numeric view.

## 3. Interpretation (carried into Gate 2)

1. **Under replicator geometry the theory is exact.** V is sufficient at short horizon. Its value decays with horizon, while f_max gains value. The CGF predictor is exact throughout. This is the regime structure that Cor. 3a and Prop. 4 predict.
2. **Under vanilla softmax PG, no aggregate statistic predicts well early.** V becomes useful only at medium and long horizons. The q²-weighted V_q2 helps at medium horizon, as Prop. 7 predicts.
3. **Under Adam with exact gradients,** V carries real information beyond f_max at short and medium horizons (0.35–0.52 vs 0.60–0.70). The CGF is best but needs its horizon fitted.
4. **Therefore V is a meaningful candidate in every geometry tested, but it is never universally sufficient.** The pre-registered competitor set (f_max, CGF, V_q2, κ3) is necessary, not decorative.

## 4. Limits of Gate 1

- **No sampling noise, no finite groups, no shared parameters, one prompt type.** All of these enter in Gate 2.
- **The "Adam" here uses exact gradients,** so it is close to sign descent. Real Adam sees noisy gradients.
- **Horizons are in effective time.** Their mapping to optimizer steps is unknown until the pilot.
- **CGF horizon fitting:** the CGF predictor's horizon parameter is fitted inside each cross-validation fold, so its advantage is honest but still only one-dimensional.

**Decision:** Gate 1 passes. V remains the primary statistic, and the competitor set is unchanged. Proceed to Gate 2.
