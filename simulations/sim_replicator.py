"""Idealized mode-level learning dynamics (master plan §7.2, §12.1).

Level of claim: analytical theorem / idealized model. Nothing here is a
statement about neural GRPO; see §7.3 for where these dynamics stop applying.

Classes are ordered [C, e_1, ..., e_M]: index 0 is the correct answer, the rest
are incorrect classes with base-policy conditional mass q0 and verifier
false-acceptance probabilities f. The verifier accepts C with probability tau.

Update geometries
-----------------
"npg"  : natural PG / multiplicative weights, dθ_k/dt = ρ_k - ρ̄  (replicator; Props 1-6 exact)
"vpg"  : vanilla softmax PG on logits, dθ_k/dt = p_k (ρ_k - ρ̄)     (Prop. 7)
"adam" : Adam on the vanilla-PG gradient with exact (noise-free) gradients.
         With exact gradients Adam is close to sign descent; gradient noise is
         added in Phase 2 (finite-G GRPO), not here.

Usage:  uv run python -m simulations.sim_replicator --out figures/out/sim_replicator
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit, logit

from verifiers import stats as vs

# ----------------------------------------------------------------------------
# Exact idealized results
# ----------------------------------------------------------------------------


def replicator_rhs(p: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """dp_k/dt = p_k (ρ_k - ρ̄)."""
    return p * (rho - p @ rho)


def initial_state(A0: float, q0) -> np.ndarray:
    q0 = np.asarray(q0, float)
    return np.r_[A0, (1.0 - A0) * q0]


def integrate_replicator(A0, q0, tau, f, t_eval, rtol=1e-12, atol=1e-14) -> np.ndarray:
    """Numerically integrate the replicator ODE; returns p(t) with shape (len(t_eval), M+1)."""
    rho = np.r_[tau, np.asarray(f, float)]
    t_eval = np.asarray(t_eval, float)
    sol = solve_ivp(
        lambda t, p: replicator_rhs(p, rho),
        (0.0, float(t_eval.max())),
        initial_state(A0, q0),
        t_eval=t_eval,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def closed_form_logit_accuracy(A0, q0, tau, f, t):
    """Prop. 2: logit A(t) = logit A(0) + τ t - K_0(t)."""
    t = np.asarray(t, float)
    return logit(A0) + tau * t - vs.cgf(q0, f, t)


def closed_form_accuracy(A0, q0, tau, f, t):
    """Prop. 2 on the probability scale."""
    return expit(closed_form_logit_accuracy(A0, q0, tau, f, t))


def effective_fpr(q0, f, t):
    """F(t) = K_0'(t): on-policy aggregate FPR under replicator dynamics (Props 1, 3)."""
    return vs.cgf_prime(q0, f, t)


def peak_time(q0, f, tau) -> float:
    """Prop. 4: the t* solving K'(t*) = τ.

    Returns 0.0 if F̄ ≥ τ (accuracy never rises) and inf if f_max ≤ τ (no interior peak).
    """
    f = np.asarray(f, float)
    if vs.aggregate_fpr(q0, f) >= tau:
        return 0.0
    if f.max() <= tau:
        return float("inf")
    g = lambda t: vs.cgf_prime(q0, f, t) - tau  # noqa: E731  (K' is non-decreasing)
    hi = 1.0
    while g(hi) <= 0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("peak_time bracket failed")
    return float(brentq(g, 0.0, hi, xtol=1e-12, rtol=1e-14))


def gibbs_logit_accuracy(A0, q0, tau, f, beta: float) -> float:
    """Prop. 6: accuracy at the optimum of E[ρ] - β KL(π || π0) = closed form at t = 1/β."""
    return float(closed_form_logit_accuracy(A0, q0, tau, f, 1.0 / beta))


def integrate_kl_npg(A0, q0, tau, f, beta, T, rtol=1e-11, atol=1e-13) -> np.ndarray:
    """Natural PG on the KL-regularised objective, in logits: dθ/dt = ρ - β(θ - θ0).

    Its fixed point is θ0 + ρ/β, i.e. the Gibbs distribution of Prop. 6.
    Returns the final class probabilities.
    """
    rho = np.r_[tau, np.asarray(f, float)]
    th0 = np.log(initial_state(A0, q0))
    sol = solve_ivp(lambda t, th: rho - beta * (th - th0), (0.0, T), th0, method="DOP853", rtol=rtol, atol=atol)
    th = sol.y[:, -1]
    p = np.exp(th - th.max())
    return p / p.sum()


# ----------------------------------------------------------------------------
# Tabular simulation under different update geometries (vectorised over configs)
# ----------------------------------------------------------------------------


def simulate_tabular(
    rule: str,
    A0: float,
    Q0: np.ndarray,
    tau: float,
    F: np.ndarray,
    T: float,
    n_steps: int = 2000,
    adam_scale: float = 0.3,
    record_every: int = 10,
):
    """Simulate N configurations at once.

    Q0, F: arrays (N, M). Returns (times, acc) with acc shape (n_records, N).
    Time is in units where npg equals the replicator clock.
    """
    Q0 = np.atleast_2d(Q0)
    F = np.atleast_2d(F)
    N, M = F.shape
    rho = np.c_[np.full(N, tau), F]
    th = np.log(np.c_[np.full(N, A0), (1 - A0) * Q0])
    h = T / n_steps
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-8
    times, accs = [0.0], [np.full(N, A0)]
    for i in range(n_steps):
        p = np.exp(th - th.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        adv = rho - (p * rho).sum(1, keepdims=True)
        if rule == "npg":
            step = adv
        elif rule == "vpg":
            step = p * adv
        elif rule == "adam":
            g = p * adv
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            step = adam_scale * (m / (1 - b1 ** (i + 1))) / (np.sqrt(v / (1 - b2 ** (i + 1))) + eps)
        else:
            raise ValueError(rule)
        th = th + h * step
        if (i + 1) % record_every == 0:
            p = np.exp(th - th.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            times.append((i + 1) * h)
            accs.append(p[:, 0])
    return np.array(times), np.array(accs)


def window_outcome(times: np.ndarray, acc: np.ndarray, T: float) -> np.ndarray:
    """Primary-outcome analogue (§14): logit of mean accuracy over the window [T/2, T]."""
    w = times >= T / 2 - 1e-12
    return logit(np.clip(acc[w].mean(0), 1e-12, 1 - 1e-12))


# ----------------------------------------------------------------------------
# Matched verifier families (simple version; the exact constrained generator
# with U/I classes and a (V, f_max) grid is Phase 2: verifiers/matched_verifiers.py)
# ----------------------------------------------------------------------------


def random_matched_family(q0, Fbar: float, n: int, rng: np.random.Generator, include_homogeneous=True, max_tries=200_000):
    """Draw n verifier vectors f in [0,1]^M with Σ q0 f = F̄ exactly (shift-and-reject).

    Spread is randomised so the family spans small to large V.
    """
    q0 = np.asarray(q0, float)
    M = q0.size
    out = [np.full(M, Fbar)] if include_homogeneous else []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not generate matched family")
        spread = rng.uniform(0.05, 1.0)
        f = Fbar + spread * (rng.uniform(-1, 1, M))
        f = f - q0 @ f + Fbar
        if (f >= 0).all() and (f <= 1).all():
            out.append(f)
    return np.array(out)


# ----------------------------------------------------------------------------
# Predictor comparison (leave-one-config-out)
# ----------------------------------------------------------------------------

from vf_stats.predictor_comparison import (  # noqa: E402,F401  (re-exported)
    PREDICTOR_SETS,
    forecast_eval,
    loco_cgf_mse as _loco_cgf_mse,
    loco_mse as _loco_mse,
    loo_parts as _loo_parts,
)


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------


@dataclass
class SweepConfig:
    M_values: list = field(default_factory=lambda: [3, 5, 8])
    alpha_values: list = field(default_factory=lambda: [0.3, 1.0, 5.0])  # Dirichlet concentration of q0
    Fbar_values: list = field(default_factory=lambda: [0.2, 0.4])
    tau_values: list = field(default_factory=lambda: [0.8, 1.0])
    horizons: list = field(default_factory=lambda: [0.5, 2.0, 8.0, 32.0])
    rules: list = field(default_factory=lambda: ["npg", "vpg", "adam"])
    n_configs: int = 40
    n_families: int = 5  # independent q0 draws per cell
    A0: float = 0.4
    n_steps: int = 1500
    seed: int = 20261102


def run_sweep(cfg: SweepConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-config outcomes, per-family predictor comparison)."""
    rng = np.random.default_rng(cfg.seed)
    outcome_rows, compare_rows = [], []
    fam_id = 0
    for M in cfg.M_values:
        for alpha in cfg.alpha_values:
            for Fbar in cfg.Fbar_values:
                for tau in cfg.tau_values:
                    for _ in range(cfg.n_families):
                        q0 = rng.dirichlet(np.full(M, alpha))
                        q0 = np.clip(q0, 1e-6, None)
                        q0 /= q0.sum()
                        Fam = random_matched_family(q0, Fbar, cfg.n_configs, rng)
                        Q = np.tile(q0, (len(Fam), 1))
                        for rule in cfg.rules:
                            for T in cfg.horizons:
                                if rule == "npg":  # exact closed form
                                    tt = np.linspace(T / 2, T, 51)
                                    acc = np.array([closed_form_accuracy(cfg.A0, q0, tau, f, tt) for f in Fam]).T
                                    y = window_outcome(tt, acc, T)
                                else:
                                    times, acc = simulate_tabular(rule, cfg.A0, Q, tau, Fam, T, n_steps=cfg.n_steps)
                                    y = window_outcome(times, acc, T)
                                for i in range(len(Fam)):
                                    outcome_rows.append(
                                        dict(family=fam_id, config=i, M=M, alpha=alpha, Fbar=Fbar, tau=tau, rule=rule, T=T, y=y[i],
                                             **vs.all_predictors(tau, q0, Fam[i]))
                                    )
                                cmp = forecast_eval(tau, Q, Fam, y)
                                cmp = cmp.assign(family=fam_id, M=M, alpha=alpha, Fbar=Fbar, tau=tau, rule=rule, T=T, sd_y=float(np.std(y)))
                                compare_rows.append(cmp)
                        fam_id += 1
                    print(f"[sweep] M={M} alpha={alpha} Fbar={Fbar} done ({fam_id} families)", flush=True)
    return pd.DataFrame(outcome_rows), pd.concat(compare_rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="figures/out/sim_replicator")
    ap.add_argument("--quick", action="store_true", help="small sweep for smoke testing")
    args = ap.parse_args(argv)
    cfg = SweepConfig()
    if args.quick:
        cfg = SweepConfig(M_values=[5], alpha_values=[1.0], Fbar_values=[0.3], tau_values=[0.9], n_configs=15, n_families=2, n_steps=500)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    outcomes, compare = run_sweep(cfg)
    outcomes.to_csv(out / "outcomes.csv.gz", index=False)
    compare.to_csv(out / "predictor_comparison.csv.gz", index=False)
    (out / "params.json").write_text(json.dumps(asdict(cfg), indent=2))
    summary = (
        compare.groupby(["rule", "T", "predictor"])["rel_mse"].median().unstack("predictor").round(3)
    )
    summary.to_csv(out / "summary_rel_mse.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
