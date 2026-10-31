"""Verifier summary statistics (master plan §7.1).

Single source of truth for every verifier-level predictor, shared by the
simulations and by the final statistical analysis.

Conventions
-----------
q   : weights over *incorrect* classes (designed modes, U, I); sums to 1.
      In analysis this is the frozen census reference distribution q_ref.
f   : per-class false-acceptance probabilities, same order as q, in [0, 1].
tau : correct-answer acceptance probability (TPR).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_ATOL = 1e-9


def _check(q, f):
    q = np.asarray(q, dtype=float)
    f = np.asarray(f, dtype=float)
    if q.shape != f.shape or q.ndim != 1:
        raise ValueError(f"q and f must be 1-D with equal shape, got {q.shape}, {f.shape}")
    if (q < -_ATOL).any() or abs(q.sum() - 1.0) > 1e-8:
        raise ValueError("q must be a probability vector")
    if (f < -_ATOL).any() or (f > 1 + _ATOL).any():
        raise ValueError("f must lie in [0, 1]")
    return q, f


def aggregate_fpr(q, f) -> float:
    """Reference-weighted aggregate false-acceptance rate F̄ = Σ q_k f_k."""
    q, f = _check(q, f)
    return float(q @ f)


def macro_fpr(f) -> float:
    """Unweighted macro average of f (reported, never matched on)."""
    return float(np.mean(f))


def weighted_var(q, f) -> float:
    """V = Σ q_k (f_k - F̄)^2 — the candidate primary heterogeneity statistic."""
    q, f = _check(q, f)
    m = q @ f
    return float(q @ (f - m) ** 2)


def unweighted_var(f) -> float:
    f = np.asarray(f, dtype=float)
    return float(np.var(f))


def kappa3(q, f) -> float:
    """Third central moment of f under q (third cumulant)."""
    q, f = _check(q, f)
    m = q @ f
    return float(q @ (f - m) ** 3)


def v_q2(q, f, rbar: float | None = None) -> float:
    """q²-weighted covariance term governing vanilla-PG drift (Prop. 7, up to the P_E factor).

    Σ q_k² (f_k - F̄)(f_k - r̄). With rbar=None, r̄ is taken as F̄, giving the
    q²-weighted variance Σ q_k² (f_k - F̄)².
    """
    q, f = _check(q, f)
    m = q @ f
    r = m if rbar is None else rbar
    return float((q**2) @ ((f - m) * (f - r)))


def fmax(f) -> float:
    return float(np.max(f))


def trigger_freq(q, f) -> float:
    """Initial mass of the most-accepted mode (ties: summed mass of all maximisers)."""
    q, f = _check(q, f)
    return float(q[np.isclose(f, f.max(), atol=1e-12)].sum())


def youden(tau: float, q, f) -> float:
    """J = τ - F̄ (net discriminative power at the reference distribution)."""
    return float(tau - aggregate_fpr(q, f))


def j_worst(tau: float, f) -> float:
    """τ - f_max: conditional advantage of correct over the most-accepted mode."""
    return float(tau - fmax(f))


def iqr_weighted(q, f) -> float:
    """Interquartile range of f under the discrete distribution q."""
    q, f = _check(q, f)
    order = np.argsort(f)
    cdf = np.cumsum(q[order])
    fs = f[order]
    lo = fs[np.searchsorted(cdf, 0.25 - 1e-12)]
    hi = fs[np.searchsorted(cdf, 0.75 - 1e-12)]
    return float(hi - lo)


def acceptance_entropy(q, f) -> float:
    """Entropy of the false-acceptance mass distribution w_k ∝ q_k f_k (concentration).

    Low entropy = false positives concentrated on few modes. Returns 0 if F̄ = 0.
    """
    q, f = _check(q, f)
    w = q * f
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w[w > 0] / s
    return float(-(w * np.log(w)).sum())


def cgf(q, f, s):
    """K(s) = log Σ q_k exp(s f_k); vectorised over s."""
    q, f = _check(q, f)
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore"):
        logq = np.log(q)
    return logsumexp(logq[None, :] + np.multiply.outer(s, f).reshape(-1, f.size), axis=1).reshape(s.shape)


def tilted(q, f, s):
    """Exponentially tilted distribution q_k e^{s f_k} / Σ — equals q(s) under replicator (Prop. 1)."""
    q, f = _check(q, f)
    with np.errstate(divide="ignore"):
        a = np.log(q) + s * f
    a = a - logsumexp(a)
    return np.exp(a)


def cgf_prime(q, f, s):
    """K'(s) = effective FPR F(s) under the tilted distribution."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.array([tilted(q, f, si) @ np.asarray(f, float) for si in s])
    return out.reshape(np.shape(s)) if out.size > 1 else float(out[0])


def cgf_second(q, f, s):
    """K''(s) = Var of f under the tilted distribution."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    f = np.asarray(f, float)
    out = []
    for si in s:
        w = tilted(q, f, si)
        m = w @ f
        out.append(w @ (f - m) ** 2)
    out = np.array(out)
    return out if out.size > 1 else float(out[0])


def cgf_predictor(tau: float, q, f, s: float) -> float:
    """Z(s) = τ s - K(s): the CGF plug-in predictor of logit-accuracy gain at horizon s."""
    return float(tau * s - cgf(q, f, s))


def all_predictors(tau: float, q, f, s: float | None = None) -> dict:
    """Every pre-registered verifier statistic (§13/§15). s is the CGF horizon, if given."""
    q, f = _check(q, f)
    out = {
        "FPR": aggregate_fpr(q, f),
        "macroFPR": macro_fpr(f),
        "TPR": float(tau),
        "J": youden(tau, q, f),
        "J_worst": j_worst(tau, f),
        "fmax": fmax(f),
        "trigger": trigger_freq(q, f),
        "V": weighted_var(q, f),
        "V_unw": unweighted_var(f),
        "V_q2": v_q2(q, f),
        "kappa3": kappa3(q, f),
        "IQR": iqr_weighted(q, f),
        "H_accept": acceptance_entropy(q, f),
        "max_qf": float(np.max(q * f)),
    }
    if s is not None:
        out["Z"] = cgf_predictor(tau, q, f, s)
    return out
