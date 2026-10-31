"""Leave-one-configuration-out predictor comparison (master plan §13, §15).

Used by the simulations (Gates 1-2) and, unchanged, by the confirmatory analysis.
The configuration is the unit: one outcome per configuration (seed-averaged upstream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from verifiers import stats as vs


PREDICTOR_SETS = {
    "V": ["V"],
    "fmax": ["fmax"],
    "trigger": ["trigger"],
    "V_q2": ["V_q2"],
    "kappa3": ["kappa3"],
    "fmax+V": ["fmax", "V"],
    "fmax+trigger": ["fmax", "trigger"],
    "max_qf": ["max_qf"],
}


def loo_parts(X: np.ndarray, y: np.ndarray):
    """OLS with intercept. Returns (LOO residuals, LOO training SSEs) via exact hat-matrix identities:

    e_(i) = e_i / (1 - h_ii)   and   SSE_(i) = SSE - e_i² / (1 - h_ii).
    """
    n = len(y)
    Xa = np.c_[np.ones(n), X]
    Xa = Xa / np.maximum(np.abs(Xa).max(0), 1e-300)  # column scaling; does not change fits
    U, s, _ = np.linalg.svd(Xa, full_matrices=False)
    U = U[:, s > s.max() * 1e-10]
    h = np.clip((U**2).sum(1), 0, 1 - 1e-12)
    e = y - U @ (U.T @ y)
    loo = e / (1 - h)
    sse_train = (e**2).sum() - e**2 / (1 - h)
    return loo, sse_train


def loco_mse(X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(loo_parts(X, y)[0] ** 2))


def loco_cgf_mse(tau, Q, F, y, s_grid) -> tuple[float, float]:
    """CGF plug-in Z(s) = τ s - K(s) with s chosen inside each training fold (honest CV)."""
    n = len(y)
    Z = np.array([[vs.cgf_predictor(tau, Q[i], F[i], s) for s in s_grid] for i in range(n)])  # (n, S)
    loo = np.empty((n, len(s_grid)))
    sse = np.empty((n, len(s_grid)))
    for j in range(len(s_grid)):
        loo[:, j], sse[:, j] = loo_parts(Z[:, [j]], y)
    j_star = sse.argmin(1)  # s minimising training-fold SSE, per held-out config
    err = loo[np.arange(n), j_star]
    return float(np.mean(err**2)), float(np.median(np.asarray(s_grid)[j_star]))


def forecast_eval(tau, Q, F, y, s_grid=None) -> pd.DataFrame:
    """LOCO-CV MSE and Spearman ρ for every predictor set on one matched family."""
    if s_grid is None:
        s_grid = np.geomspace(0.25, 64, 25)
    preds = pd.DataFrame([vs.all_predictors(tau, Q[i], F[i]) for i in range(len(y))])
    var_y = float(np.var(y))
    rows = []
    for name, cols in PREDICTOR_SETS.items():
        X = preds[cols].to_numpy()
        rho = spearmanr(preds[cols[-1]], y).statistic if len(cols) == 1 else np.nan
        rows.append({"predictor": name, "loco_mse": loco_mse(X, y), "spearman": rho})
    mse, s_med = loco_cgf_mse(tau, Q, F, y, s_grid)
    z_med = np.array([vs.cgf_predictor(tau, Q[i], F[i], s_med) for i in range(len(y))])
    rows.append({"predictor": "CGF", "loco_mse": mse, "spearman": spearmanr(z_med, y).statistic, "s_chosen": s_med})
    rows.append({"predictor": "null(intercept)", "loco_mse": loco_mse(np.zeros((len(y), 0)), y), "spearman": np.nan})
    df = pd.DataFrame(rows)
    df["rel_mse"] = df["loco_mse"] / var_y if var_y > 0 else np.nan
    return df
