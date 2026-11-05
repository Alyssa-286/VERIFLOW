"""Confirmatory analysis (master plan §14–§15), written and tested before any confirmatory data.

Unit = verifier configuration. Seeds are averaged within a configuration before any test.

Outcome per run:  Y = logit(mean greedy held-out accuracy over eval steps in [T/2, T]).
                  Accuracies are clipped to [1/(2n), 1 - 1/(2n)] (n = eval-set size) so the logit is
                  finite; this only matters at 0 or 1 accuracy.
H1  matched arm:  OLS  Ȳ_r = b0 + bV·V_r + bf·fmax_r, HC3 SEs, two-sided t-test on bV (df = R - 3).
H2  held-out:     MSE of {fmax, V} vs {fmax}, both fitted on training configs, evaluated on held-out
                  configs; paired squared-error difference with a 10,000-resample config bootstrap CI.
                  Also leave-one-config-out MSE on all matched configs.
H3  held-out:     CGF plug-in Z(s) = τs − K(s) (s chosen by LOCO on training configs, then a linear map
                  Ȳ ~ Z fitted on training configs) vs {fmax, V}; same paired bootstrap.
Holm adjustment over the three p-values (H2/H3: two-sided bootstrap p-values).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from vf_stats.predictor_comparison import loco_mse
from verifiers import stats as vs

S_GRID = np.geomspace(0.25, 64, 25)


# ------------------------------------------------------------------ outcomes

def window_logit(eval_df: pd.DataFrame, T: int, n_eval: int) -> float:
    w = eval_df[(eval_df["step"] >= T / 2) & (eval_df["step"] <= T)]
    if w.empty:
        raise ValueError(f"no eval points in [{T / 2}, {T}]")
    lo = 1.0 / (2 * n_eval)
    a = float(np.clip(w["greedy_accuracy"].mean(), lo, 1 - lo))
    return float(np.log(a / (1 - a)))


def run_outcome(run_dir: Path, T: int) -> dict:
    rows = [json.loads(l) for l in (Path(run_dir) / "eval.jsonl").read_text().splitlines() if l]
    ev = pd.DataFrame(rows).drop_duplicates("step", keep="last")
    man = json.loads((Path(run_dir) / "manifest.json").read_text())
    return {"run": Path(run_dir).name, "config_id": man["verifier_config"]["config_id"],
            "seed": man["args"]["seed"], "Y": window_logit(ev, T, int(ev["n_eval"].iloc[0])),
            "status": man["status"]}


def config_means(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed-average Y per configuration (the unit of analysis)."""
    return runs.groupby("config_id", as_index=False).agg(Y=("Y", "mean"), n_seeds=("Y", "size"))


def icc(runs: pd.DataFrame) -> dict:
    """One-way random-effects ICC(1): share of Y variance between configurations."""
    g = runs.groupby("config_id")["Y"]
    k = g.size().mean()
    msb = (g.size() * (g.mean() - runs["Y"].mean()) ** 2).sum() / (g.ngroups - 1)
    msw = ((runs["Y"] - g.transform("mean")) ** 2).sum() / (len(runs) - g.ngroups)
    return {"icc": float((msb - msw) / (msb + (k - 1) * msw)), "ms_between": float(msb), "ms_within": float(msw)}


# ------------------------------------------------------------------ H1

def ols_hc3(X: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    n = len(y)
    Xa = np.c_[np.ones(n), X]
    XtX_inv = np.linalg.inv(Xa.T @ Xa)
    beta = XtX_inv @ Xa.T @ y
    e = y - Xa @ beta
    h = np.einsum("ij,jk,ik->i", Xa, XtX_inv, Xa)
    meat = Xa.T @ (Xa * (e / (1 - h))[:, None] ** 2)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    df = n - Xa.shape[1]
    t = beta / se
    p = 2 * st.t.sf(np.abs(t), df)
    return pd.DataFrame({"term": ["intercept", *names], "coef": beta, "se_hc3": se, "t": t, "p": p, "df": df})


def h1(cfg: pd.DataFrame) -> dict:
    tab = ols_hc3(cfg[["V", "fmax"]].to_numpy(), cfg["Y"].to_numpy(), ["V", "fmax"])
    r = tab.set_index("term").loc["V"]
    return {"coef_V": float(r.coef), "se_hc3": float(r.se_hc3), "t": float(r.t), "p": float(r.p),
            "df": int(r.df), "table": tab.to_dict("records")}


# ------------------------------------------------------------------ H2 / H3

def _fit_predict(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(ytr)), Xtr]
    b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.c_[np.ones(len(Xte)), Xte] @ b


def _paired_boot(d: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p": float(min(1.0, max(p, 1.0 / n_boot)))}


def cgf_column(cfg: pd.DataFrame, s: float) -> np.ndarray:
    return np.array([vs.cgf_predictor(t, q, f, s) for t, q, f in zip(cfg["tau"], cfg["q"], cfg["f"])])


def choose_s(train: pd.DataFrame) -> float:
    y = train["Y"].to_numpy()
    return float(min(S_GRID, key=lambda s: loco_mse(cgf_column(train, s)[:, None], y)))


def h2_h3(train: pd.DataFrame, held: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict:
    ytr, yte = train["Y"].to_numpy(), held["Y"].to_numpy()
    p_fv = _fit_predict(train[["fmax", "V"]].to_numpy(), ytr, held[["fmax", "V"]].to_numpy())
    p_f = _fit_predict(train[["fmax"]].to_numpy(), ytr, held[["fmax"]].to_numpy())
    s = choose_s(train)
    p_z = _fit_predict(cgf_column(train, s)[:, None], ytr, cgf_column(held, s)[:, None])
    se_fv, se_f, se_z = (yte - p_fv) ** 2, (yte - p_f) ** 2, (yte - p_z) ** 2
    return {
        "H2": {"mse_fmax_V": float(se_fv.mean()), "mse_fmax": float(se_f.mean()),
               **_paired_boot(se_fv - se_f, n_boot, seed)},  # negative = {fmax, V} better
        "H3": {"mse_cgf": float(se_z.mean()), "mse_fmax_V": float(se_fv.mean()), "s": s,
               **_paired_boot(se_z - se_fv, n_boot, seed + 1)},  # negative = CGF better
        "n_train": len(train), "n_heldout": len(held),
    }


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def analyse(cfg: pd.DataFrame, heldout_ids: set, n_boot: int = 10_000) -> dict:
    """cfg: one row per matched configuration with Y, V, fmax, tau, q (array), f (array), config_id."""
    held = cfg[cfg["config_id"].isin(heldout_ids)]
    train = cfg[~cfg["config_id"].isin(heldout_ids)]
    r1 = h1(cfg)
    r23 = h2_h3(train, held, n_boot=n_boot)
    y = cfg["Y"].to_numpy()
    loco = {"fmax+V": loco_mse(cfg[["fmax", "V"]].to_numpy(), y), "fmax": loco_mse(cfg[["fmax"]].to_numpy(), y)}
    adj = holm({"H1": r1["p"], "H2": r23["H2"]["p"], "H3": r23["H3"]["p"]})
    return {"H1": r1, **{k: r23[k] for k in ("H2", "H3")}, "loco_all_matched": loco, "holm": adj,
            "n_configs": len(cfg), "n_train": r23["n_train"], "n_heldout": r23["n_heldout"]}
