#!/usr/bin/env python
"""Final figures for the confirmatory grid (inputs: results/grid/{analysis.json,runs.csv}, configs/grid, runs/grid).

fig_grid_h1:         configuration mean outcome Ybar vs V (matched arm), coloured by f_max level, with the H1 fit
                     at the median f_max and its HC3 95 % band; auxiliary configurations shown separately.
fig_grid_heldout:    held-out configurations: observed Ybar vs predictions of {f_max}, {f_max, V} and the CGF plug-in,
                     each fitted on the training configurations (the H2/H3 comparison).
fig_grid_modes:      share of each config's most-accepted mode among wrong answers over training, by V tertile
                     (matched arm, seed-averaged), with the homogeneous control.
fig_grid_predictors: leave-one-config-out MSE relative to Var(Ybar) for every registered predictor set (all configs,
                     exploratory).
Each figure is written as pdf + png, with a .sources.json listing input files and their SHA-256.
"""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"
RUNS = Path(os.environ.get("VERIFLOW_GRID_RUNS", "runs/grid"))  # or results/grid/run_records
RUNS = RUNS if RUNS.is_absolute() else ROOT / RUNS


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig, name, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    (OUT / f"{name}.sources.json").write_text(json.dumps({str(Path(s).relative_to(ROOT)): sha(Path(s)) for s in sources},
                                                         indent=2) + "\n")
    plt.close(fig)


def main() -> int:
    from experiments.analyse_grid import config_table
    from vf_stats import confirmatory as cf
    a_path, r_path = ROOT / "results/grid/analysis.json", ROOT / "results/grid/runs.csv"
    an = json.loads(a_path.read_text())
    runs = pd.read_csv(r_path)
    table = pd.read_csv(ROOT / "configs/grid/runs_grid.csv")
    split = json.loads((ROOT / "configs/grid/split.json").read_text())
    cfg = config_table(table)
    valid = runs[runs["status"] == "completed"]
    means = valid.groupby("config_id")["Y"].mean().rename("Y")
    c = cfg.set_index("config_id").join(means, how="inner").reset_index()
    m = c[c["arm"] == "matched"].copy()
    aux = c[c["arm"] == "auxiliary"].copy()
    base_sources = [a_path, r_path, ROOT / "configs/grid/runs_grid.csv", ROOT / "configs/grid/split.json"]

    # H1
    def fmax_level(x):
        return "f_max 0.95" if abs(x - 0.95) < 1e-6 else ("f_max 0.6" if abs(x - 0.6) < 1e-6 else "f_max free")
    m["lvl"] = m["fmax"].map(fmax_level)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cols = {"f_max free": "#4C72B0", "f_max 0.6": "#55A868", "f_max 0.95": "#C44E52"}
    for lvl, d in m.groupby("lvl"):
        ax.scatter(d["V"], d["Y"], s=22, color=cols[lvl], label=f"matched, {lvl}", zorder=3)
    ax.scatter(aux["V"], aux["Y"], s=22, marker="x", color="0.4", label="auxiliary (F̄ 0.15/0.45)", zorder=3)
    tab = cf.ols_hc3(m[["V", "fmax"]].to_numpy(), m["Y"].to_numpy(), ["V", "fmax"]).set_index("term")
    vv = np.linspace(m["V"].min(), m["V"].max(), 50)
    fm = float(m["fmax"].median())
    yhat = tab.loc["intercept", "coef"] + tab.loc["V", "coef"] * vv + tab.loc["fmax", "coef"] * fm
    h1 = an["primary"]["H1"]
    ax.plot(vv, yhat, color="k", lw=1.2, label=f"H1 fit at median f_max (b_V = {h1['coef_V']:.2f})")
    ax.set_xlabel("V (q_ref-weighted variance of f)")
    ax.set_ylabel("Ȳ = logit window-mean accuracy")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"H1: b_V = {h1['coef_V']:.2f}, 95% CI [{h1['ci95'][0]:.2f}, {h1['ci95'][1]:.2f}] (HC3)", fontsize=8)
    save(fig, "fig_grid_h1", base_sources)

    # Held-out
    held = m[m["config_id"].isin(split["heldout"])]
    train = m[~m["config_id"].isin(split["heldout"])]
    s = cf.choose_s(train)
    preds = {
        "{f_max}": cf._fit_predict(train[["fmax"]].to_numpy(), train["Y"].to_numpy(), held[["fmax"]].to_numpy()),
        "{f_max, V}": cf._fit_predict(train[["fmax", "V"]].to_numpy(), train["Y"].to_numpy(), held[["fmax", "V"]].to_numpy()),
        f"CGF Z(s), s={s:.2f}": cf._fit_predict(cf.cgf_column(train, s)[:, None], train["Y"].to_numpy(),
                                               cf.cgf_column(held, s)[:, None]),
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo, hi = float(min(held["Y"].min(), *(p.min() for p in preds.values()))), float(max(held["Y"].max(), *(p.max() for p in preds.values())))
    ax.plot([lo, hi], [lo, hi], color="0.7", lw=1)
    for (name, p), mk in zip(preds.items(), ["o", "s", "^"]):
        mse = float(np.mean((held["Y"].to_numpy() - p) ** 2))
        ax.scatter(p, held["Y"], marker=mk, s=24, alpha=0.8, label=f"{name}: MSE {mse:.3f}")
    ax.set_xlabel("predicted Ȳ (fitted on training configs)")
    ax.set_ylabel("observed Ȳ (held-out configs)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_grid_heldout", base_sources)

    # Mode trajectories by V tertile (matched)
    m["tert"] = pd.qcut(m["V"].rank(method="first"), 3, labels=["low V", "mid V", "high V"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    srcs = list(base_sources)
    curves = {}
    for r in valid[valid["arm"] == "matched"].itertuples():
        row = m[m["config_id"] == r.config_id]
        if row.empty:
            continue
        fd = row["f_dict"].iloc[0]
        designed = {k: float(v) for k, v in fd.items() if k not in ("U", "I")}
        mode = max(sorted(designed), key=lambda k: designed[k])
        p = RUNS / r.run / "eval.jsonl"
        srcs.append(p)
        ev = [json.loads(l) for l in p.read_text().splitlines() if l]
        key = "homogeneous control" if r.config_id == "m000" else str(row["tert"].iloc[0])
        curves.setdefault(key, []).append(pd.Series({e["step"]: (e["q_t"] or {}).get(mode, 0.0) for e in ev}))
    style = {"low V": "#4C72B0", "mid V": "#55A868", "high V": "#C44E52", "homogeneous control": "k"}
    for key in ["low V", "mid V", "high V", "homogeneous control"]:
        if key in curves:
            df = pd.concat(curves[key], axis=1)
            mu, se = df.mean(1), df.std(1, ddof=1) / np.sqrt(df.shape[1]) if df.shape[1] > 1 else 0 * df.mean(1)
            ax.plot(mu.index, mu, color=style[key], label=f"{key} (n={df.shape[1]} runs)")
            ax.fill_between(mu.index, mu - se, mu + se, color=style[key], alpha=0.15)
    ax.axvspan(50, 100, color="0.93", zorder=0)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("share of most-accepted mode among errors")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_grid_modes", srcs)

    # Predictor comparison (exploratory)
    fe = pd.DataFrame(an["secondary_predictors_all_configs_exploratory"])
    namecol = "predictor" if "predictor" in fe.columns else fe.columns[0]
    valcol = "rel_mse" if "rel_mse" in fe.columns else "loco_mse"
    fe = fe.sort_values(valcol)
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(fe) + 1))
    ax.barh(fe[namecol].astype(str), fe[valcol], color="#4C72B0")
    ax.axvline(1.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel(f"{valcol} (LOCO, all configs; exploratory)")
    ax.invert_yaxis()
    save(fig, "fig_grid_predictors", base_sources)
    print("wrote fig_grid_h1, fig_grid_heldout, fig_grid_modes, fig_grid_predictors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
