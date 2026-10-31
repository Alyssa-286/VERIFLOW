"""Figure 3: idealized dynamics and the predictor regime map (master plan §22).

(a) logit true accuracy vs effective time for matched verifiers of increasing V (Prop. 2)
(b) effective FPR F(t) = K'(t), with the initial slope V marked (Prop. 3)
(c) LOCO relative MSE of each predictor vs horizon, one panel per update geometry
    (reads figures/out/sim_replicator/predictor_comparison.csv.gz)

Usage:  uv run python -m figures.make_fig3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simulations import sim_replicator as sr  # noqa: E402
from verifiers import stats as vs  # noqa: E402

OUT = Path("figures/out")
SIM = OUT / "sim_replicator"

# Reference palette (dataviz skill, light mode). Ordinal blue ramp for ordered V;
# fixed categorical order for predictors.
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e4e3dd"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False,
})

# Example matched family: fixed q0, F̄ = 0.3, τ = 0.9, A0 = 0.4.
Q0 = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
FBAR, TAU, A0 = 0.3, 0.9, 0.4


def example_family() -> np.ndarray:
    """Five verifiers with identical F̄ and τ and increasing V (one mode made progressively more acceptable)."""
    fam = []
    for d in [0.0, 0.15, 0.3, 0.45, 0.6]:
        f = np.full(5, FBAR)
        f[3] += d  # raise mode 4 by d
        f[[0, 1]] -= d * Q0[3] / (Q0[0] + Q0[1])  # lower modes 1-2 to keep F̄ fixed
        assert abs(Q0 @ f - FBAR) < 1e-14 and (f >= 0).all() and (f <= 1).all()
        fam.append(f)
    return np.array(fam)


def fig3ab():
    fam = example_family()
    t = np.linspace(0, 12, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for f, c in zip(fam, ORDINAL_BLUE):
        V = vs.weighted_var(Q0, f)
        lab = f"V={V:.3f}, f_max={f.max():.2f}"
        ax1.plot(t, sr.closed_form_logit_accuracy(A0, Q0, TAU, f, t), color=c, label=lab)
        F = vs.cgf_prime(Q0, f, t)
        ax2.plot(t, F, color=c)
        ax2.plot(t[:60], FBAR + V * t[:60], color=c, lw=1, ls=(0, (2, 2)))
    ax2.axhline(TAU, color=INK2, lw=1)
    ax2.text(12, TAU, " τ", va="center", color=INK2)
    ax1.set(xlabel="effective time  t̃", ylabel="logit true accuracy", title="(a) matched F̄=0.3, τ=0.9")
    ax2.set(xlabel="effective time  t̃", ylabel="effective FPR  F(t̃)=K′(t̃)", title="(b) initial slope = V (dashed)")
    ax1.legend(fontsize=7, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3ab.{ext}", dpi=200)
    plt.close(fig)
    return fam


def fig3c():
    cmp = pd.read_csv(SIM / "predictor_comparison.csv.gz")
    show = ["V", "fmax", "fmax+V", "V_q2", "CGF"]
    med = cmp[cmp.predictor.isin(show)].groupby(["rule", "T", "predictor"])["rel_mse"].median().reset_index()
    rules = [("npg", "natural PG (replicator)"), ("vpg", "vanilla softmax PG"), ("adam", "Adam (exact gradients)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True, constrained_layout=True)
    for ax, (rule, title) in zip(axes, rules):
        d = med[med.rule == rule]
        for p, c in zip(show, CATEGORICAL):
            s = d[d.predictor == p].sort_values("T")
            ax.plot(s["T"], s["rel_mse"], color=c, marker="o", ms=4, label=p)
        ax.set_xscale("log", base=2)
        ax.set(title=title, xlabel="horizon T (effective time)")
        ax.axhline(1.0, color=INK2, lw=1)
    axes[0].set_ylabel("median LOCO MSE / Var(y)\n(lower = better; 1 = no information)")
    axes[0].set_ylim(0, 1.3)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3c_regime_map.{ext}", dpi=200)
    plt.close(fig)

    # tabular companion (table view of the figure)
    full = cmp.groupby(["rule", "T", "predictor"])["rel_mse"].median().unstack("predictor")
    full.round(3).to_csv(OUT / "fig3c_table.csv")
    best = full.drop(columns=["null(intercept)"]).idxmin(axis=1).rename("best_predictor")
    best.to_csv(OUT / "fig3c_best.csv")
    return full, best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fam = fig3ab()
    full, best = fig3c()
    sources = {
        "fig3ab": {"q0": Q0.tolist(), "Fbar": FBAR, "tau": TAU, "A0": A0, "family": fam.tolist()},
        "fig3c": {"source": str(SIM / "predictor_comparison.csv.gz"), "params": json.loads((SIM / "params.json").read_text())},
    }
    (OUT / "fig3.sources.json").write_text(json.dumps(sources, indent=2))
    print(full.round(3).to_string())
    print(best.to_string())


if __name__ == "__main__":
    main()
