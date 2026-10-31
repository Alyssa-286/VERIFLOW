"""Gate 2 figure: incremental held-out value of V beyond f_max, per dynamics setting and horizon.

Reads figures/out/sim_shared_policy/summary.csv. Usage: uv run python -m figures.make_gate2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from figures.make_fig3 import CATEGORICAL, INK2  # noqa: E402  (shared style)

OUT = Path("figures/out")
ORDINAL_BLUE3 = ["#86b6ef", "#2a78d6", "#104281"]  # horizons, light -> dark


def main():
    s = pd.read_csv(OUT / "sim_shared_policy" / "summary.csv")
    settings = list(dict.fromkeys(s["setting"]))
    horizons = sorted(s["T"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True, constrained_layout=True)
    y0 = np.arange(len(settings))
    for ax, (col, lo, hi, title) in zip(
        axes,
        [("dV_given_fmax", "dV_given_fmax_lo", "dV_given_fmax_hi", "(a) adding V to f_max"),
         ("dCGF_vs_fmaxV", "dCGF_lo", "dCGF_hi", "(b) CGF vs f_max+V")],
    ):
        for j, (T, c) in enumerate(zip(horizons, ORDINAL_BLUE3)):
            d = s[s["T"] == T].set_index("setting").loc[settings]
            y = y0 + (j - 1) * 0.22
            ax.errorbar(d[col], y, xerr=[d[col] - d[lo], d[hi] - d[col]], fmt="o", ms=5, color=c, lw=1.5,
                        capsize=0, label=f"T = {T} steps")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title)
        ax.set_xlabel("Δ LOCO MSE / Var(y)   (negative = improvement)")
    axes[0].set_yticks(y0, settings)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_gate2.{ext}", dpi=200)
    print("wrote", OUT / "fig_gate2.png")


if __name__ == "__main__":
    main()
