#!/usr/bin/env python
"""Figure: heterogeneity pilot (exploratory; array 4020, D25). Greedy accuracy and the share of the
ONE_HI mode (the high-V verifier's most-accepted mode) among wrong answers over training, per configuration,
seeds shown individually. Writes figures/out/fig_pilot_het.{pdf,png} and figures/out/fig_pilot_het.sources.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [("pilot_het_v0", "homogeneous (V floor)"), ("pilot_het_vmid", "V 50 %"),
           ("pilot_het_vhigh", "V 90 %"), ("pilot_het_fmaxhigh_vlow", "f_max 0.95, V min")]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def main() -> int:
    table = pd.read_csv(ROOT / "configs/pilot/runs_het.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    sources = {}
    for (cid, label), col in zip(CONFIGS, COLORS):
        for i, r in enumerate(table[table["config"].str.contains(cid + ".json")].itertuples()):
            p = ROOT / "runs/pilot" / r.run_name / "eval.jsonl"
            sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
            ev = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l])
            kw = dict(color=col, alpha=0.8, lw=1.2, label=label if i == 0 else None)
            ax[0].plot(ev["step"], ev["greedy_accuracy"], **kw)
            ax[1].plot(ev["step"], [q.get("ONE_HI", 0.0) for q in ev["q_t"]], **kw)
    for a in ax:
        a.axvspan(50, 100, color="0.92", zorder=0)
        a.set_xlabel("GRPO step")
    ax[0].set_ylabel("greedy held-out accuracy")
    ax[1].set_ylabel("ONE_HI share of wrong answers")
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Heterogeneity pilot (exploratory, 4 configs x 3 seeds; shaded = outcome window)", fontsize=9)
    fig.tight_layout()
    out = ROOT / "figures/out"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_pilot_het.{ext}", dpi=200)
    (out / "fig_pilot_het.sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"wrote fig_pilot_het from {len(sources)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
