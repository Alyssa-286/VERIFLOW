#!/usr/bin/env python
"""Supplementary grid figure (written after the registered analysis, D33; exploratory/descriptive).

fig_grid_supp: (a) realised false-acceptance rate on wrong rollouts over [T/2, T] (from the verifier logs) against
V, matched arm, one point per unique verifier (seed mean), coloured by f_max level; (b) greedy accuracy and the
share of the most-accepted mode among errors for every run with Y < 0.3, against the seed-mean curve of the
homogeneous control. Inputs: results/grid/runs_realised.csv, results/grid/supplement.json, runs/grid/*/eval.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from figures.make_grid_figs import ROOT, save  # noqa: E402


def main() -> int:
    rr_path, sup_path = ROOT / "results/grid/runs_realised.csv", ROOT / "results/grid/supplement.json"
    rr = pd.read_csv(rr_path)
    sup = json.loads(sup_path.read_text())
    m = rr[rr["arm"] == "matched"].groupby("config_id").agg(V=("V_design", "first"), fmax=("fmax_design", "first"),
                                                            F=("F_real_window", "mean")).reset_index()
    m = m.drop_duplicates(["V", "fmax", "F"])
    lvl = m["fmax"].map(lambda x: "f_max 0.95" if abs(x - .95) < 1e-6 else ("f_max 0.6" if abs(x - .6) < 1e-6 else "f_max free"))
    cols = {"f_max free": "#4C72B0", "f_max 0.6": "#55A868", "f_max 0.95": "#C44E52"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for k, d in m.groupby(lvl):
        a.scatter(d["V"], d["F"], s=22, color=cols[k], label=k)
    a.axhline(0.3, color="0.5", ls="--", lw=0.8, label="design F̄ = 0.3")
    a.set_xlabel("V (design)")
    a.set_ylabel("realised false-acceptance, steps 50–100")
    a.set_title("(a) matched arm, unique verifiers", fontsize=9)
    a.legend(fontsize=7, frameon=False)
    srcs = [rr_path, sup_path]
    hom = []
    for s in (1, 2):
        p = ROOT / f"runs/grid/grid_m000_s{s}/eval.jsonl"
        srcs.append(p)
        hom.append(pd.Series({e["step"]: e["greedy_accuracy"] for e in map(json.loads, p.read_text().splitlines())}))
    h = pd.concat(hom, axis=1).mean(axis=1)
    b.plot(h.index, h.values, color="k", lw=1.5, label="accuracy, homogeneous control")
    seen = set()
    for i, (run, t) in enumerate(sup["F_unusual_runs_exploratory"]["low_Y_trajectories"].items()):
        key = tuple(t["acc"])
        if key in seen:
            continue
        seen.add(key)
        steps = list(range(0, 10 * len(t["acc"]), 10))
        c = f"C{i}"
        b.plot(steps, t["acc"], color=c, lw=1.2, label=f"accuracy, {run.replace('grid_', '')}")
        b.plot(steps, t["q_top"], color=c, lw=1.0, ls=":", label=f"{t['most_accepted_mode']} share of errors")
    b.axvspan(50, 100, color="0.93", zorder=0)
    b.set_xlabel("GRPO step")
    b.set_title("(b) runs with Y < 0.3 (distinct trajectories)", fontsize=9)
    b.legend(fontsize=6, frameon=False, ncol=1)
    save(fig, "fig_grid_supp", srcs)
    print("wrote fig_grid_supp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
