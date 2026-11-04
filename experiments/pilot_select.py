#!/usr/bin/env python
"""Pilot selection rules (master plan §13, pre-declared in D19) and the heterogeneity run table.

LR rule: among the three homogeneous LR-pilot runs that completed, choose the LR with the highest
mean greedy eval accuracy over steps [400, 800]; ties (within 0.005) go to the smaller LR.
A run whose accuracy falls below 0.01 for 3 consecutive evals is ineligible (diverged).

Horizon rule (§13): T = the first eval step at which the chosen run reaches 90 % of its maximum
greedy-accuracy gain over step 0, rounded up to a multiple of 50, with a floor of 100 so that the
outcome window [T/2, T] contains at least two eval points (eval every 50 steps).

Writes configs/pilot/selection.json and configs/pilot/runs_het.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIE = 0.005
WINDOW = (400, 800)


def read_eval(run_dir: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines() if l]
    return pd.DataFrame(rows).drop_duplicates("step", keep="last").sort_values("step")


def diverged(ev: pd.DataFrame) -> bool:
    low = (ev["greedy_accuracy"] < 0.01).astype(int)
    return bool((low.rolling(3).sum() >= 3).any())


def choose_lr(runs: dict[float, pd.DataFrame]) -> tuple[float, dict]:
    scores = {}
    for lr, ev in runs.items():
        w = ev[(ev["step"] >= WINDOW[0]) & (ev["step"] <= WINDOW[1])]
        scores[lr] = None if (diverged(ev) or w.empty) else float(w["greedy_accuracy"].mean())
    ok = {lr: s for lr, s in scores.items() if s is not None}
    if not ok:
        raise ValueError("no eligible LR-pilot run")
    best = max(ok.values())
    return min(lr for lr, s in ok.items() if s >= best - TIE), scores


def horizon(ev: pd.DataFrame) -> int:
    a0 = float(ev.loc[ev["step"] == 0, "greedy_accuracy"].iloc[0])
    gain = float(ev["greedy_accuracy"].max()) - a0
    if gain <= 0:
        raise ValueError("the chosen run shows no accuracy gain; horizon undefined (Gate 5 concern)")
    hit = ev.loc[ev["greedy_accuracy"] >= a0 + 0.9 * gain, "step"].iloc[0]
    return max(100, int(math.ceil(hit / 50) * 50))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(ROOT / "runs" / "pilot"))
    ap.add_argument("--out", default=str(ROOT / "configs" / "pilot"))
    args = ap.parse_args(argv)
    table = pd.read_csv(Path(args.out) / "runs_lr.csv")
    homog = table[table["config"].str.endswith("pilot_homog.json")]
    runs = {float(r.lr): read_eval(Path(args.runs_root) / r.run_name) for r in homog.itertuples()}
    lr, scores = choose_lr(runs)
    T = horizon(runs[lr])
    het = sorted(p.name for p in (Path(args.out) / "het").glob("pilot_het_*.json"))
    rows = [{"run_name": f"het_{h[:-5]}_s{s}", "config": f"configs/pilot/het/{h}", "lr": lr, "seed": s,
             "max_steps": T} for h in het for s in (1, 2, 3)]
    pd.DataFrame(rows).to_csv(Path(args.out) / "runs_het.csv", index=False)
    sel = {"lr": lr, "T": T, "window_scores": {str(k): v for k, v in scores.items()},
           "rule": "D19", "het_runs": len(rows)}
    (Path(args.out) / "selection.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps(sel, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
