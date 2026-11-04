#!/usr/bin/env python
"""Pilot verifier configurations and run tables (master plan §13, D19). Requires q_ref.lock.

LR pilot (4 runs, 800 steps, seed 1):
  homogeneous verifier (minimum dispersion at F̄*, τ*) × lr ∈ {5e-7, 1e-6, 3e-6}
  perfect verifier (τ = 1, every f = 0) × lr = 1e-6 (the middle LR, fixed in D19)
Heterogeneity pilot (4 configs × 3 seeds, at the pilot-chosen lr and horizon T):
  V = floor (homogeneous), V at 50 % and 90 % of the feasible range (f_max free),
  and a high-f_max / low-V config (f_max = 0.95, V at the minimum feasible for that mode).
Pilot data are exploratory (§13): these configurations are excluded from the confirmatory set.

Writes configs/pilot/{lr,het}/*.json (+ index.csv), configs/pilot/runs_lr.csv.
runs_het.csv is written by experiments/pilot_select.py once lr and T are chosen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LRS = (5e-7, 1e-6, 3e-6)
PERFECT_LR = 1e-6
LR_PILOT_STEPS = 800
HET_SEEDS = (1, 2, 3)
GEN_SEED = 20261106


def build(q_ref: dict, Fbar: float, tau: float, seed: int = GEN_SEED):
    from verifiers.matched_verifiers import (feasible_v_max, feasible_v_range_by_mode, generate_config,
                                             homogeneous_config, v_floor)
    designed = [k for k in q_ref if k not in ("U", "I")]
    rng = np.random.default_rng(seed)
    lr_cfgs = [homogeneous_config(q_ref, Fbar, tau, "pilot_homog"),
               homogeneous_config(q_ref, 0.0, 1.0, "pilot_perfect")]
    lo, hi = v_floor(q_ref, Fbar), feasible_v_max(q_ref, Fbar)
    het = [homogeneous_config(q_ref, Fbar, tau, "pilot_het_v0")]
    for name, frac in (("pilot_het_vmid", 0.5), ("pilot_het_vhigh", 0.9)):
        het.append(generate_config(q_ref, designed, Fbar, lo + frac * (hi - lo), tau, rng,
                                   config_id=name, grid_cell=f"Vfrac={frac}", generator_seed=seed))
    ranges = feasible_v_range_by_mode(q_ref, Fbar, 0.95)
    mode = min(ranges, key=lambda m: ranges[m][0])
    het.append(generate_config(q_ref, designed, Fbar, ranges[mode][0] + 1e-9, tau, rng, fmax=0.95,
                               fmax_mode=mode, config_id="pilot_het_fmaxhigh_vlow",
                               grid_cell="fmax=0.95|Vmin", generator_seed=seed))
    return lr_cfgs, het


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "configs" / "pilot"))
    args = ap.parse_args(argv)
    from verifiers.matched_verifiers import save_configs
    from verifiers.qref import load_q_ref

    q = load_q_ref()  # frozen lock only; raises otherwise
    base = json.loads((ROOT / "configs" / "base.json").read_text())
    lr_cfgs, het = build(q["q_ref"], base["verifier"]["Fbar"], base["verifier"]["tau"])
    out = Path(args.out)
    save_configs(lr_cfgs, out / "lr")
    save_configs(het, out / "het")
    rows = [{"run_name": f"lr{lr:g}_homog_s1", "config": f"configs/pilot/lr/pilot_homog.json",
             "lr": lr, "seed": 1, "max_steps": LR_PILOT_STEPS} for lr in LRS]
    rows.append({"run_name": f"lr{PERFECT_LR:g}_perfect_s1", "config": "configs/pilot/lr/pilot_perfect.json",
                 "lr": PERFECT_LR, "seed": 1, "max_steps": LR_PILOT_STEPS})
    pd.DataFrame(rows).to_csv(out / "runs_lr.csv", index=False)
    (out / "q_ref_source.json").write_text(json.dumps({"lock_sha256": q["lock_sha256"], "merged": q["merged"],
                                                       "q_ref": q["q_ref"]}, indent=2) + "\n")
    print(json.dumps({"lr_runs": len(rows), "het_configs": [c.config_id for c in het],
                      "het_V": [round(c.V, 6) for c in het]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
