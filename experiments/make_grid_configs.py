#!/usr/bin/env python
"""Confirmatory configuration grid and run table (master plan §10.2, §14). Requires q_ref.lock,
configs/pilot/selection.json (lr, T) and the R chosen by the power analysis (D23).

Matched arm (F̄ = F̄*, τ = τ*): a pool from design_grid over V fractions {0, .25, .5, .75, 1} x f_max
{free, 0.6, 0.95}, 3 per cell, plus the homogeneous control (generator seed GRID_SEED). R configs are taken
from the pool: the homogeneous control, then one config per grid cell in a seeded random cell order,
cycling over cells until R are chosen (balanced over the (V, f_max) grid).
Auxiliary arm (unmatched F̄): F̄ ∈ {0.5 F̄*, 1.5 F̄*} = {0.15, 0.45}, f_max free, V fractions spread evenly
over [0, 0.9] (the exact maximum is numerically unreachable at F̄ = 0.45; the pilot also capped at 0.9); 4 per F̄ for R = 24 (standard), 6 per F̄ for R = 36 (extended), none for R = 12 (minimum), §14.
Held-out: one third of the matched configs, stratified by grid cell (homogeneous always in training).
Seeds: 2 per config. The run order is a seeded random permutation, so cluster time cannot confound configs.

Writes configs/grid/{matched,aux}/*.json (+ index.csv), configs/grid/split.json, configs/grid/runs_grid.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID_SEED = 20261108
SPLIT_SEED = 20261109
ORDER_SEED = 20261110
V_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
FMAX_LEVELS = [None, 0.6, 0.95]
AUX_PER_FBAR = {12: 0, 24: 4, 36: 6}
SEEDS = (1, 2)


def pick_matched(pool: list, R: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    homog = [c for c in pool if c.grid_cell == "homog"]
    cells: dict[str, list] = {}
    for c in pool:
        if c.grid_cell != "homog":
            cells.setdefault(c.grid_cell, []).append(c)
    order = [sorted(cells)[i] for i in rng.permutation(len(cells))]
    chosen = list(homog[:1])
    depth = 0
    while len(chosen) < R:
        added = False
        for cell in order:
            if depth < len(cells[cell]) and len(chosen) < R:
                chosen.append(cells[cell][depth])
                added = True
        if not added:
            raise ValueError(f"pool too small for R = {R}")
        depth += 1
    return chosen


def build(q_ref: dict, Fbar: float, tau: float, R: int):
    from verifiers.matched_verifiers import design_grid, feasible_v_max, generate_config, v_floor
    pool = design_grid(q_ref, Fbar, tau, V_FRACS, FMAX_LEVELS, 3, GRID_SEED, prefix="m")
    matched = pick_matched(pool, R, GRID_SEED)
    designed = [k for k in q_ref if k not in ("U", "I")]
    rng = np.random.default_rng(GRID_SEED + 1)
    aux = []
    n = AUX_PER_FBAR[R]
    for fb in (round(0.5 * Fbar, 6), round(1.5 * Fbar, 6)):
        lo, hi = v_floor(q_ref, fb), feasible_v_max(q_ref, fb)
        for i, frac in enumerate(np.linspace(0, 0.9, n) if n else []):
            aux.append(generate_config(q_ref, designed, fb, lo + float(frac) * (hi - lo), tau, rng,
                                       config_id=f"a{int(fb * 100):02d}_{i}", grid_cell=f"Fbar={fb}|Vfrac={frac:.2f}",
                                       generator_seed=GRID_SEED + 1))
    return matched, aux


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--R", type=int, required=True, choices=sorted(AUX_PER_FBAR))
    ap.add_argument("--out", default=str(ROOT / "configs" / "grid"))
    args = ap.parse_args(argv)
    from verifiers.matched_verifiers import save_configs, split_heldout
    from verifiers.qref import load_q_ref

    q = load_q_ref()
    base = json.loads((ROOT / "configs" / "base.json").read_text())["verifier"]
    sel = json.loads((ROOT / "configs" / "pilot" / "selection.json").read_text())
    matched, aux = build(q["q_ref"], base["Fbar"], base["tau"], args.R)
    train, held = split_heldout(matched, 1 / 3, SPLIT_SEED)
    out = Path(args.out)
    save_configs(matched, out / "matched")
    if aux:
        save_configs(aux, out / "aux")
    (out / "split.json").write_text(json.dumps({
        "train": sorted(c.config_id for c in train), "heldout": sorted(c.config_id for c in held),
        "seed": SPLIT_SEED, "q_ref_lock_sha256": q["sha256"]}, indent=2) + "\n")
    rows = [{"run_name": f"grid_{c.config_id}_s{s}", "config": f"configs/grid/{arm}/{c.config_id}.json",
             "lr": sel["lr"], "seed": s, "max_steps": sel["T"]}
            for arm, cs in (("matched", matched), ("aux", aux)) for c in cs for s in SEEDS]
    order = np.random.default_rng(ORDER_SEED).permutation(len(rows))
    pd.DataFrame([rows[i] for i in order]).to_csv(out / "runs_grid.csv", index=False)
    print(json.dumps({"R": args.R, "matched": len(matched), "aux": len(aux), "heldout": len(held),
                      "runs": len(rows), "lr": sel["lr"], "T": sel["T"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
