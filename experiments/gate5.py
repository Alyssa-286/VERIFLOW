#!/usr/bin/env python
"""Gate 5 (empirical engagement) evaluation of the heterogeneity pilot, pre-declared in D21.

Master plan §27 kill condition: "modes are not differentially amplified at all (all q_k(t) flat;
the homogeneous and high-V configs are indistinguishable in F(t))". D21 makes it operational.
This file was committed before any heterogeneity-pilot data existed.

Per run (config c, seed s), with T from configs/pilot/selection.json:
  m*_c     = argmax_k f_k over the designed wrong-answer modes of c (U and I excluded).
  G_c,s    = log(q_m*(T) + eps) - log(q_m*(0) + eps), q from the sampled eval-mode census
             (eval.jsonl "q_t", the distribution of wrong answers), eps = 1e-3.
  F_c,s    = realised acceptance rate of wrong rollouts (verifier_log: mean reward where
             correct is False) over training steps in [T/2, T].
  Y_c,s    = logit of mean greedy accuracy over eval steps in [T/2, T] (primary outcome, §14).

Contrasts against the homogeneous config v0 (seed means; SE of a difference of means with the
per-config seed SDs; 3 seeds give a descriptive check, never a significance claim, §13):
  D_c   = mean_s G_c,s - mean_s G_v0,s   for the SAME mode m*_c, computed in the v0 runs.
  dF_c  = mean_s F_c,s - mean_s F_v0,s.
  A config is engaged on D  if |D_c|  > 2 * SE and |D_c|  >= 0.10;
  engaged on F  if |dF_c| > 2 * SE and |dF_c| >= 0.02.

Gate 5 FAILS (the §27 kill condition) iff no non-homogeneous config is engaged on D AND the
high-V config is not engaged on F. On FAIL: lengthen the horizon once (T' = 2T, 12 new runs);
if still FAIL, stop the GPU arm. Accuracy differences (Y) are reported but do not enter the gate:
a null in accuracy is a result, not a kill.

Writes <out>/gate5.json and prints it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-3
D_FLOOR = 0.10
F_FLOOR = 0.02
SE_MULT = 2.0
EXCLUDED = {"U", "I"}
HOMOG = "pilot_het_v0"
HIGH_V = "pilot_het_vhigh"


def argmax_mode(cfg: dict) -> str:
    f = {k: float(v) for k, v in cfg["f"].items() if k not in EXCLUDED}
    return max(sorted(f), key=lambda k: f[k])


def read_eval(run_dir: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines() if l]
    return pd.DataFrame(rows).drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)


def mode_growth(ev: pd.DataFrame, mode: str, T: int) -> float:
    q0 = float((ev.loc[ev["step"] == 0, "q_t"].iloc[0] or {}).get(mode, 0.0))
    qT = float((ev.loc[ev["step"] == T, "q_t"].iloc[0] or {}).get(mode, 0.0))
    return math.log(qT + EPS) - math.log(q0 + EPS)


def window_logit(ev: pd.DataFrame, T: int) -> float:
    w = ev[(ev["step"] >= T / 2) & (ev["step"] <= T)]["greedy_accuracy"]
    a = float(w.mean())
    return math.log(a / (1 - a))


def false_accept(run_dir: Path, T: int) -> float:
    vl = pd.concat([pd.read_parquet(p, columns=["step", "correct", "reward"])
                    for p in sorted((run_dir / "verifier_log").glob("*.parquet"))])
    w = vl[(vl["step"] >= T / 2) & (vl["step"] <= T) & (~vl["correct"].astype(bool))]
    return float(w["reward"].mean())


def se_diff(a: list[float], b: list[float]) -> float:
    return math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))


def evaluate(table: pd.DataFrame, runs_root: Path, T: int) -> dict:
    cfgs = {Path(c).stem: json.loads((ROOT / c).read_text()) for c in table["config"].unique()}
    per_run = []
    for r in table.itertuples():
        cid = Path(r.config).stem
        rd = runs_root / r.run_name
        man = json.loads((rd / "manifest.json").read_text())
        if man.get("status") != "completed" or int(man.get("final_step", -1)) != T:
            raise RuntimeError(f"{r.run_name}: not a completed run to T={T} ({man.get('status')})")
        ev = read_eval(rd)
        per_run.append({"run": r.run_name, "config": cid, "seed": int(r.seed), "ev": ev,
                        "Y": window_logit(ev, T), "F": false_accept(rd, T),
                        "acc_T": float(ev.loc[ev["step"] == T, "greedy_accuracy"].iloc[0])})
    by = lambda cid: [p for p in per_run if p["config"] == cid]
    out = {"T": T, "rule": "D21", "configs": {}}
    for cid, cfg in sorted(cfgs.items()):
        m = argmax_mode(cfg)
        g = [mode_growth(p["ev"], m, T) for p in by(cid)]
        out["configs"][cid] = {
            "argmax_f_mode": m, "f_argmax": float(cfg["f"][m]),
            "G_seeds": g, "F_seeds": [p["F"] for p in by(cid)], "Y_seeds": [p["Y"] for p in by(cid)],
            "acc_T_seeds": [p["acc_T"] for p in by(cid)],
            "q_t_T_seeds": [p["ev"].loc[p["ev"]["step"] == T, "q_t"].iloc[0] for p in by(cid)],
        }
    base = by(HOMOG)
    engaged_D, engaged_F_high = [], False
    for cid, rec in out["configs"].items():
        if cid == HOMOG:
            continue
        g0 = [mode_growth(p["ev"], rec["argmax_f_mode"], T) for p in base]
        D, seD = float(np.mean(rec["G_seeds"]) - np.mean(g0)), se_diff(rec["G_seeds"], g0)
        f0 = [p["F"] for p in base]
        dF, seF = float(np.mean(rec["F_seeds"]) - np.mean(f0)), se_diff(rec["F_seeds"], f0)
        y0 = [p["Y"] for p in base]
        eD = abs(D) > SE_MULT * seD and abs(D) >= D_FLOOR
        eF = abs(dF) > SE_MULT * seF and abs(dF) >= F_FLOOR
        rec.update({"G_v0_same_mode": g0, "D": D, "SE_D": seD, "engaged_D": eD,
                    "dF": dF, "SE_dF": seF, "engaged_F": eF,
                    "dY": float(np.mean(rec["Y_seeds"]) - np.mean(y0)), "SE_dY": se_diff(rec["Y_seeds"], y0)})
        if eD:
            engaged_D.append(cid)
        if cid == HIGH_V and eF:
            engaged_F_high = True
    out["engaged_D"] = engaged_D
    out["high_V_engaged_F"] = engaged_F_high
    out["gate5_pass"] = bool(engaged_D or engaged_F_high)
    out["seed_sd_Y"] = {cid: float(np.std(r["Y_seeds"], ddof=1)) for cid, r in out["configs"].items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(ROOT / "runs" / "pilot"))
    ap.add_argument("--pilot-dir", default=str(ROOT / "configs" / "pilot"))
    ap.add_argument("--out", default=str(ROOT / "configs" / "pilot"))
    args = ap.parse_args(argv)
    pdir = Path(args.pilot_dir)
    T = int(json.loads((pdir / "selection.json").read_text())["T"])
    res = evaluate(pd.read_csv(pdir / "runs_het.csv"), Path(args.runs_root), T)
    text = json.dumps(res, indent=2, default=float)
    (Path(args.out) / "gate5.json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
