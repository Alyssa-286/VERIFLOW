#!/usr/bin/env python
"""Registered analysis of the confirmatory grid (PREREGISTRATION.md §4–§6, D25: estimation with CIs).

Inputs: configs/grid/runs_grid.csv, configs/grid/split.json, configs/grid/{matched,auxiliary}/*.json,
runs/grid/<run>/{manifest.json, eval.jsonl}. T from configs/pilot/selection.json.

Primary (matched arm, configuration = unit):
  - Y per run (window-mean logit accuracy over [T/2, T], vf_stats.confirmatory.run_outcome); Ybar per config.
  - vf_stats.confirmatory.analyse(): H1 OLS (HC3) with 95 % CIs, H2/H3 held-out paired bootstrap (10,000),
    LOCO MSE, Holm-adjusted p over H1-H3. Reported as estimates with CIs (no confirmatory claim, D25).
  - ICC(1) between configurations and seeds.
Secondary / exploratory (labelled):
  - every registered predictor's LOCO MSE and Spearman rho on all configurations (both arms), from config means;
    with 2 seeds per config the random-intercept model's fixed effects equal OLS on config means (balanced);
  - secondary outcomes per run: final accuracy, hacking indicator H (threshold 0.029, D25), growth of the
    most-accepted mode, KL(q(T)||q(0)), I+LEN share at T, divergence (< 1 % accuracy for 3 consecutive evals);
  - robustness: H1 on configs with 2 valid seeds only; H1 without the homogeneous control; H1 on final accuracy.
Missing data (§6): a config with one valid seed uses that seed; no run is excluded for "looking wrong".

Writes results/grid/analysis.json and results/grid/runs.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from vf_stats import confirmatory as cf

ROOT = Path(__file__).resolve().parents[1]
H_THRESHOLD = 0.029
EPS = 1e-3


def _clip_logit(a: float, n: int = 1000) -> float:
    lo = 1 / (2 * n)
    a = min(max(a, lo), 1 - lo)
    return math.log(a / (1 - a))


def run_secondary(run_dir: Path, T: int, f: dict) -> dict:
    ev = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines() if l]
    ev = pd.DataFrame(ev).drop_duplicates("step", keep="last").sort_values("step")
    acc = ev.set_index("step")["greedy_accuracy"]
    q0 = ev.loc[ev["step"] == 0, "q_t"].iloc[0] or {}
    qT = ev.loc[ev["step"] == T, "q_t"].iloc[0] or {}
    designed = {k: float(v) for k, v in f.items() if k not in ("U", "I")}
    m = max(sorted(designed), key=lambda k: designed[k])
    modes = sorted(set(q0) | set(qT))
    p = np.array([qT.get(k, 0.0) for k in modes]) + EPS
    q = np.array([q0.get(k, 0.0) for k in modes]) + EPS
    p, q = p / p.sum(), q / q.sum()
    low = (acc < 0.01).astype(int).rolling(3).sum()
    return {"acc_T": float(acc.loc[T]),
            "hacking": int(acc.max() - acc.loc[T] >= H_THRESHOLD),
            "growth_most_accepted": math.log(qT.get(m, 0) + EPS) - math.log(q0.get(m, 0) + EPS),
            "kl_qT_q0": float(np.sum(p * np.log(p / q))),
            "I_LEN_share_T": float(qT.get("I", 0) + qT.get("LEN", 0)),
            "diverged": bool((low >= 3).any())}


def config_table(table: pd.DataFrame) -> pd.DataFrame:
    from verifiers.matched_verifiers import load_config
    rows = {}
    for path in table["config"].unique():
        c = load_config(ROOT / path)
        names, qq, ff = c.vectors()
        rows[c.config_id] = {"config_id": c.config_id, "arm": Path(path).parent.name, "tau": c.tau, "q": qq, "f": ff,
                             "f_dict": dict(c.f), "Fbar": c.target_Fbar, **c.predictors()}
    return pd.DataFrame(rows.values())


def h1_ci(tab: dict) -> dict:
    t = st.t.ppf(0.975, tab["df"])
    return {**{k: tab[k] for k in ("coef_V", "se_hc3", "t", "p", "df")},
            "ci95": [tab["coef_V"] - t * tab["se_hc3"], tab["coef_V"] + t * tab["se_hc3"]]}


def analyse(runs_root: Path, grid_dir: Path, T: int, n_boot: int = 10_000) -> tuple[dict, pd.DataFrame]:
    table = pd.read_csv(grid_dir / "runs_grid.csv")
    split = json.loads((grid_dir / "split.json").read_text())
    cfg = config_table(table).set_index("config_id")
    runs = []
    for r in table.itertuples():
        rd = runs_root / r.run_name
        cid = Path(r.config).stem
        rec = {"run": r.run_name, "config_id": cid, "seed": int(r.seed), "arm": cfg.loc[cid, "arm"]}
        if not (rd / "manifest.json").exists():
            runs.append({**rec, "status": "missing"})
            continue
        out = cf.run_outcome(rd, T)
        runs.append({**rec, "status": out["status"], "Y": out["Y"], **run_secondary(rd, T, cfg.loc[cid, "f_dict"])})
    runs = pd.DataFrame(runs)
    valid = runs[runs["status"] == "completed"].copy()
    means = valid.groupby("config_id").agg(Y=("Y", "mean"), n_seeds=("Y", "size"),
                                           Y_final=("acc_T", lambda a: float(np.mean([_clip_logit(x) for x in a]))))
    allc = cfg.join(means, how="inner").reset_index()
    matched = allc[allc["arm"] == "matched"].copy()
    held = set(split["heldout"])
    res = {"T": T, "claim": "estimation with CIs (D25)", "n_runs": len(runs), "n_valid": len(valid),
           "status_counts": runs["status"].value_counts().to_dict()}
    prim = cf.analyse(matched, held, n_boot=n_boot)
    prim["H1"] = h1_ci(prim["H1"])
    res["primary"] = prim
    res["icc_matched"] = cf.icc(valid[valid["arm"] == "matched"])
    res["icc_all"] = cf.icc(valid)
    rob = {}
    two = matched[matched["n_seeds"] >= 2]
    rob["H1_two_seed_configs_only"] = h1_ci(cf.h1(two)) if len(two) >= 5 else None
    nohom = matched[matched["config_id"] != "m000"]
    rob["H1_without_homogeneous_control"] = h1_ci(cf.h1(nohom))
    rob["H1_on_final_accuracy"] = h1_ci(cf.h1(matched.assign(Y=matched["Y_final"])))
    res["robustness_exploratory"] = rob
    from vf_stats.predictor_comparison import forecast_eval
    Q = np.stack(allc["q"].to_numpy())
    F = np.stack(allc["f"].to_numpy())
    fe = forecast_eval(float(allc["tau"].iloc[0]), Q, F, allc["Y"].to_numpy())
    res["secondary_predictors_all_configs_exploratory"] = fe.to_dict("records")
    sec_cols = ["acc_T", "hacking", "growth_most_accepted", "kl_qT_q0", "I_LEN_share_T", "diverged"]
    res["secondary_outcomes_by_arm"] = valid.groupby("arm")[sec_cols].mean(numeric_only=True).to_dict("index")
    res["divergences_by_arm"] = valid.groupby("arm")["diverged"].sum().astype(int).to_dict()
    res["configs_with_one_valid_seed"] = sorted(means.index[means["n_seeds"] < 2])
    res["spearman_Y_V_matched"] = float(st.spearmanr(matched["V"], matched["Y"]).statistic)
    return res, runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(ROOT / "runs" / "grid"))
    ap.add_argument("--grid-dir", default=str(ROOT / "configs" / "grid"))
    ap.add_argument("--out", default=str(ROOT / "results" / "grid"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    T = int(json.loads((ROOT / "configs" / "pilot" / "selection.json").read_text())["T"])
    res, runs = analyse(Path(args.runs_root), Path(args.grid_dir), T, args.n_boot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    runs.to_csv(out / "runs.csv", index=False)
    print(json.dumps({k: res[k] for k in ("n_runs", "n_valid", "status_counts")}, default=str))
    print(json.dumps(res["primary"]["H1"], default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
