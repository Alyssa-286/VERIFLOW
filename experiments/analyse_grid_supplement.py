#!/usr/bin/env python
"""Supplementary analyses of the confirmatory grid, written AFTER the registered driver had been run (D33).

Nothing here changes a registered hypothesis, outcome, unit, test, threshold or exclusion rule. Each section is
labelled with its status:

  A  REGISTERED, not printed by the driver: b_f with its HC3 95 % CI from the same H1 fit (§5: "b_V and b_f with
     HC3 CIs"). Same function (vf_stats.confirmatory.h1), same data.
  B  REGISTERED secondary model (§5, all arms): config-mean OLS with HC3 on the registered predictor list. With
     2 seeds per config (balanced, no missing runs) its fixed effects equal the random-intercept model's (D27).
     tau = 0.9 in every config, so J = tau - FPR and J_worst = tau - f_max ("conditional advantage", stats.py)
     are exact aliases of FPR and f_max and are not separately estimable; they are reported as aliases.
  C  REGISTERED reporting commitment (Addendum 1, A1): realised false-acceptance rate on wrong rollouts and
     acceptance rate on correct rollouts, per run, from every logged verifier decision.
  D  EXPLORATORY (post hoc): aggregate verifier quality vs heterogeneity across all 48 configs (both arms):
     nested OLS fits and LOCO MSE for {FPR}, {FPR, f_max}, {FPR, V}, {FPR, f_max, V}; and the same with the
     realised window FPR in place of the design FPR.
  E  EXPLORATORY (post hoc) robustness of b_V: leave-one-config-out range; run-level OLS with config-clustered
     SE; rank (Spearman) partial correlation of Y and V given f_max.
  F  EXPLORATORY: unusual runs (largest within-config seed gaps, hacking-flagged runs) and whether seed spread
     grows with V (the pilot's high-V seed spread).
  G  Descriptive: compute used.

Writes results/grid/supplement.json and results/grid/runs_realised.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from experiments.analyse_grid import config_table
from vf_stats import confirmatory as cf
from vf_stats.predictor_comparison import loco_mse

ROOT = Path(__file__).resolve().parents[1]
SECONDARY = ["FPR", "fmax", "trigger", "V", "kappa3", "V_q2", "max_qf"]  # registered list minus exact aliases


def tab(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        t = st.t.ppf(0.975, r["df"])
        out.append({**r, "ci95": [r["coef"] - t * r["se_hc3"], r["coef"] + t * r["se_hc3"]]})
    return out


def vif(X: np.ndarray, names: list[str]) -> dict:
    out = {}
    for j, n in enumerate(names):
        others = np.delete(X, j, axis=1)
        A = np.c_[np.ones(len(X)), others]
        b, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        r2 = 1 - np.var(X[:, j] - A @ b) / np.var(X[:, j])
        out[n] = float(1 / (1 - r2)) if r2 < 1 else math.inf
    return out


def realised(run_dir: Path, T: int) -> dict:
    vl = pd.concat([pd.read_parquet(p, columns=["step", "mode", "correct", "reward"])
                    for p in sorted((run_dir / "verifier_log").glob("*.parquet"))])
    vl = vl[vl["step"] >= 1]
    wrong, right = vl[~vl["correct"].astype(bool)], vl[vl["correct"].astype(bool)]
    win = (vl["step"] >= T / 2) & (vl["step"] <= T)
    ww = wrong[(wrong["step"] >= T / 2) & (wrong["step"] <= T)]
    rw = right[(right["step"] >= T / 2) & (right["step"] <= T)]
    per_mode = wrong.groupby("mode")["reward"].agg(["mean", "size"])
    return {"F_real_all": float(wrong["reward"].mean()), "F_real_window": float(ww["reward"].mean()),
            "tau_real_all": float(right["reward"].mean()), "tau_real_window": float(rw["reward"].mean()),
            "wrong_share_window": float(len(ww) / max(1, int(win.sum()))),
            "n_decisions": int(len(vl)),
            **{f"f_real_{m}": float(r["mean"]) for m, r in per_mode.iterrows()}}


def clustered_ols(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.c_[np.ones(len(y)), X]
    XtX_inv = np.linalg.inv(A.T @ A)
    b = XtX_inv @ A.T @ y
    e = y - A @ b
    meat = np.zeros((A.shape[1], A.shape[1]))
    for g in np.unique(groups):
        s = A[groups == g].T @ e[groups == g]
        meat += np.outer(s, s)
    G, n, k = len(np.unique(groups)), len(y), A.shape[1]
    c = G / (G - 1) * (n - 1) / (n - k)
    return b, np.sqrt(np.diag(c * XtX_inv @ meat @ XtX_inv))


def partial_spearman(y, x, z) -> float:
    ry, rx, rz = (st.rankdata(v) for v in (y, x, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(st.pearsonr(res(ry), res(rx)).statistic)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(ROOT / "runs" / "grid"))
    ap.add_argument("--grid-dir", default=str(ROOT / "configs" / "grid"))
    ap.add_argument("--results", default=str(ROOT / "results" / "grid"))
    args = ap.parse_args(argv)
    T = int(json.loads((ROOT / "configs" / "pilot" / "selection.json").read_text())["T"])
    res_dir = Path(args.results)
    table = pd.read_csv(Path(args.grid_dir) / "runs_grid.csv")
    runs = pd.read_csv(res_dir / "runs.csv")
    cfg = config_table(table).set_index("config_id")
    out: dict = {"T": T, "status": "post-registration supplement (D33); see section labels in the module docstring"}

    # C: realised rates (needed by D)
    rr = []
    for r in runs.itertuples():
        rr.append({"run": r.run, **realised(Path(args.runs_root) / r.run, T)})
    rr = pd.DataFrame(rr)
    runs = runs.merge(rr, on="run")
    runs = runs.merge(cfg[["FPR", "V", "fmax"]].rename(columns={"FPR": "FPR_design", "V": "V_design",
                                                                 "fmax": "fmax_design"}),
                      left_on="config_id", right_index=True)
    runs.to_csv(res_dir / "runs_realised.csv", index=False)
    out["C_realised_rates_registered_commitment"] = {
        arm: {"F_real_window_mean": float(g["F_real_window"].mean()),
              "F_real_window_min": float(g["F_real_window"].min()),
              "F_real_window_max": float(g["F_real_window"].max()),
              "F_real_all_mean": float(g["F_real_all"].mean()),
              "tau_real_all_mean": float(g["tau_real_all"].mean()),
              "tau_real_all_min": float(g["tau_real_all"].min()),
              "tau_real_all_max": float(g["tau_real_all"].max()),
              "design_FPR_values": sorted({round(float(x), 4) for x in g["FPR_design"]})}
        for arm, g in runs.groupby("arm")}
    m_runs = runs[runs["arm"] == "matched"]
    out["C_realised_rates_registered_commitment"]["matched_corr_F_real_window_vs_V"] = float(
        st.spearmanr(m_runs["F_real_window"], m_runs["V_design"]).statistic)

    # config means
    agg = runs.groupby("config_id").agg(Y=("Y", "mean"), n=("Y", "size"), Ysd=("Y", "std"),
                                       F_real_window=("F_real_window", "mean"), acc_T=("acc_T", "mean"))
    allc = cfg.join(agg, how="inner").reset_index()
    matched = allc[allc["arm"] == "matched"].reset_index(drop=True)

    # A: b_f
    h1 = cf.h1(matched)
    out["A_H1_full_table_registered"] = tab(pd.DataFrame(h1["table"]))
    out["A_H1_r2"] = float(1 - np.var(matched["Y"] - np.c_[np.ones(len(matched)), matched[["V", "fmax"]]]
                                       @ np.linalg.lstsq(np.c_[np.ones(len(matched)), matched[["V", "fmax"]]],
                                                         matched["Y"], rcond=None)[0]) / np.var(matched["Y"]))

    # B: registered secondary model, all arms
    X = allc[SECONDARY].to_numpy()
    y = allc["Y"].to_numpy()
    out["B_secondary_model_registered"] = {
        "n_configs": len(allc), "predictors": SECONDARY,
        "aliases_not_estimable": {"J": "tau - FPR (tau = 0.9 in all configs)",
                                  "J_worst / conditional advantage": "tau - f_max"},
        "tau_values": sorted({float(t) for t in allc["tau"]}),
        "table": tab(cf.ols_hc3(X, y, SECONDARY)), "vif": vif(X, SECONDARY),
        "loco_mse": loco_mse(X, y), "var_Y": float(np.var(y)),
    }

    # D: aggregate vs heterogeneity, all configs (exploratory)
    sets = {"FPR": ["FPR"], "FPR+fmax": ["FPR", "fmax"], "FPR+V": ["FPR", "V"], "FPR+fmax+V": ["FPR", "fmax", "V"],
            "FPRreal": ["F_real_window"], "FPRreal+fmax": ["F_real_window", "fmax"],
            "FPRreal+V": ["F_real_window", "V"], "FPRreal+fmax+V": ["F_real_window", "fmax", "V"],
            "null": []}
    D = {}
    for name, cols in sets.items():
        Xs = allc[cols].to_numpy() if cols else np.zeros((len(allc), 0))
        rec = {"loco_mse": loco_mse(Xs, y), "rel_mse": loco_mse(Xs, y) / float(np.var(y))}
        if cols:
            rec["table"] = tab(cf.ols_hc3(Xs, y, cols))
        D[name] = rec
    out["D_aggregate_vs_heterogeneity_exploratory"] = D

    # E: robustness (exploratory)
    loo = []
    for i in range(len(matched)):
        sub = matched.drop(index=i)
        loo.append(cf.h1(sub)["coef_V"])
    mr = runs[runs["arm"] == "matched"]
    b, se = clustered_ols(mr[["V_design", "fmax_design"]].to_numpy(), mr["Y"].to_numpy(), mr["config_id"].to_numpy())
    out["E_robustness_exploratory"] = {
        "loo_config_bV_min": float(min(loo)), "loo_config_bV_max": float(max(loo)),
        "loo_config_most_influential": str(matched["config_id"].iloc[int(np.argmax(np.abs(np.array(loo) - h1["coef_V"])))]),
        "run_level_clustered": {"coef_V": float(b[1]), "se_cluster": float(se[1]),
                                "ci95": [float(b[1] - st.t.ppf(.975, 35) * se[1]), float(b[1] + st.t.ppf(.975, 35) * se[1])],
                                "n_runs": len(mr), "n_clusters": int(mr["config_id"].nunique())},
        "spearman_partial_Y_V_given_fmax": partial_spearman(matched["Y"].to_numpy(), matched["V"].to_numpy(),
                                                            matched["fmax"].to_numpy()),
        "H1_without_runs_Y_below_0.3_sensitivity_only": (lambda k: tab(pd.DataFrame(cf.h1(
            cfg.join(k.groupby("config_id").agg(Y=("Y", "mean")), how="inner").reset_index())["table"])))(
            mr[mr["Y"] >= 0.3]),
        "H1_with_realised_FPR_covariate_mediator_not_confounder":tab(cf.ols_hc3(matched[["V", "fmax", "F_real_window"]].to_numpy(),
                                                        matched["Y"].to_numpy(), ["V", "fmax", "F_real_window"])),
    }

    # F: unusual runs (exploratory)
    gaps = runs.pivot_table(index="config_id", columns="seed", values="Y")
    gaps["absdiff"] = (gaps[1] - gaps[2]).abs()
    gaps = gaps.join(cfg[["arm", "V", "fmax", "FPR"]])
    top = gaps.sort_values("absdiff", ascending=False).head(6)
    mg = gaps[gaps["arm"] == "matched"]
    lev = st.spearmanr(mg["V"], mg["absdiff"])
    hk = runs[runs["hacking"] == 1]
    out["F_unusual_runs_exploratory"] = {
        "largest_seed_gaps": [{"config_id": k, "arm": r["arm"], "V": float(r["V"]), "fmax": float(r["fmax"]),
                               "Y_seed1": float(r[1]), "Y_seed2": float(r[2]), "absdiff": float(r["absdiff"])}
                              for k, r in top.iterrows()],
        "median_seed_gap": float(gaps["absdiff"].median()),
        "spearman_seed_gap_vs_V_matched": {"rho": float(lev.statistic), "p": float(lev.pvalue)},
        "hacking_flagged_runs": [{"run": r.run, "config_id": r.config_id, "arm": r.arm, "Y": float(r.Y),
                                  "acc_T": float(r.acc_T), "V": float(r.V_design)} for r in hk.itertuples()],
        "min_Y_run": runs.loc[runs["Y"].idxmin(), ["run", "config_id", "Y", "acc_T", "V_design"]].to_dict(),
        "max_Y_run": runs.loc[runs["Y"].idxmax(), ["run", "config_id", "Y", "acc_T", "V_design"]].to_dict(),
        "diverged_runs": int(runs["diverged"].sum()),
    }

    # F2: trajectories of the low-Y runs (exploratory description of the high-V seed split)
    traj = {}
    for r in runs[runs["Y"] < 0.3].itertuples():
        ev = [json.loads(l) for l in (Path(args.runs_root) / r.run / "eval.jsonl").read_text().splitlines() if l]
        f = cfg.loc[r.config_id, "f_dict"]
        top = max((k for k in f if k not in ("U", "I")), key=lambda k: f[k])
        traj[r.run] = {"most_accepted_mode": top, "f_top": float(f[top]),
                       "acc": [e["greedy_accuracy"] for e in ev], "q_top": [e["q_t"].get(top, 0.0) for e in ev]}
    out["F_unusual_runs_exploratory"]["low_Y_trajectories"] = traj
    mm = runs[runs["arm"] == "matched"].copy()
    hiV = mm["V_design"] > mm["V_design"].median()
    out["F_unusual_runs_exploratory"]["runs_Y_below_0.3"] = {"matched_highV": int((mm[hiV]["Y"] < 0.3).sum()),
                                                            "matched_lowV": int((mm[~hiV]["Y"] < 0.3).sum()),
                                                            "auxiliary": int((runs[(runs["arm"] == "auxiliary")]["Y"] < 0.3).sum())}

    # H: duplicate verifier configurations (disclosure + exploratory sensitivity)
    key = allc["f_dict"].apply(lambda d: tuple(sorted((k, round(float(v), 12)) for k, v in d.items())))
    allc["vkey"] = key.map({k: i for i, k in enumerate(dict.fromkeys(key))})
    split = json.loads((Path(args.grid_dir) / "split.json").read_text())
    held = set(split["heldout"])
    groups = [sorted(g["config_id"]) for _, g in allc.groupby("vkey") if len(g) > 1]
    um = allc[allc["arm"] == "matched"].drop_duplicates("vkey").reset_index(drop=True)
    ua = allc.drop_duplicates("vkey").reset_index(drop=True)
    h1u = cf.h1(um)
    # leave-one-unique-verifier-out paired comparison
    rng = np.random.default_rng(20261114)
    se_fv, se_f, se_z, s_used = [], [], [], []
    for i in range(len(um)):
        tr, te = um.drop(index=i), um.iloc[[i]]
        se_fv.append(float((te["Y"].iloc[0] - cf._fit_predict(tr[["fmax", "V"]].to_numpy(), tr["Y"].to_numpy(),
                                                               te[["fmax", "V"]].to_numpy())[0]) ** 2))
        se_f.append(float((te["Y"].iloc[0] - cf._fit_predict(tr[["fmax"]].to_numpy(), tr["Y"].to_numpy(),
                                                             te[["fmax"]].to_numpy())[0]) ** 2))
        s = cf.choose_s(tr)
        s_used.append(s)
        se_z.append(float((te["Y"].iloc[0] - cf._fit_predict(cf.cgf_column(tr, s)[:, None], tr["Y"].to_numpy(),
                                                             cf.cgf_column(te, s)[:, None])[0]) ** 2))
    se_fv, se_f, se_z = map(np.array, (se_fv, se_f, se_z))
    # registered held-out split with twins removed: held-out verifiers that have no identical config in training
    train_keys = set(allc[(allc["arm"] == "matched") & ~allc["config_id"].isin(held)]["vkey"])
    pure_held = um[um["config_id"].isin(held) | um["vkey"].isin(
        allc[allc["config_id"].isin(held)]["vkey"])]
    pure_held = pure_held[~pure_held["vkey"].isin(train_keys)]
    um_train = um[um["vkey"].isin(train_keys)]
    pure = cf.h2_h3(um_train, pure_held, n_boot=10_000) if len(pure_held) >= 3 else None
    out["H_duplicate_configs_disclosure"] = {
        "status": "design property of the registered grid (D26), found after the registered analysis (D33); "
                  "runs are deterministic in (verifier, seed), so duplicates give identical outcomes",
        "n_unique_verifiers_all": int(allc["vkey"].nunique()), "n_unique_matched": len(um),
        "n_configs_matched": int((allc["arm"] == "matched").sum()),
        "duplicate_groups": groups,
        "heldout_with_identical_training_twin": sorted(c for c in held if allc.loc[allc["config_id"] == c, "vkey"].iloc[0]
                                                       in train_keys),
        "H1_unique_matched_exploratory": tab(pd.DataFrame(h1u["table"])),
        "loo_unique_matched_exploratory": {
            "mse_fmax_V": float(se_fv.mean()), "mse_fmax": float(se_f.mean()), "mse_cgf": float(se_z.mean()),
            "fmaxV_minus_fmax": cf._paired_boot(se_fv - se_f, 10_000, 11),
            "cgf_minus_fmaxV": cf._paired_boot(se_z - se_fv, 10_000, 12),
            "s_chosen_median": float(np.median(s_used)), "var_Y": float(np.var(um["Y"]))},
        "heldout_without_twins_exploratory": pure,
        "secondary_model_unique_all_exploratory": tab(cf.ols_hc3(ua[SECONDARY].to_numpy(), ua["Y"].to_numpy(), SECONDARY)),
        "aggregate_vs_V_unique_all_exploratory": {
            name: {"rel_mse": loco_mse(ua[c].to_numpy(), ua["Y"].to_numpy()) / float(np.var(ua["Y"])),
                   "table": tab(cf.ols_hc3(ua[c].to_numpy(), ua["Y"].to_numpy(), c))}
            for name, c in {"FPR": ["FPR"], "FPR+fmax": ["FPR", "fmax"], "FPR+V": ["FPR", "V"],
                            "FPR+fmax+V": ["FPR", "fmax", "V"], "FPRreal+fmax+V": ["F_real_window", "fmax", "V"]}.items()},
        "spearman_Y_V_unique_matched": float(st.spearmanr(um["V"], um["Y"]).statistic),
    }
    del rng

    # G: compute
    secs = []
    for r in runs.itertuples():
        met = [json.loads(l) for l in (Path(args.runs_root) / r.run / "metrics.jsonl").read_text().splitlines() if l]
        secs += [m["step_seconds"] for m in met if "step_seconds" in m]
    out["G_compute"] = {"median_step_seconds": float(np.median(secs)), "p95_step_seconds": float(np.percentile(secs, 95)),
                        "train_gpu_hours_sum": float(np.sum(secs) / 3600), "n_steps": len(secs)}

    (res_dir / "supplement.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps({k: out[k] for k in ("A_H1_full_table_registered",)}, default=float)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
