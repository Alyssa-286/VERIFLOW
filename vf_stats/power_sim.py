#!/usr/bin/env python
"""Power analysis for H1 by simulation (master plan §15), with the choices §15 leaves open fixed in D23.

H1: OLS  Ybar_r = b0 + bV V_r + bf fmax_r + e_r  (HC3), one row per matched configuration, Ybar_r the
mean of 2 seeds. Data model: Y_rs = b0 + bV V_r + u_r + e_rs, u_r ~ N(0, sigma_r^2), e_rs ~ N(0, sigma_s^2);
bf = 0 (fmax still enters the fitted model, so its collinearity with V costs power as it will in the grid).

Inputs, all from existing evidence:
- sigma_s: pooled within-config seed SD of Y in the heterogeneity pilot (4 configs x 3 seeds).
- sigma_r: conservative. Every between-config difference in the pilot counts as noise:
  sigma_r = max( sqrt(max(var(config means) - sigma_s^2 / 3, 0)), sigma_s ).
- bV ("effect from the shared-policy simulation scaled to the pilot's time scale"):
  in the Gate 2 outputs (figures/out/sim_shared_policy/outcomes.csv.gz; y = window-mean accuracy), per
  setting x family (y there is already the window-mean LOGIT accuracy; A0 is an accuracy), the horizon is
  the earliest T in {75, 150, 300} where the homogeneous config reaches 90 % of its maximal accuracy gain
  over A0 (the pilot's T rule, which is in accuracy). There, fit y ~ V + fmax over the family's configs and
  express the V effect as a share of the homogeneous logit gain: rho = bV_sim * (Vmax - Vfloor) / gain_sim,
  gain_sim = y_homog(T) - logit(A0). Real effect: bV = rho * gain_pilot / (Vmax - Vfloor)_real, with
  gain_pilot = mean Y of v0 in the pilot - logit(greedy accuracy at step 0).
  The median rho over setting x family is the planning effect; the 25th percentile of |rho| is the
  pessimistic sensitivity case.
- Design: V and fmax of R configs drawn without replacement, per replicate, from the standard (V, fmax)
  grid generated from q_ref.lock (V fracs {0, .25, .5, .75, 1} x fmax {free, 0.6, 0.95}, 3 per cell + homog).
- alpha: 0.05 / 3, the Holm threshold H1 must meet if it is the smallest of three p-values (conservative).

Rule (§15): R = the smallest of {12, 24, 36} with power >= 0.80 at the planning effect. If power(36) < 0.50
the confirmatory claim is downgraded to estimation with CIs, stated in the pre-registration in advance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from vf_stats.confirmatory import ols_hc3

ROOT = Path(__file__).resolve().parents[1]
R_OPTIONS = (12, 24, 36)
ALPHA = 0.05 / 3
TARGET, FLOOR = 0.80, 0.50
SIM_DIR = ROOT / "figures" / "out" / "sim_shared_policy"


def logit(a):
    a = np.clip(np.asarray(a, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(a / (1 - a))


def sim_rho(outcomes: pd.DataFrame, A0: float) -> pd.DataFrame:
    rows = []
    for (setting, fam), d in outcomes.groupby(["setting", "family"]):
        homog = d[d["grid_cell"] == "homog"].set_index("T")["y"].sort_index()
        acc_gain = 1 / (1 + np.exp(-homog.to_numpy())) - A0
        if acc_gain.max() <= 0:
            continue
        i = int(np.argmax(acc_gain >= 0.9 * acc_gain.max()))
        T = int(homog.index[i])
        g = float(homog.to_numpy()[i] - logit(A0))
        dT = d[d["T"] == T]
        tab = ols_hc3(dT[["V", "fmax"]].to_numpy(), dT["y"].to_numpy(), ["V", "fmax"]).set_index("term")
        vr = float(dT["V"].max() - dT["V"].min())
        rows.append({"setting": setting, "family": fam, "T": T, "gain": g,
                     "bV": float(tab.loc["V", "coef"]), "V_range": vr,
                     "rho": float(tab.loc["V", "coef"]) * vr / g})
    return pd.DataFrame(rows)


def pilot_noise(runs: pd.DataFrame) -> dict:
    """runs: columns config_id, seed, Y. Returns sigma_s, sigma_r and the config means."""
    g = runs.groupby("config_id")["Y"]
    sigma_s = float(math.sqrt((g.var(ddof=1) * (g.size() - 1)).sum() / (g.size() - 1).sum()))
    means = g.mean()
    k = float(g.size().mean())
    between = max(float(means.var(ddof=1)) - sigma_s ** 2 / k, 0.0)
    return {"sigma_s": sigma_s, "sigma_r": max(math.sqrt(between), sigma_s),
            "sigma_r_raw": math.sqrt(between), "config_means": means.to_dict()}


def power(design: pd.DataFrame, R: int, bV: float, sigma_s: float, sigma_r: float,
          n_rep: int = 2000, seeds_per_config: int = 2, alpha: float = ALPHA, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    X_all = design[["V", "fmax"]].to_numpy()
    hits = 0
    for _ in range(n_rep):
        idx = rng.choice(len(X_all), size=R, replace=False)
        X = X_all[idx]
        ybar = bV * X[:, 0] + rng.normal(0, sigma_r, R) + rng.normal(0, sigma_s, (R, seeds_per_config)).mean(1)
        p = float(ols_hc3(X, ybar, ["V", "fmax"]).set_index("term").loc["V", "p"])
        hits += p < alpha
    return hits / n_rep


def choose_R(powers: dict) -> dict:
    ok = [R for R in R_OPTIONS if powers[R] >= TARGET]
    if ok:
        return {"R": min(ok), "claim": "confirmatory test"}
    if powers[max(R_OPTIONS)] < FLOOR:
        return {"R": max(R_OPTIONS), "claim": "estimation with CIs (power(36) < 0.50, §15)"}
    return {"R": max(R_OPTIONS), "claim": f"confirmatory test, underpowered (power {powers[36]:.2f} < 0.80, stated in advance)"}


def standard_design(seed: int = 20261107) -> pd.DataFrame:
    from verifiers.matched_verifiers import design_grid
    from verifiers.qref import load_q_ref
    q = load_q_ref()
    base = json.loads((ROOT / "configs" / "base.json").read_text())["verifier"]
    cfgs = design_grid(q["q_ref"], base["Fbar"], base["tau"], [0.0, 0.25, 0.5, 0.75, 1.0], [None, 0.6, 0.95],
                       3, seed)
    return pd.DataFrame([{"config_id": c.config_id, **{k: float(c.predictors()[k]) for k in ("V", "fmax")}}
                         for c in cfgs])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate5", default=str(ROOT / "configs" / "pilot" / "gate5.json"))
    ap.add_argument("--acc0", type=float, required=True, help="greedy accuracy at step 0 (pilot)")
    ap.add_argument("--n-rep", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "configs" / "pilot" / "power.json"))
    args = ap.parse_args(argv)
    g5 = json.loads(Path(args.gate5).read_text())
    runs = pd.DataFrame([{"config_id": c, "seed": i, "Y": y} for c, r in g5["configs"].items()
                         for i, y in enumerate(r["Y_seeds"])])
    noise = pilot_noise(runs)
    params = json.loads((SIM_DIR / "params.json").read_text())
    rho = sim_rho(pd.read_csv(SIM_DIR / "outcomes.csv.gz"), float(params["A0"]))
    design = standard_design()
    v_range = float(design["V"].max() - design["V"].min())
    gain_pilot = float(np.mean(g5["configs"]["pilot_het_v0"]["Y_seeds"]) - logit(args.acc0))
    rho_med = float(rho["rho"].median())
    rho_pess = float(np.sign(rho_med) * rho["rho"].abs().quantile(0.25))
    out = {"rule": "D23", "alpha": ALPHA, "sigma": noise, "gain_pilot": gain_pilot, "V_range_real": v_range,
           "n_design_pool": len(design), "rho_median": rho_med, "rho_pessimistic": rho_pess,
           "rho_by_setting": rho.groupby("setting")["rho"].median().to_dict(), "powers": {}}
    for label, r in (("planning", rho_med), ("pessimistic", rho_pess)):
        bV = r * gain_pilot / v_range
        out["powers"][label] = {"bV": bV, **{str(R): power(design, R, bV, noise["sigma_s"], noise["sigma_r"],
                                                           n_rep=args.n_rep, seed=R) for R in R_OPTIONS}}
    out["choice"] = choose_R({R: out["powers"]["planning"][str(R)] for R in R_OPTIONS})
    text = json.dumps(out, indent=2, default=float)
    Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
