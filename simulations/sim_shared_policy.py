"""Gate 2 driver: does V add held-out predictive information under realistic dynamics? (master plan §12.3)

For each dynamics setting (policy × optimiser × coupling × KL proxy × reward scaling):
  1. draw a base model (prompt population) with heterogeneous per-prompt error modes;
  2. 'census' = pooled q_ref of that population;
  3. build an exact matched (V, f_max) grid of verifiers on q_ref (verifiers/matched_verifiers.py);
  4. run finite-G GRPO for every config × seed (common prompt batches across configs);
  5. config-level outcome (seed mean of the window outcome) at several horizons;
  6. leave-one-config-out comparison of predictors (vf_stats/predictor_comparison.py).

Usage:  uv run python -m simulations.sim_shared_policy --out figures/out/sim_shared_policy [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.sim_grpo_tabular import GRPOSimConfig, make_population, simulate, window_outcome
from verifiers import matched_verifiers as mv
from vf_stats.predictor_comparison import forecast_eval

DESIGNED = ["ECHO", "CARRY", "TRUNC", "TRANS", "DIGIT"]
CLASSES = DESIGNED + ["U", "I"]


@dataclass
class Gate2Config:
    tau: float = 0.9
    Fbar: float = 0.3
    A0: float = 0.4
    n_prompts: int = 96
    n_families: int = 6
    seeds_per_config: int = 2
    V_fracs: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    fmax_levels: list = field(default_factory=lambda: [None, 0.6, 0.95])
    n_per_cell: int = 2
    prompt_conc: float = 20.0
    seed: int = 20261116
    settings: dict = field(default_factory=lambda: {
        # name: GRPOSimConfig overrides (learning rates calibrated so the homogeneous verifier
        # gains substantially but does not saturate by the final step; see docs/GATE2.md)
        "tabular-sgd": dict(policy="tabular", optimizer="sgd", lr=10.0),
        "tabular-adam": dict(policy="tabular", optimizer="adam", lr=0.03),
        "linear-sgd": dict(policy="linear", optimizer="sgd", lr=0.1),
        "linear-adam": dict(policy="linear", optimizer="adam", lr=0.01),
        "linear-adam-coupled": dict(policy="linear", optimizer="adam", lr=0.01, coupling=0.6),
        "linear-adam-kl": dict(policy="linear", optimizer="adam", lr=0.01, l2_to_init=0.05),
        "linear-adam-drgrpo": dict(policy="linear", optimizer="adam", lr=0.01, scale_rewards=False),
        "mlp-adam": dict(policy="mlp", optimizer="adam", lr=0.01),
        "mlp-adam-coupled": dict(policy="mlp", optimizer="adam", lr=0.01, coupling=0.6),
    })
    base: dict = field(default_factory=lambda: asdict(GRPOSimConfig(steps=300, prompts_per_step=16, G=8)))


def draw_q_target(rng) -> np.ndarray:
    """Plausible census shape: designed modes Dirichlet(1), U ≈ 0.10, I ≈ 0.05."""
    qd = rng.dirichlet(np.ones(len(DESIGNED))) * 0.85
    q = np.r_[qd, 0.10, 0.05]
    return q / q.sum()


def run_setting(name: str, overrides: dict, g2: Gate2Config, horizons: list[int]):
    cfg = GRPOSimConfig(**{**g2.base, **overrides})
    rng = np.random.default_rng(g2.seed)  # same families for every setting
    compare_rows, outcome_rows = [], []
    for fam in range(g2.n_families):
        pop = make_population(draw_q_target(rng), g2.A0, g2.n_prompts, rng, prompt_conc=g2.prompt_conc, class_names=CLASSES)
        q_ref = dict(zip(CLASSES, pop.q_ref.tolist()))
        grid = mv.design_grid(q_ref, g2.Fbar, g2.tau, g2.V_fracs, g2.fmax_levels, g2.n_per_cell, seed=g2.seed + fam)
        F = np.array([[c.f[k] for k in CLASSES] for c in grid])
        Q = np.tile(pop.q_ref, (len(grid), 1))
        rho = np.c_[np.full(len(grid), g2.tau), F]
        rho_runs = np.repeat(rho, g2.seeds_per_config, axis=0)
        res = simulate(pop, rho_runs, cfg, seed=g2.seed * 7 + fam, batch_seed=g2.seed * 11 + fam)
        for T in horizons:
            y_runs = window_outcome(res["steps"], res["acc"], T)
            y = y_runs.reshape(len(grid), g2.seeds_per_config).mean(1)
            seed_sd = float(y_runs.reshape(len(grid), g2.seeds_per_config).std(1, ddof=1).mean()) if g2.seeds_per_config > 1 else np.nan
            cmp = forecast_eval(g2.tau, Q, F, y)
            # sign of V given f_max (standardised OLS coefficient)
            X = np.c_[np.ones(len(y)), [c.predictors()["fmax"] for c in grid], [c.predictors()["V"] for c in grid]]
            Xs = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
            beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(y)), Xs], (y - y.mean()) / max(y.std(), 1e-12), rcond=None)
            cmp = cmp.assign(setting=name, family=fam, T=T, sd_y=float(np.std(y)), seed_sd=seed_sd,
                             beta_V_given_fmax=float(beta[2]), beta_fmax_given_V=float(beta[1]), n_configs=len(grid))
            compare_rows.append(cmp)
            for i, c in enumerate(grid):
                outcome_rows.append(dict(setting=name, family=fam, T=T, config_id=c.config_id, grid_cell=c.grid_cell, y=y[i], **c.predictors()))
        # homogeneous-verifier trajectory, for calibration reporting
        homog = [i for i, c in enumerate(grid) if c.grid_cell == "homog"][0] * g2.seeds_per_config
        outcome_rows.append(dict(setting=name, family=fam, T=-1, config_id="homog_traj",
                                 grid_cell=json.dumps(res["acc"][:, homog].round(4).tolist()), y=np.nan))
    return pd.concat(compare_rows, ignore_index=True), pd.DataFrame(outcome_rows)


def summarise(compare: pd.DataFrame) -> pd.DataFrame:
    """Per setting × horizon: median relative LOCO MSE per predictor, paired deltas with bootstrap CIs."""
    rows = []
    boot_rng = np.random.default_rng(0)
    for (s, T), d in compare.groupby(["setting", "T"]):
        piv = d.pivot_table(index="family", columns="predictor", values="loco_mse")
        rel = d.pivot_table(index="family", columns="predictor", values="rel_mse").median()

        def paired(a, b):
            diff = (piv[a] - piv[b]).to_numpy() / d.groupby("family")["sd_y"].first().to_numpy() ** 2
            bs = [boot_rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

        dv = paired("fmax+V", "fmax")
        dc = paired("CGF", "fmax+V")
        dvn = paired("V", "null(intercept)")
        row = dict(setting=s, T=T, **{f"rel_{k}": v for k, v in rel.items()},
                   dV_given_fmax=dv[0], dV_given_fmax_lo=dv[1], dV_given_fmax_hi=dv[2],
                   dCGF_vs_fmaxV=dc[0], dCGF_lo=dc[1], dCGF_hi=dc[2],
                   dV_vs_null=dvn[0], dV_vs_null_lo=dvn[1], dV_vs_null_hi=dvn[2],
                   beta_V_given_fmax_median=d.groupby("family")["beta_V_given_fmax"].first().median(),
                   frac_beta_V_negative=(d.groupby("family")["beta_V_given_fmax"].first() < 0).mean(),
                   seed_sd=d["seed_sd"].mean(), sd_y=d["sd_y"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="figures/out/sim_shared_policy")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset of setting names")
    args = ap.parse_args(argv)
    g2 = Gate2Config()
    if args.quick:
        g2 = replace(g2, n_families=2, n_prompts=48, V_fracs=[0.0, 0.5, 1.0], fmax_levels=[None, 0.95], n_per_cell=1)
        g2.base["steps"] = 100
    steps = g2.base["steps"]
    horizons = [steps // 4, steps // 2, steps]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    comps, outs = [], []
    for name, ov in g2.settings.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        c, o = run_setting(name, ov, g2, horizons)
        comps.append(c)
        outs.append(o)
        print(f"[gate2] {name}: {time.time() - t0:.0f}s", flush=True)
    compare = pd.concat(comps, ignore_index=True)
    outcomes = pd.concat(outs, ignore_index=True)
    compare.to_csv(out / "predictor_comparison.csv.gz", index=False)
    outcomes.to_csv(out / "outcomes.csv.gz", index=False)
    (out / "params.json").write_text(json.dumps(asdict(g2), indent=2, default=str))
    summ = summarise(compare)
    summ.to_csv(out / "summary.csv", index=False)
    cols = ["setting", "T", "rel_V", "rel_fmax", "rel_fmax+V", "rel_CGF", "rel_V_q2", "dV_given_fmax", "dV_given_fmax_lo",
            "dV_given_fmax_hi", "dCGF_vs_fmaxV", "frac_beta_V_negative", "seed_sd", "sd_y"]
    print(summ[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
