"""Phase 4 closure: diagnose the Gate 3′ split-half TV results (exploratory, post hoc).

Reads finished census run directories (classification.parquet + stability.json) and,
for each run:

1. recomputes the declared statistic with the unchanged gate code
   (`census_core.split_half_tv`) and checks it equals the stored value;
2. reports the half balance and each class's contribution to the TV;
3. builds two reference distributions for the TV:
   - *random prompt halves*: every prompt goes to a half with probability 1/2,
     independently. This is the same law as the declared hash split, so it is the
     sampling distribution of the declared statistic for this census;
   - *i.i.d. errors*: error labels are shuffled across prompts, keeping each half's
     error count. This is the TV a census of independent errors would give;
   the ratio of the two measures how much prompt clustering inflates the TV;
4. measures prompt-level clustering directly: the share of within-prompt error pairs
   with the same class, against sum_k q_k^2 (its value without clustering);
5. stratifies q by an operand feature (whether the true sum has n+1 digits) to see
   whether the prompt heterogeneity is predictable from the operands.

With `--qref-lock` and `--grid`, it also propagates the locked census's own
cluster-bootstrap uncertainty in q_ref into F̄ and V of the generated grid configs.

None of this changes a gate statistic, threshold, split, census or lock. It is a
post hoc diagnosis and is labelled as such wherever it is reported.

    uv run python -m experiments.diagnose_split_half_tv RUN_DIR ... \\
        --qref-run RUN_DIR --grid configs/grid --json docs/gate3_tv_diagnosis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import census_core

SEED = 20261107
N_DRAWS = 2000


def _q(modes: np.ndarray, classes: list[str]) -> np.ndarray:
    n = modes.size
    return np.array([(modes == c).sum() / n for c in classes]) if n else np.full(len(classes), np.nan)


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def _summ(x: np.ndarray, obs: float | None = None) -> dict:
    out = {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "p_gt_0.03": float((x > census_core.GATE_TV_MAX).mean()),
    }
    if obs is not None:
        out["observed_percentile"] = float((x < obs).mean() * 100)
    return out


def diagnose_run(run_dir: Path, rng: np.random.Generator) -> dict:
    cls = pd.read_parquet(run_dir / "classification.parquet")
    params = json.loads((run_dir / "params.json").read_text())
    stored = json.loads((run_dir / "stability.json").read_text())["split_half_tv"]
    tv = census_core.split_half_tv(cls)
    classes = census_core.incorrect_classes(census_core.scheme_of(cls))

    bad = cls.loc[~cls["correct"], ["prompt_id", "a", "b", "mode"]].reset_index(drop=True)
    modes = bad["mode"].to_numpy()
    q_all = _q(modes, classes)

    # declared halves (hash split, unchanged)
    half_p = {p: census_core.half_of_prompt(p) for p in cls["prompt_id"].unique()}
    bh = bad["prompt_id"].map(half_p).to_numpy()
    q0, q1 = _q(modes[bh == 0], classes), _q(modes[bh == 1], classes)
    contrib = {c: 0.5 * abs(q0[i] - q1[i]) for i, c in enumerate(classes)}

    # random prompt halves (sampling law of the declared split)
    pids = bad["prompt_id"].to_numpy()
    uniq, inv = np.unique(pids, return_inverse=True)
    all_prompts = cls["prompt_id"].unique()
    tv_perm = np.empty(N_DRAWS)
    q_perm_half = np.empty((N_DRAWS, len(classes)))
    for d in range(N_DRAWS):
        h = rng.integers(0, 2, uniq.size)[inv]
        a, b = _q(modes[h == 0], classes), _q(modes[h == 1], classes)
        tv_perm[d] = _tv(a, b)
        q_perm_half[d] = a
    # i.i.d. errors with the declared half sizes
    n0 = int((bh == 0).sum())
    tv_iid = np.empty(N_DRAWS)
    q_iid_half = np.empty((N_DRAWS, len(classes)))
    for d in range(N_DRAWS):
        m = rng.permutation(modes)
        a, b = _q(m[:n0], classes), _q(m[n0:], classes)
        tv_iid[d] = _tv(a, b)
        q_iid_half[d] = a
    deff = {
        c: float(q_perm_half[:, i].var() / q_iid_half[:, i].var()) if q_iid_half[:, i].var() > 0 else None
        for i, c in enumerate(classes)
    }

    # within-prompt concordance of error classes
    same = pairs = 0
    for _, grp in bad.groupby("prompt_id"):
        k = len(grp)
        if k < 2:
            continue
        vc = grp["mode"].value_counts().to_numpy()
        same += int((vc * (vc - 1) // 2).sum())
        pairs += k * (k - 1) // 2
    concord = same / pairs if pairs else float("nan")
    concord_null = float((q_all ** 2).sum())

    # operand stratification: does the true sum carry out to n+1 digits?
    n = int(params["n_digits"])
    s = bad["a"].astype(object) + bad["b"].astype(object)
    over = np.array([len(str(v)) > n for v in s])
    strata = {}
    for name, mask in (("sum_n_digits", ~over), ("sum_n_plus_1_digits", over)):
        strata[name] = {"n_errors": int(mask.sum()),
                        "q": {c: float(v) for c, v in zip(classes, _q(modes[mask], classes))}}
    tv_strata = _tv(_q(modes[over], classes), _q(modes[~over], classes)) if over.any() and (~over).any() else None
    # overflow share per declared half (does the split unbalance the strata?)
    over_half = {h: float(over[bh == h].mean()) for h in (0, 1)}

    # order split: first vs second half of prompts in census order (generation-batch drift)
    order = {p: i for i, p in enumerate(pd.unique(cls["prompt_id"]))}
    first = bad["prompt_id"].map(order).to_numpy() < len(order) / 2
    tv_order = _tv(_q(modes[first], classes), _q(modes[~first], classes))

    prompts_with_err = bad.groupby("prompt_id").size()
    return {
        "run_id": run_dir.name,
        "model": params["model"]["name"],
        "n_digits": n,
        "n_prompts": int(len(all_prompts)),
        "samples_per_prompt": int(params["samples"]),
        "base_accuracy": float(cls["correct"].mean()),
        "n_errors": int(len(bad)),
        "n_prompts_with_errors": int(prompts_with_err.size),
        "errors_per_erring_prompt": float(prompts_with_err.mean()),
        "tv_declared": tv,
        "tv_stored": stored,
        "tv_recomputed_equal": bool(math.isclose(tv, stored, rel_tol=0, abs_tol=1e-12)),
        "halves": {
            "prompts": [int(sum(1 for v in half_p.values() if v == h)) for h in (0, 1)],
            "errors": [int((bh == h).sum()) for h in (0, 1)],
            "overflow_share_of_errors": over_half,
        },
        "q_all": {c: float(v) for c, v in zip(classes, q_all)},
        "q_half0": {c: float(v) for c, v in zip(classes, q0)},
        "q_half1": {c: float(v) for c, v in zip(classes, q1)},
        "tv_contribution": contrib,
        "tv_random_prompt_halves": _summ(tv_perm, tv),
        "tv_iid_errors": _summ(tv_iid, tv),
        "tv_census_order_halves": tv_order,
        "tv_census_order_halves_percentile": float((tv_perm < tv_order).mean() * 100),
        "clustering_inflation_mean_tv": float(tv_perm.mean() / tv_iid.mean()),
        "design_effect_by_class": deff,
        "within_prompt_same_class_share": concord,
        "same_class_share_without_clustering": concord_null,
        "operand_strata": strata,
        "tv_between_operand_strata": tv_strata,
    }


def qref_uncertainty(run_dir: Path, grid: Path, lock: Path, rng: np.random.Generator) -> dict:
    """Propagate prompt-cluster bootstrap draws of q_ref into F̄ and V of the grid configs."""
    payload = json.loads(lock.read_text())["payload"]
    q_lock = payload["q_ref"]
    cls = pd.read_parquet(run_dir / "classification.parquet")
    classes = census_core.incorrect_classes(census_core.scheme_of(cls))
    bad = cls.loc[~cls["correct"], ["prompt_id", "mode"]]
    q_point = _q(bad["mode"].to_numpy(), classes)
    assert all(abs(q_point[i] - q_lock[c]) < 1e-12 for i, c in enumerate(classes)), "run does not reproduce the lock"
    assert hashlib.sha256((run_dir / "q_ref.json").read_bytes()).hexdigest() == payload["census_run_q_ref_sha256"]

    # NEG merges into LEN (Amendment 1); configs carry I, LEN, MULTI, ONE_HI, ONE_LO
    keys = ["I", "LEN", "MULTI", "ONE_HI", "ONE_LO"]
    def merged(qv: np.ndarray) -> np.ndarray:
        d = dict(zip(classes, qv))
        d["LEN"] = d["LEN"] + d.get("NEG", 0.0)
        return np.array([d[k] for k in keys])

    pids = bad["prompt_id"].to_numpy()
    modes = bad["mode"].to_numpy()
    uniq, inv = np.unique(pids, return_inverse=True)
    idx = [np.where(inv == j)[0] for j in range(uniq.size)]
    draws = np.empty((N_DRAWS, len(keys)))
    for d in range(N_DRAWS):
        pick = rng.integers(0, uniq.size, uniq.size)
        sel = np.concatenate([idx[j] for j in pick])
        draws[d] = merged(_q(modes[sel], classes))

    cfgs = []
    for sub in ("matched", "auxiliary"):
        for p in sorted((grid / sub).glob("*.json")):
            c = json.loads(p.read_text())
            f = np.array([float(c["f"][k]) for k in keys])
            cfgs.append((sub, c["config_id"], f, float(c["target_Fbar"]), c.get("target_V")))
    q0 = merged(q_point)
    out_cfg = []
    fbar_dev = np.empty((N_DRAWS, len(cfgs)))
    v_draw = np.empty((N_DRAWS, len(cfgs)))
    v_point = np.empty(len(cfgs))
    for j, (sub, cid, f, fbar_t, _) in enumerate(cfgs):
        fb0 = float(q0 @ f)
        v_point[j] = float(q0 @ (f - fb0) ** 2)
        fb = draws @ f
        v = ((f[None, :] - fb[:, None]) ** 2 * draws).sum(1)
        fbar_dev[:, j] = fb - fbar_t
        v_draw[:, j] = v
        out_cfg.append({"arm": sub, "config_id": cid, "Fbar_at_lock": fb0, "Fbar_target": fbar_t,
                        "Fbar_ci95": [float(np.percentile(fb, 2.5)), float(np.percentile(fb, 97.5))],
                        "V_at_lock": v_point[j],
                        "V_ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]})
    matched = [j for j, c in enumerate(cfgs) if c[0] == "matched"]
    rank_corr = []
    rp = pd.Series(v_point[matched]).rank().to_numpy()
    for d in range(N_DRAWS):
        rd = pd.Series(v_draw[d, matched]).rank().to_numpy()
        rank_corr.append(float(np.corrcoef(rp, rd)[0, 1]))
    rank_corr = np.array(rank_corr)
    vr = v_point[matched].max() - v_point[matched].min()
    return {
        "lock_sha256": json.loads(lock.read_text())["sha256"],
        "q_ref_bootstrap_sd": {k: float(draws[:, i].std()) for i, k in enumerate(keys)},
        "n_configs": len(cfgs),
        "max_abs_Fbar_deviation_p95_over_draws": float(np.percentile(np.abs(fbar_dev).max(1), 95)),
        "max_abs_Fbar_deviation_any_draw": float(np.abs(fbar_dev).max()),
        "matched_V_range_at_lock": float(vr),
        "max_abs_V_deviation_p95_over_draws": float(np.percentile(np.abs(v_draw - v_point).max(1), 95)),
        "max_abs_V_deviation_p95_as_share_of_range": float(np.percentile(np.abs(v_draw - v_point).max(1), 95) / vr),
        "matched_V_spearman_vs_lock": {"min": float(rank_corr.min()), "p05": float(np.percentile(rank_corr, 5)),
                                       "median": float(np.median(rank_corr))},
        "configs": out_cfg,
    }


def replication_tv(run_a: Path, run_b: Path) -> dict:
    """TV between two independent censuses, both classified under taxonomy v3."""
    qs = []
    for r in (run_a, run_b):
        cls = pd.read_parquet(r / "classification.parquet")
        if census_core.scheme_of(cls) != "add_v3":
            gens = pd.read_parquet(r / "generations.parquet")
            cls = census_core.classify_generations(gens, "add_v3")
        qs.append(census_core.q_ref_point(cls))
    keys = list(qs[0])
    return {"runs": [run_a.name, run_b.name], "q": qs,
            "tv": 0.5 * sum(abs(qs[0][k] - qs[1][k]) for k in keys)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--qref-run", type=Path, help="census run behind q_ref.lock")
    ap.add_argument("--qref-lock", type=Path, default=Path("configs/locked/q_ref.lock"))
    ap.add_argument("--grid", type=Path, help="configs/grid (matched/ and auxiliary/)")
    ap.add_argument("--replicate", type=Path, help="independent exploratory census to compare with --qref-run")
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED)
    out = {
        "label": "EXPLORATORY post hoc diagnosis (Phase 4 closure); no gate statistic, threshold, split or lock changed",
        "seed": SEED, "draws": N_DRAWS, "tv_threshold": census_core.GATE_TV_MAX,
        "runs": [diagnose_run(r, rng) for r in a.runs],
    }
    pct = np.array([r["tv_random_prompt_halves"]["observed_percentile"] for r in out["runs"]]) / 100
    from scipy.stats import kstest
    out["declared_tv_percentiles_vs_uniform"] = {
        "percentiles": pct.tolist(), "ks_stat": float(kstest(pct, "uniform").statistic),
        "ks_p": float(kstest(pct, "uniform").pvalue),
    }
    if a.qref_run and a.grid:
        out["q_ref_uncertainty"] = qref_uncertainty(a.qref_run, a.grid, a.qref_lock, rng)
    if a.qref_run and a.replicate:
        out["independent_replication"] = replication_tv(a.replicate, a.qref_run)
    a.json.write_text(json.dumps(out, indent=1) + "\n")
    for r in out["runs"]:
        p = r["tv_random_prompt_halves"]
        print(f"{r['run_id']:28s} acc {r['base_accuracy']:.3f} err {r['n_errors']:5d} TV {r['tv_declared']:.4f} "
              f"(eq {r['tv_recomputed_equal']}) rand-halves mean {p['mean']:.4f} p95 {p['p95']:.4f} "
              f"P>.03 {p['p_gt_0.03']:.2f} pct {p['observed_percentile']:.0f} | iid mean "
              f"{r['tv_iid_errors']['mean']:.4f} infl {r['clustering_inflation_mean_tv']:.2f} | "
              f"concord {r['within_prompt_same_class_share']:.3f} vs {r['same_class_share_without_clustering']:.3f} | "
              f"order-TV {r['tv_census_order_halves']:.4f} pct {r['tv_census_order_halves_percentile']:.0f}")
    print("KS of declared-TV percentiles vs U(0,1):", out["declared_tv_percentiles_vs_uniform"]["ks_p"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
