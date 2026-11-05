"""Confirmatory analysis (§15) on synthetic data with known effects, before any real data."""

import numpy as np
import pandas as pd
import pytest

from verifiers.matched_verifiers import design_grid
from verifiers.qref import merge_small_modes
from vf_stats import confirmatory as cf

Q5 = {"I": 0.0301, "LEN": 0.1623, "MULTI": 0.4201, "NEG": 0.0, "ONE_HI": 0.181, "ONE_LO": 0.2065}


def _configs():
    q = merge_small_modes(Q5)[0]
    cfgs = design_grid(q, 0.3, 0.9, V_fracs=[0.1, 0.3, 0.5, 0.7, 0.9], fmax_levels=[None, 0.6, 0.95],
                       n_per_cell=2, seed=11)
    rows = []
    for c in cfgs:
        names, qq, ff = c.vectors()
        p = c.predictors()
        rows.append({"config_id": c.config_id, "V": p["V"], "fmax": p["fmax"], "tau": c.tau, "q": qq, "f": ff})
    return pd.DataFrame(rows)


def _with_y(cfg, bV, bf, sd, seed):
    rng = np.random.default_rng(seed)
    out = cfg.copy()
    out["Y"] = 0.5 + bV * out["V"] + bf * out["fmax"] + rng.normal(0, sd, len(out))
    return out


def test_window_logit_and_clipping():
    ev = pd.DataFrame({"step": [0, 50, 100, 150, 200], "greedy_accuracy": [0.3, 0.5, 0.6, 0.7, 0.8]})
    assert cf.window_logit(ev, 200, 1000) == pytest.approx(np.log(0.7 / 0.3))  # steps 100..200
    assert np.isfinite(cf.window_logit(ev.assign(greedy_accuracy=1.0), 200, 1000))


def test_hc3_matches_known_coefficients():
    cfg = _with_y(_configs(), bV=-8.0, bf=-1.0, sd=1e-6, seed=0)
    r = cf.h1(cfg)
    assert r["coef_V"] == pytest.approx(-8.0, rel=1e-3) and r["p"] < 1e-6


def test_h1_false_positive_rate_is_nominal():
    cfg = _configs()
    ps = [cf.h1(_with_y(cfg, 0.0, -1.0, 0.1, s))["p"] for s in range(300)]
    assert 0.01 <= np.mean(np.array(ps) < 0.05) <= 0.11


def test_h2_detects_v_and_holm_is_monotone():
    cfg = _with_y(_configs(), bV=-15.0, bf=-1.0, sd=0.02, seed=3)
    held = set(cfg["config_id"].iloc[::3])
    res = cf.analyse(cfg, held, n_boot=2000)
    assert res["H2"]["mean_diff"] < 0 and res["H2"]["ci95"][1] < 0
    h = res["holm"]
    assert all(h[k] >= p for k, p in (("H1", res["H1"]["p"]), ("H2", res["H2"]["p"]), ("H3", res["H3"]["p"])))
    assert res["n_train"] + res["n_heldout"] == len(cfg)


def test_icc():
    runs = pd.DataFrame({"config_id": np.repeat(["a", "b", "c", "d"], 2), "Y": [0, 0.01, 1, 1.01, 2, 2.01, 3, 3.01]})
    assert cf.icc(runs)["icc"] > 0.99
    assert cf.config_means(runs)["n_seeds"].tolist() == [2, 2, 2, 2]
