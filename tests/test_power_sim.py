import numpy as np
import pandas as pd

from vf_stats import power_sim as ps


def _design(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"V": rng.uniform(0, 0.17, n), "fmax": rng.uniform(0.3, 0.95, n)})


def test_power_increases_with_effect_and_R():
    d = _design()
    p0 = ps.power(d, 24, 0.0, 0.1, 0.1, n_rep=300, seed=1)
    p_small = ps.power(d, 12, -3.0, 0.1, 0.1, n_rep=300, seed=1)
    p_big = ps.power(d, 36, -3.0, 0.1, 0.1, n_rep=300, seed=1)
    assert p0 < 0.06  # size close to alpha = 0.0167
    assert p_big > p_small


def test_pilot_noise_conservative_floor():
    runs = pd.DataFrame({"config_id": ["a"] * 3 + ["b"] * 3, "Y": [1.0, 1.1, 0.9, 1.0, 1.1, 0.9]})
    n = ps.pilot_noise(runs)
    assert abs(n["sigma_s"] - 0.1) < 1e-12
    assert n["sigma_r"] == n["sigma_s"]  # no between-config spread -> floor at sigma_s


def test_choose_r_rules():
    assert ps.choose_R({12: 0.5, 24: 0.85, 36: 0.95}) == {"R": 24, "claim": "confirmatory test"}
    assert ps.choose_R({12: 0.1, 24: 0.2, 36: 0.4})["claim"].startswith("estimation")
    assert ps.choose_R({12: 0.3, 24: 0.6, 36: 0.7})["R"] == 36
