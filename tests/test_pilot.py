"""Pilot tooling (D19): configuration builder and the pre-declared LR / horizon rules."""

import pandas as pd
import pytest

from experiments.make_pilot_configs import build
from experiments.pilot_select import choose_lr, diverged, horizon
from verifiers.qref import merge_small_modes

Q5 = {"I": 0.0301, "LEN": 0.1623, "MULTI": 0.4201, "NEG": 0.0, "ONE_HI": 0.181, "ONE_LO": 0.2065}


def test_pilot_configs():
    q = merge_small_modes(Q5)[0]
    lr_cfgs, het = build(q, 0.3, 0.9)
    homog, perfect = lr_cfgs
    assert abs(homog.Fbar - 0.3) < 1e-11 and homog.tau == 0.9
    assert perfect.tau == 1.0 and all(v == 0.0 for v in perfect.f.values())
    Vs = [c.V for c in het]
    assert Vs[0] < Vs[1] < Vs[2]  # V floor < mid < high
    for c in het:
        assert abs(c.Fbar - 0.3) < 1e-11 and c.tau == 0.9 and all(0 <= v <= 1 for v in c.f.values())
    fm = het[3]
    assert max(fm.f[k] for k in q if k != "I") == pytest.approx(0.95) and fm.V < Vs[2]


def _ev(acc):
    return pd.DataFrame({"step": [50 * i for i in range(len(acc))], "greedy_accuracy": acc})


def test_lr_rule_and_ties_and_divergence():
    flat = [0.35] + [0.6] * 16
    runs = {5e-7: _ev([0.35] + [0.5] * 16), 1e-6: _ev(flat), 3e-6: _ev([0.35] + [0.603] * 16)}
    lr, scores = choose_lr(runs)
    assert lr == 1e-6  # 3e-6 is within the 0.005 tie band, so the smaller LR wins
    runs[1e-6] = _ev([0.35, 0.5, 0.005, 0.004, 0.003] + [0.9] * 12)
    assert diverged(runs[1e-6]) and choose_lr(runs)[0] == 3e-6


def test_horizon_rule():
    ev = _ev([0.35, 0.6, 0.75, 0.8, 0.82, 0.84, 0.85, 0.85])
    # gain = 0.5; 90 % => 0.80 first reached at step 150 -> 150
    assert horizon(ev) == 150
    assert horizon(_ev([0.35, 0.9, 0.9])) == 100  # floor
    with pytest.raises(ValueError):
        horizon(_ev([0.4, 0.3, 0.2]))
