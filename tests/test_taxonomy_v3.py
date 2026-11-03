"""Taxonomy v3 (Amendment 1, D15): outcome partition of wrong addition answers.
Hand cases, exhaustiveness/disjointness, synthetic recovery (§9.3 item 5), and the
census core / audit export on a v3 frame."""

import numpy as np
import pandas as pd
import pytest

from error_modes.classify_errors import DESIGNED_MODES_V3, classify, classify_value_v3, v3_label
from experiments import census_core


def test_v3_hand_cases():
    # 12345 + 54321 = 66666 (L = 5, high half = positions 0, 1)
    a, b = 12345, 54321
    assert classify(a, b, "Answer: 66666", "add_v3").primary == "C"
    assert classify(a, b, "66666", "add_v3").primary == "C"
    assert classify(a, b, "The answer is 66666", "add_v3").primary == "I"
    assert classify_value_v3(a, b, 76666).primary == "ONE_HI"   # position 0
    assert classify_value_v3(a, b, 67666).primary == "ONE_HI"   # position 1
    assert classify_value_v3(a, b, 66766).primary == "ONE_LO"   # position 2 = floor(5/2)
    assert classify_value_v3(a, b, 66667).primary == "ONE_LO"
    assert classify_value_v3(a, b, 66677).primary == "MULTI"
    assert classify_value_v3(a, b, 6666).primary == "LEN"
    assert classify_value_v3(a, b, 166666).primary == "LEN"
    assert classify_value_v3(a, b, -66666).primary == "NEG"
    c = classify_value_v3(a, b, 12345)  # echo of a: same length, several digits differ
    assert c.primary == "MULTI" and c.matches == ("MULTI",) and "ECHO" in c.diag
    assert not c.ambiguous


def test_v3_is_an_exhaustive_disjoint_partition():
    rng = np.random.default_rng(15)
    for _ in range(3000):
        a, b = int(rng.integers(10**4, 10**5)), int(rng.integers(10**4, 10**5))
        v = int(rng.integers(-10**6, 10**6))
        c = classify_value_v3(a, b, v)
        if v == a + b:
            assert c.primary == "C"
            continue
        assert c.primary in DESIGNED_MODES_V3 and c.matches == (c.primary,)


def _inject_v3(mode, s, rng):
    ds = str(s)
    L = len(ds)
    if mode == "NEG":
        return -int(rng.integers(1, 10 * s))
    if mode == "LEN":
        return int(ds[:-1]) if rng.random() < 0.5 else int(ds + str(rng.integers(10)))
    if mode in ("ONE_HI", "ONE_LO"):
        lo, hi = (0, L // 2) if mode == "ONE_HI" else (L // 2, L)
        i = int(rng.integers(lo, hi))
        choices = [d for d in "0123456789" if d != ds[i] and not (i == 0 and d == "0")]
        return int(ds[:i] + choices[rng.integers(len(choices))] + ds[i + 1:])
    if mode == "MULTI":
        idx = rng.choice(L, size=int(rng.integers(2, L + 1)), replace=False)
        out = list(ds)
        for i in idx:
            out[i] = str(rng.choice([d for d in "0123456789" if d != ds[i] and not (i == 0 and d == "0")]))
        return int("".join(out))
    raise ValueError(mode)


@pytest.mark.parametrize("n_digits", [5, 8])
@pytest.mark.parametrize("mode", DESIGNED_MODES_V3)
def test_v3_synthetic_recovery(mode, n_digits):
    rng = np.random.default_rng(abs(hash(("v3", mode, n_digits))) % 2**32)
    lo, hi = 10 ** (n_digits - 1), 10**n_digits
    n = hit = 0
    while n < 1000:
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        v = _inject_v3(mode, a + b, rng)
        if v == a + b:
            continue
        n += 1
        hit += classify_value_v3(a, b, v).primary == mode
    assert hit / n >= 0.99, (mode, hit / n)


def test_census_core_and_audit_on_v3_frame(tmp_path, monkeypatch):
    rows = []
    for i, text in enumerate(["66666", "76666", "66766", "66677", "6666", "x", "66666", "67666"]):
        rows.append({"prompt_id": f"p{i % 4}", "a": 12345, "b": 54321, "completion": text})
    cls = census_core.classify_generations(pd.DataFrame(rows), "add_v3")
    q = census_core.q_ref_point(cls)
    assert set(q) == {"I", "NEG", "LEN", "ONE_HI", "ONE_LO", "MULTI"}
    cov = census_core.coverage_and_ambiguity(cls)
    assert cov["ambiguity_rate"] == 0.0 and cov["n_incorrect"] == 6 and cov["n_designed"] == 5
    assert v3_label(66766, 66666) == "ONE_LO"
    from experiments import check_construct_gate as gate
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    blind, key = gate.write_audit_sample(cls, tmp_path, n=6, seed=1)
    text = (tmp_path / "audit_instructions.md").read_text(encoding="utf-8")
    assert "ONE_HI" in text and "CARRY" not in text
    assert "mode" not in blind.read_text().splitlines()[0]
