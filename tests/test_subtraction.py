"""Subtraction task (§9.3 ladder step 2, DECISIONS.md D13): splits, taxonomy sub-v1,
synthetic recovery (§9.3 item 5) and the census core on a subtraction frame."""

import numpy as np
import pandas as pd
import pytest

from error_modes.classify_errors import (DESIGNED_MODES_SUB, borrow_drop_answers, classify,
                                         classify_value_sub)
from experiments import census_core
from vf_datasets.arithmetic import check_disjoint, frame_sha256, make_splits


# ---------------- splits ----------------

def test_sub_splits_ordered_disjoint_and_separate_from_addition():
    sizes = {"train": 500}
    sub = make_splits(6, sizes=sizes, task="sub")
    check_disjoint(sub)
    for df in sub.values():
        assert (df["a"] > df["b"]).all()
        assert (df["answer"] == df["a"] - df["b"]).all()
        assert df["prompt"].iloc[0].startswith(f"Compute {df['a'].iloc[0]} - {df['b'].iloc[0]}.")
        assert df["prompt_id"].iloc[0].startswith(df["prompt_id"].iloc[0].split("-")[0] + "-sub-n6-")
    add = make_splits(6, sizes=sizes)
    assert frame_sha256(add["census"]) == frame_sha256(make_splits(6, sizes=sizes, task="add")["census"])
    add_pairs = set(zip(add["census"]["a"], add["census"]["b"]))
    assert not add_pairs & set(zip(sub["census"]["a"], sub["census"]["b"]))


# ---------------- hand-checked taxonomy ----------------

def test_borrow_drop_examples():
    # 523 - 187 = 336: borrows arise in the units and tens columns
    assert borrow_drop_answers(523, 187) == {346, 436, 446}
    # chained borrow across zeros: 10000 - 1 = 9999
    assert 10009 in borrow_drop_answers(10000, 1) and 19999 in borrow_drop_answers(10000, 1)
    # no borrow needed -> no dropped-borrow answers
    assert borrow_drop_answers(987, 123) == frozenset()


def test_sub_classification_and_diagnostics():
    assert classify(523, 187, "Answer: 336", "sub").primary == "C"
    assert classify(523, 187, "336", "sub").primary == "C"  # bare integer (taxonomy v2 parser)
    assert classify_value_sub(523, 187, 446).primary == "BORROW"
    assert classify_value_sub(523, 187, 523).primary == "ECHO"
    assert classify_value_sub(523, 187, 33).primary == "TRUNC"
    assert classify_value_sub(523, 187, 363).primary == "TRANS"
    assert classify_value_sub(523, 187, 396).primary == "DIGIT"
    c = classify_value_sub(523, 187, 346)  # one dropped borrow is also a one-digit slip
    assert c.matches == ("BORROW", "DIGIT") and c.ambiguous
    assert classify_value_sub(523, 187, 710).diag == ("OPSWAP",)
    assert classify_value_sub(523, 187, -336).diag == ("NEG",)
    assert classify_value_sub(523, 187, 464).diag == ("SFL",)
    assert classify(523, 187, "no answer", "sub").primary == "I"
    with pytest.raises(ValueError):
        classify_value_sub(187, 523, 1)


# ---------------- synthetic recovery (§9.3.5) ----------------

def _inject_sub(mode, a, b, rng):
    ds = str(a - b)
    if mode == "ECHO":
        return [a, b, int(f"{a}{b}"), int(f"{b}{a}")][rng.integers(4)]
    if mode == "BORROW":
        opts = sorted(borrow_drop_answers(a, b))
        return None if not opts else opts[rng.integers(len(opts))]
    if mode == "TRUNC":
        if len(ds) < 2:
            return None
        k = int(rng.integers(1, len(ds)))
        return int(ds[:k] if rng.random() < 0.5 else ds[-k:])
    if mode == "TRANS":
        pairs = [i for i in range(len(ds) - 1) if ds[i] != ds[i + 1] and not (i == 0 and ds[1] == "0")]
        if not pairs:
            return None
        i = pairs[rng.integers(len(pairs))]
        return int(ds[:i] + ds[i + 1] + ds[i] + ds[i + 2:])
    if mode == "DIGIT":
        i = int(rng.integers(len(ds)))
        choices = [d for d in "0123456789" if d != ds[i] and not (i == 0 and d == "0" and len(ds) > 1)]
        return int(ds[:i] + choices[rng.integers(len(choices))] + ds[i + 1:])
    raise ValueError(mode)


@pytest.mark.parametrize("n_digits", [5, 8])
@pytest.mark.parametrize("mode", DESIGNED_MODES_SUB)
def test_sub_synthetic_recovery(mode, n_digits):
    rng = np.random.default_rng(abs(hash(("sub", mode, n_digits))) % 2**32)
    lo, hi = 10 ** (n_digits - 1), 10**n_digits
    n = matched = 0
    while n < 1000:
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        if a == b:
            continue
        a, b = max(a, b), min(a, b)
        v = _inject_sub(mode, a, b, rng)
        if v is None or v == a - b:
            continue
        n += 1
        matched += mode in classify_value_sub(a, b, v).matches
    assert matched / n >= 0.99, (mode, matched / n)


# ---------------- census core on a subtraction frame ----------------

def test_census_core_uses_subtraction_classes():
    rows = []
    for i, (a, b, text) in enumerate([(523, 187, "336"), (523, 187, "446"), (523, 187, "396"),
                                      (523, 187, "523"), (984, 351, "633"), (984, 351, "1335"),
                                      (612, 345, "333"), (612, 345, "267")]):
        rows.append({"prompt_id": f"p{i % 4}", "a": a, "b": b, "completion": text})
    cls = census_core.classify_generations(pd.DataFrame(rows), "sub")
    q = census_core.q_ref_point(cls)
    assert set(q) == {"I", "ECHO", "BORROW", "TRUNC", "TRANS", "DIGIT", "U"}
    assert abs(sum(q.values()) - 1) < 1e-12
    cov = census_core.coverage_and_ambiguity(cls)
    assert cov["n_incorrect"] == 5 and cov["n_designed"] == 3  # BORROW, DIGIT, ECHO; 1335 and 333 are U
    assert set(cls.loc[cls["mode"] == "U", "diag"]) == {"OPSWAP", "SFL"}
