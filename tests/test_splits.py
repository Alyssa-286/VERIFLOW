import pytest

from vf_datasets.arithmetic import PROMPT_TEMPLATE_SHA256, check_disjoint, frame_sha256, make_splits, pair_key

SMALL = {"eval": 300, "eval_mode": 300, "census": 500, "train": 3000}


def test_splits_are_disjoint_and_correct():
    sp = make_splits(6, SMALL)
    check_disjoint(sp)
    for name, df in sp.items():
        assert len(df) == SMALL[name]
        assert (df["answer"] == df["a"] + df["b"]).all()
        assert df["a"].between(100000, 999999).all() and df["b"].between(100000, 999999).all()
        assert df["prompt_id"].is_unique


def test_disjoint_even_when_pairs_are_scarce():
    # 2-digit operands: 90*91/2 = 4095 unordered pairs; forces heavy de-duplication
    sp = make_splits(2, {"eval": 400, "eval_mode": 400, "census": 400, "train": 2000})
    check_disjoint(sp)


def test_deterministic_and_hash_stable():
    s1, s2 = make_splits(5, SMALL), make_splits(5, SMALL)
    for k in s1:
        assert frame_sha256(s1[k]) == frame_sha256(s2[k])
    s3 = make_splits(5, SMALL, base_seed=1)
    assert frame_sha256(s1["eval"]) != frame_sha256(s3["eval"])


def test_eval_set_independent_of_train_size():
    """Changing the training-set size must not change the eval set (precedence order)."""
    a = make_splits(7, {**SMALL, "train": 1000})
    b = make_splits(7, {**SMALL, "train": 5000})
    assert frame_sha256(a["eval"]) == frame_sha256(b["eval"])


def test_check_disjoint_detects_leak():
    sp = make_splits(4, {"eval": 50, "eval_mode": 0, "census": 50, "train": 50})
    sp["train"].loc[0, ["a", "b"]] = sp["eval"].loc[0, ["b", "a"]].to_numpy()  # swapped order still leaks
    with pytest.raises(AssertionError):
        check_disjoint(sp)


def test_template_hash_frozen():
    assert len(PROMPT_TEMPLATE_SHA256) == 64
    assert pair_key(5, 3) == (3, 5)
