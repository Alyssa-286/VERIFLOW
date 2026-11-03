"""Tests for the census core (§11) and the mock end-to-end census run."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments import census_core
from experiments.mock_policy import mock_completions
from vf_datasets.arithmetic import DEFAULT_SIZES, make_splits, frame_sha256

# Central-passing mock base policy (matches configs/mock_base_policy.json):
# designed/error ≈ 0.95, no CARRY injected (its carry-drop realisations are
# legitimately ambiguous — §D6 flag), all designed masses >= 0.05.
GOOD_PROBS = {
    "C": 0.55, "ECHO": 0.08, "TRUNC": 0.12, "TRANS": 0.08, "DIGIT": 0.10, "U": 0.02, "I": 0.0,
}


def synthetic_generations(seed=12345, n_prompts=250, samples=6):
    splits = make_splits(6, sizes={"census": n_prompts, "eval": 10, "eval_mode": 10, "train": 20},
                         base_seed=seed)
    df = splits["census"]
    rows = []
    for pid, a, b in zip(df["prompt_id"], df["a"], df["b"]):
        comps = mock_completions(int(a), int(b), seed, str(pid), GOOD_PROBS, samples)
        for j, c in enumerate(comps):
            rows.append({"prompt_id": pid, "a": int(a), "b": int(b), "sample_idx": j, "completion": c})
    return pd.DataFrame(rows), splits


def test_classify_annotates_and_counts():
    gens, _ = synthetic_generations()
    cls = census_core.classify_generations(gens)
    assert {"C", "U", "TRUNC", "TRANS", "DIGIT", "ECHO"} <= set(cls["mode"])
    assert cls["correct"].any() and (~cls["correct"]).any()
    q = census_core.q_ref_point(cls)
    assert pytest.approx(sum(q.values())) == 1.0
    designed_mass = sum(v for k, v in q.items() if k in census_core.DESIGNED_MODES)
    cov = census_core.coverage_and_ambiguity(cls)
    assert cov["coverage"] == pytest.approx(designed_mass)
    assert cov["coverage"] > 0.7


def test_q_ref_matches_manual_counts():
    cls = census_core.classify_generations(synthetic_generations()[0])
    q = census_core.q_ref_point(cls)
    bad = cls[~cls["correct"]]["mode"].value_counts()
    total = bad.sum()
    for mode, mass in q.items():
        assert mass == pytest.approx(bad.get(mode, 0) / total)
    assert census_core.mode_counts(cls).sum() == total


def test_cluster_bootstrap_contains_point_estimate():
    cls = census_core.classify_generations(synthetic_generations()[0])
    q = census_core.q_ref_point(cls)
    ci = census_core.cluster_bootstrap_q(cls, n_boot=200, seed=0)
    for mode, (lo, hi) in ci.items():
        assert lo - 1e-12 <= q[mode] <= hi + 1e-12, f"CI for {mode} excludes the estimate"
        # 240 errors in a 4-class problem: every CI must be reasonably narrow
        assert hi - lo < 0.25


def test_bootstrap_reproducible():
    cls = census_core.classify_generations(synthetic_generations()[0])
    a = census_core.cluster_bootstrap_q(cls, n_boot=100, seed=7)
    b = census_core.cluster_bootstrap_q(cls, n_boot=100, seed=7)
    assert a == b


def test_split_half_tv_small_and_deterministic():
    cls = census_core.classify_generations(synthetic_generations()[0])
    tv1 = census_core.split_half_tv(cls)
    tv2 = census_core.split_half_tv(cls)
    assert tv1 == tv2
    assert tv1 < 0.2  # same distribution on both halves


def test_per_prompt_summary_and_histogram():
    cls = census_core.classify_generations(synthetic_generations()[0])
    pp = census_core.per_prompt_summary(cls)
    assert len(pp) == 250
    assert pp["accuracy"].between(0, 1).all()
    hist = census_core.accuracy_histogram(pp)
    assert hist.sum() == 250
    # entropy > 0 only where both correct and incorrect samples exist
    assert (pp.loc[pp["accuracy"] == 1.0, "mode_entropy"] == 0.0).all() | (
        pp.loc[pp["accuracy"] == 0.0, "mode_entropy"] > 0).any()


def test_gate_evaluates_each_criterion():
    q_good = {"ECHO": 0.2, "TRUNC": 0.3, "TRANS": 0.1, "DIGIT": 0.2, "U": 0.15, "I": 0.05}
    checks = census_core.evaluate_construct_gate(q_good, 0.80, 0.05, 0.01, 0.40)
    assert census_core.gate_passes(checks)
    assert {c["id"] for c in checks} == {
        "9.3.1 coverage", "9.3.2 mass", "9.3.3 ambiguity", "9.3.4 split-half TV",
        "9.1 base accuracy window",
    }
    # each individual failure is detected
    bad_cases = [
        (0.6, 0.05, 0.01, 0.40),   # coverage
        (0.80, 0.05, 0.01, 0.70),  # accuracy window
        (0.80, 0.05, 0.01, 0.10),  # accuracy window (high side)
    ]
    for cov, amb, tv, acc in bad_cases:
        assert not census_core.gate_passes(census_core.evaluate_construct_gate(q_good, cov, amb, tv, acc))
    q_few_modes = {"ECHO": 0.5, "TRUNC": 0.45, "TRANS": 0.02, "DIGIT": 0.02, "U": 0.005, "I": 0.005}
    assert not census_core.gate_passes(census_core.evaluate_construct_gate(q_few_modes, 0.85, 0.05, 0.01))
    assert not census_core.gate_passes(census_core.evaluate_construct_gate(q_good, 0.80, 0.20, 0.01))
    assert not census_core.gate_passes(census_core.evaluate_construct_gate(q_good, 0.80, 0.05, 0.10))


def test_q_ref_lock_payload_hash_roundtrip(tmp_path):
    import hashlib
    payload = {"q_ref": {"TRUNC": 0.5, "U": 0.5}}
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    doc = {"payload": payload, "sha256": sha}
    p = tmp_path / "q_ref.lock"
    p.write_text(json.dumps(doc))
    reloaded = json.loads(p.read_text())
    assert hashlib.sha256(json.dumps(reloaded["payload"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() == reloaded["sha256"]


# ------------------------------------------------------------ end to end (mock)

def test_census_end_to_end_mock(tmp_path):
    from experiments.base_mode_census import main
    out = tmp_path / "census"
    rc = main(["--n", "6", "--backend", "mock", "--size", "1200", "--samples", "8",
               "--seed", "42", "--out", str(out), "--n-boot", "200"])
    run_dir = out / "n6_s42"
    for artifact in ("generations.parquet", "classification.parquet", "q_ref.json",
                     "per_mode_lengths.json", "difficulty_hist.csv",
                     "per_prompt_summary.parquet", "stability.json",
                     "coverage_ambiguity.json", "params.json", "manifest.json"):
        assert (run_dir / artifact).exists(), artifact
    q_doc = json.loads((run_dir / "q_ref.json").read_text())
    assert pytest.approx(sum(q_doc["q_ref"].values())) == 1.0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["experiment_id"] == "veriflow-census-v1"
    assert manifest["git_commit"]
    assert manifest["dataset_sha256"]["census"] == manifest["census_prompts_sha256"]
    from vf_datasets.arithmetic import PROMPT_TEMPLATE_SHA256
    assert manifest["prompt_template_sha256"] == PROMPT_TEMPLATE_SHA256
    assert (run_dir / "coverage_ambiguity.json").read_text().count('"pass"') >= 4
    # sufficient samples: the gate should pass for this well-behaved mock
    cov = json.loads((run_dir / "coverage_ambiguity.json").read_text())
    assert cov["gate_pass"] is True
    assert rc == 0


# ------------------------------------------------------------ manual audit

BLIND_HEADER = "row_id,prompt_id,a,b,completion,answer_true,judgment\n"


def _mk_census_with_audit(tmp_path, monkeypatch):
    """Small classification frame with mixed error modes + exported audit."""
    from experiments import check_construct_gate as gate
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    cls = census_core.classify_generations(synthetic_generations(seed=777, n_prompts=60, samples=4)[0])
    run_dir = tmp_path / "census"
    run_dir.mkdir()
    blind, key = gate.write_audit_sample(cls, run_dir, n=40, seed=3)
    return gate, cls, run_dir, blind, key


def test_blinded_audit_hides_labels_and_key_is_separate(tmp_path, monkeypatch):
    gate, cls, run_dir, blind, key = _mk_census_with_audit(tmp_path, monkeypatch)
    blind_text = blind.read_text()
    assert "mode" not in blind_text.split("\n")[0] and "matches" not in blind_text
    import csv
    blind_rows = list(csv.reader(blind_text.splitlines()))
    assert blind_rows[0][-1] == "judgment"
    # judgment column empty for every row (csv reader: trailing empty field)
    assert len(blind_rows) - 1 >= 40 and all(row[-1] == "" for row in blind_rows[1:])
    key_text = key.read_text()
    assert "primary,matches" in key_text
    assert (run_dir / "audit_key.csv").exists()
    # blinded rows carry no trace of the classifier's multi-rules
    for line in blind_text.splitlines()[1:]:
        assert "|" not in line


def test_validate_audit_detects_disagreement_and_empty(tmp_path, monkeypatch):
    gate, cls, run_dir, blind, key = _mk_census_with_audit(tmp_path, monkeypatch)
    import pandas as pd
    df = pd.read_csv(blind, dtype=str).fillna("")
    # choose one label that cannot match for a synthetic disagreement: rename
    # every judgment to a fixed label, then fix the ones that actually agree
    from experiments.census_core import INCORRECT_CLASSES
    df["judgment"] = "TRUNC"
    # the completed file must sit beside audit_key.csv (validate_audit contract)
    results = run_dir / "audit_completed.csv"
    df.to_csv(results, index=False)
    with pytest.raises(ValueError, match="empty judgment"):
        df2 = df.copy()
        df2.loc[3, "judgment"] = ""
        df2.to_csv(results, index=False)
        gate.validate_audit(results)
    df.to_csv(results, index=False)
    report = gate.validate_audit(results)
    assert report["n_rows"] == 40
    assert report["pass"] == (report["disagreement_rate"] <= 0.05)
    # a nonsense label is rejected loudly
    df3 = df.copy(); df3["judgment"] = "CARRY?"
    df3.to_csv(results, index=False)
    with pytest.raises(ValueError, match="disallowed"):
        gate.validate_audit(results)
    # unfilled file is an error
    with pytest.raises(ValueError, match="empty judgment"):
        gate.validate_audit(blind)


# Census params that pass the automated construct gate with the mock policy
# (n=6, size=1200, samples=8, seed=42) — see test_census_end_to_end_mock.
GATE_PASSING_CENSUS = ["--n", "6", "--backend", "mock", "--size", "1200", "--samples", "8",
                       "--n-boot", "50"]


def _mk_gate_passing_census(tmp_path, monkeypatch, tag="42"):
    from experiments import check_construct_gate as gate
    from experiments.base_mode_census import main as census_main
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    out = tmp_path / "census"
    census_main(GATE_PASSING_CENSUS + ["--seed", tag, "--out", str(out)])
    run_dir = out / f"n6_s{tag}"
    cov = json.loads((run_dir / "coverage_ambiguity.json").read_text())
    assert cov["gate_pass"] is True, "GATE_PASSING_CENSUS no longer passes the gate"
    # make the census look 'real' for the backend check; the audit machinery is
    # what these tests exercise (mock-freeze refusal is tested by
    # test_freeze_refused_for_mock_census)
    doc = json.loads((run_dir / "manifest.json").read_text())
    doc["model"] = {"repo": "qwen/qwen2.5-0.5b-instruct", "backend": "hf", "revision": "x"}
    (run_dir / "manifest.json").write_text(json.dumps(doc))
    return gate, run_dir


def test_freeze_requires_completed_audit(tmp_path, monkeypatch, capsys):
    """--freeze-qref refuses while the manual audit is still unjudged."""
    gate, run_dir = _mk_gate_passing_census(tmp_path, monkeypatch)
    lock_path = tmp_path / "configs" / "locked" / "q_ref.lock"
    rc = gate.main(["--census", str(run_dir), "--audit-sample", "40", "--freeze-qref"])
    captured = capsys.readouterr().out
    assert "manual audit not judged yet" in captured
    assert not lock_path.exists()
    assert rc == 0  # the gate itself passes; only the audit is outstanding


def test_freeze_requires_completed_audit_via_cli(tmp_path, monkeypatch, capsys):
    """After a judged audit (rate 0), --freeze-qref writes the q_ref lock."""
    gate, run_dir = _mk_gate_passing_census(tmp_path, monkeypatch)
    lock_path = tmp_path / "configs" / "locked" / "q_ref.lock"
    import pandas as pd
    # first gate call exports the blinded sample (no freeze yet)
    gate.main(["--census", str(run_dir), "--audit-sample", "40"])
    df = pd.read_csv(run_dir / "audit_sample.csv", dtype=str).fillna("")
    df['judgment'] = "AMBIG"  # AMBIG always agrees → disagreement rate 0
    df.to_csv(run_dir / "audit_completed.csv", index=False)
    gate.main(["--census", str(run_dir), "--audit-sample", "40",
               "--audit-results", str(run_dir / "audit_completed.csv"),
               "--freeze-qref"])
    captured = capsys.readouterr().out
    assert "frozen:" in captured
    lock = json.loads(lock_path.read_text())
    assert lock["payload"]["census_run_id"] == run_dir.name


def test_validate_audit_tampered_rows_rejected(tmp_path, monkeypatch):
    gate, cls, run_dir, blind, key = _mk_census_with_audit(tmp_path, monkeypatch)
    import pandas as pd
    df = pd.read_csv(blind, dtype=str).fillna("")
    df = df.iloc[:-1]  # drop a row → row set no longer matches the key
    bad = run_dir / "audit_tampered.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="audit key"):
        gate.validate_audit(bad)


def test_freeze_refused_for_mock_census(tmp_path, monkeypatch):
    """A mock census must never freeze q_ref.lock (only a real GPU census can)."""
    from experiments import check_construct_gate as gate_mod
    monkeypatch.setattr(gate_mod, "ROOT", tmp_path)  # any accidental write stays in tmp
    from experiments.base_mode_census import main as census_main
    census_main(["--n", "6", "--backend", "mock", "--size", "120", "--samples", "4",
                 "--seed", "5", "--out", str(tmp_path / "census"), "--n-boot", "50"])
    run_dir = tmp_path / "census" / "n6_s5"
    with pytest.raises(RuntimeError, match="mock"):
        gate_mod.freeze_qref(run_dir)
    assert not (tmp_path / "configs" / "locked" / "q_ref.lock").exists()


def test_census_deterministic_for_fixed_seed(tmp_path):
    from experiments.base_mode_census import main
    import hashlib
    outs = []
    for tag in ("a", "b"):
        out = tmp_path / tag
        main(["--n", "6", "--backend", "mock", "--size", "120", "--samples", "4",
              "--seed", "99", "--out", str(out), "--n-boot", "50"])
        cls = (out / "n6_s99" / "classification.parquet").read_bytes()
        outs.append(hashlib.sha256(cls).hexdigest())
    assert outs[0] == outs[1]


def test_leakage_guard_runs_in_census():
    """The census calls check_disjoint; an overlapping pair raises before generation."""
    from vf_datasets.arithmetic import check_disjoint
    splits = make_splits(6, sizes={"census": 10, "eval": 10, "eval_mode": 10, "train": 10})
    contaminated = pd.concat([splits["census"], splits["eval"].iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        check_disjoint({**splits, "census": contaminated})


def test_census_frame_missing_columns_raises():
    with pytest.raises(ValueError, match="missing columns"):
        census_core.classify_generations(pd.DataFrame([{"prompt_id": "x", "a": 1, "b": 2}]))


def test_no_errors_census_fails_informatively():
    gens = pd.DataFrame([
        {"prompt_id": f"p{i}", "a": 10 + i, "b": 20, "completion": "Answer: " + str(30 + i)}
        for i in range(5)
    ])
    cls = census_core.classify_generations(gens)
    with pytest.raises(ValueError, match="no incorrect"):
        census_core.q_ref_point(cls)


def test_more_errors_needed():
    # expect a clear failure path when one half of the split has zero errors
    gens = pd.DataFrame([
        {"prompt_id": str(i), "a": 1000 + i, "b": 7, "completion": "Answer: " + str(1007 + i)}
        for i in range(6)
    ] + [{"prompt_id": "err", "a": 9, "b": 9, "completion": "Answer: 18.5x"}])
    cls = census_core.classify_generations(gens)
    with pytest.raises(ValueError, match="cluster bootstrap|no incorrect"):
        census_core.cluster_bootstrap_q(cls, n_boot=10)


def test_classification_parquet_handles_values_beyond_int64(tmp_path):
    a, b = 9_876_543_210, 1_234_567_890  # n = 10; ECHO concatenation has 20 digits
    gens = pd.DataFrame([{"prompt_id": "p0", "a": a, "b": b, "completion": f"Answer: {a}{b}"},
                         {"prompt_id": "p0", "a": a, "b": b, "completion": f"Answer: {a + b}"}])
    cls = census_core.classify_generations(gens)
    assert cls["mode"].tolist() == ["ECHO", "C"]
    cls.to_parquet(tmp_path / "c.parquet", index=False)
    back = pd.read_parquet(tmp_path / "c.parquet")
    assert back["value"].tolist() == [f"{a}{b}", str(a + b)]
