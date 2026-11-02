"""Manifest/run-id/run-table tests (master plan §21)."""
from __future__ import annotations

import json

import pytest

from scripts.manifest_lib import (REQUIRED_MANIFEST_KEYS, build_run_manifest,
                                  collect_git, finalize_run_manifest, file_sha256, new_run_id)
from scripts.run_spec import read_spec


def test_build_run_manifest_has_required_keys(tmp_path):
    spec = {"experiment": "veriflow-grid-v1", "config_id": "cfg001", "seed": 1}
    m = build_run_manifest(spec)
    for k in REQUIRED_MANIFEST_KEYS:
        assert k in m, k
    assert m["run_id"].startswith("veriflow-grid-v1_cfg001_1_")
    assert m["git_commit"]
    assert isinstance(m["git_dirty"], bool)
    assert m["model"] == {}
    assert m["env"]["python"]


def test_finalize_sets_status_from_exit_code(tmp_path):
    m = build_run_manifest({"experiment": "e", "config_id": "c", "seed": 0})
    m["checkpoints"] = [{"step": 10, "sha256": "x", "path": "runs/x/ckpt-10"}]
    (tmp_path / "checkpoint_steps.json").write_text(json.dumps(m["checkpoints"]))
    fin = finalize_run_manifest(m, tmp_path, exit_code=3)
    assert fin["exit_code"] == 3
    assert fin["time"]["status"] == "failed"
    assert fin["time"]["end"]
    # explicit status (a recorded divergence) wins over the exit-code default
    fin2 = finalize_run_manifest(m, tmp_path, exit_code=0, status="diverged")
    assert fin2["time"]["status"] == "diverged"


def test_run_id_deterministic_and_spec_sensitive():
    s1 = {"experiment": "e", "config_id": "cfg001", "seed": 1}
    s2 = {"experiment": "e", "config_id": "cfg002", "seed": 1}
    s1b = {"seed": 1, "config_id": "cfg001", "experiment": "e"}  # different key order
    assert new_run_id(s1) == new_run_id(s1b)
    assert new_run_id(s1) != new_run_id(s2)


def test_read_spec_coerces_ints_and_bounds(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("experiment,config_id,seed,note\nveriflow-grid,cfg001,1,hello\nveriflow-grid,cfg002,2,\n")
    s0 = read_spec(csv, 0)
    assert s0["config_id"] == "cfg001" and s0["seed"] == 1
    assert s0["note"] == "hello"
    s1 = read_spec(csv, 1)
    assert s1["config_id"] == "cfg002" and pd_isna(s1["note"])
    with pytest.raises(IndexError):
        read_spec(csv, 2)
    with pytest.raises(FileNotFoundError):
        read_spec(tmp_path / "missing.csv", 0)


def pd_isna(v):
    return v is None


def test_lock_tamper_detection(tmp_path, monkeypatch):
    """A hand-edited lock must fail the payload-hash verify."""
    import hashlib
    lock_dir = tmp_path / "configs" / "locked"
    lock_dir.mkdir(parents=True)
    lock = lock_dir / "taxonomy.lock"
    payload = {"version": "v1"}
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    doc = {"payload": payload, "sha256": sha}
    lock.write_text(json.dumps(doc))
    from scripts import preflight
    preflight.check_locks(tmp_path)  # ok
    doc["payload"]["version"] = "v2"  # tamper without rehash
    lock.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="mismatch"):
        preflight.check_locks(tmp_path)


def test_collect_git_matches_repo():
    git = collect_git()
    assert len(git["commit"]) == 40
    # any checked-out branch name is a legitimate provenance value
    assert git["branch"] and "HEAD" not in git["branch"]


def test_file_sha256_stable(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"veriflow")
    assert file_sha256(p) == file_sha256(p)
