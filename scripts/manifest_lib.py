"""Shared manifest helpers (master plan §21).

Manifests are schema-tested in tests/test_manifest.py. Every run records:
git commit + dirty flag, config/dataset/lock hashes, env (python, packages,
cuda, gpu, driver, hostname), seed, start/end times, status, exit code.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RUN_ID = Path(__file__).resolve().parents[1] / "configs" / "locked"

TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "pyarrow", "matplotlib",
    "torch", "transformers", "accelerate", "trl", "peft", "vllm", "datasets",
)


def collect_git(root: Path | None = None) -> dict:
    root = root or _ROOT
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty_out = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                   text=True, check=True).stdout
        # ignore the .freebuff workspace-mirror and untracked runtime dirs
        tracked_dirty = [l for l in dirty_out.splitlines() if l.strip() and not l.startswith("??")]
        return {"commit": commit, "dirty": bool(tracked_dirty), "branch":
                subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"git metadata unavailable in {root}: {e}") from e


def collect_env() -> dict:
    import platform
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else platform.node(),
        "packages": packages,
        "cuda_runtime": None,
        "driver": None,
        "gpu": None,
    }
    try:
        import torch  # noqa
        env["cuda_runtime"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = props.name
            env["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, drv = out.stdout.strip().splitlines()[0].split(", ")[:2]
            env["gpu"], env["driver"] = env["gpu"] or name, drv
    except Exception:
        pass
    return env


def _lock_payload_sha(path: Path) -> str | None:
    try:
        doc = json.loads(path.read_text())
        return doc.get("sha256")
    except FileNotFoundError:
        return None


def lock_hashes(root: Path | None = None) -> dict:
    """SHA-256 of every configs/locked payload (None if not yet frozen)."""
    root = root or _ROOT
    d = root / "configs" / "locked"
    return {
        "prompt_lock_sha256": _lock_payload_sha(d / "prompt.lock"),
        "taxonomy_lock_sha256": _lock_payload_sha(d / "taxonomy.lock"),
        "q_ref_lock_sha256": _lock_payload_sha(d / "q_ref.lock"),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_run_id(spec: dict) -> str:
    """Deterministic run id from a run spec dict."""
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(payload.encode()).hexdigest()[:10]
    base = "_".join(str(spec.get(k, "")) for k in ("experiment", "config_id", "seed"))
    return f"{base}_{h}"


def build_run_manifest(spec: dict, root: Path | None = None) -> dict:
    """Full manifest skeleton for a training/eval run (schema in §21)."""
    root = root or _ROOT
    git = collect_git(root)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "experiment_id": spec.get("experiment", "veriflow-run-v1"),
        "run_id": new_run_id(spec),
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "model": spec.get("model", {}),
        "prompt_template_sha256": spec.get("prompt_template_sha256"),
        "dataset_sha256": spec.get("dataset_sha256", {}),
        **lock_hashes(root),
        "q_ref_sha256": spec.get("q_ref_sha256"),
        "verifier_config_sha256": spec.get("verifier_config_sha256"),
        "verifier_config_id": spec.get("verifier_config_id"),
        "taxonomy_version": spec.get("taxonomy_version"),
        "seed": spec.get("seed"), "verifier_seed": spec.get("verifier_seed"),
        "training": spec.get("training", {}),
        "env": collect_env(),
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"),
                  "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                  "job_name": os.environ.get("SLURM_JOB_NAME")},
        "time": {"start": t0, "end": None, "status": "started"},
        "checkpoints": [], "eval": None, "exit_code": None,
    }


REQUIRED_MANIFEST_KEYS = (
    "experiment_id", "run_id", "git_commit", "git_dirty", "model",
    "prompt_template_sha256", "dataset_sha256", "q_ref_sha256",
    "verifier_config_sha256", "taxonomy_version", "seed", "verifier_seed",
    "env", "slurm", "time", "checkpoints", "eval", "exit_code",
)


def finalize_run_manifest(manifest: dict, run_dir: Path, *, exit_code: int,
                          status: str | None = None) -> dict:
    """Fill end fields in place. Never masks failures: status is derived from
    the exit code unless the trainer recorded an explicit status first."""
    manifest["time"]["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["exit_code"] = int(exit_code)
    if status is not None:
        manifest["time"]["status"] = status
    else:
        manifest["time"]["status"] = "completed" if exit_code == 0 else "failed"
    ckpt = run_dir / "checkpoint_steps.json"
    manifest["checkpoints"] = json.loads(ckpt.read_text()) if ckpt.exists() else []
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
