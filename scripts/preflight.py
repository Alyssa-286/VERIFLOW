#!/usr/bin/env python
"""veriflow-preflight (master plan §16.3 job 1). CPU-only; login node or any box.

Checks, in order, with nonzero exit on the first hard failure:
  1. repo layout (pyproject, veriflow packages, slurm/, configs/locked);
  2. required import versions (numpy scipy pandas pyarrow matplotlib);
  3. isolation: cwd is a VERIFLOW tree; no ErrorEcho references in code paths;
  4. git state: a commit exists; --require-clean rejects tracked modifications;
  5. locks: prompt.lock + taxonomy.lock payloads hash-verify;
  6. optionally the CPU unit tests (--tests, adds a few minutes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_IMPORTS = ("numpy", "scipy", "pandas", "pyarrow", "matplotlib")
REQUIRED_PATHS = (
    "pyproject.toml", "uv.lock", "verifiers/stats.py", "verifiers/matched_verifiers.py",
    "verifiers/noisy_verifier.py", "error_modes/classify_errors.py",
    "vf_datasets/arithmetic.py", "experiments/base_mode_census.py",
    "slurm/_common.sh", "configs/locked/prompt.lock", "configs/locked/taxonomy.lock",
)


def check_layout() -> list[str]:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    return missing


def check_imports() -> dict:
    from importlib.metadata import version
    out = {}
    for name in REQUIRED_IMPORTS:
        out[name] = version(name)
    return out


import re as _re

# Offending = a real path reference into the ErrorEcho project's directories.
# The bare project name in a doc comment is not an offence (docs discuss it
# legitimately); a hard-coded directory/deps reference is.
_ISOLATION_PATTERNS = (_re.compile(r"ErrorEcho[-_/]"), _re.compile(r"~/ErrorEcho"),
                       _re.compile(r"phase11-20"))
_ISOLATION_DIRS = ("verifiers", "error_modes", "vf_datasets", "vf_stats", "simulations",
                   "experiments", "scripts", "slurm", "bin", "configs", "figures", "models")
_ISOLATION_EXTS = (".py", ".sh", ".sbatch", ".yaml", ".json", ".toml", ".csv", ".md")


def check_isolation(root: Path | None = None) -> list[str]:
    """No ErrorEcho path references in code, slurm scripts, wrappers or configs."""
    root = (root or ROOT)
    offenders: list[str] = []
    for sub in (*_ISOLATION_DIRS, ""):
        base = root / sub if sub else root
        if sub == "":
            candidates = list(root.glob("*.py")) + list(root.glob("*.sh")) + list(root.glob("*.toml"))
        else:
            candidates = base.rglob("*") if base.is_dir() else []
        for p in sorted(candidates):
            if not p.is_file() or p.suffix not in _ISOLATION_EXTS:
                continue
            if p == Path(__file__):
                continue  # this module embeds the detection patterns themselves
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue  # skip hidden/venv/caches
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            if any(pat.search(text) for pat in _ISOLATION_PATTERNS):
                offenders.append(str(p.relative_to(root)))
    return offenders


def check_git(require_clean: bool) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.splitlines()
    tracked_dirty = [l for l in status if l.strip() and not l.startswith("??")]
    if require_clean and tracked_dirty:
        raise RuntimeError("git tree has tracked modifications — confirmatory rule §21 forbids dirty runs")
    return {"commit": commit, "dirty": bool(tracked_dirty)}


def check_locks(root: Path | None = None) -> dict:
    d = (root or ROOT) / "configs" / "locked"
    out = {}
    for name in ("prompt.lock", "taxonomy.lock", "prompt_sub.lock", "taxonomy_sub.lock", "q_ref.lock"):
        path = d / name
        if not path.exists():
            out[name] = None
            continue
        doc = json.loads(path.read_text())
        sha = hashlib.sha256(json.dumps(doc["payload"], sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
        if sha != doc["sha256"]:
            raise RuntimeError(f"{name}: payload hash mismatch (file corrupted or hand-edited)")
        out[name] = sha
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests", action="store_true", help="also run the CPU test suite")
    ap.add_argument("--require-clean", action="store_true")
    args = ap.parse_args(argv)

    report: dict = {"stage": "preflight"}
    missing = check_layout()
    if missing:
        report["layout"] = {"ok": False, "missing": missing}
        print(json.dumps(report, indent=2))
        return 2
    report["layout"] = {"ok": True}

    try:
        report["imports"] = check_imports()
        report["git"] = check_git(args.require_clean)
        report["locks"] = check_locks()
    except Exception as e:
        report["error"] = repr(e)
        print(json.dumps(report, indent=2))
        return 1

    offenders = check_isolation()
    report["isolation"] = {"ok": not offenders, "offenders": offenders}
    ok = not offenders
    if args.tests:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
        report["tests_exit"] = proc.returncode
        ok = ok and proc.returncode == 0

    report["ok"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
