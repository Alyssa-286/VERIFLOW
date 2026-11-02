#!/usr/bin/env python
"""Finalize <run-dir>/manifest.json after a run ends (§21).

Usage: python -m scripts.finalize_manifest --run-dir runs/<id> --exit-code $STATUS
                               [--slurm-job 3580] [--status diverged|completed|failed]
Reads the already-started manifest, sets end time/exit code/status, and copies
the result to manifests/ (the project-wide audit index).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manifest_lib import finalize_run_manifest, REQUIRED_MANIFEST_KEYS

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--exit-code", type=int, required=True)
    ap.add_argument("--slurm-job", default=None)
    ap.add_argument("--status", default=None, choices=(None, "completed", "failed", "diverged"))
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if args.slurm_job is not None:
        manifest["slurm"]["job_id"] = str(args.slurm_job)
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise RuntimeError(f"manifest is missing required keys (schema §21): {missing}")
    finalized = finalize_run_manifest(manifest, run_dir, exit_code=args.exit_code, status=args.status)

    index = ROOT / "manifests"
    index.mkdir(exist_ok=True)
    shutil.copy(path, index / f"{finalized['run_id']}.json")
    print(json.dumps({"run_id": finalized["run_id"], "status": finalized["time"]["status"],
                      "exit_code": finalized["exit_code"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
