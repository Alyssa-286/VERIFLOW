#!/usr/bin/env python
"""Write <run-dir>/manifest.json from a run spec (master plan §21).

Usage: python -m scripts.write_manifest --spec spec.json --out runs/<run_id>
--require-clean aborts when the git tree has tracked modifications (§21 rule:
git_dirty=true aborts confirmatory runs; the census tolerates it because the
census precedes the pre-registration).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manifest_lib import build_run_manifest, collect_git

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-clean", action="store_true")
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text())
    if args.require_clean and collect_git(ROOT)["dirty"]:
        raise RuntimeError("git tree dirty; refusing to start a confirmatory run (§21)")
    manifest = build_run_manifest(spec, ROOT)
    out = Path(args.out)
    if out.name != manifest["run_id"]:
        # allow caller-provided dirs, but keep the manifest authoritative
        pass
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_id": manifest["run_id"], "out": str(out / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
