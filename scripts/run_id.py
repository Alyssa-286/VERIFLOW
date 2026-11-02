#!/usr/bin/env python
"""Print a deterministic run id for a run spec (spec JSON on stdin or --json)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manifest_lib import new_run_id


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="spec JSON file; stdin if omitted")
    args = ap.parse_args(argv)
    raw = Path(args.json).read_text() if args.json else sys.stdin.read()
    spec = json.loads(raw)
    print(new_run_id(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
