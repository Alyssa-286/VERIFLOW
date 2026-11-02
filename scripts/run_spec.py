#!/usr/bin/env python
"""Reads a run table (configs/*.csv) and prints row `--index` as JSON.

Run tables (pilot_runs.csv, grid_runs.csv, later; census_runs.csv for Phase 4)
are the single source of which run a Slurm array task executes. Columns at
minimum: experiment, config_id, seed; extra columns pass through to the spec.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_spec(table: Path, index: int) -> dict:
    if not table.exists():
        raise FileNotFoundError(f"run table not found: {table}")
    df = pd.read_csv(table, comment="#")
    if index < 0 or index >= len(df):
        raise IndexError(f"index {index} out of range: table has {len(df)} rows")
    spec = {k: (None if pd.isna(v) else v) for k, v in df.iloc[index].items()}
    for k, v in list(spec.items()):
        # numeric seeds/config ids stay ints when possible
        if isinstance(v, str) and v.lstrip("-").isdigit() and k not in ("prompt", "run"):
            spec[k] = int(v)
    return spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--index", type=int, required=True)
    args = ap.parse_args(argv)
    spec = read_spec(Path(args.table) if Path(args.table).is_absolute() else ROOT / args.table, args.index)
    print(json.dumps(spec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
