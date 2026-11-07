#!/usr/bin/env python
"""Validate completed GRPO runs against their run table and the declared protocol (not just exit codes).

Per run in the table, checks:
  manifest.json   status completed; final_step == max_steps; args.lr/seed/max_steps/eval_every match;
                  verifier_config_sha256 == sha256 of the config file's canonical JSON; q_ref lock hash ==
                  the frozen lock; one git commit across all runs (reported)
  eval.jsonl      steps exactly {0, e, 2e, ..., max_steps}; accuracies in [0, 1]; n_eval == 1000
  metrics.jsonl   one step_seconds row per step 1..max_steps
  verifier_log    rows for every step 1..max_steps (G x prompts_per_step per step), config sha matches,
                  reward in {0, 1}
Writes <out> (JSON) and exits non-zero if any run fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA = "2c34f6d23b7c81cfe0ef6ef7eb3b5207b1b42adb05779760a812ad7cd8bdd5cb"


def check_run(run_dir: Path, row, eval_every: int) -> dict:
    errs: list[str] = []
    m = json.loads((run_dir / "manifest.json").read_text())
    a = m.get("args", {})
    T = int(row.max_steps)
    if m.get("status") != "completed":
        errs.append(f"status={m.get('status')}")
    if int(m.get("final_step", -1)) != T:
        errs.append(f"final_step={m.get('final_step')}")
    for k, v in (("lr", float(row.lr)), ("seed", int(row.seed)), ("max_steps", T), ("eval_every", eval_every)):
        if a.get(k) != v:
            errs.append(f"args.{k}={a.get(k)} != {v}")
    from verifiers.matched_verifiers import load_config
    cfg_sha = m.get("verifier_config_sha256")
    want = load_config(ROOT / row.config).sha256
    if cfg_sha != want:
        errs.append(f"verifier_config_sha256 {cfg_sha} != config file {want}")
    if (m.get("git") or {}).get("dirty"):
        errs.append("run started from a dirty tree")
    if m.get("q_ref_lock_sha256") != LOCK_SHA:
        errs.append("q_ref lock hash mismatch")
    ev = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines() if l]
    steps = sorted({e["step"] for e in ev})
    if steps != list(range(0, T + 1, eval_every)):
        errs.append(f"eval steps {steps}")
    if any(not 0 <= e["greedy_accuracy"] <= 1 or e.get("n_eval") != 1000 for e in ev):
        errs.append("eval values out of range or n_eval != 1000")
    met = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l]
    st = sorted({r["step"] for r in met if "step_seconds" in r})
    if st != list(range(1, T + 1)):
        errs.append(f"metrics step_seconds steps {st[:3]}..{st[-3:]} (n={len(st)})")
    vl = pd.concat([pd.read_parquet(p) for p in sorted((run_dir / "verifier_log").glob("*.parquet"))])
    per_step = vl.groupby("step").size()
    expected = int(m.get("G", 8)) * int(m.get("prompts_per_step", 32))
    vsteps = sorted(per_step.index)
    if vsteps[0] > 1 or vsteps[-1] < T - 1 or len(vsteps) < T:
        errs.append(f"verifier_log steps {vsteps[0]}..{vsteps[-1]} n={len(vsteps)}")
    if (per_step != expected).any():
        errs.append(f"verifier_log rows/step != {expected}: {per_step[per_step != expected].to_dict()}")
    if set(vl["verifier_config_sha256"]) != {cfg_sha}:
        errs.append("verifier_log config sha differs from manifest")
    if not set(vl["reward"].unique()) <= {0.0, 1.0}:
        errs.append("non-binary reward")
    st_s = [r["step_seconds"] for r in met if "step_seconds" in r]
    return {"run": run_dir.name, "ok": not errs, "errors": errs, "commit": (m.get("git") or {}).get("commit"),
            "slurm": m.get("slurm"), "wall_seconds": (m.get("time") or {}).get("wall_seconds"),
            "peak_vram_mib": m.get("peak_vram_mib"), "median_step_s": float(pd.Series(st_s).median()),
            "verifier_rows": int(len(vl)), "config_sha_file": want, "config_sha_manifest": cfg_sha}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--eval-every", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    table = pd.read_csv(args.table)
    res = []
    for row in table.itertuples():
        rd = Path(args.runs_root) / row.run_name
        if not (rd / "manifest.json").exists():
            res.append({"run": row.run_name, "ok": False, "errors": ["missing run dir/manifest"]})
            continue
        try:
            res.append(check_run(rd, row, args.eval_every))
        except Exception as e:  # report, never hide
            res.append({"run": row.run_name, "ok": False, "errors": [f"{type(e).__name__}: {e}"]})
    commits = sorted({r.get("commit") for r in res if r.get("commit")})
    out = {"n_runs": len(res), "n_ok": sum(r["ok"] for r in res), "commits": commits, "runs": res}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("n_runs", "n_ok", "commits")}))
    for r in res:
        if not r["ok"]:
            print(r["run"], r["errors"])
    return 0 if out["n_ok"] == out["n_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
