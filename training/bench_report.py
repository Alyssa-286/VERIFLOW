#!/usr/bin/env python
"""Summarise slurm/grpo_bench.sbatch and project run and grid costs (master plan §17, Gate 4)."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GRID_RUNS = {"pilot_lr": 4, "pilot_het": 12, "minimum": 24, "standard": 64, "replication_1p5b": 8}
GPU_HOURS_BUDGET = 150  # Gate 4 threshold (COMPLETION_PLAN.md Stage 2)


def summarize(out: Path) -> dict:
    rep: dict = {"configs": {}}
    for d in sorted(out.glob("G*")):
        m = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l]
        st = [x["step_seconds"] for x in m if "step_seconds" in x]
        steady = st[2:] if len(st) > 3 else st
        man = json.loads((d / "manifest.json").read_text())
        ev = []
        if (d / "eval.jsonl").exists():
            ev = [json.loads(l)["eval_seconds"] for l in (d / "eval.jsonl").read_text().splitlines() if l]
        rep["configs"][d.name] = {"step_seconds_median": round(statistics.median(steady), 2),
                                  "step_seconds_all": st, "eval_seconds": ev,
                                  "peak_vram_mib": man.get("peak_vram_mib"), "status": man["status"],
                                  "load_and_setup_seconds": round(man["time"]["wall_seconds"] - sum(st), 1)}
    g8 = rep["configs"].get("G8")
    if g8:
        step = g8["step_seconds_median"]
        ev = statistics.median(g8["eval_seconds"]) if g8["eval_seconds"] else 0.0
        proj = {}
        for T in (500, 800):
            sec = T * step + (T // 50 + 1) * ev + g8["load_and_setup_seconds"]
            proj[f"T{T}_gpu_hours"] = round(sec / 3600, 2)
        rep["run_projection_G8"] = proj
        h500 = proj["T500_gpu_hours"]
        h800 = proj["T800_gpu_hours"]
        rep["grid_projection_gpu_hours"] = {
            "pilot (4 LR runs x T800 + 12 het x T500)": round(4 * h800 + 12 * h500, 1),
            "minimum grid (24 x T500)": round(24 * h500, 1),
            "standard grid (64 x T500)": round(64 * h500, 1),
        }
        rep["gate4"] = {"budget_gpu_hours": GPU_HOURS_BUDGET,
                        "standard_fits": 64 * h500 <= GPU_HOURS_BUDGET,
                        "minimum_fits": 24 * h500 <= GPU_HOURS_BUDGET}
    return rep


if __name__ == "__main__":
    r = summarize(Path(sys.argv[1]))
    Path(sys.argv[1], "bench_report.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
