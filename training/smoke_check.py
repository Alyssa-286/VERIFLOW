#!/usr/bin/env python
"""Assertions for the Gate 4 smoke job (slurm/grpo_smoke.sbatch). Prints a JSON report;
exits nonzero on any failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiers.noisy_verifier import prf_uniform


def _vlog(run: Path) -> pd.DataFrame:
    parts = sorted((run / "verifier_log").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def check(out: Path) -> dict:
    rep: dict = {"checks": {}}
    c = rep["checks"]
    for name, steps in (("one_step", 1), ("ten_step", 10), ("resumed", 10)):
        run = out / name
        man = json.loads((run / "manifest.json").read_text())
        c[f"{name}: completed"] = man["status"] == "completed" and man["final_step"] == steps
        log = _vlog(run)
        c[f"{name}: verifier log has every step"] = (not log.empty and
                                                     set(log["step"]) >= set(range(0 if name != "resumed" else 5, steps)))
        G, pps = man["G"], man["prompts_per_step"]
        per_step = log.groupby("step").size()
        c[f"{name}: G*prompts per step"] = bool((per_step == G * pps).all())
        gen = json.loads((run / "generation_config.json").read_text())["trainer"]
        c[f"{name}: neutral sampling"] = (gen["temperature"] == 1.0 and gen["top_p"] == 1.0
                                          and gen["top_k"] in (0, None) and gen["repetition_penalty"] == 1.0)
        rep[name] = {"wall_seconds": man["time"].get("wall_seconds"), "peak_vram_mib": man.get("peak_vram_mib"),
                     "resumed_from": man.get("resumed_from"), "log_rows": len(log)}
        if name == "ten_step":
            vs, rs = man["base"]["verifier"]["verifier_seed"], man["args"]["seed"]
            u = [prf_uniform("veriflow-v1", vs, rs, s, p, j)
                 for s, p, j in zip(log["step"], log["prompt_id"], log["sample_idx"])]
            c["ten_step: verifier replay is exact"] = bool((pd.Series(u) == log["u"].reset_index(drop=True)).all())
            c["ten_step: reward = 1[u < rho]"] = bool(((log["u"] < log["rho"]).astype(float) == log["reward"]).all())
            c["ten_step: I never rewarded"] = bool((log.loc[log["mode"] == "I", "reward"] == 0).all())
            rep["ten_step"]["true_accuracy_by_step"] = log.groupby("step")["correct"].mean().round(4).to_dict()
            rep["ten_step"]["reward_by_step"] = log.groupby("step")["reward"].mean().round(4).to_dict()
            ev = [json.loads(l) for l in (run / "eval.jsonl").read_text().splitlines() if l]
            rep["ten_step"]["eval"] = [{k: e[k] for k in ("step", "greedy_accuracy", "sampled_accuracy", "eval_seconds")}
                                       for e in ev]
            c["ten_step: eval at 0, 5, 10"] = [e["step"] for e in ev] == [0, 5, 10]
            m = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l]
            st = [x["step_seconds"] for x in m if "step_seconds" in x]
            rep["ten_step"]["step_seconds"] = st
    res = json.loads((out / "resumed" / "manifest.json").read_text())
    c["resumed: started from checkpoint-5"] = str(res.get("resumed_from", "")).endswith("checkpoint-5")
    rep["ok"] = all(c.values())
    return rep


if __name__ == "__main__":
    r = check(Path(sys.argv[1]))
    print(json.dumps(r, indent=2, default=str))
    raise SystemExit(0 if r["ok"] else 1)
