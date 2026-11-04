#!/usr/bin/env python
"""Verify completed census runs and build the Gate 3 comparison (§9.1, §9.3).

For each run directory this re-derives q_ref, coverage, ambiguity and the
split-half TV from classification.parquet and checks them against the stored
artifacts, checks the manifest (real backend, pinned revision, hashes, Slurm id)
and that every §11 artifact is present and nonempty. It then applies the §9.1
selection rule: n = the smallest n whose base accuracy is in [0.25, 0.60] and
whose automated §9.3 gate passes (coverage >= 0.70 is part of that gate).

Usage:
  python -m experiments.compare_census runs/census/n5_s20261102_j3594 ... \
      --json runs/census/gate3_comparison.json --md runs/census/gate3_comparison.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from error_modes.classify_errors import classes_for
from experiments import census_core
from models.loading import load_spec
from vf_datasets.arithmetic import prompt_template_sha256

ARTIFACTS = ("generations.parquet", "classification.parquet", "q_ref.json", "per_mode_lengths.json",
             "difficulty_hist.csv", "per_prompt_summary.parquet", "stability.json",
             "coverage_ambiguity.json", "params.json", "manifest.json")


def verify_run(run_dir: Path, model: str = "qwen0p5b") -> dict:
    problems: list[str] = []
    for f in ARTIFACTS:
        p = run_dir / f
        if not p.is_file() or p.stat().st_size == 0:
            problems.append(f"missing or empty {f}")
    if problems:
        return {"run_id": run_dir.name, "verified": False, "problems": problems}

    man = json.loads((run_dir / "manifest.json").read_text())
    params = json.loads((run_dir / "params.json").read_text())
    q_doc = json.loads((run_dir / "q_ref.json").read_text())
    cov = json.loads((run_dir / "coverage_ambiguity.json").read_text())
    tv_stored = json.loads((run_dir / "stability.json").read_text())["split_half_tv"]
    cls = pd.read_parquet(run_dir / "classification.parquet")
    gens = pd.read_parquet(run_dir / "generations.parquet")

    spec = load_spec((man.get("model") or {}).get("name") or model)  # the run's own pinned model
    if (man.get("model") or {}).get("revision") != spec.revision:
        problems.append("manifest model revision != pinned revision")
    if (man.get("model") or {}).get("backend") == "mock":
        problems.append("mock backend")
    task = q_doc.get("task", "add")
    if man.get("prompt_template_sha256") != prompt_template_sha256(task):
        problems.append("prompt template hash mismatch")
    if man.get("q_ref_sha256") != hashlib.sha256((run_dir / "q_ref.json").read_bytes()).hexdigest():
        problems.append("q_ref.json hash != manifest")
    expected = params["split_sizes"]["census"] * params["samples"]
    if len(gens) != expected or len(cls) != expected:
        problems.append(f"generation count {len(gens)}/{len(cls)} != {expected}")
    if gens["completion"].isna().any():
        problems.append("null completions")

    q = q_doc["q_ref"]
    if not math.isclose(sum(q.values()), 1.0, abs_tol=1e-9):
        problems.append(f"q_ref sums to {sum(q.values())}")
    q_re = census_core.q_ref_point(cls)
    if any(not math.isclose(q[c], q_re[c], abs_tol=1e-12) for c in q):
        problems.append("q_ref does not match classification.parquet")
    for c, (lo, hi) in q_doc["q_ref_ci95"].items():
        if not (0.0 <= lo <= hi <= 1.0) or not (lo - 1e-9 <= q[c] <= hi + 1e-9):
            problems.append(f"CI for {c} [{lo}, {hi}] does not bracket {q[c]}")
    cov_re = census_core.coverage_and_ambiguity(cls)
    if not (math.isclose(cov_re["coverage"], cov["coverage"]) and
            math.isclose(cov_re["ambiguity_rate"], cov["ambiguity_rate"])):
        problems.append("coverage/ambiguity do not match classification.parquet")
    if not math.isclose(census_core.split_half_tv(cls), tv_stored):
        problems.append("split-half TV does not match classification.parquet")

    checks = {c["id"]: c for c in cov["gate"]}
    n_heavy = sum(1 for m in classes_for(q_doc.get("scheme", task))[1] if q[m] >= census_core.GATE_MASS_MIN)
    bad = cls[~cls["correct"]]
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "verified": not problems,
        "problems": problems,
        "n_digits": q_doc["n_digits"],
        "task": task,
        "scheme": q_doc.get("scheme", task),
        "u_diagnostics": (bad.loc[bad["mode"] == "U", "diag"].value_counts(normalize=True).to_dict()
                          if "diag" in bad.columns else {}),
        # taxonomy v3 keeps the v2 mechanism labels as a descriptive annotation (Amendment 1)
        "mechanism_annotation": (bad.loc[bad["mode"] != "I", "diag"].str.split("|").str[0]
                                 .value_counts(normalize=True).round(4).to_dict()
                                 if q_doc.get("scheme") == "add_v3" else {}),
        "slurm_job_id": (man.get("slurm") or {}).get("job_id"),
        "wall_seconds": (man.get("time") or {}).get("wall_seconds"),
        "hardware": man.get("hardware"),
        "model_revision": (man.get("model") or {}).get("revision"),
        "seed": man.get("seed"),
        "git_commit": man.get("git_commit"),
        "n_generations": len(cls),
        "base_accuracy": q_doc["base_accuracy"],
        "q_ref": q,
        "q_ref_ci95": q_doc["q_ref_ci95"],
        "coverage": cov["coverage"],
        "modes_ge_5pct": n_heavy,
        "ambiguity": cov["ambiguity_rate"],
        "split_half_tv": tv_stored,
        "parse_status_of_errors": bad["parse_status"].value_counts().to_dict(),
        "checks": {k: v["pass"] for k, v in checks.items() if k != "9.3.2 merge note"},
        "gate_pass_automated": bool(cov["gate_pass"]),
    }


def select_n(rows: list[dict]) -> dict:
    """§9.1: smallest n with base accuracy in the window and the automated §9.3 gate passing."""
    lo, hi = census_core.GATE_ACC_RANGE
    ok = sorted((r for r in rows if r["verified"] and lo <= r["base_accuracy"] <= hi
                 and r["gate_pass_automated"]), key=lambda r: r["n_digits"])
    if ok:
        return {"selected_n": ok[0]["n_digits"], "run_id": ok[0]["run_id"]}
    return {"selected_n": None, "run_id": None,
            "reason": "no n satisfies §9.1 accuracy window + §9.3 automated gate"}


def to_markdown(rows: list[dict]) -> str:
    first = rows[0]["q_ref"] if rows else {}
    tail = [m for m in ("U", "I") if m in first]  # taxonomy v3 has no U
    modes = [m for m in first if m not in ("U", "I")] + tail
    head = ("| n | Run ID (Slurm job) | Base acc | " + " | ".join(f"q {m} [95% CI]" for m in modes)
            + " | Coverage | Modes ≥5% | Ambiguity | Split-half TV | Gate |")
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for r in sorted(rows, key=lambda r: r["n_digits"]):
        qs = " | ".join(f"{r['q_ref'][m]:.3f} [{r['q_ref_ci95'][m][0]:.3f}, {r['q_ref_ci95'][m][1]:.3f}]"
                        for m in modes)
        fails = [k for k, v in r["checks"].items() if not v]
        verdict = "PASS" if r["gate_pass_automated"] else "FAIL (" + ", ".join(fails) + ")"
        lines.append(f"| {r['n_digits']} | `{r['run_id']}` ({r['slurm_job_id']}) | {r['base_accuracy']:.3f} | {qs} | "
                     f"{r['coverage']:.3f} | {r['modes_ge_5pct']} | {r['ambiguity']:.3f} | "
                     f"{r['split_half_tv']:.4f} | {verdict} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    rows = [verify_run(Path(r)) for r in args.runs]
    report = {"runs": rows, "selection": select_n(rows),
              "rule": "§9.1: smallest n with base accuracy in [0.25, 0.60] and §9.3 automated gate passing"}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    md = to_markdown([r for r in rows if "q_ref" in r])
    if args.md:
        Path(args.md).write_text(md)
    print(md)
    print(json.dumps({"selection": report["selection"],
                      "verified": {r["run_id"]: (r["verified"], r["problems"]) for r in rows}}, indent=2))
    return 0 if all(r["verified"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
