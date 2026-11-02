#!/usr/bin/env python
"""Gate 3 checker: evaluate §9.3 against a completed census run.

Usage:
  python -m experiments.check_construct_gate --census runs/census/n7_s20261102 \
      [--json out] [--audit-sample 200] [--freeze-qref]

The synthetic-recovery criterion (§9.3.5) is `tests/test_classifier.py` (Phase 3,
D6: 100% recovery at n=5 and n=8). The manual audit (§9.3.6) is emitted here as
`audit_sample.csv` with the classifier label column hidden.

--freeze-qref writes configs/locked/q_ref.lock ONLY if all automated criteria
AND the §9.3.6 manual audit (≤ 5% disagreement) pass; it records the q_ref.json
SHA-256 and the census run id. A mock census can never freeze. Any later change
invalidates every configuration generated from the old lock.

Manual-audit files (§9.3.6):
  audit_sample.csv  blinded: prompt_id, a, b, completion, answer_true, judgment
                    — the classifier label column is NOT present; the reviewer
                    fills `judgment` and returns the file.
  audit_key.csv     separate file, kept out of the reviewer's hands: row_id,
                    primary (classifier label), matches (multi-hot).
Validation: a judgment agrees iff its label is one of the *rules the classifier
matched* (multi-hot), or is AMBIG. Disagreement rate must be ≤ 0.05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import census_core

ROOT = Path(__file__).resolve().parents[1]

# The reviewer may name any rule the classifier knows about, say it is genuinely
# ambiguous (AMBIG), or disagree outright with the parsed class by naming a
# designed/U/I mode. AMBIG agrees with any multi-rule match.
ALLOWED_JUDGMENTS = ("ECHO", "CARRY", "TRUNC", "TRANS", "DIGIT", "U", "I", "AMBIG")
AUDIT_DISAGREEMENT_MAX = 0.05  # §9.3.6


def load_census(run_dir: Path) -> tuple:
    for f in ("classification.parquet", "q_ref.json", "coverage_ambiguity.json",
              "stability.json", "per_prompt_summary.parquet"):
        if not (run_dir / f).exists():
            raise FileNotFoundError(f"{run_dir} is missing {f}; run the census first")
    cls = pd.read_parquet(run_dir / "classification.parquet")
    q_doc = json.loads((run_dir / "q_ref.json").read_text())
    cov = json.loads((run_dir / "coverage_ambiguity.json").read_text())
    return cls, q_doc, cov


def write_audit_sample(cls: pd.DataFrame, run_dir: Path, n: int = 200, seed: int = 7) -> tuple[Path, Path]:
    """Manual-audit sample (§9.3.6): n random incorrect completions with the
    classifier label hidden (blinded file) and the labels kept in a SEPARATE
    key file. The researcher fills `judgment` on a copy of the blinded file.
    disagreement > 5% of judged rows triggers redesign."""
    bad = cls[~cls["correct"]].reset_index(drop=True)
    if len(bad) == 0:
        raise ValueError("no incorrect generations to audit")
    sample = bad.sample(n=min(n, len(bad)), random_state=seed).copy()
    sample["row_id"] = [f"AUD-{i:04d}" for i in range(len(sample))]
    blind = sample[["row_id", "prompt_id", "a", "b", "completion"]].copy()
    blind["answer_true"] = blind["a"] + blind["b"]
    blind["judgment"] = ""
    blind_path = run_dir / "audit_sample.csv"
    blind.to_csv(blind_path, index=False)
    key = sample[["row_id", "prompt_id", "mode", "matches"]].rename(columns={"mode": "primary"})
    key_path = run_dir / "audit_key.csv"
    key.to_csv(key_path, index=False)
    return blind_path, key_path


def _audit_agrees(label: str, matches: str) -> bool:
    """A judgment agrees iff it names a rule the classifier itself matched, or
    the reviewer says the example is genuinely ambiguous (AMBIG)."""
    if label == "AMBIG":
        return True
    return label in matches.split("|")


def validate_audit(results_path: Path) -> dict:
    """Validate the COMPLETED blinded audit against the classifier's labels.

    The results file is the reviewer's copy of audit_sample.csv. Every row must
    carry a judgment from ALLOWED_JUDGMENTS (empty cells are an error, not a
    silent exclusion). Disagreement = judged label names no matched rule and is
    not AMBIG. Returns a report dict; never raises for a low rate."""
    df = pd.read_csv(results_path, dtype=str).fillna("")
    required = {"row_id", "prompt_id", "a", "b", "completion", "judgment"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"audit results file is missing columns: {sorted(missing_cols)}")
    run_dir = results_path.parent
    key = pd.read_csv(run_dir / "audit_key.csv", dtype=str).fillna("")
    if len(key) != len(df) or set(key["row_id"]) != set(df["row_id"]):
        raise ValueError(
            f"audit results do not match the audit key ({len(df)} rows vs {len(key)} key rows); "
            "re-export the audit before judging"
        )
    merged = df[["row_id", "judgment"]].merge(key, on="row_id", validate="1:1")
    empty = int((merged["judgment"].str.strip() == "").sum())
    if empty:
        raise ValueError(f"{empty} audit rows have an empty judgment; all rows must be judged")
    bad_labels = sorted(set(merged["judgment"].str.strip()) - set(ALLOWED_JUDGMENTS))
    if bad_labels:
        raise ValueError(f"disallowed judgment labels {bad_labels}; allowed: {list(ALLOWED_JUDGMENTS)}")
    merged["agree"] = [_audit_agrees(j.strip(), m) for j, m in zip(merged["judgment"], merged["matches"])]
    n = int(len(merged))
    disagreements = int((~merged["agree"]).sum())
    rate = disagreements / n
    return {
        "file": str(results_path),
        "n_rows": n,
        "disagreements": disagreements,
        "disagreement_rate": rate,
        "threshold": AUDIT_DISAGREEMENT_MAX,
        "pass": rate <= AUDIT_DISAGREEMENT_MAX,
        "disagreed_rows": merged.loc[~merged["agree"], "row_id"].tolist()[:50],
    }


def freeze_qref(run_dir: Path) -> Path:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        backend = (json.loads(manifest_path.read_text()).get("model") or {}).get("backend")
        if backend == "mock":
            raise RuntimeError(
                "refusing to freeze a mock census into q_ref.lock — "
                "only a real (hf-backend) census run may produce the locked reference"
            )
    q_doc = json.loads((run_dir / "q_ref.json").read_text())
    q_ref_json = run_dir / "q_ref.json"
    payload = {
        "n_digits": q_doc["n_digits"],
        "taxonomy_version": q_doc["taxonomy_version"],
        "q_ref": q_doc["q_ref"],
        "q_ref_ci95": q_doc["q_ref_ci95"],
        "base_accuracy": q_doc["base_accuracy"],
        "census_run_id": run_dir.name,
        "census_run_q_ref_sha256": hashlib.sha256(q_ref_json.read_bytes()).hexdigest(),
        "frozen_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lock_path = ROOT / "configs" / "locked" / "q_ref.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"payload": payload, "sha256": sha}, indent=2, sort_keys=True) + "\n")
    return lock_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", required=True)
    ap.add_argument("--json", default=None, help="write the full gate report here")
    ap.add_argument("--audit-sample", type=int, default=200)
    ap.add_argument("--audit-results", default=None,
                    help="path to the COMPLETED blinded audit CSV (judgment column filled)")
    ap.add_argument("--freeze-qref", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.census)
    cls, q_doc, cov = load_census(run_dir)
    base_accuracy = q_doc.get("base_accuracy")
    tv = json.loads((run_dir / "stability.json").read_text())["split_half_tv"]

    checks = census_core.evaluate_construct_gate(q_doc["q_ref"], cov["coverage"],
                                                 cov["ambiguity_rate"], tv, base_accuracy)
    gate_pass = census_core.gate_passes(checks)

    audit_path, key_path = write_audit_sample(cls, run_dir, args.audit_sample)
    report = {
        "census_run": str(run_dir),
        "checks": checks,
        "gate_pass": gate_pass,
        "manual_audit": {"blinded_file": str(audit_path),
                         "key_file": str(key_path),
                         "status": "pending",
                         "criterion": f"disagreement <= {AUDIT_DISAGREEMENT_MAX}; audit is part of Gate 3"},
        "synthetic_recovery": "see tests/test_classifier.py (D6: 100% at n=5, n=8)",
    }
    audit_result = None
    if args.audit_results:
        audit_result = validate_audit(Path(args.audit_results))
        report["manual_audit"].update({"status": "judged", **audit_result})
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.freeze_qref:
        reasons = []
        if not gate_pass:
            reasons.append("automated construct gate failed")
        if audit_result is None:
            reasons.append("manual audit not judged yet (pass --audit-results <completed CSV>)")
        elif not audit_result["pass"]:
            reasons.append(f"manual-audit disagreement {audit_result['disagreement_rate']:.3f} over threshold")
        if reasons:
            print("NOT frozen: " + "; ".join(reasons))
        else:
            lock = freeze_qref(run_dir)
            print(f"frozen: {lock}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
