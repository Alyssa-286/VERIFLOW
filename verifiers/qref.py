"""Reference error-mode distribution for verifier configurations (master plan §10.2, Amendment 1).

`load_q_ref()` reads the frozen `configs/locked/q_ref.lock` and refuses anything else, unless
`provisional=<census run dir>` is passed explicitly. Provisional q_ref values are for
engineering only (smoke tests, throughput benchmarks) and are tagged as such in every config
and manifest; they never feed a pilot or confirmatory run.

`merge_small_modes()` applies the pre-declared Amendment 1 rule before configurations are
generated: NEG merges into LEN if q(NEG) < 0.02; any other designed v3 mode below 0.02
merges into MULTI.
"""

from __future__ import annotations

import json
from pathlib import Path

from error_modes.classify_errors import DESIGNED_MODES_V3

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "locked" / "q_ref.lock"
MERGE_BELOW = 0.02


def merge_small_modes(q: dict) -> tuple[dict, dict]:
    """Return (merged q_ref, {merged_mode: target}) per Amendment 1. Masses are preserved."""
    q = {k: float(v) for k, v in q.items()}
    merged: dict = {}
    if "NEG" in q and q["NEG"] < MERGE_BELOW:
        q["LEN"] = q.get("LEN", 0.0) + q.pop("NEG")
        merged["NEG"] = "LEN"
    for m in [m for m in DESIGNED_MODES_V3 if m in q and m not in ("NEG", "MULTI")]:
        if q[m] < MERGE_BELOW:
            q["MULTI"] = q.get("MULTI", 0.0) + q.pop(m)
            merged[m] = "MULTI"
    total = sum(q.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"q_ref does not sum to 1 after merging ({total})")
    return q, merged


def load_q_ref(provisional: str | Path | None = None) -> dict:
    """{'q_ref': merged q, 'merged': {...}, 'source': ..., 'provisional': bool, 'sha256': ...}."""
    if provisional is None:
        if not LOCK.exists():
            raise FileNotFoundError(
                f"{LOCK} does not exist: q_ref is frozen only after Gate 3' (incl. the manual audit). "
                "Pass provisional=<census run dir> for engineering runs only."
            )
        doc = json.loads(LOCK.read_text())
        q_raw, source, sha, prov = doc["payload"]["q_ref"], str(LOCK), doc["sha256"], False
    else:
        run = Path(provisional)
        doc = json.loads((run / "q_ref.json").read_text())
        if doc.get("scheme") != "add_v3":
            raise ValueError(f"{run} is not a taxonomy-v3 census")
        q_raw, source, sha, prov = doc["q_ref"], str(run / "q_ref.json"), None, True
    q, merged = merge_small_modes(q_raw)
    return {"q_ref": q, "merged": merged, "source": source, "provisional": prov, "lock_sha256": sha}
