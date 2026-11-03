#!/usr/bin/env python
"""Generate configs/locked/prompt.lock and configs/locked/taxonomy.lock.

Run once at Phase 4; re-running overwrites the file, so any change must go
through docs/DECISIONS.md first (locked-protocol rule, §28).

Each lock file is {"payload": <dict>, "sha256": <h>} where h is the SHA-256 of
the canonical JSON of <payload> alone. Verifiers re-hash the payload to check.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from error_modes.classify_errors import (DESIGNED_MODES, DESIGNED_MODES_SUB, PRIORITY, PRIORITY_SUB,
                                        DESIGNED_MODES_V3, PRIORITY_V3, SUB_DIAGNOSTICS,
                                        SUB_TAXONOMY_VERSION, TAXONOMY_VERSION, V3_TAXONOMY_VERSION)
from vf_datasets.arithmetic import (PROMPT_TEMPLATE, PROMPT_TEMPLATE_SHA256, PROMPT_VERSION,
                                    SUB_PROMPT_TEMPLATE, SUB_PROMPT_TEMPLATE_SHA256)

from experiments.base_mode_census import DEFAULT_MAX_NEW_TOKENS

LOCK_DIR =Path(__file__).resolve().parents[1] / "configs" / "locked"


def write_lock(name: str, payload: dict) -> Path:
    sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = LOCK_DIR / name
    path.write_text(json.dumps({"payload": payload, "sha256": sha}, indent=2, sort_keys=True) + "\n")
    print(f"{path}: payload_sha256={sha}")
    return path


def main() -> int:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    write_lock("prompt.lock", {"version": PROMPT_VERSION, "template": PROMPT_TEMPLATE,
                               "sha256": PROMPT_TEMPLATE_SHA256,
                               "max_new_tokens": DEFAULT_MAX_NEW_TOKENS})
    write_lock(
        "taxonomy.lock",
        {
            "version": TAXONOMY_VERSION,
            "priority": list(PRIORITY),
            "designed_modes": list(DESIGNED_MODES),
            "spec": "docs/VERIFLOW_MASTER_PLAN.md §9.2",
        },
    )
    # Subtraction (§9.3 ladder step 2, D13): separate files so the addition locks are untouched.
    write_lock("prompt_sub.lock", {"version": PROMPT_VERSION, "task": "sub", "template": SUB_PROMPT_TEMPLATE,
                                   "sha256": SUB_PROMPT_TEMPLATE_SHA256,
                                   "max_new_tokens": DEFAULT_MAX_NEW_TOKENS})
    write_lock(
        "taxonomy_sub.lock",
        {
            "version": SUB_TAXONOMY_VERSION,
            "parser": TAXONOMY_VERSION,
            "priority": list(PRIORITY_SUB),
            "designed_modes": list(DESIGNED_MODES_SUB),
            "u_diagnostics": list(SUB_DIAGNOSTICS),
            "spec": "docs/DECISIONS.md D13",
        },
    )
    # Amendment 1 (D15): taxonomy v3 outcome partition for addition.
    write_lock(
        "taxonomy_v3.lock",
        {
            "version": V3_TAXONOMY_VERSION,
            "parser": TAXONOMY_VERSION,
            "priority": list(PRIORITY_V3),
            "designed_modes": list(DESIGNED_MODES_V3),
            "one_digit_split": "ONE_HI iff position < floor(L/2) from the most significant digit",
            "split_seed": 20261104,
            "spec": "docs/AMENDMENT_1.md",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
