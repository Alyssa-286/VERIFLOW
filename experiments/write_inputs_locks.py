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

from error_modes.classify_errors import DESIGNED_MODES, PRIORITY, TAXONOMY_VERSION
from vf_datasets.arithmetic import PROMPT_TEMPLATE, PROMPT_TEMPLATE_SHA256

LOCK_DIR = Path(__file__).resolve().parents[1] / "configs" / "locked"


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
    write_lock("prompt.lock", {"template": PROMPT_TEMPLATE, "sha256": PROMPT_TEMPLATE_SHA256})
    write_lock(
        "taxonomy.lock",
        {
            "version": TAXONOMY_VERSION,
            "priority": list(PRIORITY),
            "designed_modes": list(DESIGNED_MODES),
            "spec": "docs/VERIFLOW_MASTER_PLAN.md §9.2",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
