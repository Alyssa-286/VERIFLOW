#!/usr/bin/env python
"""Export the per-run records of a run tree for the public archive, with private infrastructure details redacted.

Copies, for every run directory under --src: manifest.json, eval.jsonl, metrics.jsonl, generation_config.json
into --dst/<run>/ and (optionally) packs verifier_log/*.parquet into one tar.gz. Checkpoints are never exported.

Redaction (manifest.json only; no other file contains these strings, which is checked):
  absolute cluster paths "/home/<user>/VERIFLOW/..." -> "<REPO>/..."; "hostname" -> "<cluster-node>".
Nothing used by the validator or the analysis is touched (status, args except the absolute base path, config and
lock hashes, git commit). An index with the SHA-256 of every original and exported file is written to
--dst/INDEX.json, so the archive can be tied to the originals kept on the cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

FILES = ("manifest.json", "eval.jsonl", "metrics.jsonl", "generation_config.json")
HOME = re.compile(r"/home/[^/\"]+/VERIFLOW")
PRIVATE = re.compile(r"/home/[^/\"]+|<cluster-user>\d+|192\.168\.\d+\.\d+|sit-gpu|<login-host>")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def redact_manifest(raw: bytes) -> bytes:
    m = json.loads(raw)
    txt = HOME.sub("<REPO>", json.dumps(m, indent=2))
    m = json.loads(txt)

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "hostname":
                    o[k] = "<cluster-node>"
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(m)
    return (json.dumps(m, indent=2) + "\n").encode()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verifier-tar", default=None)
    a = ap.parse_args(argv)
    src, dst = Path(a.src), Path(a.dst)
    index = {}
    runs = sorted(p for p in src.iterdir() if (p / "manifest.json").exists())
    for rd in runs:
        (dst / rd.name).mkdir(parents=True, exist_ok=True)
        for f in FILES:
            raw = (rd / f).read_bytes()
            out = redact_manifest(raw) if f == "manifest.json" else raw
            if PRIVATE.search(out.decode("utf-8", "replace")):
                raise SystemExit(f"private string left in {rd.name}/{f}")
            (dst / rd.name / f).write_bytes(out)
            index[f"{rd.name}/{f}"] = {"original_sha256": sha(raw), "exported_sha256": sha(out)}
    if a.verifier_tar:
        with tarfile.open(a.verifier_tar, "w:gz") as tf:
            for rd in runs:
                for p in sorted((rd / "verifier_log").glob("*.parquet")):
                    index[f"{rd.name}/verifier_log/{p.name}"] = {"original_sha256": sha(p.read_bytes())}
                    tf.add(p, arcname=f"{rd.name}/verifier_log/{p.name}")
    (dst / "INDEX.json").write_text(json.dumps({"n_runs": len(runs), "files": index}, indent=1) + "\n")
    print(f"exported {len(runs)} runs to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
