"""Pinned model and tokenizer loading (master plan §16.5, §21).

Every CUDA run must load the model at exactly the revision recorded in
configs/model/<model>.yaml. Loading "main" is forbidden: it silently changes
the weights between runs. This module:

  - resolves a model spec from the pinned YAML files (no PyYAML dependency);
  - refuses an unknown or unpinned revision;
  - exposes load_model_and_tokenizer() plus revision helpers;
  - is cache- and offline-friendly: callers set HF_HOME/HF_HUB_OFFLINE
    (see slurm/_common.sh), which env.py also checks.

torch/transformers are imported lazily so the CPU test suite runs without them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = _ROOT / "configs" / "model"


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    tokenizer_revision: str
    dtype: str
    name: str  # yaml filename stem

    def as_dict(self) -> dict:
        return {
            "name": self.name, "repo": self.repo, "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision, "dtype": self.dtype,
        }


def load_spec(name: str) -> ModelSpec:
    """Resolve configs/model/<name>.yaml. `_sha`-style self-hashes make this
    parsing deliberately tiny: the expected form is documented in the file."""
    path = MODELS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown model spec {name!r}: {path} does not exist")
    vals: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, v = line.partition(":")
        vals[key.strip()] = v.strip()
    missing = {"repo", "revision", "tokenizer_revision", "dtype"} - vals.keys()
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    for k in ("revision", "tokenizer_revision"):
        if len(vals[k]) != 40 or not all(c in "0123456789abcdef" for c in vals[k]):
            raise ValueError(f"{path}: {k} must be a 40-hex git revision, got {vals[k]!r}")
    return ModelSpec(repo=vals["repo"], revision=vals["revision"],
                     tokenizer_revision=vals["tokenizer_revision"],
                     dtype=vals["dtype"], name=name)


def expected_revision(name: str) -> str:
    return load_spec(name).revision


def local_revision(hf_home: Path, repo: str) -> str | None:
    """Revision recorded in a downloaded HF snapshot cache, if present."""
    refs = hf_home / "hub" / f"models--{repo.replace('/', '--')}" / "refs"
    for ref in ("main",):
        p = refs / ref
        if p.is_file():
            return p.read_text().strip() or None
    return None


def snapshot_dir(hf_home: Path, repo: str, revision: str) -> Path:
    return hf_home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def check_revision(spec: ModelSpec, hf_home: Path) -> dict:
    """Compare the cached snapshot with the pinned revision.

    Downloading by commit hash creates snapshots/<sha>/ without a refs/main, so
    the pinned snapshot directory (config + safetensors weights) is the primary
    evidence; a refs/main that names another commit is a mismatch.
    Returns a dict {ok, cached, pinned, snapshot_present, repo}. Never raises."""
    cached = local_revision(hf_home, spec.repo)
    snap = snapshot_dir(hf_home, spec.repo, spec.revision)
    present = (snap / "config.json").is_file() and any(snap.glob("*.safetensors"))
    return {"repo": spec.repo, "cached": cached, "pinned": spec.revision,
            "snapshot_present": present,
            "ok": present and (cached is None or cached == spec.revision)}


def load_model_and_tokenizer(spec: ModelSpec, *, offline: bool = True):
    """Load the pinned model/tokenizer. Raises when the cache ref disagrees
    with the pinned revision while offline (weights would be wrong)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import os
    env_hf = os.environ.get("HF_HOME")
    hf_home = Path(env_hf) if env_hf else Path.home() / ".cache" / "huggingface"
    chk = check_revision(spec, hf_home)
    if offline and not chk["ok"]:
        raise RuntimeError(
            f"cached revision for {spec.repo} is {chk['cached']} but {spec.revision} "
            f"is pinned; refusing offline load (fix the cache: delete refs and re-download)"
        )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[spec.dtype]
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.tokenizer_revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.repo, revision=spec.revision, dtype=dtype, device_map=None
    )
    return model, tok


def manifest_entry(spec: ModelSpec) -> dict:
    return spec.as_dict()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
