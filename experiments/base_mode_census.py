#!/usr/bin/env python
"""Base-policy error census (master plan §11) — GPU job, first VERIFLOW GPU use.

Generates completions for the census split at 1.0 temperature / top_p 1.0 with
the pinned model, classifies them with taxonomy v1, and writes every §11
artifact plus a run manifest. All heavy numerical work lives in
`census_core.py`, which is unit-tested on CPU without torch.

Backends:
  hf   : transformers generate on a CUDA device (the real census).
  mock : a deterministic seeded pseudo-policy that realizes each error mode by
         construction (CPU tests / smoke runs; never used for decisions).

Usage (cluster):
  python -m experiments.base_mode_census --n 7 --model qwen0p5b \
      --samples 4 --seed 20261102 --out runs/census

Outputs in <out>/n<digits>_s<seed>/:
  generations.parquet, classification.parquet, q_ref.json, per_mode_lengths.json,
  difficulty_hist.csv, per_prompt_summary.parquet, stability.json,
  coverage_ambiguity.json, params.json, manifest.json
q_ref.lock freezing is a separate, explicit step (check_construct_gate.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from error_modes.classify_errors import TAXONOMY_VERSION
from experiments import census_core
from experiments.mock_policy import mock_completions
from scripts.manifest_lib import collect_env, collect_git, lock_hashes
from vf_datasets.arithmetic import DEFAULT_SIZES, check_disjoint, frame_sha256, make_splits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_NEW_TOKENS = 128  # answers are short; recorded in params.json


# ------------------------------------------------------------------ backends

def generate_hf(splits, n_digits, model_name, samples, seed, max_new_tokens, out_dir, batch_size=64):
    """Real census generation, chunked to bound GPU memory. Returns (generations
    DataFrame, model metadata, generation kwargs)."""
    import torch
    from models.loading import load_model_and_tokenizer, load_spec

    spec = load_spec(model_name)
    model, tok = load_model_and_tokenizer(spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("generate_hf requires CUDA; use --backend mock for CPU smoke runs")
    model = model.eval().to(device)

    torch.manual_seed(seed)  # a single stream; run params record it
    gen_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,          # census sampling must match training sampling (§11)
        "top_k": 0,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": samples,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
    }
    df = splits["census"]
    rows = []
    prompt_id = df["prompt_id"].tolist()
    for lo in range(0, len(df), batch_size):
        chunk = df.iloc[lo:lo + batch_size]
        chat = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True)
                for p in chunk["prompt"].tolist()]
        enc = tok(chat, return_tensors="pt", padding=True, padding_side="left").to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        gen_only = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        cis = chunk["prompt_id"].to_numpy().repeat(samples)
        as_ = chunk["a"].to_numpy().repeat(samples)
        bs = chunk["b"].to_numpy().repeat(samples)
        sample_idx = np.tile(np.arange(samples), len(chunk))
        for pid, a, b, s_idx, text in zip(cis, as_, bs, sample_idx, texts):
            rows.append({"prompt_id": pid, "a": int(a), "b": int(b),
                         "sample_idx": int(s_idx), "completion": text})
    return pd.DataFrame(rows), spec.as_dict(), gen_kwargs


def generate_mock(splits, n_digits, samples, seed, mock_probs):
    """Deterministic pseudo-policy for CPU tests. mock_probs maps mode -> prob
    (C included). See mock_policy.py for how each mode is realized."""
    rows = []
    df = splits["census"]
    answers = df["answer"].to_numpy()
    prompts = df["prompt_id"].to_numpy()
    for pid, a, b, s in zip(prompts, df["a"], df["b"], answers):
        comps = mock_completions(int(a), int(b), seed, str(pid), mock_probs, samples)
        for j, text in enumerate(comps):
            rows.append({"prompt_id": pid, "a": int(a), "b": int(b),
                         "sample_idx": j, "completion": text})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=7, help="n_digits (censuses run n=5..10)")
    ap.add_argument("--model", default="qwen0p5b")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--size", type=int, default=None, help="census split size (default 2000)")
    ap.add_argument("--seed", type=int, default=20261102, help="generation + prompt seed")
    ap.add_argument("--base-seed", type=int, default=20261102, help="split construction seed")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--backend", choices=("hf", "mock"), default="hf")
    ap.add_argument("--mock-probs", default=None, help="JSON file {mode: prob} for the mock backend")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/census")
    args = ap.parse_args(argv)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(args.out) / f"n{args.n}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = {**DEFAULT_SIZES}
    if args.size is not None:
        sizes["census"] = args.size
    splits = make_splits(args.n, sizes=sizes, base_seed=args.base_seed)
    check_disjoint(splits)  # leakage guard; raises on any overlap

    mock_probs: dict | None = None
    if args.backend == "mock":
        if args.mock_probs is None:
            mock_probs = json.loads((ROOT / "configs" / "mock_base_policy.json").read_text())
        else:
            mock_probs = json.loads(Path(args.mock_probs).read_text())

    gens_meta = {}
    spec_meta = {"backend": args.backend}
    if args.backend == "hf":
        gens, spec_meta, gen_kwargs = generate_hf(splits, args.n, args.model,
                                                  args.samples, args.seed,
                                                  args.max_new_tokens, out_dir)
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
        gens_meta = {"sampling": gen_kwargs}
    else:
        gens = generate_mock(splits, args.n, args.samples, args.seed, mock_probs)
        gens_meta = {"mock_probs": mock_probs, "max_new_tokens": args.max_new_tokens}

    cls = census_core.classify_generations(gens)
    cls["sample_idx"] = gens["sample_idx"].to_numpy()  # row order is preserved
    per_prompt = census_core.per_prompt_summary(cls)
    q_point = census_core.q_ref_point(cls)
    q_ci = census_core.cluster_bootstrap_q(cls, n_boot=args.n_boot, seed=args.seed)
    coverage = census_core.coverage_and_ambiguity(cls)
    tv = census_core.split_half_tv(cls)
    base_accuracy = float(per_prompt["accuracy"].mean())

    checks = census_core.evaluate_construct_gate(q_point, coverage["coverage"],
                                                 coverage["ambiguity_rate"], tv, base_accuracy)
    gate_pass = census_core.gate_passes(checks)

    cls.to_parquet(out_dir / "classification.parquet", index=False)
    gens.merge(cls[["prompt_id", "sample_idx", "completion", "mode", "matches"]],
               on=["prompt_id", "sample_idx", "completion"], how="left"
               ).to_parquet(out_dir / "generations.parquet", index=False)
    per_prompt.to_parquet(out_dir / "per_prompt_summary.parquet", index=False)
    census_core.per_mode_lengths(cls).to_json(out_dir / "per_mode_lengths.json", orient="records")
    census_core.accuracy_histogram(per_prompt).rename("n_prompts").to_csv(out_dir / "difficulty_hist.csv")

    q_ref_doc = {
        "n_digits": args.n,
        "taxonomy_version": TAXONOMY_VERSION,
        "q_ref": q_point,
        "q_ref_ci95": q_ci,
        "base_accuracy": base_accuracy,
        "n_incorrect": coverage["n_incorrect"],
        "n_generations": int(len(gens)),
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    q_ref_json = out_dir / "q_ref.json"
    q_ref_json.write_text(json.dumps(q_ref_doc, indent=2, sort_keys=True) + "\n")
    stability = {"split_half_tv": tv, "tv_threshold": census_core.GATE_TV_MAX}
    (out_dir / "stability.json").write_text(json.dumps(stability, indent=2) + "\n")
    (out_dir / "coverage_ambiguity.json").write_text(json.dumps({**coverage, "gate": checks, "gate_pass": gate_pass}, indent=2) + "\n")

    git = collect_git(ROOT)
    params = {
        "kind": "census", "n_digits": args.n, "model": spec_meta, "samples": args.samples,
        "seed": args.seed, "base_seed": args.base_seed, "split_sizes": sizes,
        "max_new_tokens": args.max_new_tokens, **gens_meta,
        "prompt_sha256": frame_sha256(splits["census"]),
        "dataset_sha256": {k: frame_sha256(v) for k, v in splits.items()},
        "git": git, "env": collect_env(), "time": {"start": started, "end": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "gate_pass": gate_pass,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    git = collect_git(ROOT)
    import os
    import hashlib
    manifest = {
        "experiment_id": "veriflow-census-v1",
        "run_id": out_dir.name,
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "model": spec_meta if args.backend == "hf" else {"backend": "mock"},
        "taxonomy_version": TAXONOMY_VERSION,
        "prompt_template_sha256": params["prompt_sha256"],
        "dataset_sha256": params["dataset_sha256"],
        **lock_hashes(ROOT),
        "q_ref_sha256": hashlib.sha256(q_ref_json.read_bytes()).hexdigest(),
        "seed": args.seed,
        "env": params["env"],
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"),
                  "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                  "job_name": os.environ.get("SLURM_JOB_NAME")},
        "time": params["time"],
        "status": "completed" if gate_pass else "gate_failed",
        "artifacts": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"out_dir": str(out_dir), "q_ref": q_point, "gate_pass": gate_pass,
                      "base_accuracy": base_accuracy, "tv": tv, **coverage}, indent=2))
    return 0 if gate_pass else 1  # nonzero: the §9.3 construct gate failed


if __name__ == "__main__":
    raise SystemExit(main())
