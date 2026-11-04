#!/usr/bin/env python
"""GRPO training run with a mode-specific noisy verifier (master plan §13–§16, Phase 5).

One run = one verifier configuration × one seed. TRL 1.17.1 GRPOTrainer with every
theory-relevant setting pinned in configs/base.json (loss_type dapo, β = 0, group
reward scaling, G = 8, T = 1.0, full fine-tune, AdamW, HF-generate backend).

Writes into --run-dir:
  manifest.json            spec, versions, hashes, Slurm id, status, timing, peak VRAM
  generation_config.json   effective sampling settings (asserted neutral)
  metrics.jsonl            trainer logs (reward, loss, ...) + step wall time
  eval.jsonl               every --eval-every steps: greedy true accuracy on the eval split,
                           v3 mode distribution of greedy errors, and a sampled (T = 1)
                           eval-mode census q(t)
  verifier_log/*.parquet   every verifier decision (true correctness kept separate from reward)
  checkpoints/checkpoint-*  model + optimizer + scheduler + RNG + trainer state (resumable)

Usage (cluster): see slurm/grpo.sbatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEUTRAL_SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                    "repetition_penalty": 1.0}


def neutralize_generation_config(model) -> dict:
    """Qwen2.5 ships T = 0.7, top_p = 0.8, top_k = 20, repetition_penalty = 1.1. transformers can
    back-fill fields left at global defaults from the model's config, so overwrite them all."""
    gc = model.generation_config
    for k, v in NEUTRAL_SAMPLING.items():
        setattr(gc, k, v)
    return {k: getattr(gc, k) for k in NEUTRAL_SAMPLING}


def chat(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def build_splits(base: dict):
    from vf_datasets.arithmetic import check_disjoint, make_splits
    splits = make_splits(base["n_digits"], base_seed=base["split_seed"], task=base["task"])
    check_disjoint(splits)
    return splits


def load_verifier_config(args, base: dict):
    from verifiers.matched_verifiers import homogeneous_config, load_config
    from verifiers.qref import load_q_ref
    if args.verifier_config:
        cfg = load_config(args.verifier_config)
        return cfg, {"source": args.verifier_config, "provisional": False}
    if not args.provisional_census:
        raise SystemExit("pass --verifier-config (generated from q_ref.lock) or --provisional-census")
    q = load_q_ref(provisional=args.provisional_census)
    cfg = homogeneous_config(q["q_ref"], base["verifier"]["Fbar"], base["verifier"]["tau"],
                             config_id="provisional_homog")
    return cfg, {k: q[k] for k in ("source", "provisional", "merged")}


class _Recorder:
    """Callbacks: metrics/eval/verifier-log persistence. Built lazily so this module imports without torch."""

    @staticmethod
    def make(trainer_cls_base, tok, splits, base, args, run_dir: Path, vlog: list):
        import numpy as np
        import pandas as pd
        import torch
        from transformers import TrainerCallback

        from error_modes.classify_errors import classify

        scheme = base["scheme"]
        ev = splits["eval"].iloc[: args.eval_size]
        em = splits["eval_mode"].iloc[: args.eval_mode_size]

        def _generate(model, prompts, do_sample: bool, seed: int):
            out = []
            bs = base["eval"]["batch_size"]
            torch.manual_seed(seed)  # sampled eval-mode census: seed = f(run seed, step)
            for lo in range(0, len(prompts), bs):
                texts = [tok.apply_chat_template(chat(p), tokenize=False, add_generation_prompt=True)
                         for p in prompts[lo:lo + bs]]
                enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
                kw = dict(max_new_tokens=base["max_completion_length"], pad_token_id=tok.pad_token_id)
                kw.update(NEUTRAL_SAMPLING if do_sample else {"do_sample": False, "temperature": None,
                                                              "top_p": None, "top_k": None})
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model.generate(**enc, **kw)
                out += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return out

        def _modes(df, texts):
            labels = [classify(int(a), int(b), t, scheme).primary for a, b, t in zip(df["a"], df["b"], texts)]
            s = pd.Series(labels)
            bad = s[s != "C"]
            return float((s == "C").mean()), (bad.value_counts(normalize=True).round(6).to_dict() if len(bad) else {})

        class Recorder(TrainerCallback):
            def __init__(self):
                self.t_last = None

            def _evaluate(self, model, step):
                was_training = model.training
                model.eval()
                t0 = time.time()
                acc, q_err = _modes(ev, _generate(model, ev["prompt"].tolist(), False, args.seed))
                acc_s, q_mode = _modes(em, _generate(model, em["prompt"].tolist(), True, args.seed * 100003 + step))
                rec = {"step": step, "greedy_accuracy": acc, "greedy_error_modes": q_err,
                       "sampled_accuracy": acc_s, "q_t": q_mode, "eval_seconds": round(time.time() - t0, 2),
                       "n_eval": len(ev), "n_eval_mode": len(em)}
                with open(run_dir / "eval.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if was_training:
                    model.train()

            def on_train_begin(self, a, state, control, model=None, **kw):
                done = set()
                if (run_dir / "eval.jsonl").exists():
                    done = {json.loads(l)["step"] for l in (run_dir / "eval.jsonl").read_text().splitlines() if l}
                if args.eval_every > 0 and state.global_step == 0 and 0 not in done:
                    self._evaluate(model, 0)
                self.t_last = time.time()

            def on_step_end(self, a, state, control, model=None, **kw):
                now = time.time()
                with open(run_dir / "metrics.jsonl", "a") as fh:
                    fh.write(json.dumps({"step": state.global_step, "step_seconds": round(now - self.t_last, 3),
                                         "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}) + "\n")
                if args.eval_every > 0 and state.global_step % args.eval_every == 0:
                    self._evaluate(model, state.global_step)
                self.t_last = time.time()

            def on_log(self, a, state, control, logs=None, **kw):
                if logs:
                    with open(run_dir / "metrics.jsonl", "a") as fh:
                        fh.write(json.dumps({"step": state.global_step, **{k: v for k, v in logs.items()
                                                                           if isinstance(v, (int, float))}}) + "\n")

            def _flush(self, step):
                if not vlog:
                    return
                d = run_dir / "verifier_log"
                d.mkdir(exist_ok=True)
                pd.DataFrame(vlog).to_parquet(d / f"upto_step_{step:06d}_{int(time.time())}.parquet", index=False)
                vlog.clear()

            def on_save(self, a, state, control, **kw):
                self._flush(state.global_step)

            def on_train_end(self, a, state, control, **kw):
                self._flush(state.global_step)

        return Recorder()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base", default=str(ROOT / "configs" / "base.json"))
    ap.add_argument("--verifier-config", default=None, help="config JSON generated from q_ref.lock")
    ap.add_argument("--provisional-census", default=None,
                    help="ENGINEERING ONLY: homogeneous config from an unfrozen v3 census q_ref")
    ap.add_argument("--run-kind", choices=("smoke", "bench", "pilot", "grid"), required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-generations", type=int, default=None)
    ap.add_argument("--prompts-per-step", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-size", type=int, default=None)
    ap.add_argument("--eval-mode-size", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--resume", choices=("auto", "none"), default="auto")
    args = ap.parse_args(argv)

    base = json.loads(Path(args.base).read_text())
    G = args.num_generations or base["grpo"]["num_generations"]
    pps = args.prompts_per_step or base["batch"]["prompts_per_step"]
    args.eval_every = base["eval"]["every"] if args.eval_every is None else args.eval_every
    args.eval_size = args.eval_size or base["eval"]["eval_size"]
    args.eval_mode_size = args.eval_mode_size or base["eval"]["eval_mode_size"]
    save_every = args.save_every or base["save_every"]
    if args.provisional_census and args.run_kind in ("pilot", "grid"):
        raise SystemExit("pilot/grid runs require a verifier config generated from q_ref.lock")

    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from models.loading import load_model_and_tokenizer, load_spec
    from scripts.manifest_lib import collect_env, collect_git, lock_hashes
    from verifiers.noisy_verifier import NoisyVerifier, make_reward_fn

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    splits = build_splits(base)
    train = splits["train"]
    train_ds = Dataset.from_dict({"prompt": [chat(p) for p in train["prompt"]],
                                  "a": train["a"].tolist(), "b": train["b"].tolist(),
                                  "prompt_id": train["prompt_id"].tolist()})

    cfg, cfg_meta = load_verifier_config(args, base)
    verifier = NoisyVerifier(cfg, base["verifier"]["verifier_seed"], args.seed, scheme=base["scheme"])
    vlog: list = []
    reward_fn = make_reward_fn(verifier, vlog)

    # fp32 master weights + bf16 autocast: bf16 weights would round away updates at lr ~1e-6.
    spec = dataclasses.replace(load_spec(base["model"]), dtype="float32")
    model, tok = load_model_and_tokenizer(spec)
    tok.padding_side = "left"
    gen_neutral = neutralize_generation_config(model)

    per_dev = base["batch"]["per_device_train_batch_size"]
    if (pps * G) % per_dev or per_dev % G:
        raise SystemExit(f"prompts_per_step*G={pps * G} must be a multiple of the micro-batch {per_dev}, "
                         f"and the micro-batch a multiple of G={G}")
    g = base["grpo"]
    o = base["optim"]
    targs = GRPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=args.lr, max_steps=args.max_steps, seed=args.seed,
        per_device_train_batch_size=per_dev, gradient_accumulation_steps=(pps * G) // per_dev,
        num_generations=G, max_completion_length=base["max_completion_length"],
        loss_type=g["loss_type"], beta=g["beta"], scale_rewards=g["scale_rewards"],
        num_iterations=g["num_iterations"], temperature=g["temperature"], top_p=g["top_p"],
        top_k=g["top_k"], min_p=g["min_p"], repetition_penalty=g["repetition_penalty"],
        epsilon=g["epsilon"], mask_truncated_completions=g["mask_truncated_completions"],
        use_vllm=g["use_vllm"], disable_dropout=g["disable_dropout"], shuffle_dataset=g["shuffle_dataset"],
        log_completions=g["log_completions"],
        optim=o["optim"], lr_scheduler_type=o["lr_scheduler_type"], warmup_steps=o["warmup_steps"],
        weight_decay=o["weight_decay"], adam_beta1=o["adam_beta1"], adam_beta2=o["adam_beta2"],
        adam_epsilon=o["adam_epsilon"], max_grad_norm=o["max_grad_norm"], bf16=o["bf16"],
        gradient_checkpointing=o["gradient_checkpointing"],
        save_strategy="steps", save_steps=save_every, save_total_limit=base["save_total_limit"],
        logging_steps=1, report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
    )
    recorder = _Recorder.make(None, tok, splits, base, args, run_dir, vlog)
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=targs, train_dataset=train_ds,
                          processing_class=tok, callbacks=[recorder])

    eff = trainer.generation_config.to_dict() if getattr(trainer, "generation_config", None) else {}
    effective = {k: eff.get(k) for k in ("do_sample", "temperature", "top_p", "top_k", "min_p",
                                          "repetition_penalty", "max_new_tokens")}
    (run_dir / "generation_config.json").write_text(json.dumps(
        {"trainer": effective, "model_defaults_after_neutralize": gen_neutral}, indent=2) + "\n")
    bad = {k: (effective.get(k), v) for k, v in NEUTRAL_SAMPLING.items()
           if effective.get(k) is not None and effective.get(k) != v}
    if bad:
        raise SystemExit(f"training sampling is not neutral: {bad}")

    ckpts = sorted((run_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    ckpts = [c for c in ckpts if (c / "trainer_state.json").exists()]
    resume = str(ckpts[-1]) if (args.resume == "auto" and ckpts) else None

    manifest = {
        "experiment_id": "veriflow-grpo-v1", "run_kind": args.run_kind, "run_id": run_dir.name,
        "args": vars(args), "base": base, "G": G, "prompts_per_step": pps,
        "verifier_config": json.loads(cfg.canonical_json()), "verifier_config_sha256": cfg.sha256,
        "verifier_config_meta": cfg_meta,
        "predictors": {k: (float(v) if hasattr(v, "__float__") else v) for k, v in cfg.predictors().items()},
        "model": spec.as_dict(), "git": collect_git(ROOT), "env": collect_env(), **lock_hashes(ROOT),
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "job_name": os.environ.get("SLURM_JOB_NAME")},
        "resumed_from": resume, "status": "running", "time": {"start": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    status = "failed"
    try:
        trainer.train(resume_from_checkpoint=resume)
        status = "completed"
    finally:
        manifest.update(status=status, final_step=int(trainer.state.global_step),
                        peak_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        device=torch.cuda.get_device_name(0))
        manifest["time"].update(end=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds=round(time.time() - t0, 1))
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
