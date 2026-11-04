# Gate 4: feasibility (Phase 5). Verdict: **PASS** (2026-11-04)

**Scope:** the training stack runs end to end on the SIT cluster, resumes from
checkpoints and logs every verifier decision replayably. The measured cost of
the planned grid fits the budget (COMPLETION_PLAN.md Stage 2: ≤ 150 GPU-h).

## Stack (D17)

| Item | Value |
|---|---|
| Trainer | TRL **1.17.1** `GRPOTrainer` (pinned), transformers 4.57.6, torch 2.8.0+cu128 |
| Generation backend | **HF generate**. vLLM is not used: NVML is broken on `sit-gpu` (driver/library mismatch) and vLLM would also force a different torch. Fixed for every run. |
| Pinned GRPO settings | `configs/base.json`: loss_type `dapo`, β = 0, scale_rewards `group`, num_iterations 1, G = 8, T = 1.0, top_p 1.0, top_k 0, repetition_penalty 1.0, ε = 0.2, no truncation masking |
| Batch | 32 prompts × G = 8 = 256 completions per optimizer step (micro-batch 64 × 4 accumulation) |
| Optimisation | full fine-tune, AdamW (β 0.9/0.999, ε 1e-8, wd 0), constant LR, grad-norm clip 1.0, **fp32 master weights + bf16 autocast** (bf16 weights would round away updates at lr ≈ 1e-6) |
| Sampling safety | Qwen2.5 ships T 0.7 / top_p 0.8 / top_k 20 / repetition 1.1. The trainer overwrites the model's generation config and **asserts** that the effective training sampling is neutral; this is recorded in each run's `generation_config.json`. |
| Data | taxonomy-v3 scheme, n = 5, split seed 20261104 (Amendment 1); train 50,000 / eval 1,000 / eval-mode 1,000, disjoint |
| Logging | `eval.jsonl` (greedy accuracy, v3 error modes, sampled q(t)), `metrics.jsonl`, `verifier_log/*.parquet` (every decision; true correctness separate from reward), `manifest.json` |

## Smoke (job 3667, 1 min 14 s): all 17 checks true

Uses a *provisional* homogeneous verifier (F̄ 0.3, τ 0.9 placeholders) built from the unfrozen census q_ref: engineering only.

- **Runs:** a 1-step run, a 10-step run saving at steps 5 and 10, and a copy
  truncated to checkpoint-5 that **resumed and completed to step 10**.
- **Verifier log:** 256 decisions per step. Offline PRF replay reproduces every
  u exactly, reward = 1[u < ρ], and `I` is never rewarded.
- **Realised vs configured rates** (10-step run, 2,560 decisions):

  | Class | n | Configured | Realised |
  |---|---|---|---|
  | C | 1,216 | 0.900 | 0.908 |
  | LEN | 166 | 0.309 | 0.289 |
  | MULTI | 520 | 0.309 | 0.298 |
  | ONE_HI | 310 | 0.309 | 0.319 |
  | ONE_LO | 313 | 0.309 | 0.294 |

- **Effective sampling:** do_sample, T 1.0, top_p 1.0, top_k 0, repetition 1.0.
- **Training moves the model:** greedy eval accuracy (100 prompts) was 0.41 at
  step 0, 0.55 at step 5 and 0.74 at step 10, at lr 1e-6. Learning is fast, so
  the pilot's horizon rule (§13) matters.

## Throughput (job 3669)

| G | Median step | Peak VRAM | Full eval (1,000 greedy + 1,000 sampled) |
|---|---|---|---|
| 8 | **1.09 s** | 17.4 GB | 2.3–3.3 s |
| 16 | 2.69 s | 41.8 GB | — |

Completions under the answer-only prompt are a few tokens long, which is why
rollouts are much cheaper than the §17 estimate (8–18 s per step). Occasional
slow steps (up to 15 s at G = 16) come from rare long completions.

## Projection and budget

| Item | GPU-h (G = 8) |
|---|---|
| One run, T = 500 / 800 (eval every 50) | 0.16 / 0.26 |
| Pilot (4 LR runs × 800 + 12 heterogeneity runs × 500) | ≈ 3 |
| Minimum grid (24 runs) | ≈ 4 |
| **Standard grid (64 runs)** | **≈ 10** |

**Gate 4: PASS.** The standard design fits the 150 GPU-h budget with more than
10× margin, so no downsizing is needed. Because the margin is so large, the
pre-registration (Stage 4) can consider the master plan's *extended* design
(more configurations); that is a researcher decision. The projections assume
completion lengths stay short during training, and the pilot measures this.

## Open items carried to Stage 3

- **F̄\* and τ\*** are not fixed by the master plan. They must be chosen and
  logged before the pilot. The current 0.3 / 0.9 are placeholders.
- **q_ref.lock** (pending the researcher's audit) is required before any
  pilot or grid configuration is generated; the trainer refuses provisional
  q_ref for `pilot`/`grid` runs.
- **1.5B replication arm:** its census failed split-half TV (D16), so its
  design must be decided before Stage 5.
