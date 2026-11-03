# VERIFLOW Decision Log

Append-only. Each entry records a date, the decision, its evidence, and what it locks. Entries are never edited after confirmatory data exist; corrections are added as new entries.

---

## D0: Gate 0 (novelty): PASS with modification. 2026-10-31

**Evidence:** `VERIFLOW_MASTER_PLAN.md` §5–6. Full-text audit (arXiv API + PDFs) of Egashira et al. 2605.02909, Rad et al. 2601.04411 and Moya et al. 2609.35677, plus 14 further papers.

**Findings:**
- No paper found holds the aggregate FPR (under a reference distribution) and TPR fixed while varying the per-error-mode false-acceptance distribution.
- Rad et al. assume block symmetry across bad modes.
- Egashira et al. already establish that outcomes are not determined by aggregate FPR or Youden's J.

**Decision:** VERIFLOW proceeds with the modified question (§2). V is the primary statistic. f_max, trigger×conditional advantage, V_q2, κ3 and the CGF plug-in are pre-declared competitors. The original title is retired.

**Caveat:** WebSearch was unavailable, so the search covered arXiv only. A Scholar/Semantic Scholar sweep is still owed; if it finds a matched-rate dispersion RLVR study, re-open Gate 0.

## D1: Fixed conventions. 2026-10-31

1. The primary aggregate FPR is weighted by the frozen census distribution q_ref, not by q(t) and not macro-averaged.
2. f_U = F̄ and f_I = 0 in every configuration.
3. τ is identical across matched configurations.
4. The primary outcome is the logit of mean held-out greedy accuracy over the window [T/2, T].
5. The inferential unit is the verifier configuration.

## D3: Gate 1 (mathematics): PASS. 2026-11-01

See `docs/GATE1.md`. Props 1–7 are verified numerically (21 tests). Regime structure: V dominates at short horizon under natural PG, f_max at long horizon; the CGF is exact. Vanilla PG is poorly predicted early, and V_q2 is relevant there. **Locks:** V stays primary; the competitor set is unchanged.

## D4: Verifier randomness uses a SHA-256 PRF. 2026-11-01

`verifiers/noisy_verifier.py` derives each uniform from SHA-256 of (verifier_seed, run_seed, step, prompt_id, sample_idx). The plan said "Philox". A hash PRF has the same required properties (counter-based, keyed, replayable, independent of global RNGs) and is simpler to replay from logs. Tested in `tests/test_verifiers.py`.

## D5: Gate 2 (simulation): PASS. 2026-11-01

See `docs/GATE2.md`. In 9 dynamics settings × 3 horizons, adding V to f_max reduced leave-one-config-out error in 27/27 cells (all bootstrap CIs exclude 0). The V coefficient given f_max was negative in all families. **Locks:** V primary (no switch to V_q2); the CGF predictor kept as H3; the horizon must be fixed from the pilot before the grid.

## D6: Phase 3 construct tooling complete; one construct risk flagged. 2026-11-01

Taxonomy v1, parser, splits and noisy verifier are implemented and tested. Synthetic recovery is 100 % for all designed modes at n = 5 and 8.

**Flag:** 32–53 % of injected dropped-carry errors also match the one-digit-slip rule (expected: a single dropped carry changes one digit by one). If the real-census ambiguity for CARRY/DIGIT exceeds 10 % (§9.3.3), merge them into one mode CARRY∪DIGIT, decided at Gate 3 before any configs are generated. TRUNC and DIGIT are primary in 92–98 % of injections; the remainder match a higher-priority rule.

## D2: Task choice. 2026-10-31

The task is two-operand integer addition, not ETH's decimal chain-sum. The reason is construct validity of the deterministic taxonomy. Direct numerical comparison with Egashira et al. is therefore not possible; a replication of one of their configurations is an extended-design item.

## D7: Phase 4 tooling complete (CPU-verified); cluster policy surveyed read-only. 2026-11-02

**Implemented and tested:** census core + watchdog-free orchestration (`experiments/census_core.py`, `experiments/base_mode_census.py`, mock backend for CPU end-to-end runs), §9.3 gate evaluator (`experiments/check_construct_gate.py`, refuses to freeze mock data), manifest machinery (`scripts/manifest_lib.py` etc., schema-tested), preflight/gpucheck (`scripts/preflight.py`, `scripts/gpucheck.py`, slurm/preflight.sbatch, slurm/gpucheck.sbatch), census sbatch, isolation guard (`scripts/preflight.check_isolation`, `tests/test_isolation.py`). `configs/model/*.yaml` pin Qwen2.5-0.5B-Instruct @ `7ae557604adf67be50417f59c2c2f167def9a775` and Qwen2.5-1.5B-Instruct @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (both verified against the HF API on 2026-11-02, Apache-2.0). `configs/locked/prompt.lock` + `taxonomy.lock` payload hashes recorded. GPU-versions of deps (torch≥2.6, transformers≥4.56.2) live in the `gpu` dependency group; TRL/vLLM wait for Phase 5.

**Cluster survey (read-only only, no job submitted):** see `docs/CLUSTER_POLICY.md`. Node sit-gpu has 2 GPUs, 96 CPUs, 515 GB RAM, MaxTime infinite. One GPU was held by the user's own `p12-exploratory` job; VERIFLOW arrays therefore throttle to `%1`. GPUs' driver version and bf16 support are gated on `veriflow-gpucheck`, not assumed.

**Not yet done (needs the cluster):** the real census for n=5..10, the q_ref.lock freeze, and the manual audit (§9.3.6). Gate 3 remains open.

## D8: Real census run; Gate 3 FAILS at ladder step 1; q_ref NOT frozen. 2026-11-03

**Executed on SIT HPC** (`~/VERIFLOW`, branch `phase4/hpc-ready`): preflight job 3590 (124 tests green, isolation ok), GPU check 3591 (failed: NVML mismatch, see below), GPU check 3593 (pass), census jobs 3594–3597, 3600, 3602 (n = 5…10; 3598/3599 crashed, see GATE3.md). Each census takes about 66 s of generation (8,000 samples) and about 72 s of job time on one L40. Peak VRAM is about 2.1 GB and the run directory is 1.6–2.4 MB. Cost for all six n is about 0.12 GPU-h, far below the §11 estimate of 0.2–0.5 GPU-h *per n*.

**Result:** every n fails §9.3.1 (coverage 0.02–0.09), §9.3.2 (0–1 modes ≥ 5 %), §9.3.3 (ambiguity 0.12–0.18) and the §9.1 accuracy window (0.015–0.176). Only §9.3.4 TV passes. No n is selected, `q_ref.lock` is not written and no audit is exported. Table and diagnosis: `docs/GATE3.md`, evidence in `docs/gate3_census_comparison.json`.

**Fixes made before or while running (none changes a research definition):**
- Census sampling now overrides Qwen2.5's shipped `generation_config` (`repetition_penalty=1.1`, T=0.7, top_p=0.8, top_k=20). Without this, `repetition_penalty` 1.1 would have silently applied and the census would not be §11's pure T=1.0/top_p=1.0 sampling. The census refuses to run if any shipped sampling default is left un-overridden.
- The parsed answer is stored as a string (int64 overflow at n ≥ 9), and raw generations are persisted before classification.
- The manifest's `prompt_template_sha256` now holds the template hash. It used to hold the census-split hash, which is now `census_prompts_sha256`.
- Each census job gets its own run id (`n<N>_s<seed>_j<jobid>`); an existing run directory is never overwritten.
- `preflight.sbatch` syncs the `gpu` group. A bare `uv sync --frozen` would uninstall torch.
- Audit validity: `AMBIG` agrees only with rows the classifier also found multi-rule. Previously an all-AMBIG audit scored 0 % disagreement. The q_ref lock refuses a missing manifest or an unpinned revision, and records the model revision, prompt/taxonomy hashes, Slurm job and git commit.

**Open decision (researcher):** the §9.3 ladder's next permitted step is an answer-only prompt. That needs (a) a new prompt template, which becomes `prompt.lock` v2, and (b) a max_new_tokens choice. 45 % of all v1 generations hit the 128-token code default, and §11 ties the cap to the not-yet-fixed training max_completion_length. Neither is predeclared, so neither was improvised. The DIGIT-only designed mass and U ≥ 15 % at n ≥ 6 make it likely that a format fix alone cannot reach 70 % coverage beyond n = 5. In that case the ladder continues to subtraction, then M = 3, then the kill rule.

## D9: Prompt v2 (answer-only) and max_new_tokens = 256, locked before any v2 data. 2026-11-03

Approved by the researcher after D8. This is the §9.3 ladder's answer-only-prompt step.

- Template v2: `Compute {a} + {b}.` / `Reply with only the final answer, exactly in the form` / `Answer: <integer>`. `prompt.lock` payload sha256 `918a83d3…e492a8`, template sha256 `5cd38864…6261c5`. v1 is kept as `PROMPT_TEMPLATE_V1` and in git history; v1 runs are evidence only and can't be selected.
- `max_new_tokens = 256`, recorded in `prompt.lock`. It becomes the training `max_completion_length` (§11).
- Unchanged: census split operands, seeds, 2,000 × 4 sampling, T=1.0/top_p=1.0/top_k=0/repetition_penalty=1.0, taxonomy v1 (`taxonomy.lock` hash unchanged), gate thresholds, §9.1 selection rule.
- No further wording variants are tried on census data. If v2 fails, the ladder continues to subtraction.
- Run ids: `n<N>_p2_s20261102_j<job>`.

## D10: Prompt v2 census also fails Gate 3 at every n; q_ref NOT frozen. 2026-11-03

Jobs 3610–3615 (n = 5…10, commit `6c04e29`, preflight 3609) all completed and verified. Every n fails §9.3.1/9.3.2 and §9.1. Ambiguity also fails at n=5. Base accuracy is 0.001–0.024 and coverage 0.003–0.014. 95–97 % of errors are `I`, and about 98 % of those are a bare integer with no `Answer:` prefix: the model obeys "only the final answer" literally. Ladder step 1 (different n, answer-only prompt) is exhausted. Details: `docs/GATE3.md`, `docs/gate3_census_comparison_p2.json`.

Not done, because each is a protocol choice: accepting a bare-integer completion (taxonomy v2, invalidates all censuses), moving to subtraction (ladder step 2, likely to hit the same format failure), or applying the stop rule. No audit was exported and `q_ref.lock` does not exist. Phase 5 must not start.
