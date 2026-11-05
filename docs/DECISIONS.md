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

## D11: Taxonomy v2: a bare-integer completion is parsed as the answer. 2026-11-03

Approved by the researcher after D10. **Chosen after seeing the prompt-v2 outputs** (about 98 % of v2 `I` errors were a bare integer). This is a post hoc parser decision and must be disclosed as such. It is fixed once and is not tuned further.

- Rule: with no `Answer:` line, a completion that is exactly one integer (same integer grammar as before; surrounding whitespace, `*`, backticks, `$`, `\boxed{}` and one trailing period stripped) is the answer, with parse status `ok_bare`. Prose ("The answer is X"), several numbers, working, or an empty completion stay `I`.
- Unchanged: classes, priority, every error-mode rule, multi-hot ambiguity, gate thresholds, prompt v2, max_new_tokens 256, sampling, split and seed.
- `TAXONOMY_VERSION = v2`; `taxonomy.lock` payload sha256 `f4bf0f24…eb575d`. All v1-taxonomy censuses are invalid for selection and are kept as evidence. Synthetic recovery (§9.3.5) was rerun: green.
- Fresh GPU census runs: `n<N>_p2t2_s20261102_j<job>`.

## D12: Phase 4 complete; Gate 3 FAILS; q_ref NOT frozen. 2026-11-03

Final configuration: prompt v2 + taxonomy v2, jobs 3620–3625 (commit `47fa0d2`, preflight 3618). All six runs were verified, and none passes. The only n inside the §9.1 window is n = 5 (acc 0.344). It has coverage 0.416, 2 designed modes ≥ 5 % (DIGIT 0.295, CARRY 0.120) and ambiguity 0.235, with U = 0.552. U rises to 0.965 at n = 10. ECHO/TRUNC/TRANS are ≤ 0.002 everywhere. Split-half TV ≤ 0.022, so the estimates are stable and the failure is real.

**Gate 3 verdict: FAIL.** No n is selected, `q_ref.lock` is not written and no audit was exported. Phase 5 (GRPO) must not start on this design.

Ladder position: step 1 is exhausted. Step 3 (M = 3) is unreachable because at most 2 designed modes reach 5 %. For addition on Qwen2.5-0.5B-Instruct, the §9.3 stop condition (fewer than 3 designed modes at 5 % mass) holds at every n. The one remaining step before the stop rule is 2-operand subtraction (step 2). It requires a subtraction taxonomy (borrow-drop rule, sign handling, synthetic-recovery tests) that the master plan does not define, so it is a new design decision, not a continuation of Phase 4. Options for the researcher: (a) define and run the subtraction step; (b) apply the stop rule and revisit the task, model (e.g. 1.5B replication) or design; (c) amend §9.3, which would be a protocol change requiring disclosure.

Measured cost: 18 censuses plus 3 preflights plus 2 GPU checks, about 0.2 GPU-h total on one L40.

## D13: Ladder step 2: 2-operand subtraction, locked before any subtraction data. 2026-11-03

Approved by the researcher after D12.

- Task: `a - b` with both operands n-digit and a > b (pairs with a = b dropped). Splits come from the seed namespace `sub|<split>` with the same sizes and disjointness rule. Addition splits are bit-identical (tested).
- Prompt: the v2 wording with `-` (`prompt_sub.lock` payload `99b4bc16…e0b8e`), max_new_tokens 256, taxonomy-v2 parser (bare integers accepted).
- Taxonomy `sub-v1` (`taxonomy_sub.lock` payload `a02392c2…52a16`): C, I, ECHO, **BORROW** (a non-empty subset of borrows dropped: the column still adds 10, the next column is not decremented; enumerated over the borrow tree), TRUNC, TRANS, DIGIT, U. It is a strict mirror of the addition taxonomy.
- U-only diagnostics, never designed modes and never in a gate statistic: OPSWAP (a+b), NEG (b−a), SFL (smaller-from-larger per column).
- Synthetic recovery ≥ 99 % for every designed mode at n = 5, 8 (`tests/test_subtraction.py`).
- Unchanged: sampling, 2,000 × 4 per n, n = 5…10, every §9.3 threshold and the §9.1 selection rule. Run ids: `n<N>_sub_p2t2_s20261102_j<job>`.
- Known risk, predeclared: a single dropped borrow is also a one-digit slip, so BORROW∩DIGIT ambiguity is expected (as CARRY∩DIGIT was).

## D14: Gate 3 verdict: STOP (§9.3 kill rule). Phase 4 closed. 2026-11-03

Subtraction census jobs 3627–3632 (commit `36de395`, preflight 3626) all completed and verified. Every n fails, with base accuracy 0.013–0.150, coverage 0.05–0.38, at most 2 designed modes ≥ 5 % (DIGIT, BORROW), ambiguity 0.13–0.17 and U 0.55–0.90. About 99.6 % of U errors match none of the OPSWAP/NEG/SFL diagnostics.

Across all 24 real censuses (addition ×3 configurations, subtraction ×1; n = 5…10) no n has 3 designed modes at ≥ 5 % mass. Ladder step 3 (M = 3) is unreachable. The §9.3 stop condition holds: **the matched-heterogeneity design cannot be built on Qwen2.5-0.5B-Instruct with two-operand integer addition/subtraction under the predeclared taxonomy.** This is the plan's kill rule (§27), not a tooling failure: split-half TV ≤ 0.022 everywhere, so the q estimates are stable.

Consequences: no `q_ref.lock`, no manual audit, Phase 5 (GRPO) does not start on this design. Any continuation is a new pre-registered design, e.g. a different base model (the pinned 1.5B replication model), a different task, or a revised §9.3. It must be logged as a protocol amendment before any data are collected.

Total compute for Phase 4: 24 censuses, 6 preflights and 2 GPU checks, about 0.3 GPU-h on one L40.

## D15: Amendment 1: taxonomy v3 (outcome-based partition) and fresh split seed 20261104. 2026-11-03

Full text: `docs/AMENDMENT_1.md`, logged before any v3 data. The taxonomy is an exhaustive, disjoint partition of wrong answers: I, NEG, LEN, ONE_HI, ONE_LO, MULTI. It was chosen after the exploratory censuses (disclosed). The confirmatory census uses a new split seed, 20261104. Thresholds, selection rule, prompt v2, parser v2 and sampling are unchanged. Stop rule: if Gate 3′ fails, the GPU arm ends (Track B).

## D16: Gate 3′ automated criteria PASS at n = 5; audit pending. 2026-11-04

Fresh-split census (Amendment 1, seed 20261104), Qwen2.5-0.5B, jobs 3640–3645, commit `ffa27a7`. Selected run `n5_v3_s20261104_j3640`, with base accuracy 0.349 and q_ref:
- LEN 0.162, MULTI 0.420, ONE_HI 0.181, ONE_LO 0.206;
- NEG 0.000, merged into LEN per Amendment 1;
- I 0.030.

Coverage is 0.970, ambiguity 0 and split-half TV 0.0177. The q_ref reproduces the exploratory estimate. The blinded 200-row audit was exported (`audit/gate3v3_n5_s20261104_j3640/`, key kept on the cluster), and `q_ref.lock` will be frozen only after the researcher's judged audit shows ≤ 5 % disagreement.

The 1.5B secondary census (jobs 3647–3652) fails split-half TV within the accuracy window (n = 8: 0.039; n = 9: 0.050). The replication arm's design (larger census or descriptive-only) is decided before Stage 5.

Report jobs 3646/3653 crashed formatting the Markdown table (taxonomy v3 has no U) after writing the selection and audit. The formatter was fixed (`508fefa`) and the tables were regenerated; the audit files were verified byte-identical.

## D17: Phase 5 training stack fixed; Gate 4 PASS. 2026-11-04

TRL 1.17.1 is pinned (exact). The backend is HF generate, with no vLLM: NVML is broken on sit-gpu and vLLM conflicts with torch < 2.9. All GRPO settings are pinned in `configs/base.json` (dapo, β 0, group scaling, G 8, 32 prompts/step, neutral sampling asserted at run start, fp32 master weights + bf16 autocast, AdamW, constant LR). Smoke job 3667 passed all 17 checks (resume, exact verifier replay, realised = configured rates). Bench job 3669: 1.09 s/step at G = 8, about 0.16 GPU-h per 500-step run, standard 64-run grid ≈ 10 GPU-h. Details: `docs/GATE4.md`. Open before Stage 3: F̄*/τ* values, q_ref.lock (audit), and the 1.5B arm design.

## D18: F̄* = 0.3 and τ* = 0.9 fixed for the pilot and grid. 2026-11-04

Chosen by the researcher before any pilot data, from Claude's proposal. These are the Gate 2 simulation values (GATE2.md), so the simulation evidence on V vs f_max applies to the same operating point. The matched arm uses F̄ = F̄* and τ = τ* for every configuration. The auxiliary arm uses F̄ ∈ {0.15, 0.45} (0.5× and 1.5× F̄*, §14). Recorded in `configs/base.json`.

## D19: Pilot rules pre-declared (Stage 3). 2026-11-04

Logged before any pilot data. Implemented in `experiments/make_pilot_configs.py` and `experiments/pilot_select.py`, tested in `tests/test_pilot.py`.

**LR pilot.** Homogeneous verifier (minimum dispersion at F̄* 0.3, τ* 0.9) × lr ∈ {5e-7, 1e-6, 3e-6}, plus a perfect verifier (τ 1, every f 0) at lr 1e-6, the middle LR (§13 leaves its LR unspecified). Seed 1, 800 steps, eval every 50.

**LR choice** (§13 says "choose the LR" without a rule): pick the highest mean greedy eval accuracy over steps 400–800. A tie within 0.005 goes to the smaller LR. A run below 0.01 accuracy for 3 consecutive evals is ineligible.

**Horizon T.** The §13 rule (90 % of the maximum gain, rounded up to a multiple of 50) with a floor of 100, so the window [T/2, T] holds at least two eval points.

**Heterogeneity pilot.** 4 configurations × seeds {1, 2, 3} at the chosen lr and T:
- V floor;
- V at 50 % and 90 % of the feasible range (f_max free);
- f_max = 0.95 at the minimum feasible V for the mode with the lowest such V.

Generator seed 20261106. Pilot data are exploratory and excluded from the confirmatory set (§13).

## D20: Gate 3′ audit performed by Claude, not by a person; q_ref.lock frozen. 2026-11-05

**Provenance, stated plainly.** §9.3.6 calls for a manual audit by a person. The researcher did not fill it in. On 2026-11-05 they instructed Claude (Opus 5.5) to "audit on behalf of me and continue". All 200 judgments in `audit_completed.csv` were made by Claude. **The researcher has not reviewed them.** It was not a human audit, and it was not human-reviewed. Any manuscript or pre-registration must describe it this way.

**Method.** Claude worked from `audit_instructions.md` and `audit_sample.csv` (SHA-256 `e76ac745…4403`) only. It did not open `audit_key.csv` (`7cedc4a9…08eb4`), `classification.parquet` or `gate3.json` before validation. 188 rows are bare integers and were labelled by a rule implementation written from the instructions (`audit/gate3v3_n5_s20261104_j3640/tools/ai_label.py`). Claude judged the 12 other rows individually, recorded a reason for each in the script, and cross-checked them against the rules. AUD-0003 (`1,982,300`) was flagged as an interpretive call before validation: it was labelled I under a strict reading of "nothing but one integer", where a lenient reading would give LEN.

**Limitation.** The rule implementation is a second implementation of the same written rules the classifier follows. The audit is therefore close to a differential test of the classifier code. It does not check whether the rules match a human reader's judgment the way §9.3.6 intends.

**Result.** `validate_audit()` found 1 of 200 rows in disagreement (0.005, threshold 0.05). The disagreeing row is AUD-0003, and its judgment was not changed after validation. The automated criteria were recomputed from the census artifacts: coverage 0.970, mass 4 modes, ambiguity 0, split-half TV 0.0177, base accuracy 0.349. All pass.

**Lock.** `configs/locked/q_ref.lock` was frozen on the cluster at 2026-11-05T12:22:34 by `freeze_qref()` called directly. `main()` was not used because it rewrites the audit sample and key. The sample and key hashes were the same before and after. Lock sha256 `2c34f6d2…d5cb` (file SHA-256 `737aab73…076f8f`), run `n5_v3_s20261104_j3640`, Slurm job 3640, Qwen2.5-0.5B-Instruct revision `7ae55760…a775` (the pinned revision), taxonomy-v3 lock `6b1ebdf9…8069`, prompt lock `918a83d3…92a8`, q_ref.json `f3b607c5…f196`.

**Open for the researcher.** The researcher may re-do or review the audit themself before the pre-registration is posted. If a human audit changes the pass/fail outcome, the lock and every configuration generated from it are invalidated.
