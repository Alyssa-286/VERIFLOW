# VERIFLOW phase ledger (final, 2026-11-14)

Each row states the exit criterion, the verdict, and the evidence a reader can check. Decisions are in [DECISIONS.md](DECISIONS.md).

| Phase | Exit criterion | Verdict | Evidence |
|---|---|---|---|
| 0 Planning and literature | Gate 0: the modified question is new | **Done**; Gate 0 not reopened (D30). Forward-citation search rate-limited (disclosed) | `docs/RELATED_WORK.md`, `paper/references.bib` (every entry verified against arXiv) |
| 1 Theory | Gate 1: maths verified numerically | **PASS** 2026-11-01 | `docs/GATE1.md`, `tests/test_sim_replicator.py`, `tests/test_stats.py` |
| 2 Simulation | Gate 2: V adds held-out information beyond f_max in shared-parameter sims | **PASS** 2026-11-01 (27/27 cells) | `docs/GATE2.md`, `figures/out/fig_gate2.*` |
| 3 Construct tooling | tests green, synthetic recovery ≥ 99 % | **Done** (D6) | `tests/` |
| 4 Cluster and census | Gate 3: construct criteria | v1 mechanism taxonomy **STOP** (D14) → Amendment 1 → Gate 3′ **PASS** at n = 5 (D16), audit 1/200 (D20, D22), `q_ref.lock` frozen. **CLOSED WITH DOCUMENTED LIMITATION** (D28, verified D29): split-half TV is noisy at this census size (P(TV > 0.03) = 0.41 for the selected census); 1.5B failure stands; q_ref adequate for the 0.5B estimand only | `docs/GATE3.md`, `docs/PHASE4_CLOSURE.md`, `docs/gate3_tv_diagnosis.json`, `docs/PREREG_ADDENDUM_1.md` |
| 5 Training stack | Gate 4: feasibility | **PASS** 2026-11-04 (D17) | `docs/GATE4.md` |
| 6 Pilot | Gate 5: manipulation reaches the policy | **PASS** 2026-11-07 (D25); power → R = 36, estimation-only | `configs/pilot/{gate5,power}.json`, `results/pilot/validate_het.json` |
| 7 Registration and grid | registration recorded; all runs complete and valid | **Done** 2026-11-14 (D31, D32): OSF bs3wq public, approved, DOI 10.17605/OSF.IO/BS3WQ; 24/24 hash commitments verified; array 4036 **96/96 valid**, one commit, grid digest = registered; no rerun, no exclusion | `results/grid/validate_grid.json`, `results/grid/run_records/INDEX.json`, `docs/PROVENANCE.md` |
| 8 Analysis | registered analysis run on real outputs | **Done** (D33): H1 b_V = −6.01 [−7.59, −4.43]; H2 −0.023 [−0.047, −0.002]; H3 +0.002 [−0.007, 0.011]; Holm p reported, not interpreted as confirmatory. Duplicated configurations disclosed (25 distinct of 36 matched); deduplicated and other post hoc analyses labelled | `results/grid/analysis.json`, `results/grid/supplement.json`, `figures/out/fig_grid_*` |
| 8 Paper | complete, numbers generated from result files | **Done**: 14 pages, all sections, every result number a generated macro | `paper/main.pdf`, `paper/main.tex`, `paper/sections/`, `paper/make_numbers.py` |
| 8 Reproducibility and release | clean checkout reproduces analysis and paper; tests pass; licence; public release | **Done** (D34): 184 tests pass; clean-clone check of analysis, figures and PDF; MIT (code) + CC BY 4.0 (docs, data); public repository with redacted history, original history kept private; release v1.0.0 with verifier-log archive | `docs/REPRODUCE.md`, `LICENSE`, `LICENSES.md`, `docs/PROVENANCE.md`, GitHub release v1.0.0 |

## Answer to the research question

In this setting (Qwen2.5-0.5B-Instruct, five-digit addition, 100 GRPO steps), error-mode heterogeneity predicted RLVR outcome beyond aggregate verifier quality: at exactly matched aggregate rates, higher dispersion V went with lower accuracy, and V predicted outcomes better than the aggregate or worst-mode rate. The registered held-out test is weakened by duplicated configurations, the mechanism analyses are exploratory, and nothing generalises beyond this model, task and horizon.

## Known limitations carried into the paper

Low pre-specified power (0.43; estimation-only claim); two seeds per configuration; duplicated configurations in the registered grid; census stability criterion noisy (Phase 4 limitation); failed 1.5B stability test, no 1.5B claim; three-seed pilot is descriptive; AI-labelled construct audit reviewed by the researcher; incomplete forward-citation coverage.

## Cluster hygiene

ErrorEcho stayed off the cluster for the whole of this work (no job, process or crontab; recovery snapshot of 2026-11-07 untouched). No other user's job was touched. Grid checkpoints (≈ 1.1 TB) remain on the cluster.
