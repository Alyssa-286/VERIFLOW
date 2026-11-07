# Related work and novelty check (Phase 0 completion)

**Date:** 2026-11-07. This completes the sweep that D0 deferred. Every work listed was verified against the arXiv API (id, title, authors, first-version date); BibTeX entries are in `paper/references.bib`. Venues are not claimed unless verified.

## Method and limits of this sweep

- **arXiv API:** the known ids, plus six full-text searches sorted by date (queries below), covering 2023–2026-11.
- **Full texts read:** abstracts of all candidates; full text of Plesner et al. (closest by design).
- **Semantic Scholar:** the citation-graph queries ("who cites Egashira et al. and Rad et al.") returned HTTP 429 (rate limit) on every attempt on 2026-11-07, so forward-citation coverage is incomplete. Google Scholar was not used, because no automated access was available. Work that is not on arXiv, or that is worded differently from the queries, may have been missed. The novelty statement below is scoped accordingly.

**Queries (arXiv `all:`, 25 newest each):**
- "noisy rewards" AND verifier AND reinforcement
- "false positive" AND verifier AND RLVR
- "reward noise" AND GRPO
- verifier AND "error mode" AND reinforcement
- "imperfect verifier"
- "reward hacking" AND verifier AND RLVR

## Closest works

| Work | What it does | Relation to VERIFLOW | Overlap |
|---|---|---|---|
| Egashira, Vero, Dekoninck, Dorner, Staab, Vechev 2026, *Delay, Plateau, or Collapse* (2605.02909) | Controlled arithmetic RLVR. Systematic false positives cause plateaus or collapse, and "outcomes are not determined by the overall error rate but by the specific pattern of introduced errors". | Establishes that the rate is not sufficient. It uses discrete hand-built FP patterns. It does not match base-weighted F̄ and τ across a continuous dispersion grid, and does not test held-out prediction by a dispersion statistic. | **High** (motivation). VERIFLOW takes its qualitative finding as given. |
| Rad, Filom, Keivan, Mohajerin Esfahani, Kamalinejad 2026, *Rate or Fate? RLVεR* (2601.04411) | Bandit/replicator theory for GRPO with FP and FN. The drift of the incorrect mass depends on Youden's J alone, with a phase transition at J = 0. | VERIFLOW generalises its block-symmetric error model to mode-specific FP rates. Under block symmetry, V = 0 by construction. | **High** (theory baseline) |
| Plesner, Guzmán, Athalye 2026, *An Imperfect Verifier is Good Enough* (2604.07666) | Qwen3-8B on MBPP. Small losses up to 20–30 % noise, and four controlled noise structures at equal rates. These are correlation patterns over the rollout × test matrix (cell, row, column, whole). | Its structures are correlation structures with equal FP and FN rates, resampled each encounter. Per-error-mode FP rates are not varied, and the authors state they do not model persistent blind spots. | **Medium** |
| Gupta, Joshi 2026, *Checking the Checkers* (2610.17824) | Survey of 58 studies: "the structure of an error matters more than its rate". | Field-level consensus that VERIFLOW must not claim as new. | Context |
| Moya, Thornley, Lin 2026, *Verifier Errors in RLVR* (2609.35677) | Gradient-flow theory of reward hacking with a fixed verifier; correction with audits. | One fixed verifier and hack set. No dispersion manipulation. | Low |
| Zhang 2026, *When the Reward Suite Is Leaky* (2607.11022) | Pre-registered contrast of natural FPs (MBPP vs MBPP+). The FP mechanism is consistent with selection of pre-existing error modes. | Natural rather than controlled per-mode FPs. Supports the selection mechanism VERIFLOW models. | Low–medium |
| Xin 2026, *Where the Verifier Fails* (2609.01354) | Category-level verifier audit: similar aggregate error, opposite failure reasons. | Measurement only, no RL. | Low |
| Ray 2026, *Before the Model Learns the Bug* (2606.01066) | Fuzzing RLVR verifiers; per-bug-class FPR and FNR. | Measured rather than controlled; no matched design. | Low |
| Cai et al. 2025 (2510.00915); El Mansouri et al. 2025 (2510.18924) | Instance-independent asymmetric noise; noise-corrected GRPO. | Class-conditional, but mode-symmetric. | Low |
| Helff et al. 2026 (2604.15149); Wang et al. 2026, *CATCH* (2609.39533) | Reward hacking via verifier loopholes; a controllable hacking testbed. | A single exploitable FP class or loophole. | Low |
| Xin 2026 (2609.06386); Kim et al. 2025 (2506.07962) | Within-group and cross-model correlation of verifier errors. | Motivates the per-sample PRF verifier design. | Context |
| Shao et al. 2025, *Spurious Rewards* (2506.10947); Rahman et al. 2026 (2604.18574) | Qwen-specific priors can dominate reward signals; weak-supervision regimes are governed by saturation. | Caution for Qwen-based conclusions and for fast saturation (seen in our pilot). | Context |
| Huang et al. 2025 (2505.22203); Wang et al. 2025, *CURE* (2506.03136) | Real verifier FP modes in maths and code. | Shows that per-mode FP heterogeneity exists in practice. | Context |
| Shao et al. 2024, *DeepSeekMath* (2402.03300); Yu et al. 2025, *DAPO* (2503.14476); Liu et al. 2025, *Dr. GRPO* (2503.20783); Schulman et al. 2017, *PPO* (1707.06347); Qwen Team 2024 (2412.15115) | Methods and model. | GRPO, the DAPO loss, the std-scaling caveat, the base model. | Methods |

## Verdict on Gate 0

**Gate 0 is not reopened.** No work found holds the base-weighted aggregate false-acceptance rate and the true-positive rate exactly fixed while varying the dispersion of per-error-mode false-acceptance rates on a designed grid, and then evaluates whether a dispersion statistic predicts RLVR outcomes on held-out verifier configurations beyond the worst-mode rate. The closest designs either vary discrete FP patterns (Egashira et al.) or vary correlation structure at equal rates (Plesner et al.).

## What VERIFLOW contributes, and what it does not

**It contributes:**
1. an exactly matched design that isolates per-mode dispersion V from F̄, τ and f_max, with theory (Props 1–7) that reduces to the Rad et al. J-dynamics when V = 0;
2. a pre-registered estimate, with confidence intervals, of the V–outcome association in one small, controlled RLVR setting (Qwen2.5-0.5B-Instruct, 5-digit addition), with held-out-configuration prediction;
3. transparent negative and limiting results: the mechanism-taxonomy stop, the noisy census stability criterion, and an underpowered confirmatory test downgraded to estimation.

**It does not show:**
- that "structure matters more than rate" (already established);
- anything about larger models, other tasks or natural verifiers;
- a significance-tested confirmatory effect.
