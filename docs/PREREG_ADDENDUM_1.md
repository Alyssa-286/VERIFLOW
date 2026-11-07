# Pre-registration addendum 1 (post-freeze disclosure)

**Date:** 2026-11-07. **Status:** written after the pre-registration was frozen at tag `prereg-v1` (commit `8ff68a3`) and **before any confirmatory run**. It changes no registered hypothesis, estimand, design value, sample size, outcome definition or analysis. `PREREGISTRATION.md` is not edited. This addendum is submitted with the OSF registration as a separate, dated document.

## A1. Phase 4 split-half stability: closed with a documented limitation (D28)

A post hoc, exploratory diagnosis (`docs/PHASE4_CLOSURE.md`; script `experiments/diagnose_split_half_tv.py`; output `docs/gate3_tv_diagnosis.json`) found the following.

- **The split-half TV criterion (§9.3.4) is noisy at the census size used** (2,000 prompts × 4 samples).
  - Under random prompt halves drawn by the declared split's law, the selected 0.5B census (job 3640, declared TV 0.0177) would exceed 0.03 with probability 0.41.
  - The failing 1.5B censuses (n = 8: 0.039; n = 9: 0.050) lie at the 74th and 92nd percentiles of their own random-split distributions.
  - The Gate 3′ TV pass is therefore weak evidence of stability on its own.
- **No implementation defect.** The recomputed TV equals the stored value in 12/12 runs.
- **The 1.5B declared failure stands.** No 1.5B q_ref exists, and no claim is made about the 1.5B model, other scales or other tasks.
- **The locked q_ref is supported for the registered estimand** (Qwen2.5-0.5B-Instruct, 5-digit addition, prompt v2, taxonomy v3), by evidence not used in the gate:
  - an independent-split replicate (census 3620, re-classified under v3) has TV 0.012 against the lock;
  - the cluster-bootstrap SD of q_ref is ≤ 0.010 per mode;
  - propagated to the 48 registered configurations, F̄ stays within ±0.02 of 0.3 (95th percentile of the worst configuration), and V moves by ≤ 4.8 % of its range, with matched-arm V ranks preserved (Spearman ≥ 0.989 in every bootstrap draw).
- **Reporting commitment.** The paper will report the Gate 3′ TV pass together with its 0.41 re-randomisation failure probability, and will treat F̄ equality at the true base distribution as holding only to about ±0.02. The realised false-acceptance rate is measured directly from each run's verifier log and reported.

## A2. Independent verification of A1 (D29)

The diagnosis was re-run independently on the SIT cluster from the original census outputs, read-only.
- Deterministic quantities match exactly: the declared TVs, and the recomputed TV equalling the stored value in 12/12 runs.
- Monte Carlo quantities match within simulation noise: selected-run failure probability 0.41 vs 0.41; replicate TV 0.0124 vs 0.0124; F̄ 95th-percentile deviation 0.0202 vs 0.0203; V-rank Spearman minimum 0.9899 vs 0.9892.

No new computation beyond this read-only check was run. The only remaining difference comes from the order in which runs consume the seeded random stream.
