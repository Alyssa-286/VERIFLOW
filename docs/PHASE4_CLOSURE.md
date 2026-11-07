# Phase 4 closure: split-half stability and the locked q_ref

**Decision (2026-11-07, D28): B — CLOSED WITH DOCUMENTED LIMITATION.**

- **0.5B primary arm.** Gate 3′ passed by its declared procedure, and `q_ref.lock`
  stays valid and unchanged for the registered estimand.
- **Stability evidence.** The declared split-half TV statistic gives weak
  evidence of stability at this census size, for either model. The stability of
  q_ref rests on other evidence (§4). That evidence is post hoc, and it must be
  disclosed as such.
- **1.5B arm.** Its declared stability requirement failed (n = 8, 9), and that
  failure stands. The diagnosis attributes it to sampling variability under
  strong prompt clustering. It does not come from a calculation defect or from
  systematic instability. Still, no 1.5B q_ref exists, and no confirmatory
  claim is made about the 1.5B model.

Everything below is an **exploratory, post hoc** analysis of existing census
outputs. No census, gate statistic, threshold, split, lock, configuration or
registered value was changed. No new GPU computation was run.

## 1. Evidence audited

| Artifact | Checked |
|---|---|
| Census runs 3640–3645 (0.5B) and 3647–3652 (1.5B) | copied read-only from `~/VERIFLOW/runs/census/`; `classification.parquet`, `params.json` and `stability.json` read for all 12 |
| Exploratory census 3620 (split 20261102) | `generations.parquet` re-classified under taxonomy v3 for an independent replicate |
| `q_ref.lock` | file SHA-256 `737aab73…757e`, lock sha256 `2c34f6d2…d5cb`, unchanged; the 3640 `q_ref.json` hashes to the locked `f3b607c5…f196`, and its point estimate reproduces the lock to 1e-12 |
| Gate code | `census_core.split_half_tv`, `half_of_prompt` (SHA-256 of `"split-half\|" + prompt_id`, low bit of byte 0), `q_ref_point`; threshold `GATE_TV_MAX = 0.03` (§9.3.4) |
| Audit | already re-validated on 2026-11-07 (D22): 1/200 disagreement, hashes unchanged; not re-done here |
| Grid configs | the 48 configs of D26 (`configs/grid/matched`, `configs/grid/auxiliary`) |
| Earlier censuses (v1, v2, p2t2, subtraction) | the TV passed in all 24 (≤ 0.022, GATE3.md). No earlier TV failure exists |

**Script:** `experiments/diagnose_split_half_tv.py`, seed 20261107, 2,000 draws per
reference distribution.
**Output:** [gate3_tv_diagnosis.json](gate3_tv_diagnosis.json).

## 2. Implementation check: no defect

- The recomputed TV equals the stored `stability.json` value in all 12 runs, to
  within 1e-12.
- The halves are balanced. For example, 3640 splits 1018/982 prompts and
  2671/2540 errors; 3650 splits 1019/981 prompts and 2029/1937 errors.
- The split is by prompt (cluster-respecting), as §9.3.4 specifies. The hash
  depends only on `prompt_id`, so the declared split is one draw from "each
  prompt to a half with probability ½".
- The TV is computed over every incorrect class (I, NEG, LEN, ONE_HI, ONE_LO,
  MULTI), as the gate defines it.

No code fix is justified, so nothing was rerun.

## 3. Diagnosis

### 3.1 The declared TV is one draw from a wide distribution

The table compares the declared TV with the TV over 2,000 random prompt halves
drawn by the same law as the declared split.

| Run | Base acc | Errors (prompts with ≥1) | Declared TV | Random-halves mean [p5, p95] | P(TV > 0.03) | Declared percentile |
|---|---|---|---|---|---|---|
| **0.5B n5 (3640, selected)** | 0.349 | 5211 (1719) | **0.0177** | 0.0287 [0.012, 0.050] | **0.41** | 17 |
| 0.5B n6 (3641) | 0.131 | 6954 | 0.0272 | 0.0227 | 0.23 | 70 |
| 1.5B n7 (3649) | 0.636 | 2914 | 0.0548 | 0.0365 | 0.60 | 86 |
| **1.5B n8 (3650)** | 0.504 | 3966 (1281) | **0.0389** | 0.0313 [0.011, 0.058] | **0.48** | 74 |
| **1.5B n9 (3651)** | 0.488 | 4098 (1352) | **0.0501** | 0.0294 [0.011, 0.055] | **0.43** | 92 |
| 1.5B n10 (3652) | 0.246 | 6029 | 0.0147 | 0.0228 | 0.22 | 24 |

All 12 runs are in the JSON. Across the 12 runs, the declared percentiles are
consistent with uniform (KS p = 0.97). The extremes are 0.5B n8 (TV 0.0028, 0th
percentile) and 1.5B n9 (92nd). The declared statistic behaves exactly
like a random partition: there is no sign of a split artifact.

The consequence is that at 2,000 prompts × 4 samples, **the criterion is close
to a coin flip for both models**:
- Re-randomising the halves would fail the selected 0.5B run 41 % of the time.
- Re-randomising would pass the 1.5B run at n = 8 52 % of the time.
- The chance that 1.5B fails at both in-window n by chance alone is
  0.48 × 0.43 ≈ 0.21.

The 0.5B pass and the 1.5B fail therefore do **not** show that the two models'
q estimates differ in stability. Their random-halves means are 0.029 (0.5B n5),
0.031 (1.5B n8) and 0.029 (1.5B n9).

### 3.2 Why the TV sits at the threshold: prompt clustering, not drift

**Clustering inflates the TV.** With independent errors and the same half
sizes, the expected TV would be 0.021 (0.5B n5) and 0.020 (1.5B n8). Prompt
clustering raises it by a factor of 1.40 (0.5B) and 1.55–1.70 (1.5B). Design
effects per designed class are 2.0–2.4 for 0.5B n5 and 2.5–3.5 for 1.5B.

**The error class is largely a property of the prompt.** The share of
within-prompt error pairs that fall in the same class is 0.62 for 0.5B n5,
against 0.28 without clustering. For 1.5B it is 0.75 at n8 (against 0.38) and
0.74 at n9 (against 0.40).

**The heterogeneity is predictable from the operands.** q differs between
prompts whose sum has n digits and those whose sum has n + 1 digits. The TV
between these two strata is 0.19 (0.5B n5), 0.21 (1.5B n8) and 0.11 (1.5B n9).
For example, in 1.5B n8, ONE_HI is 0.185 when the sum has n digits and 0.041
when it carries out to n + 1.

**There is no drift over the census.** Splitting by census order (first versus
last 1,000 prompts) gives TVs at percentiles 10 (0.5B n5), 90 (1.5B n8) and
61 (1.5B n9) of the random-halves distribution, so there is no generation-batch
effect.

The **1.5B model is more accurate**, so it has fewer errors (3966 against 5211)
spread over fewer prompts (1281 against 1719), with stronger clustering. These
three facts together explain its slightly wider TV distribution.

### 3.3 The large designed modes drive the TV, not rare classes

| Run | Contributions ½\|q₀ − q₁\| (largest first) |
|---|---|
| 1.5B n8 | MULTI 0.0141, ONE_HI 0.0123, LEN 0.0072, I 0.0028, ONE_LO 0.0026 |
| 1.5B n9 | ONE_LO 0.0246, ONE_HI 0.0131, LEN 0.0063, MULTI 0.0057, I 0.0005 |
| 0.5B n5 | ONE_LO 0.0071, LEN 0.0056, ONE_HI 0.0021, I 0.0017, MULTI 0.0012 |

I and NEG, the rare classes, contribute ≤ 0.003. The discrepancy sits in the
modes that carry 10–50 % of the mass.

### 3.4 Answers to the six diagnostic questions

1. **What caused the 1.5B failure?** The declared hash split gave TV 0.039
   (n8) and 0.050 (n9), at the 74th and 92nd percentiles of a random-partition
   distribution whose mean (0.029–0.031) already sits at the 0.03 threshold.
2. **Genuine instability, or an artifact?** Neither an implementation,
   partition nor calculation defect (§2), nor systematic instability (uniform
   percentiles, no order drift). It is sampling variability of a half-sample
   statistic under strong, operand-driven prompt heterogeneity. That
   heterogeneity is a real property of the error distribution. It is not
   instability of the population q.
3. **Which modes contribute?** MULTI and ONE_HI at n8; ONE_LO and ONE_HI at
   n9 (§3.3).
4. **Is it concentrated in rare categories?** No.
5. **Do the existing outputs separate sampling variability from systematic
   instability?** Yes, for the question of whether the declared split is
   anomalous: it is not. They cannot certify that the 1.5B q meets the declared
   criterion, because the statistic is too noisy at this size to discriminate.
6. **Is the locked q_ref valid for the confirmatory estimand?** Yes, within
   the bounds in §4–5.

## 4. Is the locked q_ref adequate for the confirmatory estimand?

The registered estimand is the association of V and f_max with GRPO outcomes
for Qwen2.5-0.5B-Instruct on 5-digit addition, with q_ref treated as a fixed
design input. The configs are generated from the locked values, so V and f_max
are defined design quantities. The relevant question is how far the true base
q could move the configs away from their design values.

- **Independent replicate.** Census 3620 used a different split (20261102), the
  same model and prompt, and was re-classified under v3. Its TV against the
  locked 3640 is **0.0124**, below the threshold. It is also below its own
  sampling expectation, about 0.029/√2 ≈ 0.020 for two full-size censuses.
  Caveat: 3620 is the exploratory census the partition was chosen on (D15).
  The partition is deterministic and fits no parameters, so the q estimate
  is still an independent sample.
- **Cluster-bootstrap SD of the locked q_ref** (prompts resampled): LEN 0.007,
  MULTI 0.010, ONE_HI 0.008, ONE_LO 0.008, I 0.003.
- **F̄ matching against the true base distribution** (target 0.3):
  - per-config 95 % intervals deviate by at most 0.018 (median 0.010);
  - the 95th percentile of the worst deviation across all 48 configs is 0.020.
- **V, the regressor of interest:**
  - the 95th percentile of the worst absolute deviation is 0.008, which is
    4.7 % of the matched V range (0.172);
  - the Spearman rank correlation of the 36 matched configs' V with the
    design V is ≥ 0.989 in every bootstrap draw (median 0.997);
  - f_max is a config parameter and is unaffected.

These uncertainties are small next to the design contrasts. In the Gate 5
pilot, the realised false-acceptance rate F was measured directly from the
verifier logs, and the grid does the same. Any F̄ mismatch is therefore
observed, not assumed.

## 5. Supported and unsupported claims

**Supported:**
- the association of V (with f_max controlled) with held-out accuracy, for
  Qwen2.5-0.5B-Instruct, 5-digit addition, prompt v2 and taxonomy v3;
- V and F̄ are design values computed from the locked q_ref, which is
  replicated on an independent split (TV 0.012) with bootstrap SD ≤ 0.010 per
  mode.

**Not supported:**
- any confirmatory statement about Qwen2.5-1.5B, other scales, other n or
  other tasks;
- describing the Gate 3′ split-half TV pass as strong evidence of stability.
  It must be reported with its re-randomisation failure probability (0.41);
- claims that rely on exact F̄ equality at the true base distribution, beyond
  about ±0.02.

**Exploratory only:** any 1.5B result, and everything in this document.

## 6. Is more computation needed?

**No, for the registered design.** The primary arm needs no new data. The
diagnosis used only existing outputs.

No 1.5B census is run, because re-running the same TV gate until it passes is
exactly what the integrity rules forbid. If a 1.5B arm is ever wanted, it needs
a new amendment logged before data. The minimum is one fresh-split 1.5B census
at the selected n, judged by a criterion that does not depend on the luck of
one split. One such criterion is the **expected** random-halves TV ≤ 0.03,
together with an independent-replicate TV ≤ the 95th percentile of its own
sampling distribution. For 1.5B n8, the existing data put that requirement at
roughly 4,300 errors (about 2,200 prompts × 4) at the current clustering. That
figure is a planning number, not a result.

## 7. Disclosure for the registration (researcher action)

`PREREGISTRATION.md` is frozen at tag `prereg-v1` (`8ff68a3`) and is not edited
here. This analysis was done after the freeze and before any confirmatory run.
If the OSF registration is not yet posted, add the following text. If it is
already posted, add it as a dated OSF addendum:

> Post hoc (2026-11-07, before any confirmatory run): the split-half TV
> criterion (§9.3.4) is noisy at the census size used. Under random prompt
> halves, the selected 0.5B census would exceed 0.03 with probability 0.41,
> and the failing 1.5B censuses lie at the 74th and 92nd percentiles of their
> own random-split distributions. The locked q_ref is supported by an
> independent-split replicate (TV 0.012) and by bootstrap propagation: V ranks
> are preserved (Spearman ≥ 0.989), and F̄ is matched to within ±0.02. No
> registered value changes. Details: docs/PHASE4_CLOSURE.md.
