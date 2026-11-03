# Amendment 1: outcome-based error partition (taxonomy v3) and a fresh confirmatory census

Logged 2026-11-03 as DECISIONS.md D15, **before any data under this amendment
exist**. It amends master plan §9.2 (taxonomy) and §11 (census split). Every
other part of the plan is unchanged.

## Why

Gate 3 stopped under the v1/v2 *mechanism* taxonomy (D14). 55–97 % of wrong
answers matched no named mechanism, for both addition and subtraction. The
research question (§2) needs a partition of wrong answers into modes that a
programmatic verifier can accept at different rates. It does not need the
modes to be mechanisms. The amendment therefore replaces the mechanism
taxonomy with a deterministic partition defined by how the answer differs
from the truth.

## Disclosure

The partition was chosen **after** seeing the exploratory censuses
(D8–D14). An analysis of job 3620 (addition, n = 5) motivated it. All 24
earlier censuses are labelled **exploratory**. The confirmatory census below
uses a split whose operands have never been generated, so its outcome is a
genuine test. The paper must report this amendment and its timing.

## Taxonomy v3 (addition)

Inputs are the operands a, b, the true sum S, and the answer v parsed by the
taxonomy-v2 parser (D11). L = number of digits of S. Classes are disjoint and
exhaustive:

| Code | Rule |
|---|---|
| `C` | v = S |
| `I` | no parseable answer (taxonomy-v2 parser) |
| `NEG` | v < 0 |
| `LEN` | v ≥ 0 and the digit count of v ≠ L |
| `ONE_HI` | same digit count, exactly one digit differs, at position i < ⌊L/2⌋ counted from the most significant digit (0-based) |
| `ONE_LO` | same digit count, exactly one digit differs, at position i ≥ ⌊L/2⌋ |
| `MULTI` | same digit count, two or more digits differ |

- **Designed modes:** NEG, LEN, ONE_HI, ONE_LO, MULTI.
- **Not designed:** `I`. There is no `U`; the partition is exhaustive.
- **Coverage and ambiguity:** coverage is therefore 1 − q(I), and ambiguity is
  0 by construction. Both are still computed and reported.
- **Secondary annotation:** the mechanism labels (ECHO, CARRY, TRUNC, TRANS,
  DIGIT, U) are recorded per answer for description only. They are never used
  in a gate statistic or a verifier configuration.
- **Small modes (§9.3.2 has no `U` to merge into here):** before
  configurations are generated, NEG merges into LEN if q(NEG) < 0.02 (both
  mean "wrong shape"). Any other designed mode below 0.02 merges into MULTI.
  A merged mode does not count towards the ≥ 3-modes criterion. NEG is
  expected to be near 0.

## Unchanged from the plan and from D9/D11

Everything below is as before:
- task: two-operand addition;
- prompt v2 (`prompt.lock`) and max_new_tokens 256;
- sampling (T 1.0, top_p 1.0, top_k 0, repetition_penalty 1.0);
- 2,000 prompts × 4 samples per n, n = 5…10;
- the §9.1 selection rule;
- every §9.3 threshold;
- the 200-row blinded manual audit (≤ 5 % disagreement);
- the 2,000-resample cluster bootstrap.

## Fresh confirmatory split

- `base_seed = seed = 20261104`, used for **all** splits from now on (census,
  train, eval, eval-mode). The 20261102 splits are retired.
- Run ids: `n<N>_v3_s20261104_j<job>`.

## Gate 3′ (pass and stop rules, fixed now)

1. Choose n by §9.1: the smallest n with base accuracy in [0.25, 0.60] whose
   automated §9.3 gate passes. Under v3 the binding criteria are mass
   (≥ 3 designed modes at ≥ 0.05) and split-half TV ≤ 0.03.
2. Synthetic recovery (§9.3.5) ≥ 99 % for every designed v3 mode.
3. The researcher's blinded audit of 200 errors from the selected run shows
   ≤ 5 % disagreement. Only then is `q_ref.lock` frozen.
4. **Stop:** if no n passes, the GPU arm ends and the project continues on
   Track B (COMPLETION_PLAN.md). No further taxonomy is tried.

## Secondary (not used for selection)

The same census is run with the pinned replication model Qwen2.5-1.5B-Instruct
(`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) to check whether the replication
arm is feasible. Run ids: `n<N>_v3q15_s20261104_j<job>`.
