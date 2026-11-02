# Gate 3 — construct validity (Phase 4 exit)

**Status: ⇨ awaiting the real census run.** All Phase 4 tooling exists and is
CPU-tested (see tests). The gate itself is decided by `experiments/check_construct_gate.py`
on a real (`--backend hf`) census output.

## What runs the gate

```bash
python -m experiments.check_construct_gate --census runs/census/n<chosen>_s20261102 \
    --json runs/census/n<chosen>_s20261102/gate3.json --audit-sample 200 --freeze-qref
```

`--freeze-qref` writes `configs/locked/q_ref.lock` **only if all automated
criteria pass and the census is not a mock run** (the checker refuses mock
data). Manual audit data go to `audit_sample.csv` with classifier labels hidden.

## Criteria (§9.3), evaluated automatically unless noted

| # | Criterion | Threshold | Checked where |
|---|---|---|---|
| 1 | Coverage: designed modes (ECHO, CARRY, TRUNC, TRANS, DIGIT) ≥ 70 % of incorrect generations | `census_core.GATE_COVERAGE_MIN` | automatic |
| 2 | ≥ 3 designed modes with q_ref ≥ 0.05 | `GATE_MASS_COUNT/MIN` | automatic (+ merge note for < 0.02) |
| 3 | Ambiguity ≤ 10 % of designed-mode assignments | `GATE_AMBIGUITY_MAX` | automatic |
| 4 | Split-half TV ≤ 0.03 (prompt-clustered halves) | `GATE_TV_MAX` | automatic |
| 5 | Synthetic recovery ≥ 99 % | — | `tests/test_classifier.py` — **already green (D6: 100 % at n=5, n=8)** |
| 6 | Manual audit of 200 errors, disagreement ≤ 5 % | human | `audit_sample.csv` after a real census |
| §9.1 | Base accuracy (temperature 1.0) in [0.25, 0.60] to choose n | automatic, per n | census report |

§9.3 result table to fill after the real census:

| n | Base acc | Coverage | Modes ≥ 5 % | Ambiguity | TV | Verdict |
|---|---|---|---|---|---|---|
| 5 | TBD | TBD | TBD | TBD | TBD | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

## Known construct risk carried from D6

32–53 % of injected dropped-carry errors also match the one-digit-slip rule.
If the real-census ambiguity for CARRY/DIGIT exceeds 10 % (§9.3.3), merge into
one mode CARRY∪DIGIT **before any configuration is generated** (Gate 3
decision, recorded in DECISIONS.md). The checker reports ambiguity per the
multi-hot rules; the merge itself is a Phase 5 config-generation decision.

## Failure ladder (§9.3)

1. different n / answer-only prompt → 2. subtraction → 3. accept M=3 modes →
   stop if fewer than 3 designed modes reach 5 % mass.
