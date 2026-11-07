# Gate 3: construct validity (Phase 4 exit)

**Status (2026-11-07): Gate 3′ (Amendment 1) passed at n = 5 by its declared procedure; Phase 4 is CLOSED WITH A DOCUMENTED LIMITATION (D28), not an unconditional pass.** Automated
criteria passed 2026-11-04 (D16). The 200-row audit (labels made by Claude
from the written rules, D20; reviewed by the researcher, D22) has 1/200
disagreement (AUD-0003, threshold 5 %), re-validated 2026-11-07. `q_ref.lock`
was frozen 2026-11-05 (lock sha256 `2c34f6d2…d5cb`, file `737aab73…6f8f`).

**Phase 4 closure (2026-11-07, D28): CLOSED WITH DOCUMENTED LIMITATION.** This is
a post hoc diagnosis of the split-half TV
([PHASE4_CLOSURE.md](PHASE4_CLOSURE.md), [gate3_tv_diagnosis.json](gate3_tv_diagnosis.json)).
It found no implementation defect. At 2,000 prompts × 4 samples the TV
statistic is close to a coin flip:
- re-randomising the halves would fail the selected 0.5B run with probability 0.41;
- the 1.5B failures sit at the 74th and 92nd percentiles of their own
  random-split distributions.

The 1.5B failure stands, and no 1.5B claim is made. The locked q_ref is
supported by an independent-split replicate (TV 0.012) and bootstrap
propagation. The pass above is reported with this limitation. It is not
re-labelled.

History: the v1/v2 mechanism taxonomy reached the §9.3 stop rule (D14; results
below). Amendment 1 (D15, `docs/AMENDMENT_1.md`, logged before data) replaced
it with an outcome partition and a never-generated split (seed 20261104). The
amended confirmatory census is the "Gate 3′" section directly below.

## What runs the gate

```bash
# verify every run + build the comparison + apply the §9.1 selection rule
uv run python -m experiments.compare_census runs/census/<run_id> ... \
    --json runs/census/gate3_comparison.json --md runs/census/gate3_comparison.md
# for the selected run only: gate report + blinded audit export (+ freeze after audit)
uv run python -m experiments.check_construct_gate --census runs/census/<run_id> \
    --json runs/census/<run_id>/gate3.json --audit-sample 200
uv run python -m experiments.check_construct_gate --census runs/census/<run_id> \
    --audit-results runs/census/<run_id>/audit_completed.csv --freeze-qref
```

`--freeze-qref` writes `configs/locked/q_ref.lock` only if all automated
criteria pass, the judged audit disagreement is ≤ 5 %, and the manifest shows
a real (non-mock) census at the pinned model revision. The lock records the run
id, Slurm job id, git commit, model revision, prompt/taxonomy hashes and the
q_ref.json SHA-256.

## Criteria (§9.3), evaluated automatically unless noted

| # | Criterion | Threshold | Checked where |
|---|---|---|---|
| 1 | Coverage: designed modes (ECHO, CARRY, TRUNC, TRANS, DIGIT) ≥ 70 % of incorrect generations | `census_core.GATE_COVERAGE_MIN` | automatic |
| 2 | ≥ 3 designed modes with q_ref ≥ 0.05 | `GATE_MASS_COUNT/MIN` | automatic (+ merge note for < 0.02) |
| 3 | Ambiguity ≤ 10 % of designed-mode assignments | `GATE_AMBIGUITY_MAX` | automatic |
| 4 | Split-half TV ≤ 0.03 (prompt-clustered halves) | `GATE_TV_MAX` | automatic |
| 5 | Synthetic recovery ≥ 99 % | none | `tests/test_classifier.py`: **green (D6: 100 % at n=5, n=8)** |
| 6 | Manual audit of 200 errors, disagreement ≤ 5 % | human | `audit_sample.csv` for the selected census |
| §9.1 | Base accuracy (temperature 1.0) in [0.25, 0.60] to choose n | automatic, per n | census report |

## Gate 3′: taxonomy v3, fresh split 20261104 (confirmatory), 2026-11-04

Census jobs 3640–3645 ran on Qwen2.5-0.5B-Instruct @ `7ae5576`, commit `ffa27a7`
(clean tree), preflight 3638, one L40, 22–33 s per n. The CPU report ran as job 3646.
Its table formatter crashed after the selection and audit were written;
it was fixed in `508fefa` and the tables were regenerated, with the audit files
byte-identical. All six runs were verified. Evidence:
[gate3v3_census_comparison.json](gate3v3_census_comparison.json).

| n | Run ID (job) | Base acc | q LEN | q MULTI | q ONE_HI | q ONE_LO | q NEG | q I | Coverage | Modes ≥5% | Ambig. | TV | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **5** | `n5_v3_s20261104_j3640` (3640) | **0.349** | 0.162 [0.148, 0.176] | 0.420 [0.401, 0.438] | 0.181 [0.166, 0.196] | 0.206 [0.190, 0.223] | 0.000 | 0.030 [0.025, 0.035] | 0.970 | **4** | 0.000 | 0.0177 | **PASS** |
| 6 | `n6_v3_s20261104_j3641` (3641) | 0.131 | 0.156 | 0.584 | 0.071 | 0.170 | 0.000 | 0.019 | 0.981 | 4 | 0.000 | 0.0272 | FAIL (9.1) |
| 7 | `n7_v3_s20261104_j3642` (3642) | 0.049 | 0.173 | 0.673 | 0.041 | 0.092 | 0.000 | 0.020 | 0.980 | 3 | 0.000 | 0.0206 | FAIL (9.1) |
| 8 | `n8_v3_s20261104_j3643` (3643) | 0.019 | 0.164 | 0.755 | 0.023 | 0.038 | 0.000 | 0.020 | 0.980 | 2 | 0.000 | 0.0028 | FAIL (9.3.2, 9.1) |
| 9 | `n9_v3_s20261104_j3644` (3644) | 0.009 | 0.141 | 0.802 | 0.012 | 0.029 | 0.000 | 0.016 | 0.984 | 2 | 0.000 | 0.0112 | FAIL (9.3.2, 9.1) |
| 10 | `n10_v3_s20261104_j3645` (3645) | 0.003 | 0.264 | 0.701 | 0.004 | 0.010 | 0.000 | 0.020 | 0.980 | 2 | 0.000 | 0.0104 | FAIL (9.3.2, 9.1) |

**Selection (§9.1): n = 5, run `n5_v3_s20261104_j3640`.**
- **Pre-declared merge:** q(NEG) = 0.000 < 0.02, so NEG merges into LEN before
  configuration generation. That leaves 4 designed modes (LEN, MULTI, ONE_HI,
  ONE_LO), all ≥ 0.16.
- **Replication of the exploratory estimate:** the fresh-split q_ref
  closely reproduces the exploratory one (job 3620: 0.167/0.408/0.182/0.211),
  which is independent evidence that the estimate is stable.
- **Mechanism annotation (descriptive):** 58 % U, 29 % DIGIT, 13 % CARRY.

**Criteria status:**
- 9.1 ✔
- 9.3.1 ✔
- 9.3.2 ✔
- 9.3.3 ✔
- 9.3.4 ✔
- 9.3.5 ✔ (`tests/test_taxonomy_v3.py`, ≥ 99 % for every designed mode at n = 5 and 8)
- **9.3.6 pending:** 200-row blinded audit at
  `audit/gate3v3_n5_s20261104_j3640/` (key kept on the cluster).

### Secondary: Qwen2.5-1.5B-Instruct (replication model; not used for selection)

Jobs 3647–3652 ran, with report 3653. Evidence:
[gate3v3_q15_census_comparison.json](gate3v3_q15_census_comparison.json).

| n | Run ID | Base acc | Modes ≥5% | TV | Gate |
|---|---|---|---|---|---|
| 5 | `n5_v3q15_s20261104_j3647` | 0.840 | 4 | 0.062 | FAIL (TV, 9.1) |
| 6 | `n6_v3q15_s20261104_j3648` | 0.744 | 4 | 0.029 | FAIL (9.1) |
| 7 | `n7_v3q15_s20261104_j3649` | 0.636 | 3 | 0.055 | FAIL (TV, 9.1) |
| 8 | `n8_v3q15_s20261104_j3650` | 0.504 | 3 | 0.039 | FAIL (TV) |
| 9 | `n9_v3q15_s20261104_j3651` | 0.488 | 3 | 0.050 | FAIL (TV) |
| 10 | `n10_v3q15_s20261104_j3652` | 0.246 | 3 | 0.015 | FAIL (9.1) |

The 1.5B model is far more accurate and almost always uses the `Answer:`
format. Within the accuracy window (n = 8, 9) the mode structure is present
(3 modes ≥ 5 %), but split-half TV fails (0.039, 0.050): its error
distribution varies more across prompts. The 1.5B replication arm therefore
needs its own decision before Stage 5. Options: a larger census for 1.5B, or
reporting it as descriptive only. It does not affect the 0.5B primary arm.

**Resolved (D28).** No confirmatory 1.5B arm. The post hoc diagnosis attributes
the failure to sampling variability under strong prompt clustering. It is not a
split artifact and not concentrated in rare classes. The declared failure is
not overturned. See [PHASE4_CLOSURE.md](PHASE4_CLOSURE.md).

## Results: prompt v1 + taxonomy v1, 2026-11-03 (SIT HPC, NVIDIA L40, bf16)

Sampling: T = 1.0, top_p = 1.0, top_k = 0, repetition_penalty = 1.0 (Qwen's
shipped 1.1 overridden), max_new_tokens = 128, seed 20261102, 2,000-bootstrap
prompt-clustered 95 % CIs. q is over incorrect generations (C excluded).
Full machine-readable evidence: [gate3_census_comparison.json](gate3_census_comparison.json).

| n | Run ID (Slurm job) | Base acc | q ECHO [95% CI] | q CARRY [95% CI] | q TRUNC [95% CI] | q TRANS [95% CI] | q DIGIT [95% CI] | q U [95% CI] | q I [95% CI] | Coverage | Modes ≥5% | Ambiguity | Split-half TV | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | `n5_s20261102_j3594` (3594) | 0.176 | 0.000 [0.000, 0.000] | 0.018 [0.014, 0.021] | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.000] | 0.072 [0.065, 0.079] | 0.087 [0.080, 0.095] | 0.823 [0.813, 0.833] | 0.090 | 1 | 0.152 | 0.0219 | FAIL (9.3.1, 9.3.2, 9.3.3, 9.1) |
| 6 | `n6_s20261102_j3595` (3595) | 0.102 | 0.000 [0.000, 0.001] | 0.011 [0.009, 0.014] | 0.001 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.074 [0.067, 0.080] | 0.155 [0.145, 0.164] | 0.759 [0.749, 0.769] | 0.087 | 1 | 0.116 | 0.0049 | FAIL (9.3.1, 9.3.2, 9.3.3, 9.1) |
| 7 | `n7_s20261102_j3596` (3596) | 0.060 | 0.000 [0.000, 0.000] | 0.011 [0.009, 0.014] | 0.001 [0.001, 0.002] | 0.000 [0.000, 0.001] | 0.065 [0.059, 0.071] | 0.226 [0.215, 0.235] | 0.696 [0.686, 0.707] | 0.078 | 1 | 0.119 | 0.0109 | FAIL (9.3.1, 9.3.2, 9.3.3, 9.1) |
| 8 | `n8_s20261102_j3597` (3597) | 0.043 | 0.000 [0.000, 0.000] | 0.008 [0.006, 0.011] | 0.002 [0.001, 0.003] | 0.001 [0.000, 0.001] | 0.042 [0.037, 0.047] | 0.300 [0.290, 0.311] | 0.647 [0.636, 0.657] | 0.053 | 0 | 0.118 | 0.0079 | FAIL (9.3.1, 9.3.2, 9.3.3, 9.1) |
| 9 | `n9_s20261102_j3600` (3600) | 0.031 | 0.000 [0.000, 0.000] | 0.007 [0.006, 0.010] | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.001] | 0.028 [0.025, 0.033] | 0.256 [0.246, 0.267] | 0.706 [0.695, 0.716] | 0.038 | 0 | 0.161 | 0.0078 | FAIL (9.3.1, 9.3.2, 9.3.3, 9.1) |
| 10 | `n10_s20261102_j3602` (3602) | 0.015 | 0.000 [0.000, 0.000] | 0.005 [0.003, 0.007] | 0.001 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.015 [0.012, 0.018] | 0.253 [0.244, 0.263] | 0.726 [0.716, 0.736] | 0.021 | 0 | 0.181 | 0.0064 | FAIL (9.3.1, 9.3.2, 9.3.3, 9.1) |

Only the stability criterion (9.3.4) passes at every n. Selection rule (§9.1):
smallest n with base accuracy in [0.25, 0.60] and the automated gate passing →
**none**.

Runs n9 job 3598 and n10 job 3599 crashed after generation (int64 overflow when
writing a 20-digit ECHO answer to parquet). Their empty run directories and logs
are kept as failure evidence; the bug was fixed (commit `4e812da`) and both n
were rerun as 3600/3602. Runs 3594–3597 used commit `9b41884`, runs 3600/3602
`4e812da`. Generation and classification code are identical between the two;
only the parquet type of the parsed answer changed.

### Diagnosis (what the failure is made of)

| n | `I` share of errors | `I` at the 128-token cap | all generations at cap | `I` that state the true sum somewhere |
|---|---|---|---|---|
| 5 | 0.823 | 0.619 | 0.448 | 0.305 |
| 6 | 0.759 | 0.626 | 0.456 | 0.257 |
| 7 | 0.696 | 0.638 | 0.461 | 0.215 |
| 8 | 0.647 | 0.606 | 0.426 | 0.195 |
| 9 | 0.706 | 0.623 | 0.459 | 0.156 |
| 10 | 0.726 | 0.609 | 0.463 | 0.110 |

- **Format failure dominates.** 65–82 % of errors are `I`, almost all
  `no_answer_line`: the model writes long step-by-step working and is cut off at
  `max_new_tokens = 128`, or ends with prose ("The final answer is X") with no
  `Answer:` line. The parser behaves exactly as §9.2 specifies.
- **The 128-token cap is not a master-plan value.** §11 asks for "the same
  max_completion_length" as training, and that is not fixed until Phase 5;
  128 was a Phase-4 code default.
- **Among parseable errors, `U` dominates as n grows** (DIGIT is the only
  designed mode near 5 %). Even with `I` removed, designed coverage is about
  0.51 at n=5 and falls with n. That is below 0.70, so fixing format alone
  is unlikely to be enough at n ≥ 6.
- **Ambiguity 0.12–0.18 > 0.10**, driven by CARRY∩DIGIT, as D6 predicted.

## Results: prompt v2 (answer-only) + taxonomy v1, 2026-11-03

Template `Compute {a} + {b}.` / `Reply with only the final answer, exactly in the form` /
`Answer: <integer>` with max_new_tokens = 256. Everything else is identical to
v1, including sampling, split, seed and taxonomy. Commit `6c04e29`, preflight 3609.
Evidence: [gate3_census_comparison_p2.json](gate3_census_comparison_p2.json).

| n | Run ID (Slurm job) | Base acc | q CARRY | q DIGIT | q U | q I | Coverage | Modes ≥5% | Ambiguity | Split-half TV | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | `n5_p2_s20261102_j3610` (3610) | 0.024 | 0.002 [0.001, 0.004] | 0.012 [0.009, 0.014] | 0.013 [0.011, 0.016] | 0.973 [0.969, 0.976] | 0.014 | 0 | 0.127 | 0.0040 | FAIL (9.3.1, 9.3.2, 9.3.3, 9.1) |
| 6 | `n6_p2_s20261102_j3611` (3611) | 0.012 | 0.001 [0.000, 0.002] | 0.010 [0.007, 0.012] | 0.021 [0.018, 0.025] | 0.968 [0.964, 0.972] | 0.011 | 0 | 0.071 | 0.0066 | FAIL (9.3.1, 9.3.2, 9.1) |
| 7 | `n7_p2_s20261102_j3612` (3612) | 0.008 | 0.001 [0.000, 0.002] | 0.007 [0.005, 0.009] | 0.031 [0.027, 0.035] | 0.961 [0.956, 0.966] | 0.008 | 0 | 0.094 | 0.0042 | FAIL (9.3.1, 9.3.2, 9.1) |
| 8 | `n8_p2_s20261102_j3613` (3613) | 0.005 | 0.001 [0.000, 0.001] | 0.006 [0.004, 0.008] | 0.041 [0.036, 0.045] | 0.953 [0.948, 0.958] | 0.006 | 0 | 0.040 | 0.0047 | FAIL (9.3.1, 9.3.2, 9.1) |
| 9 | `n9_p2_s20261102_j3614` (3614) | 0.002 | 0.001 [0.000, 0.001] | 0.004 [0.003, 0.006] | 0.035 [0.031, 0.040] | 0.960 [0.955, 0.964] | 0.005 | 0 | 0.098 | 0.0034 | FAIL (9.3.1, 9.3.2, 9.1) |
| 10 | `n10_p2_s20261102_j3615` (3615) | 0.001 | 0.000 [0.000, 0.001] | 0.003 [0.002, 0.004] | 0.037 [0.033, 0.042] | 0.959 [0.955, 0.964] | 0.003 | 0 | 0.077 | 0.0034 | FAIL (9.3.1, 9.3.2, 9.1) |

ECHO, TRUNC and TRANS are 0.000 (upper CI ≤ 0.001) at every n. Each job took
23–35 s of generation and ≤ 2.5 GB of VRAM.

**Diagnosis.** The model obeys "only the final answer" literally and drops the
`Answer:` prefix:

| n | `I` share of errors | `I` that are a bare integer | `I` containing the true sum |
|---|---|---|---|
| 5 | 0.973 | 0.983 | 0.347 |
| 6 | 0.968 | 0.987 | 0.134 |
| 7 | 0.961 | 0.987 | 0.051 |
| 8 | 0.953 | 0.986 | 0.019 |
| 9 | 0.960 | 0.987 | 0.011 |
| 10 | 0.959 | 0.982 | 0.002 |

Under taxonomy v1 (§9.2) a completion with no `Answer:` line is `I`, so these
count as format failures. The last column is diagnostic only, not a gate
statistic. Even if bare integers were accepted, accuracy would only plausibly
reach the [0.25, 0.60] window at n = 5. Accepting them is a taxonomy change
(v2), which by §9.2 invalidates every census so far, so it was not done
without a decision.

## Results: prompt v2 + taxonomy v2 (addition, final), 2026-11-03

Commit `47fa0d2`, preflight 3618 (132 tests green, synthetic recovery included).
Sampling is identical to the earlier runs. Evidence:
[gate3_census_comparison_p2t2.json](gate3_census_comparison_p2t2.json).
Each job took 23–35 s of generation and ≤ 2.5 GB of VRAM. All six runs were
verified (artifacts, q normalisation, CIs, re-derived statistics, pinned
revision, template hash).

| n | Run ID (Slurm job) | Base acc | q ECHO [95% CI] | q CARRY [95% CI] | q TRUNC [95% CI] | q TRANS [95% CI] | q DIGIT [95% CI] | q U [95% CI] | q I [95% CI] | Coverage | Modes ≥5% | Ambiguity | Split-half TV | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | `n5_p2t2_s20261102_j3620` (3620) | 0.344 | 0.000 [0.000, 0.000] | 0.120 [0.106, 0.134] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 0.295 [0.277, 0.313] | 0.552 [0.532, 0.571] | 0.032 [0.027, 0.037] | 0.416 | 2 | 0.235 | 0.0219 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity) |
| 6 | `n6_p2t2_s20261102_j3621` (3621) | 0.136 | 0.000 [0.000, 0.000] | 0.052 [0.045, 0.060] | 0.001 [0.000, 0.002] | 0.002 [0.001, 0.004] | 0.205 [0.190, 0.218] | 0.720 [0.705, 0.736] | 0.020 [0.016, 0.023] | 0.260 | 2 | 0.156 | 0.0185 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 7 | `n7_p2t2_s20261102_j3622` (3622) | 0.052 | 0.000 [0.000, 0.000] | 0.025 [0.020, 0.030] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.001] | 0.116 [0.106, 0.125] | 0.838 [0.827, 0.849] | 0.020 [0.016, 0.023] | 0.143 | 1 | 0.143 | 0.0183 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 8 | `n8_p2t2_s20261102_j3623` (3623) | 0.021 | 0.000 [0.000, 0.000] | 0.011 [0.009, 0.014] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.061 [0.054, 0.068] | 0.904 [0.895, 0.912] | 0.022 [0.019, 0.026] | 0.074 | 1 | 0.110 | 0.0134 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 9 | `n9_p2t2_s20261102_j3624` (3624) | 0.011 | 0.000 [0.000, 0.000] | 0.007 [0.005, 0.009] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.033 [0.028, 0.038] | 0.943 [0.937, 0.949] | 0.016 [0.013, 0.019] | 0.040 | 0 | 0.122 | 0.0045 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 10 | `n10_p2t2_s20261102_j3625` (3625) | 0.003 | 0.000 [0.000, 0.000] | 0.002 [0.001, 0.004] | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.001] | 0.013 [0.010, 0.016] | 0.965 [0.960, 0.969] | 0.020 [0.016, 0.023] | 0.015 | 0 | 0.106 | 0.0059 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |

- **n = 5** is the only n inside the §9.1 accuracy window (0.344), but it fails
  coverage (0.416 < 0.70), mass (2 designed modes ≥ 5 %: DIGIT 0.295, CARRY
  0.120) and ambiguity (0.235 > 0.10, CARRY∩DIGIT, as D6 predicted). Merging
  CARRY∪DIGIT as §9.3.3 prescribes would leave **one** designed mode.
- ECHO, TRUNC and TRANS are ≤ 0.002 at every n: the model does not make those
  errors. Rising n moves mass into `U`, not into designed modes.
- Split-half TV (≤ 0.022) passes everywhere, so these q estimates are stable:
  the failure is real, not noise.

## Results: subtraction (ladder step 2), 2026-11-03

`a − b`, a > b, prompt `Compute {a} - {b}.` (v2 wording), taxonomy `sub-v1`
(BORROW in CARRY's place), taxonomy-v2 parser, max_new_tokens 256. Commit
`36de395`, preflight 3626 (146 tests green, subtraction synthetic recovery
≥ 99 % for every designed mode at n = 5, 8). Each job took 27–39 s and ≤ 2.5 GB
of VRAM. All six runs were verified. Evidence:
[gate3_census_comparison_sub.json](gate3_census_comparison_sub.json).

| n | Run ID (Slurm job) | Base acc | q BORROW [95% CI] | q DIGIT [95% CI] | q ECHO [95% CI] | q TRANS [95% CI] | q TRUNC [95% CI] | q U [95% CI] | q I [95% CI] | Coverage | Modes ≥5% | Ambiguity | Split-half TV | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | `n5_sub_p2t2_s20261102_j3627` (3627) | 0.150 | 0.095 [0.086, 0.104] | 0.283 [0.268, 0.298] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 0.004 [0.002, 0.006] | 0.554 [0.538, 0.570] | 0.064 [0.058, 0.070] | 0.383 | 2 | 0.171 | 0.0218 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 6 | `n6_sub_p2t2_s20261102_j3628` (3628) | 0.105 | 0.078 [0.071, 0.086] | 0.230 [0.217, 0.243] | 0.000 [0.000, 0.001] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.003] | 0.647 [0.632, 0.661] | 0.042 [0.037, 0.047] | 0.311 | 2 | 0.160 | 0.0185 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 7 | `n7_sub_p2t2_s20261102_j3629` (3629) | 0.083 | 0.045 [0.039, 0.052] | 0.138 [0.128, 0.147] | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.001] | 0.004 [0.002, 0.006] | 0.771 [0.760, 0.782] | 0.041 [0.037, 0.046] | 0.187 | 1 | 0.140 | 0.0145 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 8 | `n8_sub_p2t2_s20261102_j3630` (3630) | 0.053 | 0.028 [0.024, 0.032] | 0.093 [0.085, 0.100] | 0.000 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.003 [0.002, 0.004] | 0.830 [0.820, 0.839] | 0.046 [0.041, 0.051] | 0.125 | 1 | 0.127 | 0.0083 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 9 | `n9_sub_p2t2_s20261102_j3631` (3631) | 0.041 | 0.024 [0.021, 0.028] | 0.067 [0.060, 0.073] | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.001] | 0.003 [0.002, 0.004] | 0.854 [0.845, 0.863] | 0.051 [0.046, 0.056] | 0.095 | 1 | 0.149 | 0.0058 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |
| 10 | `n10_sub_p2t2_s20261102_j3632` (3632) | 0.013 | 0.010 [0.008, 0.012] | 0.034 [0.029, 0.039] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.001] | 0.004 [0.002, 0.005] | 0.897 [0.890, 0.904] | 0.054 [0.049, 0.059] | 0.049 | 0 | 0.126 | 0.0087 | FAIL (9.3.1 coverage, 9.3.2 mass, 9.3.3 ambiguity, 9.1 base accuracy window) |

- Base accuracy is 0.150 at n = 5 and falls with n, so it is below the
  [0.25, 0.60] window everywhere. Only DIGIT and BORROW ever reach 5 %, and
  ambiguity is 0.13–0.17 (BORROW∩DIGIT, as predeclared in D13).
- The U-only diagnostics explain almost none of `U`: about 99.6 % of `U`
  errors match none of OPSWAP (a+b), NEG (b−a) or SFL. The errors are not a
  known structured mechanism the taxonomy is missing.

## Failure ladder (§9.3): where it stands

| Step | Status |
|---|---|
| 1. different n | done: fails at all n = 5…10 (all addition configurations) |
| 1. answer-only prompt | done: prompt v2 (D9), plus taxonomy v2 parse fix (D11); fails |
| 2. 2-operand subtraction | done (D13): fails at all n; ≤ 2 designed modes at 5 % |
| 3. accept M = 3 modes | **unreachable**: at most 2 designed modes reach 5 % for either task at any n |
| 4. stop if < 3 designed modes reach 5 % | **TRIGGERED**: Gate 3 verdict is STOP (D14) |

## Manual audit tooling (ready, unused)

`check_construct_gate.py` writes `audit_sample.csv` (row_id, prompt_id, a, b,
completion, answer_true, empty `judgment`; no classifier labels),
a separate `audit_key.csv`, and `audit_instructions.md` with the permitted
labels. As of D8, `AMBIG` agrees only with rows the classifier itself found
multi-rule, so an all-AMBIG audit cannot pass.
