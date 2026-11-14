# VERIFLOW completion plan: from the Gate 3 stop to a released paper

Written 2026-11-03, after D14. Supersedes the post-Phase-4 parts of the
ROADMAP calendar; the master plan stays the scientific reference.

> **Status 2026-11-07:** Stages 1–2 are done (D16–D20, D22). Stage 3 is in progress: LR pilot
> complete and analysed (D21). The calendar below is the original plan and is superseded by the
> dates in DECISIONS.md.
>
> **Stage 1 closed 2026-11-07 (D28): closed with a documented limitation.** The
> split-half TV diagnosis ([PHASE4_CLOSURE.md](PHASE4_CLOSURE.md)) finds no defect.
> The criterion is noisy at the census size used, the 1.5B failure stands (no 1.5B
> arm), and the locked q_ref is adequate for the 0.5B estimand. The §7 disclosure was
> filed as Addendum 1 in the OSF registration (D29, D31).
>
> **Status 2026-11-14:** all stages complete. Grid 96/96 valid (D32); registered analysis,
> supplement and duplicate-configuration disclosure (D33); paper and release (D34).

## Where the project stands

| Done | Evidence |
|---|---|
| Gate 0 novelty, Gate 1 maths (Props 1–7), Gate 2 simulation (27/27 cells) | GATE1.md, GATE2.md |
| Construct tooling, cluster stack, 24 real censuses | GATE3.md, D7–D14 |
| **Gate 3: STOP** under the v1 *mechanism* taxonomy | D14 |

The STOP is narrow. It says the base model's errors are **not explained by
named arithmetic mechanisms** (dropped carry, transposition, …): 55–97 % are
`U`. It does not say the errors lack structure. The experiment needs a
partition of wrong answers into modes, so that a verifier can accept
different modes at different rates. It does not need those modes to be
mechanisms.

**Exploratory check (existing addition census n=5, job 3620; not
confirmatory).** An *outcome-based* partition of wrong answers works:
- one wrong digit in the low half: 21 %;
- one wrong digit in the high half: 18 %;
- several wrong digits: 41 %;
- wrong length: 17 %;
- unparseable: 3 %.

That is four modes ≥ 5 %, no overlap, and 100 % coverage, at base accuracy
0.344 (inside the [0.25, 0.60] window). The project can continue with a
pre-registered amendment, tested on **fresh** data.

## The two tracks

- **Track A (main):** Amendment 1 → full GRPO experiment → paper. This is the
  strongest resume outcome: a pre-registered empirical study with theory.
- **Track B (guaranteed floor):** a paper built from Gates 1–2, plus the
  Gate 3 finding as an empirical result: "small-model arithmetic errors are
  not mechanism-structured". Its text is written alongside Track A (Stage 7),
  so a paper exists whatever happens at any later gate.

## Stages

Each stage ends at a gate with a written pass/stop rule. Stages run in order.
Cost figures are estimates until measured, and every gate decision is logged
in DECISIONS.md.

### Stage 1: Amendment 1 + Gate 3′ (construct, re-run). ~3 days, < 0.5 GPU-h

1. Write `docs/AMENDMENT_1.md` (D15) **before new data**:
   - Taxonomy v3 is an outcome-based partition, computed against the true
     answer: `I`, `LEN` (wrong length), `ONE_LO`, `ONE_HI` (exactly one wrong
     digit; position in the low or high half), `MULTI` (≥ 2 wrong digits,
     same length), `NEG`.
   - The old mechanism labels are kept as secondary annotations.
   - The task stays addition with prompt v2 and the v2 parser.
   - Disclose that the partition was chosen on exploratory census data
     (all 24 earlier runs are exploratory).
2. Run a **confirmatory census on a fresh split** (new base seed, so the
   operands have never been seen) at n = 5…10. The §9.1 selection rule and
   §9.3 thresholds are unchanged. Coverage is trivially 1 and ambiguity 0 under
   a partition, so the binding criteria are mass (≥ 3 modes ≥ 5 %), split-half
   TV ≤ 0.03 and the accuracy window.
3. Secondary check: run the census on the pinned **Qwen2.5-1.5B** as well,
   for the planned replication.
4. **Manual audit:** you judge the 200 blinded rows (~1–2 h). Disagreement
   must be ≤ 5 %.
5. Freeze `q_ref.lock`.
6. **Gate 3′ stop rule:** if the fresh census fails, go to Track B only.

### Stage 2: Phase 5, training stack (Gate 4: feasibility). ~1 week, ~5 GPU-h

1. Matched-configuration generator (§10.2), driven by `q_ref.lock`: exact
   matching of aggregate FPR and TPR, with the V / f_max grid.
2. Reward = the existing noisy verifier (SHA-256 PRF) wrapped as a TRL GRPO
   reward. Pin the TRL version in the `gpu` group.
3. `veriflow-smoke`: 1-step and 10-step GRPO, checkpoint save and resume,
   verifier log, per-mode logging.
4. `veriflow-bench`: steps/s for HF generate vs vLLM × G ∈ {8, 16}.
5. **Gate 4:** the projected grid (Stage 5) fits in ≤ 150 GPU-h on one GPU at
   a time. Otherwise drop to the master plan's "Minimum" design (24 runs)
   **before** any confirmatory run.

### Stage 3: Phase 6, pilot (Gate 5: empirical engagement). ~1–2 weeks, ~20 GPU-h

1. LR pilot on the homogeneous verifier: 3 learning rates plus a perfect
   verifier, 800 steps each. Fix the horizon T by the §13 rule.
2. Heterogeneity pilot: 4 configs × 3 seeds.
3. **Gate 5:** training measurably moves accuracy, and the per-mode q(t)
   responds to f as logged. If heterogeneity has *no* visible effect, the
   confirmatory test is still run and reported as a two-sided test, and the
   null result is the finding.
4. Power analysis by simulation (§15) chooses R ∈ {12, 24, 36} configs.

### Stage 4: Pre-registration. ~3 days, CPU

- `docs/PREREGISTRATION.md` covers H1–H3, the outcome window, the analysis
  code (run on pilot data as a dry run) and the exclusion and failure rules.
- Generate all configurations once and split them into training and held-out
  configs.
- Post it on OSF before the first confirmatory run. This turns the project from
  "a student ran experiments" into "a pre-registered study".

### Stage 5: Phase 7, confirmatory grid. ~2–3 weeks wall clock, ~60–150 GPU-h

- Standard design: 24 matched + 8 auxiliary configs × 2 seeds = 64 runs, plus
  the 1.5B replication (8 runs).
- One GPU job at a time, with ErrorEcho keeping priority.
- Each run carries a clean manifest and is resumable. No run is excluded for
  "looking wrong".

### Stage 6: Phase 8, analysis. ~1 week, CPU

- H1 (OLS with HC3), H2 and H3 (held-out MSE with a config-level bootstrap),
  and the hierarchical secondary model.
- Figures per §22, robustness checks, and theory-versus-experiment
  comparison plots.

### Stage 7: Paper, release, applications. ~2 weeks, CPU

- **Paper:** an 8–9 page workshop/conference format (ICLR / NeurIPS workshop
  or TMLR). Track B text is written alongside so a paper exists whatever
  happens.
- **arXiv preprint.** Code release on GitHub with a reproducibility bundle:
  locks, manifests, census data, and the one-command figure rebuild.
- **Resume/CV entry and a one-page project summary.**

## Applications track (runs in parallel; dated)

- **Summer@EPFL deadline: 2026-11-29**, before Stage 3 ends. Apply with the
  Gate 1–2 results, the census findings and the pre-registered design. A
  one-page summary will be ready by **2026-11-20** (Stage A1).
- Other programmes are verified individually as their dates come up.

## Calendar (one GPU, shared with ErrorEcho)

| Stage | Target |
|---|---|
| 1 Amendment + Gate 3′ | Nov 4 – Nov 7 |
| A1 application summary | by Nov 20 |
| 2 Training stack, Gate 4 | Nov 8 – Nov 16 |
| 3 Pilot, Gate 5 | Nov 17 – Nov 30 |
| 4 Pre-registration | Dec 1 – Dec 4 |
| 5 Confirmatory grid | Dec 5 – Dec 31 |
| 6 Analysis | Jan 1 – Jan 10 |
| 7 Paper + release | Jan 11 – Jan 31 |

## Integrity rules for every stage

- Decisions are logged **before** the data that could influence them.
  Post hoc changes are labelled as post hoc (D9, D11, Amendment 1).
- No q_ref, configuration or hypothesis is changed after confirmatory data
  exist.
- Every claim in the paper traces to a manifest (git commit, Slurm job id,
  hashes).
- Human-only steps (the manual audit, OSF posting, applications) are done by
  you. They are never simulated.
