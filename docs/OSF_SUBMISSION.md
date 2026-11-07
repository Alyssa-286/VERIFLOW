# OSF submission pack (researcher action)

Posting the registration is the one step Claude cannot do: it needs your OSF account. Nothing confirmatory runs until it is posted (§15, §28).

## What to register

- **Registration document:** `docs/PREREGISTRATION.md` at the **frozen commit** named in the GitHub tag `prereg-v1` (`git show prereg-v1:docs/PREREGISTRATION.md`).
- **Supporting files to attach:**
  - `docs/DECISIONS.md` (D0–D26);
  - `docs/AMENDMENT_1.md`;
  - `configs/locked/q_ref.lock`;
  - `configs/pilot/{selection,gate5,power}.json`;
  - `configs/grid/runs_grid.csv` and `configs/grid/split.json`;
  - `results/pilot/validate_het.json`.
- **Code snapshot:** the tag `prereg-v1` on `github.com/Alyssa-286/VERIFLOW`. If the repository is private, attach a zip: `git archive -o veriflow-prereg-v1.zip prereg-v1`.

## Steps

1. Go to osf.io → *Create new project* "VERIFLOW: error-mode structure of verifier false positives in RLVR".
2. Upload the files above to the project's storage.
3. *Registrations* → *New registration* → template **"OSF Preregistration"**. Fill each field by copying the matching section of `PREREGISTRATION.md`:
   - **Study information:** title and question (§1); hypotheses H1–H3 (§5), and state that the claim type is *estimation with CIs* (§5, §7).
   - **Design plan:** §3 (matched 36 + auxiliary 12 configurations, 2 seeds, held-out 13). Blinding: not applicable (automated verifier). Randomisation: config generation, mode assignment, held-out split and run order are all seeded (§3, D26).
   - **Sampling plan:** existing data: *registration prior to creation of data* (the pilot is exploratory and disclosed). Data collection: §2 and §3. Sample size: 48 configurations × 2 seeds = 96 runs. Rationale: the power analysis (§7). Stopping rule: the grid runs to completion with no interim looks (§6).
   - **Variables:** manipulated: the verifier configuration (V, f_max, F̄). Measured: Y and the secondary outcomes (§4).
   - **Analysis plan:** §5 (models, HC3, bootstrap, Holm, ICC); inference criteria (estimation with CIs); data exclusion and missing data (§6); exploratory analyses (§5, secondary).
   - **Other:** the disclosures in §2 (prompt and parser changes, Amendment 1, the AI-labelled audit reviewed by the researcher, eval cadence, the RNG re-seed, CPU count), and the pilot results (§7).
4. Choose an embargo if you want one (OSF allows up to 4 years). Then *Register*.

## What to send back to Claude

The registration URL or DOI, and the date. Claude then:
1. logs it as D27;
2. submits the 96-run grid on the frozen commit:

   ```bash
   sbatch --array=0-95%2 --cpus-per-task=2 --export=ALL,RUN_TABLE=configs/grid/runs_grid.csv,RUN_KIND=grid,RUNS_ROOT=runs/grid,EVAL_EVERY=10 slurm/grpo_array.sbatch
   ```

3. validates every run with `experiments/validate_runs.py`;
4. runs the registered analysis.
