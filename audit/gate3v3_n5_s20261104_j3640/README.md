# Your Gate 3′ manual audit (§9.3.6): what to do

This is the only Gate 3′ step that has to be done by a person. Your judgments decide
whether `q_ref.lock` can be frozen.

1. Copy `audit_sample.csv` to `audit_completed.csv` in this folder.
2. Follow `audit_instructions.md`. Fill the `judgment` column for all 200 rows
   with one of `NEG, LEN, ONE_HI, ONE_LO, MULTI, I`, and change nothing else.
   For example, for the true sum 169440 (L = 6, high half = positions 0, 1, 2),
   the answer 179440 differs only at position 1, so it is `ONE_HI`.
3. Tell Claude it is done. The completed file is copied next to the answer key
   on the cluster (`~/VERIFLOW/runs/census/n5_v3_s20261104_j3640/`) and checked with:

   ```bash
   cd ~/VERIFLOW && uv run python -m experiments.check_construct_gate \
       --census runs/census/n5_v3_s20261104_j3640 \
       --audit-results runs/census/n5_v3_s20261104_j3640/audit_completed.csv --freeze-qref
   ```

   It prints the disagreement rate. The lock is written only if the rate is ≤ 5 %.

The answer key (`audit_key.csv`) deliberately stays on the cluster. Don't look at
it, or at any classifier output, before you finish: that is what keeps the audit
blinded.
