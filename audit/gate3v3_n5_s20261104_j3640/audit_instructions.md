# Manual audit (master plan §9.3.6), census run `n5_v3_s20261104_j3640`

Do NOT open `audit_key.csv` until every row is judged.

1. Copy `audit_sample.csv` to `audit_completed.csv` in this same directory.
2. For each of the 200 rows, read `completion` and find the final answer: the
   integer on the `Answer:` line, or, if there is no `Answer:` line, the
   completion itself when it is nothing but one integer. Compare it with
   `answer_true` (= a + b) and write ONE label in `judgment`: NEG, LEN, ONE_HI, ONE_LO, MULTI, I.
     I      no Answer: line and not a bare integer, an unparseable Answer: value,
            or conflicting Answer: lines
     NEG    the answer is negative
     LEN    the answer has a different number of digits than the true sum
     ONE_HI same number of digits, exactly one digit differs, and it is in the
            high half: position < floor(L/2) counting from the leftmost digit
            as 0, where L is the number of digits of the true sum
     ONE_LO same number of digits, exactly one digit differs, at position >= floor(L/2)
     MULTI  same number of digits, two or more digits differ
   Exactly one label fits each row (the classes do not overlap).
3. Leave every other column unchanged. Validate with
   `python -m experiments.check_construct_gate --census <this dir> --audit-results <this dir>/audit_completed.csv`.
   Disagreement must be <= 5 %.
