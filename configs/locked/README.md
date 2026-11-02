# configs/locked/

Frozen inputs. Every file here carries its own SHA-256; a change must be a new
file, never an edit, and is recorded in `docs/DECISIONS.md` before any GPU data
it affects exist.

| File | Contents | Written by |
|---|---|---|
| `prompt.lock` | prompt template + its SHA-256 (from `vf_datasets.arithmetic`) | frozen at Phase 4 (D7) |
| `taxonomy.lock` | taxonomy version + rule table reference | frozen at Phase 4 (D7) |
| `q_ref.lock` | frozen census distribution q_ref + census-run hash | `experiments/check_construct_gate.py --freeze-qref` after a passing census (Gate 3) |
| `prereg.lock` | pre-registration hash | Phase 7 (§15) |
