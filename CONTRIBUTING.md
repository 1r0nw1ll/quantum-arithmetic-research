# Contributing

## Quick start (local)

All validators are deterministic and require only Python 3.10+ stdlib. No GPU, no pip install, no external downloads.

Run the full suite from the repo root:

```bash
cd qa_alphageometry_ptolemy
python qa_meta_validator.py           # 15 tests: tetrad + conjectures + FST + agent security + Kayser
python qa_conjecture_core.py          # 5 checks: factories, ledger, guards
python qa_fst/qa_fst_validate.py      # 8 checks: spine, certs, manifest
python qa_kayser/qa_kayser_validate.py --all  # 28 checks: C1-C6 correspondence suite (merkle-rooted)
```

## Reporting a failure (as an obstruction)

Open a GitHub Issue and include:

1. Command run + OS + Python version
2. Exact error output
3. If applicable: the certificate or input file that triggered it
4. Any invariant diffs reported by the validator

Tag with one of:

- `obstruction:invariant` -- invariant mismatch or violation
- `obstruction:non-reachability` -- generator reachability failure
- `obstruction:drift` -- numeric or behavior drift between environments
- `obstruction:schema` -- structural or schema-version violation

## CI

All validators run automatically on push and PR via GitHub Actions. See the badge at the top of the README for current status.
