# QA Rule 30 Negative Fixtures

These fixtures are schema-realistic but intentionally invalid to prove Family [34] gates are enforced.

Expected failures (exit code == 1):
- cert_neg_missing_invariant_diff.json -> MISSING_INVARIANT_DIFF
- cert_neg_scope_invalid.json          -> SCOPE_INVALID
- cert_neg_aggregate_mismatch.json     -> AGGREGATE_MISMATCH

Meta-validator asserts exit code, fail marker, and a human-readable reason substring.
