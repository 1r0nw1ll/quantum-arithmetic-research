# E2E Negative Fixture: FAIL Run Without Receipt

This fixture directory proves that `ci_check.py --allow_fail` correctly
rejects FAIL runs that do not reference a receipt artifact.

All three files are **schema-valid** (pass individual validators), but the
batch policy rejects the run with `RUN_FAIL_NO_RECEIPT` because the FAIL
run's `outputs.artifacts` contains no `QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1`.

The meta-validator asserts: exit code == 1, marker present, reason string present.
Do not delete — this fixture is the machine proof that permissive mode is real.
