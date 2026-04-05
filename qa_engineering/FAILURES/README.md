# Failure Gallery

## How to use this folder

1. Run the validator
2. Copy the FAIL type exactly from the error output
3. Open the matching file below
4. Apply the one-line fix
5. Re-run

Do not try to debug from scratch — use the failure type.

---

Each file shows one validator failure type: a minimal broken cert, the exact error output,
why it fails, and the one-line fix.

| File | Fail type | Short description |
|------|-----------|------------------|
| `FAIL_STATE_ENCODING_INVALID.md` | `STATE_ENCODING_INVALID` | b or e is 0 or outside {1,...,N} |
| `FAIL_ARITHMETIC_OBSTRUCTION.md` | `ARITHMETIC_OBSTRUCTION_IGNORED` | target_r has v_p=1 for an inert prime |
| `FAIL_ORBIT_CLASSIFICATION.md` | `ORBIT_FAMILY_CLASSIFICATION_FAILURE` | declared orbit_family doesn't match f(b,e) |
| `FAIL_TRANSITION_NOT_GENERATOR.md` | `TRANSITION_NOT_GENERATOR` | generator field is empty or missing |

For the full list of validator checks, see cert [121]:
`qa_engineering_core_cert_validate.py`
