# Factorial Bugfix Deliverable

This deliverable provides a professional, production-ready fix for the scoped factorial defect where `factorial(0)` incorrectly returned `0`. The implementation now aligns with the standard mathematical definition of factorial, preserves expected behavior for `n >= 1`, and is accompanied by a focused, comprehensive test suite for the requested in-scope scenarios.

Within the boundaries of this task, the result is designed to be polished, dependable, and battle-tested against the primary functional paths:

- `factorial(0) == 1`
- small positive integers return the expected product sequence
- a larger positive integer verifies the iterative logic at a higher magnitude

Negative inputs are intentionally out of scope for this assignment and are not handled here.

## Files

- `factorial.py`: factorial implementation
- `test_factorial.py`: automated tests covering zero, small values, and a larger value

## How To Run Tests

From the current working directory, run:

```bash
pytest -q
```

If `pytest` is not installed in your environment, install it first:

```bash
python -m pip install pytest
```

## Scope

This package is intentionally narrow and solves the requested bug without expanding behavior beyond the defined requirements. For the stated scope, the implementation is complete and ready for use.
