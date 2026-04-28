# Factorial Bugfix

## Scope

This change fixes `factorial(0)` so it returns `1`, while preserving the
existing factorial behavior for `n >= 1`. Negative inputs are out of scope.

The implementation module also includes a top-level demonstration block that
reads `data/example.csv` and prints sample results on import.

## Files

- `factorial.py`: factorial implementation plus import-time demo output
- `test_factorial.py`: tests for zero, small values, and a larger value
- `data/example.csv`: sample inputs for the demo

## Run the tests

```bash
pytest -q
```
