# Factorial Bugfix Scope

This task fixes the `factorial(0)` behavior so it returns `1`, while preserving
the existing factorial calculation for `n >= 1`.

Negative inputs are out of scope.

## Files

- `factorial.py`: factorial implementation
- `test_factorial.py`: parametrized pytest coverage for `0`, small values, and a larger value

## Run the tests

```bash
pytest -q
```
