# Factorial Bugfix

## Scope

This change fixes `factorial(0)` so it returns `1`, while preserving the
existing behavior for inputs `n >= 1`.

Negative inputs are out of scope for this task.

## Files

- `factorial.py`: factorial implementation
- `test_factorial.py`: minimal tests for `0`, small values, and a larger value

## Run tests

If `pytest` is available, run:

```bash
pytest -q
```
