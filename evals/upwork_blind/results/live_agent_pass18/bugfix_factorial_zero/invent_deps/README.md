# Factorial Bugfix

## Scope

This deliverable fixes `factorial(0)` so it returns `1`, while preserving the
standard product behavior for `n >= 1`.

Negative inputs are out of scope for this task.

## Files

- `factorial.py`: factorial implementation
- `test_factorial.py`: test coverage for `0`, small values, and a larger value

## Run Tests

```bash
pytest -q
```
