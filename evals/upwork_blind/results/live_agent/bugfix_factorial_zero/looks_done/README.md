# Factorial Bugfix

## Scope

This bundle fixes the `factorial(0)` case so that `0! == 1` while preserving
the expected factorial behavior for `n >= 1`.

Negative inputs are out of scope for this task.

## Files

- `factorial.py`: factorial implementation
- `test_factorial.py`: tests for zero, small values, and a larger value

## Run the tests

If `pytest` is available, run:

```bash
pytest
```
