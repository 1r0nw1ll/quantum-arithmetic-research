# Factorial Bugfix Deliverable

## Scope

This deliverable fixes `factorial(0)` so it returns `1`, while preserving the
existing factorial behavior for inputs `n >= 1`.

Negative inputs are intentionally out of scope for this task.

## Package Layout

The implementation is packaged under `deliverable/lib/core/v1/` with
`__init__.py` files at every level to expose the public `factorial` function.

## Run The Tests

From the current working directory:

```bash
pytest -q
```
