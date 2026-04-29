# Factorial Package (v1)

Scope:
- Fix factorial so that 0! == 1 while preserving existing behavior for n >= 1.
- Negative inputs are out of scope for this task.

How to run tests:
- Ensure Python 3.x is installed.
- From the repository root, run:
  - `pytest -q deliverable/lib/core/v1/test_factorial.py`
  - or simply `pytest -q` if you want to run all tests in the repo (if any).

Expected outcome:
- `factorial(0)` returns `1`, `factorial(n)` for n >= 1 returns `1 * 2 * ... * n`.
