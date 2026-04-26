# Bugfix: factorial(0) returns 0

The current implementation returns 0 for `factorial(0)`. Standard factorial
says `0! = 1`. Fix the function so:

- `factorial(0) == 1`
- `factorial(n)` for n >= 1 returns the product `1 * 2 * ... * n`
- existing behavior for n >= 1 is preserved

Deliver:
- the fixed function as `factorial.py`
- a `test_factorial.py` that exercises `factorial(0)`, small n, and a larger n
- a README with scope and how to run the tests

Negative n is out of scope for this task.
