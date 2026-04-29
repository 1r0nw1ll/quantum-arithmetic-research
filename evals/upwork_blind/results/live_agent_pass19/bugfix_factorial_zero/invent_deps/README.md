# Factorial Bugfix

Scope
- Fix factorial(0) to return 1 (0! = 1).
- Preserve existing behavior for n >= 1 (n! is product 1..n).
- Negative n is out of scope for this task.

What I changed
- Implemented a correct iterative factorial maintaining simple, robust input validation.
- Added tests covering 0, small n, and a larger n to ensure behavior stays correct.

How to run tests
- Ensure you have Python available and pytest installed (the project assumes a standard Python data-science stack).
- Run:

```
pytest -q
```

Expected result: all tests pass.
