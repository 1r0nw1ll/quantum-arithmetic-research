# Factorial (0! = 1) Bugfix Demonstration

Scope:
- Fix factorial(0) so that 0! == 1 and preserve existing behavior for n >= 1.
- Provide a small demo at module import time to aid reviewers.
- Include tests for 0, small n, and a larger n.

How to run tests:
- Ensure Python 3 is available.
- Install pytest if needed: `pip install pytest`.
- Run tests from the project root: `pytest -q`.

Files:
- `factorial.py` - fixed implementation with top-level demo
- `test_factorial.py` - pytest-based tests for 0, small n, and large n
- `data/example.csv` - sample input values for the demo
- `README.md` - this guide
