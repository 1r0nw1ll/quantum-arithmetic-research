# Factorial Bugfix and Tests

Scope:
- Fix factorial(0) to return 1 and preserve correct behavior for n >= 1.
- Add thorough pytest tests using parameterization with advisory metadata.
- Provide README with how to run tests.

How to run:
- Ensure Python 3.x is installed.
- Install pytest if needed: `pip install pytest`.
- Run tests: `pytest -q`.

Files delivered:
- `factorial.py`: fixed factorial implementation.
- `test_factorial.py`: parameterized tests with 4 columns per case (input, expected, label, note).
- `README.md`: scope and test instructions.
