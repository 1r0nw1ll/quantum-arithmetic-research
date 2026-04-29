Project Scope
- Bug fix: factorial(0) should be 1; 0! = 1
- Preserve existing behavior for n >= 1; factorial(n) remains the product 1..n
- Negative n out of scope

How to Run Tests
- Prereqs: Python and pytest (optional; can use unittest as fallback)
- Install dependencies if needed (none for this task)
- Run: `pytest -q`
- Expected: all tests pass (including factorial(0) == 1)

Notes
- The core function is intentionally compact, using a lambda with a chained ternary and a reduce over a list comprehension.