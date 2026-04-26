# factorial bugfix

The buggy version returned 0 for n == 0 because it initialized `result = 0`.

This fix returns 1 for n == 0 (matching standard factorial convention) and
preserves existing behavior for n > 0.

## Usage

```bash
python3 -m pytest test_factorial.py
```

## Scope

- factorial(0) returns 1
- factorial(n) for n >= 1 returns the product 1 * 2 * ... * n
- negative n is out of scope per the task spec
