"""
Compute the factorial of non-negative integers.
0! == 1 by definition.
"""


def factorial(n: int) -> int:
    """Return the factorial of n.

    - 0! == 1
    - n! == 1 * 2 * ... * n for n >= 1
    """
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")

    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
