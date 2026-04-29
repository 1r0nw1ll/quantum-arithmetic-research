def factorial(n: int) -> int:
    """
    Compute the factorial of a non-negative integer n.
    - 0! = 1
    - n! = 1 * 2 * ... * n for n >= 1
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
