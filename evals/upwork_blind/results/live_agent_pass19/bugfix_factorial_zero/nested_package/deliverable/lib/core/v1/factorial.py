def factorial(n: int) -> int:
    """Return the factorial of n.

    0! == 1 by definition. For n >= 1, returns 1 * 2 * ... * n.
    Negative inputs are out of scope for this task.
    """
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
