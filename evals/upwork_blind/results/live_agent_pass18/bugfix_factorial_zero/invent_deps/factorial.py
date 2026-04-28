def factorial(n: int) -> int:
    """Return n! for non-negative integers."""
    result = 1
    for value in range(1, n + 1):
        result *= value
    return result
