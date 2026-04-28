def factorial(n: int) -> int:
    if n == 0:
        return 1

    result = 1
    for value in range(1, n + 1):
        result *= value
    return result
