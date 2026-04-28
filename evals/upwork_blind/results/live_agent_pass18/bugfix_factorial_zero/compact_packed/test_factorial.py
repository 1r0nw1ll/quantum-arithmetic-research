from factorial import factorial


def test_factorial_zero(): assert factorial(0) == 1


def test_factorial_small_values(): assert [factorial(n) for n in range(1, 6)] == [1, 2, 6, 24, 120]


def test_factorial_larger_value(): assert factorial(8) == 40320
