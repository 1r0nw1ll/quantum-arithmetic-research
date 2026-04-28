from deliverable import factorial


def test_factorial_zero_is_one() -> None:
    assert factorial(0) == 1


def test_factorial_small_values() -> None:
    assert factorial(1) == 1
    assert factorial(3) == 6
    assert factorial(5) == 120


def test_factorial_larger_value() -> None:
    assert factorial(10) == 3628800
