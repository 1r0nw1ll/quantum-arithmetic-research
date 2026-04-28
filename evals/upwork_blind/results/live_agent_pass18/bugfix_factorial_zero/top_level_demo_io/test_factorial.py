from factorial import factorial


def test_factorial_zero():
    assert factorial(0) == 1


def test_factorial_small_values():
    assert factorial(1) == 1
    assert factorial(3) == 6
    assert factorial(5) == 120


def test_factorial_larger_value():
    assert factorial(8) == 40320
