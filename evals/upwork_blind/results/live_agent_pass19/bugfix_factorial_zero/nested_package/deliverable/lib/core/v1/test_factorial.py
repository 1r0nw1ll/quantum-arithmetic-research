from deliverable.lib.core.v1 import factorial


def test_factorial_zero():
    assert factorial(0) == 1


def test_factorial_small():
    assert factorial(1) == 1
    assert factorial(5) == 120


def test_factorial_large():
    assert factorial(10) == 3628800
