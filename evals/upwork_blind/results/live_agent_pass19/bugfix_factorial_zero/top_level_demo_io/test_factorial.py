import pytest

from factorial import factorial


def test_factorial_zero():
    assert factorial(0) == 1


def test_factorial_small():
    assert factorial(1) == 1
    assert factorial(3) == 6


def test_factorial_large():
    assert factorial(10) == 3628800
