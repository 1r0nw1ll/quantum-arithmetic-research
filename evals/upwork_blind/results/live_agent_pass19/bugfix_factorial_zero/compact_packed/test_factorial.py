from factorial import factorial

def test_zero():
    assert factorial(0) == 1


def test_small():
    assert factorial(1) == 1
    assert factorial(3) == 6
    assert factorial(5) == 120


def test_large():
    assert factorial(10) == 3628800
