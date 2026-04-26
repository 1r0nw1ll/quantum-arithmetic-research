from factorial import factorial

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_small():
    assert factorial(1) == 1
    assert factorial(5) == 120

def test_factorial_larger():
    assert factorial(10) == 3628800
