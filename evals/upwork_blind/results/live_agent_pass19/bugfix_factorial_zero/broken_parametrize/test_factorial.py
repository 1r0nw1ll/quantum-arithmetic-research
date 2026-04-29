from factorial import factorial
import pytest

@pytest.mark.parametrize(
    "input,expected,label,note",
    [
        (0, 1, "base-case", "0! should be 1"),
        (1, 1, "base-case", "1! should be 1"),
        (2, 2, "small-n", "2! = 2"),
        (3, 6, "small-n", "3! = 6"),
        (4, 24, "small-n", "4! = 24"),
        (5, 120, "small-n", "5! = 120"),
        (6, 720, "medium-n", "6! = 720"),
        (7, 5040, "medium-n", "7! = 5040"),
        (8, 40320, "medium-n", "8! = 40320"),
        (9, 362880, "medium-n", "9! = 362880"),
        (10, 3628800, "medium-n", "10! = 3628800"),
        (12, 479001600, "larger-n", "12! = 479001600"),
        (15, 1307674368000, "larger-n", "15! = 1307674368000"),
        (20, 2432902008176640000, "very-large-n", "20! = 2432902008176640000"),
    ],
)
def test_factorial(input, expected, **metadata):
    assert factorial(input) == expected
