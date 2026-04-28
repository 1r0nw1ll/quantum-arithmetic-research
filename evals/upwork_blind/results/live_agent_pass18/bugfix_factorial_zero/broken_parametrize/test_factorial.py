import pytest

from factorial import factorial


CASES = [
    (0, 1, "zero", "Defines the corrected base case: 0! is 1."),
    (1, 1, "identity", "Confirms the multiplicative identity case."),
    (2, 2, "two", "Checks the smallest non-trivial product."),
    (3, 6, "three", "Verifies a short multiplication chain."),
    (4, 24, "four", "Covers another small composite input."),
    (5, 120, "five", "Exercises a commonly memorized factorial."),
    (6, 720, "six", "Extends coverage beyond very small values."),
    (7, 5040, "seven", "Checks a larger product with more iterations."),
    (8, 40320, "eight", "Provides a mid-range regression case."),
    (10, 3628800, "ten", "Validates a larger, standard benchmark input."),
]


@pytest.mark.parametrize(
    "n, expected, _label, _note",
    CASES,
    ids=[label for _, _, label, _ in CASES],
)
def test_factorial_values(n, expected, _label, _note):
    assert factorial(n) == expected
