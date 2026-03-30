
# qa_fft.py
from sympy import symbols, Rational, simplify, cos, sin, pi

def qa_fft_transform(tuples, k):
    theta_k = Rational(2 * pi * k, 24)
    result = 0
    for b, e, d, a in tuples:
        result += a * cos(theta_k) + d * sin(theta_k)
    return simplify(result)
