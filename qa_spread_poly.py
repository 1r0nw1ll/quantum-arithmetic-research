#!/usr/bin/env python3
"""
QA Spread Polynomials Module
Implements spread polynomial generator (Wildberger–Goh recursion)
Sn(s) = 2(1-2s)Sn-1 - Sn-2 + 2s, S0=0, S1=s
"""

import numpy as np

def spread_polynomials(s: float, N: int):
    """Return list [S0,...,SN] via recurrence (1)."""
    S = [0.0, s]
    for n in range(2, N+1):
        Sn = 2*(1-2*s)*S[-1] - S[-2] + 2*s
        S.append(Sn)
    return np.array(S)

def spread_periodicity(s: float, p: int, N: int=50):
    """Return sequence mod p and detected period m."""
    S = [0, s % p]
    for n in range(2, N+1):
        Sn = (2*(1-2*s)*S[-1] - S[-2] + 2*s) % p
        S.append(Sn)
    # Period detection (skip first 2)
    for m in range(3, N//2):
        if all((S[i] == S[i+m]) for i in range(N-m)):
            return m, S
    return None, S

def analyze_spread_periodicity(primes=None, s=1/3, N=36):
    """Analyze spread periodicity for given primes."""
    if primes is None:
        primes = [5, 7, 11, 13, 17, 19, 23, 29]
    periods = []
    for p in primes:
        m, S = spread_periodicity(s, p, N)
        periods.append({'p': p, 'period': m or 0})
    return periods