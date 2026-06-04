#!/usr/bin/env python3
"""
QA Synchronous Harmonics Ceiling Cert [318] -- validator

Primary source:
  Iverson, B. (1993) Quantum Arithmetic Book 2 (QA-2), ITAM, Portland,
    ISBN 1-883401-07-0, Chapter 6: "Synchronous Harmonics", pp.84-99.

Companion cert: [147] QA Synchronous Harmonics -- covers wavelet sync at
  product, par interference rules, QN divisible by 6 (pp.65-83).

This cert covers the CEILING of the QA number system, which Iverson develops
in the final third of Ch.6. The key integer: 5040 = 7! = 2^4 * 3^2 * 5 * 7.

Iverson's statements (QA-2 Ch.6):
  "5040 ... has more divisors than any other integer below it." (p.84)
  "It comes from 1 x 2 x 3 x 4 x 5 x 6 x 7 = 5040. It is divisible by all
   composite integers from 8 through 2520 which have all of their factors
   being 2 through 10." (p.84)
  "In order to jump from one Quantum Number to the next, requires a change
   in 6 units." (p.88)  [because 2 and 3 are in every QN]
  "If the student takes these four prime factors, 2,3,5,&7, and adds one,
   two or three larger prime factors to them, the smallest number we can
   achieve for a product is: 2*3*5*7*11 = 2310 units." (p.75)
  "These fractions represent the harmonic points of a wavelength of 5040
   units." (p.85)  [the fractions k/d * 5040 for d in {2..7}]

SAMEKH (open frontier per Iverson):
  Iverson proposes that somewhere above 5040 there exists a number that
  "divides into prime factors in more than one way" (p.83), contradicting
  Euclid's unique factorization. He calls this hypothetical number "Samekh"
  and states it is bounded between ~10,000 and ~700,000 (pp.56,83). This is
  NOT certified here -- it is flagged as explicitly open per primary source.

Five claims:
  C1  5040 extremal divisor count: 5040=2^4*3^2*5*7 has exactly tau(5040)=60
      divisors; no positive integer less than 5040 has as many (max below=48);
      5040 is divisible by every integer from 1 to 10 inclusive; 5039 is prime;
      5041=71^2 has exactly 3 divisors (the sharpest possible drop after 5040).
  C2  5040 harmonic fractions: all 21 fractions {k/d * 5040} for
      d in {2,3,4,5,6,7} and k in {1,...,d-1} are positive integers; they
      range from 720 to 4320; these are the "harmonic points of a wavelength
      of 5040 units" per Iverson.
  C3  Minimum quantum wavelengths: the smallest integer with exactly 5
      distinct prime factors containing both 2 and 3 is 2310=2*3*5*7*11
      (primorial of first 5 primes); the smallest with 7 is 510510=
      2*3*5*7*11*13*17; both are divisible by 6 (since they contain 2 and 3).
  C4  5040-neighbourhood gap: the integers 5041..5045 are each not divisible
      by 6 (a necessary condition for a quantum wavelength), so the nearest
      quantum-eligible number above 5040 is at least 5046; 5046=2*3*29^2 has
      only 3 distinct prime factors (below the minimum 5), so the actual gap
      to the next minimum-5-factor QW candidate above 5040 is 5040->5040+6k.
  C5  Theorem NT and Samekh: 5040 harmonic fractions, tau(5040)=60, and
      minimum QW wavelengths 2310/510510 are exact integer claims; the
      continuous-frequency model (sine waves, spectral lines) is the observer
      projection; Samekh (unique-factorization violation above 5040) is
      explicitly open per Iverson QA-2 p.83.
"""

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic: divisor count by trial division; "
    "primality test; harmonic fraction products exact integer division; "
    "factorization for distinct-prime-count checks; "
    "Theorem NT: frequency spectra observer; Samekh open per primary source"
)

import json
import sys
from math import gcd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def factorize(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def num_divisors(n):
    t = 1
    for e in factorize(n).values():
        t *= (e + 1)
    return t


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


N = 5040   # the QA ceiling number

checks = {}
passed = 0
failed = 0


# C1 -- 5040 extremal divisor count
tau_N = num_divisors(N)
f_N   = factorize(N)
div_1_to_10 = all(N % k == 0 for k in range(1, 11))
max_tau_below = max(num_divisors(k) for k in range(1, N))  # ~1s
ok_c1 = (
    f_N == {2: 4, 3: 2, 5: 1, 7: 1}       # 5040 = 2^4 * 3^2 * 5 * 7
    and tau_N == 60
    and div_1_to_10
    and max_tau_below < tau_N              # 5040 strictly maximal
    and factorize(5041) == {71: 2}         # 5041 = 71^2
    and num_divisors(5041) == 3
    and is_prime(5039)
)
checks["C1_5040_tau60_extremal_div1to10_5041eq71sq_5039prime"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1


# C2 -- 5040 harmonic fractions: 21 values, all positive integers, 720..4320
harmonic_vals = []
for d in [2, 3, 4, 5, 6, 7]:
    for k in range(1, d):
        assert N * k % d == 0, f"NOT INTEGER: {k}/{d} * {N}"
        harmonic_vals.append(N * k // d)

ok_c2 = (
    len(harmonic_vals) == 21
    and min(harmonic_vals) == N // 7       # 720
    and max(harmonic_vals) == N * 6 // 7  # 4320
    and all(isinstance(v, int) and v > 0 for v in harmonic_vals)
    and N // 7 == 720
    and N * 6 // 7 == 4320
)
checks["C2_5040_harmonic_fractions_21_integers_720_to_4320"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1


# C3 -- Minimum QW: 2310 (5 primes) and 510510 (7 primes), both contain 2 & 3
MIN_QW_5 = 2 * 3 * 5 * 7 * 11          # = 2310
MIN_QW_7 = 2 * 3 * 5 * 7 * 11 * 13 * 17  # = 510510
f5 = factorize(MIN_QW_5)
f7 = factorize(MIN_QW_7)

ok_c3 = (
    MIN_QW_5 == 2310
    and MIN_QW_7 == 510510
    and len(f5) == 5 and 2 in f5 and 3 in f5
    and len(f7) == 7 and 2 in f7 and 3 in f7
    and MIN_QW_5 % 6 == 0
    and MIN_QW_7 % 6 == 0
    # Primorial structure: each is the product of the first k primes
    and f5 == {2: 1, 3: 1, 5: 1, 7: 1, 11: 1}
    and f7 == {2: 1, 3: 1, 5: 1, 7: 1, 11: 1, 13: 1, 17: 1}
)
checks["C3_min_qw_2310_five_primes_510510_seven_primes_both_div6"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1


# C4 -- 5040-neighbourhood: 5041..5045 not div 6; 5046 only 3 distinct primes
ok_c4 = (
    all(k % 6 != 0 for k in range(5041, 5046))   # none divisible by 6
    and 5046 % 6 == 0                              # 5046 IS div by 6
    and len(factorize(5046)) == 3                  # but only 3 distinct primes
    and factorize(5046) == {2: 1, 3: 1, 29: 2}    # = 2 * 3 * 29^2
)
checks["C4_5040_gap_5041to5045_not_div6_5046_only_3_primes"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1


# C5 -- Theorem NT: all quantities are integers; Samekh open (not computed)
ok_c5 = (
    isinstance(tau_N, int)
    and isinstance(max_tau_below, int)
    and isinstance(MIN_QW_5, int)
    and isinstance(MIN_QW_7, int)
    and all(isinstance(v, int) for v in harmonic_vals)
    # No float entered any computation above
    and tau_N == 60
    and max_tau_below == 48
    # Samekh: not asserted here -- it is open per Iverson QA-2 p.83
    # "Samekh states: There is a number which divides into prime factors
    #  in more than one way" -- flagged open, NOT a cert claim
)
checks["C5_theorem_nt_all_int_no_float_samekh_open"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1


print(json.dumps(checks, indent=2))
print(f"\nTotal: {passed} PASS, {failed} FAIL")
if failed:
    sys.exit(1)
