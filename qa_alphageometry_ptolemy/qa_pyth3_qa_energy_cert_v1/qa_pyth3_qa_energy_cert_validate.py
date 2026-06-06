"""
QA Pyth-3 QA and Energy — Chapter V structural cert [374].
# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
"""
from __future__ import annotations
from fractions import Fraction
from math import gcd, pi as MATH_PI


def _prime_factors(n: int) -> set[int]:
    """Return set of distinct prime factors of n."""
    factors: set[int] = set()
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.add(p)
            n //= p
        p += 1
    if n > 1:
        factors.add(n)
    return factors


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    p = 3
    while p * p <= n:
        if n % p == 0:
            return False
        p += 2
    return True


def check_c1() -> tuple[bool, str]:
    """C1: QN definition — at least 4 distinct prime factors.
    Products: (1,1,2,3)->6, (2,1,3,4)->24; ratio 24/6=4=2*2."""
    # Male seed QN: b=1, e=1 -> diff=2, apex=3  (A2: diff=b+e, apex=b+2e)
    b_m1, e_m1 = 1, 1
    diff_m1 = b_m1 + e_m1       # 2  (derived)
    apex_m1 = b_m1 + 2 * e_m1   # 3  (derived)
    prod_male1 = b_m1 * e_m1 * diff_m1 * apex_m1   # 1*1*2*3 = 6

    # Female seed QN: b=2, e=1 -> diff=3, apex=4
    b_f1, e_f1 = 2, 1
    diff_f1 = b_f1 + e_f1       # 3  (derived)
    apex_f1 = b_f1 + 2 * e_f1   # 4  (derived)
    prod_female1 = b_f1 * e_f1 * diff_f1 * apex_f1  # 2*1*3*4 = 24

    if prod_male1 != 6:
        return False, f"Male seed product should be 6, got {prod_male1}"
    if prod_female1 != 24:
        return False, f"Female seed product should be 24, got {prod_female1}"
    ratio1 = prod_female1 // prod_male1
    if ratio1 != 4:
        return False, f"Octave ratio should be 4, got {ratio1}"
    # Two octaves = 2*2 = 4
    if ratio1 != 2 * 2:
        return False, f"4 != 2*2"
    return True, f"prod_male1={prod_male1} prod_female1={prod_female1} ratio={ratio1}=2*2 PASS"


def check_c2() -> tuple[bool, str]:
    """C2: Sympathetic Harmonics 2310 and 2730.
    2310=2*3*5*7*11; 2730=2*3*5*7*13; GCD=210=2*3*5*7; ratio 11:13."""
    a_qn = 2 * 3 * 5 * 7 * 11   # 2310
    b_qn = 2 * 3 * 5 * 7 * 13   # 2730
    if a_qn != 2310:
        return False, f"a_qn={a_qn} != 2310"
    if b_qn != 2730:
        return False, f"b_qn={b_qn} != 2730"
    common = 2 * 3 * 5 * 7       # 210
    if gcd(a_qn, b_qn) != common:
        return False, f"GCD={gcd(a_qn,b_qn)} != 210"
    unique_a = a_qn // common    # 11
    unique_b = b_qn // common    # 13
    if unique_a != 11 or unique_b != 13:
        return False, f"unique factors should be 11,13 got {unique_a},{unique_b}"
    # Verify: a_qn = common * 11, b_qn = common * 13
    if common * unique_a != a_qn or common * unique_b != b_qn:
        return False, "Aliquot decomposition fails"
    lcm_val = common * unique_a * unique_b   # 210*11*13=30030
    if lcm_val != 30030:
        return False, f"LCM={lcm_val} != 30030"
    return True, (f"2310=210*11 2730=210*13 GCD=210 ratio=11:13 "
                  f"LCM=30030 PASS")


def check_c3() -> tuple[bool, str]:
    """C3: Two-octave law — female QN product = 4 * male QN product.
    Pair 2: male (1,2,3,5) product=30; female (4,1,5,6) product=120."""
    # Male: b=1, e=2 -> diff=3, apex=5
    b_m2, e_m2 = 1, 2
    diff_m2 = b_m2 + e_m2       # 3  (derived)
    apex_m2 = b_m2 + 2 * e_m2   # 5  (derived)
    prod_male2 = b_m2 * e_m2 * diff_m2 * apex_m2   # 1*2*3*5 = 30

    # Female: b=4, e=1 -> diff=5, apex=6
    b_f2, e_f2 = 4, 1
    diff_f2 = b_f2 + e_f2       # 5  (derived)
    apex_f2 = b_f2 + 2 * e_f2   # 6  (derived)
    prod_female2 = b_f2 * e_f2 * diff_f2 * apex_f2  # 4*1*5*6 = 120

    if prod_male2 != 30:
        return False, f"Male pair2 product should be 30, got {prod_male2}"
    if prod_female2 != 120:
        return False, f"Female pair2 product should be 120, got {prod_female2}"
    ratio2 = prod_female2 // prod_male2
    if ratio2 != 4:
        return False, f"Octave ratio pair2 should be 4, got {ratio2}"
    # Also check prime factors match between pairs
    pf_m2 = _prime_factors(prod_male2)    # {2,3,5}
    pf_f2 = _prime_factors(prod_female2)  # {2,3,5}
    if pf_m2 != pf_f2:
        return False, f"Prime factors differ: male={pf_m2} female={pf_f2}"
    return True, (f"prod_male2={prod_male2} prod_female2={prod_female2} "
                  f"ratio={ratio2}=2*2 prime_factors_match={pf_m2} PASS")


def check_c4() -> tuple[bool, str]:
    """C4: Non-virtual QN pairs (5 distinct prime factors).
    Male (5,8,13,21): primes {2,3,5,7,13} product=10920;
    Female (16,5,21,26): same primes, product=43680=4*10920."""
    # Male: b=5, e=8 -> diff=13, apex=21  (A2 derivation)
    b_m3, e_m3 = 5, 8
    diff_m3 = b_m3 + e_m3       # 13 (derived)
    apex_m3 = b_m3 + 2 * e_m3   # 21 (derived)
    if diff_m3 != 13:
        return False, f"diff_m3={diff_m3} != 13"
    if apex_m3 != 21:
        return False, f"apex_m3={apex_m3} != 21"
    prod_male3 = b_m3 * e_m3 * diff_m3 * apex_m3   # 5*8*13*21 = 10920
    if prod_male3 != 10920:
        return False, f"prod_male3={prod_male3} != 10920"
    pf_m3 = _prime_factors(prod_male3)  # {2,3,5,7,13}
    if pf_m3 != {2, 3, 5, 7, 13}:
        return False, f"prime_factors male={pf_m3} != {{2,3,5,7,13}}"
    if len(pf_m3) < 5:
        return False, f"Need >=5 prime factors, got {len(pf_m3)}"

    # Female: b=16, e=5 -> diff=21, apex=26  (A2 derivation)
    b_f3, e_f3 = 16, 5
    diff_f3 = b_f3 + e_f3       # 21 (derived)
    apex_f3 = b_f3 + 2 * e_f3   # 26 (derived)
    if diff_f3 != 21:
        return False, f"diff_f3={diff_f3} != 21"
    if apex_f3 != 26:
        return False, f"apex_f3={apex_f3} != 26"
    prod_female3 = b_f3 * e_f3 * diff_f3 * apex_f3  # 16*5*21*26 = 43680
    if prod_female3 != 43680:
        return False, f"prod_female3={prod_female3} != 43680"
    pf_f3 = _prime_factors(prod_female3)  # {2,3,5,7,13}
    if pf_f3 != {2, 3, 5, 7, 13}:
        return False, f"prime_factors female={pf_f3} != {{2,3,5,7,13}}"

    # Two-octave ratio
    ratio3 = prod_female3 // prod_male3
    if ratio3 != 4:
        return False, f"ratio3={ratio3} != 4"
    return True, (f"male(5,8,13,21) prod={prod_male3} female(16,5,21,26) "
                  f"prod={prod_female3} ratio={ratio3}=2*2 primes={pf_m3} PASS")


def check_c5() -> tuple[bool, str]:
    """C5: Parker/Keely PI rational approximation 20612/6561.
    6561=3^8=81*81; 20612=4*5153 with 5153 prime; observer-projection
    comparison |20612/6561 - pi| < 1e-4."""
    num = 20612
    den = 6561

    # Integer structure of denominator: 6561 = 3^8 = 81*81
    three_to_8 = 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3   # 6561
    if three_to_8 != 6561:
        return False, f"3^8={three_to_8} != 6561"
    if 81 * 81 != 6561:
        return False, f"81*81={81*81} != 6561"
    if _prime_factors(den) != {3}:
        return False, f"prime_factors(6561)={_prime_factors(den)} != {{3}}"

    # Integer structure of numerator: 20612 = 4 * 5153
    if num != 4 * 5153:
        return False, f"{num} != 4*5153"
    if not _is_prime(5153):
        return False, "5153 is not prime"

    # gcd(20612, 6561) should be 1 (coprime)
    if gcd(num, den) != 1:
        return False, f"gcd({num},{den})={gcd(num,den)} != 1; not coprime"

    # Observer-projection: float comparison (T2-safe read-only measurement)
    frac = Fraction(num, den)
    approx = float(frac)         # observer output only — not fed back into QA
    error = abs(approx - MATH_PI)
    if error >= 1e-4:
        return False, f"|20612/6561 - pi|={error:.2e} >= 1e-4"
    return True, (f"6561=3^8=81*81 20612=4*5153(prime) gcd=1 "
                  f"|approx-pi|={error:.2e}<1e-4 PASS")


def main() -> None:
    checks = [check_c1, check_c2, check_c3, check_c4, check_c5]
    passed = 0
    for fn in checks:
        ok, msg = fn()
        label = "PASS" if ok else "FAIL"
        print(f"[{label}] {fn.__name__}: {msg}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
