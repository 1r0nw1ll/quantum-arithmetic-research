# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.IV Primeness, Parity and Shape: "
    "C=2be+2e^2; C^2=4E^2+4EF; G has no prime factor <5; H and I have no prime factor <7; "
    "F is 5-par iff a,b same quaternary parity); "
    "Theorem NT: 'gnomon', 'rectangle', 'prime', 'shape' are observer projection labels for "
    "integer arithmetic; no float state, no QA orbit evolution"
)

"""
Cert [361] — QA Pyth-1 Primeness, Parity and Shape (Ch.IV)

Source: Iverson (1993) Pythagorean Arithmetic Vol I, Chapter IV pp.42-49
  p.44: 'C must satisfy the formula: C = 2be + 2e^2.'
        'C^2 must satisfy the formula C^2 = 4E^2 + 4EF.'
  p.41: 'The value of G must always be a 5-par number... It is also often a 5-pent number
         and can have no divisor smaller than 5.'
  p.41-42: 'H and I... can have no divisor less than 7.'
  p.44: 'F will be either a 3-par or a 5-par number... These two conditions occur when a
         and b are of the same, or the opposite, quaternary parity.'

Five claims:
  C1: C = 2be + 2e^2 for all prime pairs; and C^2 = 4E^2 + 4EF
      (proof: C=2de=2(b+e)e=2be+2e^2; C^2=4d^2e^2=4e^2(e^2+ab)=4E^2+4EF since ab=d^2-e^2)
  C2: F^2 = G^2 - C^2 (complement of Pythagorean theorem); F=(d-e)(d+e)=d^2-e^2 (algebraic)
  C3: G has no prime factor < 5 for all prime Pythagorean pairs;
      proof: G≡1(mod 4) → 2∤G; gcd(d,e)=1 → impossible for 3|d and 3|e simultaneously → 3∤G
  C4: H and I have no prime factor < 7;
      proof: H=C+F=even+odd=odd → 2∤H,I; for p∈{3,5}: 2 is not a QR mod p → (t±1)^2≢2(mod p)
      where t=d/e, so p∤(C±F); verified numerically for all pairs (b,e)<=35
  C5: F is 5-par iff a≡b(mod 4) (same quaternary parity); F is 3-par iff a≢b(mod 4);
      proof: a,b both odd; (4j+r)(4k+r)≡r^2≡1(mod 4); (4j+r)(4k+s)≡rs≡3(mod 4) for r≠s∈{1,3}
"""

from math import gcd


def _prime_pairs(max_b: int, max_e: int):
    """Yield (b, e, d, a) with b odd, gcd(b,e)=1."""
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) == 1:
                d = b + e
                a = d + e
                yield b, e, d, a


def _min_prime_factor(n: int) -> int:
    """Return smallest prime factor of n, or n if prime (n >= 2)."""
    if n <= 1:
        return n
    if n % 2 == 0:
        return 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return i
        i += 2
    return n


def check_c1() -> tuple[bool, str]:
    """C = 2be + 2e^2 and C^2 = 4E^2 + 4EF for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(25, 25):
        C = 2 * d * e
        E = e * e
        F = a * b
        # C = 2be + 2e^2 (substituting d = b+e)
        C_form = 2 * b * e + 2 * e * e
        assert C == C_form, f"C={C} != 2be+2e^2={C_form} at b={b},e={e}"
        # C^2 = 4E^2 + 4EF
        # Proof: C^2=4d^2e^2=4e^2(e^2+ab)=4E^2+4EF since ab=F and e^2=E
        lhs = C * C
        rhs = 4 * E * E + 4 * E * F
        assert lhs == rhs, f"C^2={lhs} != 4E^2+4EF={rhs} at b={b},e={e}"
        # Verify the proof: ab = d^2 - e^2
        assert F == d * d - e * e, f"ab={F} != d^2-e^2={d*d-e*e}"
        count += 1
    return True, (
        f"C=2be+2e^2 and C^2=4E^2+4EF verified for all {count} prime pairs (b,e)<=25; "
        f"proof: C=2(b+e)e=2be+2e^2; C^2=4d^2e^2=4e^2(e^2+ab)=4E^2+4EF (since F=ab=d^2-e^2) ✓"
    )


def check_c2() -> tuple[bool, str]:
    """F^2 = G^2 - C^2 and F = d^2 - e^2 for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(25, 25):
        C = 2 * d * e
        F = a * b
        G = d * d + e * e
        # C^2 + F^2 = G^2 (Pythagorean theorem)
        assert C * C + F * F == G * G, f"Pythagorean theorem fails at b={b},e={e}"
        # F^2 = G^2 - C^2
        assert F * F == G * G - C * C
        # F = d^2 - e^2
        assert F == d * d - e * e, f"F={F} != d^2-e^2={d*d-e*e}"
        # F = (d-e)(d+e) = b*a ✓
        assert F == (d - e) * (d + e) == b * a
        count += 1
    return True, (
        f"F^2=G^2-C^2 and F=d^2-e^2=(d-e)(d+e)=ab verified for {count} prime pairs (b,e)<=25; "
        f"equivalent to Pythagorean theorem C^2+F^2=G^2 ✓"
    )


def check_c3() -> tuple[bool, str]:
    """G has no prime factor < 5 for all prime Pythagorean pairs."""
    count = 0
    min_factor_found = 100
    for b, e, d, a in _prime_pairs(35, 35):
        G = d * d + e * e
        mpf = _min_prime_factor(G)
        assert mpf >= 5, f"G={G} has prime factor {mpf} < 5 at b={b},e={e}"
        if mpf < min_factor_found:
            min_factor_found = mpf
        count += 1
    # Algebraic proof:
    # 2∤G: G=d^2+e^2 ≡ 1(mod 4) (from cert [360] C1), so G is odd ✓
    # 3∤G: 3|G=d^2+e^2 would require 3|d AND 3|e (since QRs mod 3 are {0,1},
    #       d^2+e^2≡0 mod 3 only if both d^2≡0 and e^2≡0, i.e., 3|d and 3|e).
    #       But gcd(d,e)=1 since gcd(b,e)=1 and d=b+e → impossible. ✓
    return True, (
        f"G has no prime factor < 5 for all {count} pairs (b,e)<=35; "
        f"smallest prime factor found = {min_factor_found} (always ≥5); "
        f"proof: 2∤G (G≡1 mod 4); 3∤G (3|d and 3|e impossible since gcd(d,e)=1) ✓"
    )


def check_c4() -> tuple[bool, str]:
    """H and I have no prime factor < 7."""
    count = 0
    min_H_factor = 100
    min_I_factor = 100
    for b, e, d, a in _prime_pairs(35, 35):
        C = 2 * d * e
        F = a * b
        H = C + F
        I = abs(C - F)
        # H and I are always odd (C even, F odd)
        assert H % 2 == 1, f"H={H} is even at b={b},e={e}"
        assert I % 2 == 1, f"I={I} is even at b={b},e={e}"
        # No prime factor < 7
        if H > 1:
            mpf_H = _min_prime_factor(H)
            assert mpf_H >= 7, f"H={H} has prime factor {mpf_H} < 7 at b={b},e={e}"
            if mpf_H < min_H_factor:
                min_H_factor = mpf_H
        if I > 1:
            mpf_I = _min_prime_factor(I)
            assert mpf_I >= 7, f"I={I} has prime factor {mpf_I} < 7 at b={b},e={e}"
            if mpf_I < min_I_factor:
                min_I_factor = mpf_I
        count += 1
    # Proof outline:
    # 2∤H,I: C is even (4-par), F is odd → H=C+F is odd; I=|C-F| is odd ✓
    # 3∤H,I: C/F = 2t/(t^2-1) where t=d/e; for 3|(C+F): (t+1)^2≡2(mod 3), impossible since 2∉QR(3)
    # 5∤H,I: same argument mod 5: (t+1)^2≡2(mod 5), impossible since 2∉QR(5)
    return True, (
        f"H and I have no prime factor < 7 for {count} pairs (b,e)<=35; "
        f"min H-factor={min_H_factor}, min I-factor={min_I_factor} (both ≥7); "
        f"proof: H,I odd (C even, F odd); for p∈{{3,5}}: 2∉QR(mod p) → p∤(C±F) ✓"
    )


def check_c5() -> tuple[bool, str]:
    """F is 5-par iff a≡b(mod 4); F is 3-par iff a≢b(mod 4)."""
    count_5par = 0
    count_3par = 0
    for b, e, d, a in _prime_pairs(25, 25):
        F = a * b
        # a and b are both odd (verified in cert [360] C3)
        assert a % 2 == 1 and b % 2 == 1
        same_quat = (a % 4 == b % 4)  # same quaternary parity
        F_mod4 = F % 4
        if same_quat:
            # both 1-par: (4j+1)(4k+1)≡1(mod 4)=5-par
            # both 3-par: (4j+3)(4k+3)≡9≡1(mod 4)=5-par
            assert F_mod4 == 1, f"F={F} not 5-par when a≡b(mod 4) at b={b},e={e}"
            count_5par += 1
        else:
            # one 1-par, one 3-par: product ≡1×3=3(mod 4)=3-par
            assert F_mod4 == 3, f"F={F} not 3-par when a≢b(mod 4) at b={b},e={e}"
            count_3par += 1
    return True, (
        f"F par-class rule verified for all 268 prime pairs (b,e)<=25: "
        f"{count_5par} have F 5-par (a≡b mod 4); {count_3par} have F 3-par (a≢b mod 4); "
        f"proof: both same-class odd: product≡1(mod 4)=5-par; different class: product≡3(mod 4)=3-par ✓"
    )


def main() -> None:
    checks = [check_c1, check_c2, check_c3, check_c4, check_c5]
    passed = 0
    for fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fn.__name__}: {msg}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise RuntimeError(f"cert [361] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
