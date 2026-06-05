# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1991) QA Vol I p.27 + Iverson (1993) QA-2 Ch.1 — primary source text, no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (gcd, modular parity checks); "
    "no QA state evolution; Theorem NT: 'male'/'female' QN and 'par-type' "
    "are observer-layer labels on integer arithmetic structure; "
    "all arithmetic exact integer, no float"
)

"""
Cert [333] — QA Female QN Derivation: First-Fourth Parity

Sources:
  QA-1 p.27 (Ch.2 Exercise 8):
    "the female number can be derived directly from the male number.
     It is done by doubling the two intermediate numbers of the male
     Quantum Number and placing them at the two ends."
  QA-2 Ch.1 "Doubling":
    "If the first integer is 2-par, the fourth integer will be 4-par.
     If the first integer is 4-par, then the fourth will be 2-par."

Male QN: (b, e, d, a) where d = b+e, a = b+2e, gcd(b,e)=1, b odd.
Female QN: double the two intermediate numbers (e, d) and place them
           at the two ends → (b_f, e_f, d_f, a_f) = (2e, b, a, 2d).

Derived coords verify:
  d_f = b_f + e_f = 2e + b = b+e = d ✓  (raw, NOT mod-reduced per A2)
  a_f = b_f + 2*e_f = 2e + 2b = 2(b+e) = 2d ✓

Parity rule (first-fourth swap):
  b_f = 2e: if e odd → 2e ≡ 2 (mod 4) → 2-par; if e even → 4-par.
  a_f = 2d = 2(b+e): since b odd and gcd(b,e)=1 → b and e have
    opposite parities (b odd, so e can be even or odd, but they're
    coprime). When e odd → b+e even → 2(b+e) ≡ 0 (mod 4) → 4-par.
    When e even → b+e odd → 2(b+e) ≡ 2 (mod 4) → 2-par.
  → b_f is 2-par iff a_f is 4-par; they always swap. QED.

Five claims certified via integer arithmetic.
"""

from math import gcd


def _par_type(n: int) -> int:
    """Return 2 if n≡2 (mod 4), 4 if n≡0 (mod 4), else 0 (odd)."""
    if n % 4 == 0:
        return 4
    if n % 2 == 0:
        return 2
    return 0


def check_c1() -> tuple[bool, str]:
    """Female QN derivation formula: (2e, b, a, 2d) from male (b, e, d, a)."""
    # Verify raw-derived coords: d_f = 2e+b and a_f = 2e+2b = 2(b+e)
    male_cases = [
        (1, 2),   # d=3, a=5  → female (4,1,5,6)
        (1, 4),   # d=5, a=9  → female (8,1,9,10)
        (3, 2),   # d=5, a=7  → female (4,3,7,10)
        (1, 6),   # d=7, a=13 → female (12,1,13,14)
        (3, 4),   # d=7, a=11 → female (8,3,11,14)
        (5, 2),   # d=7, a=9  → female (4,5,9,14)
        (5, 4),   # d=9, a=13 → female (8,5,13,18)
        (7, 2),   # d=9, a=11 → female (4,7,11,18)
    ]
    for b, e in male_cases:
        assert gcd(b, e) == 1, f"({b},{e}) not coprime"
        assert b % 2 == 1, f"b={b} must be odd for male QN"
        d = b + e          # raw A2 — no mod reduction
        a = b + 2 * e
        bf, ef, df_raw, af_raw = 2 * e, b, a, 2 * d
        # Verify internal consistency: d_f = bf + ef, a_f = bf + 2*ef
        assert bf + ef == df_raw, f"({b},{e}): d_f mismatch {bf+ef} != {df_raw}"
        assert bf + 2 * ef == af_raw, f"({b},{e}): a_f mismatch {bf+2*ef} != {af_raw}"
    return True, f"Female (2e,b,a,2d) derivation verified for {len(male_cases)} male pairs"


def check_c2() -> tuple[bool, str]:
    """Parity argument: gcd(e,d)=gcd(b,e)=1; e and d have different parities."""
    # For male (b odd, gcd(b,e)=1): gcd(e,d) = gcd(e, b+e) = gcd(e,b) = 1
    # Coprimeness → e and d cannot both be even; b odd → b+e parity flips with e
    for b in range(1, 18, 2):   # odd b
        for e in range(1, 18):
            if gcd(b, e) != 1:
                continue
            d = b + e
            # gcd(e,d) must equal gcd(b,e) = 1
            assert gcd(e, d) == 1, f"({b},{e}): gcd(e,d)={gcd(e,d)} != 1"
            # parity of e and d differ (since b is odd)
            assert (e % 2) != (d % 2), f"({b},{e}): e and d same parity"
    return True, "gcd(e,d)=1 and e,d have opposite parities for all male pairs tested"


def check_c3() -> tuple[bool, str]:
    """e odd → b_f=2e is 2-par, a_f=2d is 4-par."""
    odd_e_cases = [
        (1, 2),  # e=2 even; skip
        (1, 3),  # e=3 odd: bf=6 (2-par), af=2*4=8 (4-par)  wait, b must be odd
        # Pairs with b odd AND e odd:
    ]
    # Systematic: b odd, e odd, gcd(b,e)=1
    for b in range(1, 16, 2):
        for e in range(1, 16, 2):
            if gcd(b, e) != 1:
                continue
            d = b + e  # b odd + e odd = even
            bf = 2 * e  # 2 × odd = ≡2 (mod 4) when e≡1 or 3 (mod 4)?
            af = 2 * d  # 2 × even: depends on d/2 parity
            # 2 × odd: if odd ≡ 1 (mod 4) → 2 (mod 4); if ≡ 3 (mod 4) → 6 ≡ 2 (mod 4)
            # Either way 2*odd ≡ 2 (mod 4) → 2-par
            assert _par_type(bf) == 2, f"({b},{e}): bf={bf} should be 2-par"
            # d = b+e = odd+odd = even; 2d = 4*(d/2); d/2 = (b+e)/2
            # Since b≡1(mod2) and e≡1(mod2): d=b+e≡0(mod2); d/2 integer
            # 2d ≡ 0 (mod 4) → 4-par
            assert _par_type(af) == 4, f"({b},{e}): af={af} should be 4-par"
    return True, "e odd → b_f=2e is 2-par, a_f=2d is 4-par; verified all coprime odd pairs up to 15"


def check_c4() -> tuple[bool, str]:
    """e even (b odd) → b_f=2e is 4-par, a_f=2d is 2-par."""
    # b odd, e even, gcd(b,e)=1
    for b in range(1, 16, 2):
        for e in range(2, 16, 2):
            if gcd(b, e) != 1:
                continue
            d = b + e  # odd + even = odd
            bf = 2 * e  # 2 × even: e=2k → bf=4k → 0 (mod 4) → 4-par
            af = 2 * d  # 2 × odd → 2 (mod 4) → 2-par
            assert _par_type(bf) == 4, f"({b},{e}): bf={bf} should be 4-par"
            assert _par_type(af) == 2, f"({b},{e}): af={af} should be 2-par"
    return True, "e even → b_f=2e is 4-par, a_f=2d is 2-par; verified all coprime pairs up to 15"


def check_c5() -> tuple[bool, str]:
    """First-fourth parity swap holds for all male QNs; par-types are always opposite."""
    total = 0
    for b in range(1, 24, 2):
        for e in range(1, 24):
            if gcd(b, e) != 1:
                continue
            d = b + e
            bf = 2 * e
            af = 2 * d
            pt_b = _par_type(bf)
            pt_a = _par_type(af)
            # Both must be even par-types (2 or 4) and opposite
            assert pt_b in (2, 4), f"({b},{e}): b_f={bf} par={pt_b}, not 2 or 4"
            assert pt_a in (2, 4), f"({b},{e}): a_f={af} par={pt_a}, not 2 or 4"
            assert pt_b != pt_a, f"({b},{e}): b_f and a_f have same par-type {pt_b}"
            total += 1
    return True, f"First-fourth parity always opposite (2-par ↔ 4-par); verified {total} male QNs"


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
        raise RuntimeError(f"cert [333] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
