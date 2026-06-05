# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (gnomon area identities, "
    "F=d^2-e^2, C^2=4E^2+4EF); no QA state evolution; Theorem NT: "
    "'gnomon', 'shape', 'rectangle' are observer-layer labels on integer "
    "algebraic identities; all arithmetic exact integer, no float"
)

"""
Cert [338] — QA Pythagorean Gnomon and Square Identities

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, pp.37-39, 43-46
  "F = d^2 - e^2" (from F=ab=d^2-e^2, since b=d-e and a=d+e)
  "C must satisfy the formula: C = 2be + 2e^2"
  "C^2 must satisfy the formula C^2 = 4E^2 + 4EF"
  "A and B will be of odd parity in the binary sense. Since they are
   squares of odd numbers they will be 5-par (4n+1) numbers."
  "In quaternary parity, the odd member of D and E will be 5-par (4n+1),
   and the even member will be 4-par (4n)."

Key identities:
  F = d^2 - e^2 = (d+e)(d-e) = a*b     [since a=d+e, b=d-e]
  C = 2de = 2be + 2e^2 = 2e(b+e) = 2ed  [expanded form]
  C^2 = 4*E^2 + 4*E*F  where E=e^2      [gnomon identity]
  A = a^2 is 5-par; B = b^2 is 5-par    [squares of odd integers]
  D = d^2 and E = e^2 have opposite par-types (4-par vs 5-par)

Five claims certified via integer arithmetic.
"""

from math import gcd


def _params(b: int, e: int) -> dict:
    """Compute parameters from bead numbers."""
    d = b + e
    a = b + 2 * e
    C = 2 * d * e
    F = a * b
    G = d * d + e * e
    A = a * a
    B = b * b
    D = d * d
    E = e * e
    return dict(b=b, e=e, d=d, a=a, C=C, F=F, G=G, A=A, B=B, D=D, E=E)


def _par_type(n: int) -> int:
    """Return 4 if n≡0 (mod 4), 5 if n≡1 (mod 4), else 0."""
    r = n % 4
    if r == 0:
        return 4
    if r == 1:
        return 5
    return r  # 2 or 3


def check_c1() -> tuple[bool, str]:
    """F = d^2 - e^2 = (d+e)(d-e) = a*b; since b=d-e and a=d+e."""
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        d, a, F, D, E = p['d'], p['a'], p['F'], p['D'], p['E']
        # b = d - e
        assert p['b'] == d - e, f"({b},{e}): b != d-e"
        # a = d + e
        assert a == d + e, f"({b},{e}): a != d+e"
        # F = d^2 - e^2 = (d+e)(d-e) = ab
        assert D - E == a * b, f"({b},{e}): d^2-e^2={D-E} != ab={a*b}"
        assert F == D - E, f"({b},{e}): F={F} != d^2-e^2={D-E}"
    return True, f"F=d^2-e^2=(d+e)(d-e)=ab since b=d-e and a=d+e; verified {len(cases)} pairs"


def check_c2() -> tuple[bool, str]:
    """C = 2de = 2be + 2e^2 = 2e(b+e); expanded form."""
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        d, C = p['d'], p['C']
        # Two forms of C
        form1 = 2 * d * e          # C = 2de
        form2 = 2 * b * e + 2 * e * e   # C = 2be + 2e^2
        form3 = 2 * e * (b + e)    # C = 2e(b+e)
        assert form1 == form2 == form3 == C, (
            f"({b},{e}): C={C}; 2de={form1}; 2be+2e^2={form2}; 2e(b+e)={form3}"
        )
    return True, f"C=2de=2be+2e^2=2e(b+e) three-form equivalence verified"


def check_c3() -> tuple[bool, str]:
    """C^2 = 4*E^2 + 4*E*F (gnomon identity: C^2 = 4e^4 + 4e^2*F)."""
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        C, F, E = p['C'], p['F'], p['E']
        gnomon = 4 * E * E + 4 * E * F
        assert C * C == gnomon, f"({b},{e}): C^2={C*C} != 4E^2+4EF={gnomon}"
        # Verify algebraically: C^2 = (2de)^2 = 4d^2*e^2 = 4e^2*(d^2) = 4E*D
        # And 4E^2+4EF = 4e^2*(e^2+F) = 4e^2*(e^2+d^2-e^2) = 4e^2*d^2 = C^2 ✓
    return True, f"C^2=4E^2+4EF (gnomon identity) verified for {len(cases)} pairs"


def check_c4() -> tuple[bool, str]:
    """A=a^2 and B=b^2 are both 5-par (≡1 mod 4); squares of odd integers."""
    # Any odd integer squared ≡ 1 (mod 4) → 5-par
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(11,2)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        a, A, B = p['a'], p['A'], p['B']
        assert b % 2 == 1, f"b={b} should be odd"
        assert a % 2 == 1, f"a={a} should be odd (b+2e = odd + even)"
        assert A % 4 == 1, f"({b},{e}): A=a^2={A} not ≡1 (mod 4)"
        assert B % 4 == 1, f"({b},{e}): B=b^2={B} not ≡1 (mod 4)"
        assert _par_type(A) == 5 and _par_type(B) == 5, f"A or B not 5-par"
    return True, f"A=a^2 and B=b^2 are 5-par (≡1 mod 4) for all {len(cases)} pairs"


def check_c5() -> tuple[bool, str]:
    """D=d^2 and E=e^2 have opposite par-types (one 4-par, one 5-par)."""
    # Since gcd(b,e)=1 with b odd: d=b+e and e have opposite parities
    # If e even: e^2≡0 (mod 4) → E is 4-par; d=b+e=odd+even=odd → d^2≡1 (mod 4) → D is 5-par
    # If e odd: e^2≡1 (mod 4) → E is 5-par; d=b+e=odd+odd=even → d^2≡0 (mod 4) → D is 4-par
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(11,2),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        d, D, E = p['d'], p['D'], p['E']
        pt_D = _par_type(D)
        pt_E = _par_type(E)
        assert pt_D in (4, 5) and pt_E in (4, 5), (
            f"({b},{e}): D par={pt_D}, E par={pt_E} not in (4,5)"
        )
        assert pt_D != pt_E, f"({b},{e}): D and E have same par-type {pt_D}"
    return True, f"D=d^2 and E=e^2 always have opposite par-types (4-par vs 5-par); verified {len(cases)} pairs"


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
        raise RuntimeError(f"cert [338] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
