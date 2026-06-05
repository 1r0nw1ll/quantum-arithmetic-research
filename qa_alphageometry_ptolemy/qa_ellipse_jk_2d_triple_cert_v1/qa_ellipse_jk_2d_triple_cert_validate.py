# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (ellipse parameter identities, "
    "2D triple decomposition); no QA state evolution; Theorem NT: 'ellipse', "
    "'major diameter', 'semimajor' are observer-layer labels on integer bead "
    "number relationships; all arithmetic exact integer, no float"
)

"""
Cert [337] — QA Ellipse J,K Parameters and 2D Triple Decomposition

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, pp.37-38, 43-44
  "J=D-C/2 and K=D+C/2. As related to the parametric numbers they are:
   J=bd and K=ad."

  "F+G = J+K = 2J+C = 2D (each equals 2d^2)"

Ellipse parameters J and K:
  J = bd  (semi-interval at left focus, J = D - C/2 = d^2 - de = db)
  K = ad  (semi-interval at right focus, K = D + C/2 = d^2 + de = da)

Three ways to express 2D = 2d^2:
  F + G  = 2d^2   (from cert [336] G+F=2d^2)
  J + K  = 2d^2   (bd + ad = d(a+b) = d*2d = 2d^2)
  2J + C = 2d^2   (2bd + 2de = 2d(b+e) = 2d^2)

Five claims certified via integer algebra.
"""

from math import gcd


def _bead_params(b: int, e: int) -> dict:
    """Compute all parameters from bead numbers (b,e)."""
    d = b + e
    a = b + 2 * e
    C = 2 * d * e
    F = a * b
    G = d * d + e * e
    D = d * d
    J = b * d
    K = a * d
    return dict(b=b, e=e, d=d, a=a, C=C, F=F, G=G, D=D, J=J, K=K)


def check_c1() -> tuple[bool, str]:
    """J=bd and K=ad; also J=D-C/2 and K=D+C/2 (where D=d^2, C=2de)."""
    test_cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(1,8),(3,8)]
    for b, e in test_cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _bead_params(b, e)
        d, a, C, D, J, K = p['d'], p['a'], p['C'], p['D'], p['J'], p['K']
        # Primary definitions
        assert J == b * d, f"({b},{e}): J={J} != bd={b*d}"
        assert K == a * d, f"({b},{e}): K={K} != ad={a*d}"
        # Alternative: J = D - C/2 (C is always even since C=2de)
        assert C % 2 == 0, f"({b},{e}): C={C} not even"
        assert J == D - C // 2, f"({b},{e}): J={J} != D-C/2={D-C//2}"
        assert K == D + C // 2, f"({b},{e}): K={K} != D+C/2={D+C//2}"
    return True, f"J=bd=D-C/2 and K=ad=D+C/2 verified for test cases"


def check_c2() -> tuple[bool, str]:
    """J + K = 2d^2: since a+b = (b+2e)+b = 2(b+e) = 2d, J+K = d(a+b) = 2d^2."""
    test_cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4)]
    for b, e in test_cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _bead_params(b, e)
        d, D, J, K = p['d'], p['D'], p['J'], p['K']
        assert J + K == 2 * D, f"({b},{e}): J+K={J+K} != 2D={2*D}"
        # Structural: a+b = 2d
        assert p['a'] + b == 2 * d, f"({b},{e}): a+b={p['a']+b} != 2d={2*d}"
    return True, f"J+K=2D=2d^2 via a+b=2d; verified {len(test_cases)} pairs"


def check_c3() -> tuple[bool, str]:
    """2J + C = 2d^2 (third way to decompose 2D)."""
    test_cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4)]
    for b, e in test_cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _bead_params(b, e)
        d, D, C, J = p['d'], p['D'], p['C'], p['J']
        assert 2 * J + C == 2 * D, f"({b},{e}): 2J+C={2*J+C} != 2D={2*D}"
        # Structural: 2J+C = 2bd + 2de = 2d(b+e) = 2d^2
    return True, f"2J+C=2D=2d^2 (decomposition via J=bd, C=2de, b+e=d); verified"


def check_c4() -> tuple[bool, str]:
    """Triple equality: F+G = J+K = 2J+C = 2D, simultaneously for all cases."""
    test_cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2)]
    for b, e in test_cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _bead_params(b, e)
        F, G, J, K, C, D = p['F'], p['G'], p['J'], p['K'], p['C'], p['D']
        two_D = 2 * D
        assert F + G == two_D, f"({b},{e}): F+G={F+G} != 2D={two_D}"
        assert J + K == two_D, f"({b},{e}): J+K={J+K} != 2D={two_D}"
        assert 2 * J + C == two_D, f"({b},{e}): 2J+C={2*J+C} != 2D={two_D}"
        # All three simultaneously
        assert F + G == J + K == 2 * J + C == 2 * D, (
            f"({b},{e}): triple equality failed"
        )
    return True, f"F+G=J+K=2J+C=2D simultaneously for {len(test_cases)} pairs"


def check_c5() -> tuple[bool, str]:
    """Algebraic proof: J=bd follows from J=D-C/2=d^2-de=d(d-e)=db (since d-e=b)."""
    # d - e = (b+e) - e = b → J = d(d-e) = db ✓
    # K = D + C/2 = d^2 + de = d(d+e) = da (since d+e = a = b+2e) ✓
    # J+K = bd+ad = d(b+a) = d*(b + b+2e) = d*(2b+2e) = 2d*(b+e) = 2d*d = 2d^2 ✓
    # 2J+C = 2bd + 2de = 2d(b+e) = 2d^2 ✓
    for b in range(1, 15, 2):
        for e in range(1, 15):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = b + 2 * e
            # Verify d-e=b and d+e=a
            assert d - e == b, f"({b},{e}): d-e={d-e} != b"
            assert d + e == a, f"({b},{e}): d+e={d+e} != a"
            J = b * d
            K = a * d
            assert J == d * (d - e), f"({b},{e}): bd != d(d-e)"
            assert K == d * (d + e), f"({b},{e}): ad != d(d+e)"
    return True, (
        "J=bd=d(d-e) since d-e=b; K=ad=d(d+e)=da since d+e=a; "
        "algebraic proof holds for all coprime pairs up to b,e<15"
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
        raise RuntimeError(f"cert [337] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
