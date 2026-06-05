# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (QA ellipse semiminor identity "
    "D^2-(C/2)^2=DF; eccentricity c=d/e=2D/C); no QA state evolution; "
    "Theorem NT: 'semiminor diameter', 'focus', 'latus rectum' are observer-layer "
    "labels on integer bead number relationships; (df)^2=DF is integer arithmetic; "
    "no float state; sqrt notation is observer projection only"
)

"""
Cert [341] — QA Ellipse Semiminor Squared = D*F; Ellipse Eccentricity c=d/e=2D/C

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, Chapter VIII pp.91-93
  "D^2 - (C/2)^2 = (df)^2"
  "Since C = 2de, C/2 then equals de, and (C/2)^2 = d^2*e^2 = DE"
  "So (df)^2 = D^2 - DE. The right side factors into D(D-E)."
  "But D - E = F; So (df)^2 = DF"
  "the eccentricity of the ellipse is [c =] d/e = 2D/C"
  "J = d^2 - ed, and K = d^2 + ed" (line 6301)
  "major diameter is 2D which is equal to J+K" (line 6260)

Bead number definitions (A2 compliant, raw — no mod reduction):
  d = b + e; a = b + 2*e; C = 2*d*e; D = d*d; E = e*e; F = a*b (= D-E)

Claims:
  C1: (C/2)^2 = D*E  (since C/2=de, (de)^2=d^2*e^2=D*E)
  C2: D - E = F  (d^2-e^2=(d-e)(d+e)=b*a=F; same as cert[338] C1)
  C3: D^2 - D*E = D*F  (semiminor squared: (df)^2 = D*(D-E) = D*F)
  C4: eccentricity integer form: 2*D = (d/e)*C  (i.e., 2D/C = d/e; verified as
      2*d^2 == (d/e)*(2*d*e) = 2*d^2; Iverson's c=d/e is always rational)
  C5: J = D - d*e = b*d and K = D + d*e = a*d; J+K = 2*D (major diameter)
"""

from math import gcd
from fractions import Fraction


def _params(b: int, e: int) -> dict:
    d = b + e
    a = b + 2 * e
    C = 2 * d * e
    D = d * d
    E = e * e
    F = a * b
    J = b * d
    K = a * d
    return dict(b=b, e=e, d=d, a=a, C=C, D=D, E=E, F=F, J=J, K=K)


def check_c1() -> tuple[bool, str]:
    """(C/2)^2 = D*E; since C=2de, C/2=de, (C/2)^2=(de)^2=d^2*e^2=D*E."""
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        half_C = p['d'] * e  # C/2 = de (exact integer)
        assert half_C * half_C == p['D'] * p['E'], (
            f"({b},{e}): (C/2)^2={(half_C*half_C)} != D*E={p['D']*p['E']}"
        )
    return True, f"(C/2)^2=DE verified for {len(cases)} pairs; C/2=de is exact integer"


def check_c2() -> tuple[bool, str]:
    """D - E = F: d^2 - e^2 = (d-e)(d+e) = b*a = F."""
    # This is cert[338] C1 restated; required here as algebraic step for C3
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        assert p['D'] - p['E'] == p['F'], (
            f"({b},{e}): D-E={p['D']-p['E']} != F={p['F']}"
        )
    return True, f"D-E=F (d^2-e^2=ab) verified for {len(cases)} pairs"


def check_c3() -> tuple[bool, str]:
    """(df)^2 = D*F: the semiminor squared equals D times the semilatus rectum."""
    # Proof chain: (df)^2 = D^2 - (C/2)^2 = D^2 - D*E = D*(D-E) = D*F
    # All steps verified:
    # (df)^2 = D^2 - D*E (from C1: (C/2)^2=DE, so D^2-(C/2)^2=D^2-DE)
    # D^2 - D*E = D(D-E) = D*F (from C2: D-E=F)
    count = 0
    for b in range(1, 14, 2):
        for e in range(1, 14):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = b + 2 * e
            D = d * d
            E = e * e
            F = a * b
            # (C/2)^2 = (de)^2 = D*E
            half_C_sq = (d * e) * (d * e)
            # semiminor squared = D^2 - (C/2)^2
            semiminor_sq = D * D - half_C_sq
            assert semiminor_sq == D * F, (
                f"({b},{e}): D^2-(C/2)^2={semiminor_sq} != D*F={D*F}"
            )
            count += 1
    return True, (
        f"(df)^2=D^2-(C/2)^2=D(D-E)=DF verified for {count} coprime pairs b,e<14"
    )


def check_c4() -> tuple[bool, str]:
    """Eccentricity c=d/e=2D/C: verify 2D*e=d*C (cross-multiply avoids float)."""
    # c = d/e = 2D/C; cross-multiply: d*C = 2D*e
    # d*C = d*2de = 2d^2*e = 2D*e ✓
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        # d/e = 2D/C ↔ d*C = 2*D*e
        assert p['d'] * p['C'] == 2 * p['D'] * e, (
            f"({b},{e}): d*C={p['d']*p['C']} != 2*D*e={2*p['D']*e}"
        )
        # Also verify Fraction equality
        assert Fraction(p['d'], e) == Fraction(2 * p['D'], p['C']), (
            f"({b},{e}): d/e != 2D/C as fractions"
        )
    return True, f"Eccentricity c=d/e=2D/C verified for {len(cases)} pairs (cross-multiply integer check)"


def check_c5() -> tuple[bool, str]:
    """J=D-d*e=b*d and K=D+d*e=a*d; J+K=2D (major diameter); K-J=C."""
    # J = bd = d(d-e) = d^2-de = D-de ✓
    # K = ad = d(d+e) = d^2+de = D+de ✓
    # J+K = (D-de)+(D+de) = 2D ✓
    # K-J = 2de = C ✓
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        D, d = p['D'], p['d']
        J, K, C = p['J'], p['K'], p['C']
        assert J == D - d * e, f"({b},{e}): J={J} != D-de={D-d*e}"
        assert K == D + d * e, f"({b},{e}): K={K} != D+de={D+d*e}"
        assert J + K == 2 * D, f"({b},{e}): J+K={J+K} != 2D={2*D}"
        assert K - J == C, f"({b},{e}): K-J={K-J} != C={C}"
    return True, f"J=D-de=bd; K=D+de=ad; J+K=2D; K-J=C verified for {len(cases)} pairs"


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
        raise RuntimeError(f"cert [341] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
