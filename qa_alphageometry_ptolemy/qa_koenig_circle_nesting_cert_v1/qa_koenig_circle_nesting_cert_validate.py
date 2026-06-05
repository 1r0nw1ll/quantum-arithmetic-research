# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pythagorean triples, "
    "annular area identity H^2-G^2=G^2-I^2=2CF); "
    "no QA state evolution; Theorem NT: circles, areas, and 'Koenig series' "
    "are observer-layer labels on integer quadratic identities; "
    "all arithmetic exact integer, no float"
)

"""
Cert [334] — QA Koenig Circle Nesting

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, Ch.7 pp.55-70
  "THE KOENIG SERIES AND THE TREE OF LIFE"

For a prime Pythagorean triangle with legs C,F and hypotenuse G:
  I = C - F  (smaller Koenig radius)
  G = hypotenuse
  H = C + F  (larger Koenig radius)

Three concentric circles with radii I,G,H form two equal annular areas.

Key identities:
  H^2 - G^2 = G^2 - I^2 = 2*C*F  (G bisects area between I and H)
  Annular area = 2*C*F = 24*L   where L = C*F/12

Nesting property: H_n = I_{n+1}
  → 1, 5, 7, 13, 17, 25, 31, ... (G values between nested steps)
  Standard chain from Iverson:
    (I,G,H) = (1, 5, 7)  — triangle (4,3,5),  L=1
    (I,G,H) = (7,13,17)  — triangle (12,5,13), L=5
    (I,G,H) = (17,25,31) — triangle (24,7,25), L=14

Five claims certified via integer arithmetic.
"""

from math import gcd, isqrt


def _is_prime_pythagorean(C: int, F: int, G: int) -> bool:
    """True if (C,F,G) is a primitive Pythagorean triple (C^2+F^2=G^2, gcd=1)."""
    return (C * C + F * F == G * G) and gcd(gcd(C, F), G) == 1


def _koenig_igh(C: int, F: int, G: int) -> tuple[int, int, int]:
    """Return (I,G,H) = (|C-F|, G, C+F) for a Pythagorean triangle."""
    return abs(C - F), G, C + F


def check_c1() -> tuple[bool, str]:
    """G bisects area: H^2-G^2 = G^2-I^2 = 2*C*F for specific cases."""
    cases = [
        # (C, F, G): C=longer leg, F=shorter leg for convention
        (4,  3,  5),   # basic triangle: (I,G,H)=(1,5,7)   L=1
        (12, 5,  13),  # (I,G,H)=(7,13,17)  L=5
        (24, 7,  25),  # (I,G,H)=(17,25,31) L=14
        (8,  15, 17),  # (I,G,H)=(7,17,23)  L=10
        (20, 21, 29),  # (I,G,H)=(1,29,21... wait: I=|20-21|=1, H=41)
        (40, 9,  41),  # (I,G,H)=(31,41,49) L=30
        (28, 45, 53),  # (I,G,H)=(17,53,73) L=105
    ]
    for C, F, G in cases:
        # Ensure canonical: C>F
        if F > C:
            C, F = F, C
        assert _is_prime_pythagorean(C, F, G), f"({C},{F},{G}) not primitive Pythagorean"
        I, _, H = _koenig_igh(C, F, G)
        outer = H * H - G * G
        inner = G * G - I * I
        assert outer == inner, f"({C},{F},{G}): H^2-G^2={outer} != G^2-I^2={inner}"
        assert outer == 2 * C * F, f"({C},{F},{G}): annular area {outer} != 2CF={2*C*F}"
    return True, f"H^2-G^2=G^2-I^2=2CF verified for {len(cases)} prime Pythagorean triangles"


def check_c2() -> tuple[bool, str]:
    """Annular area = 24*L where L = C*F/12; one leg divisible by 4, other by 3."""
    # Primitive triples: one leg ≡ 0 (mod 4), other ≡ 0 (mod 3)
    # → product C*F ≡ 0 (mod 12) → 2*C*F ≡ 0 (mod 24)
    cases = [
        (4, 3, 5, 1),    # L=1, area=24
        (12, 5, 13, 5),  # L=5, area=120
        (24, 7, 25, 14), # L=14, area=336
        (8, 15, 17, 10), # L=10, area=240
        (40, 9, 41, 30), # L=30, area=720
        (28, 45, 53, 105), # L=105, area=2520
    ]
    for C, F, G, L_expected in cases:
        if F > C:
            C, F = F, C
        assert _is_prime_pythagorean(C, F, G), f"({C},{F},{G}) not Pythagorean"
        area = 2 * C * F
        assert area % 24 == 0, f"({C},{F},{G}): area={area} not divisible by 24"
        L = area // 24
        assert L == L_expected, f"({C},{F},{G}): L={L} expected {L_expected}"
        # Also verify CF divisible by 12
        assert (C * F) % 12 == 0, f"({C},{F},{G}): C*F={C*F} not div by 12"
        assert C * F // 12 == L_expected, f"({C},{F},{G}): CF/12={C*F//12} != L={L_expected}"
    return True, f"Annular area=24L; CF divisible by 12; L=CF/12 verified for {len(cases)} triangles"


def check_c3() -> tuple[bool, str]:
    """Nesting: H_n = I_{n+1}. Chain: (1,5,7)→(7,13,17)→(17,25,31)."""
    chain = [
        (4, 3, 5),    # (I,G,H)=(1,5,7)
        (12, 5, 13),  # (I,G,H)=(7,13,17)
        (24, 7, 25),  # (I,G,H)=(17,25,31)
    ]
    igh_chain = []
    for C, F, G in chain:
        if F > C:
            C, F = F, C
        I, _, H = _koenig_igh(C, F, G)
        igh_chain.append((I, G, H))

    # Verify H_n = I_{n+1}
    for n in range(len(igh_chain) - 1):
        I_n, G_n, H_n = igh_chain[n]
        I_next, G_next, H_next = igh_chain[n + 1]
        assert H_n == I_next, (
            f"Nesting failure: H_{n}={H_n} != I_{n+1}={I_next}"
        )

    # Verify specific values
    assert igh_chain[0] == (1, 5, 7), f"Step 0: {igh_chain[0]} != (1,5,7)"
    assert igh_chain[1] == (7, 13, 17), f"Step 1: {igh_chain[1]} != (7,13,17)"
    assert igh_chain[2] == (17, 25, 31), f"Step 2: {igh_chain[2]} != (17,25,31)"

    radii = [igh_chain[0][0]]
    for I, G, H in igh_chain:
        radii.extend([G, H])

    return True, f"Nesting H_n=I_(n+1) for chain (1,5,7)→(7,13,17)→(17,25,31); radii={radii}"


def check_c4() -> tuple[bool, str]:
    """Annular area identity holds for all primitive Pythagorean triples up to hypotenuse 200."""
    # Generate all primitive triples via Euclid's formula: m>n>0, gcd(m,n)=1, m-n odd
    # C=m^2-n^2, F=2mn, G=m^2+n^2 (or swap C,F)
    count = 0
    for m in range(2, 20):
        for n in range(1, m):
            if gcd(m, n) != 1:
                continue
            if (m - n) % 2 == 0:
                continue
            C = m * m - n * n
            F = 2 * m * n
            G = m * m + n * n
            if G > 200:
                continue
            # Canonical: ensure C>F
            if F > C:
                C, F = F, C
            assert _is_prime_pythagorean(C, F, G), f"({C},{F},{G}) generation error"
            I, _, H = _koenig_igh(C, F, G)
            outer = H * H - G * G
            inner = G * G - I * I
            assert outer == inner == 2 * C * F, (
                f"({C},{F},{G}): annular area mismatch"
            )
            area = 2 * C * F
            assert area % 24 == 0, f"({C},{F},{G}): area={area} not div by 24"
            count += 1
    return True, f"H^2-G^2=G^2-I^2=2CF and area divisible by 24 for {count} primitive triples (G<=200)"


def check_c5() -> tuple[bool, str]:
    """Extended nesting: each H value in the chain equals the I of the next step."""
    # Build next step: given (I_n, G_n, H_n), find a triple with I_{n+1}=H_n
    # I = C-F = H_n → C-F = H_n; C^2+F^2=G_{n+1}^2; C+F=H_{n+1}
    # From C-F=h and C^2+F^2=G^2: G^2=h^2+2F^2+2Fh → not directly useful
    # Instead: generate triples with I=target directly

    def triples_with_I(target: int, max_G: int = 500) -> list[tuple[int, int, int, int, int]]:
        """Find primitive triples (C,F,G) with |C-F|=target, return (I,G,H)."""
        results = []
        for m in range(2, 40):
            for n in range(1, m):
                if gcd(m, n) != 1 or (m - n) % 2 == 0:
                    continue
                C = m * m - n * n
                F = 2 * m * n
                G = m * m + n * n
                if G > max_G:
                    continue
                if F > C:
                    C, F = F, C
                if abs(C - F) == target:
                    I, _, H = _koenig_igh(C, F, G)
                    results.append((I, G, H))
        return results

    # Verify the chain extends: H_0=7 → H_1=17 → H_2=31
    h0_triples = triples_with_I(7)
    assert (7, 13, 17) in h0_triples, f"(7,13,17) not found among I=7 triples: {h0_triples}"

    h1_triples = triples_with_I(17)
    assert (17, 25, 31) in h1_triples, f"(17,25,31) not found among I=17 triples: {h1_triples}"

    h2_triples = triples_with_I(31)
    assert len(h2_triples) >= 1, f"No triples with I=31 found"

    return True, (
        f"Nesting chain validated: I=7 gives {len(h0_triples)} options incl. (7,13,17); "
        f"I=17 gives {len(h1_triples)} options incl. (17,25,31); "
        f"I=31 gives {len(h2_triples)} continuation options"
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
        raise RuntimeError(f"cert [334] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
