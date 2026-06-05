# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.VI External Relationships: "
    "column step F(b,e+1)-F(b,e)=2b; row step C(b+2,e)-C(b,e)=4e; G-F=2E; G-C=B; "
    "A(b,e)=B(b+2e,e); D(b,e)=E(b,e+b); 2-par appears only in {e,d,J,K}); "
    "Theorem NT: 'table', 'ellipse', 'apogee', 'perigee' are observer projection labels; "
    "no float state, no QA orbit evolution"
)

"""
Cert [363] — QA Pyth-1 External Relationships (Ch.VI)

Source: Iverson (1993) Pythagorean Arithmetic Vol I, Chapter VI pp.67-71
  p.62: 'In the first column, the value of F increases in steps of 2 units.
         In the remainder of the columns, F increases by 2b units at each step.'
  p.62: 'the value C increases by 4e units at each step for the base of the triangles.'
  p.62: 'The difference between F and G remains constant in the blocks across the page,
         and the difference between C and G remains constant, progressing down the columns.'
  p.62: 'the value of A becomes the value of B moving to the right e-blocks.'
  p.62: 'A given value of D becomes the same value of E at a distance of b-blocks down.'
  p.63: 'complete absence of all 2-par numbers except as values of e, d, J or K.'

Five claims (algebraic — all derivable from A2: d=b+e, a=b+2e RAW):
  C1: Column step: F(b,e+1)-F(b,e) = 2b for all prime pairs (b,e);
      Row step: C(b+2,e)-C(b,e) = 4e for all prime pairs (b,e)
      proof: F(b,e)=b(b+2e); F(b,e+1)=b(b+2e+2)=F+2b; C(b,e)=2(b+e)e; C(b+2,e)=2(b+2+e)e=C+4e
  C2: G(b,e)-F(b,e) = 2e^2 = 2E (constant along rows fixing e);
      G(b,e)-C(b,e) = b^2 = B (constant along columns fixing b)
      proof: G-F=(d^2+e^2)-(d^2-e^2)=2e^2; G-C=(d^2+e^2)-2de=(d-e)^2=b^2
  C3: A(b,e) = B(b+2e, e): the A-value of any prime pair equals the B-value
      of the prime pair (b+2e, e) [moving e-blocks to the right in Table 3]
      proof: A(b,e)=(b+2e)^2=B(b+2e,e) since B(b',e)=b'^2 with b'=b+2e
  C4: D(b,e) = E(b, e+b): the D-value of (b,e) equals the E-value of (b,e+b)
      [moving b-blocks down the column]
      proof: D(b,e)=(b+e)^2=d^2; E(b,e+b)=(e+b)^2=d^2
  C5: 2-par integers (n≡2 mod 4) appear only among {e, d, J, K};
      they never appear in {A, B, C, D, E, F, G, H, I} for any prime pair
      proof: A=a^2,B=b^2,D=d^2,E=e^2 are always squares (0 or 1 mod 4, never 2);
             C≡0 mod 4 (cert [355]); F is 3-par or 5-par (cert [361]);
             G,H,I are all ≡1 mod 4 (odd; G 5-par; H,I no factor <7 means odd)
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


def check_c1() -> tuple[bool, str]:
    """Column step F(b,e+1)-F(b,e)=2b; Row step C(b+2,e)-C(b,e)=4e."""
    col_count = 0
    row_count = 0
    for b, e, d, a in _prime_pairs(30, 30):
        F = a * b            # F(b,e)
        C = 2 * d * e        # C(b,e)
        # Column step: F(b,e+1) for any new e+1 (gcd(b,e+1) may not be 1, but formula holds)
        e1 = e + 1
        d1 = b + e1
        a1 = d1 + e1
        F1 = a1 * b          # F(b,e+1)
        assert F1 - F == 2 * b, f"F(b,e+1)-F(b,e)={F1-F} != 2b={2*b} at b={b},e={e}"
        col_count += 1
        # Row step: C(b+2,e) — need b+2 (still odd) with gcd(b+2,e) possibly !=1
        b2 = b + 2
        d2 = b2 + e
        C2 = 2 * d2 * e      # C(b+2,e)
        assert C2 - C == 4 * e, f"C(b+2,e)-C(b,e)={C2-C} != 4e={4*e} at b={b},e={e}"
        row_count += 1
    return True, (
        f"Column step F(b,e+1)-F(b,e)=2b verified for {col_count} pairs (b,e)<=30; "
        f"Row step C(b+2,e)-C(b,e)=4e verified for {row_count} pairs; "
        f"proof: F=b(b+2e)→F(b,e+1)=b(b+2e+2)=F+2b; C=2(b+e)e→C(b+2,e)=2(b+2+e)e=C+4e ✓"
    )


def check_c2() -> tuple[bool, str]:
    """G-F=2E (constant along rows); G-C=B (constant along columns)."""
    count = 0
    for b, e, d, a in _prime_pairs(30, 30):
        C = 2 * d * e
        F = a * b
        G = d * d + e * e
        B = b * b
        E = e * e
        # G - F = 2e^2 = 2E (independent of b, constant for fixed e)
        assert G - F == 2 * E, f"G-F={G-F} != 2E={2*E} at b={b},e={e}"
        # G - C = b^2 = B (independent of e, constant for fixed b)
        assert G - C == B, f"G-C={G-C} != B={B} at b={b},e={e}"
        count += 1
    # Proofs:
    # G-F = (d^2+e^2)-(d^2-e^2) = 2e^2 = 2E ✓
    # G-C = (d^2+e^2)-2de = (d-e)^2 = b^2 = B ✓
    return True, (
        f"G-F=2E and G-C=B verified for all {count} prime pairs (b,e)<=30; "
        f"proof: G-F=(d^2+e^2)-(d^2-e^2)=2e^2; G-C=(d-e)^2=b^2 ✓"
    )


def check_c3() -> tuple[bool, str]:
    """A(b,e) = B(b+2e, e): A-value of (b,e) equals B-value of (b+2e, e)."""
    count = 0
    for b, e, d, a in _prime_pairs(20, 20):
        A = a * a                  # A(b,e) = (b+2e)^2
        b_new = b + 2 * e         # = a (the 'a' value of current pair)
        B_new = b_new * b_new     # B(b+2e, e) = b_new^2 = (b+2e)^2
        assert A == B_new, f"A(b,e)={A} != B(b+2e,e)={B_new} at b={b},e={e}"
        count += 1
    # Proof: A(b,e) = a^2 = (b+2e)^2; B(b_new,e) = b_new^2 with b_new=b+2e → equal ✓
    # This means: the 'a' bead of (b,e) is the 'b' bead of (b+2e,e)
    # and A values propagate e-blocks to the right in Table 3 as B values
    return True, (
        f"A(b,e)=B(b+2e,e) verified for {count} prime pairs (b,e)<=20; "
        f"proof: A=(b+2e)^2=B(b+2e,e) since B(b',e)=b'^2 with b'=b+2e; "
        f"the 'a'-bead of (b,e) equals the 'b'-bead of (a,e) — A-B migration ✓"
    )


def check_c4() -> tuple[bool, str]:
    """D(b,e) = E(b, e+b): D-value of (b,e) equals E-value of (b,e+b)."""
    count = 0
    for b, e, d, a in _prime_pairs(20, 20):
        D = d * d                  # D(b,e) = (b+e)^2
        e_new = e + b             # b-blocks down the column
        d_new = b + e_new         # d(b, e+b) = b + (e+b) = 2b+e = d+b
        E_new = e_new * e_new     # E(b, e+b) = (e+b)^2
        assert D == E_new, f"D(b,e)={D} != E(b,e+b)={E_new} at b={b},e={e}"
        count += 1
    # Proof: D(b,e) = d^2 = (b+e)^2; E(b,e+b) = (e+b)^2 → equal ✓
    # This means: the 'd' bead of (b,e) is the 'e' bead of (b,e+b)
    # and D values propagate b-blocks down in Table 3 as E values
    return True, (
        f"D(b,e)=E(b,e+b) verified for {count} prime pairs (b,e)<=20; "
        f"proof: D=(b+e)^2; E(b,e+b)=(e+b)^2; equal by symmetry; "
        f"the 'd'-bead of (b,e) equals the 'e'-bead of (b,e+b) — D-E migration ✓"
    )


def check_c5() -> tuple[bool, str]:
    """2-par integers appear only in {e,d,J,K}; never in {A,B,C,D,E,F,G,H,I}."""
    count_2par_in_excluded = 0
    total = 0
    for b, e, d, a in _prime_pairs(35, 35):
        A = a * a
        B = b * b
        C = 2 * d * e
        D = d * d
        E = e * e
        F = a * b
        G = d * d + e * e
        H = C + F
        I = abs(C - F)
        excluded_ids = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "H": H, "I": I}
        for name, val in excluded_ids.items():
            if val % 4 == 2:  # is 2-par
                count_2par_in_excluded += 1
                assert False, f"{name}={val} is 2-par at b={b},e={e}"
        total += 1
    # Verify 2-par CAN appear in {e, d, J, K}
    # b=1,e=1: d=2 (2-par); J=bd=2 (2-par); K=ad=6 (2-par); e=1 (not 2-par)
    b, e, d, a = 1, 1, 2, 3
    J = b * d
    K = a * d
    assert d % 4 == 2, f"Expected d=2 to be 2-par"
    assert J % 4 == 2, f"Expected J=2 to be 2-par"
    assert K % 4 == 2, f"Expected K=6 to be 2-par"
    # e=2: e is 2-par (e=2 mod 4)
    # b=1,e=2: d=3,a=5. d is odd, J=3,K=15. e=2 is 2-par.
    assert (2 % 4) == 2, "e=2 is 2-par as expected"
    return True, (
        f"2-par (≡2 mod 4) never appears in {{A,B,C,D,E,F,G,H,I}} across {total} pairs (b,e)<=35 ✓; "
        f"2-par CAN appear in d (d=2 at b=1,e=1), J (J=2 at b=1,e=1), K (K=6 at b=1,e=1), "
        f"e (e=2 itself is 2-par); "
        f"proof: A,B,D,E are squares (0 or 1 mod 4); C≡0 mod 4; F is 3-par or 5-par; G,H,I odd ✓"
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
        raise RuntimeError(f"cert [363] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
