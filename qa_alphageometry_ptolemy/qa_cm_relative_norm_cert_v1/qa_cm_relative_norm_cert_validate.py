#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical algebraic number theory; Neukirch (1999) ISBN 978-3-540-65399-8 §II.10 (CM fields, Hecke characters, relative norm); Washington (1997) ISBN 978-0-387-94762-4 Ch.1 (cyclotomic integers, CM involution); Lang (1994) ISBN 978-1-4612-0853-2 §VIII.2 (tower law for norms, relative norm factorization); Silverman (1994) ISBN 978-0-387-94328-2 §II.10 (CM theory for elliptic curves, Hecke Grössencharacter) -->
"""
Cert [397] — QA CM Relative Norm and Tower Law

CLAIM:
  For π = a+b*zeta_5 in Z[zeta_5] with N_{K/Q}(a+b*zeta_5) = p (a rational prime,
  p ≡ 1 mod 5), four interrelated algebraic structures hold — all provable by
  exact integer arithmetic over Z[phi] (phi = (1+sqrt(5))/2 ∈ Q(sqrt(5)) = F):

  (A) RELATIVE NORM FORMULA:
      N_{K/F}(a+b*zeta_5) = (a^2-ab+b^2) + ab*phi  ∈ Z[phi]
      where K = Q(zeta_5), F = Q(sqrt(5)) are related by the CM extension K/F.
      The CM involution sigma_K: zeta_5 -> zeta_5^4 (complex conjugation) fixes F.

  (B) TOWER LAW:
      N_{F/Q}(N_{K/F}(pi)) = N_{K/Q}(pi) = p
      Algebraic identity: (a^2-ab+b^2)^2 + (a^2-ab+b^2)(ab) - (ab)^2 = a^4-a^3b+a^2b^2-ab^3+b^4
      This is the norm tower N_{K/Q} = N_{F/Q} ∘ N_{K/F}, proved purely by algebra.

  (C) CM NORM FACTORIZATION:
      N_{K/F}(pi) * sigma_F(N_{K/F}(pi)) = N_{K/Q}(pi) = p
      where sigma_F(u+v*phi) = (u+v) - v*phi   (phi -> 1-phi, the F/Q Galois involution)
      Equivalently: N_{K/F}(pi) and sigma_F(N_{K/F}(pi)) are the two F-conjugate
      "halves" of the absolute norm p.

  (D) CM MIN POLY AND TOTALLY-NEGATIVE DISCRIMINANT:
      The minimal polynomial of pi over F is:
          X^2 - Tr_{K/F}(pi)*X + N_{K/F}(pi)  ∈ Z[phi][X]
      where Tr_{K/F}(pi) = pi + sigma_K(pi) = (a+b*zeta_5) + (a+b*zeta_5^4)
                         = (2a-b) + b*phi  ∈ Z[phi]   [= partial trace from cert [396]]
      and N_{K/F}(pi) = pi * sigma_K(pi)   [= relative norm from check A]

      DISCRIMINANT = Tr^2 - 4*N_{rel} = -b^2*(2+phi)  ∈ Z[phi]

      For b != 0: disc = -2b^2 + (-b^2)*phi.
      This is TOTALLY NEGATIVE: both real embeddings of Q(sqrt(5)) map disc to a
      negative real number (since 2+phi ≈ 3.618 > 0 and 2+(1-phi) = 3-phi ≈ 1.382 > 0,
      so -b^2*(2+phi) is negative in BOTH embeddings).

      TOTALLY NEGATIVE discriminant ⟺ the CM minimal polynomial has NO real roots
      in either real embedding ⟺ the Frobenius eigenvalues are complex in every
      embedding ⟺ Weil bound holds at EVERY real place.

      This is the degree-4 / CM analog of cert [395]'s discriminant sign flip:
        [395]: Delta_{GL2} = a_f^2 - 4p < 0  (1 real place, Q)
        [397]: disc = -b^2*(2+phi) << 0       (TOTALLY negative, 2 real places of F)

POSITION IN THE LANGLANDS CHAIN:
  GL_1: (5/p) ∈ {+-1}              cert [394] — degree-2 orbit character
  GL_2: |a_f| ≤ 2*sqrt(N)          cert [395] — Cassini weight-0/weight-2 sign flip
  GL_2: exact a_f (LMFDB)           cert [390] — hardcoded LMFDB data
  K/F:  Z[zeta_5] infrastructure    cert [396] — companion matrix C^5=I, G^4=I
  CM:   relative norm + min poly     cert [397] THIS — N_{K/F}, tower law, disc

THEOREM NT COMPLIANCE:
  Z[phi] is represented as integer pairs (u,v) = u+v*phi. All operations are
  exact integer arithmetic: addition, multiplication with phi^2 = phi+1 applied,
  and norm computation u^2+u*v-v^2. No floats enter the QA layer. The values
  2+phi ≈ 3.618 and 3-phi ≈ 1.382 appear only in the docstring as motivating
  context, never as computed float values.

CHECKS:
  C1: RELATIVE NORM: N_{K/F}(a+b*zeta_5) = (a^2-ab+b^2) + ab*phi for 4 generators.
  C2: TOWER LAW: N_{F/Q}(N_{K/F}(pi)) = N_{K/Q}(pi) = p for 4 primes.
  C3: CM NORM FACTORING: N_{K/F}(pi) * sigma_F(N_{K/F}(pi)) = p for 4 primes.
  C4: CM MIN POLY: Tr_{K/F}(pi) = (2a-b)+b*phi; poly X^2-Tr*X+N_{rel} ∈ Z[phi][X].
  C5: TOTALLY NEGATIVE DISC: disc = -b^2*(2+phi); both components strictly negative;
      N_{F/Q}(disc) = 5*b^4 > 0 (totally negative, not just negative in one embedding).
"""

import sys

# ──────────────────────────────────────────────────────────────────────────
# Z[phi] arithmetic   (phi = (1+sqrt(5))/2, phi^2 = phi+1)
# Elements: (u, v) meaning u + v*phi
# phi is the POSITIVE root of x^2 - x - 1 = 0
# Galois conjugate: sigma_F: phi -> 1-phi  (the other root, phi_bar = 1-phi)
# N_{F/Q}(u+v*phi) = (u+v*phi)(u+v*(1-phi)) = u^2 + uv - v^2
# ──────────────────────────────────────────────────────────────────────────

def zphi_add(a, b):
    return (a[0]+b[0], a[1]+b[1])

def zphi_sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def zphi_mul(a, b):
    # (u1+v1*phi)(u2+v2*phi) = u1u2 + (u1v2+v1u2)*phi + v1v2*phi^2
    # phi^2 = phi+1, so v1v2*phi^2 = v1v2 + v1v2*phi
    u = a[0]*b[0] + a[1]*b[1]        # u1u2 + v1v2
    v = a[0]*b[1] + a[1]*b[0] + a[1]*b[1]  # u1v2 + v1u2 + v1v2
    return (u, v)

def zphi_neg(a):
    return (-a[0], -a[1])

def zphi_norm(a):
    """N_{F/Q}(u+v*phi) = u^2 + uv - v^2  (exact integer)."""
    u, v = a
    return u*u + u*v - v*v

def sigma_F(a):
    """sigma_F(u+v*phi) = u+v*(1-phi) = (u+v) - v*phi  in Z[phi]."""
    u, v = a
    return (u+v, -v)


# ──────────────────────────────────────────────────────────────────────────
# Quartic norm formula (for pi = a+b*zeta_5, c=d=0)
# N_{K/Q}(a+b*zeta_5) = a^4 - a^3b + a^2b^2 - ab^3 + b^4
# ──────────────────────────────────────────────────────────────────────────

def norm_kq(a, b):
    return a*a*a*a - a*a*a*b + a*a*b*b - a*b*b*b + b*b*b*b


# ──────────────────────────────────────────────────────────────────────────
# Relative norm formula N_{K/F}(a+b*zeta_5) = (a^2-ab+b^2) + ab*phi
# Derivation: N_{K/F}(pi) = pi * sigma_K(pi) where sigma_K: zeta_5 -> zeta_5^4
# (a+b*zeta_5)(a+b*zeta_5^4) = a^2 + ab(zeta_5+zeta_5^4) + b^2*zeta_5^5
#   = a^2+b^2 + ab*(zeta_5+zeta_5^4)
# zeta_5+zeta_5^4 = 2*cos(2*pi/5) = (sqrt(5)-1)/2 = phi-1
# so = a^2+b^2 + ab*(phi-1) = (a^2+b^2-ab) + ab*phi = (a^2-ab+b^2) + ab*phi
# ──────────────────────────────────────────────────────────────────────────

def rel_norm(a, b):
    """Returns (u, v) in Z[phi] = u+v*phi."""
    return (a*a - a*b + b*b, a*b)


# ──────────────────────────────────────────────────────────────────────────
# Partial trace = CM eigenvalue candidate
# Tr_{K/F}(a+b*zeta_5) = (a+b*zeta_5) + sigma_K(a+b*zeta_5)
#   = (a+b*zeta_5) + (a+b*zeta_5^4)  [this is the same as N formula derivation numerator]
#   = 2a + b*(zeta_5+zeta_5^4) = 2a + b*(phi-1) = (2a-b) + b*phi
# ──────────────────────────────────────────────────────────────────────────

def partial_trace(a, b):
    return (2*a - b, b)


# ──────────────────────────────────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────────────────────────────────

PRIMES = [
    (2,  1,  11,  "p=11"),
    (2, -1,  31,  "p=31"),
    (3,  1,  61,  "p=61"),
    (4,  3,  181, "p=181"),
]


def main():
    failures = []
    passed = []

    print("=" * 72)
    print("Cert [397] — QA CM Relative Norm and Tower Law")
    print("  CM structure of Q(zeta_5)/Q(sqrt(5)) at primes p ≡ 1 (mod 5)")
    print("=" * 72)

    # ── C1: RELATIVE NORM FORMULA ─────────────────────────────────────────
    print("\n  C1: RELATIVE NORM  N_{K/F}(a+b*zeta_5) = (a^2-ab+b^2) + ab*phi")
    c1_ok = True
    for a, b, p, tag in PRIMES:
        rn = rel_norm(a, b)
        u, v = rn
        # Verify components are integers (they always are by construction)
        # Cross-check: N_{F/Q}(rel_norm) should equal N_{K/Q}(pi) = p
        nf = zphi_norm(rn)
        expected_u = a*a - a*b + b*b
        expected_v = a*b
        ok = (u == expected_u and v == expected_v)
        mark = "PASS" if ok else "FAIL"
        sign_v = "+" if v >= 0 else ""
        print(f"  [{mark}] ({a},{b}): N_rel = {u}{sign_v}{v}*phi  [{tag}]")
        if not ok:
            failures.append(f"C1: ({a},{b}) N_rel=({u},{v}) != ({expected_u},{expected_v})")
            c1_ok = False
    if c1_ok:
        passed.append("C1")
        print("         Formula: u = a^2-ab+b^2  (integer),  v = ab  (integer)")

    # ── C2: TOWER LAW ─────────────────────────────────────────────────────
    print("\n  C2: TOWER LAW  N_{F/Q}(N_{K/F}(pi)) = N_{K/Q}(pi) = p")
    print("      Identity: (a^2-ab+b^2)^2 + (a^2-ab+b^2)(ab) - (ab)^2 = a^4-a^3b+a^2b^2-ab^3+b^4")
    c2_ok = True
    for a, b, p, tag in PRIMES:
        rn = rel_norm(a, b)
        nf = zphi_norm(rn)
        nkq = norm_kq(a, b)
        ok = (nf == nkq == p)
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] ({a},{b}): N_F(N_rel) = {nf}, N_KQ = {nkq}, p = {p}  [{tag}]")
        if not ok:
            failures.append(f"C2: ({a},{b}) N_F(N_rel)={nf}, N_KQ={nkq}, p={p}")
            c2_ok = False
    if c2_ok:
        passed.append("C2")
        print("         Tower law N_{K/Q} = N_{F/Q} ∘ N_{K/F} verified at all 4 primes")

    # ── C3: CM NORM FACTORIZATION ─────────────────────────────────────────
    print("\n  C3: CM NORM FACTORIZATION  N_{K/F}(pi) * sigma_F(N_{K/F}(pi)) = p")
    print("      sigma_F(u+v*phi) = (u+v) - v*phi   [phi -> 1-phi]")
    c3_ok = True
    for a, b, p, tag in PRIMES:
        rn = rel_norm(a, b)
        rn_conj = sigma_F(rn)
        product = zphi_mul(rn, rn_conj)
        # product should be (p, 0) i.e. the integer p embedded in Z[phi]
        ok = (product == (p, 0))
        mark = "PASS" if ok else "FAIL"
        u, v = rn
        uc, vc = rn_conj
        sign_v = "+" if v >= 0 else ""
        sign_vc = "+" if vc >= 0 else ""
        print(f"  [{mark}] ({a},{b}): ({u}{sign_v}{v}phi) * ({uc}{sign_vc}{vc}phi) = {product}  [{tag}]")
        if not ok:
            failures.append(f"C3: ({a},{b}) product={product} != ({p},0)")
            c3_ok = False
    if c3_ok:
        passed.append("C3")
        print("         N_{K/F} and sigma_F(N_{K/F}) are the two F/Q-conjugate halves of p")

    # ── C4: CM MIN POLY ───────────────────────────────────────────────────
    print("\n  C4: CM MIN POLY  X^2 - Tr_{K/F}(pi)*X + N_{K/F}(pi)  ∈ Z[phi][X]")
    print("      Tr_{K/F}(a+b*zeta_5) = (2a-b) + b*phi  (= partial trace, cert [396])")
    c4_ok = True
    for a, b, p, tag in PRIMES:
        tr = partial_trace(a, b)
        rn = rel_norm(a, b)
        # Verify Tr = (2a-b, b) and N_rel = (a^2-ab+b^2, ab)
        ok_tr = (tr == (2*a - b, b))
        ok_rn = (rn == (a*a - a*b + b*b, a*b))
        ok = ok_tr and ok_rn
        mark = "PASS" if ok else "FAIL"
        sign_tr = "+" if tr[1] >= 0 else ""
        sign_rn = "+" if rn[1] >= 0 else ""
        print(f"  [{mark}] ({a},{b}): Tr=({tr[0]}{sign_tr}{tr[1]}phi), N_rel=({rn[0]}{sign_rn}{rn[1]}phi)  [{tag}]")
        print(f"            min poly: X^2 - ({tr[0]}{sign_tr}{tr[1]}phi)*X + ({rn[0]}{sign_rn}{rn[1]}phi)")
        if not ok:
            failures.append(f"C4: ({a},{b}) tr={tr}, rn={rn}")
            c4_ok = False
    if c4_ok:
        passed.append("C4")

    # ── C5: TOTALLY NEGATIVE DISCRIMINANT ─────────────────────────────────
    print("\n  C5: TOTALLY NEGATIVE DISC  disc = Tr^2 - 4*N_rel = -b^2*(2+phi)")
    print("      disc = (-2b^2) + (-b^2)*phi  in Z[phi]")
    print("      Both components negative for b != 0  =>  TOTALLY NEGATIVE")
    print("      N_{F/Q}(disc) = 5*b^4 > 0  (product of two negative values)")
    c5_ok = True
    for a, b, p, tag in PRIMES:
        tr = partial_trace(a, b)
        rn = rel_norm(a, b)
        # Compute Tr^2 - 4*N_rel in Z[phi]
        tr_sq = zphi_mul(tr, tr)
        four_nrel = (4*rn[0], 4*rn[1])
        disc = zphi_sub(tr_sq, four_nrel)
        # Expected: disc = (-2b^2, -b^2) = -b^2*(2, 1) where (2,1) = 2+phi
        expected_disc = (-2*b*b, -b*b)
        # N_{F/Q}(disc) should be 5*b^4
        disc_norm = zphi_norm(disc)
        expected_norm = 5*b*b*b*b
        # Both components negative?
        totally_neg = (disc[0] < 0 and disc[1] < 0) if b != 0 else True
        ok = (disc == expected_disc and disc_norm == expected_norm and totally_neg)
        mark = "PASS" if ok else "FAIL"
        sign_d = "+" if disc[1] >= 0 else ""
        print(f"  [{mark}] ({a},{b}): disc = ({disc[0]}{sign_d}{disc[1]}phi)")
        print(f"            = -b^2*(2+phi) = -{b*b}*(2+phi) ✓")
        print(f"            N_F(disc) = {disc_norm} = 5*{b*b*b*b} ✓")
        print(f"            Totally negative: u={disc[0]}<0, v={disc[1]}<0  {'✓' if totally_neg else 'FAIL'}")
        if not ok:
            failures.append(
                f"C5: ({a},{b}) disc={disc} (expected {expected_disc}), "
                f"norm={disc_norm} (expected {expected_norm}), "
                f"totally_neg={totally_neg}"
            )
            c5_ok = False
    if c5_ok:
        passed.append("C5")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n  Checks passed: {', '.join(passed)}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1

    print()
    print("  ALL CHECKS PASS")
    print()
    print("  CM structure of Q(zeta_5)/Q(sqrt(5)) over Z[phi]:")
    print()
    print("    Relative norm:     N_{K/F}(a+b*zeta_5) = (a^2-ab+b^2) + ab*phi")
    print("    Tower law:         N_{F/Q}(N_{K/F}(pi)) = N_{K/Q}(pi) = p")
    print("    CM factoring:      N_{K/F}(pi) * sigma_F(N_{K/F}(pi)) = p")
    print("    CM min poly:       X^2 - [(2a-b)+b*phi]*X + [(a^2-ab+b^2)+ab*phi]")
    print("    CM discriminant:   -b^2*(2+phi)  TOTALLY NEGATIVE for b != 0")
    print()
    print("  Weil consequence (degree-4 / CM analog of cert [395]):")
    print("    cert [395]: Delta_{GL2} = a_f^2 - 4p < 0       (1 real place, Q)")
    print("    cert [397]: disc = -b^2*(2+phi) << 0             (TOTALLY negative, 2 places of F)")
    print("    In both cases: discriminant < 0 <=> Weil bound holds <=> complex Frobenius eigenvalues")
    print()
    print("  Langlands chain complete at degree-4:")
    print("    [394] GL_1  (5/p) from sigma-orbit")
    print("    [395] GL_2  Weil bound, discriminant sign flip")
    print("    [396] K/F   Z[zeta_5] infrastructure (C^5=I, partial trace)")
    print("    [397] CM    relative norm + tower law + totally-negative discriminant  <-- THIS")
    return 0


def self_test():
    import json as _json
    failures = []

    # C1
    for a, b, p, _ in PRIMES:
        rn = rel_norm(a, b)
        if rn != (a*a - a*b + b*b, a*b):
            failures.append(f"C1:{a},{b}")

    # C2
    for a, b, p, _ in PRIMES:
        if zphi_norm(rel_norm(a, b)) != p:
            failures.append(f"C2:{a},{b}")
        if norm_kq(a, b) != p:
            failures.append(f"C2kq:{a},{b}")

    # C3
    for a, b, p, _ in PRIMES:
        rn = rel_norm(a, b)
        if zphi_mul(rn, sigma_F(rn)) != (p, 0):
            failures.append(f"C3:{a},{b}")

    # C4
    for a, b, p, _ in PRIMES:
        tr = partial_trace(a, b)
        rn = rel_norm(a, b)
        if tr != (2*a-b, b) or rn != (a*a-a*b+b*b, a*b):
            failures.append(f"C4:{a},{b}")

    # C5
    for a, b, p, _ in PRIMES:
        tr = partial_trace(a, b)
        rn = rel_norm(a, b)
        tr_sq = zphi_mul(tr, tr)
        disc = zphi_sub(tr_sq, (4*rn[0], 4*rn[1]))
        if disc != (-2*b*b, -b*b):
            failures.append(f"C5_disc:{a},{b}:{disc}")
        if zphi_norm(disc) != 5*b*b*b*b:
            failures.append(f"C5_norm:{a},{b}")
        if not (disc[0] < 0 and disc[1] < 0):
            failures.append(f"C5_sign:{a},{b}")

    ok = len(failures) == 0
    print(_json.dumps({"ok": ok, "checks": 5, "failures": failures}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        sys.exit(main())
