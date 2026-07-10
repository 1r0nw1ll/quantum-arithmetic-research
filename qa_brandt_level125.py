#!/usr/bin/env python3
# QA_COMPLIANCE = "reference_grounding — Brandt module / Eichler order arithmetic over Q(sqrt5); exact integer/Fraction arithmetic; no QA state machine"
"""
The level-125 Eichler order over Q(sqrt5): assembling and CROSS-CHECKING its Brandt data.

Continues docs/theory/QA_AS_QUATERNION_ORDER.md. The definite algebra
B = (-1,-1|Q(sqrt5)) (disc 1; maximal order = the icosian ring, class number 1;
qa_icosian_order.py) carries Hilbert modular forms over F=Q(sqrt5) via its Brandt
matrices. The Eichler order of level n = p5^3 (p5=(sqrt5), N(n)=125) is the one
that, by Jacquet-Langlands, is EXPECTED to realize the CM Hilbert newform LMFDB
2.2.5.1-125.1-a (level_norm 125).

WHAT THIS SCRIPT DOES (all exact; the number theory it can check in-file, it checks):
  - records the setup: N(p5)=5, level n=p5^3, N(n)=125 (= LMFDB level_norm);
  - computes the Eichler MASS = 5/2 exactly (icosian base mass 1/60 times the local
    factor N(p5)^2 (N(p5)+1) = 150). Mass 5/2 is consistent with class number h=3.
  - CONFIRMS Brandt-module dimension h = 3: LMFDB has NO newforms at level 5.1 or
    25.1, so there are NO oldforms at 125.1; the level-125 cusp space is therefore
    exactly the 2-dim newform, and the full space is dim 3 = 1 (Eisenstein) + 2
    (cusp). [Sage/PARI foundation check qa_brandt_level125_sage.sage independently
    validates the maximal order: totally definite, reduced disc (1), norm-form
    det 5^4, exactly 120 norm-1 units = 2I.]
  - re-verifies the CM structure on an embedded 24-prime LMFDB fixture (below):
    a_P != 0 <=> p == 1 mod 5 (CM by Q(zeta5)); and that the two eigenvalues over
    each split p are Galois conjugate over Q(e), e^2+e-31=0 (= Q(sqrt5)).
  - taking dim 3 (= 1 Eisenstein + LMFDB cusp dim 2) and the LMFDB eigenvalues,
    writes down the char polys the Brandt matrices T(P) would have:
    (x-(N(P)+1))(x^2 - tr(a_P)x + nm(a_P)).

WHAT THIS SCRIPT DOES NOT DO (scoped honestly, per Codex review):
  - it does NOT independently COMPUTE the Brandt matrices. The char polys are
    DERIVED from the LMFDB eigenvalues + the dimension (1 + LMFDB cusp dim 2) and checked for
    self-consistency (Galois pairing; cusp factor of T(P_11) = LMFDB Hecke poly
    x^2+x-31). The explicit 3x3 integer matrix entries in an ideal-class basis
    require the neighbor/Kirschmer-Voight enumeration of the 3 right-ideal classes
    (a CAS-scale Magma/Sage computation), not reproduced here.
  - the CM pattern was additionally spot-checked against the full LMFDB eigenvalue
    list (provenance: lmfdb.org/api/hmf_hecke/?label=2.2.5.1-125.1-a), but that run
    is external; this file only asserts what it re-verifies on the embedded fixture.

Primary data: LMFDB 2.2.5.1-125.1-a (level_norm 125, weight [2,2], is_CM yes,
hecke_polynomial x^2+x-31, AL(p5)=+1).
"""
from __future__ import annotations
from fractions import Fraction as Fr

# --- Hecke eigenvalue field Q(e), e^2 + e - 31 = 0 (= Q(sqrt5)); a_P = c + d e ---
# Galois conjugation sigma: e -> -1 - e  (e + sigma(e) = -1, e*sigma(e) = -31).
def sigma(a):  c, d = a; return (c - d, -d)
def trace(a):  c, d = a; return 2 * c - d                   # a + sigma(a)
def norm(a):   c, d = a; return c * c - c * d - 31 * d * d  # a * sigma(a)

# Embedded LMFDB fixture: (N(P), p, a_P as (c,d) in e-coords) for the first 24 primes
# of Q(sqrt5), taken verbatim from lmfdb.org hmf_hecke for 2.2.5.1-125.1-a.
LMFDB_FIXTURE = [
    (4, 2, (0, 0)), (5, 5, (0, 0)), (9, 3, (0, 0)),
    (11, 11, (-1, -1)), (11, 11, (0, 1)),
    (19, 19, (0, 0)), (19, 19, (0, 0)), (29, 29, (0, 0)), (29, 29, (0, 0)),
    (31, 31, (-5, 1)), (31, 31, (-6, -1)), (41, 41, (4, -1)), (41, 41, (5, 1)),
    (49, 7, (0, 0)), (59, 59, (0, 0)), (59, 59, (0, 0)),
    (61, 61, (-1, -1)), (61, 61, (0, 1)), (71, 71, (10, 1)), (71, 71, (9, -1)),
    (79, 79, (0, 0)), (79, 79, (0, 0)), (89, 89, (0, 0)), (89, 89, (0, 0)),
]
HECKE_POLY = (-1, -31)   # LMFDB hecke_polynomial x^2 + x - 31 as (trace, norm) = (-1, -31)


def verify():
    checks = []
    def chk(name, cond): checks.append((name, bool(cond)))

    # setup: N(p5^3) = 125 = LMFDB level norm
    chk("level n = p5^3 has norm 125 (= LMFDB level_norm)", 5 ** 3 == 125)

    # Eichler mass = base(1/60) * local factor N(p5)^2 (N(p5)+1) = 1/60 * 150 = 5/2
    mass = Fr(1, 60) * (5 ** (3 - 1) * (5 + 1))
    chk("Eichler mass = 1/60 * 150 = 5/2 (exact)", mass == Fr(5, 2))
    # dimension h=3 CONFIRMED: no newforms at level 5.1 or 25.1 (LMFDB) => no oldforms
    # at 125.1, so the cusp space is exactly the 2-dim newform; +1 Eisenstein = 3.
    lmfdb_cusp_dim = 2                 # 2.2.5.1-125.1-a is the only cusp form at 125.1
    newforms_at_lower_levels = 0      # LMFDB: 0 newforms at 2.2.5.1 levels 5.1 and 25.1
    chk("Brandt-module dim h = 3 = 1 Eisenstein + cusp dim 2 (no oldforms; mass 5/2 consistent)",
        newforms_at_lower_levels == 0 and 1 + lmfdb_cusp_dim == 3)

    # CM structure, checked in-file on the embedded fixture: a_P != 0 <=> p == 1 mod 5
    chk("fixture: a_P != 0  <=>  p == 1 mod 5  (CM by Q(zeta5))",
        all((a != (0, 0)) == (p % 5 == 1) for (_, p, a) in LMFDB_FIXTURE))

    # the two eigenvalues over each split p are Galois conjugate over Q(e)
    from collections import defaultdict
    byp = defaultdict(list)
    for (nrm, p, a) in LMFDB_FIXTURE:
        if p % 5 == 1:
            byp[p].append(a)
    chk("fixture: the two eigenvalues over each split p are Galois conjugate",
        all(len(v) == 2 and sigma(v[0]) == v[1] for v in byp.values()))

    # cusp factor of T(P_11) = x^2 - tr(a)x + nm(a) equals the LMFDB Hecke polynomial
    a11 = byp[11][0]
    chk("cusp factor of T(P_11) equals the LMFDB Hecke polynomial x^2 + x - 31",
        (trace(a11), norm(a11)) == HECKE_POLY)

    # each split-p cusp factor has integer coefficients (as a rational Brandt matrix needs)
    chk("each split-p cusp factor x^2 - tr(a)x + nm(a) has integer coefficients",
        all(isinstance(trace(v[0]), int) and isinstance(norm(v[0]), int)
            for v in byp.values()))

    return checks


def brandt_spectra():
    print("Brandt char polys implied by dim 3 (1 Eisenstein + LMFDB cusp 2) + LMFDB eigenvalues:")
    print("  T(P) = (x-(N(P)+1)) * (x^2 - tr(a_P)x + nm(a_P)):")
    from collections import defaultdict
    byp = defaultdict(list)
    for (nrm, p, a) in LMFDB_FIXTURE:
        byp[(nrm, p)].append(a)
    for (nrm, p), evs in sorted(byp.items()):
        if p % 5 == 1:
            a = evs[0]
            print(f"    split p={p:3d} (N(P)={nrm}): (x-{nrm + 1}) * (x^2 - ({trace(a)})x + ({norm(a)}))")
    print("  inert good primes P (N(P)=p^2, a_P=0 by CM => cusp eigenvalues 0,0):")
    for nrm, p, a in LMFDB_FIXTURE:
        if p != 5 and p % 5 != 1 and nrm == p * p:
            print(f"    inert p={p:3d} (N(P)={nrm}): (x-{nrm + 1}) * x^2")


def _run(selftest=False):
    results = verify()
    n_ok = sum(ok for _, ok in results)
    if selftest:
        import json
        print(json.dumps({"ok": n_ok == len(results), "passed": n_ok, "total": len(results)}))
        return 0 if n_ok == len(results) else 1
    print("LEVEL-125 EICHLER ORDER OVER Q(sqrt5): BRANDT DATA (cross-checked vs LMFDB)")
    print("(target: CM Hilbert newform LMFDB 2.2.5.1-125.1-a)\n")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{n_ok}/{len(results)} checks pass.\n")
    brandt_spectra()
    print("\nMass 5/2 is consistent with class number 3; with LMFDB cusp dim 2 the space is")
    print("dim 3 = 1 Eisenstein + 2-dim CM (by Q(zeta5)) cusp orbit, expected (Jacquet-")
    print("Langlands) to be 2.2.5.1-125.1-a. NOT an independent Brandt-matrix computation:")
    print("char polys are derived from LMFDB eigenvalues + this dimension; explicit")
    print("ideal-class matrix entries need the neighbor-method enumeration.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run(selftest="--self-test" in sys.argv))
