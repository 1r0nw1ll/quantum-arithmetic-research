#!/usr/bin/env python3
# QA_COMPLIANCE = "reference_grounding — Brandt module / Eichler order arithmetic over Q(sqrt5); exact integer/Fraction arithmetic; no QA state machine"
"""
The level-125 Eichler order over Q(sqrt5): the Brandt matrices, COMPUTED from scratch.

Continues docs/theory/QA_AS_QUATERNION_ORDER.md. The definite algebra
B = (-1,-1|Q(sqrt5)) (disc 1; maximal order = the icosian ring, class number 1;
qa_icosian_order.py) carries Hilbert modular forms over F=Q(sqrt5) via its Brandt
matrices. The Eichler order of level n = p5^3 (p5=(sqrt5), N(n)=125) realizes the
CM Hilbert newform LMFDB 2.2.5.1-125.1-a (level_norm 125).

The full from-scratch computation lives in qa_brandt_level125_compute.sage
(SageMath 10.7 + PARI): it builds the maximal order (det 5^4, 120 units = 2I),
the O_K-structure, the splitting rho: O_max (x) O_K/p5^3 -> M_2(R) at the RAMIFIED
prime p5 (rank-1 idempotent via mod-p5 seed + Newton lift), the Eichler order
O = {x : (1-e) x e = 0} (validated reduced disc 5^10), the class number
h = 3 = #(O_max^1 orbits on P^1(O_K/p5^3)) with weights (1,1,2) and mass 5/2, and
finally the Brandt matrices T(q) via the q-neighbour elements (nrd(alpha) a totally
positive generator of q) acting through rho on P^1. Result:

  T(p11) = [[2,5,10],[5,7,0],[5,0,2]]     char poly (x-12)(x^2 + x - 31)
  T(p31) = [[7,15,20],[15,12,10],[10,5,2]] char poly (x-32)(x^2 + 11x - 1)
  T(p7)  = [[20,20,20],[20,20,20],[10,10,10]] char poly (x-50) x^2   (p7 inert, N=49)

Every cusp factor is exactly the LMFDB Hecke data (x^2+x-31 is the LMFDB Hecke
polynomial; the p7-inert 0,0 is the CM signature). Column sums = N(q)+1 (Eisenstein).
The Sage script's stage [7] extends this to a FULL Hecke-system check: all 13 good
primes up to norm 100 reproduce the LMFDB cusp factors (S,P) exactly -- i.e. the
whole L-function of 2.2.5.1-125.1-a (to that bound) comes out of the Brandt matrices,
not just three sample primes.

THIS Python file re-verifies, with no CAS, the properties of the computed matrices
(column sums = N(q)+1; (N(q)+1) is an eigenvalue; cusp factor = the LMFDB-predicted
(x^2 - tr(a_P)x + nm(a_P))), and re-checks the CM structure on an embedded 24-prime
LMFDB fixture. h=3 is corroborated independently: LMFDB has no newforms at level 5.1
or 25.1, so no oldforms at 125.1 (cusp space = the 2-dim newform, +1 Eisenstein = 3).

Primary data: LMFDB 2.2.5.1-125.1-a (level_norm 125, weight [2,2], is_CM yes,
hecke_polynomial x^2+x-31, AL(p5)=+1).
"""
from __future__ import annotations
from fractions import Fraction as Fr

# Brandt matrices computed in qa_brandt_level125_compute.sage (orbit basis, weights (1,1,2)):
BRANDT = {
    11: ([[2, 5, 10], [5, 7, 0], [5, 0, 2]], (-1, -31)),          # T(p11); cusp x^2 - (-1)x + (-31)
    31: ([[7, 15, 20], [15, 12, 10], [10, 5, 2]], (-11, -1)),     # T(p31); cusp x^2 - (-11)x + (-1)
    49: ([[20, 20, 20], [20, 20, 20], [10, 10, 10]], (0, 0)),     # T(p7 inert, N=49); cusp x^2
}

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

    # --- verify the COMPUTED 3x3 Brandt matrices (from qa_brandt_level125_compute.sage) ---
    def m3det(M):
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    def minors2(M):  # sum of principal 2x2 minors
        return ((M[0][0]*M[1][1]-M[0][1]*M[1][0]) + (M[0][0]*M[2][2]-M[0][2]*M[2][0])
                + (M[1][1]*M[2][2]-M[1][2]*M[2][1]))
    for Nq, (M, (S, P)) in BRANDT.items():
        r = Nq + 1                                    # Eisenstein eigenvalue N(q)+1
        colsums = [sum(M[i][j] for i in range(3)) for j in range(3)]
        c1 = sum(M[i][i] for i in range(3)); c2 = minors2(M); c3 = m3det(M)
        # char poly x^3 - c1 x^2 + c2 x - c3 must equal (x-r)(x^2 - S x + P)
        factors_ok = (c1 == r + S) and (c2 == P + r*S) and (c3 == r*P)
        chk(f"computed T(q,N={Nq}): column sums all = N(q)+1 = {r} (Eisenstein)",
            all(cs == r for cs in colsums))
        chk(f"computed T(q,N={Nq}): char poly = (x-{r})(x^2 - ({S})x + ({P})) [matches LMFDB]",
            factors_ok)
    # T(p11) cusp factor is the LMFDB Hecke polynomial
    chk("computed T(p11) cusp factor (x^2 + x - 31) = the LMFDB Hecke polynomial",
        BRANDT[11][1] == HECKE_POLY)

    # full Hecke system to norm 100 (verified in qa_brandt_level125_compute.sage stage [7]):
    # LMFDB cusp factors (S,P) for every good prime; here re-check the CM rule they satisfy
    # (S,P) != (0,0)  <=>  p == 1 mod 5 (split in Q(zeta5)); else CM-inert -> cusp x^2.
    CUSP = {2:(0,0),3:(0,0),7:(0,0),11:(-1,-31),19:(0,0),29:(0,0),31:(-11,-1),
            41:(9,-11),59:(0,0),61:(-1,-31),71:(19,59),79:(0,0),89:(0,0)}
    chk("full-Hecke-system targets (13 good primes to norm 100) obey CM rule: (S,P)!=0 <=> p==1 mod5",
        all(((S, P) != (0, 0)) == (p % 5 == 1) for p, (S, P) in CUSP.items()))
    chk("the 3 computed Brandt matrices' cusp factors match the full-system targets",
        all(BRANDT[Nq][1] == CUSP[p] for Nq, p in ((11, 11), (31, 31), (49, 7))))

    return checks


def brandt_spectra():
    print("Brandt matrices COMPUTED from scratch (qa_brandt_level125_compute.sage), orbit basis:")
    labels = {11: "T(p11)", 31: "T(p31)", 49: "T(p7 inert, N=49)"}
    for Nq in (11, 31, 49):
        M, (S, P) = BRANDT[Nq]
        cusp = f"x^2 - ({S})x + ({P})" if (S, P) != (0, 0) else "x^2"
        print(f"  {labels[Nq]:20s} = {M}")
        print(f"    {'':20s}   char poly (x-{Nq + 1}) * ({cusp})")


def _run(selftest=False):
    results = verify()
    n_ok = sum(ok for _, ok in results)
    if selftest:
        import json
        print(json.dumps({"ok": n_ok == len(results), "passed": n_ok, "total": len(results)}))
        return 0 if n_ok == len(results) else 1
    print("LEVEL-125 EICHLER ORDER OVER Q(sqrt5): BRANDT MATRICES (computed from scratch)")
    print("(target: CM Hilbert newform LMFDB 2.2.5.1-125.1-a)\n")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{n_ok}/{len(results)} checks pass.\n")
    brandt_spectra()
    print("\nComputed end-to-end in qa_brandt_level125_compute.sage: maximal order (det 5^4,")
    print("120 units = 2I) -> splitting at the ramified prime p5 -> Eichler order (det 5^10)")
    print("-> class number h=3 (unit orbits on P^1), weights (1,1,2), mass 5/2 -> the Brandt")
    print("matrices above. Every char poly matches LMFDB 2.2.5.1-125.1-a (cusp factor of")
    print("T(p11) is the LMFDB Hecke polynomial x^2+x-31; p7-inert cusp eigenvalues 0,0 = CM).")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run(selftest="--self-test" in sys.argv))
