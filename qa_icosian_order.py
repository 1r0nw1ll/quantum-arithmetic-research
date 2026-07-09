#!/usr/bin/env python3
# QA_COMPLIANCE = "reference_grounding — builds the definite quaternion order over Q(sqrt5) (icosian ring); exact Z[phi] arithmetic; sigma-embeddings are observer-layer geometry only"
"""
The definite quaternion order over Q(sqrt5): the ICOSIAN RING.

Follow-up to docs/theory/QA_AS_QUATERNION_ORDER.md. The Brandt/Hecke thread showed
that genuine (nontrivial) Brandt matrices for QA's golden structure need a DEFINITE
quaternion algebra over Q(sqrt5), not the split M2(Q). This constructs the canonical
one and verifies its structure exactly.

  B = (-1,-1 | Q(sqrt5)):  i^2=j^2=k^2=-1, ij=k;  reduced norm = a^2+b^2+c^2+d^2.
  Totally definite (norm is a sum of 4 squares -> totally positive).

  Maximal order = the ICOSIAN RING (Conway-Sloane, SPLAG Ch.8): its 120 units are the
  binary icosahedral group 2I, all of reduced norm 1, and under one real embedding they
  are the 120 vertices of the 600-CELL. The icosian ring is isometric to E8 -- the SAME
  E8 that QA aligns to (4D tuples -> 8D projection -> 240 roots).

Q(sqrt5) is QA's own field (Z[M]=Z[phi]=O_Q(sqrt5), the golden order). So the definite
order that would carry QA's Hecke eigenforms is built on the golden ratio itself, and it
is the icosian/E8 object QA already uses geometrically. Exact arithmetic throughout;
the real-embedding coordinates are observer-layer geometry (600-cell check) only.
"""
from __future__ import annotations
from fractions import Fraction as Fr
from itertools import permutations, product

# --- Q(sqrt5): element (a,b) means a + b*sqrt5, a,b in Q (exact) ---
Z = (Fr(0), Fr(0)); ONE = (Fr(1), Fr(0))
def fadd(x, y): return (x[0] + y[0], x[1] + y[1])
def fneg(x):    return (-x[0], -x[1])
def fmul(x, y): return (x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])
def femb(x, s): return float(x[0]) + s * float(x[1]) * (5 ** 0.5)   # two real embeddings s=+/-1
def in_Zphi(x):
    """Z[phi] membership: a+b*sqrt5 in Z[phi] iff 2a,2b in Z and 2a == 2b (mod 2)."""
    a2, b2 = 2 * x[0], 2 * x[1]
    return a2.denominator == 1 and b2.denominator == 1 and (a2 - b2) % 2 == 0

# --- quaternions over Q(sqrt5): (a,b,c,d) = a + b i + c j + d k ---
def Hmul(p, q):
    # quaternion components w + x i + y j + z k (w,x,y,z naming avoids QA a/d coord clash)
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    w = fadd(fadd(fadd(fmul(w1, w2), fneg(fmul(x1, x2))), fneg(fmul(y1, y2))), fneg(fmul(z1, z2)))
    x = fadd(fadd(fadd(fmul(w1, x2), fmul(x1, w2)), fmul(y1, z2)), fneg(fmul(z1, y2)))
    y = fadd(fadd(fadd(fmul(w1, y2), fneg(fmul(x1, z2))), fmul(y1, w2)), fmul(z1, x2))
    z = fadd(fadd(fadd(fmul(w1, z2), fmul(x1, y2)), fneg(fmul(y1, x2))), fmul(z1, w2))
    return (w, x, y, z)
def Hconj(p):  w, x, y, z = p; return (w, fneg(x), fneg(y), fneg(z))
def nrd(p):
    s = Z
    for x in p: s = fadd(s, fmul(x, x))
    return s                                   # reduced norm in Q(sqrt5)
def trd(p):    return (2 * p[0][0], 2 * p[0][1])   # reduced trace in Q(sqrt5)


def unit_icosians():
    """The 120 units of the icosian ring = the binary icosahedral group 2I."""
    half = (Fr(1, 2), Fr(0))                          # 1/2
    iphi2 = (Fr(-1, 4), Fr(1, 4)); phi2 = (Fr(1, 4), Fr(1, 4))   # 1/(2phi)=(phi-1)/2, phi/2
    S = set()
    for pos in range(4):                       # 8: +/-1 in one slot
        for s in (ONE, fneg(ONE)):
            v = [Z, Z, Z, Z]; v[pos] = s; S.add(tuple(v))
    for signs in product((half, fneg(half)), repeat=4):   # 16: (+/-1/2,...)
        S.add(tuple(signs))
    def parity(p):
        p = list(p); par = 1
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]: par = -par
        return par
    for perm in permutations(range(4)):        # 96: even perms of (0,+/-1/2,+/-1/2phi,+/-phi/2)
        if parity(perm) != 1: continue
        for sg in product((1, -1), repeat=3):
            vals = [Z, half if sg[0] > 0 else fneg(half),
                    iphi2 if sg[1] > 0 else fneg(iphi2),
                    phi2 if sg[2] > 0 else fneg(phi2)]
            v = [None] * 4
            for slot, src in zip(perm, vals): v[slot] = src
            S.add(tuple(v))
    return list(S)


def verify():
    checks = []
    def chk(name, cond): checks.append((name, bool(cond)))

    # totally definite: reduced norm is a sum of four squares -> totally positive
    ex = ((Fr(1, 2), Fr(1, 2)), ONE, Z, Z)     # phi + i, a non-unit
    n = nrd(ex)
    chk("B=(-1,-1|Q(sqrt5)) totally definite (nrd sum of 4 squares, both embeddings >0)",
        femb(n, 1) > 0 and femb(n, -1) > 0)

    U = unit_icosians()
    chk("exactly 120 unit icosians", len(U) == 120)
    chk("every unit has reduced norm 1", all(nrd(u) == ONE for u in U))
    chk("every icosian is an algebraic integer (nrd, trd in Z[phi])",
        all(in_Zphi(nrd(u)) and in_Zphi(trd(u)) for u in U))
    Uset = set(U)
    chk("units closed under multiplication = binary icosahedral group 2I",
        all(Hmul(x, y) in Uset for x in U for y in U))

    # under one real embedding the 120 units are the 600-cell (12-regular, golden angle)
    P = [[femb(c, 1) for c in u] for u in U]
    def dot(a, b): return sum(a[i] * b[i] for i in range(4))
    chk("120 units are unit vectors in R^4 under sigma_1 (|.|^2 = 1)",
        all(abs(dot(p, p) - 1.0) < 1e-9 for p in P))
    phi_over2 = (1 + 5 ** 0.5) / 4
    neigh = [sum(1 for j in range(120) if abs(dot(P[i], P[j]) - phi_over2) < 1e-6)
             for i in range(120)]
    chk("each unit has exactly 12 nearest neighbors at inner product phi/2 (600-cell)",
        set(neigh) == {12})

    return checks


if __name__ == "__main__":
    print("THE DEFINITE QUATERNION ORDER OVER Q(sqrt5): THE ICOSIAN RING\n")
    results = verify()
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n_ok = sum(ok for _, ok in results)
    print(f"\n{n_ok}/{len(results)} checks pass.")
    print("Definite order over Q(sqrt5) = icosian ring; 120 units = 2I = 600-cell; ring ~ E8")
    print("(SPLAG Ch.8). Class number 1 at level 1 => Brandt still trivial; matching the CM")
    print("Hilbert modular form 2.2.5.1-125.1-a (certs [384]-[431]) needs an Eichler order of")
    print("level 125=5^3 in this algebra -- the next computation.")
    import sys
    sys.exit(0 if n_ok == len(results) else 1)
