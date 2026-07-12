#!/usr/bin/env python3
# QA_COMPLIANCE = "EXACT integer arithmetic: icosian quaternion coordinates are Z[phi] elements (a+b*phi)/2 stored as integer pairs; Hamilton product, norm and group closure are all exact integer operations. No floats, no rounding, no observer exemption."
"""
Exact remediation of Phase K (qa_icosahedral_quasicrystal.py): the 120 icosians form the
binary icosahedral group 2I -- verified there by float rounding to 6 decimals (the weakest
check in the arc; rounding can mask a non-closure). Redone here in EXACT Z[phi] arithmetic.

Every icosian coordinate is (a + b*phi)/2 with a,b integers:
  0=(0,0)/2  1=(2,0)/2  1/2=(1,0)/2  phi/2=(0,1)/2  (phi-1)/2=(-1,1)/2   (phi=(1+sqrt5)/2)
Z[phi] arithmetic is exact: (a,b)+(c,d)=(a+c,b+d); (a,b)*(c,d)=(ac+bd, ad+bc+bd) via phi^2=phi+1.
The quaternion Hamilton product of two denom-2 icosians has denom-4 integer-pair coordinates;
the 120 icosians re-expressed at denom 4 form the comparison set. Group closure (14400/14400)
and unit norm (=1 exactly) are then EXACT integer facts, with no float or rounding anywhere.
"""
from __future__ import annotations
import itertools


# ---- Z[phi] exact integer-pair arithmetic ----
def za(x, y):
    return (x[0] + y[0], x[1] + y[1])


def zs(x, y):
    return (x[0] - y[0], x[1] - y[1])


def zm(x, y):
    a, b = x
    c, d = y
    return (a * c + b * d, a * d + b * c + b * d)      # phi^2 = phi + 1


def qmul(q, p):
    """Hamilton product of quaternions whose 4 coords are Z[phi] pairs.
    Inputs at denom D -> output numerators at denom D^2 (here 2 -> 4)."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    w = zs(zs(zs(zm(w1, w2), zm(x1, x2)), zm(y1, y2)), zm(z1, z2))
    x = za(za(zm(w1, x2), zm(x1, w2)), zs(zm(y1, z2), zm(z1, y2)))
    y = za(za(zs(zm(w1, y2), zm(x1, z2)), zm(y1, w2)), zm(z1, x2))
    z = za(za(za(zm(w1, z2), zm(x1, y2)), zs((0, 0), zm(y1, x2))), zm(z1, w2))
    return (w, x, y, z)


def even_perms():
    return [p for p in itertools.permutations(range(4))
            if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]


def icosians_denom2():
    """The 120 icosians, each coord = (a+b*phi)/2 as an integer pair (denominator 2 implicit)."""
    Z, HALF, PHIH, PM = (0, 0), (1, 0), (0, 1), (-1, 1)   # 0, 1/2, phi/2, (phi-1)/2  (numerators/2)
    ONE = (2, 0)                                          # 1 = 2/2
    def neg(t):
        return (-t[0], -t[1])
    pts = set()
    # 8: (+-1,0,0,0) and permutations
    for i in range(4):
        for s in (ONE, neg(ONE)):
            q = [Z, Z, Z, Z]; q[i] = s
            pts.add(tuple(q))
    # 16: (+-1/2)^4
    for signs in itertools.product((HALF, neg(HALF)), repeat=4):
        pts.add(tuple(signs))
    # 96: even permutations of (0, +-1/2, +-phi/2, +-(phi-1)/2)
    base = [Z, HALF, PHIH, PM]
    for p in even_perms():
        vals = [base[p[i]] for i in range(4)]
        nz = [i for i in range(4) if vals[i] != Z]
        for sgn in itertools.product((1, -1), repeat=len(nz)):
            q = list(vals)
            for idx, sg in zip(nz, sgn):
                q[idx] = neg(vals[idx]) if sg < 0 else vals[idx]
            pts.add(tuple(q))
    return sorted(pts)


def to_denom4(q):
    """Re-express a denom-2 quaternion (coords (a,b)/2) at denom 4: coords (2a,2b)/4."""
    return tuple((2 * a, 2 * b) for (a, b) in q)


def norm_denom4(q_d4):
    """Sum of squares of the 4 coords, as a Z[phi] pair at denom 16 (coords at denom 4)."""
    s = (0, 0)
    for c in q_d4:
        s = za(s, zm(c, c))                              # (coord)^2, denom 16
    return s                                             # unit norm 1 -> (16,0) at denom 16


def run():
    print("Exact Z[phi] verification: the 120 icosians are the group 2I (no floats, no rounding)\n")
    ic2 = icosians_denom2()
    ic4 = {to_denom4(q) for q in ic2}
    print(f"[1] constructed {len(ic2)} distinct icosians (expected 120): {len(ic2) == 120}")

    # unit norm exactly: sum of squared coords = 1  (=(16,0) at denom 16)
    norms_ok = all(norm_denom4(to_denom4(q)) == (16, 0) for q in ic2)
    print(f"[2] every icosian has EXACT unit norm (Z[phi] sum-of-squares = 1): {norms_ok}")

    # group closure exactly: every product of two icosians is an icosian
    closed = 0
    for q in ic2:
        for p in ic2:
            if qmul(q, p) in ic4:                        # product at denom 4, compared exactly
                closed += 1
    total = len(ic2) * len(ic2)
    print(f"[3] EXACT group closure: {closed}/{total} products land back in the set "
          f"-> {'CLOSED = binary icosahedral group 2I (order 120)' if closed == total else 'NOT closed'}")

    ok = len(ic2) == 120 and norms_ok and closed == total
    print("\nVERDICT (exact remediation of Phase K):")
    print(f"  * The 120 icosians (600-cell / E8 minimal-vector structure) form the group 2I,")
    print(f"    verified in PURE EXACT Z[phi] INTEGER arithmetic: {closed}/{total} closure, exact")
    print(f"    unit norms, zero floats, zero rounding, no RT1_OBSERVER_FILE exemption.")
    print(f"  * This upgrades Phase K's float-rounded closure (14400/14400 at 6 decimals) to an")
    print(f"    exact integer proof -- confirming the result and removing the weakest check in the")
    print(f"    golden arc. The golden/icosian/E8 structure is EXACTLY discrete-constructible, as")
    print(f"    the retraction (now corrected) failed to recognize.")
    print(f"\n  STATUS: {'EXACT -- 2I closure proven in Z[phi] integers' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
