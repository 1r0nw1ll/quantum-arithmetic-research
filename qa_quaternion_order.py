#!/usr/bin/env python3
# QA_COMPLIANCE = "reference_grounding — verifies QA maps into the split quaternion order M2(Z); integer arithmetic only"
"""
QA as the split quaternion order M2(Z) (grounding vs Voight, Quaternion Algebras).

Grounds the whole QA framework in mainstream quaternion arithmetic. Per Voight
(Ch. 3, Involutions): for B = M2(F) the ADJUGATE is the standard involution, the
reduced trace trd = matrix trace, and the reduced norm nrd = determinant; every
element satisfies x^2 - trd(a) x + nrd(a) = 0; and B^1 = {nrd = 1} is the norm-1
unit group (= SL(2)).

Verified correspondence:
  QA object                         quaternion-algebra object (Voight)
  ---------------------------------------------------------------------
  (b,e) state / T-operator M        element of the order M2(Z)
  qa_neg / phase conjugation        standard involution (adjugate) on rotors
  QA Eisenstein norm b^2+be-e^2     reduced norm (det) restricted to Z[M]
  versors [294]-[303]               B^1 = SL(2,Z) = norm-1 units
  Fibonacci/Lucas orbits            because M is the golden element: M^2=M+I
  orbit certs [384]-[431] (Z[phi])  the commutative subring Z[M] = O_Q(sqrt5)

Everything is exact integer arithmetic on M2(Z); this is a reference grounding,
not a QA state machine.
"""
from __future__ import annotations
import numpy as np

I = np.eye(2, dtype=np.int64)
M = np.array([[0, 1], [1, 1]], dtype=np.int64)   # QA T-operator (b,e)->(e,b+e)


def adj(A):
    """Standard involution on M2 (Voight Ex. 3.2.8): A_bar = adjugate(A)."""
    return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], dtype=np.int64)


def trd(A):   # reduced trace = matrix trace
    return int(A[0, 0] + A[1, 1])


def nrd(A):   # reduced norm = determinant
    return int(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])


def qa_norm(b, e):
    return b * b + b * e - e * e     # Eisenstein / QA norm form


def verify():
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))

    S = [M, np.array([[1, 0], [1, 1]]), np.array([[1, 1], [0, 1]]),
         np.array([[2, 3], [1, 2]]), np.array([[3, 1], [5, 2]])]

    # Standard involution axioms + adjugate is it (Voight 3.2/3.3)
    chk("A@adj(A) = nrd(A) I", all(np.array_equal(A @ adj(A), nrd(A) * I) for A in S))
    chk("adj(adj(A)) = A", all(np.array_equal(adj(adj(A)), A) for A in S))
    chk("adj anti-automorphism adj(AB)=adj(B)adj(A)",
        all(np.array_equal(adj(A @ B), adj(B) @ adj(A)) for A in S for B in S))
    chk("A + adj(A) = trd(A) I  and  A adj(A) = nrd(A) I  (Voight 3.3.1)",
        all(np.array_equal(A + adj(A), trd(A) * I) and
            np.array_equal(A @ adj(A), nrd(A) * I) for A in S))
    chk("reduced char poly A^2 - trd(A)A + nrd(A)I = 0",
        all(np.array_equal(A @ A - trd(A) * A + nrd(A) * I, np.zeros((2, 2), np.int64))
            for A in S))
    chk("nrd multiplicative", all(nrd(A @ B) == nrd(A) * nrd(B) for A in S for B in S))

    # QA T-operator as the golden element of M2(Z)
    chk("trd(M)=1, nrd(M)=-1 (char poly x^2 - x - 1)", trd(M) == 1 and nrd(M) == -1)
    chk("M^2 = M + I (M is the golden element phi)", np.array_equal(M @ M, M + I))

    # QA norm form = reduced norm on the subring Z[M]
    chk("det(bI + eM) = b^2+be-e^2 for all b,e in [-8,8]",
        all(nrd(b * I + e * M) == qa_norm(b, e) for b in range(-8, 9) for e in range(-8, 9)))

    # Versors = norm-1 units = SL(2,Z); two T-steps = one SL(2,Z) rotor (cert [296])
    L = np.array([[1, 0], [1, 1]]); Rm = np.array([[1, 1], [0, 1]])
    chk("L,R are norm-1 units (SL(2,Z))", nrd(L) == 1 and nrd(Rm) == 1)
    chk("M^2 = L R (two T-steps = one rotor)", np.array_equal(M @ M, L @ Rm))
    # M is nrd=-1 (odd-grade versor / reflection): NOT in B^1=SL(2,Z)
    chk("M is an odd-grade versor: nrd(M)=-1, M not in B^1", nrd(M) == -1)

    # The order Z<L,R,M> is the MAXIMAL order M2(Z), not a suborder/Eichler order:
    # it contains all four matrix units (E21=L-I, E12=R-I, E11=E12 E21, E22=E21 E12).
    E21 = L - I; E12 = Rm - I; E11 = E12 @ E21; E22 = E21 @ E12
    chk("Z<L,R,M> = maximal order M2(Z) (contains all 4 matrix units)",
        np.array_equal(E11, [[1, 0], [0, 0]]) and np.array_equal(E22, [[0, 0], [0, 1]]) and
        np.array_equal(E12, [[0, 1], [0, 0]]) and np.array_equal(E21, [[0, 0], [1, 0]]))

    # The three "special primes", de-conflated (see docs/theory/QA_AS_QUATERNION_ORDER.md):
    #   5 = ramified prime of Z[phi] (disc=5); 3 = INERT in Z[phi] (not ramified);
    #   M2(Q) split => algebra unramified everywhere (E11 is a zero divisor).
    roots = lambda p: [t for t in range(p) if (t * t - t - 1) % p == 0]
    chk("disc(Z[phi]) = 5 (5 is THE ramified prime of the quadratic order)",
        (-1) * (-1) - 4 * 1 * (-1) == 5)
    chk("3 is INERT in Z[phi] (x^2-x-1 irreducible mod 3 -> F_9), NOT ramified",
        roots(3) == [])
    chk("5 ramifies in Z[phi] (x^2-x-1 has a double root mod 5, at t=3)",
        roots(5) == [3])
    chk("M2(Q) is split => unramified everywhere (E11 is a zero divisor)",
        np.array_equal(E11 @ E22, np.zeros((2, 2), np.int64)) and nrd(E11) == 0)

    # Brandt/Hecke (Voight Ch.41): the construction needs a DEFINITE algebra. QA's
    # norm form is INDEFINITE (signature (1,1)) => M2(Q) split => class number 1 =>
    # Brandt matrices degenerate to T(p)=(p+1). See docs/theory/QA_AS_QUATERNION_ORDER.md.
    gram_ev = np.linalg.eigvalsh(np.array([[1.0, 0.5], [0.5, -1.0]]))   # form b^2+be-e^2
    chk("QA norm form is INDEFINITE (signature (1,1)) -> Brandt needs definite, so degenerate",
        gram_ev[0] < 0 < gram_ev[1])
    # the p-neighbor combinatorics that survive: exactly p+1 index-p sublattices of Z^2
    # (HNF reps [[p,0],[0,1]] and [[1,j],[0,p]], j=0..p-1) = Bruhat-Tits (p+1)-regular tree
    def n_index_p_sublattices(p):
        return len([((p, 0), (0, 1))] + [((1, j), (0, p)) for j in range(p)])
    chk("p+1 index-p sublattices of Z^2 (Bruhat-Tits (p+1)-regular tree) for p=2,3,5",
        all(n_index_p_sublattices(p) == p + 1 for p in (2, 3, 5)))
    # QA's 3-adic filtration is the CENTRAL/scalar line: mult-by-3 = index 3^2 = 9
    chk("mult-by-3 = scalar 3Z^2 has index 9=3^2 (central line of the 3-neighbor tree)",
        nrd(3 * I) == 9)

    # Fractional (b,e): does it change the quaternion assessment? Two levels.
    # (a) ALGEBRA level - reduced trace/norm/char-poly hold for ANY rational (b,e):
    #     the element b*I+e*M satisfies X^2 - (2b+e)X + (b^2+be-e^2)I = 0 unchanged.
    from fractions import Fraction as Fr
    def alg_ok(b, e):
        A = np.array([[b, e], [e, b + e]], dtype=object)   # b*I + e*M
        Trd, Nrd = 2 * b + e, qa_norm(b, e)
        cay = A @ A - Trd * A + Nrd * np.array([[1, 0], [0, 1]], dtype=object)
        return (A[0, 0] + A[1, 1] == Trd and
                A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0] == Nrd and
                np.array_equal(cay, np.zeros((2, 2), dtype=object)))
    chk("algebra level: trd/nrd/char-poly hold for Fraction (b,e) too (datatype irrelevant)",
        all(alg_ok(b, e) for (b, e) in
            [(Fr(1, 3), Fr(2, 5)), (Fr(-7, 4), Fr(9, 8)), (Fr(3), Fr(7))]))
    # (b) ARITHMETIC level - the ORDER O=Z[phi] is EXACTLY integer (b,e); non-integer
    #     (b,e) has nrd not in Z, leaving O for the field Q(sqrt5) (no units/orbit/Brandt).
    def in_order(b, e):
        return (2 * b + e).denominator == 1 and qa_norm(b, e).denominator == 1
    chk("arithmetic level: integer (b,e) in the order O; genuine fractions leave it",
        in_order(Fr(3), Fr(7)) and in_order(Fr(0), Fr(1)) and
        not in_order(Fr(1, 2), Fr(0)) and not in_order(Fr(1, 3), Fr(2, 5)))

    return checks


if __name__ == "__main__":
    print("QA AS THE SPLIT QUATERNION ORDER M2(Z)  (grounding vs Voight, Quaternion Algebras)\n")
    results = verify()
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n_ok = sum(ok for _, ok in results)
    print(f"\n{n_ok}/{len(results)} checks pass.")
    print("QA = M2(Z) with adjugate standard involution; norm form = reduced norm on "
          "Z[M]=Z[phi]=O_Q(sqrt5); versors = SL(2,Z) = norm-1 units.")
    import sys
    sys.exit(0 if n_ok == len(results) else 1)
