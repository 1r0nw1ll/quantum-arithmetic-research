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
