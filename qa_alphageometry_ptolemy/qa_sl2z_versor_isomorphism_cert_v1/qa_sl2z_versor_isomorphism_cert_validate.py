#!/usr/bin/env python3
"""
QA SL(2,Z) Versor Isomorphism Cert [296] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Hestenes and Sobczyk (1984) Clifford Algebra to Geometric Calculus, Reidel, ISBN 978-90-277-1673-6
  Brocot (1861) Calcul des rouages par approximation, Revue Chronometrique 3:186-194

Five claims:
  C1  Stern-Brocot bijection: W(b,e)*[1,1]^T = [b,e]^T for all primitive (b,e)
  C2  T-operator = Fibonacci matrix M = [[0,1],[1,1]], det(M) = -1
  C3  Rotor identity: M^2 = L*R as integer matrices
  C4  Singularity fixpoint: (9,9) is the unique fixed state of T in {1,...,9}^2
  C5  Orbit-grade: partition under M on {1,...,9}^2 is 1+8+72
"""

import sys
from math import gcd

# ---------------------------------------------------------------------------
# SL(2,Z) generators (2x2 integer matrices as nested lists)
# ---------------------------------------------------------------------------
L = [[1, 0], [1, 1]]   # (b,e) -> (b, b+e)
R = [[1, 1], [0, 1]]   # (b,e) -> (b+e, e)
M = [[0, 1], [1, 1]]   # QA T-step: (b,e) -> (e, b+e)


def mat_mul(A, B):
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0],  A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0],  A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


def mat_vec(A, v):
    return [A[0][0]*v[0] + A[0][1]*v[1],
            A[1][0]*v[0] + A[1][1]*v[1]]


def det(A):
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]


def qa_t_step(b, e, m=9):
    """QA T-operator with A1 arithmetic (states in {1,...,m})."""
    new_b = e
    new_e = ((b + e - 1) % m) + 1
    return new_b, new_e


def sb_word(b, e):
    """
    Compute Stern-Brocot word W(b,e) such that W*[1,1]^T = [b,e]^T.
    Uses subtractive Euclidean algorithm on (b,e), reading steps in reverse
    (from (b,e) back to (1,1)) to reconstruct the forward path.

    When cb > ce, the parent of (b,e) in the tree is (b-e, e), reached by R.
    When ce > cb, the parent of (b,e) is (b, e-b), reached by L.
    Collecting these parent-labels going BACKWARD gives the FORWARD word.
    """
    steps = []
    cb, ce = b, e
    while (cb, ce) != (1, 1):
        if cb > ce:
            steps.append('R')
            cb -= ce
        else:
            steps.append('L')
            ce -= cb
    # steps are collected backward (child→root); reverse to get the forward path
    word = ''.join(reversed(steps))
    # Reconstruct W by LEFT-multiplying each generator in forward order.
    # Path G_1 G_2 ... G_k maps [1,1]^T via G_k·...·G_1·[1,1]^T,
    # so we accumulate W = G_k·...·G_1 by prepending each new step.
    W = [[1, 0], [0, 1]]   # identity
    for ch in word:
        G = L if ch == 'L' else R
        W = mat_mul(G, W)  # left-multiply: W ← G·W
    return word, W, len(word)


# ---------------------------------------------------------------------------
# C1 — Stern-Brocot bijection
# ---------------------------------------------------------------------------
def check_c1(max_sum=30):
    """Verify W(b,e)*[1,1]^T = [b,e]^T for all primitive (b,e) with b+e <= max_sum."""
    failures = []
    seen_words = {}
    for b in range(1, max_sum):
        for e in range(1, max_sum - b + 1):
            if gcd(b, e) != 1:
                continue
            word, W, _ = sb_word(b, e)
            result = mat_vec(W, [1, 1])
            if result != [b, e]:
                failures.append(f"W({b},{e})={word!r} -> {result} != [{b},{e}]")
            if word in seen_words and seen_words[word] != (b, e):
                failures.append(f"Word {word!r} collision: ({b},{e}) vs {seen_words[word]}")
            seen_words[word] = (b, e)
    return failures


# ---------------------------------------------------------------------------
# C2 — T-operator = M, det(M) = -1
# ---------------------------------------------------------------------------
def check_c2():
    """Verify M*[b,e]^T = [e, b+e]^T (raw, no modular reduction)."""
    failures = []
    for b in range(1, 15):
        for e in range(1, 15):
            result = mat_vec(M, [b, e])
            if result != [e, b + e]:
                failures.append(f"M*[{b},{e}]^T = {result} != [{e},{b+e}]")
    if det(M) != -1:
        failures.append(f"det(M) = {det(M)}, expected -1")
    return failures


# ---------------------------------------------------------------------------
# C3 — M^2 = L*R as integer matrices
# ---------------------------------------------------------------------------
def check_c3():
    M2 = mat_mul(M, M)
    LR = mat_mul(L, R)
    failures = []
    if M2 != LR:
        failures.append(f"M^2 = {M2}, L*R = {LR}")
    if det(LR) != 1:
        failures.append(f"det(L*R) = {det(LR)}, expected 1")
    return failures


# ---------------------------------------------------------------------------
# C4 — Singularity fixpoint
# ---------------------------------------------------------------------------
def check_c4(m=9):
    fixed = [(b, e)
             for b in range(1, m + 1)
             for e in range(1, m + 1)
             if qa_t_step(b, e, m) == (b, e)]
    failures = []
    if fixed != [(9, 9)]:
        failures.append(f"Fixed points = {fixed}, expected [(9,9)]")
    # Direct A1 arithmetic check
    new_e = ((9 + 9 - 1) % 9) + 1
    if (9, new_e) != (9, 9):
        failures.append(f"A1 check: T(9,9) = (9,{new_e}), expected (9,9)")
    return failures


# ---------------------------------------------------------------------------
# C5 — Orbit partition 1+8+72 and grade correspondence
# ---------------------------------------------------------------------------
def check_c5(m=9):
    failures = []

    # Build all T-orbits on {1,...,m}^2
    all_states = {(b, e) for b in range(1, m + 1) for e in range(1, m + 1)}
    visited = set()
    orbits = []
    for start in sorted(all_states):
        if start in visited:
            continue
        orbit = []
        cur = start
        while cur not in visited:
            visited.add(cur)
            orbit.append(cur)
            cur = qa_t_step(*cur, m=m)
        orbits.append(orbit)

    size_counts = {}
    for o in orbits:
        size_counts[len(o)] = size_counts.get(len(o), 0) + 1

    expected = {1: 1, 8: 1, 24: 3}
    if size_counts != expected:
        failures.append(f"Orbit size distribution = {size_counts}, expected {expected}")

    sing = sum(len(o) for o in orbits if len(o) == 1)
    sat  = sum(len(o) for o in orbits if len(o) == 8)
    cos  = sum(len(o) for o in orbits if len(o) == 24)
    if (sing, sat, cos) != (1, 8, 72):
        failures.append(f"Partition (sing,sat,cos) = ({sing},{sat},{cos}), expected (1,8,72)")

    # Verify all SB-word matrices for primitive states have det=+1 (they're in SL(2,Z))
    # This is a consistency check: since L,R ∈ SL(2,Z), every product has det=+1.
    det_failures = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if gcd(b, e) != 1:
                continue
            _, W, _ = sb_word(b, e)
            if det(W) != 1:
                det_failures += 1
                failures.append(f"det(W({b},{e})) = {det(W)}, expected 1")
    if det_failures > 0:
        failures.append(f"SB word det invariant violated for {det_failures} primitive states")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_bijection",   check_c1,  {"max_sum": 30}),
        ("C2_T_equals_M",  check_c2,  {}),
        ("C3_rotor",       check_c3,  {}),
        ("C4_singularity", check_c4,  {}),
        ("C5_orbits",      check_c5,  {}),
    ]
    all_pass = True
    results = {}
    for label, fn, kwargs in checks:
        failures = fn(**kwargs)
        status = "PASS" if not failures else "FAIL"
        if failures:
            all_pass = False
        results[label] = {"status": status, "failures": failures}
        suffix = f" — {failures[0]}" if failures else ""
        print(f"  {label}: {status}{suffix}")

    print()
    if all_pass:
        print("CERT [296] PASS — QA SL(2,Z) Versor Isomorphism")
    else:
        print("CERT [296] FAIL")
        for k, v in results.items():
            for f in v["failures"]:
                print(f"  FAIL {k}: {f}")
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()
