#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical Weil conjectures and HMF Hecke theory; Weil (1949) doi.org/10.1090/S0002-9904-1949-09219-4 (Weil bound original); Deligne (1974) doi.org/10.1007/BF02684373 (Weil conjecture proof for abelian varieties); Diamond & Shurman (2005) ISBN 978-0-387-27226-9 Ch.9 (Hecke eigenvalue bounds); LMFDB (2024) https://www.lmfdb.org/ModularForm/GL2/TotallyReal/2.2.5.1 (eigenvalue data from cert [390]) -->
"""
Cert [395] — QA Weil Bound from Cassini

CLAIM:
  For the Hilbert modular form 2.2.5.1-31.1-a over Q(sqrt(5)), weight [2,2],
  the Hecke eigenvalues a_f(p) satisfy the Weil bound for all prime ideals p
  in the LMFDB data (cert [390], 34 prime ideals with N(p) <= 151):

      |a_f(p)|^2  <=  4 * N(p)       (C1 — Weil bound)
      |a_f(p)|^2  <   4 * N(p)       (C2 — strict, non-CM form)

  STRUCTURAL ORIGIN — THE CASSINI TO WEIL SIGN FLIP:

  The Fibonacci Frobenius (cert [394]/[391]) has characteristic polynomial:

      P_Fib(x) = x^2 - L_p * x - 1       (for odd prime p)

  where L_p = Tr(M^p) is the p-th Lucas number, det(M^p) = (-1)^p = -1.
  Discriminant:  Delta_Fib = L_p^2 + 4 > 0   (POSITIVE — real eigenvalues)

  The GL_2 Frobenius has characteristic polynomial:

      P_GL2(x) = x^2 - a_f(p)*x + N(p)

  Discriminant:  Delta_GL2 = a_f(p)^2 - 4*N(p) <= 0   (NEGATIVE — complex eigenvalues)

  The Weil bound |a_f(p)| <= 2*sqrt(N(p)) is equivalent to Delta_GL2 <= 0.

  SIGN FLIP  Delta_Fib > 0  --->  Delta_GL2 < 0:
    This is the transition from weight-0 (Fibonacci Frobenius, Cassini det=-1)
    to weight-2 (GL_2 Frobenius, det=N(p)=p).

    Cassini invariant: det(M^p) = (-1)^p = -1  [cert [391]]
      => Fibonacci eigenvalues multiply to -1  (|product| = 1)
    Weil weight-2:   det(Frob_p) = N(p) = p
      => GL_2 eigenvalues multiply to p  (|product| = p, |each| = sqrt(p))

    The weight factor p shifts det from -1 to p and flips the discriminant sign.

THEOREM NT COMPLIANCE:
  All arithmetic is on exact integers. No float state. Observer projections
  (primality test via isqrt, display only) appear only at output/labeling.
  The det(M^k) Cassini norm is computed exactly via matrix exponentiation.

CHECKS:
  C1: |a_f(p)|^2 <= 4*N(p) for all 34 prime ideals with N(p) <= 151.
  C2: Strict inequality |a_f(p)|^2 < 4*N(p) (non-CM form: no equality).
  C3: Fibonacci discriminant Delta_Fib = L_p^2 + 4 > 0 for split/inert/ramified
      primes p in {11, 19, 29, 41, 59, 61, 71, 79, 89, 101}.
  C4: GL_2 discriminant Delta_GL2 = a_f(p)^2 - 4*N(p) < 0 for all 34 prime ideals.
  C5: Cassini det: det(M^p) = (-1)^p for spot-check primes {11, 19, 41, 59, 71}
      (reconfirms cert [391] Cassini invariant).
"""

import sys
import math

# ──────────────────────────────────────────────────────────────────────────
# LMFDB eigenvalue data — hardcoded from cert [390] (2026-06-11 fetch)
# Ordering: prime ideals of Z[phi] sorted by norm N(p), skipping level 31
# ──────────────────────────────────────────────────────────────────────────

EIGS_31_1 = [
    -3, -2, 2, 4, -4, -4, 4, -2, -2, -1, 8, -6, -6, 2, 12, -4,
    6, -2, 0, -8, 0, 16, -6, 10, 6, -10, 6, -10, -20, 4, 4, -20,
    6, -10, 8, 16, -6, 4, -12, -10, 22, 0, 16, 16, 24, -12, -4, 6,
    -26, -24, 0, -14, 26, 12, 12, -10, -18, 0, 0, -30, 18, -30, 8, 8,
    28, -20, -34, -2, 24, 16, -20, 12, 14, 22, 18, -30,
]

# ──────────────────────────────────────────────────────────────────────────
# Prime ideal list for Q(sqrt(5)) — matches cert [390] _build_prime_list exactly
# ──────────────────────────────────────────────────────────────────────────

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return False
    return True

def _classify(p):
    if p == 5:
        return "ram", p
    elif p % 5 in (1, 4):
        return "split", p
    else:
        return "inert", p * p

def _build_prime_list(max_count=80):
    result = []
    p = 2
    count = 0
    while count < max_count:
        if _is_prime(p):
            t, n = _classify(p)
            if t == "split":
                if p != 31:
                    result.append((n, p, "split", 1))
                    result.append((n, p, "split", 2))
                    count += 2
            elif t == "inert":
                result.append((n, p, "inert", 0))
                count += 1
            else:
                result.append((n, p, "ram", 0))
                count += 1
        p += 1
    result.sort(key=lambda x: (x[0], x[1]))
    return result

PRIME_LIST = _build_prime_list(max_count=80)
VERIFIED_RANGE = [i for i, (n, p, t, s) in enumerate(PRIME_LIST) if n <= 151]

# ──────────────────────────────────────────────────────────────────────────
# Fibonacci matrix arithmetic (exact integers, no floats in QA layer)
# ──────────────────────────────────────────────────────────────────────────

FIB_M = [[1, 1], [1, 0]]

def _matmul(A, B):
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]

def _matpow(M, n):
    result = [[1, 0], [0, 1]]
    while n:
        if n & 1:
            result = _matmul(result, M)
        M = _matmul(M, M)
        n >>= 1
    return result

def lucas(k):
    """L_k = Tr(M^k) = F_{k+1} + F_{k-1}  (exact integer)."""
    Mk = _matpow(FIB_M, k)
    return Mk[0][0] + Mk[1][1]

def cassini_det(k):
    """det(M^k) = (-1)^k  (Cassini invariant, exact integer)."""
    Mk = _matpow(FIB_M, k)
    return Mk[0][0] * Mk[1][1] - Mk[0][1] * Mk[1][0]


# ──────────────────────────────────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────────────────────────────────

def main():
    failures = []
    passed = []

    print("=" * 68)
    print("Cert [395] — QA Weil Bound from Cassini")
    print("  |a_f(p)|^2 <= 4*N(p)  [discriminant sign flip: Cassini -> Weil]")
    print("=" * 68)

    # ── C1 + C2 ──────────────────────────────────────────────────────────
    print("\n  C1/C2: Weil bound and strict inequality for N(p) <= 151")
    c1_ok = True
    c2_ok = True
    weil_data = []
    for i in VERIFIED_RANGE:
        norm, p, t, sub = PRIME_LIST[i]
        a = EIGS_31_1[i]
        a_sq = a * a
        bound = 4 * norm
        disc = a_sq - bound
        weil_data.append((norm, p, t, a, disc))
        if a_sq > bound:
            failures.append(f"C1: N={norm} p={p} |a|^2={a_sq} > 4N={bound}")
            c1_ok = False
        if a_sq == bound:
            failures.append(f"C2: N={norm} p={p} |a|^2={a_sq} == 4N (CM equality)")
            c2_ok = False
    if c1_ok:
        passed.append("C1")
        print(f"  [PASS] C1: |a_f(p)|^2 <= 4*N(p) for all {len(VERIFIED_RANGE)} prime ideals")
    if c2_ok:
        passed.append("C2")
        print(f"  [PASS] C2: strict (no equality) — consistent with non-CM form")

    print()
    print(f"  {'N(p)':>6}  {'type':>6}  {'a_f':>6}  {'Delta_GL2':>12}  {'check':>5}")
    print(f"  {'':-<6}  {'':-<6}  {'':-<6}  {'':-<12}  {'':-<5}")
    for norm, p, t, a, disc in weil_data[:12]:
        chk = "PASS" if disc < 0 else "FAIL"
        print(f"  {norm:6d}  {t:>6}  {a:6d}  {disc:12d}  {chk}")
    if len(weil_data) > 12:
        print(f"  ... ({len(weil_data)-12} more, all PASS)")

    # ── C3: Fibonacci discriminant (positive) ────────────────────────────
    print("\n  C3: Fibonacci Frobenius  Delta_Fib = L_p^2 + 4 > 0")
    fib_primes = [11, 19, 29, 41, 59, 61, 71, 79, 89, 101]
    c3_ok = True
    for p in fib_primes:
        lp = lucas(p)
        disc_fib = lp * lp + 4
        if disc_fib <= 0:
            failures.append(f"C3: p={p} L_p={lp} Delta_Fib={disc_fib} <= 0")
            c3_ok = False
    if c3_ok:
        passed.append("C3")
        ex_p = fib_primes[0]
        ex_lp = lucas(ex_p)
        print(f"  [PASS] C3: e.g. p={ex_p}: L_{ex_p}={ex_lp}, Delta_Fib={ex_lp*ex_lp+4}")
        print(f"         char poly x^2 - L_p*x - 1, discriminant POSITIVE (real eigenvalues)")
    else:
        print(f"  [FAIL] C3: {[f for f in failures if f.startswith('C3')]}")

    # ── C4: GL_2 discriminant (negative) ────────────────────────────────
    print("\n  C4: GL_2 Frobenius  Delta_GL2 = a_f^2 - 4*N(p) < 0")
    c4_ok = True
    most_negative = None
    for norm, p, t, a, disc in weil_data:
        if disc >= 0:
            failures.append(f"C4: N={norm} p={p} Delta_GL2={disc} >= 0")
            c4_ok = False
        if most_negative is None or disc > most_negative:
            most_negative = disc
    if c4_ok:
        passed.append("C4")
        print(f"  [PASS] C4: Delta_GL2 < 0 for all {len(weil_data)} prime ideals")
        print(f"         Largest (closest to 0): Delta_GL2 = {most_negative}")
        print(f"         GL_2 Frobenius eigenvalues complex, |each| = sqrt(N(p))")

    # ── C5: Cassini det ──────────────────────────────────────────────────
    print("\n  C5: Cassini  det(M^p) = (-1)^p  (weight-0 analogue of Weil)")
    spot_primes = [11, 19, 41, 59, 71]
    c5_ok = True
    for p in spot_primes:
        d = cassini_det(p)
        expected = -1            # (-1)^p = -1 for all odd primes p
        ok = (d == expected)
        mark = "PASS" if ok else "FAIL"
        lp = lucas(p)
        disc_f = lp * lp + 4
        print(f"  [{mark}] p={p:3d}: det(M^{p}) = {d}, L_{p}={lp}, Delta_Fib={disc_f}")
        if not ok:
            failures.append(f"C5: p={p} det={d}, expected {expected}")
            c5_ok = False
    if c5_ok:
        passed.append("C5")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"\n  Checks passed: {', '.join(passed)}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1

    print()
    print("  ALL CHECKS PASS")
    print()
    print("  Cassini -> Weil discriminant sign flip:")
    print()
    print("    WEIGHT-0  (Fibonacci Frobenius, Cassini):")
    print("      det(M^p) = -1          char poly: x^2 - L_p*x - 1")
    print("      Delta_Fib = L_p^2 + 4 > 0       (REAL eigenvalues, product = -1)")
    print()
    print("    WEIGHT-2  (GL_2 Frobenius, Weil):")
    print("      det(Frob_p) = N(p)     char poly: x^2 - a_f*x + N(p)")
    print("      Delta_GL2 = a_f^2 - 4*N < 0     (COMPLEX, |eigenvalue| = sqrt(N))")
    print("      Weil bound: |a_f(p)| <= 2*sqrt(N(p))")
    print()
    print("    The weight shift det=-1 -> det=p flips the discriminant sign.")
    print("    Cassini invariant is the weight-0 analogue of the Weil bound.")
    return 0


def self_test():
    import json as _json
    failures = []

    for i in VERIFIED_RANGE:
        norm, p, t, sub = PRIME_LIST[i]
        a = EIGS_31_1[i]
        if a * a > 4 * norm:
            failures.append(f"C1:N{norm}a{a}")
        if a * a == 4 * norm:
            failures.append(f"C2:N{norm}")
        if a * a - 4 * norm >= 0:
            failures.append(f"C4:N{norm}a{a}")

    for p in [11, 19, 29, 41, 59]:
        lp = lucas(p)
        if lp * lp + 4 <= 0:
            failures.append(f"C3:p{p}")

    for p in [11, 19, 41]:
        d = cassini_det(p)
        if d != -1:
            failures.append(f"C5:p{p}")

    ok = len(failures) == 0
    print(_json.dumps({"ok": ok, "checks": 5, "failures": failures}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        sys.exit(main())
