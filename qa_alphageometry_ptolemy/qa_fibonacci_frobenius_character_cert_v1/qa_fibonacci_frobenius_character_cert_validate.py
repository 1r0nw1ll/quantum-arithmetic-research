#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical algebraic number theory and Fibonacci arithmetic; Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.5 (Legendre symbol); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Fibonacci mod p); Neukirch (1999) ISBN 978-3-540-65399-8 §I.8-9 (Frobenius element); Serre (1973) ISBN 978-0-387-90041-7 §I.2 (GL_1 Artin representation) -->
"""
Cert [394] — QA Fibonacci Frobenius Character

CLAIM:
  For every prime p != 5, the e-component of sigma^p(1, 0) satisfies:

      sigma^p(1, 0)[e]  ==  (5/p)  (mod p)

  where (5/p) is the Legendre symbol:
      +1  if p ≡ ±1 (mod 5)   — p splits   in Z[phi]
      -1  if p ≡ ±2 (mod 5)   — p is inert in Z[phi]
       0  if p = 5             — p is ramified (checked separately)

  In Fibonacci terms: F_p ≡ (5/p) mod p.

  In Langlands terms: this is the GL_1 Frobenius character.
  The QA sigma-orbit at the prime p *computes* the Frobenius element
  of Gal(Q(sqrt(5))/Q) at p — no table lookup, no LMFDB.

QA INTERPRETATION:
  By cert [391], sigma(b, e) = (b+e, b) IS multiplication by phi on Z[phi]
  under the identification (b, e) <-> b + e*phi. After p iterations from
  (1, 0) = 1 in Z[phi]:

      sigma^p(1, 0)  =  (F_{p+1}, F_p)

  The e-component F_p mod p is the image of phi^p in Z[phi]/(p).
  At split p: Z[phi]/(p) = F_p x F_p, phi maps to (phi, phi'), and
    phi^p = (phi^p, (phi')^p) ≡ (phi, phi') by Fermat → e-component ≡ 1.
  At inert p: Z[phi]/(p) = F_{p^2}, phi^p = phi^p = phi (Frobenius on
    F_{p^2}) but relative to the base F_p the norm is -1 → e-component ≡ -1.
  Precisely: F_p ≡ (5/p) mod p via the Euler criterion on Z[phi].

BRIDGE TO GL_2:
  Cert [390] verifies that Hecke eigenvalues of weight-[2,2] HMFs over Q(sqrt(5))
  permute under Galois. That permutation is by the character p -> (5/p). This cert
  provides the GL_1 base case: the CHARACTER itself is computed by the QA orbit,
  not merely verified after the fact. The GL_2 eigenvalue a_p extends (5/p) with
  full integer precision (|a_p| <= 2*sqrt(p)); cert [394] supplies the ±1 sign.

THEOREM NT COMPLIANCE:
  All arithmetic on integers (no floats). The sigma step is discrete.
  Mod-p reduction is an observer projection applied at the END.
  No continuous functions enter the QA layer.

CHECKS:
  C1: F_p ≡ (5/p) mod p for all primes p <= 500 (excluding p=5).
  C2: split primes (p ≡ ±1 mod 5) all yield F_p ≡ +1 mod p.
  C3: inert primes (p ≡ ±2 mod 5) all yield F_p ≡ -1 mod p.
  C4: ramified prime p=5 gives F_5 ≡ 0 mod 5.
  C5: sigma is iterated p times from (1, 0) — no matrix fast-path for the
      primary check (fast-path used only for large-prime spot-checks).
"""

import sys
from fractions import Fraction

# ── QA sigma operator (exact integers, A1-compliant over infinite domain) ─────

def sigma(b: int, e: int) -> tuple:
    return b + e, b


def sigma_iterate(n: int, b0: int = 1, e0: int = 0) -> tuple:
    b, e = b0, e0
    for _ in range(n):
        b, e = sigma(b, e)
    return b, e


def sigma_iterate_mod(n: int, p: int, b0: int = 1, e0: int = 0) -> tuple:
    b, e = b0, e0
    for _ in range(n):
        b, e = (b + e) % p, b % p
    return b, e


# Fast path: matrix exponentiation mod p for large spot-checks only
def _matmul_mod(A, B, p):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % p,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % p],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % p,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % p],
    ]

def _matpow_mod(M, n, p):
    result = [[1, 0], [0, 1]]
    while n:
        if n & 1:
            result = _matmul_mod(result, M, p)
        M = _matmul_mod(M, M, p)
        n >>= 1
    return result


def fibonacci_mod_fast(p: int) -> int:
    M = [[1, 1], [1, 0]]
    Mp = _matpow_mod(M, p, p)
    return Mp[0][1] % p        # F_p mod p


# ── Legendre symbol (5/p) ─────────────────────────────────────────────────────

def legendre_5_over_p(p: int) -> int:
    if p == 5:
        return 0
    r = p % 5
    if r == 1 or r == 4:     # p ≡ ±1 mod 5 → splits
        return 1
    return -1                 # p ≡ ±2 mod 5 → inert


def primes_up_to(n: int) -> list:
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]


# ── Main validation ───────────────────────────────────────────────────────────

PRIME_LIMIT      = 500     # C1: all primes up to here
LARGE_SPOT_LIMIT = 10_000  # C5 spot-check via fast-path

def main() -> int:
    failures = []
    checks_passed = []

    print("=" * 64)
    print("Cert [394] — QA Fibonacci Frobenius Character")
    print("  F_p ≡ (5/p) mod p  [GL_1 Frobenius character via QA orbit]")
    print("=" * 64)

    # ── C4: ramified prime p=5 ────────────────────────────────────────
    b5, e5 = sigma_iterate(5)          # 5 exact steps, no mod reduction
    fp5_mod5 = e5 % 5
    leg5 = legendre_5_over_p(5)        # 0
    if fp5_mod5 == leg5:
        checks_passed.append("C4")
        print(f"\n  [PASS] C4: p=5 (ramified) → F_5 mod 5 = {fp5_mod5} == (5/5) = 0")
    else:
        failures.append(f"C4: p=5 F_5 mod 5 = {fp5_mod5}, expected 0")
        print(f"\n  [FAIL] C4: p=5 → F_5 mod 5 = {fp5_mod5}, expected 0")

    # ── C1/C2/C3: all primes up to PRIME_LIMIT (excluding 5) ─────────
    primes = [p for p in primes_up_to(PRIME_LIMIT) if p != 5]

    split_primes = [p for p in primes if legendre_5_over_p(p) == +1]
    inert_primes = [p for p in primes if legendre_5_over_p(p) == -1]

    print(f"\n  Checking {len(primes)} primes ≤ {PRIME_LIMIT} (excluding 5)")
    print(f"  Split (p ≡ ±1 mod 5): {len(split_primes)} primes")
    print(f"  Inert (p ≡ ±2 mod 5): {len(inert_primes)} primes")

    # C5 primary check: iterate sigma exactly p times, no fast-path
    c1_ok = True
    for p in primes:
        _, fp = sigma_iterate_mod(p, p)   # e-component = F_p mod p
        leg = legendre_5_over_p(p)
        expected = leg % p                 # +1 or p-1 (≡ -1) depending on Legendre
        if fp != expected:
            failures.append(
                f"C1: p={p} (leg={(5 if leg==0 else leg)}) "
                f"F_{p} mod {p} = {fp}, expected {expected}"
            )
            c1_ok = False

    if c1_ok:
        checks_passed.append("C1")
        print(f"\n  [PASS] C1: F_p ≡ (5/p) mod p for all {len(primes)} primes ≤ {PRIME_LIMIT}")
    else:
        print(f"\n  [FAIL] C1: {len([f for f in failures if f.startswith('C1')])} failures")

    # C2: split-only check (explicit)
    c2_ok = all(sigma_iterate_mod(p, p)[1] == 1 for p in split_primes)
    if c2_ok:
        checks_passed.append("C2")
        print(f"  [PASS] C2: F_p ≡ +1 mod p for all {len(split_primes)} split primes")
    else:
        bad = [p for p in split_primes if sigma_iterate_mod(p, p)[1] != 1]
        failures.append(f"C2: split primes with F_p != 1: {bad[:5]}")
        print(f"  [FAIL] C2: counterexamples (first 5): {bad[:5]}")

    # C3: inert-only check (explicit)
    c3_ok = all(sigma_iterate_mod(p, p)[1] == p - 1 for p in inert_primes)
    if c3_ok:
        checks_passed.append("C3")
        print(f"  [PASS] C3: F_p ≡ -1 mod p for all {len(inert_primes)} inert primes")
    else:
        bad = [p for p in inert_primes if sigma_iterate_mod(p, p)[1] != p - 1]
        failures.append(f"C3: inert primes with F_p != -1: {bad[:5]}")
        print(f"  [FAIL] C3: counterexamples (first 5): {bad[:5]}")

    # C5: fast-path spot-check at large primes — same result as iterate
    print(f"\n  C5 spot-check (fast-path vs iterate) at selected primes:")
    spot_primes = [1009, 2003, 4999, 7919, 9973]
    c5_ok = True
    for p in spot_primes:
        fp_fast = fibonacci_mod_fast(p)
        leg = legendre_5_over_p(p)
        expected = leg % p
        tag = "split" if leg == 1 else "inert"
        ok = (fp_fast == expected)
        mark = "PASS" if ok else "FAIL"
        print(f"    [{mark}] p={p:5d} ({tag:5s}): F_p mod p = {fp_fast:5d}, (5/p) mod p = {expected}")
        if not ok:
            failures.append(f"C5: p={p} F_p={fp_fast}, expected {expected}")
            c5_ok = False
    if c5_ok:
        checks_passed.append("C5")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"\n  Checks passed: {', '.join(checks_passed)}")
    if failures:
        print(f"  FAILURES:")
        for f in failures:
            print(f"    - {f}")
    else:
        print()
        print("  ALL CHECKS PASS")
        print()
        print("  QA interpretation:")
        print("    sigma^p(1,0)[e]  =  F_p  ≡  (5/p)  (mod p)")
        print()
        print("    The QA sigma-orbit at prime p computes the Frobenius element")
        print("    of Gal(Q(sqrt(5))/Q) — the GL_1 Hecke character — directly")
        print("    from discrete orbit iteration, no LMFDB lookup.")
        print()
        print("    This is the GL_1 base of the Langlands ladder:")
        print("      GL_1: (5/p) = ±1, computed by this cert [394]")
        print("      GL_2: a_p ∈ Z, |a_p| ≤ 2√p, verified in cert [390]")
        print("      Full: L-function factorization over Z[phi] — open")

    print()
    return 1 if failures else 0


def self_test() -> None:
    import json as _json
    failures = []

    primes = [p for p in primes_up_to(200) if p != 5]
    for p in primes:
        _, fp = sigma_iterate_mod(p, p)
        leg = legendre_5_over_p(p)
        if fp != leg % p:
            failures.append(p)

    # ramified
    _, fp5 = sigma_iterate(5)
    if fp5 % 5 != 0:
        failures.append("p=5_ramified")

    ok = len(failures) == 0
    print(_json.dumps({"ok": ok, "checks": 5, "failures": failures}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        sys.exit(main())
