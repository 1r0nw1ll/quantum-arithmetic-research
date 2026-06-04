#!/usr/bin/env python3
"""
QA Quantize Algorithm Cert [320] -- validator

Primary source:
  Iverson, B. (1995) Quantum Arithmetic Book 3 (QA-3), ITAM, Portland,
    ISBN 1-883401-08-9, Chapter 1: "Quantize".

The Quantize algorithm is Ben's bridge from empirical measurement to the QA
integer world. Given two quantum-exact measurement arms JO = d*b and KO = d*a,
the diagonal DO = (JO+KO)/2 = d*(b+a)/2. Since a = b+2e, we have b+a = 2(b+e)
= 2d, so DO = d*d exactly. Inversion: d = isqrt(DO), b = JO // d, e = d - b.

Five claims:

  C1  Midpoint identity: b+a=2d holds for ALL 81 pairs in {1..9}^2 (algebraic
      completeness — zero exceptions; Iverson's foundation for the Quantize step).

  C2  DO=d^2 theorem: for all 72 Cosmos pairs (b,e), the measurement triple
      (JO=d*b, KO=d*a, DO=(JO+KO)//2) satisfies DO == d*d exactly; the
      midpoint of the outer arms equals the squared inner coordinate.

  C3  Quantize reconstruction lossless: for all 72 Cosmos pairs, given
      (DO=d*d, JO=d*b), the inverse d=isqrt(DO), b_rec=JO//d, e_rec=d-b_rec
      recovers (b,e) exactly — zero reconstruction errors.

  C4  Fingerprint uniqueness: all 72 Cosmos pairs have distinct (DO, JO)
      encodings; the (d^2, d*b) fingerprint is injective on the Cosmos orbit.

  C5  Satellite extension: midpoint identity and exact Quantize reconstruction
      hold for all 8 Satellite pairs; isqrt(d*d)==d for every Satellite d;
      the 8-cycle is fully Quantize-consistent.
"""

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic: d=b+e raw, a=b+2e raw; "
    "DO=d*d exact; JO=d*b, KO=d*a observer projections; "
    "Quantize inverse: d=isqrt(DO), b=JO//d, e=d-b; "
    "Theorem NT: (JO,KO,DO) are observer-layer measurements; (b,e) is the "
    "QA causal layer; round-trip (b,e)->(DO,JO)->(b,e) lossless is a QA claim"
)

from fractions import Fraction
from math import isqrt

# ---------------------------------------------------------------------------
# QA primitives (mod-9, A1-compliant)
# ---------------------------------------------------------------------------

M = 9

def qa_step(b: int, e: int, m: int = M) -> int:
    """A1-compliant T-step: new b = ((b+e-1) % m) + 1."""
    return ((b + e - 1) % m) + 1


def beda(b: int, e: int) -> tuple[int, int, int, int]:
    """BEDA(b,e) = (b, e, d, a) with d=b+e RAW, a=b+2e RAW (no mod)."""
    d = b + e
    a = b + 2 * e
    return (b, e, d, a)


def cosmos_states(m: int = M) -> list[tuple[int, int]]:
    """All 72 Cosmos pairs: orbit_family == Cosmos (not Satellite/Singularity)."""
    cosmos = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if b == m and e == m:
                continue  # Singularity (9,9)
            # Satellite: b == e (diagonal excluding (9,9))
            if b == e:
                continue
            cosmos.append((b, e))
    return cosmos


def satellite_states(m: int = M) -> list[tuple[int, int]]:
    """All 8 Satellite pairs: b == e, excluding Singularity (9,9)."""
    return [(b, b) for b in range(1, m) if b != m]


# ---------------------------------------------------------------------------
# Quantize helpers
# ---------------------------------------------------------------------------

def quantize(DO: int, JO: int) -> tuple[int, int]:
    """
    Inverse Quantize: given DO=d^2 and JO=d*b, recover (b, e).
    Requires DO to be a perfect square and JO divisible by sqrt(DO).
    """
    d = isqrt(DO)
    b = JO // d
    e = d - b
    return (b, e)


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------

def check_c1() -> list[str]:
    """C1: b+a=2d for all 81 pairs in {1..9}^2."""
    failures = []
    for b in range(1, M + 1):
        for e in range(1, M + 1):
            _, _, d, a = beda(b, e)
            if b + a != 2 * d:
                failures.append(f"({b},{e}): b+a={b+a} != 2d={2*d}")
    return failures


def check_c2() -> list[str]:
    """C2: DO=d^2 for all 72 Cosmos pairs."""
    failures = []
    for b, e in cosmos_states():
        _, _, d, a = beda(b, e)
        JO = d * b
        KO = d * a
        DO = (JO + KO) // 2
        if DO != d * d:
            failures.append(
                f"({b},{e}): DO=(JO+KO)/2={DO}, d^2={d*d} — mismatch"
            )
        if (JO + KO) % 2 != 0:
            failures.append(
                f"({b},{e}): JO+KO={JO+KO} is odd — midpoint not integer"
            )
    return failures


def check_c3() -> list[str]:
    """C3: Quantize reconstruction lossless for all 72 Cosmos pairs."""
    failures = []
    for b, e in cosmos_states():
        _, _, d, a = beda(b, e)
        DO = d * d
        JO = d * b
        b_rec, e_rec = quantize(DO, JO)
        if (b_rec, e_rec) != (b, e):
            failures.append(
                f"({b},{e}): reconstructed ({b_rec},{e_rec}) — mismatch"
            )
    return failures


def check_c4() -> list[str]:
    """C4: all 72 Cosmos (DO, JO) fingerprints are distinct."""
    failures = []
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for b, e in cosmos_states():
        _, _, d, a = beda(b, e)
        fp = (d * d, d * b)
        if fp in seen:
            failures.append(
                f"({b},{e}) and {seen[fp]} share fingerprint DO={fp[0]}, JO={fp[1]}"
            )
        seen[fp] = (b, e)
    return failures


def check_c5() -> list[str]:
    """C5: midpoint identity and Quantize reconstruction for all 8 Satellite pairs."""
    failures = []
    for b, e in satellite_states():
        # midpoint identity (same as C1 but asserting for Satellite explicitly)
        _, _, d, a = beda(b, e)
        if b + a != 2 * d:
            failures.append(f"Satellite ({b},{e}): b+a={b+a} != 2d={2*d}")
        # isqrt exactness
        if isqrt(d * d) != d:
            failures.append(
                f"Satellite ({b},{e}): isqrt(d^2)={isqrt(d*d)} != d={d}"
            )
        # reconstruction
        DO = d * d
        JO = d * b
        b_rec, e_rec = quantize(DO, JO)
        if (b_rec, e_rec) != (b, e):
            failures.append(
                f"Satellite ({b},{e}): reconstructed ({b_rec},{e_rec}) — mismatch"
            )
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("C1", "Midpoint identity b+a=2d (all 81 pairs)",        check_c1),
    ("C2", "DO=d^2 theorem (72 Cosmos pairs)",                check_c2),
    ("C3", "Quantize reconstruction lossless (72 Cosmos)",    check_c3),
    ("C4", "Fingerprint uniqueness (72 Cosmos)",              check_c4),
    ("C5", "Satellite extension (8 Satellite pairs)",         check_c5),
]


def main() -> int:
    all_pass = True
    for cid, desc, fn in CHECKS:
        failures = fn()
        status = "PASS" if not failures else "FAIL"
        print(f"  [{status}] {cid}: {desc}")
        for f in failures:
            print(f"        {f}")
        if failures:
            all_pass = False
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
