#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=reproduction script for cert [515]; sources cited in mapping_protocol_ref.json -->
"""
Reproduction script for cert [515]'s empirical record.

Requires `fpylll` (not a repo dependency; install into a venv with
`pip install fpylll cysignals` -- both installed cleanly from prebuilt
wheels on macOS arm64 as of 2026-07-04, no need to compile fplll/GMP from
source). Not run by the cert's --self-test (that only checks the
deterministic, stdlib-only mathematical properties); run this file
directly to regenerate the numbers cited in mapping_protocol_ref.json and
docs/families/515_qa_orbit_lattice_mod3_collapse.md.

Builds a real NTRU-lattice attack surface (ring Z[x]/(x^N-1), standard
2N-dim lattice [I_N|H; 0|q*I_N]) and runs real LLL/BKZ (via fpylll)
against two key-generation methods: random ternary keys (baseline) and
QA-orbit-derived ternary keys (coefficient = (b mod 3) - 1).
"""
from __future__ import annotations

import random
import sys
import time
from math import gcd

try:
    from fpylll import IntegerMatrix, LLL, BKZ
except ImportError:
    print("fpylll not installed -- `pip install fpylll cysignals` into a venv first.",
          file=sys.stderr)
    raise


# --- QA orbit dynamics (A1: states in {1..m}; A2: derived; integer only) ---

def qa_step(b: int, e: int, m: int) -> tuple[int, int]:
    return e, ((b + e - 1) % m) + 1


def orbit_sequence(b0: int, e0: int, m: int, length: int) -> list[int]:
    seq = []
    b, e = b0, e0
    for _ in range(length):
        seq.append(b)
        b, e = qa_step(b, e, m)
    return seq


def orbit_period(b0: int, e0: int, m: int) -> int:
    b, e = b0, e0
    n = 0
    while True:
        b, e = qa_step(b, e, m)
        n += 1
        if (b, e) == (b0, e0):
            return n


def qa_poly(b0: int, e0: int, m: int, N: int) -> list[int]:
    return [((v % 3) - 1) for v in orbit_sequence(b0, e0, m, N)]


def seeds_with_period(m: int, period: int, limit: int = 200) -> list[tuple[int, int]]:
    out = []
    for b0 in range(1, m + 1):
        for e0 in range(1, m + 1):
            if orbit_period(b0, e0, m) == period:
                out.append((b0, e0))
                if len(out) >= limit:
                    return out
    return out


# --- NTRU ring arithmetic ---------------------------------------------------

def poly_mul_mod(a: list[int], b: list[int], N: int, mod: int | None = None) -> list[int]:
    result = [0] * N
    for i in range(N):
        if a[i] == 0:
            continue
        ai = a[i]
        for j in range(N):
            result[(i + j) % N] += ai * b[j]
    if mod is not None:
        result = [c % mod for c in result]
    return result


def _gf2_xgcd(a_coeffs: list[int], N: int):
    def trim(p):
        while len(p) > 1 and p[-1] == 0:
            p.pop()
        return p

    def pmul(p, q_):
        r = [0] * (len(p) + len(q_) - 1)
        for i, pi in enumerate(p):
            if pi:
                for j, qj in enumerate(q_):
                    r[i + j] ^= (pi * qj) % 2
        return trim(r)

    modulus = [1] + [0] * (N - 1) + [1]
    r0 = modulus[:]
    r1 = trim([c % 2 for c in a_coeffs] + [0] * max(0, N - len(a_coeffs)))
    s0, s1 = [0], [1]
    while any(r1):
        rem = r0[:]
        quot = [0] * max(1, len(r0) - len(r1) + 1)
        while len(rem) >= len(r1) and any(rem):
            if rem[-1] == 0:
                rem.pop()
                continue
            shift = len(rem) - len(r1)
            quot[shift] ^= 1
            for i, rc in enumerate(r1):
                rem[i + shift] ^= rc
            rem = trim(rem)
        r0, r1 = r1, rem
        qs1 = pmul(quot, s1)
        newlen = max(len(s0), len(qs1))
        s0p = s0 + [0] * (newlen - len(s0))
        qs1p = qs1 + [0] * (newlen - len(qs1))
        s0, s1 = s1, trim([(s0p[i] ^ qs1p[i]) for i in range(newlen)])
    return r0, s0


def poly_inverse_mod_2power(a: list[int], N: int, q: int) -> list[int] | None:
    """Invert a in Z[x]/(x^N-1) mod q=2^k via GF(2) xgcd + Hensel lift."""
    gcd_, s0 = _gf2_xgcd(a, N)
    if gcd_ != [1]:
        return None
    inv = ([c % 2 for c in s0] + [0] * N)[:N]
    mod = 2
    while mod < q:
        mod = mod * mod
        two_minus = poly_mul_mod(a, inv, N, mod)
        two_minus = [(-c) % mod for c in two_minus]
        two_minus[0] = (two_minus[0] + 2) % mod
        inv = poly_mul_mod(inv, two_minus, N, mod)
    return [c % q for c in inv]


def verify_inverse(a: list[int], inv: list[int], N: int, q: int) -> bool:
    return poly_mul_mod(a, inv, N, q) == [1] + [0] * (N - 1)


def keygen_random(N: int, q: int, rng: random.Random):
    while True:
        f = [rng.choice([-1, 0, 1]) for _ in range(N)]
        f_inv = poly_inverse_mod_2power(f, N, q)
        if f_inv is None or not verify_inverse(f, f_inv, N, q):
            continue
        g = [rng.choice([-1, 0, 1]) for _ in range(N)]
        return f, g, poly_mul_mod(g, f_inv, N, q)


def keygen_qa(N: int, q: int, m: int, rng: random.Random, seed_pool=None):
    attempts = 0
    while True:
        attempts += 1
        if seed_pool:
            b0, e0 = rng.choice(seed_pool)
        else:
            b0, e0 = rng.randint(1, m), rng.randint(1, m)
        f = qa_poly(b0, e0, m, N)
        f_inv = poly_inverse_mod_2power(f, N, q)
        if f_inv is None or not verify_inverse(f, f_inv, N, q):
            if attempts > 20000:
                raise RuntimeError("no invertible QA-derived f found")
            continue
        if seed_pool:
            b1, e1 = rng.choice(seed_pool)
        else:
            b1, e1 = rng.randint(1, m), rng.randint(1, m)
        g = qa_poly(b1, e1, m, N)
        return f, g, poly_mul_mod(g, f_inv, N, q)


def target_norm2(f: list[int], g: list[int]) -> int:
    return sum(v * v for v in f) + sum(v * v for v in g)


def check_fg_in_lattice(f: list[int], g: list[int], h: list[int], N: int, q: int) -> bool:
    fh = poly_mul_mod(f, h, N)
    return all((fh[i] - g[i]) % q == 0 for i in range(N))


# --- NTRU lattice + fpylll attack ------------------------------------------

def circulant(h, N):
    return [[h[(j - i) % N] for j in range(N)] for i in range(N)]


def ntru_lattice_basis(h, N, q):
    H = circulant(h, N)
    rows = [([1 if k == i else 0 for k in range(N)] + H[i]) for i in range(N)]
    rows += [([0] * N + [q if k == i else 0 for k in range(N)]) for i in range(N)]
    return rows


def run_reduction(basis_rows, dim, use_bkz=False, bkz_block=10):
    A = IntegerMatrix(dim, dim)
    for i, row in enumerate(basis_rows):
        for j, val in enumerate(row):
            A[i, j] = val
    LLL.reduction(A)
    if use_bkz:
        BKZ.reduction(A, BKZ.Param(block_size=bkz_block))
    return A


def attack_once(f, g, h, N, q, use_bkz=False, bkz_block=10):
    basis = ntru_lattice_basis(h, N, q)
    A = run_reduction(basis, 2 * N, use_bkz=use_bkz, bkz_block=bkz_block)
    target = target_norm2(f, g)
    best = min(sum(A[i, j] * A[i, j] for j in range(A.ncols)) for i in range(A.nrows))
    return best, target


def summarize(ratios, label, trials):
    broke = sum(1 for r in ratios if r <= 1.5)
    avg = sum(ratios) / len(ratios)
    print(f"  {label:24s}: broken {broke}/{trials}  avg(best/target)={avg:8.3f}")


if __name__ == "__main__":
    N, q, trials = 83, 256, 10
    rng = random.Random(42)

    print(f"=== Cert [515] reproduction: N={N} q={q}, LLL only ===")
    ratios_random, ratios_qa9, ratios_qa80 = [], [], []
    for _ in range(trials):
        f, g, h = keygen_random(N, q, rng)
        assert check_fg_in_lattice(f, g, h, N, q)
        best, target = attack_once(f, g, h, N, q)
        ratios_random.append(best / target)

        f2, g2, h2 = keygen_qa(N, q, 9, rng)
        assert check_fg_in_lattice(f2, g2, h2, N, q)
        best2, target2 = attack_once(f2, g2, h2, N, q)
        ratios_qa9.append(best2 / target2)

    pool80 = seeds_with_period(80, 120, limit=trials * 4)
    for _ in range(trials):
        f3, g3, h3 = keygen_qa(N, q, 80, rng, seed_pool=pool80)
        assert check_fg_in_lattice(f3, g3, h3, N, q)
        best3, target3 = attack_once(f3, g3, h3, N, q)
        ratios_qa80.append(best3 / target3)

    summarize(ratios_random, "random-key baseline", trials)
    summarize(ratios_qa9, "QA m=9 (3|9, unsafe)", trials)
    summarize(ratios_qa80, "QA m=80 (gcd=1, safe)", trials)
