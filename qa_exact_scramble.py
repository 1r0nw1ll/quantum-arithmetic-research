#!/usr/bin/env python3
# QA_COMPLIANCE = "QA layer = EXACT integer pixel positions under the Arnold cat map (a permutation on the n x n grid, A1, reversible). The float trajectory is the DECIMAL representation under test; never fed back as QA state. Real scored task: lossless recovery of a real data block."
# RT1_OBSERVER_FILE: the float positions, recovery rates, horizons are observer-layer readouts; QA positions are exact integers.
"""
A REAL scored task where exact-integer QA can WIN or LOSE against float: reversible data
scrambling (the Arnold cat map -- a genuine image-encryption scheme). Round-trip a real data
block forward N rounds then inverse N rounds; score how many bytes come back byte-exact.

  * EXACT integers: the cat map C=[[1,1],[1,2]] (det 1) is a PERMUTATION of the n x n grid,
    so forward-N then inverse-N is the identity -> 100% recovery at EVERY N, by construction.
  * DECIMAL float64: the SAME positions computed as continuous (x,y) in [0,1)^2 drift at the
    Lyapunov rate; once the drift exceeds one cell (1/n) the recovered position rounds to the
    WRONG pixel and the byte is lost.

This is a fair fight: float64 (double precision, the standard) gets every chance, and for
SMALL N it recovers perfectly -- so exactness is UNNECESSARY there (it can lose/tie). The
advantage appears only past the drift horizon; the test reports the crossover N so we see
which regime is which -- not "float would drift" but a scored task where the drift loses data.

COMMITTED PREDICTION (no hedge): exact recovery = 100% at all N. Float64 recovery = 100% for
small N, collapsing toward chance past a horizon of a few tens of rounds (it shortens as the
grid n grows). FALSIFIABLE: float64 could recover perfectly out to N in the hundreds (then
exactness is a purist point at these scales), or exact could fail to be a permutation (bug).
"""
from __future__ import annotations
import math


def load_block(n, path="GEMINI.md"):
    """A real n*n data block from the bytes of a real file (repeated to fill)."""
    with open(path, "rb") as f:
        raw = f.read()
    return [raw[k % len(raw)] for k in range(n * n)]


def roundtrip_exact(n, N):
    """For each grid cell, forward-N then inverse-N in EXACT integers; fraction returning home."""
    home = 0
    for i in range(n):
        for j in range(n):
            x, y = i, j
            for _ in range(N):                       # forward C=[[1,1],[1,2]]
                x, y = (x + y) % n, (x + 2 * y) % n
            for _ in range(N):                       # inverse C^-1=[[2,-1],[-1,1]]
                x, y = (2 * x - y) % n, (-x + y) % n
            home += (x, y) == (i, j)
    return home / (n * n)


def roundtrip_float(n, N):
    """Same round-trip but positions carried as DECIMAL floats in [0,1)^2; fraction home."""
    home = 0
    for i in range(n):
        for j in range(n):
            x, y = i / n, j / n
            for _ in range(N):
                x, y = (x + y) % 1.0, (x + 2 * y) % 1.0
            for _ in range(N):
                x, y = (2 * x - y) % 1.0, (-x + y) % 1.0
            if (round(x * n) % n, round(y * n) % n) == (i, j):
                home += 1
    return home / (n * n)


def recover_bytes(n, N, use_float):
    """Actually scramble+unscramble a real byte block; fraction of bytes byte-exactly recovered."""
    data = load_block(n)
    perm = [0] * (n * n)                              # forward destination of each source cell
    for i in range(n):
        for j in range(n):
            if use_float:
                x, y = i / n, j / n
                for _ in range(N):
                    x, y = (x + y) % 1.0, (x + 2 * y) % 1.0
                di, dj = round(x * n) % n, round(y * n) % n
            else:
                x, y = i, j
                for _ in range(N):
                    x, y = (x + y) % n, (x + 2 * y) % n
                di, dj = x, y
            perm[i * n + j] = di * n + dj
    scrambled = [0] * (n * n)
    for k in range(n * n):
        scrambled[perm[k]] = data[k]                 # collisions (float) silently drop bytes
    # unscramble with the inverse map computed the same way
    inv = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            if use_float:
                x, y = i / n, j / n
                for _ in range(N):
                    x, y = (2 * x - y) % 1.0, (-x + y) % 1.0
                di, dj = round(x * n) % n, round(y * n) % n
            else:
                x, y = i, j
                for _ in range(N):
                    x, y = (2 * x - y) % n, (-x + y) % n
                di, dj = x, y
            inv[i * n + j] = di * n + dj
    recovered = [0] * (n * n)
    for k in range(n * n):
        recovered[inv[k]] = scrambled[k]
    return sum(recovered[k] == data[k] for k in range(n * n)) / (n * n)


def run():
    print("Exact-integer vs decimal-float on a REAL reversible scramble (Arnold cat map)\n")
    n = 100
    Ns = [1, 5, 10, 15, 20, 25, 30, 40, 60, 100, 150]
    k_pred = (52 * math.log(2) - math.log(n)) / (2 * math.log((1 + math.sqrt(5)) / 2))
    print(f"[setup] {n}x{n} grid, real byte block (GEMINI.md); forward-N + inverse-N round trip.")
    print(f"[predicted] float drift exceeds one cell (1/n) near N ~ {k_pred:.0f} rounds.\n")

    print(f"{'N':>4} {'exact recover':>15} {'float recover':>15} {'exact bytes':>13} {'float bytes':>12}")
    print("-" * 62)
    exact_all, float_horizon = True, None
    for N in Ns:
        er = roundtrip_exact(n, N)
        fr = roundtrip_float(n, N)
        eb = recover_bytes(n, N, use_float=False)
        fb = recover_bytes(n, N, use_float=True)
        exact_all = exact_all and er > 0.999 and eb > 0.999
        if float_horizon is None and fr < 0.99:
            float_horizon = N
        print(f"{N:>4} {er:>14.1%} {fr:>14.1%} {eb:>12.1%} {fb:>11.1%}")

    print(f"\n[result] EXACT integers: 100% recovery at EVERY N (byte-perfect) -- {exact_all}.")
    print(f"    FLOAT64: recovery first drops below 99% at N = {float_horizon} rounds "
          f"(predicted ~{k_pred:.0f}); it collapses past there as decimal drift exceeds one cell.")

    ok = exact_all and float_horizon is not None and float_horizon < 100
    print("\nVERDICT (scored, real task -- can lose):")
    print(f"  * For SMALL N (< ~{float_horizon}) float64 recovers perfectly too -> exactness is")
    print(f"    UNNECESSARY there; it does not win, it ties. (The honest 'float is fine' regime.)")
    print(f"  * Past N = {float_horizon} rounds float64 LOSES DATA (recovery collapses) while exact")
    print(f"    integers stay byte-perfect forever -- because the decimal positions drift at the")
    print(f"    Lyapunov rate and round to the wrong cell. Here exactness genuinely WINS a scored")
    print(f"    task (lossless recovery), and the horizon shortens as the grid gets finer.")
    print(f"  * So QA's exact-integer rule is not a purist tic and not always needed: it is exactly")
    print(f"    what a reversible/long/chaotic computation requires, and irrelevant to a short one.")
    print(f"    Real signal-processing agrees -- lossless codecs (JPEG2000 lifting) use integer")
    print(f"    arithmetic for precisely this reason.")
    print(f"\n  STATUS: {'SCORED -- exact wins past N=' + str(float_horizon) + ', ties below it' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
