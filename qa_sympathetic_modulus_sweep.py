#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=selectivity readout (Theorem NT); QA layer = integer phase patterns {1..M}^N under qa_add (A1). The sweep documents that mod-M is a resolution knob, not an orbit modulus."
# RT1_OBSERVER_FILE: coupling scores / separations are observer-layer readouts.
"""
Modulus-robustness of the SVP sympathetic-resonance operator (companion to
qa_sympathetic_resonance.py and the [522] real-seismic robustness): does the sympathetic
selectivity depend on mod-24, and is 24 (the Pisano period of the mod-9 orbit) the right choice?

Finding: same-chord always couples at 1.000 for every M; the different-chord false-coupling
FLOOR falls as 1/M-ish, so the separation INCREASES monotonically with M. mod-24 is fine but
SUBOPTIMAL (M=48/96 cleaner). So the selectivity is robust to M, and '24' is a resolution knob
(a borrowed iteration count), not an orbit-derived modulus.
"""
from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(0)


def separation(M, N=64, K=5, trials=300):
    def add(a, b):
        return ((a - 1 + b) % M) + 1
    tun = [RNG.integers(1, M + 1, N) for _ in range(K)]
    same = diff = 0.0
    for _ in range(trials):
        t = RNG.integers(K); phi = int(RNG.integers(1, M + 1))
        pi = tun[t]; pj = add(tun[t], phi)
        t2 = (t + 1 + RNG.integers(K - 1)) % K; pk = add(tun[t2], phi)
        same += max(np.sum(add(pi, psi) == pj) for psi in range(1, M + 1)) / N
        diff += max(np.sum(add(pi, psi) == pk) for psi in range(1, M + 1)) / N
    return same / trials, diff / trials


def run():
    print("SVP sympathetic-resonance operator -- modulus robustness (certified uses mod-24)\n")
    print(f"{'M':>4} {'same-chord':>11} {'diff-floor':>11} {'separation':>11}")
    seps = []
    for M in (6, 9, 12, 24, 48, 96):
        s, d = separation(M); seps.append(s - d)
        print(f"{M:>4} {s:11.3f} {d:11.3f} {s-d:11.3f}{'  <- cert mod-24' if M == 24 else ''}")
    print(f"\nselectivity holds for every sampled M (same-chord=1.000 throughout); separation grows")
    print(f"with M -> mod-24 is a resolution knob (suboptimal here), not an orbit-derived modulus.")
    return 0 if all(x > 0.3 for x in seps) else 1


if __name__ == "__main__":
    raise SystemExit(run())
