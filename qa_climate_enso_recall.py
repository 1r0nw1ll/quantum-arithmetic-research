#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=climate_index_to_phase, state_alphabet=mod24_A1_compliant"
"""
Thread 3: phase-conjugate memory in a NEW domain — climate ENSO regime recall.

The cert [518]/[519] phase-conjugate associative memory, applied to real NOAA
climate data: each month is a 5-channel climate state (ONI, NAO, AO, PDO, AMO)
-> QA phase vector. Store labelled months; recall the ENSO regime (El Nino /
La Nina / Neutral) from a partial (dropped-channel) or noisy state, and test
distortion tolerance under a systemic index shift via phase-lock.

Honest stress: only 5 channels (vs 23-92 for EEG, 256 for morphology) -> very
little redundancy and ENSO phases are NOT cleanly separable in these 5 indices,
so this probes where the method's robustness genuinely degrades.

Observer boundary (Theorem NT): continuous climate indices -> phase vector
[inbound]; recalled regime [outbound]. QA layer integer.
"""
import importlib.util
from pathlib import Path
import numpy as np

from qa_phase_conjugate_memory import QAPhaseConjugateMemory, qa_add, qa_mod, M

# load the numeric-prefixed teleconnection module for its NOAA fetchers
_spec = importlib.util.spec_from_file_location(
    "tc", str(Path(__file__).resolve().parent / "48_teleconnection_topographic_observer.py"))
tc = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(tc)
except SystemExit:
    pass

RNG = np.random.default_rng(42)
ENSO = {"El Nino": 2, "La Nina": 0, "Neutral": 1}


def to_phase(F):
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return qa_mod(np.rint(np.clip(Z, -3, 3) * (M / 6.0) + (M / 2.0)).astype(np.int64))


def drop_channels(x, k, rng):
    x = x.copy()
    idx = rng.choice(len(x), k, replace=False)
    x[idx] = rng.integers(1, M + 1, k)
    return x


def run():
    df = tc.fetch_all_indices()
    chans = ["ONI", "NAO", "AO", "PDO", "AMO"]
    F = df[chans].to_numpy(dtype=float)
    phases = df["ONI"].to_numpy(dtype=float)
    y = np.array([ENSO["El Nino"] if v >= 0.5 else ENSO["La Nina"] if v <= -0.5 else ENSO["Neutral"]
                  for v in phases])
    X = to_phase(F)
    n = len(X)
    chance = np.bincount(y).max() / n
    print(f"Climate ENSO regime recall — {n} months, 5 channels, phase dim={X.shape[1]}")
    print(f"  ElNino={int((y==2).sum())} LaNina={int((y==0).sum())} Neutral={int((y==1).sum())} "
          f"chance(majority)={chance:.3f}\n")

    rng = np.random.default_rng(0)
    idx = rng.permutation(n); tr, te = idx[:n // 2], idx[n // 2:]
    mem = QAPhaseConjugateMemory(X[tr], sharpen=6.0)
    ys = y[tr]

    def direct(probe): return ys[int(np.argmax(mem.overlap(probe)))]

    def pl(probe):
        bk, best = 0, -1
        for psi in range(1, M + 1):
            C = mem.overlap(qa_add(probe, psi)); k = int(np.argmax(C))
            if C[k] > best: best, bk = C[k], k
        return ys[bk]

    print("[1] ENSO regime recall vs dropped channels (of 5):")
    print(f"{'dropped':>8s} {'accuracy':>9s}")
    for k in (0, 1, 2, 3):
        acc = np.mean([direct(drop_channels(X[i], k, rng)) == y[i] for i in te])
        print(f"{k:8d} {acc:9.3f}")

    print("\n[2] Systemic index shift (global bioelectric-analog offset phi):")
    print(f"{'phi':>5s} {'naive':>8s} {'phase-lock':>11s}")
    for phi in (0, 3, 6, 12):
        dn = np.mean([direct(qa_add(drop_channels(X[i], 1, rng), phi)) == y[i] for i in te])
        plk = np.mean([pl(qa_add(drop_channels(X[i], 1, rng), phi)) == y[i] for i in te])
        print(f"{phi:5d} {dn:8.3f} {plk:11.3f}")
    print(f"\n(chance/majority = {chance:.3f})")


if __name__ == "__main__":
    run()
