#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=pattern_to_phase + float_correlation_scores, state_alphabet=mod24_A1_compliant"
"""
qa_phase_conjugate_memory.py — QA Phase-Conjugate Holographic Associative Memory.

Builds a content-addressable memory on the exact phase-conjugate primitive of
cert [518]. Stored patterns are attractors of a phase-conjugate feedback map; a
corrupted or partial probe flows to the nearest stored pattern — the discrete
analog of an optical phase-conjugate resonator (Owechko 1987; Soffer et al 1986)
and of Michael Levin's "target morphology reached as an attractor from many
perturbed starting states."

## Mechanism (grounded in cert [518])

Patterns and probes are QA phase vectors in {1,...,m}^N. Two phases are
phase-conjugate-matched iff their conjugate sum is the additive identity:
    qa_add(x_i, qa_neg(p_i)) == m      <=>      x_i == p_i
so the phase-conjugate overlap of a probe x with stored pattern k is
    C_k = #{ i : qa_add(x_i, qa_neg(p^k_i)) == m }.
Recall is holographic playback: each site is reconstructed by superposing the
stored patterns weighted by their (sharpened) overlap — the phase-conjugate
readout — and iterating to a fixed point.

For sufficiently separated (near-orthogonal) pattern families, stored patterns
are fixed points (a stored pattern has maximal overlap with itself) and a
corrupted probe converges to the stored pattern whose basin it lies in. These
are demonstrated empirical properties under favorable separation, NOT guaranteed
for arbitrary memories: near-duplicate or highly correlated patterns can outvote
the self-pattern at a site, and recall degrades toward chance as patterns
approach identical or corruption reaches the alphabet noise floor.

## Axiom compliance
Phase state (patterns, probe, reconstruction) is integer in {1,...,m} (A1/S2);
the additive identity is m, never 0. Correlation scores and vote weights are
float observer-layer quantities (like E8-alignment / harmonic-index scores),
never fed back as QA state — the argmax that writes state is over integer phases
(Theorem NT: the observer boundary is crossed only in scoring, not in state).
No **2 (S1).
"""
from __future__ import annotations
import numpy as np

M = 24


def qa_mod(x):
    return ((np.asarray(x, dtype=np.int64) - 1) % M) + 1


def qa_add(a, b):
    return qa_mod(np.asarray(a, np.int64) + np.asarray(b, np.int64))


def qa_neg(a):
    return qa_mod(-np.asarray(a, np.int64))


IDENTITY = M  # additive identity (No-Zero representative of 0)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class QAPhaseConjugateMemory:
    def __init__(self, patterns: np.ndarray, sharpen: float = 6.0):
        """patterns: (K, N) integer array in {1,...,M}. sharpen: overlap
        exponent (1 = linear holographic superposition; large = winner-take-all)."""
        self.P = qa_mod(patterns)          # (K, N)
        self.K, self.N = self.P.shape
        self.sharpen = sharpen

    def overlap(self, x: np.ndarray) -> np.ndarray:
        """Phase-conjugate overlap of probe x with each stored pattern:
        C_k = #{ i : qa_add(x_i, qa_neg(P_k_i)) == identity }  (== #matches)."""
        conj = qa_neg(self.P)                          # (K, N)
        match = (qa_add(x[None, :], conj) == IDENTITY)  # (K, N) bool
        return match.sum(axis=1).astype(float)          # (K,)

    def _playback(self, x: np.ndarray) -> np.ndarray:
        """One phase-conjugate holographic readout step: reconstruct each site by
        superposing stored patterns weighted by sharpened overlap, argmax phase."""
        C = self.overlap(x)
        w = (C / self.N) ** self.sharpen               # observer-layer weights
        if w.sum() <= 0:
            return x
        # per-site weighted vote over the M phase values
        votes = np.zeros((self.N, M + 1))              # index 1..M
        for k in range(self.K):
            np.add.at(votes, (np.arange(self.N), self.P[k]), w[k])
        return votes[:, 1:].argmax(axis=1) + 1          # integer phase in {1..M}

    def recall(self, probe: np.ndarray, iters: int = 25) -> np.ndarray:
        x = qa_mod(probe)
        for _ in range(iters):
            nxt = self._playback(x)
            if np.array_equal(nxt, x):
                break
            x = nxt
        return x

    def recall_phase_locked(self, probe: np.ndarray, iters: int = 25) -> np.ndarray:
        """Distortion-tolerant recall (cert [518] property): the resonator locks
        onto the global compensation phase psi that maximizes overlap — the
        discrete analog of a phase-conjugate mirror self-adjusting to the
        distorting medium — recalls in that frame, then reports in the probe's
        frame. Recovers the stored pattern through an unknown global phase screen."""
        probe = qa_mod(probe)
        best_psi, best_score = IDENTITY, -1.0
        for psi in range(1, M + 1):
            score = self.overlap(qa_add(probe, psi)).max()  # observer-layer score
            if score > best_score:
                best_score, best_psi = score, psi
        rec = self.recall(qa_add(probe, best_psi), iters=iters)
        return qa_add(rec, qa_neg(best_psi))  # shift back into the probe's frame


# ---------------------------------------------------------------------------
# Corruption / distortion (observer-layer probe generation)
# ---------------------------------------------------------------------------
def corrupt(pattern, frac, rng):
    """Replace a fraction of sites with uniform-random phases."""
    x = pattern.copy()
    n = len(x)
    idx = rng.choice(n, int(frac * n), replace=False)
    x[idx] = rng.integers(1, M + 1, len(idx))
    return x


def global_phase_screen(pattern, phi):
    """Apply a global modular phase shift (the [518] distortion medium)."""
    return qa_add(pattern, phi)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_fixed_points(mem, rng):
    """Stored patterns must be exact fixed points of recall."""
    ok = all(np.array_equal(mem.recall(mem.P[k]), mem.P[k]) for k in range(mem.K))
    return ok


def exp_recall_vs_corruption(N=120, K=8, sharpen=6.0, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.integers(1, M + 1, (K, N))
    mem = QAPhaseConjugateMemory(P, sharpen=sharpen)
    fp_ok = exp_fixed_points(mem, rng)
    rows = []
    for frac in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        exact = 0
        for _ in range(trials):
            s = rng.integers(K)
            probe = corrupt(P[s], frac, rng)
            rec = mem.recall(probe)
            if np.array_equal(rec, P[s]):
                exact += 1
        rows.append((frac, exact / trials))
    return fp_ok, rows


def exp_capacity(N=120, sharpen=6.0, frac=0.2, trials=120, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for K in (16, 64, 128, 256, 512):
        P = rng.integers(1, M + 1, (K, N))
        mem = QAPhaseConjugateMemory(P, sharpen=sharpen)
        exact = 0
        for _ in range(trials):
            s = rng.integers(K)
            rec = mem.recall(corrupt(P[s], frac, rng))
            if np.array_equal(rec, P[s]):
                exact += 1
        rows.append((K, K / N, exact / trials))
    return rows


def exp_correlated_capacity(N=120, n_proto=4, sharpen=6.0, frac=0.15, trials=150, seed=7):
    """Honest stress: patterns are noisy variants of a few shared prototypes, so
    they are CORRELATED (real crosstalk), unlike near-orthogonal random patterns.
    Recall rate as the per-prototype family grows."""
    rng = np.random.default_rng(seed)
    rows = []
    for per_proto in (2, 4, 8, 16, 32):
        protos = rng.integers(1, M + 1, (n_proto, N))
        pats = []
        for p in protos:
            for _ in range(per_proto):
                v = p.copy()
                # 30% of sites jittered off the prototype -> shared 70% structure
                idx = rng.choice(N, int(0.30 * N), replace=False)
                v[idx] = rng.integers(1, M + 1, len(idx))
                pats.append(v)
        P = np.array(pats)
        mem = QAPhaseConjugateMemory(P, sharpen=sharpen)
        exact = 0
        for _ in range(trials):
            s = rng.integers(len(P))
            rec = mem.recall(corrupt(P[s], frac, rng))
            if np.array_equal(rec, P[s]):
                exact += 1
        rows.append((n_proto * per_proto, per_proto, exact / trials))
    return rows


def exp_content_addressable(N=120, K=8, sharpen=6.0, trials=200, seed=2):
    """Probe near pattern s must converge to s, not another stored pattern."""
    rng = np.random.default_rng(seed)
    P = rng.integers(1, M + 1, (K, N))
    mem = QAPhaseConjugateMemory(P, sharpen=sharpen)
    to_nearest = 0
    for _ in range(trials):
        s = rng.integers(K)
        rec = mem.recall(corrupt(P[s], 0.35, rng))
        # nearest stored pattern to the recalled state
        d = [(rec != P[k]).sum() for k in range(K)]
        if int(np.argmin(d)) == s and d[s] == 0:
            to_nearest += 1
    return to_nearest / trials


def exp_distortion_tolerance(N=120, K=8, sharpen=6.0, trials=200, seed=3):
    """Global phase screen on the probe (the [518] distortion medium): does
    recall still land the correct pattern (up to the global shift)?"""
    rng = np.random.default_rng(seed)
    P = rng.integers(1, M + 1, (K, N))
    mem = QAPhaseConjugateMemory(P, sharpen=sharpen)
    rows = []
    for phi in (1, 3, 6, 12):
        naive, locked = 0, 0
        for _ in range(trials):
            s = rng.integers(K)
            probe = global_phase_screen(corrupt(P[s], 0.2, rng), phi)
            if np.array_equal(mem.recall(probe), qa_add(P[s], phi)):
                naive += 1
            if np.array_equal(mem.recall_phase_locked(probe), qa_add(P[s], phi)):
                locked += 1
        rows.append((phi, naive / trials, locked / trials))
    return rows


if __name__ == "__main__":
    print("QA PHASE-CONJUGATE HOLOGRAPHIC ASSOCIATIVE MEMORY  (m=24)\n")

    fp_ok, rc = exp_recall_vs_corruption()
    print(f"[1] Stored patterns are exact fixed points: {fp_ok}")
    print("    Recall vs corruption (N=120, K=8, exact-recovery rate over 200 trials):")
    for frac, acc in rc:
        bar = "#" * round(acc * 40)
        print(f"      corrupt {frac:.0%}   exact recall {acc:6.3f}  {bar}")

    print("\n[2] Capacity (N=120, 20% corruption, exact-recall rate vs load K/N):")
    for K, load, acc in exp_capacity():
        print(f"      K={K:3d}  load={load:5.2f}   exact recall {acc:6.3f}")

    print("\n[3] Correlated-pattern stress (noisy variants of 4 prototypes, 70% shared;"
          " 15% corruption):")
    for K, per, acc in exp_correlated_capacity():
        print(f"      K={K:3d}  ({per:2d}/prototype)   exact recall {acc:6.3f}")

    ca = exp_content_addressable()
    print(f"\n[4] Content-addressable (probe@35% corruption -> exact nearest stored): {ca:.3f}")

    print("\n[5] Distortion tolerance — global phase screen on probe (cert [518] medium):")
    print("    (naive recall vs phase-locked recall; correct = stored pattern in probe's frame)")
    for phi, naive, locked in exp_distortion_tolerance():
        print(f"      phi={phi:2d}   naive {naive:6.3f}   phase-locked {locked:6.3f}")
