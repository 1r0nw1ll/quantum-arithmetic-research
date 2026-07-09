#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=synthetic_pattern_generator, state_alphabet=mod24_A1_compliant"
"""
Controlled dimensionality sweep for the phase-conjugate associative memory
(follow-up to the climate ENSO boundary result).

Thread-3 found the phase-lock artifact-robustness collapses on low-dim patterns
(climate, 5 channels) but is strong on high-dim ones (EEG 23-92, morphology 256).
This isolates the mechanism: hold the modulus m and #classes K fixed, vary the
pattern dimension N, and measure whether phase-locked recall beats naive under a
systemic shift, plus the false-lock rate.

Theory / prediction: phase-lock scans m compensation phases x K patterns = m*K
candidate alignments; the TRUE pattern scores overlap ~ N*(1-corruption) while
each spurious (psi,k) pair scores ~ Binomial(N, 1/m). Phase-lock false-locks when
the max of the ~m*K spurious scores exceeds the true score -- likely for small N.
PREDICTION (committed): phase-lock advantage <= 0 for N <~ 12-16, clearly > 0 for
N >~ 32; crossover N* ~ 16-24.
"""
from __future__ import annotations
import numpy as np

M = 24
RNG = np.random.default_rng(7)


def qa_mod(x):
    return ((np.asarray(x, np.int64) - 1) % M) + 1


def qa_add(a, b):
    return qa_mod(np.asarray(a, np.int64) + np.asarray(b, np.int64))


def overlap(x, P):
    return (P == x[None, :]).sum(axis=1)          # (K,) match counts


def naive_cls(probe, P, labels):
    return labels[int(np.argmax(overlap(probe, P)))]


def pl_cls(probe, P, labels):
    """Phase-locked: scan psi, take global argmax over (psi, pattern). Returns
    (label, locked_correctly_flag placeholder)."""
    best_k, best = 0, -1
    for psi in range(1, M + 1):
        C = overlap(qa_add(probe, psi), P)
        k = int(np.argmax(C))
        if C[k] > best:
            best, best_k = C[k], k
    return labels[best_k]


def corrupt(x, frac, rng):
    x = x.copy(); idx = rng.choice(len(x), int(frac * len(x)), replace=False)
    x[idx] = rng.integers(1, M + 1, len(idx)); return x


def make_classes(N, K, per_class, distinct_frac, within_noise, rng):
    """Structured classes: each class has a prototype that agrees with the others
    on (1-distinct_frac)*N shared 'background' dims and differs on the rest.
    distinct_frac -> 1 = well-separated; -> 0 = overlapping. Members = prototype +
    within-class jitter. Returns (stored P, labels, prototypes)."""
    n_distinct = max(1, int(distinct_frac * N))
    background = rng.integers(1, M + 1, N)
    protos = []
    for c in range(K):
        p = background.copy()
        didx = rng.choice(N, n_distinct, replace=False)
        p[didx] = rng.integers(1, M + 1, n_distinct)
        protos.append(p)
    P, labels = [], []
    for c in range(K):
        for _ in range(per_class):
            member = corrupt(protos[c].copy(), within_noise, rng)
            P.append(member); labels.append(c)
    return np.array(P), np.array(labels), protos


def sweep_recall(K=4, per_class=4, corruption=0.2, phi=6, trials=400):
    """Original RECALL setting (probe = corrupted STORED pattern) — the [519]
    setup. Shows phase-lock works at any N because the probe is in memory."""
    print("=== RECALL (probe is a corrupted stored pattern) ===")
    print(f"{'N':>5s} {'naive_shift':>11s} {'PL_shift':>9s} {'advantage':>9s}")
    for N in (4, 8, 16, 64, 256):
        rng = np.random.default_rng(100 + N)
        naive_s = pl_s = 0
        for _ in range(trials):
            P = rng.integers(1, M + 1, (K * per_class, N))
            labels = np.repeat(np.arange(K), per_class)
            s = rng.integers(len(P))
            probe = qa_add(corrupt(P[s], corruption, rng), phi)
            naive_s += (naive_cls(probe, P, labels) == labels[s])
            pl_s += (pl_cls(probe, P, labels) == labels[s])
        naive_s /= trials; pl_s /= trials
        print(f"{N:5d} {naive_s:11.3f} {pl_s:9.3f} {pl_s-naive_s:+9.3f}")


def sweep_generalize(N=32, K=4, per_class=8, corruption=0.2, phi=6, within=0.1, trials=400):
    """GENERALIZATION / classification (probe = HELD-OUT member of a class) — the
    climate/EEG setup. Vary between-class SEPARABILITY (distinct_frac)."""
    print(f"\n=== GENERALIZATION (held-out probe -> nearest class), N={N}, "
          f"within-class noise={within:.0%}, systemic phi={phi} ===")
    print(f"{'distinct_frac':>13s} {'clean':>7s} {'naive_shift':>11s} {'PL_shift':>9s} {'advantage':>9s}")
    for df in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        rng = np.random.default_rng(int(1000 * df) + N)
        clean = naive_s = pl_s = 0
        for _ in range(trials):
            P, labels, protos = make_classes(N, K, per_class, df, within, rng)
            c = rng.integers(K)
            heldout = corrupt(protos[c].copy(), within, rng)   # a NEW member of class c
            probe0 = corrupt(heldout, corruption, rng)
            clean += (naive_cls(probe0, P, labels) == c)
            probe = qa_add(probe0, phi)
            naive_s += (naive_cls(probe, P, labels) == c)
            pl_s += (pl_cls(probe, P, labels) == c)
        clean /= trials; naive_s /= trials; pl_s /= trials
        print(f"{df:13.2f} {clean:7.3f} {naive_s:11.3f} {pl_s:9.3f} {pl_s-naive_s:+9.3f}")


def sweep_crowding(N=8, K=3, phi=6, within=0.1, trials=300):
    """Crowding: fix N and K, vary the number of stored patterns per class.
    Climate stores ~450 patterns in a 5-D space."""
    print(f"\n=== CROWDING (N={N}, K={K}, distinct_frac=0.5, phi={phi}) ===")
    print(f"{'per_class':>9s} {'total':>6s} {'clean':>7s} {'naive_shift':>11s} {'PL_shift':>9s} {'adv':>7s}")
    for per_class in (4, 12, 40, 100, 200):
        rng = np.random.default_rng(per_class + N)
        clean = naive_s = pl_s = 0
        for _ in range(trials):
            P, labels, protos = make_classes(N, K, per_class, 0.5, within, rng)
            c = rng.integers(K)
            probe0 = corrupt(corrupt(protos[c].copy(), within, rng), 0.2, rng)
            clean += (naive_cls(probe0, P, labels) == c)
            probe = qa_add(probe0, phi)
            naive_s += (naive_cls(probe, P, labels) == c)
            pl_s += (pl_cls(probe, P, labels) == c)
        clean /= trials; naive_s /= trials; pl_s /= trials
        print(f"{per_class:9d} {K*per_class:6d} {clean:7.3f} {naive_s:11.3f} {pl_s:9.3f} {pl_s-naive_s:+7.3f}")


def sweep_label_relevance(N=8, K=3, per_class=40, phi=6, trials=300):
    """Label-metric alignment: the class is defined by a threshold on `rel_dims`
    of N dims; the rest are label-IRRELEVANT noise (like ENSO defined by ONI while
    NAO/AO/PDO/AMO are ~noise). Fewer relevant dims = more misalignment."""
    print(f"\n=== LABEL-METRIC MISALIGNMENT (N={N}, {per_class}/class, crowded) ===")
    print(f"{'rel_dims':>9s} {'clean':>7s} {'naive_shift':>11s} {'PL_shift':>9s} {'adv':>7s}")
    for rel in (N, N // 2, 2, 1):
        rng = np.random.default_rng(rel + N)
        clean = naive_s = pl_s = 0
        for _ in range(trials):
            # class = pattern of the first `rel` dims (label-relevant); the other
            # N-rel dims are independent noise for every stored member and probe.
            protos = [rng.integers(1, M + 1, rel) for _ in range(K)]
            P, labels = [], []
            for c in range(K):
                for _ in range(per_class):
                    mem_rel = corrupt(protos[c].copy(), 0.1, rng)
                    noise = rng.integers(1, M + 1, N - rel)
                    P.append(np.concatenate([mem_rel, noise])); labels.append(c)
            P = np.array(P); labels = np.array(labels)
            c = rng.integers(K)
            probe_rel = corrupt(protos[c].copy(), 0.1, rng)
            probe0 = np.concatenate([probe_rel, rng.integers(1, M + 1, N - rel)])
            clean += (naive_cls(probe0, P, labels) == c)
            probe = qa_add(probe0, phi)
            naive_s += (naive_cls(probe, P, labels) == c)
            pl_s += (pl_cls(probe, P, labels) == c)
        clean /= trials; naive_s /= trials; pl_s /= trials
        print(f"{rel:9d} {clean:7.3f} {naive_s:11.3f} {pl_s:9.3f} {pl_s-naive_s:+7.3f}")


def sweep_class_structure(N=12, K=3, per_class=60, phi=6, trials=300):
    """Discrete-attractor classes vs threshold-on-a-continuum classes (like ENSO).
    Discrete: each class = corrupted copy of a fixed prototype (an attractor).
    Continuum: a latent scalar per sample -> thresholded into K bins -> the WHOLE
    pattern is a phase-gradient of that latent (members have SPREAD phases, no
    shared prototype). Tests whether phase-lock needs discrete attractors."""
    print(f"\n=== CLASS STRUCTURE (N={N}, K={K}, {per_class}/class, phi={phi}) ===")
    print(f"{'structure':>12s} {'clean':>7s} {'naive_shift':>11s} {'PL_shift':>9s} {'adv':>7s}")

    def eval_set(builder):
        rng = np.random.default_rng(N + K)
        clean = naive_s = pl_s = 0
        for _ in range(trials):
            P, labels, probe0, c = builder(rng)
            clean += (naive_cls(probe0, P, labels) == c)
            probe = qa_add(probe0, phi)
            naive_s += (naive_cls(probe, P, labels) == c)
            pl_s += (pl_cls(probe, P, labels) == c)
        return clean / trials, naive_s / trials, pl_s / trials

    def discrete(rng):
        protos = [rng.integers(1, M + 1, N) for _ in range(K)]
        P, labels = [], []
        for cc in range(K):
            for _ in range(per_class):
                P.append(corrupt(protos[cc].copy(), 0.1, rng)); labels.append(cc)
        c = rng.integers(K)
        probe0 = corrupt(corrupt(protos[c].copy(), 0.1, rng), 0.2, rng)
        return np.array(P), np.array(labels), probe0, c

    def continuum(rng):
        # latent scalar t in [0,1] -> class = bin(t); pattern = phase gradient of t
        base = rng.integers(1, M + 1, N)
        P, labels, ts = [], [], []
        for _ in range(K * per_class):
            t = rng.random()
            cc = min(K - 1, int(t * K))
            pat = qa_add(base, int(round(t * (M - 1))))     # global phase set by t
            P.append(corrupt(pat, 0.15, rng)); labels.append(cc); ts.append(t)
        c = rng.integers(K)
        t = (c + 0.5) / K
        probe0 = corrupt(qa_add(base, int(round(t * (M - 1)))), 0.2, rng)
        return np.array(P), np.array(labels), probe0, c

    for name, b in (("discrete", discrete), ("continuum", continuum)):
        cl, ns, ps = eval_set(b)
        print(f"{name:>12s} {cl:7.3f} {ns:11.3f} {ps:9.3f} {ps-ns:+7.3f}")


if __name__ == "__main__":
    print("Phase-conjugate memory: what actually drives the climate boundary?\n")
    sweep_recall()
    sweep_generalize(N=32)
    sweep_generalize(N=8)
    sweep_crowding()
    sweep_label_relevance()
    sweep_class_structure()
    print("\nclean/naive/PL are CLASS accuracy under the systemic shift; "
          "advantage>0 means phase-lock helps.")
