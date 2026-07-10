"""Shared metric helpers for QA experiments — A1-compliant integer tuples."""

from __future__ import annotations

QA_COMPLIANCE = "observer=integer_coherence_field, state_alphabet=mod{1..m}, A1+S2_compliant"

import numpy as np


def qa_tuples(b: np.ndarray, e: np.ndarray, modulus: int) -> np.ndarray:
    """Return canonical QA tuples `(b, e, d, a)` modulo ``modulus``.

    Inputs b, e must be integer arrays with values in {1, ..., modulus}.
    Derived coords use A1-compliant reduction `((x - 1) % m) + 1`, never
    the naive `x % m` form that can produce 0.
    """

    d = ((b + e - 1) % modulus) + 1  # noqa: A2-1 — canonical A2 derivation, stored before stacking
    a = ((b + 2 * e - 1) % modulus) + 1  # noqa: A2-2 — canonical A2 derivation, stored before stacking
    return np.stack([b, e, d, a], axis=1)


def harmonic_loss(tuples: np.ndarray, modulus: int) -> float:
    """Measure modular deviation from the canonical ellipse identity."""

    e = tuples[:, 1]
    d = tuples[:, 2]
    a = tuples[:, 3]
    lhs = (a * a) % modulus
    rhs = (d * d + 2 * d * e + e * e) % modulus
    diff = np.abs(lhs - rhs)
    loss = np.minimum(diff, modulus - diff)
    return float(np.mean(loss * loss))


def e8_alignment_legacy(tuples: np.ndarray) -> float:
    """LEGACY (pre-2026-07-10, kept for reproducibility): NOT an E8 alignment.

    This was the historical `e8_alignment`: it zero-pads (b,e,d,a) to 8D (so it stays
    4D) and scores mean |cosine| to a SINGLE hardcoded vector [1,1,2,3,0,0,0,0] -- a
    Fibonacci direction that is NOT an E8 root (norm^2 = 15, roots have norm^2 = 2).
    It never touched the E8 root system. Superseded by the icosian-grounded
    `e8_alignment` below; retained only to reproduce pre-fix Harmonic Index values.
    See qa_e8_icosian_grounding.sage and docs/theory/QA_AS_QUATERNION_ORDER.md."""

    qa_vectors_8d = np.zeros((tuples.shape[0], 8))
    qa_vectors_8d[:, :4] = tuples
    norms = np.linalg.norm(qa_vectors_8d, axis=1)
    norms[norms == 0] = 1
    normalized_vectors = qa_vectors_8d / norms[:, np.newaxis]
    ideal_root = np.array([1, 1, 2, 3, 0, 0, 0, 0], dtype=float)
    ideal_root_norm = ideal_root / np.linalg.norm(ideal_root)
    return float(np.mean(np.abs(normalized_vectors @ ideal_root_norm)))


def _icosian_600cell_directions() -> np.ndarray:
    """The 120 unit icosians (binary icosahedral group 2I) under one real embedding
    = the 120 vertices of the 600-CELL, as unit direction vectors in R^4.

    This IS the golden/E8 structure QA is grounded in (E8 = two 600-cells; the
    icosian ring over Q(sqrt5) is E8 -- Voight GTM 288 / Conway-Sloane SPLAG;
    verified in qa_icosian_order.py). Grounds the E8-alignment metric rigorously,
    at the correct prime (5 / sqrt5), rather than the legacy single Fibonacci vector.
    """
    from itertools import permutations, product
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    half, iphi2, phi2 = 0.5, (phi - 1) / 2.0, phi / 2.0
    S = set()
    for pos in range(4):                                  # 8: +-1 in one slot
        for s in (1.0, -1.0):
            v = [0.0, 0.0, 0.0, 0.0]; v[pos] = s; S.add(tuple(v))
    for signs in product((half, -half), repeat=4):        # 16: (+-1/2)^4
        S.add(tuple(signs))
    def parity(p):
        p = list(p); par = 1
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]: par = -par
        return par
    for perm in permutations(range(4)):                   # 96: even perms of (0,+-1/2,+-1/2phi,+-phi/2)
        if parity(perm) != 1: continue
        for sg in product((1, -1), repeat=3):
            vals = [0.0, half * sg[0], iphi2 * sg[1], phi2 * sg[2]]
            v = [None] * 4
            for slot, src in zip(perm, vals): v[slot] = src
            S.add(tuple(v))
    D = np.array(sorted(S), dtype=float)                  # 120 x 4, already unit norm
    return D / np.linalg.norm(D, axis=1, keepdims=True)


_ICOSIAN_DIRS = _icosian_600cell_directions()


def e8_alignment(tuples: np.ndarray) -> float:
    """Alignment of QA tuples to the golden/E8 (icosian) geometry.

    Each tuple (b,e,d,a) is a direction in R^4; score the mean over tuples of its
    best (max) cosine similarity to the 120 icosian 600-cell directions -- the
    binary icosahedral group 2I, i.e. QA's genuine E8 structure over Q(sqrt5)
    (the icosian ring; see qa_icosian_order.py). Replaces the legacy single-vector
    metric (kept as e8_alignment_legacy). Observer-layer geometry (Theorem NT).

    HONEST CAVEAT (qa_e8_alignment_hi_comparison.py): this is the mathematically
    correct E8 alignment, but it does NOT discriminate QA harmonicity -- the
    600-cell covers directions in R^4 finely enough that this max-cosine readout
    saturates -- random tuples align ~as well as structured ones
    (~0.96 either way). The legacy metric, though mislabeled, weakly discriminated
    (it aligned to the Fibonacci direction, and QA orbits are Fibonacci). And since
    the QA loss ~ 0, HI = alignment. So the Harmonic Index needs rethinking, not
    just a corrected root vector. Choosing this as the default is a legitimacy-over-
    discrimination trade-off, made explicit here.
    """
    v = np.asarray(tuples, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cos = (v / norms) @ _ICOSIAN_DIRS.T                   # (n_tuples, 120)
    return float(np.mean(np.max(cos, axis=1)))


def harmonic_index(loss: float, alignment: float) -> float:
    """Combine loss and alignment into the repo's harmonic index score."""

    return float(alignment * np.exp(-0.1 * loss))
