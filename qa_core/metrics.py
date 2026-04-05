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


def e8_alignment(tuples: np.ndarray) -> float:
    """Project QA tuples into 8D and score alignment to a fixed ideal root."""

    qa_vectors_8d = np.zeros((tuples.shape[0], 8))
    qa_vectors_8d[:, :4] = tuples
    norms = np.linalg.norm(qa_vectors_8d, axis=1)
    norms[norms == 0] = 1
    normalized_vectors = qa_vectors_8d / norms[:, np.newaxis]
    ideal_root = np.array([1, 1, 2, 3, 0, 0, 0, 0], dtype=float)
    ideal_root_norm = ideal_root / np.linalg.norm(ideal_root)
    return float(np.mean(np.abs(normalized_vectors @ ideal_root_norm)))


def harmonic_index(loss: float, alignment: float) -> float:
    """Combine loss and alignment into the repo's harmonic index score."""

    return float(alignment * np.exp(-0.1 * loss))
