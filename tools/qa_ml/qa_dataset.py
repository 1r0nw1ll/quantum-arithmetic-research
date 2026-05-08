"""Orbit-labeled (b, e) dataset for QA-ML benchmarks.

Enumerates the full grid {1,...,m}^2 (deterministic; no stochastic generators)
and produces parallel raw, QA-algebraic, and QA-full feature matrices alongside
integer orbit labels from qa_orbit_rules.orbit_family.

QA_COMPLIANCE = "qa_ml_dataset — exhaustive integer enumeration, A1-compliant range"
"""

from __future__ import annotations

from .qa_features import qa_packet, qa_packet_full, label


def all_pairs(m: int) -> list[tuple[int, int]]:
    """All (b, e) pairs in {1,...,m}^2 (A1-compliant, no zero state)."""
    return [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]


def build_dataset(m: int) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int, int, int, int, int, int]],
    list[tuple[int, int, int, int, int, int, int, int, int]],
    list[int],
    list[tuple[int, int]],
]:
    """Return (X_raw, X_qa, X_qa_full, y, pairs) for modulus m.

    X_raw[i]     = (b, e)
    X_qa[i]      = (b, e, d, a, C, F, G)                       — algebraic packet
    X_qa_full[i] = (b, e, d, a, C, F, G, phi_b, phi_e)         — algebraic + mod phase
    y[i]         = orbit label int
    pairs[i]     = same as X_raw[i]
    """
    pairs = all_pairs(m)
    x_raw = [(b, e) for (b, e) in pairs]
    x_qa = [qa_packet(b, e) for (b, e) in pairs]
    x_qa_full = [qa_packet_full(b, e, m) for (b, e) in pairs]
    y = [label(b, e, m) for (b, e) in pairs]
    return x_raw, x_qa, x_qa_full, y, pairs
