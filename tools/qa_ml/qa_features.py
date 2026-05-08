"""QA structural feature extraction for ML models.

Maps a discrete state (b, e) in {1,...,m}^2 to the structural packet
(b, e, d, a, C, F, G) with raw integer arithmetic:

    A2:  d = b + e          (raw, never mod-reduced here — T-operator only)
         a = b + 2e          (raw)
    S1:  G = d*d + e*e       (no ** to avoid libm ULP drift)
         all products spelled out
    S2:  b, e are Python int
    A1:  states asserted in {1,...,m} by qa_orbit_rules.orbit_family

The optional qa_packet_full extends the packet with the satellite-divisor
modular phase (b mod (m // 3), e mod (m // 3)) used by QA-ML v1+ baselines.

QA_COMPLIANCE = "qa_ml_features — A1/A2/S1/S2 compliant"
"""

from __future__ import annotations

from typing import Iterable

from qa_orbit_rules import orbit_family


FEATURE_NAMES_RAW: tuple[str, ...] = ("b", "e")
FEATURE_NAMES_QA: tuple[str, ...] = ("b", "e", "d", "a", "C", "F", "G")
FEATURE_NAMES_QA_FULL: tuple[str, ...] = (
    "b", "e", "d", "a", "C", "F", "G", "phi_b", "phi_e",
)

ORBIT_LABELS: tuple[str, ...] = ("singularity", "satellite", "cosmos")
ORBIT_TO_INT: dict[str, int] = {name: idx for idx, name in enumerate(ORBIT_LABELS)}
INT_TO_ORBIT: dict[int, str] = {idx: name for name, idx in ORBIT_TO_INT.items()}


def qa_packet(b: int, e: int) -> tuple[int, int, int, int, int, int, int]:
    """QA algebraic packet (b, e, d, a, C, F, G) — modulus-agnostic."""
    assert isinstance(b, int) and isinstance(e, int), (
        f"S2: b={b!r}, e={e!r} must be Python int"
    )
    d = b + e
    a = b + 2 * e
    c_elem = 2 * d * e
    f_elem = a * b
    g_elem = d * d + e * e
    return (b, e, d, a, c_elem, f_elem, g_elem)


def qa_packet_full(b: int, e: int, m: int) -> tuple[int, int, int, int, int, int, int, int, int]:
    """QA full packet (b, e, d, a, C, F, G, phi_b, phi_e) — algebraic + modular phase.

    phi_b, phi_e are residues modulo (m // 3), the satellite divisor in
    qa_orbit_rules.orbit_family. This phase is observer-side metadata about
    the QA layer and is NOT used as a QA state; it is a structural feature
    fed to the downstream ML observer.
    """
    assert isinstance(b, int) and isinstance(e, int) and isinstance(m, int), (
        f"S2: b={b!r}, e={e!r}, m={m!r} must be Python int"
    )
    base = qa_packet(b, e)
    sat_div = m // 3
    phi_b = b % sat_div
    phi_e = e % sat_div
    return (*base, phi_b, phi_e)


def qa_packets(pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int, int, int, int, int, int]]:
    return [qa_packet(b, e) for b, e in pairs]


def label(b: int, e: int, m: int) -> int:
    """Integer orbit label for (b, e) under modulus m. Uses canonical qa_orbit_rules."""
    return ORBIT_TO_INT[orbit_family(b, e, m)]
