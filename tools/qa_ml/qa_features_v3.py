"""QA-ML v3 features — extends qa_features with mod-5 phase, gcd structure,
modulus-level factorization for the structure-discovery model.

Used by experiments/qa_ml/04_orbit_structure_discovery.py.

Feature groups:
  base       — qa_packet_full: (b, e, d, a, C, F, G, phi_b, phi_e)
  phase5     — (psi_b, psi_e) = (b mod 5, e mod 5); orbit invariant under qa_step
               for m where 5|m (per the [277] proof pass)
  gcd        — (gcd(b, m), gcd(e, m)); structurally distinguishes the [277]
               under-count signatures (k, 3k), (k, k), (3k, k) at m = 15k
  singular   — is_singularity flag (b == m AND e == m)
  modulus    — (m, m_div_3, m_mod_3, m_mod_5, factor count of 2/3/5/7/11)
               — these features are constant per-modulus but let the model
               key off m's structure

Integer arithmetic throughout; observer-side cast happens at the model
input boundary.

QA_COMPLIANCE = "qa_ml_features_v3 — A1/A2/S1/S2 compliant; integer features"
"""

from __future__ import annotations

from math import gcd
from typing import Iterable

from .qa_features import qa_packet_full


FEATURE_NAMES_V3: tuple[str, ...] = (
    "b", "e", "d", "a", "C", "F", "G", "phi_b", "phi_e",          # base v1+v2
    "psi_b", "psi_e",                                              # mod-5 phase
    "gcd_b_m", "gcd_e_m",                                          # gcd structure
    "is_singularity",                                              # boundary flag
    "m", "m_div_3", "m_mod_3", "m_mod_5",                          # modulus
    "fac_2", "fac_3", "fac_5", "fac_7", "fac_11",                  # factorization
)


def _factor_count(n: int, p: int) -> int:
    c = 0
    while n % p == 0:
        n //= p
        c += 1
    return c


def qa_packet_v3(b: int, e: int, m: int) -> tuple[int, ...]:
    """Full v3 feature vector for state (b, e) under modulus m.

    Returns 23 integer features in the order of FEATURE_NAMES_V3.
    """
    assert isinstance(b, int) and isinstance(e, int) and isinstance(m, int), (
        f"S2: b={b!r}, e={e!r}, m={m!r} must be Python int"
    )
    assert 1 <= b <= m and 1 <= e <= m, (
        f"A1: ({b},{e}) out of {{1,...,{m}}}"
    )

    base = qa_packet_full(b, e, m)  # (b, e, d, a, C, F, G, phi_b, phi_e)
    psi_b = b % 5
    psi_e = e % 5
    gcd_b = gcd(b, m)
    gcd_e = gcd(e, m)
    is_sing = 1 if (b == m and e == m) else 0
    m_div_3 = m // 3
    m_mod_3 = m % 3
    m_mod_5 = m % 5
    fac_2 = _factor_count(m, 2)
    fac_3 = _factor_count(m, 3)
    fac_5 = _factor_count(m, 5)
    fac_7 = _factor_count(m, 7)
    fac_11 = _factor_count(m, 11)

    return (
        *base,
        psi_b, psi_e,
        gcd_b, gcd_e,
        is_sing,
        m, m_div_3, m_mod_3, m_mod_5,
        fac_2, fac_3, fac_5, fac_7, fac_11,
    )


def qa_packets_v3(triples: Iterable[tuple[int, int, int]]) -> list[tuple[int, ...]]:
    """Vectorized v3 packets for a sequence of (b, e, m) triples."""
    return [qa_packet_v3(b, e, m) for (b, e, m) in triples]
