QA_COMPLIANCE = "canonical_orbit_module — not an empirical script"
"""qa-arithmetic — Quantum Arithmetic core primitives.

Pure Python, zero dependencies. The foundation for qa-observer, qa-pim, qa-graph.

Core operations (see core.py for the canonical signatures):
    qa_step, qa_mod, orbit_family, orbit_period, norm_f, v3,
    qa_tuple, identities
"""

__version__ = "0.1.0"

from qa_arithmetic.core import (
    qa_step,
    qa_mod,
    orbit_family,
    orbit_period,
    norm_f,
    v3,
    qa_tuple,
    KNOWN_MODULI,
    self_test,
)

from qa_arithmetic.identities import identities, IDENTITY_NAMES

__all__ = [
    "qa_step",
    "qa_mod",
    "orbit_family",
    "orbit_period",
    "norm_f",
    "v3",
    "qa_tuple",
    "identities",
    "IDENTITY_NAMES",
    "KNOWN_MODULI",
    "self_test",
]
