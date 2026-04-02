QA_COMPLIANCE = "library_module — wraps canonical qa_orbit_rules"
"""QA orbit arithmetic — re-exports from canonical qa_orbit_rules."""

import sys
import os

# Add the repo root so we can import the canonical module
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qa_orbit_rules import (  # noqa: E402, ORBIT-4
    qa_step,
    orbit_family,
    orbit_period,
    norm_f,
    v3,
)

__all__ = ["qa_mod", "qa_step", "orbit_family", "orbit_period", "norm_f", "v3"]


def qa_mod(x: int, m: int) -> int:
    """A1-compliant modular reduction: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1
