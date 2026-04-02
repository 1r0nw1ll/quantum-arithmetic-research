QA_COMPLIANCE = "library_module — not an empirical script"
"""qa_observer — QA Coherence Index for domain-general structural analysis."""

from qa_observer.core import TopographicObserver, QCI
from qa_observer.orbits import orbit_family, qa_step, qa_mod
from qa_observer.surrogates import SurrogateTest

__version__ = "0.1.0"
__all__ = [
    "TopographicObserver",
    "QCI",
    "SurrogateTest",
    "orbit_family",
    "qa_step",
    "qa_mod",
]
