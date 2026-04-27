"""test_equivalence_template.py — algorithm-database equivalence test scaffold.

QA_COMPLIANCE = "test scaffold — copy to entries/<slug>/test_equivalence.py only when a real equivalence claim exists; see TEMPLATE/ALGORITHM_ENTRY_TEMPLATE.md 'When to add a test'"

Usage:
    1. Copy this file to entries/<algorithm_slug>/test_equivalence.py.
    2. Fill in the classical_run and qa_native_run helpers with the actual algorithms.
    3. Replace the canonical_inputs and assertion bodies with the real equivalence claim.
    4. Delete this docstring header and replace with a per-test docstring citing the evidence.

Anti-pattern: do NOT add a test where the mapping is conceptual rather than
output-level. A test that reduces to 'both implementations return strings'
is noise, not evidence. See TEMPLATE/ALGORITHM_ENTRY_TEMPLATE.md.
"""

import sys
from pathlib import Path

# Repo root on sys.path so existing utilities are reachable.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent  # algorithm_database/entries/<slug>/ → repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def classical_run(*args, **kwargs):
    """Classical algorithm under test. Replace body with real implementation
    or import from <entry>/classical.py."""
    raise NotImplementedError("fill in classical_run from <entry>/classical.py")


def qa_native_run(*args, **kwargs):
    """QA-native equivalent. Replace body with real implementation
    or import from <entry>/qa_native.py (which usually wraps an existing
    utility like tools/qa_kg/orbit_failure_enumeration.py)."""
    raise NotImplementedError("fill in qa_native_run from <entry>/qa_native.py")


def test_equivalence_on_canonical_inputs():
    """Real equivalence test. Cite the evidence in the docstring.

    Example (cert [263] reuse):
        # Per cert [263] qa_failure_density_enumeration_cert_v1 fixture
        # pass_mod9_cognition_morphospace.json, the orbit-class enumeration
        # produces {1/81, 8/81, 72/81} for {singularity, satellite, cosmos}.
        # This test runs the same enumeration via this entry's qa_native and
        # verifies the same ratios.
    """
    # Fill in canonical inputs — small, self-contained, no external data.
    canonical_inputs = []  # list of test cases

    # Run both implementations. Compare in a way that makes the equivalence
    # claim explicit. Avoid exception-suppression patterns; the test should
    # FAIL loudly if the equivalence breaks.
    for inp in canonical_inputs:
        classical_out = classical_run(inp)
        qa_native_out = qa_native_run(inp)
        # Real assertion goes here. Pick the strongest comparison the claim
        # supports: bit-exact equality / equality-up-to-tolerance / set-equal /
        # distribution-equal-up-to-Bernoulli-variance / etc.
        assert classical_out == qa_native_out, (
            f"equivalence broken on input={inp}: "
            f"classical={classical_out}, qa_native={qa_native_out}"
        )


if __name__ == "__main__":
    test_equivalence_on_canonical_inputs()
    print("PASS")
