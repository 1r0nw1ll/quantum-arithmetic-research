#!/usr/bin/env python3
"""
qa_empirical_template_nt_compliant.py — Scaffold for NT-Compliant Track D Scripts

Copy this file and fill in the three sections marked TODO.

Architecture (Theorem NT — QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1):

  [OBSERVER LAYER]    domain signal → discrete class label
                          ↓  (continuous computation confined here)
  [QA LAYER]          class label → declared (b,e) int pair → orbit family
                          ↓  (integers only — no floats cross this boundary)
  [PROJECTION LAYER]  orbit statistics → empirical comparison → reporting

Axiom checklist (A1, A2, T1, T2, S1, S2):
  A1: All (b,e) values in {1,...,MODULUS}. Never 0. Use STATE_ALPHABET lookup only.
  A2: d = b + e  and  a = b + 2 * e  — always derived, never assigned independently.
  T1: QA time = path step count k (integer). No continuous time variable in QA layer.
  T2: Continuous signal enters ONLY in observer layer. Never in QA layer.
  S1: Write b*b, never b-squared (avoids libm ULP drift).
  S2: b, e are Python int throughout QA layer. No numpy scalars, no floats.
"""

# ── REQUIRED: fill in before use ──────────────────────────────────────────────

from qa_orbit_rules import norm_f, v3, orbit_family, qa_step

QA_COMPLIANCE = {
    # TODO: fill in
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "cert_family": "[???] CERT_NAME",                    # e.g. "[110] QA_SEISMIC_CONTROL_CERT.v1"
    "axioms_checked": ["A1", "A2", "T1", "T2", "S1", "S2"],
    "observer": "TODO: describe the observer function",
    "state_alphabet": ["TODO: list discrete states"],
    "qa_layer_types": "int",
    "projection_types": "float",
}

# ── Discrete state alphabet ────────────────────────────────────────────────────
# A1: All (b, e) values MUST be in {1,...,MODULUS}. No zeros.
# Declare a priori — NOT derived from any signal value.
# Each state label maps to a fixed (b, e) pair.
#
# TODO: define your state alphabet and (b,e) assignments.

MODULUS = 24   # TODO: choose modulus (9 or 24)

STATE_ALPHABET: dict[str, tuple[int, int]] = {  # noqa: ORBIT-6 — template: fill in states then run qa_observer_alphabet_audit.py
    # "state_name": (b, e),   # orbit: cosmos / satellite / singularity
    # Example:
    # "quiet":        (9,  9),  # singularity
    # "active":       (1,  8),  # cosmos
}


# ── QA layer — integers only past this line ───────────────────────────────────
# norm_f, v3, orbit_family, qa_step imported from qa_orbit_rules above.

def _assert_states_valid() -> None:
    """A1 gate: called at import time. Fails fast if any declared state violates A1."""
    for name, (b, e) in STATE_ALPHABET.items():
        assert 1 <= b <= MODULUS, f"State '{name}': b={b} violates A1 (must be in {{1,...,{MODULUS}}})"
        assert 1 <= e <= MODULUS, f"State '{name}': e={e} violates A1 (must be in {{1,...,{MODULUS}}})"


def state_to_qa(label: str) -> tuple[int, int]:
    """QA layer entry: map discrete label to declared (b,e) pair."""
    b, e = STATE_ALPHABET[label]
    return int(b), int(e)  # S2: ensure Python int, not numpy scalar


def classify_sequence(labels: list[str]) -> list[str]:
    """QA layer: map label sequence → orbit family sequence. Integer arithmetic only."""
    return [orbit_family(*state_to_qa(lbl), m=MODULUS) for lbl in labels]


def orbit_counts(orbits: list[str]) -> dict[str, int]:
    """QA layer: count orbit families (integer counts)."""
    counts: dict[str, int] = {"singularity": 0, "satellite": 0, "cosmos": 0}
    for o in orbits:
        if o in counts:
            counts[o] += 1
    return counts


# ── Observer layer — floats permitted ─────────────────────────────────────────
# TODO: implement domain-specific observer that maps continuous signal → label.
# Floats are allowed ONLY here. The output must be one of STATE_ALPHABET.keys().

def observe(signal_window) -> str:
    """
    Observer: map a signal window to a discrete state label.

    TODO: implement frequency analysis / amplitude thresholds / regime detection.
    Return value must be in STATE_ALPHABET.keys().
    """
    raise NotImplementedError("TODO: implement observer for this domain")


# ── Projection layer — floats permitted ───────────────────────────────────────

def orbit_fractions(orbits: list[str]) -> dict[str, float]:
    """Projection: convert orbit counts to fractions (floats OK here)."""
    counts = orbit_counts(orbits)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def null_model_comparison(group_a_orbits: list[list[str]],
                          group_b_orbits: list[list[str]],
                          orbit: str = "singularity") -> dict:
    """
    Projection: t-test comparing orbit fractions between two groups.

    Returns dict with t-statistic, p-value, group means.
    """
    from scipy import stats as scipy_stats
    import numpy as np

    fracs_a = [orbit_fractions(o)[orbit] for o in group_a_orbits]
    fracs_b = [orbit_fractions(o)[orbit] for o in group_b_orbits]

    t_stat, p_val = scipy_stats.ttest_ind(fracs_a, fracs_b)
    return {
        "orbit": orbit,
        "group_a_mean": float(np.mean(fracs_a)),
        "group_b_mean": float(np.mean(fracs_b)),
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "direction": "A>B" if np.mean(fracs_a) > np.mean(fracs_b) else "B>A",
    }


# ── Module init ───────────────────────────────────────────────────────────────

_assert_states_valid()   # A1 gate fires at import time
