"""qa_kg.orbit_failure_enumeration — exact failure-density enumeration on QA orbit graphs.

QA_COMPLIANCE = "memory_infra utility — exact integer enumeration over finite QA state spaces; no observer projection, no float in QA state path; sampling helpers (boundary-only) use int RNG"

Provides reusable enumeration + sampling primitives that recast Kochenderfer
Ch. 7 failure-probability estimation (Kochenderfer, 2026; Algorithms for
Validation, MIT Press, val.pdf §7.1) onto finite QA orbit graphs. The
load-bearing observation: Kochenderfer's general estimator
    p_fail = E_{tau ~ p(.)}[1{tau not in psi}] = integral 1{tau not in psi} p(tau) d tau
specializes, on a finite QA state space S_m of cardinality m^2, to an
exact finite sum
    p_fail = |{s in S_m : s not in psi}| / |S_m|
which can be computed by enumeration with zero variance, while the
canonical direct-sampling estimator (Algorithm 7.1) has standard error
sqrt(p (1 - p) / N) per Kochenderfer eq. 7.3.

This module is the utility-factored core of cert family [263]
qa_failure_density_enumeration_cert_v1 and is designed for reuse by
the existing reachability / orbit-classification certs:
- cert [191] qa_bateson_learning_levels_cert_v1
- cert [193] qa_levin_cognitive_lightcone_cert_v1
- cert [194] qa_cognition_space_morphospace_cert_v1

The orbit-family classifier reproduced here is the canonical S_9
classifier from cert [194] (Sole, 2026; arxiv:2601.12837 via Dale, 2026):
- singularity   <=> (b, e) == (9, 9)                           1 state
- satellite     <=> (b mod 3 == 0) AND (e mod 3 == 0) AND not singularity   8 states
- cosmos        <=> otherwise                                  72 states

Public API (stable; cert validators import these names):
    enumerate_orbit_class_counts(modulus)
        -> {'singularity': n1, 'satellite': n2, 'cosmos': n3, 'total': N}
    exact_success_failure_probability(modulus, target_class)
        -> {'p_success': Fraction, 'p_failure': Fraction, 'numerator': k, 'denominator': N}
    direct_sampling_estimate(modulus, target_class, n_samples, seed)
        -> {'p_hat': float, 'n_in_target': int, 'n_samples': int, 'seed': int}
    theoretical_standard_error(p, n_samples)
        -> sigma = sqrt(p (1 - p) / n_samples) as float

Theorem NT (firewall) compliance:
- enumerate_orbit_class_counts and exact_success_failure_probability return
  exact integer / Fraction values; no floats in the QA-discrete path.
- direct_sampling_estimate is an OBSERVER PROJECTION at the output boundary:
  the only continuous-domain operation is the empirical mean float division
  inside p_hat, executed AFTER all QA-state enumeration. The sampling RNG
  is integer (random.Random with explicit seed) and selects an integer index
  into the enumerated state list, so the QA state path stays integer
  throughout.
- theoretical_standard_error is closed-form Bernoulli variance from
  Kochenderfer eq. 7.3; same firewall classification as direct_sampling.

Source attribution: Kochenderfer, Wheeler, Katz, Corso, Moss (Kochenderfer, 2026),
Algorithms for Validation, MIT Press CC-BY-NC-ND. Anchored at
docs/theory/kochenderfer_validation_excerpts.md#val-7-1-direct-estimation-pfail
and the QA-side bridge at docs/specs/QA_KOCHENDERFER_BRIDGE.md.
"""

from __future__ import annotations

QA_COMPLIANCE = "memory_infra utility — exact integer enumeration over finite QA state spaces; no observer projection, no float in QA state path; sampling helpers (boundary-only) use int RNG"

import random
from fractions import Fraction


# Closed set of orbit class names recognized by this utility. Mirrors the
# S_9 classifier in qa_cognition_space_morphospace_cert_v1 [194].
ORBIT_CLASSES: tuple[str, ...] = ("singularity", "satellite", "cosmos")


def qa_mod(x: int, m: int) -> int:
    """A1-compliant modular reduction: result in {1, ..., m}, never 0.

    Mirrors qa_cognition_space_morphospace_cert_validate.py qa_mod (cert [194])
    and the canonical specification in QA_AXIOMS_BLOCK.md A1.
    """
    return ((int(x) - 1) % int(m)) + 1


def qa_step(b: int, e: int, m: int) -> tuple[int, int]:
    """Fibonacci dynamic on (Z/mZ)^* with A1-compliant cosets.

    Mirrors qa_cognition_space_morphospace_cert_validate.py qa_step (cert [194]):
    (b, e) -> (e, qa_mod(b + e, m)).
    """
    return (int(e), qa_mod(int(b) + int(e), int(m)))


def orbit_family_s9(b: int, e: int) -> str:
    """Canonical S_9 orbit-family classifier reproduced from cert [194].

    Returns one of ORBIT_CLASSES.
    """
    if int(b) == 9 and int(e) == 9:
        return "singularity"
    if (int(b) % 3 == 0) and (int(e) % 3 == 0):
        return "satellite"
    return "cosmos"


def _classifier_for_modulus(modulus: int):
    """Return the canonical orbit-family classifier callable for modulus.

    Currently only m=9 has a registered canonical classifier (from cert [194]).
    Other moduli raise NotImplementedError so the cert never silently extrapolates.
    """
    if int(modulus) == 9:
        return orbit_family_s9
    raise NotImplementedError(
        f"orbit_failure_enumeration: no canonical orbit-family classifier "
        f"registered for modulus={modulus!r}. Currently only m=9 is canonical "
        f"(cert [194] qa_cognition_space_morphospace_cert_v1). Mod-24 / other "
        f"moduli would need a published classifier landed in a future cert."
    )


def enumerate_orbit_class_counts(modulus: int) -> dict[str, int]:
    """Exact enumeration of orbit-family populations on S_modulus.

    Returns a dict keyed by ORBIT_CLASSES plus 'total'. For m=9 reproduces
    the canonical cert [194] populations:
        {'singularity': 1, 'satellite': 8, 'cosmos': 72, 'total': 81}
    """
    classifier = _classifier_for_modulus(modulus)
    counts: dict[str, int] = {cls: 0 for cls in ORBIT_CLASSES}
    total = 0
    for b in range(1, int(modulus) + 1):
        for e in range(1, int(modulus) + 1):
            counts[classifier(b, e)] += 1
            total += 1
    counts["total"] = total
    return counts


def exact_success_failure_probability(
    modulus: int, target_class: str
) -> dict[str, object]:
    """Exact rational p_success / p_failure for a target orbit family.

    p_success = |{s in S_m : class(s) == target_class}| / |S_m|
    p_failure = 1 - p_success

    For m=9 and target_class in {'singularity', 'satellite', 'cosmos'}
    reproduces the cert [194] ratios {1/81, 8/81, 72/81} exactly (as Fractions).

    Theorem NT compliant: returned probabilities are Fractions (exact integer
    ratio), never floats; this stays inside the QA discrete layer.
    """
    if target_class not in ORBIT_CLASSES:
        raise ValueError(
            f"target_class={target_class!r} not in {ORBIT_CLASSES!r}"
        )
    counts = enumerate_orbit_class_counts(modulus)
    total = counts["total"]
    numerator = counts[target_class]
    if total <= 0:
        raise RuntimeError(f"enumeration produced total={total} on m={modulus}")
    p_success = Fraction(numerator, total)
    p_failure = Fraction(1, 1) - p_success
    return {
        "p_success": p_success,
        "p_failure": p_failure,
        "numerator": int(numerator),
        "denominator": int(total),
    }


def direct_sampling_estimate(
    modulus: int,
    target_class: str,
    n_samples: int,
    seed: int,
) -> dict[str, object]:
    """Kochenderfer-style direct-sampling estimator (Algorithm 7.1).

    Boundary-only operation per Theorem NT: integer sampling over the
    enumerated state set, then a single float division at the output
    boundary to produce p_hat. The QA state path stays integer.

    Reproducibility requirement: seed is REQUIRED and explicit. Cert
    family [263] FDE_SAMPLING gate enforces seeded reproducibility.
    """
    if target_class not in ORBIT_CLASSES:
        raise ValueError(
            f"target_class={target_class!r} not in {ORBIT_CLASSES!r}"
        )
    if int(n_samples) <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    if seed is None:
        raise ValueError(
            "direct_sampling_estimate requires an explicit integer seed for "
            "reproducibility (cert [263] FDE_SAMPLING gate enforces this)."
        )
    classifier = _classifier_for_modulus(modulus)
    states: list[tuple[int, int]] = [
        (b, e)
        for b in range(1, int(modulus) + 1)
        for e in range(1, int(modulus) + 1)
    ]
    rng = random.Random(int(seed))
    n_in_target = 0
    for _ in range(int(n_samples)):
        b, e = rng.choice(states)
        if classifier(b, e) == target_class:
            n_in_target += 1
    # OBSERVER PROJECTION (output boundary): single float division for p_hat.
    p_hat = float(n_in_target) / float(int(n_samples))
    return {
        "p_hat": p_hat,
        "n_in_target": int(n_in_target),
        "n_samples": int(n_samples),
        "seed": int(seed),
    }


def theoretical_standard_error(p: Fraction | float, n_samples: int) -> float:
    """Closed-form Bernoulli standard error per Kochenderfer eq. 7.3.

    sigma_hat = sqrt(p (1 - p) / N)

    Output is a float (observer projection at the output boundary). p may
    be passed as a Fraction (preferred — exact rational from the
    enumeration path) or a float (e.g., empirical p_hat).
    """
    if int(n_samples) <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    p_float = float(p)
    if not 0.0 <= p_float <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p_float!r}")
    variance = p_float * (1.0 - p_float) / float(int(n_samples))
    return variance ** 0.5


# Convenience alias so callers can spell the metric the way Kochenderfer
# Ch. 7 spells it.
direct_estimation = direct_sampling_estimate


__all__ = (
    "ORBIT_CLASSES",
    "qa_mod",
    "qa_step",
    "orbit_family_s9",
    "enumerate_orbit_class_counts",
    "exact_success_failure_probability",
    "direct_sampling_estimate",
    "direct_estimation",
    "theoretical_standard_error",
    "QA_COMPLIANCE",
)
