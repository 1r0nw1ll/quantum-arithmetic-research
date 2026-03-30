"""
qa_hot_mess_certificate.py

QA Certificate Schema for "Hot Mess" (incoherence / variance-dominant failure).

Maps bias–variance decomposition style measurements to QA obstruction logic:
- Bias-dominant error: stable wrong-basin reachability (repeatably wrong)
- Variance-dominant error: coherence invariant violation (run-to-run divergence)

Hard constraints:
- Exact scalars only (int/Fraction) — no floats in certificates
- Deterministic serialization (via qa_cert_core.canonical_json)
- Failure-completeness: success OR structured obstruction
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional

from qa_cert_core import (
    Scalar,
    canonical_json,
    cert_id,
    to_scalar_strict,
    utc_now_iso,
)


def _s(x: Any) -> Scalar:
    """Strict exact-scalar conversion (reject floats)."""
    return to_scalar_strict(x)


def _scalar_str(x: Optional[Scalar]) -> Optional[str]:
    if x is None:
        return None
    return str(x)


@dataclass(frozen=True)
class RunOutcomeWitness:
    run_id: int
    rng_seed: int
    step_count: int
    output_hash: str
    score: Scalar
    success: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _s(self.score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rng_seed": self.rng_seed,
            "step_count": self.step_count,
            "output_hash": self.output_hash,
            "score": str(self.score),
            "success": self.success,
        }


@dataclass(frozen=True)
class IncoherenceDecompositionWitness:
    metric_id: str
    total_error: Scalar
    bias_component: Scalar
    variance_component: Scalar
    incoherence_ratio: Scalar
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_error", _s(self.total_error))
        object.__setattr__(self, "bias_component", _s(self.bias_component))
        object.__setattr__(self, "variance_component", _s(self.variance_component))
        object.__setattr__(self, "incoherence_ratio", _s(self.incoherence_ratio))

    def verify_arithmetic(self) -> bool:
        t = Fraction(self.total_error)
        b = Fraction(self.bias_component)
        v = Fraction(self.variance_component)
        r = Fraction(self.incoherence_ratio)

        if t != b + v:
            return False
        if t == 0:
            return r == 0
        return r == v / t

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "total_error": str(self.total_error),
            "bias_component": str(self.bias_component),
            "variance_component": str(self.variance_component),
            "incoherence_ratio": str(self.incoherence_ratio),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CoherenceInvariant:
    metric_id: str
    max_incoherence_ratio: Scalar
    min_agreement_rate: Optional[Scalar] = None
    max_overthinking_spike: Optional[Scalar] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_incoherence_ratio", _s(self.max_incoherence_ratio))
        if self.min_agreement_rate is not None:
            object.__setattr__(self, "min_agreement_rate", _s(self.min_agreement_rate))
        if self.max_overthinking_spike is not None:
            object.__setattr__(self, "max_overthinking_spike", _s(self.max_overthinking_spike))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "max_incoherence_ratio": str(self.max_incoherence_ratio),
            "min_agreement_rate": _scalar_str(self.min_agreement_rate),
            "max_overthinking_spike": _scalar_str(self.max_overthinking_spike),
        }


@dataclass(frozen=True)
class ReasoningLengthWitness:
    mean_step_count: Scalar
    median_step_count: Scalar
    p95_step_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean_step_count", _s(self.mean_step_count))
        object.__setattr__(self, "median_step_count", _s(self.median_step_count))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_step_count": str(self.mean_step_count),
            "median_step_count": str(self.median_step_count),
            "p95_step_count": self.p95_step_count,
        }


@dataclass(frozen=True)
class OverthinkingSpikeWitness:
    overthinking_threshold_steps: int
    incoherence_ratio_baseline: Scalar
    incoherence_ratio_overthinking: Scalar
    spike_delta: Scalar
    spike_detected: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "incoherence_ratio_baseline", _s(self.incoherence_ratio_baseline))
        object.__setattr__(self, "incoherence_ratio_overthinking", _s(self.incoherence_ratio_overthinking))
        object.__setattr__(self, "spike_delta", _s(self.spike_delta))

    def verify_delta(self) -> bool:
        base = Fraction(self.incoherence_ratio_baseline)
        over = Fraction(self.incoherence_ratio_overthinking)
        delta = Fraction(self.spike_delta)
        return (over - base) == delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overthinking_threshold_steps": self.overthinking_threshold_steps,
            "incoherence_ratio_baseline": str(self.incoherence_ratio_baseline),
            "incoherence_ratio_overthinking": str(self.incoherence_ratio_overthinking),
            "spike_delta": str(self.spike_delta),
            "spike_detected": self.spike_detected,
        }


@dataclass(frozen=True)
class EnsembleEffectWitness:
    ensemble_size: int
    error_single: Scalar
    error_ensemble: Scalar
    variance_reduction_factor: Scalar
    ensemble_feasible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "error_single", _s(self.error_single))
        object.__setattr__(self, "error_ensemble", _s(self.error_ensemble))
        object.__setattr__(self, "variance_reduction_factor", _s(self.variance_reduction_factor))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ensemble_size": self.ensemble_size,
            "error_single": str(self.error_single),
            "error_ensemble": str(self.error_ensemble),
            "variance_reduction_factor": str(self.variance_reduction_factor),
            "ensemble_feasible": self.ensemble_feasible,
        }


def compute_agreement_rate(outcomes: List[RunOutcomeWitness]) -> Fraction:
    """
    Agreement rate = frequency of modal output (by output_hash).

    This is a coarse but certificate-friendly proxy for run-to-run coherence.
    """
    if not outcomes:
        return Fraction(0)
    counts: Dict[str, int] = {}
    for o in outcomes:
        counts[o.output_hash] = counts.get(o.output_hash, 0) + 1
    max_count = max(counts.values())
    return Fraction(max_count, len(outcomes))


def compute_reasoning_length_witness(outcomes: List[RunOutcomeWitness]) -> ReasoningLengthWitness:
    if not outcomes:
        return ReasoningLengthWitness(mean_step_count=0, median_step_count=0, p95_step_count=0)
    steps = sorted(int(o.step_count) for o in outcomes)
    n = len(steps)

    mean = Fraction(sum(steps), n)
    if n % 2 == 1:
        med: Scalar = steps[n // 2]
    else:
        med = Fraction(steps[n // 2 - 1] + steps[n // 2], 2)

    # p95 index (ceil(0.95*n) - 1) without floats
    idx = (95 * n + 100 - 1) // 100 - 1
    idx = max(0, min(n - 1, idx))
    p95 = steps[idx]

    return ReasoningLengthWitness(mean_step_count=mean, median_step_count=med, p95_step_count=p95)


@dataclass(frozen=True)
class HotMessIncoherenceCertificateV1:
    certificate_id: str
    timestamp: str
    model_id: str
    task_family: str
    eval_metric_id: str
    run_outcomes: List[RunOutcomeWitness]
    decomposition_witness: IncoherenceDecompositionWitness
    coherence_invariant: CoherenceInvariant
    success: bool

    version: str = "1.0.0"
    schema: str = "QA_HOT_MESS_INCOHERENCE_CERT.v1"
    certificate_type: str = "HOT_MESS_INCOHERENCE_CERT"

    agreement_rate: Optional[Scalar] = None
    reasoning_length_witness: Optional[ReasoningLengthWitness] = None
    overthinking_spike_witness: Optional[OverthinkingSpikeWitness] = None
    ensemble_effect_witness: Optional[EnsembleEffectWitness] = None

    failure_mode: Optional[str] = None
    failure_witness: Optional[Dict[str, Any]] = None
    qa_interpretation: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.agreement_rate is None:
            object.__setattr__(self, "agreement_rate", compute_agreement_rate(self.run_outcomes))
        else:
            object.__setattr__(self, "agreement_rate", _s(self.agreement_rate))

        if self.reasoning_length_witness is None:
            object.__setattr__(self, "reasoning_length_witness", compute_reasoning_length_witness(self.run_outcomes))

        if not self.decomposition_witness.verify_arithmetic():
            raise ValueError("decomposition_witness arithmetic invalid (expected total=bias+variance and ratio=variance/total)")

        if self.overthinking_spike_witness is not None and not self.overthinking_spike_witness.verify_delta():
            raise ValueError("overthinking_spike_witness.spike_delta mismatch (expected overthinking-baseline)")

        if self.success is False:
            if not self.failure_mode:
                raise ValueError("failure_mode required when success=False")
            if not isinstance(self.failure_witness, dict):
                raise ValueError("failure_witness dict required when success=False")

    @property
    def num_runs(self) -> int:
        return len(self.run_outcomes)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "certificate_id": self.certificate_id,
            "certificate_type": self.certificate_type,
            "timestamp": self.timestamp,
            "version": self.version,
            "schema": self.schema,
            "success": self.success,
            "model_id": self.model_id,
            "task_family": self.task_family,
            "eval_metric_id": self.eval_metric_id,
            "num_runs": self.num_runs,
            "run_outcomes": [o.to_dict() for o in self.run_outcomes],
            "decomposition_witness": self.decomposition_witness.to_dict(),
            "coherence_invariant": self.coherence_invariant.to_dict(),
            # convenience mirrors
            "total_error": str(self.decomposition_witness.total_error),
            "bias_component": str(self.decomposition_witness.bias_component),
            "variance_component": str(self.decomposition_witness.variance_component),
            "incoherence_ratio": str(self.decomposition_witness.incoherence_ratio),
            "agreement_rate": str(self.agreement_rate) if self.agreement_rate is not None else None,
            "reasoning_length_witness": self.reasoning_length_witness.to_dict() if self.reasoning_length_witness else None,
            "overthinking_spike_witness": self.overthinking_spike_witness.to_dict() if self.overthinking_spike_witness else None,
            "ensemble_effect_witness": self.ensemble_effect_witness.to_dict() if self.ensemble_effect_witness else None,
            "qa_interpretation": self.qa_interpretation,
        }
        if not self.success:
            d["failure_mode"] = self.failure_mode
            d["failure_witness"] = self.failure_witness
        return d

    def to_json(self, indent: int = 2) -> str:
        return canonical_json(self.to_dict(), indent=indent)


def create_demo_hot_mess_success_certificate() -> Dict[str, Any]:
    outcomes = [
        RunOutcomeWitness(run_id=1, rng_seed=11, step_count=120, output_hash="sha256:a", score="9/10", success=True),
        RunOutcomeWitness(run_id=2, rng_seed=12, step_count=130, output_hash="sha256:a", score="9/10", success=True),
        RunOutcomeWitness(run_id=3, rng_seed=13, step_count=140, output_hash="sha256:a", score="4/5", success=True),
        RunOutcomeWitness(run_id=4, rng_seed=14, step_count=155, output_hash="sha256:b", score="3/5", success=False),
        RunOutcomeWitness(run_id=5, rng_seed=15, step_count=200, output_hash="sha256:a", score="9/10", success=True),
    ]
    dec = IncoherenceDecompositionWitness(
        metric_id="brier",
        total_error="1/10",
        bias_component="1/20",
        variance_component="1/20",
        incoherence_ratio="1/2",
        notes="Demo-only arithmetic-consistent decomposition witness",
    )
    inv = CoherenceInvariant(
        metric_id="brier",
        max_incoherence_ratio="3/5",
        min_agreement_rate="3/4",
    )
    cert = HotMessIncoherenceCertificateV1(
        certificate_id=cert_id("hot-mess-demo-success"),
        timestamp=utc_now_iso(),
        model_id="demo_model",
        task_family="demo_task_family",
        eval_metric_id="brier",
        run_outcomes=outcomes,
        decomposition_witness=dec,
        coherence_invariant=inv,
        success=True,
        qa_interpretation={
            "success_type": "I_COH_PASSED",
            "note": "Agreement and incoherence below thresholds (demo).",
        },
    )
    return cert.to_dict()


def create_demo_hot_mess_failure_certificate() -> Dict[str, Any]:
    outcomes = [
        RunOutcomeWitness(run_id=1, rng_seed=21, step_count=200, output_hash="sha256:a", score="1/2", success=False),
        RunOutcomeWitness(run_id=2, rng_seed=22, step_count=210, output_hash="sha256:b", score="1/2", success=False),
        RunOutcomeWitness(run_id=3, rng_seed=23, step_count=220, output_hash="sha256:c", score="1/2", success=False),
        RunOutcomeWitness(run_id=4, rng_seed=24, step_count=230, output_hash="sha256:d", score="1/2", success=False),
        RunOutcomeWitness(run_id=5, rng_seed=25, step_count=240, output_hash="sha256:e", score="1/2", success=False),
    ]
    dec = IncoherenceDecompositionWitness(
        metric_id="brier",
        total_error="1/2",
        bias_component="1/10",
        variance_component="2/5",
        incoherence_ratio="4/5",
        notes="Demo-only: variance dominates; violates I_coh.",
    )
    inv = CoherenceInvariant(
        metric_id="brier",
        max_incoherence_ratio="3/5",
        min_agreement_rate="3/4",
    )
    cert = HotMessIncoherenceCertificateV1(
        certificate_id=cert_id("hot-mess-demo-failure"),
        timestamp=utc_now_iso(),
        model_id="demo_model",
        task_family="demo_task_family_hard",
        eval_metric_id="brier",
        run_outcomes=outcomes,
        decomposition_witness=dec,
        coherence_invariant=inv,
        success=False,
        failure_mode="HOT_MESS_INCOHERENCE",
        failure_witness={
            "agreement_rate": str(compute_agreement_rate(outcomes)),
            "incoherence_ratio": str(dec.incoherence_ratio),
            "note": "High run-to-run divergence in demo.",
        },
        qa_interpretation={
            "failure_type": "I_COH_VIOLATED",
            "note": "Variance-dominant hot-mess regime (demo).",
        },
    )
    return cert.to_dict()


if __name__ == "__main__":
    # Minimal module smoke: dump demo certs
    print(create_demo_hot_mess_success_certificate()["schema"])
    print(create_demo_hot_mess_failure_certificate()["schema"])

