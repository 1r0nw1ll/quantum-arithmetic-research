#!/usr/bin/env python3
"""
eeg_hi2_0_override_gate_from_artifacts.py
=========================================

Finish the conservative override-gate experiment from saved artifacts rather than
rerunning raw EDF extraction on a low-resource host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ABLATION = Path("results/eeg_hi2_0_feature_ablation_audit.json")
DEFAULT_GATE = Path("results/eeg_hi2_0_family_gated_classifier.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_override_gated_classifier.json")
OVERRIDE_MARGIN = 0.08
FULL_WEAK_THRESHOLD = 0.55
OVERRIDE_FAMILIES = ["coords_only", "no_geometry"]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_patient_lookup(ablation_doc: dict[str, object]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for cohort in ["stable_positive", "hard_negative"]:
        for patient in ablation_doc["patients"][cohort]:
            out[patient["patient_id"]] = patient
    return out


def selected_outer_f1(
    gated_patient: dict[str, object],
    ablation_patient: dict[str, object] | None,
    selected_family: str,
) -> float:
    if ablation_patient is not None:
        return float(ablation_patient["feature_sets"][selected_family]["f1"])
    if selected_family == "full":
        return float(gated_patient["f1_full_hi2"])
    if selected_family == gated_patient["selected_family"]:
        return float(gated_patient["f1_gated"])
    raise KeyError(f"Missing outer-test F1 for {gated_patient['patient_id']} family {selected_family}")


def override_patient(
    gated_patient: dict[str, object],
    ablation_patient: dict[str, object] | None,
) -> dict[str, object]:
    validation = gated_patient["inner_validation_f1"]
    full_validation = float(validation["full"])
    best_override = max(OVERRIDE_FAMILIES, key=lambda family: float(validation[family]))
    best_override_validation = float(validation[best_override])
    override_gap = float(best_override_validation - full_validation)
    triggered = bool(full_validation <= FULL_WEAK_THRESHOLD and override_gap >= OVERRIDE_MARGIN)
    selected_family = best_override if triggered else "full"
    selected_f1 = selected_outer_f1(gated_patient, ablation_patient, selected_family)
    full_f1 = float(ablation_patient["feature_sets"]["full"]["f1"]) if ablation_patient is not None else float(gated_patient["f1_full_hi2"])
    hi1_f1 = float(ablation_patient["f1_hi1"]) if ablation_patient is not None else float(gated_patient["f1_hi1"])
    return {
        "patient_id": gated_patient["patient_id"],
        "f1_hi1": hi1_f1,
        "f1_full_hi2": full_f1,
        "selected_family": selected_family,
        "f1_override_gated": selected_f1,
        "decision": {
            "override_triggered": triggered,
            "full_validation_f1": full_validation,
            "best_override_family": best_override,
            "best_override_validation_f1": best_override_validation,
            "override_gap": override_gap,
            "override_margin": OVERRIDE_MARGIN,
            "full_weak_threshold": FULL_WEAK_THRESHOLD,
        },
        "deltas": {
            "override_minus_full": float(selected_f1 - full_f1),
            "override_minus_hi1": float(selected_f1 - hi1_f1),
            "full_minus_hi1": float(full_f1 - hi1_f1),
        },
    }


def aggregate(patients: list[dict[str, object]]) -> dict[str, object]:
    override_minus_full = [patient["deltas"]["override_minus_full"] for patient in patients]
    nonnegative_count = sum(1 for value in override_minus_full if value >= 0.0)
    share_nonnegative = nonnegative_count / len(patients) if patients else 0.0
    family_counts = {"full": 0, "coords_only": 0, "no_geometry": 0}
    overrides = []
    for patient in patients:
        family_counts[patient["selected_family"]] += 1
        if patient["decision"]["override_triggered"]:
            overrides.append(patient["patient_id"])
    hard_negative_reroutes = [
        patient["patient_id"]
        for patient in patients
        if patient["patient_id"] in {"chb02", "chb08"}
        and patient["decision"]["override_triggered"]
        and patient["deltas"]["override_minus_full"] > 0.0
    ]
    if override_minus_full and float(sum(override_minus_full) / len(override_minus_full)) > 0.0 and share_nonnegative >= 0.8 and hard_negative_reroutes:
        verdict = "PASS"
        verdict_reason = "Artifact-reconstructed override gate improved mean F1 over full HI 2.0, kept most patients non-negative, and rerouted at least one hard-negative patient successfully."
    elif override_minus_full and (float(sum(override_minus_full) / len(override_minus_full)) > 0.0 or hard_negative_reroutes):
        verdict = "PARTIAL"
        verdict_reason = "Override gate showed some benefit, but not enough for a strong aggregate claim."
    else:
        verdict = "FAIL"
        verdict_reason = "Override gate did not improve over full HI 2.0."
    return {
        "n_patients": len(patients),
        "mean_f1_hi1": float(sum(patient["f1_hi1"] for patient in patients) / len(patients)) if patients else 0.0,
        "mean_f1_full_hi2": float(sum(patient["f1_full_hi2"] for patient in patients) / len(patients)) if patients else 0.0,
        "mean_f1_override_gated": float(sum(patient["f1_override_gated"] for patient in patients) / len(patients)) if patients else 0.0,
        "mean_override_minus_full": float(sum(override_minus_full) / len(override_minus_full)) if override_minus_full else 0.0,
        "count_nonnegative_override_minus_full": nonnegative_count,
        "share_nonnegative_override_minus_full": float(share_nonnegative),
        "selected_family_counts": family_counts,
        "override_triggered_patients": overrides,
        "hard_negative_reroutes": hard_negative_reroutes,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def print_summary(doc: dict[str, object]) -> None:
    print("=" * 80)
    print("EEG HI 2.0 OVERRIDE GATE FROM ARTIFACTS")
    print("=" * 80)
    print(f"{'Patient':<8} {'Family':<12} {'Trig':<5} {'Full':>7} {'Gate':>7} {'O-F':>7}")
    print(f"{'-'*8} {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    for patient in doc["patients"]:
        print(
            f"{patient['patient_id']:<8} {patient['selected_family']:<12} "
            f"{str(patient['decision']['override_triggered']):<5} "
            f"{patient['f1_full_hi2']:>7.3f} {patient['f1_override_gated']:>7.3f} "
            f"{patient['deltas']['override_minus_full']:>+7.3f}"
        )
    aggregate_doc = doc["aggregate"]
    print()
    print(f"Mean full HI2 F1:  {aggregate_doc['mean_f1_full_hi2']:.4f}")
    print(f"Mean override F1:  {aggregate_doc['mean_f1_override_gated']:.4f}")
    print(f"Mean override-full:{aggregate_doc['mean_override_minus_full']:+.4f}")
    print(f"Triggered overrides: {aggregate_doc['override_triggered_patients']}")
    print(f"Hard-negative reroutes: {aggregate_doc['hard_negative_reroutes']}")
    print(f"Verdict: {aggregate_doc['verdict']}")
    print(f"Reason:  {aggregate_doc['verdict_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish the override-gate EEG experiment from saved artifacts.")
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ablation_doc = json.loads(args.ablation.read_text(encoding="utf-8"))
    gate_doc = json.loads(args.gate.read_text(encoding="utf-8"))
    ablation_lookup = build_patient_lookup(ablation_doc)

    patients = [
        override_patient(gated_patient, ablation_lookup.get(gated_patient["patient_id"]))
        for gated_patient in gate_doc["patients"]
    ]
    doc = {
        "experiment": {
            "id": "eeg_hi2_0_override_gated_classifier_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_override_gate_from_artifacts.py",
            "hypothesis": "A conservative override gate that defaults to full HI 2.0 and reroutes only when a coordinate-dominant family wins validation by margin under weak-full conditions improves aggregate F1 over fixed full HI 2.0.",
            "success_criteria": "PASS if mean override-minus-full F1 is positive, at least 80% of patients are non-negative, and at least one hard-negative patient is rerouted with improvement; PARTIAL if mixed; FAIL otherwise.",
            "artifact_inputs": [str(args.ablation), str(args.gate)],
            "override_families": OVERRIDE_FAMILIES,
            "override_margin": OVERRIDE_MARGIN,
            "full_weak_threshold": FULL_WEAK_THRESHOLD,
            "resource_profile": "local_resource_profile.json",
            "notes": [
                "This result is reconstructed from saved validation and outer-test artifacts to avoid rerunning raw EDF extraction on the low-resource host."
            ]
        },
        "patients": patients,
        "aggregate": aggregate(patients),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print_summary(doc)
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
