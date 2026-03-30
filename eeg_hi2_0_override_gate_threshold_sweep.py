#!/usr/bin/env python3
"""
eeg_hi2_0_override_gate_threshold_sweep.py
==========================================

Artifact-only threshold sweep for the conservative EEG HI 2.0 override gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ABLATION = Path("results/eeg_hi2_0_feature_ablation_audit.json")
DEFAULT_GATE = Path("results/eeg_hi2_0_family_gated_classifier.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_override_gate_threshold_sweep.json")
OVERRIDE_FAMILIES = ["coords_only", "no_geometry"]
MARGINS = [0.00, 0.04, 0.08, 0.12, 0.16]
WEAK_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_ablation_lookup(ablation_doc: dict[str, object]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for cohort in ["stable_positive", "hard_negative"]:
        for patient in ablation_doc["patients"][cohort]:
            lookup[patient["patient_id"]] = patient
    return lookup


def family_f1(gated_patient: dict[str, object], ablation_patient: dict[str, object] | None, family: str) -> float:
    if ablation_patient is not None:
        return float(ablation_patient["feature_sets"][family]["f1"])
    if family == "full":
        return float(gated_patient["f1_full_hi2"])
    if family == gated_patient["selected_family"]:
        return float(gated_patient["f1_gated"])
    raise KeyError(f"Missing outer-test F1 for {gated_patient['patient_id']} family {family}")


def evaluate_pair(
    gated_doc: dict[str, object],
    ablation_lookup: dict[str, dict[str, object]],
    margin: float,
    weak_threshold: float,
) -> dict[str, object]:
    patients = []
    override_minus_full_values = []
    override_patients = []
    hard_negative_reroutes = []
    for gated_patient in gated_doc["patients"]:
        patient_id = gated_patient["patient_id"]
        validation = gated_patient["inner_validation_f1"]
        full_validation = float(validation["full"])
        best_override = max(OVERRIDE_FAMILIES, key=lambda family: float(validation[family]))
        best_override_score = float(validation[best_override])
        gap = float(best_override_score - full_validation)
        triggered = bool(full_validation <= weak_threshold and gap >= margin)
        selected_family = best_override if triggered else "full"
        selected_f1 = family_f1(gated_patient, ablation_lookup.get(patient_id), selected_family)
        full_f1 = float(gated_patient["f1_full_hi2"])
        delta = float(selected_f1 - full_f1)
        if triggered:
            override_patients.append(patient_id)
        if patient_id in {"chb02", "chb08"} and triggered and delta > 0.0:
            hard_negative_reroutes.append(patient_id)
        override_minus_full_values.append(delta)
        patients.append(
            {
                "patient_id": patient_id,
                "selected_family": selected_family,
                "override_triggered": triggered,
                "override_gap": gap,
                "full_validation_f1": full_validation,
                "selected_f1": selected_f1,
                "full_f1": full_f1,
                "override_minus_full": delta,
            }
        )

    mean_delta = float(sum(override_minus_full_values) / len(override_minus_full_values))
    nonnegative_count = sum(1 for value in override_minus_full_values if value >= 0.0)
    share_nonnegative = nonnegative_count / len(override_minus_full_values)
    if mean_delta > 0.0 and share_nonnegative >= 0.8 and hard_negative_reroutes:
        verdict = "PASS"
    elif mean_delta > 0.0 or hard_negative_reroutes:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return {
        "override_margin": margin,
        "full_weak_threshold": weak_threshold,
        "mean_override_minus_full": mean_delta,
        "count_nonnegative": int(nonnegative_count),
        "share_nonnegative": float(share_nonnegative),
        "override_patients": override_patients,
        "hard_negative_reroutes": hard_negative_reroutes,
        "verdict": verdict,
        "patients": patients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep conservative override-gate thresholds using saved EEG artifacts.")
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ablation_doc = json.loads(args.ablation.read_text(encoding="utf-8"))
    gated_doc = json.loads(args.gate.read_text(encoding="utf-8"))
    ablation_lookup = build_ablation_lookup(ablation_doc)

    results = [
        evaluate_pair(gated_doc, ablation_lookup, margin, threshold)
        for margin in MARGINS
        for threshold in WEAK_THRESHOLDS
    ]
    passing = [
        result for result in results
        if result["verdict"] == "PASS"
    ]
    anchor = next(
        result for result in results
        if result["override_margin"] == 0.08 and result["full_weak_threshold"] == 0.55
    )
    doc = {
        "experiment": {
            "id": "eeg_hi2_0_override_gate_threshold_sweep_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_override_gate_threshold_sweep.py",
            "hypothesis": "The passing override-gate rule around margin=0.08 and full_weak_threshold=0.55 lies in a stable threshold region rather than being a single fragile point.",
            "success_criteria": "PASS if multiple nearby threshold pairs preserve positive mean override-minus-full F1 and a successful hard-negative reroute; PARTIAL if only a narrow subset works; FAIL otherwise.",
            "artifact_inputs": [str(args.ablation), str(args.gate)],
            "margins": MARGINS,
            "weak_thresholds": WEAK_THRESHOLDS,
        },
        "results": results,
        "summary": {
            "n_pairs": len(results),
            "n_pass": len(passing),
            "anchor_pair": anchor,
            "passing_pairs": [
                {
                    "override_margin": result["override_margin"],
                    "full_weak_threshold": result["full_weak_threshold"],
                    "mean_override_minus_full": result["mean_override_minus_full"],
                    "override_patients": result["override_patients"],
                }
                for result in passing
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print(json.dumps(doc["summary"], indent=2, sort_keys=True, ensure_ascii=False))
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
