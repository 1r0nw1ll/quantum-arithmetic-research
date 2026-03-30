#!/usr/bin/env python3
"""
eeg_hi2_0_robust_only_override_policy_audit.py
==============================================

Artifact-only audit for a stability-aware robust-only EEG HI 2.0 override policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OVERRIDE = Path("results/eeg_hi2_0_override_gated_classifier.json")
DEFAULT_STABILITY = Path("results/eeg_hi2_0_override_gate_patient_stability_audit.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_robust_only_override_policy_audit.json")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_stability_lookup(stability_path: Path) -> dict[str, dict[str, object]]:
    source = json.loads(stability_path.read_text(encoding="utf-8"))
    return {
        patient["patient_id"]: patient
        for patient in source["patients"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a stability-aware robust-only override policy from saved EEG artifacts.")
    parser.add_argument("--override-artifact", type=Path, default=DEFAULT_OVERRIDE)
    parser.add_argument("--stability-audit", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    override_doc = json.loads(args.override_artifact.read_text(encoding="utf-8"))
    stability_lookup = load_stability_lookup(args.stability_audit)

    patients = []
    for patient in override_doc["patients"]:
        patient_id = patient["patient_id"]
        stability = stability_lookup.get(patient_id, {})
        stability_label = stability.get("stability_label", "stable_full_default")
        robust_triggered = bool(
            patient["decision"]["override_triggered"]
            and stability_label == "robust_coordinate_override"
        )
        robust_f1 = float(patient["f1_override_gated"]) if robust_triggered else float(patient["f1_full_hi2"])
        patients.append(
            {
                "patient_id": patient_id,
                "stability_label": stability_label,
                "f1_hi1": float(patient["f1_hi1"]),
                "f1_full_hi2": float(patient["f1_full_hi2"]),
                "f1_anchor_override": float(patient["f1_override_gated"]),
                "f1_robust_only": robust_f1,
                "anchor_override_triggered": bool(patient["decision"]["override_triggered"]),
                "robust_only_triggered": robust_triggered,
                "anchor_selected_family": patient["selected_family"],
                "robust_selected_family": patient["selected_family"] if robust_triggered else "full",
                "anchor_minus_full": float(patient["f1_override_gated"] - patient["f1_full_hi2"]),
                "robust_minus_full": float(robust_f1 - patient["f1_full_hi2"]),
            }
        )

    mean_full = float(sum(patient["f1_full_hi2"] for patient in patients) / len(patients)) if patients else 0.0
    mean_anchor = float(sum(patient["f1_anchor_override"] for patient in patients) / len(patients)) if patients else 0.0
    mean_robust = float(sum(patient["f1_robust_only"] for patient in patients) / len(patients)) if patients else 0.0
    anchor_gain = float(mean_anchor - mean_full)
    robust_gain = float(mean_robust - mean_full)
    retention_ratio = float(robust_gain / anchor_gain) if anchor_gain > 0.0 else 0.0
    anchor_count = sum(1 for patient in patients if patient["anchor_override_triggered"])
    robust_count = sum(1 for patient in patients if patient["robust_only_triggered"])
    if robust_gain > 0.0 and robust_count < anchor_count and retention_ratio >= 0.8:
        verdict = "PASS"
    elif robust_gain > 0.0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    doc = {
        "experiment": {
            "id": "eeg_hi2_0_robust_only_override_policy_audit_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_robust_only_override_policy_audit.py",
            "hypothesis": "A stability-aware robust_only policy retains at least 80% of the anchor override gain while using fewer overrides than the anchor policy.",
            "success_criteria": "PASS if robust_only mean gain over full is positive, uses fewer overrides, and retains at least 80% of the anchor mean gain; PARTIAL if positive but below 80%; FAIL otherwise.",
            "artifact_inputs": [str(args.override_artifact), str(args.stability_audit)],
        },
        "patients": patients,
        "aggregate": {
            "n_patients": int(len(patients)),
            "mean_f1_full_hi2": mean_full,
            "mean_f1_anchor_override": mean_anchor,
            "mean_f1_robust_only": mean_robust,
            "mean_anchor_minus_full": anchor_gain,
            "mean_robust_minus_full": robust_gain,
            "robust_gain_retention_ratio": retention_ratio,
            "anchor_override_count": int(anchor_count),
            "robust_only_override_count": int(robust_count),
            "robust_override_patients": [
                patient["patient_id"]
                for patient in patients
                if patient["robust_only_triggered"]
            ],
            "borderline_patients_left_on_full": [
                patient["patient_id"]
                for patient in patients
                if patient["stability_label"] == "borderline_coordinate_override"
                and not patient["robust_only_triggered"]
            ],
            "verdict": verdict,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print(json.dumps(doc["aggregate"], indent=2, sort_keys=True, ensure_ascii=False))
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
