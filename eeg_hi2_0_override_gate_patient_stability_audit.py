#!/usr/bin/env python3
"""
eeg_hi2_0_override_gate_patient_stability_audit.py
==================================================

Artifact-only patient-level stability audit for the EEG HI 2.0 override gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SWEEP = Path("results/eeg_hi2_0_override_gate_threshold_sweep.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_override_gate_patient_stability_audit.json")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def classify_patient(trigger_rate: float) -> str:
    if trigger_rate >= 1.0:
        return "robust_coordinate_override"
    if trigger_rate > 0.0:
        return "borderline_coordinate_override"
    return "stable_full_default"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit patient-level trigger stability from the saved override-gate threshold sweep.")
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sweep_doc = json.loads(args.sweep.read_text(encoding="utf-8"))
    results = sweep_doc["results"]
    n_pairs = len(results)
    patient_rows: dict[str, dict[str, object]] = {}

    for result in results:
        for patient in result["patients"]:
            patient_id = patient["patient_id"]
            row = patient_rows.setdefault(
                patient_id,
                {
                    "patient_id": patient_id,
                    "n_pairs": n_pairs,
                    "n_triggered": 0,
                    "n_positive_when_triggered": 0,
                    "selected_family_counts_when_triggered": {},
                    "full_f1": float(patient["full_f1"]),
                    "triggered_override_minus_full": [],
                    "triggered_override_gaps": [],
                },
            )
            if patient["override_triggered"]:
                row["n_triggered"] += 1
                if float(patient["override_minus_full"]) > 0.0:
                    row["n_positive_when_triggered"] += 1
                family = patient["selected_family"]
                family_counts = row["selected_family_counts_when_triggered"]
                family_counts[family] = int(family_counts.get(family, 0)) + 1
                row["triggered_override_minus_full"].append(float(patient["override_minus_full"]))
                row["triggered_override_gaps"].append(float(patient["override_gap"]))

    patients = []
    for patient_id in sorted(patient_rows):
        row = patient_rows[patient_id]
        n_triggered = int(row["n_triggered"])
        trigger_rate = float(n_triggered / n_pairs) if n_pairs else 0.0
        label = classify_patient(trigger_rate)
        deltas = row["triggered_override_minus_full"]
        gaps = row["triggered_override_gaps"]
        patients.append(
            {
                "patient_id": patient_id,
                "stability_label": label,
                "n_pairs": int(row["n_pairs"]),
                "n_triggered": n_triggered,
                "trigger_rate": trigger_rate,
                "n_positive_when_triggered": int(row["n_positive_when_triggered"]),
                "mean_override_minus_full_when_triggered": float(sum(deltas) / len(deltas)) if deltas else 0.0,
                "mean_override_gap_when_triggered": float(sum(gaps) / len(gaps)) if gaps else 0.0,
                "selected_family_counts_when_triggered": row["selected_family_counts_when_triggered"],
                "full_f1": float(row["full_f1"]),
            }
        )

    label_counts = {
        "robust_coordinate_override": int(sum(1 for patient in patients if patient["stability_label"] == "robust_coordinate_override")),
        "borderline_coordinate_override": int(sum(1 for patient in patients if patient["stability_label"] == "borderline_coordinate_override")),
        "stable_full_default": int(sum(1 for patient in patients if patient["stability_label"] == "stable_full_default")),
    }
    verdict = "PASS" if (
        label_counts["robust_coordinate_override"] >= 1
        and label_counts["borderline_coordinate_override"] >= 1
        and label_counts["stable_full_default"] >= 1
    ) else "FAIL"
    doc = {
        "experiment": {
            "id": "eeg_hi2_0_override_gate_patient_stability_audit_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_override_gate_patient_stability_audit.py",
            "hypothesis": "chb08 is a robust coordinate-dominant override across the full passing threshold region, chb01 is a borderline override that only triggers in a subset of the passing region, and the remaining patients are stable full defaults.",
            "success_criteria": "PASS if the audit separates at least one robust override patient, at least one borderline override patient, and a stable full-default cohort with zero trigger frequency; FAIL otherwise.",
            "artifact_inputs": [str(args.sweep)],
            "n_threshold_pairs": int(n_pairs),
        },
        "patients": patients,
        "aggregate": {
            "n_patients": int(len(patients)),
            "label_counts": label_counts,
            "robust_override_patients": [
                patient["patient_id"]
                for patient in patients
                if patient["stability_label"] == "robust_coordinate_override"
            ],
            "borderline_override_patients": [
                patient["patient_id"]
                for patient in patients
                if patient["stability_label"] == "borderline_coordinate_override"
            ],
            "stable_full_default_patients": [
                patient["patient_id"]
                for patient in patients
                if patient["stability_label"] == "stable_full_default"
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
