#!/usr/bin/env python3
"""
eeg_hi2_0_stability_threshold_policy_sweep.py
=============================================

Artifact-only sweep over minimum stability trigger-rate thresholds for allowing
coordinate-dominant EEG HI 2.0 overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OVERRIDE = Path("results/eeg_hi2_0_override_gated_classifier.json")
DEFAULT_STABILITY = Path("results/eeg_hi2_0_override_gate_patient_stability_audit.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_stability_threshold_policy_sweep.json")
THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_stability_lookup(stability_path: Path) -> dict[str, dict[str, object]]:
    source = json.loads(stability_path.read_text(encoding="utf-8"))
    return {
        patient["patient_id"]: patient
        for patient in source["patients"]
    }


def evaluate_threshold(
    override_doc: dict[str, object],
    stability_lookup: dict[str, dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    patients = []
    for patient in override_doc["patients"]:
        patient_id = patient["patient_id"]
        stability = stability_lookup.get(patient_id, {})
        trigger_rate = float(stability.get("trigger_rate", 0.0))
        anchor_triggered = bool(patient["decision"]["override_triggered"])
        threshold_triggered = bool(anchor_triggered and trigger_rate >= threshold)
        selected_family = patient["selected_family"] if threshold_triggered else "full"
        selected_f1 = float(patient["f1_override_gated"]) if threshold_triggered else float(patient["f1_full_hi2"])
        patients.append(
            {
                "patient_id": patient_id,
                "trigger_rate": trigger_rate,
                "stability_label": stability.get("stability_label", "stable_full_default"),
                "anchor_triggered": anchor_triggered,
                "threshold_triggered": threshold_triggered,
                "selected_family": selected_family,
                "f1_full_hi2": float(patient["f1_full_hi2"]),
                "f1_selected": selected_f1,
                "selected_minus_full": float(selected_f1 - patient["f1_full_hi2"]),
            }
        )

    mean_full = float(sum(patient["f1_full_hi2"] for patient in patients) / len(patients)) if patients else 0.0
    mean_selected = float(sum(patient["f1_selected"] for patient in patients) / len(patients)) if patients else 0.0
    selected_gain = float(mean_selected - mean_full)
    anchor_gain = float(override_doc["aggregate"]["mean_override_minus_full"])
    retention_ratio = float(selected_gain / anchor_gain) if anchor_gain > 0.0 else 0.0
    override_count = sum(1 for patient in patients if patient["threshold_triggered"])
    anchor_count = int(len(override_doc["aggregate"]["override_triggered_patients"]))
    if selected_gain > 0.0 and override_count < anchor_count and retention_ratio >= 0.8:
        verdict = "PASS"
    elif selected_gain > 0.0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return {
        "minimum_trigger_rate": threshold,
        "mean_selected_minus_full": selected_gain,
        "retention_ratio_vs_anchor": retention_ratio,
        "override_count": int(override_count),
        "anchor_override_count": int(anchor_count),
        "override_patients": [
            patient["patient_id"]
            for patient in patients
            if patient["threshold_triggered"]
        ],
        "verdict": verdict,
        "patients": patients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep minimum stability trigger-rate thresholds for EEG HI 2.0 override admission.")
    parser.add_argument("--override-artifact", type=Path, default=DEFAULT_OVERRIDE)
    parser.add_argument("--stability-audit", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    override_doc = json.loads(args.override_artifact.read_text(encoding="utf-8"))
    stability_lookup = load_stability_lookup(args.stability_audit)
    results = [
        evaluate_threshold(override_doc, stability_lookup, threshold)
        for threshold in THRESHOLDS
    ]
    passing = [result for result in results if result["verdict"] == "PASS"]
    doc = {
        "experiment": {
            "id": "eeg_hi2_0_stability_threshold_policy_sweep_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_stability_threshold_policy_sweep.py",
            "hypothesis": "There is a stable minimum-trigger-rate threshold region at or above 0.50 where mean gain over full remains positive, override count is below the anchor policy, and retained gain stays at least 80% of the anchor gain.",
            "success_criteria": "PASS if multiple thresholds satisfy that rule; PARTIAL if only one threshold does; FAIL otherwise.",
            "artifact_inputs": [str(args.override_artifact), str(args.stability_audit)],
            "thresholds": THRESHOLDS,
        },
        "results": results,
        "summary": {
            "n_thresholds": len(results),
            "n_pass": len(passing),
            "passing_thresholds": [
                {
                    "minimum_trigger_rate": result["minimum_trigger_rate"],
                    "mean_selected_minus_full": result["mean_selected_minus_full"],
                    "retention_ratio_vs_anchor": result["retention_ratio_vs_anchor"],
                    "override_patients": result["override_patients"],
                }
                for result in passing
            ],
            "verdict": "PASS" if len(passing) >= 2 else ("PARTIAL" if len(passing) == 1 else "FAIL"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print(json.dumps(doc["summary"], indent=2, sort_keys=True, ensure_ascii=False))
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
