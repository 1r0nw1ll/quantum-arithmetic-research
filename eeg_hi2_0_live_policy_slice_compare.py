#!/usr/bin/env python3
"""
eeg_hi2_0_live_policy_slice_compare.py
======================================

Compare two live EEG HI 2.0 scale-run artifacts produced under different regime
policies on the same patient slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ANCHOR = Path("results/eeg_hi2_0_live_slice_anchor.json")
DEFAULT_THRESHOLD = Path("results/eeg_hi2_0_live_slice_threshold_0p5.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_live_policy_slice_compare.json")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def patient_lookup(doc: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        patient["patient_id"]: patient
        for patient in doc["patients"]
        if not patient["skipped"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two live EEG policy-run artifacts on the same patient slice.")
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-only-reroute", nargs="*", default=[], help="Patients expected to reroute only under anchor")
    parser.add_argument("--both-reroute", nargs="*", default=[], help="Patients expected to reroute under both policies")
    parser.add_argument("--both-full", nargs="*", default=[], help="Patients expected to stay on full under both policies")
    args = parser.parse_args()

    anchor_doc = json.loads(args.anchor.read_text(encoding="utf-8"))
    threshold_doc = json.loads(args.threshold.read_text(encoding="utf-8"))
    anchor_patients = patient_lookup(anchor_doc)
    threshold_patients = patient_lookup(threshold_doc)
    shared_ids = sorted(set(anchor_patients) & set(threshold_patients))

    patients = []
    for patient_id in shared_ids:
        anchor = anchor_patients[patient_id]
        threshold = threshold_patients[patient_id]
        patients.append(
            {
                "patient_id": patient_id,
                "anchor_selected_family": anchor["regime_detector"]["selected_family"],
                "threshold_selected_family": threshold["regime_detector"]["selected_family"],
                "anchor_override_triggered": anchor["regime_detector"]["policy_override_triggered"],
                "threshold_override_triggered": threshold["regime_detector"]["policy_override_triggered"],
                "anchor_f1": float(anchor["HI_2.0_override_gated"]["f1"]),
                "threshold_f1": float(threshold["HI_2.0_override_gated"]["f1"]),
                "threshold_minus_anchor": float(threshold["HI_2.0_override_gated"]["f1"] - anchor["HI_2.0_override_gated"]["f1"]),
            }
        )

    patient_rows = {patient["patient_id"]: patient for patient in patients}
    anchor_gate = anchor_doc["aggregate"]["override_gate"]
    threshold_gate = threshold_doc["aggregate"]["override_gate"]
    threshold_positive = float(threshold_gate["mean_f1_delta_vs_full"]) > 0.0
    override_reduction = int(len(threshold_gate["override_triggered_patients"])) < int(len(anchor_gate["override_triggered_patients"]))
    expected_pattern = True
    for patient_id in args.anchor_only_reroute:
        row = patient_rows[patient_id]
        expected_pattern = expected_pattern and bool(row["anchor_override_triggered"]) and (not bool(row["threshold_override_triggered"]))
    for patient_id in args.both_reroute:
        row = patient_rows[patient_id]
        expected_pattern = expected_pattern and bool(row["anchor_override_triggered"]) and bool(row["threshold_override_triggered"])
    for patient_id in args.both_full:
        row = patient_rows[patient_id]
        expected_pattern = expected_pattern and (not bool(row["anchor_override_triggered"])) and (not bool(row["threshold_override_triggered"]))
    verdict = "PASS" if expected_pattern and threshold_positive and override_reduction else ("PARTIAL" if expected_pattern or threshold_positive else "FAIL")

    doc = {
        "experiment": {
            "id": "eeg_hi2_0_live_policy_slice_compare_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_live_policy_slice_compare.py",
            "hypothesis": "On a live three-patient slice spanning borderline, stable-full, and robust-override regimes, threshold-0.5 should reroute only the robust case while preserving positive mean gate gain and reducing override count versus anchor.",
            "success_criteria": "PASS if anchor reroutes chb01 and chb08, threshold-0.5 reroutes only chb08, chb03 stays on full under both, and threshold-0.5 keeps positive mean gate gain while reducing override count.",
            "artifact_inputs": [str(args.anchor), str(args.threshold)],
            "anchor_only_reroute": args.anchor_only_reroute,
            "both_reroute": args.both_reroute,
            "both_full": args.both_full,
        },
        "patients": patients,
        "aggregate": {
            "anchor_override_triggered_patients": anchor_gate["override_triggered_patients"],
            "threshold_override_triggered_patients": threshold_gate["override_triggered_patients"],
            "anchor_mean_f1_delta_vs_full": float(anchor_gate["mean_f1_delta_vs_full"]),
            "threshold_mean_f1_delta_vs_full": float(threshold_gate["mean_f1_delta_vs_full"]),
            "override_reduction": override_reduction,
            "expected_pattern": expected_pattern,
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
