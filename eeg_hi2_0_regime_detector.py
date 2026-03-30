#!/usr/bin/env python3
"""
eeg_hi2_0_regime_detector.py
============================

Reusable conservative regime detector for EEG HI 2.0 family routing.

This tool is deliberately artifact-first:
- it can read saved family-gated validation scores and emit override decisions
- it can also score a single patient decision from a provided score map
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ARTIFACT = Path("results/eeg_hi2_0_family_gated_classifier.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_regime_detector_decisions.json")
DEFAULT_STABILITY_AUDIT = Path("results/eeg_hi2_0_override_gate_patient_stability_audit.json")
OVERRIDE_FAMILIES = ["coords_only", "no_geometry"]
DEFAULT_OVERRIDE_MARGIN = 0.08
DEFAULT_FULL_WEAK_THRESHOLD = 0.55
DEFAULT_POLICY_MODE = "anchor"
DEFAULT_MIN_TRIGGER_RATE = 0.5


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def detect_regime(
    validation_scores: dict[str, float],
    override_margin: float,
    full_weak_threshold: float,
) -> dict[str, object]:
    full_validation = float(validation_scores["full"])
    best_override_family = max(
        OVERRIDE_FAMILIES,
        key=lambda family: float(validation_scores.get(family, float("-inf"))),
    )
    best_override_validation = float(validation_scores[best_override_family])
    override_gap = float(best_override_validation - full_validation)
    override_triggered = bool(
        full_validation <= full_weak_threshold and override_gap >= override_margin
    )
    selected_family = best_override_family if override_triggered else "full"
    regime = "coordinate_dominant_override" if override_triggered else "full_geometry_default"
    return {
        "selected_family": selected_family,
        "regime": regime,
        "decision": {
            "override_triggered": override_triggered,
            "full_validation_f1": full_validation,
            "best_override_family": best_override_family,
            "best_override_validation_f1": best_override_validation,
            "override_gap": override_gap,
            "override_margin": float(override_margin),
            "full_weak_threshold": float(full_weak_threshold),
        },
    }


def apply_policy_mode(
    detected: dict[str, object],
    stability: dict[str, object] | None,
    policy_mode: str,
    min_trigger_rate: float,
) -> dict[str, object]:
    if policy_mode == "anchor":
        return {
            "selected_family": detected["selected_family"],
            "regime": detected["regime"],
            "policy": {
                "policy_mode": policy_mode,
                "policy_override_triggered": bool(detected["decision"]["override_triggered"]),
                "min_trigger_rate": None,
            },
        }

    trigger_rate = 0.0 if stability is None else float(stability.get("trigger_rate", 0.0))
    effective_min_trigger_rate = 1.0 if policy_mode == "robust_only" else float(min_trigger_rate)
    threshold_allowed = trigger_rate >= effective_min_trigger_rate
    if bool(detected["decision"]["override_triggered"]) and threshold_allowed:
        return {
            "selected_family": detected["selected_family"],
            "regime": detected["regime"],
            "policy": {
                "policy_mode": policy_mode,
                "policy_override_triggered": True,
                "min_trigger_rate": effective_min_trigger_rate,
            },
        }
    return {
        "selected_family": "full",
        "regime": "full_geometry_default",
        "policy": {
            "policy_mode": policy_mode,
            "policy_override_triggered": False,
            "min_trigger_rate": effective_min_trigger_rate,
        },
    }


def load_stability_lookup(stability_audit_path: Path | None) -> dict[str, dict[str, object]]:
    if stability_audit_path is None or not stability_audit_path.exists():
        return {}
    source = json.loads(stability_audit_path.read_text(encoding="utf-8"))
    return {
        patient["patient_id"]: patient
        for patient in source.get("patients", [])
    }


def from_artifact(
    artifact_path: Path,
    override_margin: float,
    full_weak_threshold: float,
    patient_filter: set[str] | None,
    stability_audit_path: Path | None,
    policy_mode: str,
    min_trigger_rate: float,
) -> dict[str, object]:
    source = json.loads(artifact_path.read_text(encoding="utf-8"))
    stability_lookup = load_stability_lookup(stability_audit_path)
    decisions = []
    for patient in source["patients"]:
        patient_id = patient["patient_id"]
        if patient_filter and patient_id not in patient_filter:
            continue
        detected = detect_regime(
            patient["inner_validation_f1"],
            override_margin=override_margin,
            full_weak_threshold=full_weak_threshold,
        )
        stability = stability_lookup.get(patient_id)
        policy_applied = apply_policy_mode(
            detected,
            stability=stability,
            policy_mode=policy_mode,
            min_trigger_rate=min_trigger_rate,
        )
        decisions.append(
            {
                "patient_id": patient_id,
                "n_segments_balanced": patient.get("n_segments_balanced"),
                "f1_hi1": patient.get("f1_hi1"),
                "f1_full_hi2": patient.get("f1_full_hi2"),
                "f1_selected_if_available": patient.get("f1_gated"),
                "validation_scores": patient["inner_validation_f1"],
                "stability": stability,
                "anchor_decision": detected,
                **policy_applied,
            }
        )

    aggregate = {
        "n_patients": int(len(decisions)),
        "override_trigger_count": int(sum(1 for item in decisions if item["policy"]["policy_override_triggered"])),
        "selected_family_counts": {
            "full": int(sum(1 for item in decisions if item["selected_family"] == "full")),
            "coords_only": int(sum(1 for item in decisions if item["selected_family"] == "coords_only")),
            "no_geometry": int(sum(1 for item in decisions if item["selected_family"] == "no_geometry")),
        },
        "override_patients": [
            item["patient_id"]
            for item in decisions
            if item["policy"]["policy_override_triggered"]
        ],
    }
    return {
        "detector": {
            "id": "eeg_hi2_0_regime_detector_2026-03-29",
            "artifact_source": str(artifact_path),
            "stability_audit_source": str(stability_audit_path) if stability_audit_path is not None and stability_audit_path.exists() else None,
            "override_families": OVERRIDE_FAMILIES,
            "override_margin": float(override_margin),
            "full_weak_threshold": float(full_weak_threshold),
            "policy_mode": policy_mode,
            "min_trigger_rate": None if policy_mode == "anchor" else (1.0 if policy_mode == "robust_only" else float(min_trigger_rate)),
        },
        "patients": decisions,
        "aggregate": aggregate,
    }


def from_scores(
    scores: dict[str, float],
    override_margin: float,
    full_weak_threshold: float,
    policy_mode: str,
    min_trigger_rate: float,
) -> dict[str, object]:
    return {
        "detector": {
            "id": "eeg_hi2_0_regime_detector_2026-03-29",
            "override_families": OVERRIDE_FAMILIES,
            "override_margin": float(override_margin),
            "full_weak_threshold": float(full_weak_threshold),
            "policy_mode": policy_mode,
            "min_trigger_rate": None if policy_mode == "anchor" else (1.0 if policy_mode == "robust_only" else float(min_trigger_rate)),
            "policy_note": None if policy_mode == "anchor" else "single-score mode lacks stability labels; returning the raw anchor detector decision",
        },
        "decision": detect_regime(
            scores,
            override_margin=override_margin,
            full_weak_threshold=full_weak_threshold,
        ),
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative regime detector for EEG HI 2.0 family routing.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Saved family-gated artifact with inner validation scores")
    parser.add_argument("--scores", help="Optional JSON object with validation scores for one decision")
    parser.add_argument("--patient", nargs="*", help="Optional patient ids to filter when reading an artifact")
    parser.add_argument("--stability-audit", type=Path, default=DEFAULT_STABILITY_AUDIT, help="Optional patient stability audit artifact")
    parser.add_argument("--policy-mode", choices=["anchor", "robust_only", "stability_threshold"], default=DEFAULT_POLICY_MODE, help="Decision policy applied after the raw anchor detector")
    parser.add_argument("--min-trigger-rate", type=float, default=DEFAULT_MIN_TRIGGER_RATE, help="Minimum stability trigger rate required when policy_mode=stability_threshold")
    parser.add_argument("--override-margin", type=float, default=DEFAULT_OVERRIDE_MARGIN)
    parser.add_argument("--full-weak-threshold", type=float, default=DEFAULT_FULL_WEAK_THRESHOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.scores:
        score_obj = json.loads(args.scores)
        doc = from_scores(
            score_obj,
            override_margin=args.override_margin,
            full_weak_threshold=args.full_weak_threshold,
            policy_mode=args.policy_mode,
            min_trigger_rate=args.min_trigger_rate,
        )
    else:
        patient_filter = set(args.patient) if args.patient else None
        doc = from_artifact(
            args.artifact,
            override_margin=args.override_margin,
            full_weak_threshold=args.full_weak_threshold,
            patient_filter=patient_filter,
            stability_audit_path=args.stability_audit,
            policy_mode=args.policy_mode,
            min_trigger_rate=args.min_trigger_rate,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
