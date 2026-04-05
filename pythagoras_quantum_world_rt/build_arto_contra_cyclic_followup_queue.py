#!/usr/bin/env python3
"""Build the local follow-up queue beneath the Arto contra-cyclic lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "arto_contra_cyclic_followup_queue.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def build_payload(lane_payload: dict[str, object]) -> dict[str, object]:
    return {
        "queue_id": "arto_contra_cyclic_followup_queue",
        "source_lane_id": lane_payload["lane_id"],
        "items": [
            {
                "actionability": "formula_audit",
                "confidence": 0.83,
                "id": "archimedean_twin_circle_radius_note",
                "note_status": "unvalidated_local_prior_work",
                "priority": "high",
                "proposed_claims": [
                    "r = d*d - d*sqrt(F) for the Archimedean twin circle",
                    "quadrance reading r^2 = D^2",
                    "scaled tuple reading ((b,e,d,a)*d)^2 = (J,X,D,A)^2",
                ],
                "risk_flags": [
                    "The local note mixes source text with later AI interpretation.",
                    "It uses speculative algebra rather than a recovered source construction.",
                    "The worked example for (1,2,3,5) produces 2.292 for r while later extrapolation claims 81 at the quadrance layer, so the note is internally unstable until definitions are fixed.",
                    "Any future code based on this note must rewrite power operations using d*d rather than d**2.",
                ],
                "source_paths": [
                    "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:14",
                    "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
                ],
                "task": "Recover precise variable meanings for r, D, J, X, A, and F before promoting any Archimedean twin-circle radius law.",
            },
            {
                "actionability": "mechanics_reproduction",
                "confidence": 0.88,
                "id": "sixto_ramos_device_mapping_note",
                "note_status": "unvalidated_local_prior_work",
                "priority": "high",
                "proposed_claims": [
                    "Treat the Sixto Ramos machine as two QA stages in series.",
                    "Use an ellipse-on-torus parameterization for driver/driven geometry.",
                    "Use a mod-24 event scheduler and non-overlapping impulse windows.",
                ],
                "risk_flags": [
                    "This note is downstream QA mapping work, not source witness material.",
                    "It uses trig and floating-angle parameterizations where the project prefers rationalized or integer-certified constructions.",
                    "The note proposes a scheduler and velocity proxy, but none of that is validated against an extracted machine diagram yet.",
                ],
                "source_paths": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:22",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:52",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:92",
                ],
                "task": "Use this only as a hypothesis generator while reconstructing the Sixto Ramos geometry from source images or diagram text.",
            },
            {
                "actionability": "device_family_queue",
                "confidence": 0.81,
                "id": "ucros_constantinesco_followups",
                "note_status": "unvalidated_local_prior_work",
                "priority": "medium",
                "proposed_claims": [
                    "Ucros can be modeled as a sub-harmonic generator side relative to the motor.",
                    "Constantinesco can be treated as an idle/lock phase integrator.",
                    "Hyperbolic impulse drive can be modeled by a tuple ramp across an impulse sector.",
                ],
                "risk_flags": [
                    "These are only model proposals from the later QA mapping note.",
                    "No corresponding source diagrams have been extracted in this subproject yet.",
                    "The queue order should remain Sixto first, Ucros second, Constantinesco third.",
                ],
                "source_paths": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:98",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:105",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:118",
                ],
                "task": "Leave these in the queue until one fully source-recovered machine packet exists.",
            },
        ],
        "next_step": {
            "id": "sixto_ramos_geometry_packet",
            "reason": "It is the only concrete machine hook named directly in the Arto contra-cyclic source and also the strongest target in later local mapping notes.",
            "task": "Build a source-graded Sixto Ramos packet that separates actual witness geometry from later QA/RT reinterpretation attempts.",
        },
        "summary": {
            "high_priority_count": 2,
            "item_count": 3,
            "series": "Pyth external prior art",
        },
    }


def self_test() -> int:
    lane_payload = {"lane_id": "arto_contra_cyclic_lane"}
    payload = build_payload(lane_payload)
    ok = (
        payload["queue_id"] == "arto_contra_cyclic_followup_queue"
        and payload["summary"]["item_count"] == 3
        and payload["summary"]["high_priority_count"] == 2
        and payload["next_step"]["id"] == "sixto_ramos_geometry_packet"
        and payload["items"][0]["note_status"] == "unvalidated_local_prior_work"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    lane_payload = read_json(OUT_DIR / "arto_contra_cyclic_lane.json")
    payload = build_payload(lane_payload)
    write_json(OUT_PATH, payload)
    print(canonical_dump({"ok": True, "path": str(OUT_PATH.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
