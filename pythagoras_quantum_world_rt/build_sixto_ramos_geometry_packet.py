#!/usr/bin/env python3
"""Build a source-graded Sixto Ramos geometry packet from the Arto contra-cyclic lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "sixto_ramos_geometry_packet.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def build_payload(
    intake_payload: dict[str, object],
    lane_payload: dict[str, object],
    followup_payload: dict[str, object],
) -> dict[str, object]:
    by_id = {entry["id"]: entry for entry in intake_payload["sources"]}
    contra_source = by_id["arto_contra_cyclic_post_live"]
    source_geometry_items = [
        {
            "confidence": 0.97,
            "formal_claim": "The primary Sixto witness in the Arto source is a relation claim: the Sixto Ramos machine is said to reveal the geometry clearly and to relate to Archimedean twin-circle (Fibonacci/golden) geometry, toroidal geometry, and part of the conchoid of de Sluze family.",
            "id": "sixto_relation_claim",
            "source_refs": [
                {
                    "kind": "live_web",
                    "line": 17,
                    "url": contra_source["url"],
                },
                {
                    "kind": "local_second_witness",
                    "path": "private/QAnotes/Nexus AI Chat Imports/2024/11/Triangle Ratios and Relations.md",
                    "source_line": 2220,
                },
            ],
            "validation_status": "double_witnessed_source_claim",
        },
        {
            "confidence": 0.96,
            "formal_claim": "The source provides a concrete Sixto asset set embedded in the article. The named image witnesses are part of the source geometry packet even before their raster contents are separately extracted.",
            "id": "sixto_article_asset_set",
            "source_assets": [
                "SixtoRam3b",
                "Sioxto3002B",
                "SixtWave2",
                "Sioxto4001D",
                "CCH-Archm1b",
            ],
            "source_refs": [
                {
                    "kind": "live_web",
                    "lines": [19, 39],
                    "url": contra_source["url"],
                }
            ],
            "validation_status": "direct_source_asset_listing",
        },
        {
            "confidence": 0.95,
            "formal_claim": "The article explicitly states that a geometric diagram includes the Archimedean twin circles and the basic geometry of the Ramos machine. This is the strongest source-side instruction for the next extraction step.",
            "id": "sixto_diagram_presence_claim",
            "source_refs": [
                {
                    "kind": "live_web",
                    "line": 37,
                    "url": contra_source["url"],
                },
                {
                    "kind": "local_second_witness",
                    "path": "private/QAnotes/Nexus AI Chat Imports/2024/11/Triangle Ratios and Relations.md",
                    "source_line": 2228,
                },
            ],
            "validation_status": "double_witnessed_source_claim",
        },
        {
            "confidence": 0.74,
            "formal_claim": "The broader Arto book/tag surface suggests the Sixto material is part of a larger book chapter set in Talking to the Birds. This is useful provenance for future source expansion, but it is weaker than the article-level witness because it currently sits at tag/snippet level in this subproject.",
            "id": "sixto_book_provenance_note",
            "source_refs": [
                {
                    "kind": "live_intake_surface",
                    "path": "pythagoras_quantum_world_rt/arto_intake_batch_001.json",
                    "source_id": "arto_geometry_tag_live",
                },
                {
                    "kind": "live_intake_surface",
                    "path": "pythagoras_quantum_world_rt/arto_intake_batch_001.json",
                    "source_id": "arto_quantum_arithmetic_tag_live",
                },
            ],
            "validation_status": "intake_level_provenance_only",
        },
    ]
    reinterpretation_candidates = [
        item
        for item in followup_payload["items"]
        if item["id"] in {"archimedean_twin_circle_radius_note", "sixto_ramos_device_mapping_note"}
    ]
    unresolved_items = [
        {
            "best_supported_interpretation": "The source packet proves that diagrams exist and names their asset labels, but the actual geometric topology of those raster figures has not yet been transcribed into machine-readable structure.",
            "id": "sixto_raster_topology_gap",
            "normalized_answer": "Extract or recover the Sixto diagram contents before promoting any geometric measurements or stage topology.",
            "source_status": contra_source["status"],
        },
        {
            "best_supported_interpretation": "The source establishes vocabulary links to Archimedean twin circles, toroidal geometry, and the conchoid family, but it does not yet yield a validated QA or RT law.",
            "id": "sixto_no_validated_qa_bridge_yet",
            "normalized_answer": "Keep QA/RT/chromogeometry mapping downstream from a recovered diagram rather than upstream of it.",
            "source_status": contra_source["status"],
        },
    ]
    return {
        "packet_id": "sixto_ramos_geometry_packet",
        "source_lane_id": lane_payload["lane_id"],
        "source_packet_id": intake_payload["batch_id"],
        "source_geometry_items": source_geometry_items,
        "reinterpretation_candidates": reinterpretation_candidates,
        "recommended_next_actions": [
            {
                "id": "extract_sixto_diagram_topology",
                "priority": "high",
                "task": "Recover the actual geometry content of SixtoRam3b, Sioxto3002B, SixtWave2, Sioxto4001D, and CCH-Archm1b into a machine-readable witness packet.",
            },
            {
                "id": "audit_archimedean_radius_note",
                "priority": "high",
                "task": "Resolve variable meanings in the local Archimedean twin-circle radius note before any formula is reused.",
            },
            {
                "id": "delay_qa_rt_mapping_until_after_topology",
                "priority": "high",
                "task": "Do not promote ellipse-on-torus, mod-24 scheduler, or two-stage QA claims until a recovered Sixto geometry has been tested against them.",
            },
        ],
        "summary": {
            "reinterpretation_candidate_count": len(reinterpretation_candidates),
            "series": "Pyth external prior art",
            "source_geometry_count": len(source_geometry_items),
            "unresolved_count": len(unresolved_items),
        },
        "unresolved_items": unresolved_items,
    }


def self_test() -> int:
    intake_payload = {
        "batch_id": "arto_intake_batch_001",
        "sources": [
            {
                "id": "arto_contra_cyclic_post_live",
                "status": "fetched_live",
                "url": "https://artoheino.com/x",
            },
            {"id": "arto_geometry_tag_live"},
            {"id": "arto_quantum_arithmetic_tag_live"},
        ],
    }
    lane_payload = {"lane_id": "arto_contra_cyclic_lane"}
    followup_payload = {
        "items": [
            {"id": "archimedean_twin_circle_radius_note"},
            {"id": "sixto_ramos_device_mapping_note"},
            {"id": "ucros_constantinesco_followups"},
        ]
    }
    payload = build_payload(intake_payload, lane_payload, followup_payload)
    ok = (
        payload["packet_id"] == "sixto_ramos_geometry_packet"
        and payload["summary"]["source_geometry_count"] == 4
        and payload["summary"]["reinterpretation_candidate_count"] == 2
        and payload["source_geometry_items"][1]["source_assets"][0] == "SixtoRam3b"
        and payload["recommended_next_actions"][0]["id"] == "extract_sixto_diagram_topology"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    intake_payload = read_json(OUT_DIR / "arto_intake_batch_001.json")
    lane_payload = read_json(OUT_DIR / "arto_contra_cyclic_lane.json")
    followup_payload = read_json(OUT_DIR / "arto_contra_cyclic_followup_queue.json")
    payload = build_payload(intake_payload, lane_payload, followup_payload)
    write_json(OUT_PATH, payload)
    print(canonical_dump({"ok": True, "path": str(OUT_PATH.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
