#!/usr/bin/env python3
"""Build the source-graded contra-cyclic prior-art lane from Arto intake batch 001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "arto_contra_cyclic_lane.json"
SELECTED_PRIOR_ART_KEYS = [
    "pythagorean_triples_and_theorem",
    "rational_trigonometry",
    "chromogeometry",
    "ptolemy_quadrance_and_uhg",
]


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def prior_art_refs(prior_art_payload: dict[str, object]) -> list[dict[str, object]]:
    by_key = {entry["key"]: entry for entry in prior_art_payload["categories"]}
    return [by_key[key] for key in SELECTED_PRIOR_ART_KEYS]


def build_stable_items(
    contra_source: dict[str, object], prior_art_payload: dict[str, object]
) -> list[dict[str, object]]:
    refs = prior_art_refs(prior_art_payload)
    return [
        {
            "actionability": "construction_prior_art",
            "formal_claim": "The contra-cyclic article is best treated as a dynamic-geometry prior-art packet, not as a finished theorem source. Its stable contribution is a program sketch: recover machine geometry first, then test any QA or RT bridge on top of the recovered construction layer.",
            "id": "arto_contra_cyclic_programmatic_reading",
            "prior_art_refs": refs,
            "source_kind": "article",
            "source_status": contra_source["status"],
            "supporting_witnesses": [
                {
                    "kind": "live_web",
                    "note": "The article explicitly asks that the area be defined as a branch of dynamic geometry.",
                    "url": contra_source["url"],
                },
                {
                    "kind": "local_second_witness",
                    "note": "The same article text is preserved in repo-local note imports.",
                    "path": "private/QAnotes/Nexus AI Chat Imports/2024/11/Triangle Ratios and Relations.md",
                    "source_line": 2213,
                },
            ],
            "validated_examples": {
                "high_priority_followups": [
                    "Archimedean twin circles",
                    "Sixto Ramos machine geometry",
                    "Constantinesco torque-converter geometry",
                ],
                "reading_discipline": "mechanics_first_then_qa_rt_bridge",
            },
            "validation_status": "direct_validated",
        },
        {
            "actionability": "mechanics_reconstruction",
            "formal_claim": "The most concrete geometry hook in the article is the Sixto Ramos sentence: it ties the machine to Archimedean twin circles, Fibonacci/golden structure, toroidal geometry, and the conchoid of de Sluze family. This is the core reconstruction lane to test next.",
            "id": "arto_contra_cyclic_ramos_twin_circle_bridge",
            "prior_art_refs": refs,
            "source_kind": "article",
            "source_status": contra_source["status"],
            "supporting_witnesses": [
                {
                    "kind": "live_web",
                    "note": "The article names Sixto Ramos as the clearest machine witness and links it to Archimedean twin circles and conchoid geometry.",
                    "url": contra_source["url"],
                },
                {
                    "kind": "local_second_witness",
                    "note": "The imported note preserves the same Sixto/Archimedean/conchoid wording.",
                    "path": "private/QAnotes/Nexus AI Chat Imports/2024/11/Triangle Ratios and Relations.md",
                    "source_line": 2220,
                },
            ],
            "validated_examples": {
                "curve_family_labels": [
                    "archimedean_twin_circle",
                    "fibonacci_golden_geometry",
                    "toroidal_geometry",
                    "conchoid_de_sluze",
                ],
                "machine_anchor": "Sixto Ramos",
            },
            "validation_status": "direct_validated",
        },
        {
            "actionability": "qa_bridge_mapping",
            "formal_claim": "The article explicitly places Quantum Arithmetic, Great Pyramid geometry, versins/haversins, toroids, pendulums, and Huygens-style curve families inside one dynamic-geometry vocabulary. This is a real terminology bridge, but not yet a validated identity bridge.",
            "id": "arto_contra_cyclic_qa_terminology_bridge",
            "prior_art_refs": refs,
            "source_kind": "article",
            "source_status": contra_source["status"],
            "supporting_witnesses": [
                {
                    "kind": "live_web",
                    "note": "The live article lists QA, Great Pyramid geometry, versins, haversins, toroids, pendulums, and spherical geometry in one field description.",
                    "url": contra_source["url"],
                },
                {
                    "kind": "local_second_witness",
                    "note": "The repo-local imported note preserves the same terminology cluster.",
                    "path": "private/QAnotes/Nexus AI Chat Imports/2024/11/Triangle Ratios and Relations.md",
                    "source_line": 2218,
                },
            ],
            "validated_examples": {
                "terminology_cluster": [
                    "Quantum Arithmetic",
                    "Great Pyramid geometry",
                    "versin",
                    "haversin",
                    "toroids",
                    "pendulums",
                    "spherical geometry",
                    "Huygens evolutes and caustics",
                ],
                "bridge_type": "terminology_and_scope_only",
            },
            "validation_status": "direct_validated",
        },
        {
            "actionability": "dynamic_geometry_typology",
            "formal_claim": "The source proposes a specific motion typology for the new branch: 3D dynamics modeled as single, double, and triple-stage spherical rotating pendulums with optional pivot locations. That is stable as a source-intended kinematic frame even though no equations are provided.",
            "id": "arto_contra_cyclic_pendulum_typology",
            "prior_art_refs": refs,
            "source_kind": "article",
            "source_status": contra_source["status"],
            "supporting_witnesses": [
                {
                    "kind": "live_web",
                    "note": "The article defines the proposed branch in terms of staged spherical rotating pendulums.",
                    "url": contra_source["url"],
                },
                {
                    "kind": "local_second_witness",
                    "note": "The local imported note preserves the same staged-pendulum frame.",
                    "path": "private/QAnotes/Nexus AI Chat Imports/2024/11/Triangle Ratios and Relations.md",
                    "source_line": 2218,
                },
            ],
            "validated_examples": {
                "stages": [1, 2, 3],
                "pivoting_axis": "optional",
                "source_label": "upside_down_rotating_spherical_pendulum",
            },
            "validation_status": "direct_validated",
        },
    ]


def build_unresolved_items(contra_source: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "best_supported_interpretation": "The article gestures at a broad historical machine lineage, but does not provide enough technical detail to validate any performance or priority claims from the text alone.",
            "id": "arto_contra_cyclic_historical_scope_claims",
            "issue_label": "broad_historical_claims_unvalidated",
            "normalized_answer": "Keep the Archimedes-to-modern-machine lineage as source rhetoric until individual machine constructions are independently reconstructed.",
            "source_claim": "The geometry has been used in many machines for thousands of years and repeatedly rediscovered.",
            "source_status": contra_source["status"],
        },
        {
            "best_supported_interpretation": "The article provides a vocabulary bridge to QA, but not an equation-level bridge. Any RT/chromogeometry/Ptolemy reinterpretation still has to be done downstream from recovered constructions.",
            "id": "arto_contra_cyclic_no_equation_layer",
            "issue_label": "missing_formalization",
            "normalized_answer": "Do not promote any numeric identity, spread law, or quadrance law from this article alone.",
            "source_claim": "QA and related geometries are said to be involved, but no explicit formulas or test cases are given.",
            "source_status": contra_source["status"],
        },
        {
            "best_supported_interpretation": "The crop-circle and sacred-geometry rhetoric belongs to source context only and should not be promoted into the validated lane.",
            "id": "arto_contra_cyclic_speculative_context",
            "issue_label": "speculative_context",
            "normalized_answer": "Quarantine the sacred-geometry / crop-circle framing as non-operational rhetoric.",
            "source_claim": "The same geometry is described as the same as all crop circles around the world.",
            "source_status": contra_source["status"],
        },
    ]


def build_lane(
    intake_payload: dict[str, object], prior_art_payload: dict[str, object]
) -> dict[str, object]:
    by_id = {entry["id"]: entry for entry in intake_payload["sources"]}
    contra_source = by_id["arto_contra_cyclic_post_live"]
    stable_items = build_stable_items(contra_source, prior_art_payload)
    unresolved_items = build_unresolved_items(contra_source)
    return {
        "lane_id": "arto_contra_cyclic_lane",
        "source_packet_id": intake_payload["batch_id"],
        "stable_items": stable_items,
        "summary": {
            "focus": "Contra-cyclic dynamic-geometry extraction",
            "series": "Pyth external prior art",
            "stable_count": len(stable_items),
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
            }
        ],
    }
    prior_art_payload = {
        "categories": [
            {"key": key, "refs": [], "summary": key} for key in SELECTED_PRIOR_ART_KEYS
        ]
    }
    payload = build_lane(intake_payload, prior_art_payload)
    ok = (
        payload["lane_id"] == "arto_contra_cyclic_lane"
        and payload["summary"]["stable_count"] == 4
        and payload["summary"]["unresolved_count"] == 3
        and payload["stable_items"][1]["validated_examples"]["machine_anchor"] == "Sixto Ramos"
        and payload["unresolved_items"][2]["issue_label"] == "speculative_context"
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
    prior_art_payload = read_json(OUT_DIR / "prior_art_bridge_map.json")
    payload = build_lane(intake_payload, prior_art_payload)
    write_json(OUT_PATH, payload)
    print(canonical_dump({"ok": True, "path": str(OUT_PATH.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
