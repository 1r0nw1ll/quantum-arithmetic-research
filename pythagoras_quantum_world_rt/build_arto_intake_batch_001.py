#!/usr/bin/env python3
"""Build the first controlled Arto Heino intake batch for the Pyth/QA/RT subproject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "arto_intake_batch_001.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def repo_grounding() -> list[dict[str, object]]:
    return [
        {
            "kind": "project_note",
            "note": "Arto is already grounded in the repo as the geometry-facing renderer of the wider QA line.",
            "path": "Documents/synopsis_for_dale_pond_2026-03-27.md",
            "quote_excerpt": "Arto Heino as geometric renderer",
            "source_line": 106,
        },
        {
            "kind": "experiment_lineage",
            "note": "The repo already has Arto-based intake and experiment work on a different branch, so this pull extends existing intake practice rather than inventing a new one.",
            "path": "experiments/arto_ternary_topology_part_count_experiment.py",
            "quote_excerpt": "https://artoheino.com/2024/03/23/ternary-coded-decimal-and-7-segment-control/",
            "source_line": 138,
        },
    ]


def build_payload() -> dict[str, object]:
    shared_grounding = repo_grounding()
    sources = [
        {
            "alignment_hypotheses": [
                "contra_cyclic_harmonic_geometry",
                "archimedean_twin_circle_mechanics",
                "qa_adjacent_machine_geometry",
                "toroidal_or_pendular_dynamic_geometry",
            ],
            "category": "external_prior_art",
            "confidence": 0.96,
            "id": "arto_contra_cyclic_post_live",
            "kind": "article",
            "notes": [
                "Directly names the contra-cyclic lane the user flagged as high-interest.",
                "Connects Archimedes, Ben Iverson QA, torque-conversion machines, versin language, toroids, and dynamic geometry.",
                "Looks like a geometry/mechanics framework rather than a finished formal derivation, so it should be mined as construction prior art first.",
            ],
            "notable_items": [
                {
                    "detail": "Frames the topic as a geometry used in many machines across Archimedes, Da Vinci, Huygens, Bessler, Keely, Constantinesco, and others.",
                    "kind": "scope_claim",
                },
                {
                    "detail": "Explicitly lists Quantum Arithmetic, Great Pyramid geometry, toroids, pendulums, spherical geometry, versins, and Archimedean twin circles in the same construction field.",
                    "kind": "bridge_claim",
                },
                {
                    "detail": "Calls for defining the area as a branch of dynamic geometry rather than isolated machine tricks.",
                    "kind": "research_direction",
                },
            ],
            "pull_priority": "high",
            "repo_grounding": shared_grounding,
            "status": "fetched_live",
            "title": "Contra-Cyclic Harmonic Archimedean Geometry",
            "url": "https://artoheino.com/2012/10/05/contra-cyclic-harmonic-archimedean-geometry/",
            "witness": {
                "kind": "web_open",
                "observed_utc": "2026-03-30T00:00:00Z",
                "source_excerpt": "The geometry of Archimedes, toroids, pendulums, spherical geometry ... Quantum Arithmetic(as per Ben Iverson, Not Quantum Physics) and Great Pyramid geometry could all be involved.",
            },
        },
        {
            "alignment_hypotheses": [
                "qa_ellipse_archaeology",
                "saqqara_ostrakon_reconstruction",
                "whole_number_ellipse_parameterization",
                "gaussian_integer_terminology_tracking",
            ],
            "category": "external_prior_art",
            "confidence": 0.93,
            "id": "arto_geometry_tag_live",
            "kind": "tag_page",
            "notes": [
                "The current open surface is older than the search-snippet surface, but it still exposes the ellipse/QA archaeology lane clearly.",
                "Useful as a hub page for QA ellipse construction posts and Egyptian artifact reinterpretation.",
            ],
            "notable_items": [
                {
                    "detail": "Saqqara Ostrakon",
                    "kind": "article_title",
                    "published_date": "2012-12-05",
                },
                {
                    "detail": "Quantum Arithmetic",
                    "kind": "article_title",
                    "published_date": "2012-11-22",
                },
                {
                    "detail": "Book Release – Talking to the Birds",
                    "kind": "article_title",
                    "published_date": "2013-11-15",
                },
            ],
            "pull_priority": "high",
            "repo_grounding": shared_grounding,
            "status": "fetched_live",
            "title": "geometry tag page",
            "url": "https://artoheino.com/tag/geometry/",
            "witness": {
                "kind": "web_open",
                "observed_utc": "2026-03-30T00:00:00Z",
                "source_excerpt": "Saqqara Ostrakon a Different and Exact Solution ... revealing the true principles of Ancient Khemitian(Egyptian) geometry",
            },
        },
        {
            "alignment_hypotheses": [
                "qa_identity_surface",
                "quantum_ellipse_construction",
                "great_pyramid_qa_parameters",
                "gaussian_integer_language_in_source",
            ],
            "category": "external_prior_art",
            "confidence": 0.95,
            "id": "arto_quantum_arithmetic_tag_live",
            "kind": "tag_page",
            "notes": [
                "Best direct Arto surface for QA-specific terminology and geometric claims.",
                "Contains both ellipse-facing and pyramid-facing QA posts, plus explicit Gaussian-integer language in the QA exposition.",
            ],
            "notable_items": [
                {
                    "detail": "A Quantum Ellipse of the Old Kingdom",
                    "kind": "article_title",
                    "published_date": "2021-11-02",
                },
                {
                    "detail": "The Great Pyramid – A Quantum Solution",
                    "kind": "article_title",
                    "published_date": "2012-12-21",
                },
                {
                    "detail": "Quantum Arithmetic",
                    "kind": "article_title",
                    "published_date": "2012-11-22",
                },
            ],
            "pull_priority": "high",
            "repo_grounding": shared_grounding,
            "status": "fetched_live",
            "title": "Quantum Arithmetic tag page",
            "url": "https://artoheino.com/tag/quantum-arithmetic/",
            "witness": {
                "kind": "web_open",
                "observed_utc": "2026-03-30T00:00:00Z",
                "source_excerpt": "There is also one dimension below the roots b, e, d, and a. They are called quaternions, and are the square roots of the root numbers. In conventional mathematics, these are called Gaussian Integers.",
            },
        },
        {
            "alignment_hypotheses": [
                "contra_cyclic_followup_surface",
                "ben_iverson_lineage",
                "recent_qa_geometry_posts",
            ],
            "category": "external_prior_art",
            "confidence": 0.78,
            "id": "arto_ben_iverson_tag_discovery",
            "kind": "tag_page",
            "notes": [
                "This surface was discovered earlier in the live browse pass and should be treated as a follow-up target rather than a fully extracted witness in this batch.",
                "High value because it links recent posts directly to Ben Iverson and surfaced the contra-cyclic post in the same orbit.",
            ],
            "notable_items": [
                {
                    "detail": "Quantum Arithmetic",
                    "kind": "article_title",
                    "published_date": "2024-11-16",
                },
                {
                    "detail": "Contra-Cyclic Harmonic Archimedean Geometry",
                    "kind": "article_title",
                    "published_date": "2024-10-31",
                },
            ],
            "pull_priority": "high",
            "repo_grounding": shared_grounding,
            "status": "discovered_live_prior_turn",
            "title": "Ben Iverson tag page",
            "url": "https://artoheino.com/tag/ben-iverson/",
            "witness": {
                "kind": "prior_turn_browse",
                "observed_utc": "2026-03-30T00:00:00Z",
                "source_excerpt": "Visible items included Quantum Arithmetic and Contra-Cyclic Harmonic Archimedean Geometry on the Ben-Iverson-tag surface.",
            },
        },
        {
            "alignment_hypotheses": [
                "implementation_artifacts",
                "code_or_diagram_exports",
                "supporting_repos_or_notebooks",
            ],
            "category": "external_prior_art",
            "confidence": 0.61,
            "id": "arto_github_profile_queue",
            "kind": "github_profile",
            "notes": [
                "Kept explicit in the intake batch so later repo/code pulls stay attached to the same witness packet.",
                "No stable repo listing was recovered during this pass, so this remains queued rather than inflated into fake repo entries.",
            ],
            "notable_items": [],
            "pull_priority": "queued_reference",
            "repo_grounding": shared_grounding,
            "status": "queued_target",
            "title": "arto-heino GitHub profile",
            "url": "https://github.com/arto-heino",
            "witness": {
                "kind": "user_supplied_target",
                "observed_utc": "2026-03-30T00:00:00Z",
                "source_excerpt": "User requested this GitHub profile be tracked as an Arto pull source.",
            },
        },
    ]
    return {
        "batch_id": "arto_intake_batch_001",
        "created_utc": "2026-03-30T00:00:00Z",
        "focus": "Disciplined Arto Heino intake for QA/RT/Pyth geometry alignment, with contra-cyclic work prioritized.",
        "series": "Pyth-2 external prior art",
        "sources": sources,
        "summary": {
            "fetched_live_count": sum(1 for source in sources if source["status"] == "fetched_live"),
            "high_priority_ids": [
                source["id"] for source in sources if source["pull_priority"] == "high"
            ],
            "queued_count": sum(1 for source in sources if source["status"] == "queued_target"),
            "repo_grounding_count": len(shared_grounding),
            "source_count": len(sources),
        },
    }


def self_test() -> int:
    payload = build_payload()
    source_ids = {source["id"] for source in payload["sources"]}
    ok = (
        payload["batch_id"] == "arto_intake_batch_001"
        and payload["summary"]["source_count"] == 5
        and payload["summary"]["fetched_live_count"] == 3
        and payload["summary"]["queued_count"] == 1
        and "arto_contra_cyclic_post_live" in source_ids
        and "arto_github_profile_queue" in source_ids
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    payload = build_payload()
    write_json(OUT_PATH, payload)
    print(canonical_dump({"ok": True, "path": str(OUT_PATH.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
