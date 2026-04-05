#!/usr/bin/env python3
"""Build a direct QA-first correspondence artifact for the Arto contra-cyclic lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "arto_contra_cyclic_qa_correspondence.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def build_payload() -> dict[str, object]:
    return {
        "artifact_id": "arto_contra_cyclic_qa_correspondence",
        "stance": {
            "derivation_direction": "qa_to_arto",
            "operational_rule": (
                "Treat Arto contra-cyclic formulas and machine sketches as downstream from the QA bead "
                "(b,e,d,a) unless a source witness forces a contradiction."
            ),
            "reason": (
                "The local QA mapping notes already formulate the contra-cyclic machine packet in QA terms, "
                "while the Arto article itself provides geometry rhetoric and diagrams rather than an independent equation layer."
            ),
        },
        "qa_core": {
            "tuple": ["b", "e", "d", "a"],
            "relations": [
                {"id": "qa_rel_01", "equation": "d = b + e"},
                {"id": "qa_rel_02", "equation": "a = b + 2e"},
                {"id": "qa_rel_03", "equation": "F = a*b"},
                {"id": "qa_rel_04", "equation": "C = 2*e*d"},
                {"id": "qa_rel_05", "equation": "G = d*d + e*e"},
            ],
            "notes": [
                "Use d*d, never d**2, in substrate-facing expressions.",
                "These are the canonical QA primitives for mapping the contra-cyclic lane.",
            ],
        },
        "derived_correspondences": [
            {
                "id": "cc_qa_01",
                "claim": "Archimedean twin-circle / golden gearing is modeled by the QA ladder.",
                "qa_form": "(b,e) = (1,1) -> (1,2) -> (2,3) -> (3,5) -> ...",
                "derived_quantities": ["d = b + e", "a = b + 2e", "gear_ratio = a/d"],
                "status": "qa_first_local_prior_work",
                "source_refs": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:41",
                    "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:36",
                ],
            },
            {
                "id": "cc_qa_02",
                "claim": "Contra-cyclic timing is a QA tuple scheduler rather than a floating-angle primitive.",
                "qa_form": "Advance tuple states on residue cycles, especially mod-24 event windows.",
                "derived_quantities": ["tick set", "impulse windows", "non-overlapping stage locks"],
                "status": "qa_first_local_prior_work",
                "source_refs": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:45",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:48",
                ],
            },
            {
                "id": "cc_qa_03",
                "claim": "The machine-stage drive law is naturally written as a QA velocity proxy.",
                "qa_form": "v = F/C = a*b / (2*e*d)",
                "derived_quantities": ["rectified transmit ticks", "impulse plateau", "stage torque proxy"],
                "status": "qa_first_local_prior_work",
                "source_refs": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:51",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:56",
                ],
            },
            {
                "id": "cc_qa_04",
                "claim": "Sixto Ramos can be treated as two QA stages in series.",
                "qa_form": "Stage A tuple plus Stage B tuple with non-overlapping mod-24 windows.",
                "derived_quantities": ["alternating rectified pushes", "blue/green wave timeline", "stage offset"],
                "status": "qa_first_local_prior_work",
                "source_refs": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:63",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:66",
                ],
            },
            {
                "id": "cc_qa_05",
                "claim": "The contra-cyclic curve families are read as QA-generated envelopes and offset loci.",
                "qa_form": "Sample the QA ellipse / stage packet on the active residue set, then form envelopes from the discrete normal or offset data.",
                "derived_quantities": ["conchoid-like family", "evolute/caustic envelope", "sparse offset family"],
                "status": "qa_first_local_prior_work",
                "source_refs": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:81",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:84",
                ],
            },
        ],
        "source_hierarchy": {
            "tier_1_article": {
                "role": "geometry rhetoric and named machine witnesses",
                "path": "pythagoras_quantum_world_rt/arto_contra_cyclic_lane.json",
            },
            "tier_2_local_qa_mapping": {
                "role": "QA-first derivation direction for formulas and machine scheduling",
                "paths": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md",
                    "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md",
                ],
            },
        },
        "quarantined_items": [
            {
                "id": "cc_qa_q01",
                "claim": "r = d*d - d*sqrt(F) for the Archimedean twin circle",
                "status": "unvalidated",
                "reason": "Local note exists, but the variable meanings and geometric target remain unstable.",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:14",
            },
            {
                "id": "cc_qa_q02",
                "claim": "quadrance reading r^2 = D^2",
                "status": "unvalidated",
                "reason": "Useful QA-facing hypothesis, but still not source-certified from Arto alone.",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
            },
        ],
        "next_rule": {
            "id": "cc_qa_next_rule",
            "instruction": (
                "Use QA-derived correspondences as the default reading for Arto contra-cyclic work. "
                "Only open a source-recovery subtask when the raster witness is needed to decide between competing QA readings."
            ),
        },
    }


def self_test() -> int:
    payload = build_payload()
    ok = True
    ok = ok and payload["stance"]["derivation_direction"] == "qa_to_arto"
    ok = ok and len(payload["derived_correspondences"]) >= 5
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    write_json(OUT_PATH, payload)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": ["pythagoras_quantum_world_rt/arto_contra_cyclic_qa_correspondence.json"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
