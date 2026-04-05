#!/usr/bin/env python3
"""Build a machine-readable topology witness for the Sixto source asset set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "sixto_topology_witness.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def build_assets() -> list[dict[str, object]]:
    return [
        {
            "asset_id": "sixtoram3b",
            "attachment_id": 488,
            "content_type": "image/jpeg",
            "downloaded_local_copy": "/tmp/sixto_assets/sixtoram3b.jpg",
            "html_source_line": 320,
            "orig_file": "https://artoheino.com/wp-content/uploads/2012/10/sixtoram3b.jpg",
            "orig_size": [3138, 3138],
            "source_title": "SixtoRam3b",
            "topology_extract": {
                "confidence": 0.86,
                "labeled_nodes_visible": [],
                "primary_primitives": [
                    "large_outer_annulus_or_spirographic_ring",
                    "left_hub_circle",
                    "right_hub_circle",
                    "left_perimeter_marker_ring",
                    "right_perimeter_marker_ring",
                    "crossing_chord_bundle_with_single_waist",
                ],
                "topology_summary": "Two small circular hub assemblies are connected by many straight chords that cross at a single narrow waist between the hubs. The pair sits inside a much larger outer annular trace produced by repeated rotational sweeps.",
            },
        },
        {
            "asset_id": "sioxto3002b",
            "attachment_id": 489,
            "content_type": "image/jpeg",
            "downloaded_local_copy": "/tmp/sixto_assets/sioxto3002b.jpg",
            "html_source_line": 321,
            "orig_file": "https://artoheino.com/wp-content/uploads/2012/10/sioxto3002b.jpg",
            "orig_size": [4308, 3023],
            "source_title": "Sioxto3002B",
            "topology_extract": {
                "confidence": 0.92,
                "labeled_nodes_visible": [
                    "A",
                    "A1",
                    "B",
                    "B1",
                    "C",
                    "C1",
                    "D",
                    "D1",
                    "H1",
                    "H2",
                    "H3",
                    "H4",
                    "N1",
                    "N2",
                    "N3",
                    "N4",
                    "K1",
                    "K2",
                    "K3",
                    "K4",
                    "X0",
                    "X1",
                    "X2",
                    "X3",
                    "Y1",
                    "Y2",
                    "Y3",
                    "Z",
                ],
                "primary_primitives": [
                    "paired_main_circles_with_construction_lines",
                    "central_right_twin_circle_packet",
                    "top_right_time_distance_graph",
                    "bottom_four_stage_circle_chain",
                ],
                "quantitative_text_visible": {
                    "A_mm": 72,
                    "A1_mm": 72,
                    "B_mm": 233,
                    "B1_mm": 233,
                    "C_mm": 243.870867468831,
                    "C1_mm": 243.870867468831,
                    "K_rule": "K1 = Distance = Circumference/2; K2 = K1 x 0.62; K3 = K2 x 0.62; K4 = K3 x 0.62",
                    "time_rule": "H1 = H2 = H3 = H4 = TimeH; N1 = N2 = N3 = N4 = TimeN; H + N = 1 Revolution in time",
                },
                "topology_summary": "This is the clearest engineering-layout witness. It shows a main geometric construction feeding a four-stage chain of circle pairs labeled H1/N1 through H4/N4, with stage distances K1 through K4 and a small timing graph indexed by X0 through X3.",
            },
        },
        {
            "asset_id": "sixtwave2",
            "attachment_id": 490,
            "content_type": "image/jpeg",
            "downloaded_local_copy": "/tmp/sixto_assets/sixtwave2.jpg",
            "html_source_line": 322,
            "orig_file": "https://artoheino.com/wp-content/uploads/2012/10/sixtwave2.jpg",
            "orig_size": [4806, 2645],
            "source_title": "SixtWave2",
            "topology_extract": {
                "confidence": 0.9,
                "labeled_nodes_visible": [
                    "X0",
                    "X1",
                    "X2",
                    "X3",
                    "X4",
                    "Z1",
                    "Z2",
                ],
                "primary_primitives": [
                    "velocity_time_graph",
                    "driven_offset_snapshot_row",
                    "multiplier_offset_snapshot_row",
                    "perspective_mechanical_linkage_view",
                ],
                "quantitative_text_visible": {
                    "rpm": 600,
                    "time_seconds": 0.1,
                    "impulse_advantage_time_seconds": 0.025,
                    "driven_offset_span_mm": 233,
                    "vertical_span_mm": 144,
                },
                "topology_summary": "This witness turns the geometry into a phase/timing picture. It couples a velocity graph to two rows of offset-circle snapshots and a perspective linkage drawing, with the Z1/Z2 split repeated below the circles and on the mechanical view.",
            },
        },
        {
            "asset_id": "sioxto4001d",
            "attachment_id": 491,
            "content_type": "image/jpeg",
            "downloaded_local_copy": "/tmp/sixto_assets/sioxto4001d.jpg",
            "html_source_line": 323,
            "orig_file": "https://artoheino.com/wp-content/uploads/2012/10/sioxto4001d.jpg",
            "orig_size": [3514, 4061],
            "source_title": "Sioxto4001D",
            "topology_extract": {
                "confidence": 0.8,
                "labeled_nodes_visible": [
                    "H1",
                    "H2",
                    "H3",
                    "H4",
                    "N1",
                    "N2",
                    "N3",
                    "N4",
                    "K1",
                    "K2",
                    "K3",
                    "K4",
                    "1A",
                    "1B",
                    "1C",
                    "2A",
                    "2B",
                    "2C",
                    "3A",
                    "3B",
                    "3C",
                    "4A",
                    "4B",
                ],
                "primary_primitives": [
                    "multi_curve_velocity_graph",
                    "four_stage_circle_chain",
                    "two_layer_perspective_linkage_bundle",
                    "bottom_phase_order_strip",
                ],
                "uncertainty_flags": [
                    "Some numeric stage-scaling text is too small for fully reliable transcription at this pass.",
                ],
                "topology_summary": "This witness extends the earlier stage-chain into a denser multistage packet: a multi-curve graph, a horizontal H/N-K chain, a layered perspective linkage bundle, and a bottom ordered strip of phase labels 1A through 4B/4C.",
            },
        },
        {
            "asset_id": "cch_archm1b",
            "attachment_id": 511,
            "content_type": "image/png",
            "downloaded_local_copy": "/tmp/sixto_assets/cch-archm1b.png",
            "html_source_line": 331,
            "orig_file": "https://artoheino.com/wp-content/uploads/2012/10/cch-archm1b.png",
            "orig_size": [3444, 1773],
            "source_title": "CCH-Archm1b",
            "topology_extract": {
                "confidence": 0.88,
                "labeled_nodes_visible": [
                    "D",
                    "F",
                    "G",
                    "AH",
                    "HG",
                    "Q1=34",
                    "17",
                    "55",
                    "89",
                    "144",
                ],
                "primary_primitives": [
                    "large_left_red_circle",
                    "central_orange_circle",
                    "upper_left_green_circle",
                    "upper_right_blue_circle",
                    "right_blue_triangular_wedge",
                    "central_black_triangle",
                    "diagonal_red_axis_lines",
                ],
                "topology_summary": "This is the explicit Archimedean twin-circle overview. It combines several colored circle families, a rightward blue wedge/triangle, a central black triangle, and numeric labels including 17, 34, 55, 89, and 144.",
            },
        },
    ]


def build_payload(packet_payload: dict[str, object]) -> dict[str, object]:
    assets = build_assets()
    return {
        "witness_id": "sixto_topology_witness",
        "source_packet_id": packet_payload["packet_id"],
        "assets": assets,
        "cross_asset_topology": {
            "confidence": 0.89,
            "invariants": [
                "A repeated two-hub geometry appears across the source set: a left assembly and a right assembly connected by a narrow waist or transfer bundle.",
                "Later source diagrams unfold that two-hub relation into staged chains H1/N1 through H4/N4 with K1 through K4.",
                "The source repeatedly mixes circle geometry with time or phase indexing via X0..X4 and Z1/Z2.",
                "The Archimedean twin-circle overview adds an explicit number ladder 17,34,55,89,144 on top of the geometry packet.",
            ],
            "machine_readable_nodes": [
                "left_hub_assembly",
                "right_hub_assembly",
                "waist_crossing_bundle",
                "stage_chain_HN",
                "distance_chain_K",
                "phase_indices_X",
                "zone_split_Z1_Z2",
                "archimedean_twin_circle_overview",
            ],
        },
        "promotion_barriers": [
            "No exact coordinates, radii, or line equations have been transcribed yet from the raster sources.",
            "The source images are sufficient for topology extraction but not yet for certified quadrance/spread law promotion.",
            "Later QA or RT reinterpretations remain downstream until one diagram is vectorized or otherwise structurally recovered.",
        ],
        "recommended_next_actions": [
            {
                "id": "vectorize_cch_archm1b",
                "priority": "high",
                "task": "Recover exact primitive relationships from the CCH-Archm1b overview first, because it is the explicit Archimedean twin-circle diagram.",
            },
            {
                "id": "transcribe_hn_k_chain",
                "priority": "high",
                "task": "Turn the H1/N1..H4/N4 and K1..K4 chain in Sioxto3002B into a structured stage graph with numeric labels.",
            },
            {
                "id": "delay_formula_promotion",
                "priority": "high",
                "task": "Keep the Archimedean-radius and mod-24 scheduler notes quarantined until the stage graph and overview diagram have exact structural recovery.",
            },
        ],
        "summary": {
            "asset_count": len(assets),
            "series": "Pyth external prior art",
        },
    }


def self_test() -> int:
    payload = build_payload({"packet_id": "sixto_ramos_geometry_packet"})
    ok = (
        payload["witness_id"] == "sixto_topology_witness"
        and payload["summary"]["asset_count"] == 5
        and payload["assets"][1]["topology_extract"]["quantitative_text_visible"]["A_mm"] == 72
        and payload["assets"][4]["topology_extract"]["labeled_nodes_visible"][0] == "D"
        and payload["recommended_next_actions"][0]["id"] == "vectorize_cch_archm1b"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    packet_payload = read_json(OUT_DIR / "sixto_ramos_geometry_packet.json")
    payload = build_payload(packet_payload)
    write_json(OUT_PATH, payload)
    print(canonical_dump({"ok": True, "path": str(OUT_PATH.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
