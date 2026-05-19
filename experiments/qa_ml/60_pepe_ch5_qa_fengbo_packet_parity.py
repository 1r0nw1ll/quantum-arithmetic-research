"""QA-Fengbo packet parity smoke test.

This is the first lawful Fengbo-to-QA step after source recovery. It does not
train a neural operator. It tests whether Pepe's Fengbo geometry multivectors
can be represented as exact QA/CGA grid packets with small observer-side field
reconstruction error after decode.

QA_COMPLIANCE = "qa_fengbo_packet_parity - exact integer packets; observer decode only for relative-L2 metrics"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.qa_ml.qa_cga_grid_packet_v1 import (
    decode_dual_normal,
    decode_vector,
    encode_pressure_packet,
    encode_velocity_packet,
    relative_l2,
)


OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_qa_fengbo_packet_parity.json"
SEED = 0
GRID_SIZE = 32
MODULI = [12, 24, 48, 72, 144, 288]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_surface_points(n_theta: int = 24, n_phi: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ellipsoid surface points and normals in [-1,1]^3."""
    points = []
    normals = []
    axes = np.asarray([0.82, 0.44, 0.30], dtype=np.float64)
    for a in np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False):
        for b in np.linspace(0.18 * np.pi, 0.82 * np.pi, n_phi):
            unit = np.asarray([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
            point = axes * unit
            normal_raw = point / (axes * axes)
            normal = normal_raw / np.linalg.norm(normal_raw)
            points.append(point)
            normals.append(normal)
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def make_volume_points(n_axis: int = 10) -> np.ndarray:
    coords = np.linspace(-0.9, 0.9, n_axis)
    pts = []
    for x in coords:
        for y in coords:
            for z in coords:
                if x*x + y*y + z*z <= 1.0:
                    pts.append((x, y, z))
    return np.asarray(pts, dtype=np.float64)


def pressure_field(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nz = normals[:, 2]
    return 1.0 + 0.35 * x - 0.22 * y + 0.18 * z + 0.31 * nz


def velocity_field(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack((-0.45 * y, 0.45 * x, 0.20 * z + 0.10 * x), axis=1)


def run() -> dict[str, object]:
    t0 = time.time()
    surface, normals = make_surface_points()
    volume = make_volume_points()
    pressure_ref = pressure_field(surface, normals)
    velocity_ref = velocity_field(volume)

    cells = {}
    for modulus in MODULI:
        pressure_packets = [
            encode_pressure_packet(p, n, grid_size=GRID_SIZE, modulus=modulus)
            for p, n in zip(surface, normals)
        ]
        velocity_packets = [
            encode_velocity_packet(p, grid_size=GRID_SIZE, modulus=modulus)
            for p in volume
        ]

        surface_dec = np.asarray([decode_vector(packet) for packet in pressure_packets])
        normals_dec = np.asarray([decode_dual_normal(packet) for packet in pressure_packets])
        normals_dec = normals_dec / np.maximum(np.linalg.norm(normals_dec, axis=1, keepdims=True), 1e-12)
        volume_dec = np.asarray([decode_vector(packet) for packet in velocity_packets])

        pressure_est = pressure_field(surface_dec, normals_dec)
        velocity_est = velocity_field(volume_dec)

        cells[str(modulus)] = {
            "modulus": modulus,
            "pressure_relative_l2": relative_l2(pressure_ref, pressure_est),
            "velocity_relative_l2": relative_l2(velocity_ref, velocity_est),
            "coord_relative_l2_surface": relative_l2(surface, surface_dec),
            "coord_relative_l2_volume": relative_l2(volume, volume_dec),
            "normal_relative_l2": relative_l2(normals, normals_dec),
        }

    pressure_errors = [cells[str(m)]["pressure_relative_l2"] for m in MODULI]
    velocity_errors = [cells[str(m)]["velocity_relative_l2"] for m in MODULI]
    monotone_pressure = all(b <= a + 1e-12 for a, b in zip(pressure_errors, pressure_errors[1:]))
    monotone_velocity = all(b <= a + 1e-12 for a, b in zip(velocity_errors, velocity_errors[1:]))
    verdict = {
        "status": "PASS_PACKET_PARITY" if cells["144"]["pressure_relative_l2"] < 0.01 and cells["144"]["velocity_relative_l2"] < 0.01 else "FAIL_PACKET_PARITY",
        "m144_pressure_relative_l2": cells["144"]["pressure_relative_l2"],
        "m144_velocity_relative_l2": cells["144"]["velocity_relative_l2"],
        "monotone_pressure": monotone_pressure,
        "monotone_velocity": monotone_velocity,
        "claim_boundary": "packet/decode parity only; no learned Fengbo operator or ShapeNet/Ahmed reproduction",
    }
    return {
        "ok": verdict["status"] == "PASS_PACKET_PARITY",
        "schema": "QA_ML_PEPE_CH5_QA_FENGBO_PACKET_PARITY.v1",
        "seed": SEED,
        "grid_size": GRID_SIZE,
        "moduli": MODULI,
        "n_pressure_surface_points": int(surface.shape[0]),
        "n_velocity_volume_points": int(volume.shape[0]),
        "cells": cells,
        "verdict": verdict,
        "runtime_s": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    result = run()
    if args.self_test:
        print(canonical_json({"ok": bool(result["ok"]), "schema": result["schema"], "verdict": result["verdict"]}))
        return 0 if result["ok"] else 1

    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json({"ok": result["ok"], "path": str(OUT_PATH.relative_to(ROOT)), "verdict": result["verdict"]}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
