"""Continuous mini-Fengbo vs QA-quantized mini-Fengbo operator parity.

This is the controlled operator rung after packet parity. It trains the same
small deterministic operator on the same synthetic geometry/field split in two
representations:

  1. continuous Fengbo-style geometry packets
  2. QA-dequantized Fengbo-style geometry packets

It does not use ShapeNet Car or Ahmed Body and does not claim full Fengbo
replication. The claim is only whether QA packetization preserves the behavior
of a Fengbo-style geometry-to-physics operator on a controlled voxel task.

QA_COMPLIANCE = "qa_fengbo_operator_parity - exact integer packets at QA boundary; observer decode for sklearn operator"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.qa_ml.qa_cga_grid_packet_v1 import (
    decode_dual_normal,
    decode_trivector,
    decode_vector,
    encode_pressure_packet,
    encode_velocity_packet,
    normal_to_dual_bivector,
    relative_l2,
)


OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_qa_fengbo_operator_parity.json"
SEED = 0
GRID_SIZE = 32
MODULI = [24, 48, 72, 144, 288]
N_SHAPES = 48
N_THETA = 18
N_PHI = 8
N_AXIS = 7


@dataclass(frozen=True)
class ShapeSpec:
    axes: tuple[float, float, float]
    inlet: float
    yaw: float


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rotation_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_shape_specs(n: int) -> list[ShapeSpec]:
    specs = []
    for idx in range(n):
        t = idx / max(1, n - 1)
        axes = (
            0.58 + 0.26 * (0.5 + 0.5 * np.sin(2.0 * np.pi * t)),
            0.32 + 0.14 * (0.5 + 0.5 * np.cos(3.0 * np.pi * t)),
            0.24 + 0.12 * (0.5 + 0.5 * np.sin(5.0 * np.pi * t + 0.3)),
        )
        inlet = -0.55 + 1.10 * ((idx * 7) % n) / max(1, n - 1)
        yaw = -0.45 + 0.90 * ((idx * 11) % n) / max(1, n - 1)
        specs.append(ShapeSpec(axes=axes, inlet=float(inlet), yaw=float(yaw)))
    return specs


def surface_points(spec: ShapeSpec) -> tuple[np.ndarray, np.ndarray]:
    axes = np.asarray(spec.axes, dtype=np.float64)
    rot = rotation_z(spec.yaw)
    points = []
    normals = []
    for a in np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False):
        for b in np.linspace(0.18 * np.pi, 0.82 * np.pi, N_PHI):
            unit = np.asarray([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
            local = axes * unit
            normal_raw = local / (axes * axes)
            normal = normal_raw / np.linalg.norm(normal_raw)
            points.append(rot @ local)
            normals.append(rot @ normal)
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def volume_points(spec: ShapeSpec) -> np.ndarray:
    coords = np.linspace(-0.9, 0.9, N_AXIS)
    rot = rotation_z(spec.yaw)
    axes = np.asarray(spec.axes, dtype=np.float64)
    pts = []
    for x in coords:
        for y in coords:
            for z in coords:
                local = np.asarray([x, y, z], dtype=np.float64)
                if np.sum((local / (axes + 0.18)) * (local / (axes + 0.18))) <= 1.0:
                    pts.append(rot @ local)
    return np.asarray(pts, dtype=np.float64)


def pressure_target(points: np.ndarray, normals: np.ndarray, spec: ShapeSpec) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    ax, ay, az = spec.axes
    return (
        1.0
        + 0.42 * spec.inlet * nx
        + 0.18 * x
        - 0.13 * y
        + 0.09 * z
        + 0.22 * nz
        + 0.12 * ax * x * x
        - 0.08 * ay * y * z
        + 0.06 * az * nx * ny
    )


def velocity_target(points: np.ndarray, spec: ShapeSpec) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax, ay, az = spec.axes
    return np.stack(
        (
            spec.inlet * (1.0 - 0.30 * y * y) + 0.10 * ax * z,
            0.22 * x - 0.12 * spec.inlet * y + 0.08 * ay * x * z,
            0.18 * z + 0.10 * az * x - 0.05 * y * z,
        ),
        axis=1,
    )


def pressure_features(points: np.ndarray, normals: np.ndarray, spec: ShapeSpec) -> np.ndarray:
    b = np.asarray([normal_to_dual_bivector(n) for n in normals], dtype=np.float64)
    shape = np.tile(np.asarray([*spec.axes, spec.inlet], dtype=np.float64), (points.shape[0], 1))
    mask = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, points, b, shape])


def velocity_features(points: np.ndarray, spec: ShapeSpec) -> np.ndarray:
    shape = np.tile(np.asarray([*spec.axes, spec.inlet], dtype=np.float64), (points.shape[0], 1))
    mask = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, points, shape])


def qa_pressure_features(points: np.ndarray, normals: np.ndarray, spec: ShapeSpec, modulus: int) -> np.ndarray:
    packets = [
        encode_pressure_packet(p, n, grid_size=GRID_SIZE, modulus=modulus, inlet_velocity=spec.inlet)
        for p, n in zip(points, normals)
    ]
    points_q = np.asarray([decode_vector(packet) for packet in packets], dtype=np.float64)
    normals_q = np.asarray([decode_dual_normal(packet) for packet in packets], dtype=np.float64)
    normals_q = normals_q / np.maximum(np.linalg.norm(normals_q, axis=1, keepdims=True), 1e-12)
    b_q = np.asarray([normal_to_dual_bivector(n) for n in normals_q], dtype=np.float64)
    inlet_q = np.asarray([decode_trivector(packet) for packet in packets], dtype=np.float64)
    shape = np.tile(np.asarray([*spec.axes], dtype=np.float64), (points.shape[0], 1))
    mask = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, points_q, b_q, shape, inlet_q[:, None]])


def qa_velocity_features(points: np.ndarray, spec: ShapeSpec, modulus: int) -> np.ndarray:
    packets = [
        encode_velocity_packet(p, grid_size=GRID_SIZE, modulus=modulus)
        for p in points
    ]
    points_q = np.asarray([decode_vector(packet) for packet in packets], dtype=np.float64)
    shape = np.tile(np.asarray([*spec.axes, spec.inlet], dtype=np.float64), (points.shape[0], 1))
    mask = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, points_q, shape])


def collect_dataset(specs: list[ShapeSpec], *, qa_modulus: int | None = None) -> dict[str, np.ndarray]:
    pressure_x = []
    pressure_y = []
    velocity_x = []
    velocity_y = []
    for spec in specs:
        pts, normals = surface_points(spec)
        vol = volume_points(spec)
        if qa_modulus is None:
            pressure_x.append(pressure_features(pts, normals, spec))
            velocity_x.append(velocity_features(vol, spec))
        else:
            pressure_x.append(qa_pressure_features(pts, normals, spec, qa_modulus))
            velocity_x.append(qa_velocity_features(vol, spec, qa_modulus))
        pressure_y.append(pressure_target(pts, normals, spec)[:, None])
        velocity_y.append(velocity_target(vol, spec))
    return {
        "pressure_x": np.vstack(pressure_x),
        "pressure_y": np.vstack(pressure_y),
        "velocity_x": np.vstack(velocity_x),
        "velocity_y": np.vstack(velocity_y),
    }


def fit_operator(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=1e-6),
    ).fit(x, y)


def evaluate(train_specs: list[ShapeSpec], test_specs: list[ShapeSpec], *, qa_modulus: int | None = None) -> dict[str, float]:
    train = collect_dataset(train_specs, qa_modulus=qa_modulus)
    test = collect_dataset(test_specs, qa_modulus=qa_modulus)
    p_model = fit_operator(train["pressure_x"], train["pressure_y"])
    v_model = fit_operator(train["velocity_x"], train["velocity_y"])
    p_pred = p_model.predict(test["pressure_x"])
    v_pred = v_model.predict(test["velocity_x"])
    return {
        "pressure_relative_l2": relative_l2(np.ravel(test["pressure_y"]), np.ravel(p_pred)),
        "velocity_relative_l2": relative_l2(np.ravel(test["velocity_y"]), np.ravel(v_pred)),
        "pressure_train_points": int(train["pressure_x"].shape[0]),
        "pressure_test_points": int(test["pressure_x"].shape[0]),
        "velocity_train_points": int(train["velocity_x"].shape[0]),
        "velocity_test_points": int(test["velocity_x"].shape[0]),
    }


def run() -> dict[str, object]:
    t0 = time.time()
    specs = make_shape_specs(N_SHAPES)
    train_specs = [spec for i, spec in enumerate(specs) if i % 4 != 0]
    test_specs = [spec for i, spec in enumerate(specs) if i % 4 == 0]

    continuous = evaluate(train_specs, test_specs, qa_modulus=None)
    qa_cells = {}
    for modulus in MODULI:
        metrics = evaluate(train_specs, test_specs, qa_modulus=modulus)
        qa_cells[str(modulus)] = {
            **metrics,
            "pressure_gap_vs_continuous": metrics["pressure_relative_l2"] - continuous["pressure_relative_l2"],
            "velocity_gap_vs_continuous": metrics["velocity_relative_l2"] - continuous["velocity_relative_l2"],
        }

    pressure_gaps = [qa_cells[str(m)]["pressure_gap_vs_continuous"] for m in MODULI]
    velocity_gaps = [qa_cells[str(m)]["velocity_gap_vs_continuous"] for m in MODULI]
    m144 = qa_cells["144"]
    pass_operator = (
        m144["pressure_gap_vs_continuous"] <= 0.01
        and m144["velocity_gap_vs_continuous"] <= 0.02
        and m144["pressure_relative_l2"] <= 0.02
        and m144["velocity_relative_l2"] <= 0.05
    )
    verdict = {
        "status": "PASS_OPERATOR_PARITY" if pass_operator else "FAIL_OPERATOR_PARITY",
        "continuous_pressure_relative_l2": continuous["pressure_relative_l2"],
        "continuous_velocity_relative_l2": continuous["velocity_relative_l2"],
        "m144_pressure_relative_l2": m144["pressure_relative_l2"],
        "m144_velocity_relative_l2": m144["velocity_relative_l2"],
        "m144_pressure_gap_vs_continuous": m144["pressure_gap_vs_continuous"],
        "m144_velocity_gap_vs_continuous": m144["velocity_gap_vs_continuous"],
        "pressure_gap_monotone_nonincreasing": all(b <= a + 1e-12 for a, b in zip(pressure_gaps, pressure_gaps[1:])),
        "velocity_gap_monotone_nonincreasing": all(b <= a + 1e-12 for a, b in zip(velocity_gaps, velocity_gaps[1:])),
        "claim_boundary": "synthetic mini-Fengbo operator parity only; no ShapeNet/Ahmed or full Clifford FNO claim",
    }
    return {
        "ok": verdict["status"] == "PASS_OPERATOR_PARITY",
        "schema": "QA_ML_PEPE_CH5_QA_FENGBO_OPERATOR_PARITY.v1",
        "seed": SEED,
        "grid_size": GRID_SIZE,
        "moduli": MODULI,
        "n_shapes": N_SHAPES,
        "n_train_shapes": len(train_specs),
        "n_test_shapes": len(test_specs),
        "continuous": continuous,
        "qa_cells": qa_cells,
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
