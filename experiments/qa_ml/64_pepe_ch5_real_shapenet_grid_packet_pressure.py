"""ShapeNet-Car grid-packet pressure smoke for QA-Fengbo.

This is the step after the descriptor/PCA pressure-field smoke. It uses actual
mesh vertices, computed vertex normals, and pressure targets from the public
processed ShapeNet-Car pressure archive. Each sampled surface point is encoded
as Pepe-style pressure geometry:

    P = mask + coordinate vector + normal-dual bivector

The continuous operator uses the floating P features. The QA operator uses
integer QA packets at the geometry boundary, then observer-decodes them for the
same deterministic regression operator. Pressure remains the direct scalar
target; there is no PCA target shortcut.

Claim boundary: real sampled pressure field over ShapeNet-Car meshes, not full
Fengbo Clifford-FNO and no velocity claim.

QA_COMPLIANCE = "real_shapenet_grid_packet_pressure - exact int P_QA packet boundary; direct pressure scalar target"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
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
    decode_vector,
    encode_pressure_packet,
    normal_to_dual_bivector,
    relative_l2,
)


OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_real_shapenet_grid_packet_pressure.json"
CACHE_DIR = Path(__file__).resolve().parent / "real_fengbo_shapenet_car_cache"
ZIP_PATH = CACHE_DIR / "processed-car-pressure-data.zip"
ZENODO_URL = "https://zenodo.org/records/13737721/files/processed-car-pressure-data.zip"
ZENODO_RECORD = "https://zenodo.org/records/13737721"
EXPECTED_MD5 = "05153df0bd3aacdbee4a42eb00074af8"
ROOT_IN_ZIP = "processed-car-pressure-data"
GRID_SIZE = 80
MODULI = [24, 48, 72, 144, 288]


@dataclass(frozen=True)
class MeshPressure:
    car_id: str
    vertices: np.ndarray
    normals: np.ndarray
    pressure: np.ndarray
    pressure_tail_count: int


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ensure_archive(timeout_s: float) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        return
    request = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "qa-ml-shapenet-grid-packet/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        ZIP_PATH.write_bytes(response.read())


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_ids(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def read_manifest(zf: zipfile.ZipFile, name: str) -> list[str]:
    return parse_ids(zf.read(f"{ROOT_IN_ZIP}/{name}.txt").decode("utf-8"))


def parse_ply(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError("PLY header missing end_header")
    header_end = end + len(b"end_header\n")
    header = raw[:header_end].decode("utf-8", errors="replace")
    vertex_match = re.search(r"element vertex (\d+)", header)
    face_match = re.search(r"element face (\d+)", header)
    if vertex_match is None or face_match is None:
        raise ValueError("PLY header missing vertex/face counts")
    n_vertices = int(vertex_match.group(1))
    n_faces = int(face_match.group(1))
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("expected binary little endian PLY")

    vertex_values = np.frombuffer(raw, dtype="<f8", count=n_vertices * 3, offset=header_end)
    vertices = vertex_values.reshape(n_vertices, 3).astype(np.float64, copy=True)
    faces = np.empty((n_faces, 3), dtype=np.int64)
    pos = header_end + n_vertices * 3 * 8
    for face_idx in range(n_faces):
        n = raw[pos]
        pos += 1
        if n != 3:
            raise ValueError("expected triangular faces")
        faces[face_idx] = np.frombuffer(raw, dtype="<u4", count=3, offset=pos)
        pos += 12
    return vertices, faces


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = np.linalg.norm(vertices - vertices.mean(axis=0, keepdims=True), axis=1, keepdims=True)
    radial = (vertices - vertices.mean(axis=0, keepdims=True)) / np.maximum(fallback, 1e-12)
    return np.where(lengths > 1e-12, normals / np.maximum(lengths, 1e-12), radial)


def normalize_coords(vertices: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[0]
    hi = bounds[1]
    span = np.maximum(hi - lo, 1e-12)
    return np.clip(2.0 * (vertices - lo) / span - 1.0, -1.0, 1.0)


def deterministic_indices(n_items: int, sample_count: int, offset: int) -> np.ndarray:
    if sample_count >= n_items:
        return np.arange(n_items, dtype=np.int64)
    base = np.linspace(0, n_items - 1, sample_count, dtype=np.int64)
    return (base + offset) % n_items


def read_mesh_pressure(zf: zipfile.ZipFile, car_id: str, bounds: np.ndarray) -> MeshPressure:
    mesh_raw = zf.read(f"{ROOT_IN_ZIP}/data/mesh_{car_id}.ply")
    pressure_raw = zf.read(f"{ROOT_IN_ZIP}/data/press_{car_id}.npy")
    vertices_raw, faces = parse_ply(mesh_raw)
    normals = vertex_normals(vertices_raw, faces)
    vertices = normalize_coords(vertices_raw, bounds)
    pressure_all = np.load(io.BytesIO(pressure_raw)).astype(np.float64, copy=False)
    n_aligned = min(vertices.shape[0], pressure_all.shape[0])
    return MeshPressure(
        car_id=car_id,
        vertices=vertices[:n_aligned],
        normals=normals[:n_aligned],
        pressure=pressure_all[:n_aligned],
        pressure_tail_count=int(max(0, pressure_all.shape[0] - n_aligned)),
    )


def read_bounds(zf: zipfile.ZipFile) -> np.ndarray:
    text = zf.read(f"{ROOT_IN_ZIP}/watertight_global_bounds.txt").decode("utf-8")
    rows = [[float(part) for part in line.split()] for line in text.strip().splitlines()]
    bounds = np.asarray(rows, dtype=np.float64)
    if bounds.shape != (2, 3):
        raise ValueError("expected two 3D bounds rows")
    return bounds


def load_split(train_limit: int, test_limit: int, timeout_s: float) -> tuple[list[MeshPressure], list[MeshPressure], dict[str, object]]:
    ensure_archive(timeout_s)
    digest = md5_file(ZIP_PATH)
    if digest != EXPECTED_MD5:
        raise ValueError(f"unexpected archive md5 {digest}; expected {EXPECTED_MD5}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        bounds = read_bounds(zf)
        train_ids = read_manifest(zf, "train")[:train_limit]
        test_ids = read_manifest(zf, "test")[:test_limit]
        train = [read_mesh_pressure(zf, car_id, bounds) for car_id in train_ids]
        test = [read_mesh_pressure(zf, car_id, bounds) for car_id in test_ids]
    metadata = {
        "archive_path": str(ZIP_PATH),
        "archive_size_bytes": ZIP_PATH.stat().st_size,
        "archive_md5": digest,
        "source_url": ZENODO_URL,
        "source_record": ZENODO_RECORD,
        "bounds": bounds.tolist(),
    }
    return train, test, metadata


def continuous_features(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    bivectors = np.asarray([normal_to_dual_bivector(normal) for normal in normals], dtype=np.float64)
    mask = np.ones((vertices.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, vertices, bivectors])


def qa_features(vertices: np.ndarray, normals: np.ndarray, modulus: int) -> np.ndarray:
    packets = [
        encode_pressure_packet(vertex, normal, grid_size=GRID_SIZE, modulus=modulus)
        for vertex, normal in zip(vertices, normals)
    ]
    vertices_q = np.asarray([decode_vector(packet) for packet in packets], dtype=np.float64)
    normals_q = np.asarray([decode_dual_normal(packet) for packet in packets], dtype=np.float64)
    normals_q = normals_q / np.maximum(np.linalg.norm(normals_q, axis=1, keepdims=True), 1e-12)
    bivectors_q = np.asarray([normal_to_dual_bivector(normal) for normal in normals_q], dtype=np.float64)
    mask = np.ones((vertices.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, vertices_q, bivectors_q])


def collect(samples: list[MeshPressure], sample_count: int, *, qa_modulus: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    xs = []
    ys = []
    used = []
    tails = []
    for sample_idx, sample in enumerate(samples):
        idx = deterministic_indices(sample.vertices.shape[0], sample_count, sample_idx * 17)
        vertices = sample.vertices[idx]
        normals = sample.normals[idx]
        if qa_modulus is None:
            xs.append(continuous_features(vertices, normals))
        else:
            xs.append(qa_features(vertices, normals, qa_modulus))
        ys.append(sample.pressure[idx, None])
        used.append(int(idx.size))
        tails.append(sample.pressure_tail_count)
    metadata = {
        "points_per_car_min": int(min(used)),
        "points_per_car_max": int(max(used)),
        "pressure_tail_count_min": int(min(tails)),
        "pressure_tail_count_max": int(max(tails)),
    }
    return np.vstack(xs), np.vstack(ys), metadata


def fit_operator(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=1e-2),
    ).fit(x, y)


def evaluate(
    train: list[MeshPressure],
    test: list[MeshPressure],
    sample_count: int,
    *,
    qa_modulus: int | None = None,
) -> dict[str, object]:
    train_x, train_y, train_meta = collect(train, sample_count, qa_modulus=qa_modulus)
    test_x, test_y, test_meta = collect(test, sample_count, qa_modulus=qa_modulus)
    model = fit_operator(train_x, train_y)
    pred = model.predict(test_x)
    return {
        "pressure_relative_l2": relative_l2(np.ravel(test_y), np.ravel(pred)),
        "pressure_mae": float(np.mean(np.abs(test_y - pred))),
        "train_points": int(train_x.shape[0]),
        "test_points": int(test_x.shape[0]),
        "train_meta": train_meta,
        "test_meta": test_meta,
    }


def run(train_limit: int, test_limit: int, sample_count: int, timeout_s: float) -> dict[str, object]:
    t0 = time.time()
    train, test, source_metadata = load_split(train_limit, test_limit, timeout_s)
    continuous = evaluate(train, test, sample_count, qa_modulus=None)
    qa_cells = {}
    for modulus in MODULI:
        metrics = evaluate(train, test, sample_count, qa_modulus=modulus)
        qa_cells[str(modulus)] = {
            **metrics,
            "pressure_gap_vs_continuous": metrics["pressure_relative_l2"] - continuous["pressure_relative_l2"],
        }
    gaps = [qa_cells[str(modulus)]["pressure_gap_vs_continuous"] for modulus in MODULI]
    abs_gaps = [abs(gap) for gap in gaps]
    m144 = qa_cells["144"]
    pass_smoke = (
        abs(m144["pressure_gap_vs_continuous"]) <= 0.02
        and abs_gaps[-1] <= abs_gaps[0] + 1e-12
    )
    return {
        "experiment": "pepe_ch5_real_shapenet_grid_packet_pressure",
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "claim_boundary": "Real sampled ShapeNet-Car pressure packets P only; not full Fengbo Clifford-FNO and no velocity claim.",
        "source_summary": {
            "name": "Three-dimensional flow dataset over ShapeNet-Car",
            "record_url": ZENODO_RECORD,
            "file_url": ZENODO_URL,
            "doi": "10.5281/zenodo.13737721",
            "metadata": source_metadata,
        },
        "train_count": len(train),
        "test_count": len(test),
        "sample_count_per_car": int(sample_count),
        "grid_size": GRID_SIZE,
        "moduli": MODULI,
        "feature_schema": ["mask", "x", "y", "z", "B12", "B13", "B23"],
        "pressure_alignment_caveat": "Archive pressure vectors have 3682 entries while meshes have 3586 vertices; this smoke aligns pressure by taking the first vertex_count pressure entries and records the 96-entry tail as unused.",
        "continuous": continuous,
        "qa": qa_cells,
        "verdict": {
            "status": "PASS_REAL_GRID_PACKET_PRESSURE_SMOKE" if pass_smoke else "FAIL_REAL_GRID_PACKET_PRESSURE_SMOKE",
            "continuous_pressure_relative_l2": float(continuous["pressure_relative_l2"]),
            "m144_pressure_relative_l2": float(m144["pressure_relative_l2"]),
            "m144_pressure_gap_vs_continuous": float(m144["pressure_gap_vs_continuous"]),
            "pressure_gap_m24_to_m288": [float(gap) for gap in gaps],
            "pressure_abs_gap_m24_to_m288": [float(gap) for gap in abs_gaps],
            "success_criterion": "abs(m=144 pressure gap) <= 0.02 and abs(m=288 gap) no worse than abs(m=24 gap)",
        },
    }


def self_test() -> dict[str, object]:
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 3\nproperty double x\nproperty double y\nproperty double z\n"
        b"element face 1\nproperty list uchar uint vertex_indices\nend_header\n"
    )
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="<f8")
    face = bytes([3]) + np.asarray([0, 1, 2], dtype="<u4").tobytes()
    parsed_vertices, parsed_faces = parse_ply(header + vertices.tobytes() + face)
    normals = vertex_normals(parsed_vertices, parsed_faces)
    feats = continuous_features(parsed_vertices, normals)
    qfeats = qa_features(np.clip(parsed_vertices, -1.0, 1.0), normals, 144)
    ok = (
        parsed_vertices.shape == (3, 3)
        and parsed_faces.shape == (1, 3)
        and feats.shape == (3, 7)
        and qfeats.shape == (3, 7)
        and np.allclose(normals[:, 2], 1.0)
    )
    return {"ok": bool(ok), "feature_count": int(feats.shape[1])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1

    result = run(args.train_limit, args.test_limit, args.sample_count, args.timeout_s)
    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0 if result["verdict"]["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
