"""ShapeNet-Car voxel CNN pressure smoke for QA-Fengbo.

This is the pressure-only neural-grid rung after script 64. It voxelizes real
ShapeNet-Car mesh vertices and computed normals into sparse Pepe pressure
packet tensors:

    P = mask + coordinate vector + normal-dual bivector

Then it trains the same small 3D CNN on:

  1. continuous P tensors
  2. QA-quantized/dequantized P_QA tensors

The target is direct pressure on occupied voxels. This removes the
polynomial-ridge operator from script 64, but it is still not full Fengbo:
there is no velocity V packet and the operator is a small CNN, not the thesis
Clifford/FNO stack.

QA_COMPLIANCE = "real_shapenet_voxel_cnn_pressure - exact int P_QA packet boundary; direct occupied-voxel pressure loss"
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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.qa_ml.qa_cga_grid_packet_v1 import (
    dequantize_unit,
    normal_to_dual_bivector,
    quantize_unit,
    relative_l2,
)


OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_real_shapenet_voxel_cnn_pressure.json"
CACHE_DIR = Path(__file__).resolve().parent / "real_fengbo_shapenet_car_cache"
ZIP_PATH = CACHE_DIR / "processed-car-pressure-data.zip"
ZENODO_URL = "https://zenodo.org/records/13737721/files/processed-car-pressure-data.zip"
ZENODO_RECORD = "https://zenodo.org/records/13737721"
EXPECTED_MD5 = "05153df0bd3aacdbee4a42eb00074af8"
ROOT_IN_ZIP = "processed-car-pressure-data"
MODULI = [24, 72, 144, 288]
SEED = 0


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
    request = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "qa-ml-shapenet-voxel-cnn/1.0"})
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

    vertices = np.frombuffer(raw, dtype="<f8", count=n_vertices * 3, offset=header_end)
    vertices = vertices.reshape(n_vertices, 3).astype(np.float64, copy=True)
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


def read_bounds(zf: zipfile.ZipFile) -> np.ndarray:
    text = zf.read(f"{ROOT_IN_ZIP}/watertight_global_bounds.txt").decode("utf-8")
    rows = [[float(part) for part in line.split()] for line in text.strip().splitlines()]
    bounds = np.asarray(rows, dtype=np.float64)
    if bounds.shape != (2, 3):
        raise ValueError("expected two 3D bounds rows")
    return bounds


def read_mesh_pressure(zf: zipfile.ZipFile, car_id: str, bounds: np.ndarray) -> MeshPressure:
    vertices_raw, faces = parse_ply(zf.read(f"{ROOT_IN_ZIP}/data/mesh_{car_id}.ply"))
    normals = vertex_normals(vertices_raw, faces)
    vertices = normalize_coords(vertices_raw, bounds)
    pressure_all = np.load(io.BytesIO(zf.read(f"{ROOT_IN_ZIP}/data/press_{car_id}.npy"))).astype(np.float64, copy=False)
    n_aligned = min(vertices.shape[0], pressure_all.shape[0])
    return MeshPressure(
        car_id=car_id,
        vertices=vertices[:n_aligned],
        normals=normals[:n_aligned],
        pressure=pressure_all[:n_aligned],
        pressure_tail_count=int(max(0, pressure_all.shape[0] - n_aligned)),
    )


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


def quantize_dequantize_array(values: np.ndarray, modulus: int) -> np.ndarray:
    flat = values.reshape(-1)
    out = np.empty_like(flat, dtype=np.float64)
    for idx, value in enumerate(flat):
        out[idx] = dequantize_unit(quantize_unit(float(value), modulus), modulus)
    return out.reshape(values.shape)


def packet_features(vertices: np.ndarray, normals: np.ndarray, qa_modulus: int | None) -> np.ndarray:
    bivectors = np.asarray([normal_to_dual_bivector(normal) for normal in normals], dtype=np.float64)
    if qa_modulus is not None:
        vertices = quantize_dequantize_array(vertices, qa_modulus)
        bivectors = quantize_dequantize_array(bivectors, qa_modulus)
        normals_dec = np.asarray(
            [[bivector[2], -bivector[1], bivector[0]] for bivector in bivectors],
            dtype=np.float64,
        )
        normals_dec = normals_dec / np.maximum(np.linalg.norm(normals_dec, axis=1, keepdims=True), 1e-12)
        bivectors = np.asarray([normal_to_dual_bivector(normal) for normal in normals_dec], dtype=np.float64)
    mask = np.ones((vertices.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, vertices, bivectors])


def voxelize_sample(
    sample: MeshPressure,
    grid_size: int,
    pressure_mean: float,
    pressure_std: float,
    *,
    qa_modulus: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = packet_features(sample.vertices, sample.normals, qa_modulus)
    idx = np.rint((sample.vertices + 1.0) * (grid_size - 1) / 2.0).astype(np.int64)
    idx = np.clip(idx, 0, grid_size - 1)
    linear = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]
    n_vox = grid_size * grid_size * grid_size
    counts = np.bincount(linear, minlength=n_vox).astype(np.float64)

    x_flat = np.zeros((7, n_vox), dtype=np.float64)
    for channel in range(7):
        x_flat[channel] = np.bincount(linear, weights=features[:, channel], minlength=n_vox)
    occupied = counts > 0.0
    x_flat[:, occupied] /= counts[occupied][None, :]

    pressure_norm = (sample.pressure - pressure_mean) / pressure_std
    y_flat = np.zeros(n_vox, dtype=np.float64)
    y_flat[:] = 0.0
    y_sum = np.bincount(linear, weights=pressure_norm, minlength=n_vox)
    y_flat[occupied] = y_sum[occupied] / counts[occupied]
    mask_flat = occupied.astype(np.float64)

    x = x_flat.reshape(7, grid_size, grid_size, grid_size)
    y = y_flat.reshape(1, grid_size, grid_size, grid_size)
    mask = mask_flat.reshape(1, grid_size, grid_size, grid_size)
    return x, y, mask


def build_tensors(
    samples: list[MeshPressure],
    grid_size: int,
    pressure_mean: float,
    pressure_std: float,
    *,
    qa_modulus: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    xs = []
    ys = []
    masks = []
    tails = []
    occupied = []
    for sample in samples:
        x, y, mask = voxelize_sample(sample, grid_size, pressure_mean, pressure_std, qa_modulus=qa_modulus)
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        tails.append(sample.pressure_tail_count)
        occupied.append(int(mask.sum()))
    metadata = {
        "occupied_voxels_min": int(min(occupied)),
        "occupied_voxels_max": int(max(occupied)),
        "occupied_voxels_mean": float(np.mean(occupied)),
        "pressure_tail_count_min": int(min(tails)),
        "pressure_tail_count_max": int(max(tails)),
    }
    return np.stack(xs), np.stack(ys), np.stack(masks), metadata


class TinyPressureCNN(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(7, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    return torch.sum(diff * diff) / torch.clamp(torch.sum(mask), min=1.0)


def train_and_eval(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_mask: np.ndarray,
    *,
    pressure_mean: float,
    pressure_std: float,
    hidden_channels: int,
    epochs: int,
    batch_size: int,
) -> dict[str, float]:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = TinyPressureCNN(hidden_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
        torch.tensor(train_mask, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model.train()
    losses = []
    for _epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, mb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(xb), yb, mb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        losses.append(epoch_loss / max(1, n_batches))

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(test_x, dtype=torch.float32)
        pred_norm = model(xt).cpu().numpy()
    pred = pred_norm * pressure_std + pressure_mean
    target = test_y * pressure_std + pressure_mean
    mask_bool = test_mask.astype(bool)
    return {
        "pressure_relative_l2": relative_l2(target[mask_bool], pred[mask_bool]),
        "pressure_mae": float(np.mean(np.abs(target[mask_bool] - pred[mask_bool]))),
        "final_train_masked_mse": float(losses[-1]),
        "first_train_masked_mse": float(losses[0]),
    }


def evaluate(
    train: list[MeshPressure],
    test: list[MeshPressure],
    *,
    grid_size: int,
    qa_modulus: int | None,
    hidden_channels: int,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    train_pressure = np.concatenate([sample.pressure for sample in train])
    pressure_mean = float(np.mean(train_pressure))
    pressure_std = float(max(np.std(train_pressure), 1e-12))
    train_x, train_y, train_mask, train_meta = build_tensors(
        train, grid_size, pressure_mean, pressure_std, qa_modulus=qa_modulus
    )
    test_x, test_y, test_mask, test_meta = build_tensors(
        test, grid_size, pressure_mean, pressure_std, qa_modulus=qa_modulus
    )
    metrics = train_and_eval(
        train_x,
        train_y,
        train_mask,
        test_x,
        test_y,
        test_mask,
        pressure_mean=pressure_mean,
        pressure_std=pressure_std,
        hidden_channels=hidden_channels,
        epochs=epochs,
        batch_size=batch_size,
    )
    return {
        **metrics,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "train_tensor_shape": list(train_x.shape),
        "test_tensor_shape": list(test_x.shape),
    }


def run(
    train_limit: int,
    test_limit: int,
    grid_size: int,
    hidden_channels: int,
    epochs: int,
    batch_size: int,
    timeout_s: float,
) -> dict[str, object]:
    t0 = time.time()
    train, test, source_metadata = load_split(train_limit, test_limit, timeout_s)
    continuous = evaluate(
        train,
        test,
        grid_size=grid_size,
        qa_modulus=None,
        hidden_channels=hidden_channels,
        epochs=epochs,
        batch_size=batch_size,
    )
    qa_cells = {}
    for modulus in MODULI:
        metrics = evaluate(
            train,
            test,
            grid_size=grid_size,
            qa_modulus=modulus,
            hidden_channels=hidden_channels,
            epochs=epochs,
            batch_size=batch_size,
        )
        qa_cells[str(modulus)] = {
            **metrics,
            "pressure_gap_vs_continuous": metrics["pressure_relative_l2"] - continuous["pressure_relative_l2"],
        }

    gaps = [qa_cells[str(modulus)]["pressure_gap_vs_continuous"] for modulus in MODULI]
    abs_gaps = [abs(gap) for gap in gaps]
    m144 = qa_cells["144"]
    pass_smoke = (
        abs(m144["pressure_gap_vs_continuous"]) <= 0.03
        and abs_gaps[-1] <= abs_gaps[0] + 1e-12
        and m144["pressure_relative_l2"] <= continuous["pressure_relative_l2"] + 0.03
    )
    return {
        "experiment": "pepe_ch5_real_shapenet_voxel_cnn_pressure",
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "claim_boundary": "Real ShapeNet-Car pressure-only voxel CNN parity; not full Fengbo Clifford-FNO and no velocity V packet claim.",
        "source_summary": {
            "name": "Three-dimensional flow dataset over ShapeNet-Car",
            "record_url": ZENODO_RECORD,
            "file_url": ZENODO_URL,
            "doi": "10.5281/zenodo.13737721",
            "metadata": source_metadata,
        },
        "train_count": len(train),
        "test_count": len(test),
        "grid_size": int(grid_size),
        "hidden_channels": int(hidden_channels),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "moduli": MODULI,
        "feature_schema": ["mask", "x", "y", "z", "B12", "B13", "B23"],
        "pressure_alignment_caveat": "Archive pressure vectors have 3682 entries while meshes have 3586 vertices; this smoke aligns pressure by taking the first vertex_count pressure entries and records the 96-entry tail as unused.",
        "continuous": continuous,
        "qa": qa_cells,
        "verdict": {
            "status": "PASS_REAL_VOXEL_CNN_PRESSURE_SMOKE" if pass_smoke else "FAIL_REAL_VOXEL_CNN_PRESSURE_SMOKE",
            "continuous_pressure_relative_l2": float(continuous["pressure_relative_l2"]),
            "m144_pressure_relative_l2": float(m144["pressure_relative_l2"]),
            "m144_pressure_gap_vs_continuous": float(m144["pressure_gap_vs_continuous"]),
            "pressure_gap_m24_to_m288": [float(gap) for gap in gaps],
            "pressure_abs_gap_m24_to_m288": [float(gap) for gap in abs_gaps],
            "success_criterion": "m=144 pressure gap <= +0.03, abs(m=144 gap) <= 0.03, and abs(m=288 gap) no worse than abs(m=24 gap)",
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
    sample = MeshPressure(
        car_id="toy",
        vertices=np.clip(parsed_vertices, -1.0, 1.0),
        normals=normals,
        pressure=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        pressure_tail_count=0,
    )
    x, y, mask = voxelize_sample(sample, 8, 2.0, 1.0, qa_modulus=144)
    model = TinyPressureCNN(4)
    out = model(torch.tensor(x[None], dtype=torch.float32))
    ok = (
        parsed_vertices.shape == (3, 3)
        and parsed_faces.shape == (1, 3)
        and x.shape == (7, 8, 8, 8)
        and y.shape == (1, 8, 8, 8)
        and mask.shape == (1, 8, 8, 8)
        and out.shape == (1, 1, 8, 8, 8)
        and int(mask.sum()) == 3
    )
    return {"ok": bool(ok), "occupied_voxels": int(mask.sum())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--test-limit", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--hidden-channels", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1

    result = run(
        args.train_limit,
        args.test_limit,
        args.grid_size,
        args.hidden_channels,
        args.epochs,
        args.batch_size,
        args.timeout_s,
    )
    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0 if result["verdict"]["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
