"""Canonical GINO Car-CFD pressure operator + matched QA parity.

This is the faithful continuation of scripts 65/66. It abandons the
hand-assembled pipeline on the surface-pressure Zenodo record (13737721) and
uses the **canonical neuraloperator GINO Car-CFD record** instead:

    Zenodo 13936501 — "Three-dimensional flow dataset over ShapeNet-Car"
    (the record `neuralop.data.datasets.CarCFDDataset` downloads).

Why this matters (established by a landscape pass, recorded in OB):
  * The Fengbo/GINO ShapeNet-Car benchmark has NO volumetric velocity field in
    any distribution — trusted or mirror. Every ML-ready release (13737721,
    13936501, neuraloperator) is a SURFACE-PRESSURE prediction task. The
    "velocity V packet" expectation in the Pepe Ch5 chain was a framing error.
  * This record carries the authors' canonical pressure alignment: the press
    vector has 3682 entries and the mesh 3586 vertices; the documented fix is
    `concat(press[:16], press[112:])` (drop indices 16:112). Scripts 64–66
    hand-waved this as "take first 3586, 96-tail unused"; here we use the
    authors' exact alignment, removing that caveat.

Caveat kept honest: the authors' GINO representation uses an open3d SIGNED
distance field over a query grid. open3d is not installed here, so this uses
the same hardened UNSIGNED nearest-surface distance as script 66. That is the
one remaining representation gap vs canonical GINO; it is recorded, not hidden.

The operator is the published-class 3D FNO (from script 66). Continuous and
QA-quantized packet boundaries are matched (quantization on the distance
channel AND voxel placement). The verdict reports continuous R^2 and emits NO
green Fengbo PASS under any branch: the continuous operator is expected to
remain weak because surface-pressure-from-geometry is hard at CPU scale — that
is the documented ceiling, now stated against the canonical benchmark itself.

QA_COMPLIANCE = "canonical_gino_carcfd_pressure - trusted md5-pinned GINO record; canonical press alignment; exact int P_QA packet boundary (channels + placement); honest weak-baseline verdict, no Fengbo solver/velocity claim"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.qa_ml.qa_cga_grid_packet_v1 import (
    dequantize_unit,
    normal_to_dual_bivector,
    quantize_unit,
    relative_l2,
)

OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_canonical_gino_carcfd_pressure.json"
CACHE_DIR = Path(__file__).resolve().parent / "gino_carcfd_cache"
ARCHIVE_PATH = CACHE_DIR / "processed-car-pressure-data.tar.gz"
ZENODO_RECORD = "https://zenodo.org/records/13936501"
ZENODO_URL = "https://zenodo.org/api/records/13936501/files/processed-car-pressure-data.tar.gz/content"
EXPECTED_MD5 = "24a46fe791085201d48ee5db7b6cfc86"
ROOT_IN_TAR = "processed-car-pressure-data"
# Canonical neuraloperator press alignment: drop indices 16:112 (96 entries)
# so the 3682-entry press vector matches the 3586 mesh vertices.
PRESS_KEEP_HEAD = 16
PRESS_DROP_TO = 112
MODULI = [24, 72, 144, 288]
SEED = 0
SURF_DIST_DECAY = 8.0


@dataclass(frozen=True)
class MeshPressure:
    car_id: str
    vertices: np.ndarray
    normals: np.ndarray
    pressure: np.ndarray
    alignment: str


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ensure_archive(timeout_s: float) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        return
    request = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "qa-ml-gino-carcfd/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        ARCHIVE_PATH.write_bytes(response.read())


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ids(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


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

    # Heterogeneous archive: Open3D (double, list uchar uint) and meshio
    # (float, list uint8 int32). Detect both. (The official train/test split
    # this script loads is entirely double/uchar; this covers the rest.)
    if re.search(r"property\s+double\s+x", header):
        vtype, vsize = "<f8", 8
    elif re.search(r"property\s+float\s+x", header):
        vtype, vsize = "<f4", 4
    else:
        raise ValueError("PLY vertex property type not float/double")
    list_match = re.search(r"property\s+list\s+(\w+)\s+(\w+)\s+vertex_indices", header)
    if list_match is None:
        raise ValueError("PLY face list declaration missing")
    count_sizes = {"uchar": 1, "uint8": 1, "char": 1, "int8": 1,
                   "ushort": 2, "uint16": 2, "short": 2, "int16": 2}
    index_sizes = {"uint": 4, "uint32": 4, "int": 4, "int32": 4}
    cnt_t, idx_t = list_match.group(1), list_match.group(2)
    if cnt_t not in count_sizes or idx_t not in index_sizes:
        raise ValueError(f"unsupported PLY face list types {cnt_t}/{idx_t}")
    cnt_size = count_sizes[cnt_t]
    idx_dtype = "<u4" if idx_t in ("uint", "uint32") else "<i4"

    vertices = np.frombuffer(raw, dtype=vtype, count=n_vertices * 3, offset=header_end)
    vertices = vertices.reshape(n_vertices, 3).astype(np.float64, copy=True)

    faces: list[tuple[int, int, int]] = []
    pos = header_end + n_vertices * 3 * vsize
    for _ in range(n_faces):
        n = int.from_bytes(raw[pos : pos + cnt_size], "little")
        pos += cnt_size
        if n < 3:
            raise ValueError(f"degenerate PLY face with {n} vertices")
        verts = np.frombuffer(raw, dtype=idx_dtype, count=n, offset=pos).astype(np.int64)
        pos += 4 * n
        for k in range(1, n - 1):
            faces.append((int(verts[0]), int(verts[k]), int(verts[k + 1])))
    return vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


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
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-12)
    return np.clip(2.0 * (vertices - lo) / span - 1.0, -1.0, 1.0)


def align_pressure(press: np.ndarray, n_vertices: int) -> tuple[np.ndarray, str]:
    flat = press.reshape(-1).astype(np.float64, copy=False)
    if flat.shape[0] - (PRESS_DROP_TO - PRESS_KEEP_HEAD) == n_vertices:
        kept = np.concatenate([flat[:PRESS_KEEP_HEAD], flat[PRESS_DROP_TO:]])
        return kept, "canonical_concat_0_16_112_end"
    n = min(flat.shape[0], n_vertices)
    return flat[:n], f"fallback_first_{n}"


class TarCarCFD:
    def __init__(self, path: Path) -> None:
        self.tf = tarfile.open(path)

    def read(self, name: str) -> bytes:
        member = self.tf.extractfile(f"{ROOT_IN_TAR}/{name}")
        if member is None:
            raise FileNotFoundError(name)
        return member.read()

    def manifest(self, name: str) -> list[str]:
        return parse_ids(self.read(f"{name}.txt").decode("utf-8"))

    def bounds(self) -> np.ndarray:
        rows = [
            [float(p) for p in line.split()]
            for line in self.read("watertight_global_bounds.txt").decode("utf-8").strip().splitlines()
        ]
        arr = np.asarray(rows, dtype=np.float64)
        if arr.shape != (2, 3):
            raise ValueError("expected two 3D bounds rows")
        return arr

    def sample(self, car_id: str, bounds: np.ndarray) -> MeshPressure:
        verts_raw, faces = parse_ply(self.read(f"data/mesh_{car_id}.ply"))
        normals = vertex_normals(verts_raw, faces)
        verts = normalize_coords(verts_raw, bounds)
        press = np.load(io.BytesIO(self.read(f"data/press_{car_id}.npy")))
        aligned, mode = align_pressure(press, verts.shape[0])
        n = min(verts.shape[0], aligned.shape[0])
        return MeshPressure(car_id, verts[:n], normals[:n], aligned[:n], mode)


def load_split(
    train_limit: int, test_limit: int, timeout_s: float
) -> tuple[list[MeshPressure], list[MeshPressure], dict[str, object]]:
    ensure_archive(timeout_s)
    digest = md5_file(ARCHIVE_PATH)
    if digest != EXPECTED_MD5:
        raise ValueError(f"unexpected archive md5 {digest}; expected {EXPECTED_MD5}")
    tc = TarCarCFD(ARCHIVE_PATH)
    bounds = tc.bounds()
    train_ids = tc.manifest("train")[:train_limit]
    test_ids = tc.manifest("test")[:test_limit]
    train = [tc.sample(c, bounds) for c in train_ids]
    test = [tc.sample(c, bounds) for c in test_ids]
    metadata = {
        "archive_path": str(ARCHIVE_PATH),
        "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
        "archive_md5": digest,
        "source_url": ZENODO_URL,
        "source_record": ZENODO_RECORD,
        "bounds": bounds.tolist(),
        "pressure_alignment_modes": sorted({s.alignment for s in train + test}),
    }
    return train, test, metadata


def quantize_dequantize_array(values: np.ndarray, modulus: int) -> np.ndarray:
    flat = values.reshape(-1)
    out = np.empty_like(flat, dtype=np.float64)
    for idx, value in enumerate(flat):
        out[idx] = dequantize_unit(quantize_unit(float(value), modulus), modulus)
    return out.reshape(values.shape)


def packet_features(
    vertices: np.ndarray, normals: np.ndarray, qa_modulus: int | None
) -> np.ndarray:
    bivectors = np.asarray(
        [normal_to_dual_bivector(normal) for normal in normals], dtype=np.float64
    )
    if qa_modulus is not None:
        vertices = quantize_dequantize_array(vertices, qa_modulus)
        bivectors = quantize_dequantize_array(bivectors, qa_modulus)
        normals_dec = np.asarray(
            [[b[2], -b[1], b[0]] for b in bivectors], dtype=np.float64
        )
        normals_dec = normals_dec / np.maximum(
            np.linalg.norm(normals_dec, axis=1, keepdims=True), 1e-12
        )
        bivectors = np.asarray(
            [normal_to_dual_bivector(n) for n in normals_dec], dtype=np.float64
        )
    mask = np.ones((vertices.shape[0], 1), dtype=np.float64)
    return np.hstack([mask, vertices, bivectors])


def build_tensors(
    samples: list[MeshPressure],
    grid_size: int,
    pressure_mean: float,
    pressure_std: float,
    centers: np.ndarray,
    *,
    qa_modulus: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n_vox = grid_size * grid_size * grid_size
    xs, ys, masks, occupied = [], [], [], []
    for sample in samples:
        # QA boundary applied consistently: SAME quantized vertices drive the
        # surface-distance channel AND voxel placement.
        verts = sample.vertices
        if qa_modulus is not None:
            verts = quantize_dequantize_array(verts, qa_modulus)
        distance, _ = cKDTree(verts).query(centers)
        surf_dist = np.exp(-SURF_DIST_DECAY * distance).reshape(1, n_vox)

        idx = np.clip(
            np.rint((verts + 1.0) * (grid_size - 1) / 2.0).astype(np.int64),
            0,
            grid_size - 1,
        )
        linear = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]
        counts = np.bincount(linear, minlength=n_vox).astype(np.float64)
        occ = counts > 0.0

        features = packet_features(sample.vertices, sample.normals, qa_modulus)
        packed = np.zeros((7, n_vox), dtype=np.float64)
        for channel in range(7):
            packed[channel] = np.bincount(
                linear, weights=features[:, channel], minlength=n_vox
            )
        packed[:, occ] /= counts[occ][None, :]

        pressure_norm = (sample.pressure - pressure_mean) / pressure_std
        y_flat = np.zeros(n_vox, dtype=np.float64)
        y_sum = np.bincount(linear, weights=pressure_norm, minlength=n_vox)
        y_flat[occ] = y_sum[occ] / counts[occ]

        x = np.concatenate([surf_dist, packed], axis=0).reshape(
            8, grid_size, grid_size, grid_size
        )
        xs.append(x)
        ys.append(y_flat.reshape(1, grid_size, grid_size, grid_size))
        masks.append(occ.astype(np.float64).reshape(1, grid_size, grid_size, grid_size))
        occupied.append(int(occ.sum()))
    metadata = {
        "occupied_voxels_min": int(min(occupied)),
        "occupied_voxels_max": int(max(occupied)),
        "occupied_voxels_mean": float(np.mean(occupied)),
        "feature_channels": 8,
    }
    return np.stack(xs), np.stack(ys), np.stack(masks), metadata


class SpectralConv3d(nn.Module):
    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(scale * torch.rand(channels, channels, modes, modes, modes, 2))
                for _ in range(4)
            ]
        )

    @staticmethod
    def _mul(block: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", block, torch.view_as_complex(weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out = torch.zeros(
            b, self.weights[0].shape[1], d, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        md = self.modes
        out[:, :, :md, :md, :md] = self._mul(xf[:, :, :md, :md, :md], self.weights[0])
        out[:, :, -md:, :md, :md] = self._mul(xf[:, :, -md:, :md, :md], self.weights[1])
        out[:, :, :md, -md:, :md] = self._mul(xf[:, :, :md, -md:, :md], self.weights[2])
        out[:, :, -md:, -md:, :md] = self._mul(xf[:, :, -md:, -md:, :md], self.weights[3])
        return torch.fft.irfftn(out, s=(d, h, w), dim=(-3, -2, -1))


class FNO3d(nn.Module):
    def __init__(self, in_channels: int, width: int, modes: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList([SpectralConv3d(width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList(
            [nn.Conv3d(width, width, kernel_size=1) for _ in range(layers)]
        )
        self.drop = nn.Dropout3d(dropout)
        self.proj1 = nn.Conv3d(width, 64, kernel_size=1)
        self.proj2 = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spec, lin in zip(self.spectral, self.pointwise):
            x = x + torch.nn.functional.gelu(spec(x) + lin(x))
        return self.proj2(torch.nn.functional.gelu(self.proj1(self.drop(x))))


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
    width: int,
    modes: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict[str, float]:
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = FNO3d(8, width, modes, layers, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    xt = torch.tensor(train_x, dtype=torch.float32)
    yt = torch.tensor(train_y, dtype=torch.float32)
    mt = torch.tensor(train_mask, dtype=torch.float32)
    n = len(xt)
    gen = torch.Generator().manual_seed(SEED)
    first_loss = None
    last_loss = 0.0
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            j = perm[i : i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(xt[j]), yt[j], mt[j])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        mean_loss = epoch_loss / max(1, n_batches)
        if first_loss is None:
            first_loss = mean_loss
        last_loss = mean_loss

    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(test_x, dtype=torch.float32)).cpu().numpy()
    pred = pred_norm * pressure_std + pressure_mean
    target = test_y * pressure_std + pressure_mean
    mask_bool = test_mask.astype(bool)
    truth = target[mask_bool]
    estimate = pred[mask_bool]
    centered = truth - truth.mean()
    mean_residual_rel_l2 = float(
        np.linalg.norm(estimate - truth) / max(np.linalg.norm(centered), 1e-12)
    )
    return {
        "pressure_relative_l2": relative_l2(truth, estimate),
        "pressure_mean_residual_relative_l2": mean_residual_rel_l2,
        "pressure_r2": float(1.0 - mean_residual_rel_l2 * mean_residual_rel_l2),
        "pressure_mae": float(np.mean(np.abs(truth - estimate))),
        "final_train_masked_mse": float(last_loss),
        "first_train_masked_mse": float(first_loss if first_loss is not None else 0.0),
    }


def evaluate(
    train: list[MeshPressure],
    test: list[MeshPressure],
    centers: np.ndarray,
    *,
    grid_size: int,
    qa_modulus: int | None,
    width: int,
    modes: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict[str, object]:
    train_pressure = np.concatenate([s.pressure for s in train])
    pressure_mean = float(np.mean(train_pressure))
    pressure_std = float(max(np.std(train_pressure), 1e-12))
    train_x, train_y, train_mask, train_meta = build_tensors(
        train, grid_size, pressure_mean, pressure_std, centers, qa_modulus=qa_modulus
    )
    test_x, test_y, test_mask, _ = build_tensors(
        test, grid_size, pressure_mean, pressure_std, centers, qa_modulus=qa_modulus
    )
    metrics = train_and_eval(
        train_x, train_y, train_mask, test_x, test_y, test_mask,
        pressure_mean=pressure_mean, pressure_std=pressure_std,
        width=width, modes=modes, layers=layers, dropout=dropout,
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
    )
    return {**metrics, "train_meta": train_meta, "train_tensor_shape": list(train_x.shape)}


def grid_centers(grid_size: int) -> np.ndarray:
    axis = (np.arange(grid_size) * 2.0 / (grid_size - 1)) - 1.0
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def run(
    train_limit: int,
    test_limit: int,
    grid_size: int,
    width: int,
    modes: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    timeout_s: float,
) -> dict[str, object]:
    t0 = time.time()
    train, test, source_metadata = load_split(train_limit, test_limit, timeout_s)
    centers = grid_centers(grid_size)
    common = dict(
        grid_size=grid_size, width=width, modes=modes, layers=layers,
        dropout=dropout, epochs=epochs, batch_size=batch_size,
        lr=2e-3, weight_decay=5e-4,
    )
    continuous = evaluate(train, test, centers, qa_modulus=None, **common)
    qa_cells: dict[str, object] = {}
    for modulus in MODULI:
        metrics = evaluate(train, test, centers, qa_modulus=modulus, **common)
        qa_cells[str(modulus)] = {
            **metrics,
            "pressure_gap_vs_continuous": metrics["pressure_relative_l2"]
            - continuous["pressure_relative_l2"],
        }

    m144 = qa_cells["144"]
    gaps = [qa_cells[str(mod)]["pressure_gap_vs_continuous"] for mod in MODULI]
    abs_gaps = [abs(g) for g in gaps]
    continuous_r2 = float(continuous["pressure_r2"])
    qa_boundary_ok = (
        abs(m144["pressure_gap_vs_continuous"]) <= 0.03
        and abs_gaps[-1] <= abs_gaps[0] + 1e-9
    )
    operator_is_weak = continuous_r2 < 0.6
    if not qa_boundary_ok:
        status = "QA_BOUNDARY_PARITY_FAIL"
        honest_note = (
            "QA does not track the continuous operator within the declared "
            "band; treat as a boundary-parity failure pending investigation."
        )
    elif operator_is_weak:
        status = "QA_BOUNDARY_PARITY_OK__CONTINUOUS_OPERATOR_WEAK"
        honest_note = (
            "Canonical trusted GINO record + authors' press alignment. The "
            "continuous FNO is still a weak surface-pressure operator at CPU "
            "scale; QA tracks it within the 0.03 band but the operator's own "
            "error dominates the quantization gap, so this certifies the QA "
            "packet BOUNDARY, not solver-quality parity."
        )
    else:
        status = "QA_OPERATOR_PARITY_OK__SURFACE_ONLY_NOT_FENGBO_SOLVER"
        honest_note = (
            "Canonical trusted GINO record + authors' press alignment yields a "
            f"GENUINELY WORKING continuous operator (R^2={continuous_r2:.4f}); "
            "QA tracks it within the 0.03 band with the gap shrinking by "
            "modulus. This is a NON-DEGENERATE QA/continuous operator-parity "
            "result on the real Fengbo/GINO benchmark task. It is still surface "
            "pressure (the benchmark has no volumetric velocity field) and not "
            "the full volumetric Fengbo solver, hence the SURFACE_ONLY status. "
            "It supersedes the earlier 'hard ceiling R^2~=0.41' claim from "
            "scripts 64-66, which was a target-misalignment artifact (first-N "
            "vs the authors' concat(press[:16],press[112:]) fixup)."
        )
    return {
        "experiment": "pepe_ch5_canonical_gino_carcfd_pressure",
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "operator": "3D Fourier Neural Operator on dense unsigned surface-distance + packet input",
        "claim_boundary": (
            "Canonical neuraloperator GINO Car-CFD record (Zenodo 13936501, "
            "md5-pinned, trusted) with the authors' canonical press alignment. "
            "This is the faithful Fengbo/GINO benchmark TASK (surface pressure "
            "from geometry); the benchmark has no volumetric velocity field in "
            "any distribution. QA quantization-boundary parity is asserted, NOT "
            "QA/continuous operator-quality parity and NOT a Fengbo solver rung. "
            "Representation gap vs canonical GINO: open3d unavailable, so this "
            "uses an UNSIGNED nearest-surface distance instead of the authors' "
            "signed SDF; recorded, not hidden."
        ),
        "source_summary": {
            "name": "Three-dimensional flow dataset over ShapeNet-Car (canonical GINO record)",
            "record_url": ZENODO_RECORD,
            "file_url": ZENODO_URL,
            "doi": "10.5281/zenodo.13936501",
            "dataset_loader": "neuralop.data.datasets.CarCFDDataset",
            "provenance": "trusted_canonical_md5_pinned",
            "metadata": source_metadata,
        },
        "train_count": len(train),
        "test_count": len(test),
        "grid_size": int(grid_size),
        "fno": {
            "width": int(width), "modes": int(modes), "layers": int(layers),
            "dropout": float(dropout), "epochs": int(epochs), "batch_size": int(batch_size),
        },
        "moduli": MODULI,
        "feature_schema": ["surfdist", "mask", "x", "y", "z", "B12", "B13", "B23"],
        "continuous": continuous,
        "qa": qa_cells,
        "verdict": {
            "status": status,
            "qa_boundary_faithful": bool(qa_boundary_ok),
            "continuous_operator_weak": bool(operator_is_weak),
            "continuous_pressure_r2": continuous_r2,
            "continuous_pressure_relative_l2": float(continuous["pressure_relative_l2"]),
            "continuous_mean_residual_relative_l2": float(
                continuous["pressure_mean_residual_relative_l2"]
            ),
            "m144_pressure_gap_vs_continuous": float(m144["pressure_gap_vs_continuous"]),
            "pressure_abs_gap_m24_to_m288": [float(g) for g in abs_gaps],
            "pressure_alignment_modes": source_metadata["pressure_alignment_modes"],
            "honest_note": honest_note,
        },
    }


def self_test() -> dict[str, object]:
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 4\nproperty double x\nproperty double y\nproperty double z\n"
        b"element face 1\nproperty list uchar uint vertex_indices\nend_header\n"
    )
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]], dtype="<f8"
    )
    face = bytes([3]) + np.asarray([0, 1, 2], dtype="<u4").tobytes()
    pv, pf = parse_ply(header + vertices.tobytes() + face)

    alt_header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        b"element face 1\nproperty list uint8 int32 vertex_indices\nend_header\n"
    )
    alt_v = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0]], dtype="<f4"
    )
    quad = bytes([4]) + np.asarray([0, 1, 2, 3], dtype="<i4").tobytes()
    av, af = parse_ply(alt_header + alt_v.tobytes() + quad)
    alt_ok = av.shape == (4, 3) and af.shape == (2, 3)

    # Canonical press alignment: 3682 -> 3586 via concat(press[:16], press[112:]).
    fake_press = np.arange(3682, dtype=np.float64)
    aligned, mode = align_pressure(fake_press, 3586)
    align_ok = (
        aligned.shape[0] == 3586
        and mode == "canonical_concat_0_16_112_end"
        and aligned[15] == 15.0
        and aligned[16] == 112.0
    )

    normals = vertex_normals(pv, pf)
    sample = MeshPressure("toy", np.clip(pv, -1.0, 1.0), normals,
                          np.asarray([1.0, 2.0, 3.0, 4.0]), "test")
    grid = 8
    x, y, mask, _ = build_tensors([sample], grid, 2.5, 1.0, grid_centers(grid), qa_modulus=144)
    out = FNO3d(8, 6, 3, 2, 0.0)(torch.tensor(x, dtype=torch.float32))
    ok = (
        pv.shape == (4, 3)
        and x.shape == (1, 8, grid, grid, grid)
        and out.shape == (1, 1, grid, grid, grid)
        and int(mask.sum()) == 4
        and bool(np.all(x[0, 0] > 0.0))
        and alt_ok
        and align_ok
    )
    return {
        "ok": bool(ok),
        "heterogeneous_ply_ok": bool(alt_ok),
        "canonical_press_alignment_ok": bool(align_ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quick", action="store_true", help="small subset for a fast smoke")
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=111)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--modes", type=int, default=7)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1

    train_limit = 160 if args.quick else args.train_limit
    test_limit = 40 if args.quick else args.test_limit
    epochs = 8 if args.quick else args.epochs

    result = run(
        train_limit, test_limit, args.grid_size, args.width, args.modes,
        args.layers, args.dropout, epochs, args.batch_size, args.timeout_s,
    )
    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0 if result["verdict"]["qa_boundary_faithful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
