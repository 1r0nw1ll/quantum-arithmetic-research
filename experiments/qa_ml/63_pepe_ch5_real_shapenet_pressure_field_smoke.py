"""Real ShapeNet-Car pressure-field smoke for QA-Fengbo mapping.

This is the first field-level rung after the AhmedML metadata/force smoke. It
uses the public processed ShapeNet-Car pressure archive from Zenodo and trains
matched continuous vs QA-quantized geometry-to-pressure-field operators on the
same official split.

The operator is intentionally small: mesh geometry descriptors -> PCA pressure
field coefficients -> reconstructed pressure field. It is not a full Fengbo
Clifford-FNO reproduction and it does not cover velocity. The claim is only
that QA quantization of the real geometry boundary preserves a matched
pressure-field baseline on a small real-data subset.

QA_COMPLIANCE = "real_shapenet_pressure_field_smoke - exact int geometry quantization; observer decode for pressure-field metrics"
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.qa_ml.qa_cga_grid_packet_v1 import dequantize_unit, quantize_unit, relative_l2


OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_real_shapenet_pressure_field_smoke.json"
CACHE_DIR = Path(__file__).resolve().parent / "real_fengbo_shapenet_car_cache"
ZIP_PATH = CACHE_DIR / "processed-car-pressure-data.zip"
ZENODO_URL = "https://zenodo.org/records/13737721/files/processed-car-pressure-data.zip"
ZENODO_RECORD = "https://zenodo.org/records/13737721"
EXPECTED_MD5 = "05153df0bd3aacdbee4a42eb00074af8"
ROOT_IN_ZIP = "processed-car-pressure-data"
MODULI = [24, 48, 72, 144, 288]


@dataclass(frozen=True)
class CarSample:
    car_id: str
    descriptors: np.ndarray
    pressure: np.ndarray


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ensure_archive(timeout_s: float) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        return
    request = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "qa-ml-shapenet-pressure-smoke/1.0"})
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


def parse_ply_vertices(raw: bytes) -> np.ndarray:
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError("PLY header missing end_header")
    header_end = end + len(b"end_header\n")
    header = raw[:header_end].decode("utf-8", errors="replace")
    vertex_match = re.search(r"element vertex (\d+)", header)
    if vertex_match is None:
        raise ValueError("PLY header missing vertex count")
    n_vertices = int(vertex_match.group(1))
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("expected binary little endian PLY")
    if "property double x" not in header or "property double y" not in header or "property double z" not in header:
        raise ValueError("expected double x/y/z vertex properties")
    n_values = n_vertices * 3
    vertices = np.frombuffer(raw, dtype="<f8", count=n_values, offset=header_end)
    if vertices.size != n_values:
        raise ValueError("truncated PLY vertex block")
    return vertices.reshape(n_vertices, 3).astype(np.float64, copy=True)


def mesh_descriptors(vertices: np.ndarray) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = maxs - mins
    center = vertices.mean(axis=0)
    centered = vertices - center
    rms = np.sqrt(np.mean(centered * centered, axis=0))
    abs_mean = np.mean(np.abs(centered), axis=0)
    q10 = np.quantile(vertices, 0.10, axis=0)
    q25 = np.quantile(vertices, 0.25, axis=0)
    q50 = np.quantile(vertices, 0.50, axis=0)
    q75 = np.quantile(vertices, 0.75, axis=0)
    q90 = np.quantile(vertices, 0.90, axis=0)
    cov = (centered.T @ centered) / max(1, vertices.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    volume_box = np.asarray([span[0] * span[1] * span[2]], dtype=np.float64)
    diagonal = np.asarray([np.sqrt(np.sum(span * span))], dtype=np.float64)
    return np.concatenate(
        [mins, maxs, span, center, rms, abs_mean, q10, q25, q50, q75, q90, eigvals, volume_box, diagonal]
    )


def read_sample(zf: zipfile.ZipFile, car_id: str) -> CarSample:
    mesh_raw = zf.read(f"{ROOT_IN_ZIP}/data/mesh_{car_id}.ply")
    pressure_raw = zf.read(f"{ROOT_IN_ZIP}/data/press_{car_id}.npy")
    vertices = parse_ply_vertices(mesh_raw)
    pressure = np.load(io.BytesIO(pressure_raw)).astype(np.float64, copy=False)
    return CarSample(car_id=car_id, descriptors=mesh_descriptors(vertices), pressure=pressure)


def load_samples(train_limit: int, test_limit: int, timeout_s: float) -> tuple[list[CarSample], list[CarSample], dict[str, object]]:
    ensure_archive(timeout_s)
    digest = md5_file(ZIP_PATH)
    if digest != EXPECTED_MD5:
        raise ValueError(f"unexpected archive md5 {digest}; expected {EXPECTED_MD5}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        train_ids = read_manifest(zf, "train")[:train_limit]
        test_ids = read_manifest(zf, "test")[:test_limit]
        train = [read_sample(zf, car_id) for car_id in train_ids]
        test = [read_sample(zf, car_id) for car_id in test_ids]
        bounds = zf.read(f"{ROOT_IN_ZIP}/watertight_global_bounds.txt").decode("utf-8").strip()
    metadata = {
        "archive_path": str(ZIP_PATH),
        "archive_size_bytes": ZIP_PATH.stat().st_size,
        "archive_md5": digest,
        "source_url": ZENODO_URL,
        "source_record": ZENODO_RECORD,
        "watertight_global_bounds": bounds,
    }
    return train, test, metadata


def descriptor_matrix(samples: list[CarSample]) -> np.ndarray:
    return np.vstack([sample.descriptors for sample in samples])


def pressure_matrix(samples: list[CarSample]) -> np.ndarray:
    return np.vstack([sample.pressure for sample in samples])


def minmax_scale(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lo = train_x.min(axis=0)
    hi = train_x.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    train_unit = np.clip(2.0 * (train_x - lo) / span - 1.0, -1.0, 1.0)
    test_unit = np.clip(2.0 * (test_x - lo) / span - 1.0, -1.0, 1.0)
    return train_unit, test_unit, lo, hi


def qa_quantize_matrix(unit_x: np.ndarray, modulus: int) -> np.ndarray:
    out = np.empty_like(unit_x, dtype=np.float64)
    for row_idx in range(unit_x.shape[0]):
        for col_idx in range(unit_x.shape[1]):
            q = quantize_unit(float(unit_x[row_idx, col_idx]), modulus)
            out[row_idx, col_idx] = dequantize_unit(q, modulus)
    return out


def fit_pressure_operator(train_x: np.ndarray, train_pressure: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    train_coeff = pca.fit_transform(train_pressure)
    regressor = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1e-2),
    ).fit(train_x, train_coeff)
    return pca, regressor


def evaluate_pressure(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_pressure: np.ndarray,
    test_pressure: np.ndarray,
    *,
    n_components: int,
) -> dict[str, object]:
    pca, regressor = fit_pressure_operator(train_x, train_pressure, n_components)
    pred_coeff = regressor.predict(test_x)
    pred_pressure = pca.inverse_transform(pred_coeff)
    per_sample = [
        relative_l2(test_pressure[idx], pred_pressure[idx])
        for idx in range(test_pressure.shape[0])
    ]
    return {
        "pressure_relative_l2": relative_l2(test_pressure, pred_pressure),
        "pressure_mean_sample_relative_l2": float(np.mean(per_sample)),
        "pressure_median_sample_relative_l2": float(np.median(per_sample)),
        "pressure_max_sample_relative_l2": float(np.max(per_sample)),
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
    }


def run(train_limit: int, test_limit: int, n_components: int, timeout_s: float) -> dict[str, object]:
    t0 = time.time()
    train, test, source_metadata = load_samples(train_limit, test_limit, timeout_s)
    train_x_raw = descriptor_matrix(train)
    test_x_raw = descriptor_matrix(test)
    train_pressure = pressure_matrix(train)
    test_pressure = pressure_matrix(test)
    train_x, test_x, desc_min, desc_max = minmax_scale(train_x_raw, test_x_raw)

    n_components_eff = min(n_components, train_pressure.shape[0], train_pressure.shape[1])
    continuous = evaluate_pressure(train_x, test_x, train_pressure, test_pressure, n_components=n_components_eff)
    qa_cells = {}
    for modulus in MODULI:
        train_q = qa_quantize_matrix(train_x, modulus)
        test_q = qa_quantize_matrix(test_x, modulus)
        metrics = evaluate_pressure(train_q, test_q, train_pressure, test_pressure, n_components=n_components_eff)
        qa_cells[str(modulus)] = {
            **metrics,
            "pressure_gap_vs_continuous": metrics["pressure_relative_l2"] - continuous["pressure_relative_l2"],
            "mean_sample_gap_vs_continuous": metrics["pressure_mean_sample_relative_l2"] - continuous["pressure_mean_sample_relative_l2"],
        }

    m144 = qa_cells["144"]
    gaps = [qa_cells[str(modulus)]["pressure_gap_vs_continuous"] for modulus in MODULI]
    abs_gaps = [abs(gap) for gap in gaps]
    pass_smoke = (
        abs(m144["pressure_gap_vs_continuous"]) <= 0.03
        and m144["pressure_relative_l2"] <= continuous["pressure_relative_l2"] + 0.03
        and abs_gaps[-1] <= abs_gaps[0] + 1e-12
    )
    return {
        "experiment": "pepe_ch5_real_shapenet_pressure_field_smoke",
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "claim_boundary": "Real ShapeNet-Car pressure-field smoke only; not full Fengbo Clifford-FNO and no velocity-field claim.",
        "source_summary": {
            "name": "Three-dimensional flow dataset over ShapeNet-Car",
            "record_url": ZENODO_RECORD,
            "file_url": ZENODO_URL,
            "doi": "10.5281/zenodo.13737721",
            "metadata": source_metadata,
        },
        "train_count": len(train),
        "test_count": len(test),
        "train_ids": [sample.car_id for sample in train],
        "test_ids": [sample.car_id for sample in test],
        "pressure_vector_length": int(train_pressure.shape[1]),
        "descriptor_count": int(train_x.shape[1]),
        "n_components": int(n_components_eff),
        "moduli": MODULI,
        "continuous": continuous,
        "qa": qa_cells,
        "descriptor_min_first8": [float(v) for v in desc_min[:8]],
        "descriptor_max_first8": [float(v) for v in desc_max[:8]],
        "verdict": {
            "status": "PASS_REAL_PRESSURE_FIELD_SMOKE" if pass_smoke else "FAIL_REAL_PRESSURE_FIELD_SMOKE",
            "m144_pressure_relative_l2": float(m144["pressure_relative_l2"]),
            "continuous_pressure_relative_l2": float(continuous["pressure_relative_l2"]),
            "m144_pressure_gap_vs_continuous": float(m144["pressure_gap_vs_continuous"]),
            "pressure_gap_m24_to_m288": [float(gap) for gap in gaps],
            "pressure_abs_gap_m24_to_m288": [float(gap) for gap in abs_gaps],
            "success_criterion": "m=144 pressure relative-L2 within +0.03 of continuous and abs(m=288 gap) no worse than abs(m=24 gap)",
        },
    }


def self_test() -> dict[str, object]:
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 2\nproperty double x\nproperty double y\nproperty double z\n"
        b"element face 0\nproperty list uchar uint vertex_indices\nend_header\n"
    )
    vertices = np.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype="<f8")
    parsed = parse_ply_vertices(header + vertices.tobytes())
    desc = mesh_descriptors(parsed)
    q = quantize_unit(0.25, 144)
    ok = (
        parsed.shape == (2, 3)
        and desc.shape[0] == 38
        and abs(dequantize_unit(q, 144) - 0.25) < 0.01
        and parse_ids("001,002\n003") == ["001", "002", "003"]
    )
    return {"ok": bool(ok), "descriptor_count": int(desc.shape[0])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1

    result = run(args.train_limit, args.test_limit, args.n_components, args.timeout_s)
    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0 if result["verdict"]["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
