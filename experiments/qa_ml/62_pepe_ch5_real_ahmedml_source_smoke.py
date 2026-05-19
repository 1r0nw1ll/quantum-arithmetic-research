"""Real AhmedML subset smoke for QA-Fengbo mapping.

This is the real-data gate after synthetic packet/operator parity. It uses
public AhmedML per-run CSV files:

  - geo_parameters_i.csv: real Ahmed-body geometry parameters
  - force_mom_i.csv: Cd/Cl force coefficients

It does NOT claim full Fengbo pressure/velocity-field replication. The goal is
smaller and stricter: verify that QA quantization of real Fengbo-adjacent
geometry metadata preserves a matched continuous geometry-to-force operator.

QA_COMPLIANCE = "real_ahmedml_smoke - exact integer geometry quantization; observer decode for regression metrics"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.qa_ml.qa_cga_grid_packet_v1 import dequantize_unit, quantize_unit, relative_l2


OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_real_ahmedml_source_smoke.json"
CACHE_DIR = Path(__file__).resolve().parent / "real_fengbo_ahmedml_cache"
BASE_URL = "https://huggingface.co/datasets/neashton/ahmedml/resolve/main"
RUN_IDS = list(range(1, 65))
MODULI = [24, 48, 72, 144, 288]


@dataclass(frozen=True)
class AhmedRun:
    run_id: int
    geometry: dict[str, float]
    target: dict[str, float]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_single_row_csv(text: str) -> dict[str, float]:
    reader = csv.DictReader(io.StringIO(text.strip()))
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one CSV row, got {len(rows)}")
    return {str(key).strip(): float(value) for key, value in rows[0].items()}


def fetch_text(url: str, timeout_s: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "qa-ml-ahmedml-smoke/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def cached_fetch(run_id: int, kind: str, timeout_s: float) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rel_name = f"run_{run_id}/{kind}_{run_id}.csv"
    cache_path = CACHE_DIR / rel_name
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    text = fetch_text(f"{BASE_URL}/{rel_name}", timeout_s)
    cache_path.write_text(text, encoding="utf-8")
    return text


def load_run(run_id: int, timeout_s: float) -> AhmedRun:
    geometry = parse_single_row_csv(cached_fetch(run_id, "geo_parameters", timeout_s))
    target = parse_single_row_csv(cached_fetch(run_id, "force_mom", timeout_s))
    return AhmedRun(run_id=run_id, geometry=geometry, target=target)


def load_runs(run_ids: list[int], timeout_s: float) -> list[AhmedRun]:
    runs = []
    failures = []
    for run_id in run_ids:
        try:
            runs.append(load_run(run_id, timeout_s))
        except Exception as exc:  # pragma: no cover - network/data availability path
            failures.append({"run_id": run_id, "error": str(exc)})
    if len(runs) < 24:
        raise RuntimeError(f"too few AhmedML runs loaded: {len(runs)} successes, failures={failures[:3]}")
    return runs


def geometry_matrix(runs: list[AhmedRun]) -> tuple[list[str], np.ndarray]:
    keys = sorted(runs[0].geometry)
    values = np.asarray([[run.geometry[key] for key in keys] for run in runs], dtype=np.float64)
    return keys, values


def target_matrix(runs: list[AhmedRun]) -> tuple[list[str], np.ndarray]:
    keys = ["cd", "cl"]
    values = np.asarray([[run.target[key] for key in keys] for run in runs], dtype=np.float64)
    return keys, values


def train_test_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_rows)
    test = indices[indices % 4 == 0]
    train = indices[indices % 4 != 0]
    return train, test


def minmax_scale(train_x: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.min(train_x, axis=0)
    hi = np.max(train_x, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    scaled = 2.0 * (all_x - lo) / span - 1.0
    return np.clip(scaled, -1.0, 1.0), lo, hi


def qa_quantize_matrix(unit_x: np.ndarray, modulus: int) -> np.ndarray:
    out = np.empty_like(unit_x, dtype=np.float64)
    for row_idx in range(unit_x.shape[0]):
        for col_idx in range(unit_x.shape[1]):
            q = quantize_unit(float(unit_x[row_idx, col_idx]), modulus)
            out[row_idx, col_idx] = dequantize_unit(q, modulus)
    return out


def fit_regressor(train_x: np.ndarray, train_y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1e-3),
    ).fit(train_x, train_y)


def evaluate(unit_x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, object]:
    model = fit_regressor(unit_x[train_idx], y[train_idx])
    pred = model.predict(unit_x[test_idx])
    cd_l2 = relative_l2(y[test_idx, 0], pred[:, 0])
    cl_l2 = relative_l2(y[test_idx, 1], pred[:, 1])
    total_l2 = relative_l2(y[test_idx], pred)
    mae = np.mean(np.abs(y[test_idx] - pred), axis=0)
    return {
        "cd_relative_l2": float(cd_l2),
        "cl_relative_l2": float(cl_l2),
        "joint_relative_l2": float(total_l2),
        "cd_mae": float(mae[0]),
        "cl_mae": float(mae[1]),
    }


def summarize_sources() -> dict[str, object]:
    return {
        "primary_smoke_source": {
            "name": "AhmedML CAE benchmark on Hugging Face",
            "url": "https://huggingface.co/datasets/neashton/ahmedml",
            "files_used": ["run_i/geo_parameters_i.csv", "run_i/force_mom_i.csv"],
        },
        "field_replication_candidates": [
            {
                "name": "NVIDIA PhysicsNeMo Ahmed Body",
                "url": "https://huggingface.co/datasets/nvidia/PhysicsNeMo-CFD-Ahmed-Body",
                "status": "candidate for heavier field/mesh acquisition",
            },
            {
                "name": "GINO ShapeNet-Car/FNO data path",
                "url": "https://github.com/neuraloperator/GINO",
                "status": "candidate for full pressure/velocity Fengbo-style field replication",
            },
        ],
    }


def run(run_ids: list[int], timeout_s: float) -> dict[str, object]:
    t0 = time.time()
    runs = load_runs(run_ids, timeout_s)
    geometry_keys, x_raw = geometry_matrix(runs)
    target_keys, y = target_matrix(runs)
    train_idx, test_idx = train_test_indices(len(runs))
    x_unit, train_min, train_max = minmax_scale(x_raw[train_idx], x_raw)

    continuous = evaluate(x_unit, y, train_idx, test_idx)
    qa_cells = {}
    for modulus in MODULI:
        x_qa = qa_quantize_matrix(x_unit, modulus)
        metrics = evaluate(x_qa, y, train_idx, test_idx)
        qa_cells[str(modulus)] = {
            **metrics,
            "joint_gap_vs_continuous": metrics["joint_relative_l2"] - continuous["joint_relative_l2"],
            "cd_gap_vs_continuous": metrics["cd_relative_l2"] - continuous["cd_relative_l2"],
            "cl_gap_vs_continuous": metrics["cl_relative_l2"] - continuous["cl_relative_l2"],
        }

    gaps = [qa_cells[str(modulus)]["joint_gap_vs_continuous"] for modulus in MODULI]
    m144 = qa_cells["144"]
    pass_smoke = (
        m144["joint_gap_vs_continuous"] <= 0.025
        and m144["cd_gap_vs_continuous"] <= 0.025
        and m144["cl_gap_vs_continuous"] <= 0.050
        and gaps[-1] <= gaps[0]
    )
    return {
        "experiment": "pepe_ch5_real_ahmedml_source_smoke",
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "claim_boundary": "Real AhmedML geometry metadata and Cd/Cl force-coefficient smoke only; not full pressure/velocity Fengbo field replication.",
        "source_summary": summarize_sources(),
        "run_count": len(runs),
        "run_ids": [run.run_id for run in runs],
        "geometry_keys": geometry_keys,
        "target_keys": target_keys,
        "train_count": int(train_idx.size),
        "test_count": int(test_idx.size),
        "train_min": {key: float(value) for key, value in zip(geometry_keys, train_min)},
        "train_max": {key: float(value) for key, value in zip(geometry_keys, train_max)},
        "moduli": MODULI,
        "continuous": continuous,
        "qa": qa_cells,
        "verdict": {
            "status": "PASS_REAL_METADATA_FORCE_SMOKE" if pass_smoke else "FAIL_REAL_METADATA_FORCE_SMOKE",
            "m144_joint_gap_vs_continuous": float(m144["joint_gap_vs_continuous"]),
            "m144_cd_gap_vs_continuous": float(m144["cd_gap_vs_continuous"]),
            "m144_cl_gap_vs_continuous": float(m144["cl_gap_vs_continuous"]),
            "joint_gap_m24_to_m288": [float(gap) for gap in gaps],
            "success_criterion": "m=144 joint gap <= 0.025, Cd gap <= 0.025, Cl gap <= 0.050, and m=288 gap no worse than m=24",
        },
    }


def self_test() -> dict[str, object]:
    geometry_text = """body-length,body-height,body-width,front-arc-diameter,slant-angle-length,slant-angle-height,slant-surface-length,slant-angle-degrees
1187.9759519038075,253.38677354709418,369.3386773547094,194.30861723446895,95.92849345377991,174.52137364637474,199.14815016898712,61.203893601097136
"""
    force_text = """cd, cl
2.3848566658e-01, -9.4516081781e-02
"""
    geometry = parse_single_row_csv(geometry_text)
    force = parse_single_row_csv(force_text)
    q = quantize_unit(0.25, 144)
    decoded = dequantize_unit(q, 144)
    ok = (
        len(geometry) == 8
        and abs(force["cd"] - 0.23848566658) < 1e-12
        and abs(decoded - 0.25) < 0.01
    )
    return {"ok": bool(ok), "decoded_quarter_m144": float(decoded)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--runs", type=int, default=len(RUN_IDS))
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1

    run_ids = RUN_IDS[: args.runs]
    result = run(run_ids, args.timeout_s)
    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0 if result["verdict"]["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
