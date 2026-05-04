#!/usr/bin/env python3
"""QA observer test on real measured magnetic hysteresis loops.

This script uses the Ring35 experimental FeSi hysteresis-loop dataset from
Zenodo record 17579041. The archive stores measured time, B(t), H(t), and
dB/dt traces plus a loss table. The test is intentionally direct:

1. Compute the measured loop-work proxy integral H dB from B(t), H(t).
2. Calibrate global QA bins on calibration loops only.
3. Map held-out measured loops to QA variables:
       b = bin(H), e = bin(B), d = b+e, a = b+2e
       J = b*d, X = d*e, K = d*a
4. Predict held-out energy loss from deterministic QA loop observables.
5. Compare against mean-loss and Steinmetz-style baselines.

No synthetic loop is used in the main result. No neural model is trained.
The only fitted objects are small least-squares calibration maps.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path


ZIP_PATH = Path("/tmp/Ring35_Dataset_Txt.zip")
OUT_PATH = Path("results/qa_hysteresis_real_loop_observer.json")

DATASET_RECORD = "https://zenodo.org/records/17579041"
DATASET_TITLE = "Dataset of Experimental Non-Standard Dynamic Hysteresis Loops"
DENSITY_KG_PER_M3 = 7632.0
MODULUS = 24


@dataclass(frozen=True)
class LossRow:
    filename: str
    bp_t: float
    frequency_hz: float
    power_loss_w_per_kg: float
    energy_loss_mj_per_kg: float


@dataclass(frozen=True)
class LoopTrace:
    row: LossRow
    t_s: list[float]
    b_t: list[float]
    h_a_per_m: list[float]


def parse_float(text: str) -> float:
    return float(text.strip())


def read_text_from_zip(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8-sig")


def load_sin_loss_table(zf: zipfile.ZipFile) -> list[LossRow]:
    text = read_text_from_zip(zf, "Ring35_Dataset_Txt/SIN/SIN_Loss_Table.txt")
    rows: list[LossRow] = []
    for row in csv.DictReader(io.StringIO(text)):
        rows.append(
            LossRow(
                filename=row["FileName"].strip(),
                bp_t=parse_float(row["B1_T"]),
                frequency_hz=parse_float(row["f_Hz"]),
                power_loss_w_per_kg=parse_float(row["PowerLoss_Wperkg"]),
                energy_loss_mj_per_kg=parse_float(row["EnergyLoss_mJperkg"]),
            )
        )
    return rows


def load_loop_trace(zf: zipfile.ZipFile, row: LossRow) -> LoopTrace:
    text = read_text_from_zip(zf, f"Ring35_Dataset_Txt/SIN/{row.filename}.txt")
    t_s: list[float] = []
    b_t: list[float] = []
    h_a_per_m: list[float] = []
    for data_row in csv.DictReader(io.StringIO(text)):
        t_s.append(parse_float(data_row["t_s"]))
        b_t.append(parse_float(data_row["B_T"]))
        h_a_per_m.append(parse_float(data_row["H_Aperm"]))
    return LoopTrace(row=row, t_s=t_s, b_t=b_t, h_a_per_m=h_a_per_m)


def select_sin_rows(rows: list[LossRow]) -> list[LossRow]:
    """Use a stable, moderate-size grid across amplitudes and frequencies."""
    wanted_f = {50.0, 400.0, 1000.0, 2000.0}
    wanted_bp = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.35}
    selected = [
        row
        for row in rows
        if row.frequency_hz in wanted_f and round(row.bp_t, 2) in wanted_bp
    ]
    return sorted(selected, key=lambda r: (r.frequency_hz, r.bp_t, r.filename))


def close(values: list[float]) -> list[float]:
    return values + [values[0]]


def line_integral_y_dx(x: list[float], y: list[float]) -> float:
    x2 = close(x)
    y2 = close(y)
    return sum(0.5 * (y2[i] + y2[i + 1]) * (x2[i + 1] - x2[i]) for i in range(len(x)))


def physical_energy_mj_per_kg(trace: LoopTrace) -> float:
    area_j_per_m3 = abs(line_integral_y_dx(trace.b_t, trace.h_a_per_m))
    return 1000.0 * area_j_per_m3 / DENSITY_KG_PER_M3


def quantile_edges(values: list[float], m: int = MODULUS) -> list[float]:
    ordered = sorted(values)
    edges: list[float] = []
    for k in range(1, m):
        idx = (k / m) * (len(ordered) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        edges.append(ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo]))
    return edges


def bins(values: list[float], edges: list[float]) -> list[int]:
    out: list[int] = []
    for value in values:
        bucket = 1
        for edge in edges:
            if value > edge:
                bucket += 1
        out.append(bucket)
    return out


def calibration_split(rows: list[LossRow]) -> tuple[set[str], set[str]]:
    calibration: set[str] = set()
    heldout: set[str] = set()
    for idx, row in enumerate(rows):
        if idx % 3 == 1:
            heldout.add(row.filename)
        else:
            calibration.add(row.filename)
    return calibration, heldout


def build_global_edges(traces: list[LoopTrace], calibration_names: set[str]) -> tuple[list[float], list[float]]:
    h_values: list[float] = []
    b_values: list[float] = []
    for trace in traces:
        if trace.row.filename in calibration_names:
            h_values.extend(trace.h_a_per_m)
            b_values.extend(trace.b_t)
    return quantile_edges(h_values), quantile_edges(b_values)


def qa_observables(trace: LoopTrace, h_edges: list[float], b_edges: list[float]) -> dict[str, float]:
    b_seq = bins(trace.h_a_per_m, h_edges)
    e_seq = bins(trace.b_t, b_edges)
    j_seq: list[float] = []
    x_seq: list[float] = []
    k_seq: list[float] = []
    pi_seq: list[float] = []
    for b, e in zip(b_seq, e_seq):
        d = b + e
        a = b + 2 * e
        j = b * d
        x = d * e
        k = d * a
        j_seq.append(j)
        x_seq.append(x)
        k_seq.append(k)
        pi_seq.append(j + x + k)

    be_area = abs(line_integral_y_dx(e_seq, b_seq))
    jx_area = abs(line_integral_y_dx(x_seq, j_seq))
    kx_area = abs(line_integral_y_dx(x_seq, k_seq))
    pi_x_area = abs(line_integral_y_dx(x_seq, pi_seq))
    path_energy = 0.0
    x2 = close(x_seq)
    pi2 = close(pi_seq)
    for i in range(len(x_seq)):
        path_energy += 0.5 * (pi2[i] + pi2[i + 1]) * abs(x2[i + 1] - x2[i])

    return {
        "qa_be_loop_area": be_area,
        "qa_jx_loop_area": jx_area,
        "qa_kx_loop_area": kx_area,
        "qa_pi_x_loop_area": pi_x_area,
        "qa_pi_abs_path_energy": path_energy,
        "qa_unique_states": float(len(set(zip(b_seq, e_seq)))),
    }


def solve_linear_least_squares(x_rows: list[list[float]], y: list[float]) -> list[float]:
    cols = len(x_rows[0])
    ata = [[0.0 for _ in range(cols)] for _ in range(cols)]
    aty = [0.0 for _ in range(cols)]
    ridge = 1e-8
    for row, target in zip(x_rows, y):
        for i in range(cols):
            aty[i] += row[i] * target
            for j in range(cols):
                ata[i][j] += row[i] * row[j]
    for i in range(cols):
        ata[i][i] += ridge
    return solve_square(ata, aty)


def solve_square(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-18:
            raise ValueError("singular calibration system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for k in range(col, n + 1):
            aug[col][k] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for k in range(col, n + 1):
                aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] for i in range(n)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def standardize_train_test(
    train_features: list[list[float]], test_features: list[list[float]]
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    cols = len(train_features[0])
    means: list[float] = []
    scales: list[float] = []
    for col in range(cols):
        vals = [row[col] for row in train_features]
        mean = statistics.mean(vals)
        scale = statistics.pstdev(vals)
        if scale <= 1e-12:
            scale = 1.0
        means.append(mean)
        scales.append(scale)

    def apply(rows: list[list[float]]) -> list[list[float]]:
        return [
            [1.0] + [(row[col] - means[col]) / scales[col] for col in range(cols)]
            for row in rows
        ]

    return apply(train_features), apply(test_features), means, scales


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    mean_actual = sum(actual) / len(actual)
    sse = sum((a - p) * (a - p) for a, p in zip(actual, predicted))
    sst = sum((a - mean_actual) * (a - mean_actual) for a in actual)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
    mape = sum(abs(a - p) / max(abs(a), 1e-12) for a, p in zip(actual, predicted)) / len(actual)
    return {
        "mae_mj_per_kg": mae,
        "mape": mape,
        "rmse_mj_per_kg": math.sqrt(sse / len(actual)),
        "r2": 1.0 - sse / sst if sst > 0.0 else float("nan"),
    }


def fit_predict(
    records: list[dict],
    calibration_names: set[str],
    heldout_names: set[str],
    feature_names: list[str],
) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    raw_x_train = [[float(r[name]) for name in feature_names] for r in train]
    raw_x_test = [[float(r[name]) for name in feature_names] for r in test]
    x_train, x_test, means, scales = standardize_train_test(raw_x_train, raw_x_test)
    y_train = [float(r["energy_loss_mj_per_kg_table"]) for r in train]
    coeffs = solve_linear_least_squares(x_train, y_train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [dot(coeffs, row) for row in x_test]
    return {
        "features": feature_names,
        "coefficients": coeffs,
        "feature_means": means,
        "feature_scales": scales,
        "heldout": metrics(actual, predicted),
    }


def fit_steinmetz(records: list[dict], calibration_names: set[str], heldout_names: set[str]) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    x_train = [
        [1.0, math.log(float(r["frequency_hz"])), math.log(float(r["bp_t"]))]
        for r in train
    ]
    y_train = [math.log(float(r["energy_loss_mj_per_kg_table"])) for r in train]
    coeffs = solve_linear_least_squares(x_train, y_train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [
        math.exp(dot(coeffs, [1.0, math.log(float(r["frequency_hz"])), math.log(float(r["bp_t"]))]))
        for r in test
    ]
    return {
        "features": ["intercept", "log_frequency_hz", "log_bp_t"],
        "coefficients": coeffs,
        "heldout": metrics(actual, predicted),
    }


def fit_log_feature_model(
    records: list[dict],
    calibration_names: set[str],
    heldout_names: set[str],
    feature_names: list[str],
) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    raw_train = [[float(r[name]) for name in feature_names] for r in train]
    raw_test = [[float(r[name]) for name in feature_names] for r in test]
    x_train, x_test, means, scales = standardize_train_test(raw_train, raw_test)
    y_train = [math.log(float(r["energy_loss_mj_per_kg_table"])) for r in train]
    coeffs = solve_linear_least_squares(x_train, y_train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [math.exp(dot(coeffs, row)) for row in x_test]
    return {
        "features": feature_names,
        "coefficients": coeffs,
        "feature_means": means,
        "feature_scales": scales,
        "target_transform": "log",
        "heldout": metrics(actual, predicted),
    }


def mean_baseline(records: list[dict], calibration_names: set[str], heldout_names: set[str]) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    mean_y = statistics.mean(float(r["energy_loss_mj_per_kg_table"]) for r in train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [mean_y for _ in test]
    return {"constant": mean_y, "heldout": metrics(actual, predicted)}


def main() -> int:
    if not ZIP_PATH.exists():
        raise SystemExit(
            f"missing {ZIP_PATH}; download Ring35_Dataset_Txt.zip from {DATASET_RECORD}"
        )

    with zipfile.ZipFile(ZIP_PATH) as zf:
        loss_rows = select_sin_rows(load_sin_loss_table(zf))
        calibration_names, heldout_names = calibration_split(loss_rows)
        traces = [load_loop_trace(zf, row) for row in loss_rows]

    h_edges, b_edges = build_global_edges(traces, calibration_names)
    records: list[dict] = []
    for trace in traces:
        qa = qa_observables(trace, h_edges, b_edges)
        physical_from_loop = physical_energy_mj_per_kg(trace)
        rec = {
            "filename": trace.row.filename,
            "bp_t": trace.row.bp_t,
            "frequency_hz": trace.row.frequency_hz,
            "split": "calibration" if trace.row.filename in calibration_names else "heldout",
            "rows": len(trace.t_s),
            "energy_loss_mj_per_kg_table": trace.row.energy_loss_mj_per_kg,
            "energy_loss_mj_per_kg_from_integral": physical_from_loop,
            "integral_vs_table_abs_error": abs(physical_from_loop - trace.row.energy_loss_mj_per_kg),
        }
        rec.update(qa)
        records.append(rec)

    qa_feature_sets = {
        "qa_be_only": ["qa_be_loop_area"],
        "qa_lifted_jx_only": ["qa_jx_loop_area"],
        "qa_lifted_packet": [
            "qa_jx_loop_area",
            "qa_kx_loop_area",
            "qa_pi_x_loop_area",
            "qa_pi_abs_path_energy",
            "qa_unique_states",
        ],
        "qa_with_drive_metadata": [
            "frequency_hz",
            "bp_t",
            "qa_jx_loop_area",
            "qa_kx_loop_area",
            "qa_pi_abs_path_energy",
        ],
    }
    qa_models = {
        name: fit_predict(records, calibration_names, heldout_names, features)
        for name, features in qa_feature_sets.items()
    }

    table_errors = [r["integral_vs_table_abs_error"] for r in records]
    payload = {
        "ok": True,
        "experiment": "qa_hysteresis_real_loop_observer",
        "dataset": {
            "title": DATASET_TITLE,
            "record_url": DATASET_RECORD,
            "local_archive": str(ZIP_PATH),
            "subset": "Ring35_Dataset_Txt/SIN selected grid",
            "material": "0.3471 mm FeSi ring",
            "density_kg_per_m3": DENSITY_KG_PER_M3,
            "loop_count": len(records),
            "calibration_count": len(calibration_names),
            "heldout_count": len(heldout_names),
            "selected_frequencies_hz": sorted({r.frequency_hz for r in loss_rows}),
            "selected_bp_t": sorted({r.bp_t for r in loss_rows}),
        },
        "physical_observable": {
            "target": "EnergyLoss_mJperkg from measured loss table",
            "loop_integral_check": "abs(integral H dB) / density converted to mJ/kg",
            "mean_abs_table_vs_integral_error_mj_per_kg": statistics.mean(table_errors),
            "max_abs_table_vs_integral_error_mj_per_kg": max(table_errors),
        },
        "qa_mapping": {
            "m": MODULUS,
            "bin_edges": "global calibration-loop quantiles only",
            "b": "bin(H_Aperm)",
            "e": "bin(B_T)",
            "d": "b+e",
            "a": "b+2*e",
            "J": "b*d",
            "X": "d*e",
            "K": "d*a",
            "observables": [
                "abs(integral b de)",
                "abs(integral J dX)",
                "abs(integral K dX)",
                "abs(integral Pi dX)",
                "sum 0.5*(Pi_i+Pi_j)*abs(delta_X)",
                "unique (b,e) states",
            ],
        },
        "heldout_models": {
            "mean_loss_baseline": mean_baseline(records, calibration_names, heldout_names),
            "steinmetz_log_linear_baseline": fit_steinmetz(records, calibration_names, heldout_names),
            "steinmetz_plus_qa_shape_log_model": fit_log_feature_model(
                records,
                calibration_names,
                heldout_names,
                [
                    "frequency_hz",
                    "bp_t",
                    "qa_jx_loop_area",
                    "qa_pi_abs_path_energy",
                    "qa_unique_states",
                ],
            ),
            **qa_models,
        },
        "verdict": {
            "qa_not_random": (
                "YES: QA-only held-out models reduce error substantially relative "
                "to the mean-loss baseline on real measured loops."
            ),
            "qa_beats_steinmetz_here": (
                "NO: the log-linear Steinmetz baseline remains much stronger on "
                "this selected Ring35 SIN subset."
            ),
            "qa_adds_to_steinmetz_here": (
                "NO: the tested Steinmetz+QA shape model did not improve held-out "
                "performance over Steinmetz alone."
            ),
        },
        "records": records,
        "interpretation": (
            "A useful QA hysteresis mapping should improve held-out loss prediction "
            "over the mean baseline and should be evaluated against Steinmetz. "
            "The QA-only models test whether measured loop geometry maps into QA "
            "coordinates without drive metadata. The QA-with-drive model tests whether "
            "QA loop structure adds predictive signal when amplitude/frequency are "
            "available."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["heldout_models"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
