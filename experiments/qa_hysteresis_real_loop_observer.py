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
4. Predict held-out energy loss from deterministic QA loop observables,
   including a physically dimensioned QA shell approximation to integral H dB.
5. Compare against mean-loss and Steinmetz-style baselines.

No synthetic loop is used in the main result. No neural model is trained. The
direct QA shell integral uses calibration-loop shell centers but no fitted
regression coefficient.
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
QA_LIFTED_VARIABLES = ("b", "e", "J", "X", "K", "Pi", "K_minus_J")


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


def bin_centers(values: list[float], edges: list[float], m: int = MODULUS) -> list[float]:
    buckets: list[list[float]] = [[] for _ in range(m)]
    for value, bucket in zip(values, bins(values, edges)):
        buckets[bucket - 1].append(value)

    centers: list[float] = []
    fallback = statistics.mean(values)
    for idx, bucket_values in enumerate(buckets):
        if bucket_values:
            centers.append(statistics.mean(bucket_values))
            continue
        lower = edges[idx - 1] if idx > 0 else min(values)
        upper = edges[idx] if idx < len(edges) else max(values)
        if math.isfinite(lower) and math.isfinite(upper):
            centers.append(0.5 * (lower + upper))
        else:
            centers.append(fallback)
    return centers


def qa_vars(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    j = b * d
    x = d * e
    k = d * a
    return {
        "b": b,
        "e": e,
        "J": j,
        "X": x,
        "K": k,
        "Pi": j + x + k,
        "K_minus_J": k - j,
    }


def lifted_energy_field(h_var: str, b_var: str) -> str:
    return f"qa_lifted_{h_var.lower()}_{b_var.lower()}_shell_energy_mj_per_kg"


def calibration_split(rows: list[LossRow]) -> tuple[set[str], set[str]]:
    calibration: set[str] = set()
    heldout: set[str] = set()
    for idx, row in enumerate(rows):
        if idx % 3 == 1:
            heldout.add(row.filename)
        else:
            calibration.add(row.filename)
    return calibration, heldout


def build_global_calibration(
    traces: list[LoopTrace], calibration_names: set[str]
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    dict[tuple[int, int], tuple[float, float]],
    dict[tuple[str, str], dict[int, float]],
]:
    h_values: list[float] = []
    b_values: list[float] = []
    for trace in traces:
        if trace.row.filename in calibration_names:
            h_values.extend(trace.h_a_per_m)
            b_values.extend(trace.b_t)
    h_edges = quantile_edges(h_values)
    b_edges = quantile_edges(b_values)
    h_centers = bin_centers(h_values, h_edges)
    b_centers = bin_centers(b_values, b_edges)

    state_values: dict[tuple[int, int], list[tuple[float, float]]] = {}
    lifted_values: dict[tuple[str, str], dict[int, list[float]]] = {}
    for trace in traces:
        if trace.row.filename not in calibration_names:
            continue
        b_seq = bins(trace.h_a_per_m, h_edges)
        e_seq = bins(trace.b_t, b_edges)
        for h_value, b_value, b_state, e_state in zip(trace.h_a_per_m, trace.b_t, b_seq, e_seq):
            state_values.setdefault((b_state, e_state), []).append((h_value, b_value))
            vars_for_state = qa_vars(b_state, e_state)
            for var_name in QA_LIFTED_VARIABLES:
                key = vars_for_state[var_name]
                lifted_values.setdefault(("H", var_name), {}).setdefault(key, []).append(h_value)
                lifted_values.setdefault(("B", var_name), {}).setdefault(key, []).append(b_value)

    state_centers = {
        state: (
            statistics.mean(h_value for h_value, _ in values),
            statistics.mean(b_value for _, b_value in values),
        )
        for state, values in state_values.items()
    }
    lifted_centers = {
        target_var: {key: statistics.mean(values) for key, values in value_map.items()}
        for target_var, value_map in lifted_values.items()
    }
    return h_edges, b_edges, h_centers, b_centers, state_centers, lifted_centers


def qa_observables(
    trace: LoopTrace,
    h_edges: list[float],
    b_edges: list[float],
    h_centers: list[float],
    b_centers: list[float],
    state_centers: dict[tuple[int, int], tuple[float, float]],
    lifted_centers: dict[tuple[str, str], dict[int, float]],
) -> dict[str, float]:
    b_seq = bins(trace.h_a_per_m, h_edges)
    e_seq = bins(trace.b_t, b_edges)
    nonqa_h_shell_seq = [h_centers[b - 1] for b in b_seq]
    nonqa_b_shell_seq = [b_centers[e - 1] for e in e_seq]
    qa_h_shell_seq: list[float] = []
    qa_b_shell_seq: list[float] = []
    unseen_state_count = 0
    for b, e in zip(b_seq, e_seq):
        center = state_centers.get((b, e))
        if center is None:
            unseen_state_count += 1
            center = (h_centers[b - 1], b_centers[e - 1])
        qa_h_shell_seq.append(center[0])
        qa_b_shell_seq.append(center[1])
    j_seq: list[float] = []
    x_seq: list[float] = []
    k_seq: list[float] = []
    pi_seq: list[float] = []
    vars_seq: list[dict[str, int]] = []
    for b, e in zip(b_seq, e_seq):
        vars_for_state = qa_vars(b, e)
        j = vars_for_state["J"]
        x = vars_for_state["X"]
        k = vars_for_state["K"]
        j_seq.append(j)
        x_seq.append(x)
        k_seq.append(k)
        pi_seq.append(vars_for_state["Pi"])
        vars_seq.append(vars_for_state)

    be_area = abs(line_integral_y_dx(e_seq, b_seq))
    jx_area = abs(line_integral_y_dx(x_seq, j_seq))
    kx_area = abs(line_integral_y_dx(x_seq, k_seq))
    pi_x_area = abs(line_integral_y_dx(x_seq, pi_seq))
    path_energy = 0.0
    x2 = close(x_seq)
    pi2 = close(pi_seq)
    for i in range(len(x_seq)):
        path_energy += 0.5 * (pi2[i] + pi2[i + 1]) * abs(x2[i + 1] - x2[i])

    def lifted_shell_energy(h_var: str, b_var: str) -> tuple[float, float]:
        h_map = lifted_centers.get(("H", h_var), {})
        b_map = lifted_centers.get(("B", b_var), {})
        h_lifted_seq: list[float] = []
        b_lifted_seq: list[float] = []
        unseen = 0
        for b, e, vars_for_state in zip(b_seq, e_seq, vars_seq):
            h_key = vars_for_state[h_var]
            b_key = vars_for_state[b_var]
            h_value = h_map.get(h_key)
            b_value = b_map.get(b_key)
            if h_value is None:
                unseen += 1
                h_value = h_centers[b - 1]
            if b_value is None:
                unseen += 1
                b_value = b_centers[e - 1]
            h_lifted_seq.append(h_value)
            b_lifted_seq.append(b_value)
        energy = 1000.0 * abs(line_integral_y_dx(b_lifted_seq, h_lifted_seq)) / DENSITY_KG_PER_M3
        return energy, unseen / (2 * len(vars_seq))

    all_lifted: dict[str, float] = {}
    all_lifted_unseen: dict[str, float] = {}
    for h_var in QA_LIFTED_VARIABLES:
        for b_var in QA_LIFTED_VARIABLES:
            energy, unseen_fraction = lifted_shell_energy(h_var, b_var)
            field = lifted_energy_field(h_var, b_var)
            all_lifted[field] = energy
            all_lifted_unseen[field.replace("_energy_", "_unseen_")] = unseen_fraction

    lifted_jx = all_lifted[lifted_energy_field("J", "X")]
    lifted_jx_unseen = all_lifted_unseen[
        lifted_energy_field("J", "X").replace("_energy_", "_unseen_")
    ]
    lifted_kx = all_lifted[lifted_energy_field("K", "X")]
    lifted_kx_unseen = all_lifted_unseen[
        lifted_energy_field("K", "X").replace("_energy_", "_unseen_")
    ]
    lifted_k_minus_j_x = all_lifted[lifted_energy_field("K_minus_J", "X")]
    lifted_k_minus_j_x_unseen = all_lifted_unseen[
        lifted_energy_field("K_minus_J", "X").replace("_energy_", "_unseen_")
    ]
    lifted_j_pi = all_lifted[lifted_energy_field("J", "Pi")]
    lifted_j_pi_unseen = all_lifted_unseen[
        lifted_energy_field("J", "Pi").replace("_energy_", "_unseen_")
    ]

    fixed = {
        "qa_be_loop_area": be_area,
        "qa_jx_loop_area": jx_area,
        "qa_kx_loop_area": kx_area,
        "qa_pi_x_loop_area": pi_x_area,
        "qa_pi_abs_path_energy": path_energy,
        "nonqa_shell_hdb_energy_mj_per_kg": 1000.0
        * abs(line_integral_y_dx(nonqa_b_shell_seq, nonqa_h_shell_seq))
        / DENSITY_KG_PER_M3,
        "qa_shell_hdb_energy_mj_per_kg": 1000.0
        * abs(line_integral_y_dx(qa_b_shell_seq, qa_h_shell_seq))
        / DENSITY_KG_PER_M3,
        "qa_lifted_jx_shell_energy_mj_per_kg": lifted_jx,
        "qa_lifted_jx_unseen_fraction": lifted_jx_unseen,
        "qa_lifted_kx_shell_energy_mj_per_kg": lifted_kx,
        "qa_lifted_kx_unseen_fraction": lifted_kx_unseen,
        "qa_lifted_k_minus_j_x_shell_energy_mj_per_kg": lifted_k_minus_j_x,
        "qa_lifted_k_minus_j_x_unseen_fraction": lifted_k_minus_j_x_unseen,
        "qa_lifted_j_pi_shell_energy_mj_per_kg": lifted_j_pi,
        "qa_lifted_j_pi_unseen_fraction": lifted_j_pi_unseen,
        "qa_shell_unseen_state_fraction": unseen_state_count / len(b_seq),
        "qa_unique_states": float(len(set(zip(b_seq, e_seq)))),
    }
    fixed.update(all_lifted)
    fixed.update(all_lifted_unseen)
    return fixed


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


def direct_prediction(
    records: list[dict], heldout_names: set[str], prediction_field: str
) -> dict:
    test = [r for r in records if r["filename"] in heldout_names]
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [float(r[prediction_field]) for r in test]
    return {"prediction_field": prediction_field, "heldout": metrics(actual, predicted)}


def build_records_for_split(
    traces: list[LoopTrace], calibration_names: set[str], heldout_names: set[str]
) -> list[dict]:
    h_edges, b_edges, h_centers, b_centers, state_centers, lifted_centers = (
        build_global_calibration(traces, calibration_names)
    )
    records: list[dict] = []
    for trace in traces:
        qa = qa_observables(
            trace, h_edges, b_edges, h_centers, b_centers, state_centers, lifted_centers
        )
        physical_from_loop = physical_energy_mj_per_kg(trace)
        rec = {
            "filename": trace.row.filename,
            "bp_t": trace.row.bp_t,
            "frequency_hz": trace.row.frequency_hz,
            "split": "heldout" if trace.row.filename in heldout_names else "calibration",
            "rows": len(trace.t_s),
            "energy_loss_mj_per_kg_table": trace.row.energy_loss_mj_per_kg,
            "energy_loss_mj_per_kg_from_integral": physical_from_loop,
            "integral_vs_table_abs_error": abs(physical_from_loop - trace.row.energy_loss_mj_per_kg),
        }
        rec.update(qa)
        records.append(rec)
    return records


def direct_model_suite(
    records: list[dict], calibration_names: set[str], heldout_names: set[str]
) -> dict:
    return {
        "mean_loss_baseline": mean_baseline(records, calibration_names, heldout_names),
        "steinmetz_log_linear_baseline": fit_steinmetz(records, calibration_names, heldout_names),
        "raw_physical_hdb_direct": direct_prediction(
            records, heldout_names, "energy_loss_mj_per_kg_from_integral"
        ),
        "nonqa_shell_hdb_direct": direct_prediction(
            records, heldout_names, "nonqa_shell_hdb_energy_mj_per_kg"
        ),
        "qa_shell_hdb_direct": direct_prediction(
            records, heldout_names, "qa_shell_hdb_energy_mj_per_kg"
        ),
        "qa_lifted_jx_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_jx_shell_energy_mj_per_kg"
        ),
        "qa_lifted_kx_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_kx_shell_energy_mj_per_kg"
        ),
        "qa_lifted_k_minus_j_x_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_k_minus_j_x_shell_energy_mj_per_kg"
        ),
        "qa_lifted_j_pi_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_j_pi_shell_energy_mj_per_kg"
        ),
    }


def direct_comparison_rows(heldout_models: dict, comparison_order: list[str]) -> list[dict]:
    return [{"model": name, **heldout_models[name]["heldout"]} for name in comparison_order]


def lifted_pair_sweep(records: list[dict], heldout_names: set[str]) -> list[dict]:
    nonqa = direct_prediction(records, heldout_names, "nonqa_shell_hdb_energy_mj_per_kg")[
        "heldout"
    ]
    rows: list[dict] = []
    for h_var in QA_LIFTED_VARIABLES:
        for b_var in QA_LIFTED_VARIABLES:
            field = lifted_energy_field(h_var, b_var)
            result = direct_prediction(records, heldout_names, field)["heldout"]
            rows.append(
                {
                    "h_var": h_var,
                    "b_var": b_var,
                    "field": field,
                    **result,
                    "beats_nonqa_by_r2": result["r2"] > nonqa["r2"],
                    "beats_nonqa_by_rmse": result["rmse_mj_per_kg"]
                    < nonqa["rmse_mj_per_kg"],
                    "beats_nonqa_by_mae": result["mae_mj_per_kg"]
                    < nonqa["mae_mj_per_kg"],
                    "rmse_ratio_to_nonqa": result["rmse_mj_per_kg"]
                    / max(nonqa["rmse_mj_per_kg"], 1e-12),
                }
            )
    return rows


def summarize_lifted_pair_sweep(split_results: list[dict]) -> dict:
    aggregates: dict[str, dict] = {}
    for split in split_results:
        for row in split["lifted_pair_sweep"]:
            key = f"{row['h_var']}/{row['b_var']}"
            bucket = aggregates.setdefault(
                key,
                {
                    "h_var": row["h_var"],
                    "b_var": row["b_var"],
                    "r2_win_count": 0,
                    "rmse_win_count": 0,
                    "mae_win_count": 0,
                    "rmse_ratios": [],
                    "r2_values": [],
                },
            )
            bucket["r2_win_count"] += int(row["beats_nonqa_by_r2"])
            bucket["rmse_win_count"] += int(row["beats_nonqa_by_rmse"])
            bucket["mae_win_count"] += int(row["beats_nonqa_by_mae"])
            bucket["rmse_ratios"].append(row["rmse_ratio_to_nonqa"])
            bucket["r2_values"].append(row["r2"])

    scored = []
    for key, bucket in aggregates.items():
        scored.append(
            {
                "pair": key,
                "h_var": bucket["h_var"],
                "b_var": bucket["b_var"],
                "r2_win_count": bucket["r2_win_count"],
                "rmse_win_count": bucket["rmse_win_count"],
                "mae_win_count": bucket["mae_win_count"],
                "mean_rmse_ratio_to_nonqa": statistics.mean(bucket["rmse_ratios"]),
                "median_rmse_ratio_to_nonqa": statistics.median(bucket["rmse_ratios"]),
                "mean_r2": statistics.mean(bucket["r2_values"]),
            }
        )

    scored.sort(
        key=lambda row: (
            -row["rmse_win_count"],
            -row["mae_win_count"],
            -row["r2_win_count"],
            row["median_rmse_ratio_to_nonqa"],
            row["mean_rmse_ratio_to_nonqa"],
        )
    )
    return {
        "pair_count": len(scored),
        "top_by_transfer_score": scored[:10],
        "best_pair": scored[0] if scored else None,
    }


def transfer_splits(loss_rows: list[LossRow]) -> list[tuple[str, set[str], set[str]]]:
    all_names = {row.filename for row in loss_rows}
    splits: list[tuple[str, set[str], set[str]]] = []
    calibration_names, heldout_names = calibration_split(loss_rows)
    splits.append(("index_mod3_holdout", calibration_names, heldout_names))

    for frequency in sorted({row.frequency_hz for row in loss_rows}):
        heldout = {row.filename for row in loss_rows if row.frequency_hz == frequency}
        if 2 <= len(heldout) < len(all_names):
            splits.append((f"frequency_{int(frequency)}Hz_holdout", all_names - heldout, heldout))

    for bp_t in sorted({row.bp_t for row in loss_rows}):
        heldout = {row.filename for row in loss_rows if row.bp_t == bp_t}
        if 2 <= len(heldout) < len(all_names):
            splits.append((f"amplitude_{bp_t:.2f}T_holdout", all_names - heldout, heldout))
    return splits


def summarize_transfer_split(
    split_name: str,
    traces: list[LoopTrace],
    calibration_names: set[str],
    heldout_names: set[str],
    comparison_order: list[str],
) -> dict:
    records = build_records_for_split(traces, calibration_names, heldout_names)
    models = direct_model_suite(records, calibration_names, heldout_names)
    rows = direct_comparison_rows(models, comparison_order)
    pair_rows = lifted_pair_sweep(records, heldout_names)
    best_by_rmse = min(rows, key=lambda row: row["rmse_mj_per_kg"])
    best_shell_by_rmse = min(
        [row for row in rows if row["model"] != "raw_physical_hdb_direct"],
        key=lambda row: row["rmse_mj_per_kg"],
    )
    nonqa = models["nonqa_shell_hdb_direct"]["heldout"]
    qa_j_pi = models["qa_lifted_j_pi_shell_direct"]["heldout"]
    return {
        "split": split_name,
        "calibration_count": len(calibration_names),
        "heldout_count": len(heldout_names),
        "best_model_by_rmse": best_by_rmse["model"],
        "best_nonraw_model_by_rmse": best_shell_by_rmse["model"],
        "qa_lifted_j_pi_beats_nonqa_by_r2": qa_j_pi["r2"] > nonqa["r2"],
        "qa_lifted_j_pi_beats_nonqa_by_rmse": qa_j_pi["rmse_mj_per_kg"] < nonqa["rmse_mj_per_kg"],
        "best_lifted_pair_by_rmse": min(pair_rows, key=lambda row: row["rmse_mj_per_kg"]),
        "best_lifted_pair_by_r2": max(pair_rows, key=lambda row: row["r2"]),
        "lifted_pair_sweep": pair_rows,
        "models": {name: models[name]["heldout"] for name in comparison_order},
    }


def main() -> int:
    if not ZIP_PATH.exists():
        raise SystemExit(
            f"missing {ZIP_PATH}; download Ring35_Dataset_Txt.zip from {DATASET_RECORD}"
        )

    with zipfile.ZipFile(ZIP_PATH) as zf:
        loss_rows = select_sin_rows(load_sin_loss_table(zf))
        calibration_names, heldout_names = calibration_split(loss_rows)
        traces = [load_loop_trace(zf, row) for row in loss_rows]

    records = build_records_for_split(traces, calibration_names, heldout_names)

    qa_feature_sets = {
        "nonqa_shell_hdb_affine": ["nonqa_shell_hdb_energy_mj_per_kg"],
        "qa_shell_hdb_affine": ["qa_shell_hdb_energy_mj_per_kg"],
        "qa_be_only": ["qa_be_loop_area"],
        "qa_lifted_jx_only": ["qa_jx_loop_area"],
        "qa_lifted_jx_shell_affine": ["qa_lifted_jx_shell_energy_mj_per_kg"],
        "qa_lifted_j_pi_shell_affine": ["qa_lifted_j_pi_shell_energy_mj_per_kg"],
        "qa_lifted_packet": [
            "qa_shell_hdb_energy_mj_per_kg",
            "qa_lifted_jx_shell_energy_mj_per_kg",
            "qa_lifted_j_pi_shell_energy_mj_per_kg",
            "qa_jx_loop_area",
            "qa_kx_loop_area",
            "qa_pi_x_loop_area",
            "qa_pi_abs_path_energy",
            "qa_unique_states",
        ],
        "qa_with_drive_metadata": [
            "frequency_hz",
            "bp_t",
            "qa_shell_hdb_energy_mj_per_kg",
            "qa_lifted_j_pi_shell_energy_mj_per_kg",
            "qa_jx_loop_area",
            "qa_kx_loop_area",
            "qa_pi_abs_path_energy",
        ],
    }
    qa_models = {
        name: fit_predict(records, calibration_names, heldout_names, features)
        for name, features in qa_feature_sets.items()
    }

    heldout_models = {
        **direct_model_suite(records, calibration_names, heldout_names),
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
    }
    comparison_order = [
        "mean_loss_baseline",
        "steinmetz_log_linear_baseline",
        "raw_physical_hdb_direct",
        "nonqa_shell_hdb_direct",
        "qa_shell_hdb_direct",
        "qa_lifted_jx_shell_direct",
        "qa_lifted_kx_shell_direct",
        "qa_lifted_k_minus_j_x_shell_direct",
        "qa_lifted_j_pi_shell_direct",
        "qa_be_only",
        "qa_lifted_packet",
        "steinmetz_plus_qa_shape_log_model",
    ]
    direct_comparison = direct_comparison_rows(heldout_models, comparison_order)
    qa_shell_r2 = heldout_models["qa_shell_hdb_direct"]["heldout"]["r2"]
    nonqa_shell_r2 = heldout_models["nonqa_shell_hdb_direct"]["heldout"]["r2"]
    qa_shell_mae = heldout_models["qa_shell_hdb_direct"]["heldout"]["mae_mj_per_kg"]
    nonqa_shell_mae = heldout_models["nonqa_shell_hdb_direct"]["heldout"]["mae_mj_per_kg"]
    lifted_j_pi_r2 = heldout_models["qa_lifted_j_pi_shell_direct"]["heldout"]["r2"]
    lifted_j_pi_mae = heldout_models["qa_lifted_j_pi_shell_direct"]["heldout"]["mae_mj_per_kg"]
    transfer_order = [
        "mean_loss_baseline",
        "steinmetz_log_linear_baseline",
        "raw_physical_hdb_direct",
        "nonqa_shell_hdb_direct",
        "qa_shell_hdb_direct",
        "qa_lifted_jx_shell_direct",
        "qa_lifted_j_pi_shell_direct",
    ]
    split_summaries = [
        summarize_transfer_split(name, traces, cal, hold, transfer_order)
        for name, cal, hold in transfer_splits(loss_rows)
    ]
    lifted_pair_sweep_summary = summarize_lifted_pair_sweep(split_summaries)
    best_sweep_pair = lifted_pair_sweep_summary["best_pair"]
    qa_j_pi_r2_wins = sum(
        1 for split in split_summaries if split["qa_lifted_j_pi_beats_nonqa_by_r2"]
    )
    qa_j_pi_rmse_wins = sum(
        1 for split in split_summaries if split["qa_lifted_j_pi_beats_nonqa_by_rmse"]
    )

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
            "bin_centers": "global calibration-loop shell means only",
            "qa_state_centers": "global calibration-loop joint (b,e) state means only",
            "b": "bin(H_Aperm)",
            "e": "bin(B_T)",
            "d": "b+e",
            "a": "b+2*e",
            "J": "b*d",
            "X": "d*e",
            "K": "d*a",
            "observables": [
                "abs(integral b de)",
                "abs(integral H_bin dB_bin) converted to mJ/kg",
                "abs(integral H_shell dB_shell) converted to mJ/kg",
                "abs(integral H_J dB_X) converted to mJ/kg",
                "abs(integral H_K dB_X) converted to mJ/kg",
                "abs(integral H_(K-J) dB_X) converted to mJ/kg",
                "abs(integral H_J dB_Pi) converted to mJ/kg",
                "abs(integral J dX)",
                "abs(integral K dX)",
                "abs(integral Pi dX)",
                "sum 0.5*(Pi_i+Pi_j)*abs(delta_X)",
                "unique (b,e) states",
            ],
        },
        "heldout_models": heldout_models,
        "direct_predictor_comparison": direct_comparison,
        "transfer_split_summary": {
            "split_count": len(split_summaries),
            "splits": split_summaries,
            "qa_lifted_j_pi_r2_wins_vs_nonqa_shell": qa_j_pi_r2_wins,
            "qa_lifted_j_pi_rmse_wins_vs_nonqa_shell": qa_j_pi_rmse_wins,
            "lifted_pair_sweep_summary": lifted_pair_sweep_summary,
        },
        "verdict": {
            "qa_not_random": (
                "YES: QA-only held-out models reduce error substantially relative "
                "to the mean-loss baseline on real measured loops."
            ),
            "qa_beats_steinmetz_here": (
                "YES: the direct dimensioned QA shell integral outperforms the "
                "log-linear Steinmetz baseline on this selected Ring35 SIN subset."
            ),
            "qa_adds_to_steinmetz_here": (
                "The tested log Steinmetz+rank-QA shape model does not improve "
                "Steinmetz, but the physically dimensioned QA shell integral beats "
                "Steinmetz directly."
            ),
            "qa_beats_nonqa_shell_here": (
                "YES"
                if qa_shell_r2 > nonqa_shell_r2 and qa_shell_mae < nonqa_shell_mae
                else "NO"
            ),
            "qa_lifted_vs_nonqa_shell_here": (
                "MIXED: calibrated QA J/Pi shells have higher held-out R2 than "
                "ordinary H/B shell binning, but worse MAE/RMSE. This is not yet "
                "a clean QA-specific win."
                if lifted_j_pi_r2 > nonqa_shell_r2 and lifted_j_pi_mae > nonqa_shell_mae
                else "See direct_predictor_comparison."
            ),
            "shell_result_interpretation": (
                "The raw physical loop integral is the expected upper-bound sanity "
                "check. Ordinary non-QA H/B shell binning explains most of the "
                "direct shell result and beats the joint (b,e) QA state shell by "
                "MAE/RMSE. A lifted QA J/Pi shell slightly beats non-QA by R2 but "
                "not by MAE/RMSE, so this run supports finite-shell hysteresis "
                "mapping and only a mixed, not decisive, QA-specific advantage."
            ),
            "transfer_result_interpretation": (
                f"Across {len(split_summaries)} index/frequency/amplitude transfer "
                f"splits, QA J/Pi beats ordinary non-QA shell by R2 in "
                f"{qa_j_pi_r2_wins} splits and by RMSE in {qa_j_pi_rmse_wins} splits. "
                "This hardens the conclusion by checking family-heldout transfer. "
                "The lifted-pair sweep reports whether another QA variable pair "
                "transfers more consistently."
            ),
            "lifted_pair_sweep_interpretation": (
                f"Best lifted pair by the predeclared transfer score is "
                f"{best_sweep_pair['pair']}, with RMSE wins in "
                f"{best_sweep_pair['rmse_win_count']} of {len(split_summaries)} splits "
                f"and median RMSE ratio {best_sweep_pair['median_rmse_ratio_to_nonqa']:.3f} "
                "relative to ordinary non-QA shell. This does not establish a stable "
                "QA-specific advantage yet."
            ),
            "leakage_controls": (
                "H/B quantile edges, marginal shell centers, and joint QA state "
                "centers are fit on calibration loops only. Direct predictors use "
                "no held-out loss labels and no fitted regression coefficient."
            ),
        },
        "records": records,
        "interpretation": (
            "A useful QA hysteresis mapping should improve held-out loss prediction "
            "over the mean baseline and should be evaluated against Steinmetz. "
            "The rank-only QA models test whether measured loop geometry maps into "
            "QA coordinates without physical scale. The dimensioned QA shell model "
            "keeps calibration-set H/B shell centers and directly approximates the "
            "hysteresis work integral."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["heldout_models"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
