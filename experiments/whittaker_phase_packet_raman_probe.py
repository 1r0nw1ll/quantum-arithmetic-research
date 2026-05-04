#!/usr/bin/env python3
"""Real-data Whittaker phase-packet probe on local RRUFF Raman spectra.

This is an experiment, not a cert. It tests whether the exact phase-packet
substrate from the Whittaker ladder produces useful real-data features on a
wave/spectral dataset already present in the repo.

Data:
  qa_lab/qa_data/raman/<label>/*.txt

Observer boundary:
  Raman wavenumber samples are read as decimal strings and converted directly
  to fractions.Fraction. Peak selection and classification use ordinary
  observer-side numerical ranking. The Whittaker/QA side is the exact phase
  argument:

      phase_arg = k * (omega dot x - v*t)

  with t=0, v=0 and x in {e_x, e_y, e_z}. No trigonometric functions are
  evaluated.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


DATA_ROOT = Path("qa_lab/qa_data/raman")
OUT_PATH = Path("results/whittaker_phase_packet_raman_probe.json")
MIN_CLASS_N = 10
TOP_PEAKS = 5
MIN_SPACING_CM = 8.0
N_FOLDS = 3
N_SHUFFLES = 50
SEED = 275

# Exact [273] S2 packets known to be present for m=3/5. Stored as
# (x_num, y_num, z_num, den).
OMEGA_PACKETS = [
    (7, 24, 0, 25),
    (24, 7, 0, 25),
    (1200, 1200, -527, 1777),
]


def parse_decimal_fraction(text: str) -> Fraction:
    return Fraction(text.strip())


def load_spectrum(path: Path) -> list[tuple[Fraction, float]]:
    points: list[tuple[Fraction, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        left, right = line.split(",", 1)
        try:
            k = parse_decimal_fraction(left)
            y = float(right.strip())
        except Exception:
            continue
        if math.isfinite(y):
            points.append((k, y))
    points.sort(key=lambda item: item[0])
    return points


def select_peaks(points: list[tuple[Fraction, float]]) -> list[tuple[Fraction, float]]:
    if len(points) < 3:
        return []
    intensities = [y for _, y in points]
    baseline = min(intensities)
    adjusted = [(k, y - baseline) for k, y in points]
    candidates: list[tuple[Fraction, float]] = []
    for idx in range(1, len(adjusted) - 1):
        k, y = adjusted[idx]
        if y > adjusted[idx - 1][1] and y >= adjusted[idx + 1][1]:
            candidates.append((k, y))
    if len(candidates) < TOP_PEAKS:
        candidates = sorted(adjusted, key=lambda item: item[1], reverse=True)
    candidates.sort(key=lambda item: item[1], reverse=True)

    selected: list[tuple[Fraction, float]] = []
    for k, y in candidates:
        if y <= 0:
            continue
        k_float = float(k)
        if all(abs(k_float - float(prev_k)) >= MIN_SPACING_CM for prev_k, _ in selected):
            selected.append((k, y))
        if len(selected) == TOP_PEAKS:
            break
    selected.sort(key=lambda item: item[0])
    return selected


def residue24(value: Fraction) -> float:
    q = math.floor(value / 24)
    r = value - 24 * q
    return float(r) / 24.0


def phase_features(peaks: list[tuple[Fraction, float]]) -> list[float]:
    features: list[float] = []
    if not peaks:
        return [0.0] * (TOP_PEAKS * len(OMEGA_PACKETS) * 3 + TOP_PEAKS)

    total_intensity = sum(max(0.0, y) for _, y in peaks) or 1.0
    padded = peaks[:]
    while len(padded) < TOP_PEAKS:
        padded.append((Fraction(0, 1), 0.0))

    for k, y in padded[:TOP_PEAKS]:
        for x_num, y_num, z_num, den in OMEGA_PACKETS:
            for component in (x_num, y_num, z_num):
                phase_arg = k * Fraction(component, den)
                features.append(residue24(phase_arg))
        features.append(max(0.0, y) / total_intensity)
    return features


def phase_no_intensity_features(peaks: list[tuple[Fraction, float]]) -> list[float]:
    features: list[float] = []
    padded = peaks[:]
    while len(padded) < TOP_PEAKS:
        padded.append((Fraction(0, 1), 0.0))

    for k, _y in padded[:TOP_PEAKS]:
        for x_num, y_num, z_num, den in OMEGA_PACKETS:
            for component in (x_num, y_num, z_num):
                phase_arg = k * Fraction(component, den)
                features.append(residue24(phase_arg))
    return features


def phase_intensity_only_features(peaks: list[tuple[Fraction, float]]) -> list[float]:
    total_intensity = sum(max(0.0, y) for _, y in peaks) or 1.0
    padded = peaks[:]
    while len(padded) < TOP_PEAKS:
        padded.append((Fraction(0, 1), 0.0))
    return [max(0.0, y) / total_intensity for _k, y in padded[:TOP_PEAKS]]


def phase_single_omega_features(
    peaks: list[tuple[Fraction, float]],
    omega_idx: int,
) -> list[float]:
    x_num, y_num, z_num, den = OMEGA_PACKETS[omega_idx]
    features: list[float] = []
    total_intensity = sum(max(0.0, y) for _, y in peaks) or 1.0
    padded = peaks[:]
    while len(padded) < TOP_PEAKS:
        padded.append((Fraction(0, 1), 0.0))
    for k, y in padded[:TOP_PEAKS]:
        for component in (x_num, y_num, z_num):
            phase_arg = k * Fraction(component, den)
            features.append(residue24(phase_arg))
        features.append(max(0.0, y) / total_intensity)
    return features


def raw_peak_features(peaks: list[tuple[Fraction, float]]) -> list[float]:
    features: list[float] = []
    total_intensity = sum(max(0.0, y) for _, y in peaks) or 1.0
    padded = peaks[:]
    while len(padded) < TOP_PEAKS:
        padded.append((Fraction(0, 1), 0.0))
    for k, y in padded[:TOP_PEAKS]:
        features.append(float(k) / 4000.0)
        features.append(max(0.0, y) / total_intensity)
    return features


def raw_peak_no_intensity_features(peaks: list[tuple[Fraction, float]]) -> list[float]:
    padded = peaks[:]
    while len(padded) < TOP_PEAKS:
        padded.append((Fraction(0, 1), 0.0))
    return [float(k) / 4000.0 for k, _y in padded[:TOP_PEAKS]]


def raw_intensity_only_features(peaks: list[tuple[Fraction, float]]) -> list[float]:
    total_intensity = sum(max(0.0, y) for _, y in peaks) or 1.0
    padded = peaks[:]
    while len(padded) < TOP_PEAKS:
        padded.append((Fraction(0, 1), 0.0))
    return [max(0.0, y) / total_intensity for _k, y in padded[:TOP_PEAKS]]


def load_dataset() -> list[dict]:
    rows = []
    for path in sorted(DATA_ROOT.glob("*/*.txt")):
        label = path.parent.name
        points = load_spectrum(path)
        peaks = select_peaks(points)
        if not peaks:
            continue
        rows.append({
            "path": str(path),
            "label": label,
            "n_points": len(points),
            "peaks": [(str(k), y) for k, y in peaks],
            "phase_features": phase_features(peaks),
            "phase_no_intensity_features": phase_no_intensity_features(peaks),
            "phase_intensity_only_features": phase_intensity_only_features(peaks),
            "phase_omega0_features": phase_single_omega_features(peaks, 0),
            "phase_omega1_features": phase_single_omega_features(peaks, 1),
            "phase_omega2_features": phase_single_omega_features(peaks, 2),
            "raw_features": raw_peak_features(peaks),
            "raw_no_intensity_features": raw_peak_no_intensity_features(peaks),
            "raw_intensity_only_features": raw_intensity_only_features(peaks),
        })
    counts = Counter(row["label"] for row in rows)
    keep = {label for label, n in counts.items() if n >= MIN_CLASS_N}
    return [row for row in rows if row["label"] in keep]


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    train: list[dict] = []
    test: list[dict] = []
    for label, items in sorted(by_label.items()):
        items = sorted(items, key=lambda row: row["path"])
        for idx, row in enumerate(items):
            if idx % 3 == 0:
                test.append(row)
            else:
                train.append(row)
    return train, test


def split_rows_fold(rows: list[dict], fold: int, n_folds: int = N_FOLDS) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    train: list[dict] = []
    test: list[dict] = []
    for label, items in sorted(by_label.items()):
        items = sorted(items, key=lambda row: row["path"])
        for idx, row in enumerate(items):
            if idx % n_folds == fold:
                test.append(row)
            else:
                train.append(row)
    return train, test


def centroid_model(rows: list[dict], feature_key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row[feature_key])
    centroids = {}
    for label, vectors in grouped.items():
        n = len(vectors)
        width = len(vectors[0])
        centroid = []
        for col in range(width):
            centroid.append(sum(vec[col] for vec in vectors) / n)
        centroids[label] = centroid
    return centroids


def sqdist(lhs: list[float], rhs: list[float]) -> float:
    total = 0.0
    for a, b in zip(lhs, rhs):
        delta = a - b
        total += delta * delta
    return total


def evaluate(train: list[dict], test: list[dict], feature_key: str) -> dict:
    centroids = centroid_model(train, feature_key)
    correct = 0
    confusion = Counter()
    for row in test:
        pred = min(
            centroids,
            key=lambda label: sqdist(row[feature_key], centroids[label]),
        )
        if pred == row["label"]:
            correct += 1
        confusion[(row["label"], pred)] += 1
    return {
        "accuracy": correct / len(test) if test else 0.0,
        "correct": correct,
        "n_test": len(test),
        "confusion_top": [
            {"true": true, "pred": pred, "n": n}
            for (true, pred), n in confusion.most_common(20)
        ],
    }


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def repeated_fold_results(rows: list[dict], feature_keys: list[str]) -> dict:
    out = {}
    for feature_key in feature_keys:
        fold_results = []
        for fold in range(N_FOLDS):
            train, test = split_rows_fold(rows, fold)
            result = evaluate(train, test, feature_key)
            fold_results.append(result["accuracy"])
        out[feature_key] = summarize(fold_results)
    return out


def label_shuffle_null(rows: list[dict], feature_key: str, n: int = N_SHUFFLES) -> dict:
    rng = random.Random(SEED)
    labels = [row["label"] for row in rows]
    accuracies = []
    for _ in range(n):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        shuffled_rows = []
        for row, label in zip(rows, shuffled):
            clone = dict(row)
            clone["label"] = label
            shuffled_rows.append(clone)
        fold_scores = []
        for fold in range(N_FOLDS):
            train, test = split_rows_fold(shuffled_rows, fold)
            fold_scores.append(evaluate(train, test, feature_key)["accuracy"])
        accuracies.append(statistics.fmean(fold_scores))
    return summarize(accuracies)


def null_p_value(observed: float, null_values: list[float]) -> float:
    ge = sum(1 for value in null_values if value >= observed)
    return (ge + 1) / (len(null_values) + 1)


def main() -> int:
    rows = load_dataset()
    train, test = split_rows(rows)
    counts = Counter(row["label"] for row in rows)
    phase_result = evaluate(train, test, "phase_features")
    raw_result = evaluate(train, test, "raw_features")
    feature_keys = [
        "phase_features",
        "phase_no_intensity_features",
        "phase_intensity_only_features",
        "phase_omega0_features",
        "phase_omega1_features",
        "phase_omega2_features",
        "raw_features",
        "raw_no_intensity_features",
        "raw_intensity_only_features",
    ]
    folds = repeated_fold_results(rows, feature_keys)
    phase_null = label_shuffle_null(rows, "phase_features")
    raw_null = label_shuffle_null(rows, "raw_features")
    phase_observed = folds["phase_features"]["mean"]
    raw_observed = folds["raw_features"]["mean"]
    payload = {
        "experiment": "whittaker_phase_packet_raman_probe",
        "status": "real_data_probe",
        "data_root": str(DATA_ROOT),
        "n_spectra": len(rows),
        "n_train": len(train),
        "n_test": len(test),
        "class_counts": dict(sorted(counts.items())),
        "observer_boundary": (
            "Raman decimal samples are observer input; peak extraction and "
            "classification are observer-side. Phase arguments are exact "
            "Fraction values; no trig functions are evaluated."
        ),
        "omega_packets": OMEGA_PACKETS,
        "top_peaks": TOP_PEAKS,
        "min_spacing_cm": MIN_SPACING_CM,
        "n_folds": N_FOLDS,
        "n_label_shuffles": N_SHUFFLES,
        "seed": SEED,
        "phase_packet_result": phase_result,
        "raw_peak_baseline_result": raw_result,
        "repeated_fold_results": folds,
        "label_shuffle_nulls": {
            "phase_features": phase_null,
            "raw_features": raw_null,
        },
        "null_p_values": {
            "phase_features": null_p_value(phase_observed, phase_null["values"]),
            "raw_features": null_p_value(raw_observed, raw_null["values"]),
        },
        "summary": {
            "phase_mean_accuracy": phase_observed,
            "raw_mean_accuracy": raw_observed,
            "phase_minus_raw_mean_accuracy": phase_observed - raw_observed,
            "best_ablation": max(
                folds,
                key=lambda key: folds[key]["mean"],
            ),
            "best_ablation_mean_accuracy": max(
                folds[key]["mean"] for key in folds
            ),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
