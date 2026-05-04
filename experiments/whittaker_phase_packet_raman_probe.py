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
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


DATA_ROOT = Path("qa_lab/qa_data/raman")
OUT_PATH = Path("results/whittaker_phase_packet_raman_probe.json")
MIN_CLASS_N = 10
TOP_PEAKS = 5
MIN_SPACING_CM = 8.0

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
            "raw_features": raw_peak_features(peaks),
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


def main() -> int:
    rows = load_dataset()
    train, test = split_rows(rows)
    counts = Counter(row["label"] for row in rows)
    phase_result = evaluate(train, test, "phase_features")
    raw_result = evaluate(train, test, "raw_features")
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
        "phase_packet_result": phase_result,
        "raw_peak_baseline_result": raw_result,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
