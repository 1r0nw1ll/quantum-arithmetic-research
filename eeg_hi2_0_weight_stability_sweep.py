#!/usr/bin/env python3
"""
eeg_hi2_0_weight_stability_sweep.py
===================================

Reduced-pressure chromogeometric weight-stability sweep for HI 2.0 on CHB-MIT.

Purpose:
- hold the balanced patient datasets fixed
- sweep a small QA-native grid of (w_ang, w_rad) settings with w_fam=0
- measure whether the HI 2.0 advantage over HI 1.0 is broad and stable
  or narrow and patient-fragile

This stays within the existing HI 2.0 formula family and classifier setup.
"""

QA_COMPLIANCE = "empirical_observer — EEG signal is observer input; QA discrete orbit is the classifier state"


from __future__ import annotations

import argparse
import json
import os
import re
import zlib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from eeg_brain_feature_extractor_fixed import EEGBrainFeatureExtractor
from qa_harmonicity_v2 import compute_hi_1_0, compute_hi_2_0, qa_tuple


DEFAULT_DATA_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_weight_stability_sweep_10patients.json")
WINDOW_SEC = 4.0
OVERLAP_SEC = 2.0
MIN_SEGMENTS_PER_CLASS = 8
MAX_SEGMENTS_PER_CLASS = 64
TEST_SIZE = 0.25
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 8
DEFAULT_PATIENT_LIMIT = 10
WEIGHT_CONFIGS = [
    {"label": "angular_1.00_radial_0.00", "w_ang": 1.0, "w_rad": 0.0, "w_fam": 0.0},
    {"label": "angular_0.75_radial_0.25", "w_ang": 0.75, "w_rad": 0.25, "w_fam": 0.0},
    {"label": "angular_0.50_radial_0.50", "w_ang": 0.50, "w_rad": 0.50, "w_fam": 0.0},
    {"label": "angular_0.25_radial_0.75", "w_ang": 0.25, "w_rad": 0.75, "w_fam": 0.0},
    {"label": "angular_0.00_radial_1.00", "w_ang": 0.00, "w_rad": 1.00, "w_fam": 0.0},
]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def patient_seed(patient_id: str) -> int:
    return zlib.crc32(patient_id.encode("utf-8")) & 0xFFFFFFFF


def parse_summary(summary_path: Path) -> dict[str, list[tuple[int, int]]]:
    annotations: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    pending_start: int | None = None

    with summary_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            file_match = re.match(r"File Name:\s+(\S+\.edf)", line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1).lower()
                annotations.setdefault(current_file, [])
                pending_start = None
                continue

            if current_file is None:
                continue

            start_match = re.match(
                r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)\s+second",
                line,
                re.IGNORECASE,
            )
            if start_match:
                pending_start = int(start_match.group(1))
                continue

            end_match = re.match(
                r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)\s+second",
                line,
                re.IGNORECASE,
            )
            if end_match and pending_start is not None:
                annotations[current_file].append((pending_start, int(end_match.group(1))))
                pending_start = None

    return {name: spans for name, spans in annotations.items() if spans}


def read_edf_all_channels(edf_path: Path) -> tuple[np.ndarray, int, list[str]]:
    with edf_path.open("rb") as handle:
        header = handle.read(256)
        ns = int(header[252:256].decode("ascii").strip())
        sig_header_raw = handle.read(ns * 256)

        def field(offset: int, width: int, count: int) -> list[str]:
            return [
                sig_header_raw[offset + i * width : offset + (i + 1) * width]
                .decode("ascii")
                .strip()
                for i in range(count)
            ]

        labels = field(0, 16, ns)
        phys_mins = [float(x) for x in field(ns * (16 + 80 + 8), 8, ns)]
        phys_maxs = [float(x) for x in field(ns * (16 + 80 + 8 + 8), 8, ns)]
        dig_mins = [int(x) for x in field(ns * (16 + 80 + 8 + 8 + 8), 8, ns)]
        dig_maxs = [int(x) for x in field(ns * (16 + 80 + 8 + 8 + 8 + 8), 8, ns)]
        n_samp = [
            int(x)
            for x in field(ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80), 8, ns)
        ]
        record_duration = float(header[244:252].decode("ascii").strip())
        n_records = int(header[236:244].decode("ascii").strip())
        sample_rate = int(n_samp[0] / record_duration)
        gains = [
            (phys_maxs[i] - phys_mins[i]) / (dig_maxs[i] - dig_mins[i])
            for i in range(ns)
        ]
        offsets = [phys_maxs[i] - gains[i] * dig_maxs[i] for i in range(ns)]
        total_per_ch = n_records * n_samp[0]
        signals = np.empty((ns, total_per_ch), dtype=np.float32)
        record_total = sum(n_samp)

        for rec in range(n_records):
            raw = handle.read(record_total * 2)
            if len(raw) < record_total * 2:
                break
            all_samp = np.frombuffer(raw, dtype=np.int16)
            pos = 0
            for ch in range(ns):
                samples = all_samp[pos : pos + n_samp[ch]].astype(np.float32)
                signals[ch, rec * n_samp[ch] : (rec + 1) * n_samp[ch]] = (
                    samples * gains[ch] + offsets[ch]
                )
                pos += n_samp[ch]

    return signals, sample_rate, labels


def segment_signals(signals: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    window_samples = int(WINDOW_SEC * sample_rate)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * sample_rate)
    segments: list[np.ndarray] = []
    start = 0
    while start + window_samples <= signals.shape[1]:
        segments.append(signals[:, start : start + window_samples])
        start += step_samples
    return segments


def label_segments(n_segments: int, seizure_times: list[tuple[int, int]]) -> np.ndarray:
    step_sec = WINDOW_SEC - OVERLAP_SEC
    labels = np.zeros(n_segments, dtype=int)
    for idx in range(n_segments):
        segment_start = idx * step_sec
        segment_end = segment_start + WINDOW_SEC
        for seizure_start, seizure_end in seizure_times:
            if not (segment_end < seizure_start or segment_start > seizure_end):
                labels[idx] = 1
                break
    return labels


def extract_7d_features(
    segments: list[np.ndarray],
    channel_names: list[str],
    extractor: EEGBrainFeatureExtractor,
    label: str,
) -> np.ndarray:
    features: list[np.ndarray] = []
    total = len(segments)
    for idx, segment in enumerate(segments, start=1):
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"    [{label}] segment {idx}/{total}", flush=True)
        channels_data = {
            channel_name: segment[ch_idx, :]
            for ch_idx, channel_name in enumerate(channel_names)
        }
        features.append(extractor.extract_network_features(channels_data))
    return np.array(features, dtype=np.float64)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_classifier(
    features_train: np.ndarray,
    features_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    clf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(features_train, y_train)
    return metric_bundle(y_test, clf.predict(features_test))


def extract_hi1_features(features_7d: np.ndarray) -> np.ndarray:
    out = np.zeros((len(features_7d), 4), dtype=np.float64)
    for idx in range(len(features_7d)):
        b = max(1, min(24, int(features_7d[idx, 0] * 23) + 1))
        e = max(1, min(24, int(features_7d[idx, 1] * 23) + 1))
        q = qa_tuple(b, e, modulus=24)
        out[idx] = [
            compute_hi_1_0(q, modulus=24),
            b / 24.0,
            e / 24.0,
            float(np.linalg.norm(features_7d[idx])),
        ]
    return out


def extract_hi2_features(
    features_7d: np.ndarray,
    w_ang: float,
    w_rad: float,
    w_fam: float,
) -> tuple[np.ndarray, dict[str, float]]:
    out = np.zeros((len(features_7d), 10), dtype=np.float64)
    hi_values: list[float] = []
    angular_values: list[float] = []
    radial_values: list[float] = []

    for idx in range(len(features_7d)):
        b = max(1, min(24, int(features_7d[idx, 0] * 23) + 1))
        e = max(1, min(24, int(features_7d[idx, 1] * 23) + 1))
        q = qa_tuple(b, e, modulus=24)
        result = compute_hi_2_0(q, w_ang=w_ang, w_rad=w_rad, w_fam=w_fam, modulus=24)
        c_val, f_val, g_val = result["pythagorean_triple"]
        out[idx] = [
            result["HI_2.0"],
            result["H_angular"],
            result["H_radial"],
            result["H_family"],
            c_val / 1000.0,
            f_val / 1000.0,
            g_val / 1000.0,
            result["gcd"],
            b / 24.0,
            e / 24.0,
        ]
        hi_values.append(float(result["HI_2.0"]))
        angular_values.append(float(result["H_angular"]))
        radial_values.append(float(result["H_radial"]))

    return out, {
        "mean_hi_2_0": float(np.mean(hi_values)) if hi_values else 0.0,
        "mean_h_angular": float(np.mean(angular_values)) if angular_values else 0.0,
        "mean_h_radial": float(np.mean(radial_values)) if radial_values else 0.0,
    }


def build_balanced_patient_dataset(
    patient_dir: Path,
    extractor: EEGBrainFeatureExtractor,
) -> dict[str, object]:
    patient_id = patient_dir.name
    print(f"[{patient_id}] building balanced dataset", flush=True)
    summary_path = patient_dir / f"{patient_id}-summary.txt"
    result: dict[str, object] = {
        "patient_id": patient_id,
        "skipped": True,
        "skip_reason": None,
        "n_files": 0,
        "n_segments_raw": 0,
        "n_segments_balanced": 0,
        "n_baseline_raw": 0,
        "n_seizure_raw": 0,
        "n_baseline": 0,
        "n_seizure": 0,
    }

    if not summary_path.exists():
        result["skip_reason"] = "missing_summary"
        return result

    annotations = parse_summary(summary_path)
    if not annotations:
        result["skip_reason"] = "no_seizure_annotations"
        return result

    candidate_meta: list[dict[str, object]] = []
    seizure_candidates: list[tuple[str, int]] = []
    baseline_candidates: list[tuple[str, int]] = []

    for filename, seizure_times in sorted(annotations.items()):
        edf_path = patient_dir / filename
        if not edf_path.exists():
            continue

        try:
            signals, sample_rate, channel_names = read_edf_all_channels(edf_path)
        except Exception as exc:
            print(f"[{patient_id}] read error {filename}: {exc}", flush=True)
            continue

        local_extractor = extractor
        if sample_rate != extractor.sample_rate:
            local_extractor = EEGBrainFeatureExtractor(sample_rate=sample_rate)
        segments = segment_signals(signals, sample_rate)
        labels = label_segments(len(segments), seizure_times)
        candidate_meta.append(
            {
                "file": filename,
                "edf_path": edf_path,
                "channel_names": channel_names,
                "sample_rate": sample_rate,
                "labels": labels,
                "extractor": local_extractor,
            }
        )
        for idx, value in enumerate(labels):
            key = (filename, idx)
            if value == 1:
                seizure_candidates.append(key)
            else:
                baseline_candidates.append(key)

    if not candidate_meta:
        result["skip_reason"] = "no_readable_edf_files"
        return result

    result["n_files"] = int(len(candidate_meta))
    result["n_segments_raw"] = int(len(seizure_candidates) + len(baseline_candidates))
    result["n_baseline_raw"] = int(len(baseline_candidates))
    result["n_seizure_raw"] = int(len(seizure_candidates))

    if len(baseline_candidates) < MIN_SEGMENTS_PER_CLASS or len(seizure_candidates) < MIN_SEGMENTS_PER_CLASS:
        result["skip_reason"] = "insufficient_segments_per_class"
        return result

    target_count = min(len(baseline_candidates), len(seizure_candidates), MAX_SEGMENTS_PER_CLASS)
    rng = np.random.default_rng(patient_seed(patient_id))
    baseline_pick_idx = rng.choice(len(baseline_candidates), target_count, replace=False)
    seizure_pick_idx = rng.choice(len(seizure_candidates), target_count, replace=False)
    selected = {baseline_candidates[idx]: 0 for idx in baseline_pick_idx}
    for idx in seizure_pick_idx:
        selected[seizure_candidates[idx]] = 1

    features: list[np.ndarray] = []
    labels_out: list[int] = []
    for meta in candidate_meta:
        file_keys = sorted(
            (key for key in selected if key[0] == meta["file"]),
            key=lambda item: item[1],
        )
        if not file_keys:
            continue
        print(f"[{patient_id}] extracting {len(file_keys)} selected segments from {meta['file']}", flush=True)
        signals, sample_rate, _ = read_edf_all_channels(meta["edf_path"])
        segments = segment_signals(signals, sample_rate)
        selected_segments = [segments[key[1]] for key in file_keys]
        file_features = extract_7d_features(
            selected_segments,
            meta["channel_names"],
            meta["extractor"],
            f"{patient_id}:{meta['file']}",
        )
        features.extend(file_features)
        labels_out.extend(selected[key] for key in file_keys)

    features_7d = np.array(features, dtype=np.float64)
    labels_array = np.array(labels_out, dtype=int)
    shuffle_idx = np.arange(len(labels_array))
    rng.shuffle(shuffle_idx)
    features_7d = features_7d[shuffle_idx]
    labels_array = labels_array[shuffle_idx]
    indices = np.arange(len(labels_array))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=patient_seed(patient_id) ^ 0xA5A5A5A5,
        stratify=labels_array,
    )

    result.update(
        {
            "skipped": False,
            "skip_reason": None,
            "n_segments_balanced": int(len(labels_array)),
            "n_baseline": int((labels_array == 0).sum()),
            "n_seizure": int((labels_array == 1).sum()),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "features_7d": features_7d,
            "labels": labels_array,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
    )
    return result


def reduce_patient_record(patient: dict[str, object]) -> dict[str, object]:
    reduced = {}
    for key, value in patient.items():
        if key in {"features_7d", "labels", "train_idx", "test_idx"}:
            continue
        reduced[key] = value
    return reduced


def evaluate_weight_config(
    patient_datasets: list[dict[str, object]],
    label: str,
    w_ang: float,
    w_rad: float,
    w_fam: float,
) -> dict[str, object]:
    print(f"[sweep] evaluating {label}", flush=True)
    patient_results: list[dict[str, object]] = []

    for patient in patient_datasets:
        base_record = reduce_patient_record(patient)
        if patient["skipped"]:
            patient_results.append(base_record)
            continue

        features_7d = patient["features_7d"]
        labels = patient["labels"]
        train_idx = patient["train_idx"]
        test_idx = patient["test_idx"]

        hi1 = extract_hi1_features(features_7d)
        hi2, feature_means = extract_hi2_features(features_7d, w_ang=w_ang, w_rad=w_rad, w_fam=w_fam)

        baseline_hi1 = evaluate_classifier(
            hi1[train_idx],
            hi1[test_idx],
            labels[train_idx],
            labels[test_idx],
        )
        candidate_hi2 = evaluate_classifier(
            hi2[train_idx],
            hi2[test_idx],
            labels[train_idx],
            labels[test_idx],
        )

        hi_values = hi2[:, 0]
        baseline_mask = labels == 0
        seizure_mask = labels == 1
        hi_seizure_mean = float(np.mean(hi_values[seizure_mask])) if np.any(seizure_mask) else 0.0
        hi_baseline_mean = float(np.mean(hi_values[baseline_mask])) if np.any(baseline_mask) else 0.0

        base_record.update(
            {
                "HI_1.0": baseline_hi1,
                "HI_2.0": candidate_hi2,
                "feature_means": feature_means,
                "score_means": {
                    "seizure_hi_2_0_mean": hi_seizure_mean,
                    "baseline_hi_2_0_mean": hi_baseline_mean,
                    "delta_hi_2_0_mean": float(hi_seizure_mean - hi_baseline_mean),
                },
                "deltas": {
                    "accuracy_delta": float(candidate_hi2["accuracy"] - baseline_hi1["accuracy"]),
                    "precision_delta": float(candidate_hi2["precision"] - baseline_hi1["precision"]),
                    "recall_delta": float(candidate_hi2["recall"] - baseline_hi1["recall"]),
                    "f1_delta": float(candidate_hi2["f1"] - baseline_hi1["f1"]),
                },
            }
        )
        patient_results.append(base_record)

    evaluable = [result for result in patient_results if not result["skipped"]]
    f1_deltas = [result["deltas"]["f1_delta"] for result in evaluable]
    accuracy_deltas = [result["deltas"]["accuracy_delta"] for result in evaluable]
    score_deltas = [result["score_means"]["delta_hi_2_0_mean"] for result in evaluable]
    count_nonnegative = sum(1 for value in f1_deltas if value >= 0.0)
    share_nonnegative = (count_nonnegative / len(evaluable)) if evaluable else 0.0

    if len(evaluable) >= 5 and (0.0 < w_rad < 1.0) and f1_deltas and float(np.mean(f1_deltas)) > 0.0 and share_nonnegative >= 0.7:
        verdict = "PASS"
        verdict_reason = "Mixed angular/radial setting preserved a positive mean F1 delta with at least 70% non-negative patient deltas."
    elif evaluable and f1_deltas and float(np.mean(f1_deltas)) > 0.0:
        verdict = "PARTIAL"
        verdict_reason = "Aggregate mean F1 delta stayed positive, but patient-level stability was weaker than the preregistered target."
    else:
        verdict = "FAIL"
        verdict_reason = "This setting did not preserve a positive patient-level HI 2.0 gain."

    return {
        "label": label,
        "weights": {"w_ang": w_ang, "w_rad": w_rad, "w_fam": w_fam},
        "patients": patient_results,
        "aggregate": {
            "n_patients_evaluable": int(len(evaluable)),
            "n_patients_skipped": int(len(patient_results) - len(evaluable)),
            "mean_accuracy_delta": float(np.mean(accuracy_deltas)) if accuracy_deltas else 0.0,
            "median_accuracy_delta": float(np.median(accuracy_deltas)) if accuracy_deltas else 0.0,
            "mean_f1_delta": float(np.mean(f1_deltas)) if f1_deltas else 0.0,
            "median_f1_delta": float(np.median(f1_deltas)) if f1_deltas else 0.0,
            "count_nonnegative_f1_delta": int(count_nonnegative),
            "share_nonnegative_f1_delta": float(share_nonnegative),
            "mean_score_delta_hi_2_0": float(np.mean(score_deltas)) if score_deltas else 0.0,
            "median_score_delta_hi_2_0": float(np.median(score_deltas)) if score_deltas else 0.0,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        },
    }


def build_cross_config_summary(config_results: list[dict[str, object]]) -> dict[str, object]:
    mixed_configs = [
        result for result in config_results if 0.0 < result["weights"]["w_rad"] < 1.0
    ]
    passing_mixed = [result for result in mixed_configs if result["aggregate"]["verdict"] == "PASS"]
    positive_mean_configs = [
        result for result in mixed_configs if result["aggregate"]["mean_f1_delta"] > 0.0
    ]
    best_config = max(
        config_results,
        key=lambda result: result["aggregate"]["mean_f1_delta"],
    )
    worst_config = min(
        config_results,
        key=lambda result: result["aggregate"]["mean_f1_delta"],
    )

    patient_ids = [
        patient["patient_id"]
        for patient in config_results[0]["patients"]
        if not patient["skipped"]
    ]
    patient_stability: list[dict[str, object]] = []
    for patient_id in patient_ids:
        patient_runs = []
        for config in config_results:
            for patient in config["patients"]:
                if patient["patient_id"] == patient_id and not patient["skipped"]:
                    patient_runs.append(
                        {
                            "label": config["label"],
                            "w_ang": config["weights"]["w_ang"],
                            "w_rad": config["weights"]["w_rad"],
                            "f1_delta": patient["deltas"]["f1_delta"],
                            "score_delta_hi_2_0": patient["score_means"]["delta_hi_2_0_mean"],
                        }
                    )
                    break

        signs = [np.sign(run["f1_delta"]) for run in patient_runs]
        best_run = max(patient_runs, key=lambda run: run["f1_delta"])
        worst_run = min(patient_runs, key=lambda run: run["f1_delta"])
        patient_stability.append(
            {
                "patient_id": patient_id,
                "always_nonnegative_f1_delta": bool(all(run["f1_delta"] >= 0.0 for run in patient_runs)),
                "always_negative_f1_delta": bool(all(run["f1_delta"] < 0.0 for run in patient_runs)),
                "sign_changes": int(sum(1 for idx in range(1, len(signs)) if signs[idx] != signs[idx - 1])),
                "best_config": best_run,
                "worst_config": worst_run,
            }
        )

    if len(passing_mixed) >= 2:
        verdict = "PASS"
        verdict_reason = "At least two mixed angular/radial settings met the patient-level stability target, so the HI 2.0 gain is not confined to a single fragile weight point."
    elif positive_mean_configs:
        verdict = "PARTIAL"
        verdict_reason = "Some mixed settings remained positive, but stability was too narrow or too patient-fragile for a strong claim."
    else:
        verdict = "FAIL"
        verdict_reason = "No mixed angular/radial setting preserved a stable positive patient-level gain."

    return {
        "n_configs_total": int(len(config_results)),
        "n_configs_mixed": int(len(mixed_configs)),
        "n_mixed_configs_pass": int(len(passing_mixed)),
        "n_mixed_configs_positive_mean_f1": int(len(positive_mean_configs)),
        "best_config_by_mean_f1_delta": {
            "label": best_config["label"],
            "weights": best_config["weights"],
            "aggregate": best_config["aggregate"],
        },
        "worst_config_by_mean_f1_delta": {
            "label": worst_config["label"],
            "weights": worst_config["weights"],
            "aggregate": worst_config["aggregate"],
        },
        "patient_stability": patient_stability,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def make_result_document(
    data_root: Path,
    patient_ids: list[str],
    config_results: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "experiment": {
            "id": "eeg_hi2_0_weight_stability_sweep_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_weight_stability_sweep.py",
            "hypothesis": "If the HI 2.0 gain is QA-stable rather than a single-point artifact, multiple mixed angular/radial weight settings should preserve a positive mean F1 delta and a mostly non-negative patient-level sign pattern across the reduced-pressure 10-patient CHB-MIT slice.",
            "success_criteria": "PASS if at least two mixed angular/radial settings achieve mean F1 delta > 0 and at least 70% non-negative patient deltas; PARTIAL if only a narrower subset stays positive; FAIL if no mixed setting preserves the gain.",
            "data_root": str(data_root),
            "patients_requested": patient_ids,
            "window_seconds": WINDOW_SEC,
            "overlap_seconds": OVERLAP_SEC,
            "min_segments_per_class": MIN_SEGMENTS_PER_CLASS,
            "max_segments_per_class": MAX_SEGMENTS_PER_CLASS,
            "test_size": TEST_SIZE,
            "weight_configs": WEIGHT_CONFIGS,
            "open_brain_context": [
                "2026-03-29 chromogeometric audit on chb07 versus chb08 found a structured counterexample rather than score noise.",
                "2026-03-29 combined 10-patient HI 2.0 slice passed only modestly, with chb08 as the main negative patient.",
            ],
        },
        "weight_configs": config_results,
        "cross_config_summary": build_cross_config_summary(config_results),
    }


def print_summary(doc: dict[str, object]) -> None:
    print("=" * 80)
    print("EEG HI 2.0 WEIGHT-STABILITY SWEEP")
    print("=" * 80)
    print(f"Data root: {doc['experiment']['data_root']}")
    print(f"Patients:  {', '.join(doc['experiment']['patients_requested'])}")
    print()
    print("Config-level summary:")
    print(f"  {'Config':<28} {'Mean F1Δ':>9} {'Median F1Δ':>11} {'Nonneg':>8} {'Verdict':>8}")
    print(f"  {'-'*28} {'-'*9} {'-'*11} {'-'*8} {'-'*8}")
    for config in doc["weight_configs"]:
        aggregate = config["aggregate"]
        print(
            f"  {config['label']:<28} "
            f"{aggregate['mean_f1_delta']:>+9.4f} "
            f"{aggregate['median_f1_delta']:>+11.4f} "
            f"{aggregate['count_nonnegative_f1_delta']:>3}/{aggregate['n_patients_evaluable']:<4} "
            f"{aggregate['verdict']:>8}"
        )
    print()
    summary = doc["cross_config_summary"]
    print(f"Cross-config verdict: {summary['verdict']}")
    print(f"Reason: {summary['verdict_reason']}")
    print(
        "Best config by mean F1 delta: "
        f"{summary['best_config_by_mean_f1_delta']['label']} "
        f"({summary['best_config_by_mean_f1_delta']['aggregate']['mean_f1_delta']:+.4f})"
    )
    print(
        "Worst config by mean F1 delta: "
        f"{summary['worst_config_by_mean_f1_delta']['label']} "
        f"({summary['worst_config_by_mean_f1_delta']['aggregate']['mean_f1_delta']:+.4f})"
    )
    print("Patient stability:")
    for patient in summary["patient_stability"]:
        print(
            f"  {patient['patient_id']}: "
            f"always_nonnegative={patient['always_nonnegative_f1_delta']} "
            f"always_negative={patient['always_negative_f1_delta']} "
            f"sign_changes={patient['sign_changes']} "
            f"best={patient['best_config']['label']}({patient['best_config']['f1_delta']:+.3f}) "
            f"worst={patient['worst_config']['label']}({patient['worst_config']['f1_delta']:+.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep HI 2.0 angular/radial weights across a reduced-pressure CHB-MIT slice."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patient", nargs="*", help="Optional patient ids to evaluate")
    parser.add_argument(
        "--patient-limit",
        type=int,
        default=DEFAULT_PATIENT_LIMIT,
        help="Maximum number of patients to evaluate when --patient is not supplied",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"Data root not found: {args.data_root}")

    patient_dirs = sorted(path for path in args.data_root.glob("chb*/") if path.is_dir())
    if args.patient:
        wanted = set(args.patient)
        patient_dirs = [path for path in patient_dirs if path.name in wanted]
    else:
        patient_dirs = patient_dirs[: max(0, args.patient_limit)]

    if not patient_dirs:
        raise SystemExit("No patient directories selected.")

    extractor = EEGBrainFeatureExtractor(sample_rate=256)
    patient_datasets = [build_balanced_patient_dataset(path, extractor) for path in patient_dirs]
    config_results = [
        evaluate_weight_config(
            patient_datasets,
            config["label"],
            config["w_ang"],
            config["w_rad"],
            config["w_fam"],
        )
        for config in WEIGHT_CONFIGS
    ]

    doc = make_result_document(args.data_root, [path.name for path in patient_dirs], config_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print_summary(doc)
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
