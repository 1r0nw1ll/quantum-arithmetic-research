#!/usr/bin/env python3
"""
eeg_hi2_0_chromogeometry_audit.py
=================================

Targeted audit of HI 2.0 / chromogeometric structure for selected CHB-MIT patients.

Purpose:
- compare one or more positive patients against a negative or ambiguous patient
- inspect whether the failure pattern is visible in canonical QA quantities
  C, F, G, I and in the HI 2.0 subcomponents H_angular / H_radial

This is a focused diagnostic experiment, not a new classifier family.
"""

QA_COMPLIANCE = "empirical_observer — EEG signal is observer input; QA discrete orbit is the classifier state"


from __future__ import annotations

import argparse
import json
import os
import re
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from eeg_brain_feature_extractor_fixed import EEGBrainFeatureExtractor
from qa_harmonicity_v2 import compute_hi_1_0, compute_hi_2_0, qa_tuple


DATA_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
WINDOW_SEC = 4.0
OVERLAP_SEC = 2.0
MAX_SEGMENTS_PER_CLASS = 64
TEST_SIZE = 0.25
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 8
HI2_CONFIG = {"w_ang": 0.5, "w_rad": 0.5, "w_fam": 0.0}


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
) -> np.ndarray:
    features: list[np.ndarray] = []
    for segment in segments:
        channels_data = {
            channel_name: segment[idx, :]
            for idx, channel_name in enumerate(channel_names)
        }
        features.append(extractor.extract_network_features(channels_data))
    return np.array(features, dtype=np.float64)


def extract_hi_features(features_7d: np.ndarray, use_hi2: bool) -> np.ndarray:
    n_samples = len(features_7d)
    if use_hi2:
        out = np.zeros((n_samples, 10), dtype=np.float64)
        for idx in range(n_samples):
            b = max(1, min(24, int(features_7d[idx, 0] * 23) + 1))
            e = max(1, min(24, int(features_7d[idx, 1] * 23) + 1))
            q = qa_tuple(b, e, modulus=24)
            result = compute_hi_2_0(
                q,
                w_ang=HI2_CONFIG["w_ang"],
                w_rad=HI2_CONFIG["w_rad"],
                w_fam=HI2_CONFIG["w_fam"],
                modulus=24,
            )
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
        return out

    out = np.zeros((n_samples, 4), dtype=np.float64)
    for idx in range(n_samples):
        b = max(1, min(24, int(features_7d[idx, 0] * 23) + 1))
        e = max(1, min(24, int(features_7d[idx, 1] * 23) + 1))
        q = qa_tuple(b, e, modulus=24)
        out[idx] = [compute_hi_1_0(q, modulus=24), b / 24.0, e / 24.0, np.linalg.norm(features_7d[idx])]
    return out


def build_balanced_patient_dataset(patient_id: str) -> tuple[np.ndarray, np.ndarray]:
    patient_dir = DATA_ROOT / patient_id
    extractor = EEGBrainFeatureExtractor(sample_rate=256)
    summary = parse_summary(patient_dir / f"{patient_id}-summary.txt")

    candidate_meta: list[dict[str, object]] = []
    seizure_candidates: list[tuple[str, int]] = []
    baseline_candidates: list[tuple[str, int]] = []

    for filename, seizure_times in sorted(summary.items()):
        edf_path = patient_dir / filename
        if not edf_path.exists():
            continue
        signals, sample_rate, channel_names = read_edf_all_channels(edf_path)
        if sample_rate != extractor.sample_rate:
            extractor = EEGBrainFeatureExtractor(sample_rate=sample_rate)
        segments = segment_signals(signals, sample_rate)
        labels = label_segments(len(segments), seizure_times)
        candidate_meta.append(
            {
                "file": filename,
                "edf_path": edf_path,
                "channel_names": channel_names,
                "sample_rate": sample_rate,
                "labels": labels,
            }
        )
        for idx, value in enumerate(labels):
            key = (filename, idx)
            if value == 1:
                seizure_candidates.append(key)
            else:
                baseline_candidates.append(key)

    rng = np.random.default_rng(patient_seed(patient_id))
    target_count = min(len(seizure_candidates), len(baseline_candidates), MAX_SEGMENTS_PER_CLASS)
    baseline_pick_idx = rng.choice(len(baseline_candidates), target_count, replace=False)
    seizure_pick_idx = rng.choice(len(seizure_candidates), target_count, replace=False)
    selected = {baseline_candidates[idx]: 0 for idx in baseline_pick_idx}
    for idx in seizure_pick_idx:
        selected[seizure_candidates[idx]] = 1

    features: list[np.ndarray] = []
    labels: list[int] = []
    extractor = EEGBrainFeatureExtractor(sample_rate=256)
    for meta in candidate_meta:
        keys = sorted((key for key in selected if key[0] == meta["file"]), key=lambda item: item[1])
        if not keys:
            continue
        signals, sample_rate, channel_names = read_edf_all_channels(meta["edf_path"])
        if sample_rate != extractor.sample_rate:
            extractor = EEGBrainFeatureExtractor(sample_rate=sample_rate)
        segments = segment_signals(signals, sample_rate)
        chosen = [segments[key[1]] for key in keys]
        file_features = extract_7d_features(chosen, channel_names, extractor)
        features.extend(file_features)
        labels.extend(selected[key] for key in keys)

    features_arr = np.array(features, dtype=np.float64)
    labels_arr = np.array(labels, dtype=int)
    shuffle_idx = np.arange(len(labels_arr))
    rng.shuffle(shuffle_idx)
    return features_arr[shuffle_idx], labels_arr[shuffle_idx]


def chromogeometric_rows(features_7d: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for feat in features_7d:
        b = max(1, min(24, int(feat[0] * 23) + 1))
        e = max(1, min(24, int(feat[1] * 23) + 1))
        q = qa_tuple(b, e, modulus=24)
        result = compute_hi_2_0(
            q,
            w_ang=HI2_CONFIG["w_ang"],
            w_rad=HI2_CONFIG["w_rad"],
            w_fam=HI2_CONFIG["w_fam"],
            modulus=24,
        )
        c_val, f_val, g_val = result["pythagorean_triple"]
        rows.append(
            {
                "HI_2.0": float(result["HI_2.0"]),
                "H_angular": float(result["H_angular"]),
                "H_radial": float(result["H_radial"]),
                "C": float(c_val),
                "F": float(f_val),
                "G": float(g_val),
                "I": float(abs(c_val - f_val)),
                "b": float(b),
                "e": float(e),
            }
        )
    return rows


def summarize_group(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["HI_2.0", "H_angular", "H_radial", "C", "F", "G", "I", "b", "e"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def audit_patient(patient_id: str) -> dict[str, object]:
    features_7d, labels = build_balanced_patient_dataset(patient_id)
    hi1 = extract_hi_features(features_7d, use_hi2=False)
    hi2 = extract_hi_features(features_7d, use_hi2=True)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=patient_seed(patient_id) ^ 0xA5A5A5A5,
        stratify=labels,
    )
    clf1 = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
        class_weight="balanced",
    )
    clf2 = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
        class_weight="balanced",
    )
    clf1.fit(hi1[train_idx], labels[train_idx])
    clf2.fit(hi2[train_idx], labels[train_idx])
    f1_hi1 = float(f1_score(labels[test_idx], clf1.predict(hi1[test_idx]), zero_division=0))
    f1_hi2 = float(f1_score(labels[test_idx], clf2.predict(hi2[test_idx]), zero_division=0))

    rows = chromogeometric_rows(features_7d)
    seizure_rows = [row for row, label in zip(rows, labels) if label == 1]
    baseline_rows = [row for row, label in zip(rows, labels) if label == 0]

    return {
        "patient_id": patient_id,
        "n_segments_balanced": int(len(labels)),
        "f1_hi1": f1_hi1,
        "f1_hi2": f1_hi2,
        "f1_delta": float(f1_hi2 - f1_hi1),
        "seizure_means": summarize_group(seizure_rows),
        "baseline_means": summarize_group(baseline_rows),
        "deltas": {
            key: float(summarize_group(seizure_rows)[key] - summarize_group(baseline_rows)[key])
            for key in ["HI_2.0", "H_angular", "H_radial", "C", "F", "G", "I", "b", "e"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chromogeometric HI 2.0 structure for selected EEG patients.")
    parser.add_argument("--patients", nargs="+", required=True, help="Patient ids, e.g. chb07 chb08")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/eeg_hi2_0_chromogeometry_audit.json"),
        help="Where to write the audit JSON",
    )
    args = parser.parse_args()

    audits = [audit_patient(patient_id) for patient_id in args.patients]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audits, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audits, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
