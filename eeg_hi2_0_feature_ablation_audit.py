#!/usr/bin/env python3
"""
eeg_hi2_0_feature_ablation_audit.py
===================================

Feature-ablation audit for HI 2.0 EEG cohorts.

Purpose:
- derive stable-positive and hard-negative cohorts from the existing
  weight-stability artifact
- test whether cohort differences live in chromogeometric features,
  coordinate features, or their interaction
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from eeg_brain_feature_extractor_fixed import EEGBrainFeatureExtractor
from qa_harmonicity_v2 import compute_hi_1_0, compute_hi_2_0, qa_tuple


DEFAULT_DATA_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
DEFAULT_WEIGHT_SWEEP = Path("results/eeg_hi2_0_weight_stability_sweep_10patients.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_feature_ablation_audit.json")
WINDOW_SEC = 4.0
OVERLAP_SEC = 2.0
MAX_SEGMENTS_PER_CLASS = 64
TEST_SIZE = 0.25
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 8
HI2_CONFIG = {"w_ang": 0.5, "w_rad": 0.5, "w_fam": 0.0}
FEATURE_NAMES = ["HI_2.0", "H_angular", "H_radial", "H_family", "C", "F", "G", "gcd", "b", "e"]
FEATURE_SETS = {
    "full": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "no_coords": [0, 1, 2, 3, 4, 5, 6, 7],
    "no_geometry": [0, 1, 2, 3, 8, 9],
    "geometry_only": [4, 5, 6, 7],
    "coords_only": [8, 9],
}


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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


def load_cohorts(weight_sweep_path: Path) -> tuple[list[str], list[str]]:
    doc = json.loads(weight_sweep_path.read_text(encoding="utf-8"))
    patient_stability = doc["cross_config_summary"]["patient_stability"]
    stable_positive = [
        row["patient_id"] for row in patient_stability if row["always_nonnegative_f1_delta"]
    ]
    hard_negative = [
        row["patient_id"] for row in patient_stability if row["always_negative_f1_delta"]
    ]
    return stable_positive, hard_negative


def build_balanced_patient_dataset(patient_id: str, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    patient_dir = data_root / patient_id
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

    target_count = min(len(seizure_candidates), len(baseline_candidates), MAX_SEGMENTS_PER_CLASS)
    rng = np.random.default_rng(patient_seed(patient_id))
    baseline_pick_idx = rng.choice(len(baseline_candidates), target_count, replace=False)
    seizure_pick_idx = rng.choice(len(seizure_candidates), target_count, replace=False)
    selected = {baseline_candidates[idx]: 0 for idx in baseline_pick_idx}
    for idx in seizure_pick_idx:
        selected[seizure_candidates[idx]] = 1

    features: list[np.ndarray] = []
    labels_out: list[int] = []
    for meta in candidate_meta:
        keys = sorted((key for key in selected if key[0] == meta["file"]), key=lambda item: item[1])
        if not keys:
            continue
        signals, sample_rate, _ = read_edf_all_channels(meta["edf_path"])
        segments = segment_signals(signals, sample_rate)
        chosen = [segments[key[1]] for key in keys]
        file_features = extract_7d_features(chosen, meta["channel_names"], meta["extractor"], f"{patient_id}:{meta['file']}")
        features.extend(file_features)
        labels_out.extend(selected[key] for key in keys)

    features_arr = np.array(features, dtype=np.float64)
    labels_arr = np.array(labels_out, dtype=int)
    shuffle_idx = np.arange(len(labels_arr))
    rng.shuffle(shuffle_idx)
    return features_arr[shuffle_idx], labels_arr[shuffle_idx]


def build_hi2_feature_matrix(features_7d: np.ndarray) -> np.ndarray:
    hi2_features = np.zeros((len(features_7d), 10), dtype=np.float64)
    for idx, feat in enumerate(features_7d):
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
        hi2_features[idx] = [
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
    return hi2_features


def build_hi1_feature_matrix(features_7d: np.ndarray) -> np.ndarray:
    hi1_features = np.zeros((len(features_7d), 4), dtype=np.float64)
    for idx, feat in enumerate(features_7d):
        b = max(1, min(24, int(feat[0] * 23) + 1))
        e = max(1, min(24, int(feat[1] * 23) + 1))
        q = qa_tuple(b, e, modulus=24)
        hi1_features[idx] = [
            compute_hi_1_0(q, modulus=24),
            b / 24.0,
            e / 24.0,
            float(np.linalg.norm(feat)),
        ]
    return hi1_features


def eval_f1(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    clf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(x[train_idx], y[train_idx])
    return float(f1_score(y[test_idx], clf.predict(x[test_idx]), zero_division=0))


def audit_patient(patient_id: str, data_root: Path) -> dict[str, object]:
    features_7d, labels = build_balanced_patient_dataset(patient_id, data_root)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=patient_seed(patient_id) ^ 0xA5A5A5A5,
        stratify=labels,
    )

    hi1 = build_hi1_feature_matrix(features_7d)
    hi2 = build_hi2_feature_matrix(features_7d)
    hi1_f1 = eval_f1(hi1, labels, train_idx, test_idx)

    results: dict[str, dict[str, object]] = {}
    full_f1 = 0.0
    for name, cols in FEATURE_SETS.items():
        f1_value = eval_f1(hi2[:, cols], labels, train_idx, test_idx)
        results[name] = {
            "f1": f1_value,
            "columns": cols,
            "features": [FEATURE_NAMES[idx] for idx in cols],
        }
        if name == "full":
            full_f1 = f1_value

    for name in FEATURE_SETS:
        results[name]["drop_from_full"] = float(full_f1 - results[name]["f1"])

    return {
        "patient_id": patient_id,
        "n_segments_balanced": int(len(labels)),
        "f1_hi1": hi1_f1,
        "feature_sets": results,
    }


def cohort_summary(name: str, patients: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"cohort": name, "patients": [p["patient_id"] for p in patients], "n_patients": int(len(patients))}
    feature_sets: dict[str, object] = {}
    for set_name in FEATURE_SETS:
        f1s = [patient["feature_sets"][set_name]["f1"] for patient in patients]
        drops = [patient["feature_sets"][set_name]["drop_from_full"] for patient in patients]
        feature_sets[set_name] = {
            "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
            "median_f1": float(np.median(f1s)) if f1s else 0.0,
            "mean_drop_from_full": float(np.mean(drops)) if drops else 0.0,
            "median_drop_from_full": float(np.median(drops)) if drops else 0.0,
        }
    hi1_values = [patient["f1_hi1"] for patient in patients]
    summary["mean_f1_hi1"] = float(np.mean(hi1_values)) if hi1_values else 0.0
    summary["feature_sets"] = feature_sets
    return summary


def compare_cohorts(stable: dict[str, object], negative: dict[str, object]) -> dict[str, object]:
    stable_sets = stable["feature_sets"]
    negative_sets = negative["feature_sets"]
    criteria = {
        "stable_geometry_only_advantage": bool(stable_sets["geometry_only"]["mean_f1"] > negative_sets["geometry_only"]["mean_f1"]),
        "negative_coords_only_advantage": bool(negative_sets["coords_only"]["mean_f1"] > stable_sets["coords_only"]["mean_f1"]),
        "geometry_drop_harder_for_stable": bool(stable_sets["no_geometry"]["mean_drop_from_full"] > negative_sets["no_geometry"]["mean_drop_from_full"]),
        "coords_drop_harder_for_negative": bool(negative_sets["no_coords"]["mean_drop_from_full"] > stable_sets["no_coords"]["mean_drop_from_full"]),
    }
    criteria_count = sum(1 for value in criteria.values() if value)
    if criteria_count >= 3:
        verdict = "PASS"
        verdict_reason = "Most preregistered ablation contrasts supported a geometry-vs-coordinate cohort split."
    elif criteria_count >= 2:
        verdict = "PARTIAL"
        verdict_reason = "The ablation pattern was mixed: some contrasts supported a cohort split, but not enough for a strong claim."
    else:
        verdict = "FAIL"
        verdict_reason = "The ablation contrasts did not support the preregistered geometry-vs-coordinate cohort split."
    return {
        "criteria": criteria,
        "criteria_count": int(criteria_count),
        "stable_positive": stable_sets,
        "hard_negative": negative_sets,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def print_summary(doc: dict[str, object]) -> None:
    print("=" * 80)
    print("EEG HI 2.0 FEATURE ABLATION AUDIT")
    print("=" * 80)
    for cohort_name in ["stable_positive", "hard_negative"]:
        cohort = doc["cohorts"][cohort_name]
        print(f"{cohort_name}: mean HI1 F1={cohort['mean_f1_hi1']:.4f}")
        for set_name in FEATURE_SETS:
            item = cohort["feature_sets"][set_name]
            print(f"  {set_name:<14} mean_f1={item['mean_f1']:.4f} mean_drop={item['mean_drop_from_full']:+.4f}")
        print()
    comparison = doc["comparison"]
    print(f"Verdict: {comparison['verdict']}")
    print(f"Reason:  {comparison['verdict_reason']}")
    for key, value in comparison["criteria"].items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HI 2.0 feature ablations across stable-positive and hard-negative EEG cohorts.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weight-sweep", type=Path, default=DEFAULT_WEIGHT_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stable_positive, hard_negative = load_cohorts(args.weight_sweep)
    stable_patients = [audit_patient(patient_id, args.data_root) for patient_id in stable_positive]
    negative_patients = [audit_patient(patient_id, args.data_root) for patient_id in hard_negative]

    stable_summary = cohort_summary("stable_positive", stable_patients)
    negative_summary = cohort_summary("hard_negative", negative_patients)
    comparison = compare_cohorts(stable_summary, negative_summary)
    doc = {
        "experiment": {
            "id": "eeg_hi2_0_feature_ablation_audit_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_feature_ablation_audit.py",
            "hypothesis": "Stable-positive HI 2.0 patients depend more on chromogeometric structure while hard-negative patients depend more on raw b/e coordinates or coordinate-geometry interactions.",
            "success_criteria": "PASS if most preregistered ablation contrasts support a geometry-vs-coordinate cohort split; PARTIAL if mixed; FAIL otherwise.",
            "data_root": str(args.data_root),
            "weight_sweep_artifact": str(args.weight_sweep),
            "hi2_config": HI2_CONFIG,
            "feature_names": FEATURE_NAMES,
            "feature_sets": {name: [FEATURE_NAMES[idx] for idx in cols] for name, cols in FEATURE_SETS.items()},
        },
        "cohorts": {
            "stable_positive": stable_summary,
            "hard_negative": negative_summary,
        },
        "patients": {
            "stable_positive": stable_patients,
            "hard_negative": negative_patients,
        },
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print_summary(doc)
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
