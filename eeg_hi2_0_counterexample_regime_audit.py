#!/usr/bin/env python3
"""
eeg_hi2_0_counterexample_regime_audit.py
========================================

Focused chromogeometric counterexample study for HI 2.0 on CHB-MIT.

Purpose:
- derive stable-positive and hard-negative patient cohorts from the existing
  weight-stability sweep
- audit whether the counterexample cohort occupies a distinct seizure regime
  in canonical HI 2.0 / chromogeometric quantities

This is a standalone diagnostic experiment. It does not define a new classifier.
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
from qa_harmonicity_v2 import PISANO_FAMILY_MAP, compute_hi_1_0, compute_hi_2_0, digital_root, qa_tuple


DEFAULT_DATA_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
DEFAULT_WEIGHT_SWEEP = Path("results/eeg_hi2_0_weight_stability_sweep_10patients.json")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_counterexample_regime_audit.json")
WINDOW_SEC = 4.0
OVERLAP_SEC = 2.0
MAX_SEGMENTS_PER_CLASS = 64
TEST_SIZE = 0.25
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 8
HI2_CONFIG = {"w_ang": 0.5, "w_rad": 0.5, "w_fam": 0.0}

MEAN_KEYS = [
    "HI_1.0",
    "HI_2.0",
    "H_angular",
    "H_radial",
    "H_family",
    "C",
    "F",
    "G",
    "I",
    "gcd",
    "b",
    "e",
]
RATE_KEYS = [
    "primitive_rate",
    "female_rate",
    "composite_rate",
    "fermat_rate",
    "pythagoras_rate",
    "plato_rate",
    "none_family_rate",
    "pisano_24cycle_rate",
    "pisano_tribonacci_rate",
    "pisano_ninbonacci_rate",
    "pisano_unknown_rate",
]


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
        keys = sorted(
            (key for key in selected if key[0] == meta["file"]),
            key=lambda item: item[1],
        )
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


def chromogeometric_rows(features_7d: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feat in features_7d:
        b = max(1, min(24, int(feat[0] * 23) + 1))
        e = max(1, min(24, int(feat[1] * 23) + 1))
        q = qa_tuple(b, e, modulus=24)
        result_hi1 = compute_hi_1_0(q, modulus=24)
        result_hi2 = compute_hi_2_0(
            q,
            w_ang=HI2_CONFIG["w_ang"],
            w_rad=HI2_CONFIG["w_rad"],
            w_fam=HI2_CONFIG["w_fam"],
            modulus=24,
        )
        c_val, f_val, g_val = result_hi2["pythagorean_triple"]
        dr_b = digital_root(b)
        dr_e = digital_root(e)
        pisano_family = PISANO_FAMILY_MAP.get((dr_b, dr_e), "Unknown")
        gcd_val = int(result_hi2["gcd"])
        rows.append(
            {
                "HI_1.0": float(result_hi1),
                "HI_2.0": float(result_hi2["HI_2.0"]),
                "H_angular": float(result_hi2["H_angular"]),
                "H_radial": float(result_hi2["H_radial"]),
                "H_family": float(result_hi2["H_family"]),
                "C": float(c_val),
                "F": float(f_val),
                "G": float(g_val),
                "I": float(abs(c_val - f_val)),
                "gcd": float(gcd_val),
                "b": float(b),
                "e": float(e),
                "primitive_rate": float(1 if gcd_val == 1 else 0),
                "female_rate": float(1 if gcd_val == 2 else 0),
                "composite_rate": float(1 if gcd_val > 2 else 0),
                "fermat_rate": float(1 if "Fermat" in result_hi2["families"] else 0),
                "pythagoras_rate": float(1 if "Pythagoras" in result_hi2["families"] else 0),
                "plato_rate": float(1 if "Plato" in result_hi2["families"] else 0),
                "none_family_rate": float(1 if result_hi2["families"] == ["None"] else 0),
                "pisano_24cycle_rate": float(1 if pisano_family in {"Fibonacci", "Lucas", "Phibonacci"} else 0),
                "pisano_tribonacci_rate": float(1 if pisano_family == "Tribonacci" else 0),
                "pisano_ninbonacci_rate": float(1 if pisano_family == "Ninbonacci" else 0),
                "pisano_unknown_rate": float(1 if pisano_family == "Unknown" else 0),
            }
        )
    return rows


def mean_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in MEAN_KEYS + RATE_KEYS:
        out[key] = float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
    return out


def patient_regime_audit(patient_id: str, data_root: Path) -> dict[str, object]:
    features_7d, labels = build_balanced_patient_dataset(patient_id, data_root)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=patient_seed(patient_id) ^ 0xA5A5A5A5,
        stratify=labels,
    )

    hi1_features = np.zeros((len(features_7d), 4), dtype=np.float64)
    hi2_features = np.zeros((len(features_7d), 10), dtype=np.float64)
    for idx, feat in enumerate(features_7d):
        b = max(1, min(24, int(feat[0] * 23) + 1))
        e = max(1, min(24, int(feat[1] * 23) + 1))
        q = qa_tuple(b, e, modulus=24)
        hi1 = compute_hi_1_0(q, modulus=24)
        hi2 = compute_hi_2_0(
            q,
            w_ang=HI2_CONFIG["w_ang"],
            w_rad=HI2_CONFIG["w_rad"],
            w_fam=HI2_CONFIG["w_fam"],
            modulus=24,
        )
        c_val, f_val, g_val = hi2["pythagorean_triple"]
        hi1_features[idx] = [hi1, b / 24.0, e / 24.0, float(np.linalg.norm(feat))]
        hi2_features[idx] = [
            hi2["HI_2.0"],
            hi2["H_angular"],
            hi2["H_radial"],
            hi2["H_family"],
            c_val / 1000.0,
            f_val / 1000.0,
            g_val / 1000.0,
            hi2["gcd"],
            b / 24.0,
            e / 24.0,
        ]

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
    clf1.fit(hi1_features[train_idx], labels[train_idx])
    clf2.fit(hi2_features[train_idx], labels[train_idx])

    f1_hi1 = float(f1_score(labels[test_idx], clf1.predict(hi1_features[test_idx]), zero_division=0))
    f1_hi2 = float(f1_score(labels[test_idx], clf2.predict(hi2_features[test_idx]), zero_division=0))

    rows = chromogeometric_rows(features_7d)
    seizure_rows = [row for row, label in zip(rows, labels) if label == 1]
    baseline_rows = [row for row, label in zip(rows, labels) if label == 0]
    seizure_mean = mean_summary(seizure_rows)
    baseline_mean = mean_summary(baseline_rows)
    deltas = {
        key: float(seizure_mean[key] - baseline_mean[key])
        for key in MEAN_KEYS + RATE_KEYS
    }

    return {
        "patient_id": patient_id,
        "n_segments_balanced": int(len(labels)),
        "n_seizure": int((labels == 1).sum()),
        "n_baseline": int((labels == 0).sum()),
        "f1_hi1": f1_hi1,
        "f1_hi2": f1_hi2,
        "f1_delta": float(f1_hi2 - f1_hi1),
        "seizure_means": seizure_mean,
        "baseline_means": baseline_mean,
        "deltas": deltas,
    }


def cohort_summary(name: str, patient_audits: list[dict[str, object]]) -> dict[str, object]:
    delta_keys = MEAN_KEYS + RATE_KEYS
    return {
        "cohort": name,
        "patients": [audit["patient_id"] for audit in patient_audits],
        "n_patients": int(len(patient_audits)),
        "mean_f1_delta": float(np.mean([audit["f1_delta"] for audit in patient_audits])) if patient_audits else 0.0,
        "median_f1_delta": float(np.median([audit["f1_delta"] for audit in patient_audits])) if patient_audits else 0.0,
        "mean_patient_delta": {
            key: float(np.mean([audit["deltas"][key] for audit in patient_audits])) if patient_audits else 0.0
            for key in delta_keys
        },
    }


def compare_cohorts(stable: dict[str, object], negative: dict[str, object]) -> dict[str, object]:
    stable_delta = stable["mean_patient_delta"]
    negative_delta = negative["mean_patient_delta"]
    gap = {key: float(stable_delta[key] - negative_delta[key]) for key in stable_delta}

    criteria = {
        "f1_sign_split": bool(stable["mean_f1_delta"] > 0.0 and negative["mean_f1_delta"] < 0.0),
        "hi2_sign_split": bool(stable_delta["HI_2.0"] > 0.0 and negative_delta["HI_2.0"] < 0.0),
        "radial_sign_split": bool(stable_delta["H_radial"] >= 0.0 and negative_delta["H_radial"] < 0.0),
        "primitivity_split": bool(
            (stable_delta["primitive_rate"] >= 0.0 and negative_delta["primitive_rate"] < 0.0)
            or (stable_delta["gcd"] < negative_delta["gcd"])
        ),
    }
    criteria_count = sum(1 for value in criteria.values() if value)

    if criteria_count >= 4:
        verdict = "PASS"
        verdict_reason = "Hard negatives separated from stable positives on the preregistered classifier, HI 2.0, radial, and primitivity axes."
    elif criteria_count >= 2:
        verdict = "PARTIAL"
        verdict_reason = "Some regime separation was present, but not enough for a strong counterexample-cohort claim."
    else:
        verdict = "FAIL"
        verdict_reason = "The hard-negative cohort did not separate cleanly from the stable-positive cohort on the preregistered regime axes."

    return {
        "criteria": criteria,
        "criteria_count": int(criteria_count),
        "group_delta_gap": gap,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def print_summary(doc: dict[str, object]) -> None:
    print("=" * 80)
    print("EEG HI 2.0 COUNTEREXAMPLE REGIME AUDIT")
    print("=" * 80)
    print(f"Stable positives: {', '.join(doc['cohorts']['stable_positive']['patients'])}")
    print(f"Hard negatives:   {', '.join(doc['cohorts']['hard_negative']['patients'])}")
    print()
    for name in ["stable_positive", "hard_negative"]:
        cohort = doc["cohorts"][name]
        delta = cohort["mean_patient_delta"]
        print(f"{name}: mean_f1_delta={cohort['mean_f1_delta']:+.4f} HI2Δ={delta['HI_2.0']:+.4f} HradΔ={delta['H_radial']:+.4f} primitiveΔ={delta['primitive_rate']:+.4f} gcdΔ={delta['gcd']:+.4f}")
    print()
    comparison = doc["comparison"]
    print(f"Verdict: {comparison['verdict']}")
    print(f"Reason:  {comparison['verdict_reason']}")
    print("Criteria:")
    for key, value in comparison["criteria"].items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chromogeometric regime separation between stable-positive and hard-negative EEG patients.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weight-sweep", type=Path, default=DEFAULT_WEIGHT_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stable_positive, hard_negative = load_cohorts(args.weight_sweep)
    stable_audits = [patient_regime_audit(patient_id, args.data_root) for patient_id in stable_positive]
    negative_audits = [patient_regime_audit(patient_id, args.data_root) for patient_id in hard_negative]

    stable_summary = cohort_summary("stable_positive", stable_audits)
    negative_summary = cohort_summary("hard_negative", negative_audits)
    comparison = compare_cohorts(stable_summary, negative_summary)

    doc = {
        "experiment": {
            "id": "eeg_hi2_0_counterexample_regime_audit_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_counterexample_regime_audit.py",
            "hypothesis": "The hard-negative HI 2.0 patients identified by the weight-stability sweep occupy a distinct seizure regime from the stable-positive patients, with opposite-signed HI 2.0 response and weaker radial/primitivity response rather than merely lower classifier luck.",
            "success_criteria": "PASS if stable positives and hard negatives separate on the preregistered classifier, HI 2.0, radial, and primitivity axes; PARTIAL if only a subset separates; FAIL otherwise.",
            "data_root": str(args.data_root),
            "weight_sweep_artifact": str(args.weight_sweep),
            "hi2_config": HI2_CONFIG,
        },
        "cohorts": {
            "stable_positive": stable_summary,
            "hard_negative": negative_summary,
        },
        "patients": {
            "stable_positive": stable_audits,
            "hard_negative": negative_audits,
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
