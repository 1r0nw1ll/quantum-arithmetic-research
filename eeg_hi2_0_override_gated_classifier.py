#!/usr/bin/env python3
"""
eeg_hi2_0_override_gated_classifier.py
======================================

Conservative override gate for reduced-pressure CHB-MIT HI 2.0.

Purpose:
- default to the full HI 2.0 feature family
- only reroute to coordinate-dominant families when validation evidence is
  strong enough to justify the switch
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
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_override_gated_classifier.json")
WINDOW_SEC = 4.0
OVERLAP_SEC = 2.0
MAX_SEGMENTS_PER_CLASS = 64
TEST_SIZE = 0.25
INNER_VAL_SIZE = 0.25
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 8
HI2_CONFIG = {"w_ang": 0.5, "w_rad": 0.5, "w_fam": 0.0}
FEATURE_NAMES = ["HI_2.0", "H_angular", "H_radial", "H_family", "C", "F", "G", "gcd", "b", "e"]
FEATURE_SETS = {
    "full": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "no_geometry": [0, 1, 2, 3, 8, 9],
    "coords_only": [8, 9],
}
OVERRIDE_FAMILIES = ["coords_only", "no_geometry"]
OVERRIDE_MARGIN = 0.08
FULL_WEAK_THRESHOLD = 0.55


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


def load_patient_ids(weight_sweep_path: Path) -> list[str]:
    doc = json.loads(weight_sweep_path.read_text(encoding="utf-8"))
    return list(doc["experiment"]["patients_requested"])


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


def build_hi1_matrix(features_7d: np.ndarray) -> np.ndarray:
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


def build_hi2_matrix(features_7d: np.ndarray) -> np.ndarray:
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


def eval_f1(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    clf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(x[train_idx], y[train_idx])
    return float(f1_score(y[test_idx], clf.predict(x[test_idx]), zero_division=0))


def choose_override_family(
    hi2: np.ndarray,
    labels: np.ndarray,
    outer_train_idx: np.ndarray,
    patient_id: str,
) -> tuple[str, dict[str, float], dict[str, object]]:
    train_sub_idx, val_sub_idx = train_test_split(
        outer_train_idx,
        test_size=INNER_VAL_SIZE,
        random_state=patient_seed(patient_id) ^ 0x5A5A5A5A,
        stratify=labels[outer_train_idx],
    )
    validation_scores = {
        family: eval_f1(hi2[:, cols], labels, train_sub_idx, val_sub_idx)
        for family, cols in FEATURE_SETS.items()
    }
    selected_family = "full"
    full_score = validation_scores["full"]
    best_override = max(OVERRIDE_FAMILIES, key=lambda family: validation_scores[family])
    override_gap = float(validation_scores[best_override] - full_score)
    override_triggered = bool(
        full_score <= FULL_WEAK_THRESHOLD and override_gap >= OVERRIDE_MARGIN
    )
    if override_triggered:
        selected_family = best_override
    return selected_family, validation_scores, {
        "full_validation_f1": float(full_score),
        "best_override_family": best_override,
        "best_override_validation_f1": float(validation_scores[best_override]),
        "override_gap": float(override_gap),
        "override_margin": float(OVERRIDE_MARGIN),
        "full_weak_threshold": float(FULL_WEAK_THRESHOLD),
        "override_triggered": override_triggered,
    }


def evaluate_patient(patient_id: str, data_root: Path) -> dict[str, object]:
    print(f"[{patient_id}] starting", flush=True)
    features_7d, labels = build_balanced_patient_dataset(patient_id, data_root)
    hi1 = build_hi1_matrix(features_7d)
    hi2 = build_hi2_matrix(features_7d)
    indices = np.arange(len(labels))
    outer_train_idx, outer_test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=patient_seed(patient_id) ^ 0xA5A5A5A5,
        stratify=labels,
    )

    hi1_f1 = eval_f1(hi1, labels, outer_train_idx, outer_test_idx)
    full_f1 = eval_f1(hi2[:, FEATURE_SETS["full"]], labels, outer_train_idx, outer_test_idx)
    selected_family, validation_scores, decision = choose_override_family(hi2, labels, outer_train_idx, patient_id)
    gated_f1 = eval_f1(hi2[:, FEATURE_SETS[selected_family]], labels, outer_train_idx, outer_test_idx)

    return {
        "patient_id": patient_id,
        "n_segments_balanced": int(len(labels)),
        "f1_hi1": hi1_f1,
        "f1_full_hi2": full_f1,
        "selected_family": selected_family,
        "selected_features": [FEATURE_NAMES[idx] for idx in FEATURE_SETS[selected_family]],
        "inner_validation_f1": validation_scores,
        "decision": decision,
        "f1_override_gated": gated_f1,
        "deltas": {
            "override_minus_full": float(gated_f1 - full_f1),
            "override_minus_hi1": float(gated_f1 - hi1_f1),
            "full_minus_hi1": float(full_f1 - hi1_f1),
        },
    }


def aggregate_results(patients: list[dict[str, object]]) -> dict[str, object]:
    override_minus_full = [patient["deltas"]["override_minus_full"] for patient in patients]
    nonnegative_count = sum(1 for value in override_minus_full if value >= 0.0)
    share_nonnegative = nonnegative_count / len(patients) if patients else 0.0
    family_counts: dict[str, int] = {name: 0 for name in FEATURE_SETS}
    overrides = []
    for patient in patients:
        family_counts[patient["selected_family"]] += 1
        if patient["decision"]["override_triggered"]:
            overrides.append(patient["patient_id"])

    hard_negative_reroutes = [
        patient["patient_id"]
        for patient in patients
        if patient["patient_id"] in {"chb02", "chb08"}
        and patient["decision"]["override_triggered"]
        and patient["deltas"]["override_minus_full"] > 0.0
    ]

    if override_minus_full and float(np.mean(override_minus_full)) > 0.0 and share_nonnegative >= 0.8 and hard_negative_reroutes:
        verdict = "PASS"
        verdict_reason = "The conservative override gate improved mean F1 over full HI 2.0, kept most patients non-negative, and rescued at least one hard-negative patient."
    elif override_minus_full and (float(np.mean(override_minus_full)) > 0.0 or hard_negative_reroutes):
        verdict = "PARTIAL"
        verdict_reason = "The override gate showed some benefit, but not enough for a strong aggregate claim."
    else:
        verdict = "FAIL"
        verdict_reason = "The conservative override gate did not improve over the fixed full HI 2.0 representation."

    return {
        "n_patients": int(len(patients)),
        "mean_f1_hi1": float(np.mean([patient["f1_hi1"] for patient in patients])) if patients else 0.0,
        "mean_f1_full_hi2": float(np.mean([patient["f1_full_hi2"] for patient in patients])) if patients else 0.0,
        "mean_f1_override_gated": float(np.mean([patient["f1_override_gated"] for patient in patients])) if patients else 0.0,
        "mean_override_minus_full": float(np.mean(override_minus_full)) if override_minus_full else 0.0,
        "median_override_minus_full": float(np.median(override_minus_full)) if override_minus_full else 0.0,
        "count_nonnegative_override_minus_full": int(nonnegative_count),
        "share_nonnegative_override_minus_full": float(share_nonnegative),
        "selected_family_counts": family_counts,
        "override_triggered_patients": overrides,
        "hard_negative_reroutes": hard_negative_reroutes,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def print_summary(doc: dict[str, object]) -> None:
    print("=" * 80)
    print("EEG HI 2.0 OVERRIDE-GATED CLASSIFIER")
    print("=" * 80)
    print(f"Patients: {', '.join(doc['experiment']['patients'])}")
    print()
    print(f"{'Patient':<8} {'Family':<12} {'Trig':<5} {'HI1':>7} {'Full':>7} {'Gate':>7} {'O-F':>7}")
    print(f"{'-'*8} {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for patient in doc["patients"]:
        print(
            f"{patient['patient_id']:<8} {patient['selected_family']:<12} "
            f"{str(patient['decision']['override_triggered']):<5} "
            f"{patient['f1_hi1']:>7.3f} {patient['f1_full_hi2']:>7.3f} "
            f"{patient['f1_override_gated']:>7.3f} {patient['deltas']['override_minus_full']:>+7.3f}"
        )
    print()
    aggregate = doc["aggregate"]
    print(f"Mean HI1 F1:       {aggregate['mean_f1_hi1']:.4f}")
    print(f"Mean full HI2 F1:  {aggregate['mean_f1_full_hi2']:.4f}")
    print(f"Mean override F1:  {aggregate['mean_f1_override_gated']:.4f}")
    print(f"Mean override-full:{aggregate['mean_override_minus_full']:+.4f}")
    print(f"Triggered overrides: {aggregate['override_triggered_patients']}")
    print(f"Hard-negative reroutes: {aggregate['hard_negative_reroutes']}")
    print(f"Verdict: {aggregate['verdict']}")
    print(f"Reason:  {aggregate['verdict_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conservative override gate over reduced-pressure CHB-MIT HI 2.0 families.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weight-sweep", type=Path, default=DEFAULT_WEIGHT_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    patient_ids = load_patient_ids(args.weight_sweep)
    patients = [evaluate_patient(patient_id, args.data_root) for patient_id in patient_ids]
    doc = {
        "experiment": {
            "id": "eeg_hi2_0_override_gated_classifier_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_override_gated_classifier.py",
            "hypothesis": "A conservative override gate that defaults to full HI 2.0 and reroutes only when a coordinate-dominant family wins validation by margin under weak-full conditions improves aggregate F1 over fixed full HI 2.0.",
            "success_criteria": "PASS if mean override-minus-full F1 is positive, at least 80% of patients are non-negative, and at least one hard-negative patient is rerouted with improvement; PARTIAL if mixed; FAIL otherwise.",
            "data_root": str(args.data_root),
            "weight_sweep_artifact": str(args.weight_sweep),
            "patients": patient_ids,
            "feature_sets": {name: [FEATURE_NAMES[idx] for idx in cols] for name, cols in FEATURE_SETS.items()},
            "override_families": OVERRIDE_FAMILIES,
            "override_margin": OVERRIDE_MARGIN,
            "full_weak_threshold": FULL_WEAK_THRESHOLD,
            "hi2_config": HI2_CONFIG,
        },
        "patients": patients,
        "aggregate": aggregate_results(patients),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print_summary(doc)
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
