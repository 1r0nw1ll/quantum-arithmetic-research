#!/usr/bin/env python3
"""
eeg_hi2_0_chbmit_scale.py
=========================

Scale the HI 1.0 vs HI 2.0 seizure-detection comparison to the locally
available CHB-MIT patient set.

Design goals:
- standalone, no network access
- patient-level results first, aggregate verdict second
- deterministic balancing and train/test splits
- honest PASS / PARTIAL / FAIL verdict against pre-declared criteria

Default data root is the canonical local archive path used elsewhere in the repo:
  archive/phase_artifacts/phase2_data/eeg/chbmit
"""

QA_COMPLIANCE = "empirical_observer — EEG signal is observer input; QA discrete orbit is the classifier state"


from __future__ import annotations

import argparse
import gc
import json
import os
import re
import zlib
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from eeg_brain_feature_extractor_fixed import EEGBrainFeatureExtractor
from qa_harmonicity_v2 import compute_hi_1_0, compute_hi_2_0, qa_tuple


DEFAULT_DATA_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
DEFAULT_OUTPUT = Path("results/eeg_hi2_0_chbmit_scale_results.json")
DEFAULT_STABILITY_AUDIT = Path("results/eeg_hi2_0_override_gate_patient_stability_audit.json")
WINDOW_SEC = 4.0
OVERLAP_SEC = 2.0
MIN_SEGMENTS_PER_CLASS = 8
MAX_SEGMENTS_PER_CLASS = 48
DEFAULT_MEMORY_LIMIT_MB = 3072
TEST_SIZE = 0.25
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 8
HI2_CONFIG = {"w_ang": 0.5, "w_rad": 0.5, "w_fam": 0.0}
OVERRIDE_FAMILIES = {
    "full": list(range(10)),
    "coords_only": [8, 9],
    "no_geometry": [0, 1, 2, 3, 8, 9],
}
OVERRIDE_MARGIN = 0.08
FULL_WEAK_THRESHOLD = 0.55
DEFAULT_REGIME_POLICY = "stability_threshold"
DEFAULT_MIN_TRIGGER_RATE = 0.5


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def patient_seed(patient_id: str) -> int:
    return zlib.crc32(patient_id.encode("utf-8")) & 0xFFFFFFFF


def parse_summary(summary_path: Path) -> dict[str, list[tuple[int, int]]]:
    """
    Parse CHB-MIT summary annotations.
    Returns: {filename.edf: [(start_s, end_s), ...]}
    """
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
    """
    Minimal EDF reader already proven elsewhere in the repo.
    Returns (signals, sample_rate, channel_labels) with signals shaped
    (n_channels, n_samples).
    """
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


def count_segments(n_samples: int, sample_rate: int) -> int:
    window_samples = int(WINDOW_SEC * sample_rate)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * sample_rate)
    if n_samples < window_samples:
        return 0
    return 1 + int((n_samples - window_samples) // step_samples)


def collect_segments_for_indices(
    signals: np.ndarray,
    sample_rate: int,
    segment_indices: list[int],
) -> list[np.ndarray]:
    window_samples = int(WINDOW_SEC * sample_rate)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * sample_rate)
    segments: list[np.ndarray] = []
    for segment_idx in segment_indices:
        start = segment_idx * step_samples
        end = start + window_samples
        segments.append(signals[:, start:end])
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
            print(
                f"    [{label}] extracting 7D features segment {idx}/{total}",
                flush=True,
            )
        channels_data = {
            channel_name: segment[idx, :]
            for idx, channel_name in enumerate(channel_names)
        }
        features.append(extractor.extract_network_features(channels_data))
    return np.array(features, dtype=np.float64)


def configure_memory_limit(memory_limit_mb: int | None) -> None:
    if memory_limit_mb is None or memory_limit_mb <= 0 or resource is None:
        return
    limit_bytes = int(memory_limit_mb) * 1024 * 1024
    for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
        if not hasattr(resource, limit_name):
            continue
        limit_key = getattr(resource, limit_name)
        try:
            soft_limit, hard_limit = resource.getrlimit(limit_key)
            new_hard = hard_limit if hard_limit == resource.RLIM_INFINITY else min(hard_limit, limit_bytes)
            resource.setrlimit(limit_key, (limit_bytes, new_hard))
        except (OSError, ValueError):
            continue


def maybe_collect_garbage() -> None:
    gc.collect()


def extract_hi_features(features_7d: np.ndarray, use_hi2: bool) -> np.ndarray:
    n_samples = len(features_7d)

    if use_hi2:
        hi_features = np.zeros((n_samples, 10), dtype=np.float64)
        for idx in range(n_samples):
            b = int(features_7d[idx, 0] * 23) + 1
            e = int(features_7d[idx, 1] * 23) + 1
            b = max(1, min(24, b))
            e = max(1, min(24, e))
            q = qa_tuple(b, e, modulus=24)
            result = compute_hi_2_0(
                q,
                w_ang=HI2_CONFIG["w_ang"],
                w_rad=HI2_CONFIG["w_rad"],
                w_fam=HI2_CONFIG["w_fam"],
                modulus=24,
            )
            c_val, f_val, g_val = result["pythagorean_triple"]
            hi_features[idx] = [
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
        return hi_features

    hi_features = np.zeros((n_samples, 4), dtype=np.float64)
    for idx in range(n_samples):
        b = int(features_7d[idx, 0] * 23) + 1
        e = int(features_7d[idx, 1] * 23) + 1
        b = max(1, min(24, b))
        e = max(1, min(24, e))
        q = qa_tuple(b, e, modulus=24)
        hi_1 = compute_hi_1_0(q, modulus=24)
        hi_features[idx] = [hi_1, b / 24.0, e / 24.0, np.linalg.norm(features_7d[idx])]
    return hi_features


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, model: RandomForestClassifier) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "feature_importance": model.feature_importances_.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Baseline", "Seizure"],
            output_dict=True,
            zero_division=0,
        ),
    }


def fit_classifier(x_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)
    return clf


def load_stability_lookup(stability_audit_path: Path | None) -> dict[str, dict[str, object]]:
    if stability_audit_path is None or not stability_audit_path.exists():
        return {}
    source = json.loads(stability_audit_path.read_text(encoding="utf-8"))
    return {
        patient["patient_id"]: patient
        for patient in source.get("patients", [])
    }


def select_override_family(
    hi2_features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    patient_id: str,
) -> tuple[str, dict[str, float], dict[str, object]]:
    inner_train_idx, inner_val_idx = train_test_split(
        train_idx,
        test_size=TEST_SIZE,
        random_state=patient_seed(patient_id) ^ 0x5A5A5A5A,
        stratify=labels[train_idx],
    )
    validation_scores: dict[str, float] = {}
    for family, columns in OVERRIDE_FAMILIES.items():
        clf = fit_classifier(hi2_features[inner_train_idx][:, columns], labels[inner_train_idx])
        score = f1_score(
            labels[inner_val_idx],
            clf.predict(hi2_features[inner_val_idx][:, columns]),
            zero_division=0,
        )
        validation_scores[family] = float(score)

    full_score = validation_scores["full"]
    best_override_family = max(
        ("coords_only", "no_geometry"),
        key=lambda family: validation_scores[family],
    )
    best_override_score = validation_scores[best_override_family]
    override_gap = float(best_override_score - full_score)
    override_triggered = bool(
        full_score <= FULL_WEAK_THRESHOLD and override_gap >= OVERRIDE_MARGIN
    )
    selected_family = best_override_family if override_triggered else "full"
    return selected_family, validation_scores, {
        "override_triggered": override_triggered,
        "full_validation_f1": float(full_score),
        "best_override_family": best_override_family,
        "best_override_validation_f1": float(best_override_score),
        "override_gap": float(override_gap),
        "override_margin": float(OVERRIDE_MARGIN),
        "full_weak_threshold": float(FULL_WEAK_THRESHOLD),
    }


def apply_regime_policy(
    anchor_selected_family: str,
    override_decision: dict[str, object],
    stability: dict[str, object] | None,
    regime_policy: str,
    min_trigger_rate: float,
) -> tuple[str, bool, float | None]:
    if regime_policy == "anchor":
        return anchor_selected_family, bool(override_decision["override_triggered"]), None

    trigger_rate = 0.0 if stability is None else float(stability.get("trigger_rate", 0.0))
    effective_min_trigger_rate = 1.0 if regime_policy == "robust_only" else float(min_trigger_rate)
    if bool(override_decision["override_triggered"]) and trigger_rate >= effective_min_trigger_rate:
        return anchor_selected_family, True, effective_min_trigger_rate
    return "full", False, effective_min_trigger_rate


def evaluate_patient(
    patient_dir: Path,
    extractor: EEGBrainFeatureExtractor,
    stability_lookup: dict[str, dict[str, object]],
    regime_policy: str,
    min_trigger_rate: float,
    max_segments_per_class: int,
) -> dict[str, object]:
    patient_id = patient_dir.name
    print(f"[{patient_id}] starting", flush=True)
    summary_path = patient_dir / f"{patient_id}-summary.txt"
    result: dict[str, object] = {
        "patient_id": patient_id,
        "skipped": True,
        "skip_reason": None,
        "n_files": 0,
        "n_segments": 0,
        "n_baseline": 0,
        "n_seizure": 0,
        "files_processed": [],
    }

    if not summary_path.exists():
        result["skip_reason"] = "missing_summary"
        return result

    annotations = parse_summary(summary_path)
    if not annotations:
        result["skip_reason"] = "no_seizure_annotations"
        return result

    candidate_meta: list[dict[str, object]] = []

    try:
        for filename, seizure_times in sorted(annotations.items()):
            edf_path = patient_dir / filename
            if not edf_path.exists():
                result["files_processed"].append({"file": filename, "status": "missing"})
                continue

            try:
                print(f"[{patient_id}] reading {filename}", flush=True)
                signals, sample_rate, channel_names = read_edf_all_channels(edf_path)
                if sample_rate != extractor.sample_rate:
                    extractor = EEGBrainFeatureExtractor(sample_rate=sample_rate)
                n_segments = count_segments(signals.shape[1], sample_rate)
                print(
                    f"[{patient_id}] {filename} -> {n_segments} segments, "
                    f"{len(seizure_times)} seizure interval(s)",
                    flush=True,
                )
                labels = label_segments(n_segments, seizure_times)
            except MemoryError:
                result["files_processed"].append(
                    {"file": filename, "status": "error", "error": "memory_limit_exceeded"}
                )
                print(f"[{patient_id}] memory limit hit on {filename}", flush=True)
                result["skip_reason"] = "memory_limit_exceeded"
                return result
            except Exception as exc:
                result["files_processed"].append(
                    {"file": filename, "status": "error", "error": str(exc)}
                )
                print(f"[{patient_id}] error on {filename}: {exc}", flush=True)
                continue
            finally:
                if "signals" in locals():
                    del signals

            candidate_meta.append(
                {
                    "file": filename,
                    "edf_path": edf_path,
                    "sample_rate": sample_rate,
                    "channel_names": channel_names,
                    "labels": labels,
                    "segment_count": int(n_segments),
                }
            )
            result["files_processed"].append(
                {
                    "file": filename,
                    "status": "ok",
                    "n_segments": int(len(labels)),
                    "n_seizure": int(labels.sum()),
                    "n_baseline": int(len(labels) - labels.sum()),
                }
            )
            maybe_collect_garbage()

        if not candidate_meta:
            result["skip_reason"] = "no_readable_edf_files"
            print(f"[{patient_id}] skipped: no readable EDF files", flush=True)
            return result

        seizure_candidates: list[tuple[str, int]] = []
        baseline_candidates: list[tuple[str, int]] = []
        total_segments = 0
        for meta in candidate_meta:
            labels = meta["labels"]
            total_segments += len(labels)
            for idx, value in enumerate(labels):
                key = (meta["file"], idx)
                if value == 1:
                    seizure_candidates.append(key)
                else:
                    baseline_candidates.append(key)

        result["n_segments"] = int(total_segments)
        result["n_baseline_raw"] = int(len(baseline_candidates))
        result["n_seizure_raw"] = int(len(seizure_candidates))
        result["n_files"] = int(sum(1 for item in result["files_processed"] if item["status"] == "ok"))

        if len(baseline_candidates) < MIN_SEGMENTS_PER_CLASS or len(seizure_candidates) < MIN_SEGMENTS_PER_CLASS:
            result["skip_reason"] = "insufficient_segments_per_class"
            print(
                f"[{patient_id}] skipped: insufficient class counts "
                f"(baseline={len(baseline_candidates)}, seizure={len(seizure_candidates)})",
                flush=True,
            )
            return result

        rng = np.random.default_rng(patient_seed(patient_id))
        target_count = min(
            len(baseline_candidates),
            len(seizure_candidates),
            max_segments_per_class,
        )
        baseline_pick_idx = rng.choice(len(baseline_candidates), target_count, replace=False)
        seizure_pick_idx = rng.choice(len(seizure_candidates), target_count, replace=False)
        selected_keys = {
            baseline_candidates[idx]: 0 for idx in baseline_pick_idx
        }
        for idx in seizure_pick_idx:
            selected_keys[seizure_candidates[idx]] = 1

        selected_by_file: dict[str, list[int]] = {}
        for file_name, segment_idx in selected_keys:
            selected_by_file.setdefault(file_name, []).append(segment_idx)

        extracted_features: list[np.ndarray] = []
        extracted_labels: list[int] = []
        for meta in candidate_meta:
            file_indices = sorted(selected_by_file.get(meta["file"], []))
            if not file_indices:
                continue
            print(
                f"[{patient_id}] extracting selected segments from {meta['file']} "
                f"({len(file_indices)} segments)",
                flush=True,
            )
            signals, sample_rate, channel_names = read_edf_all_channels(meta["edf_path"])
            if sample_rate != extractor.sample_rate:
                extractor = EEGBrainFeatureExtractor(sample_rate=sample_rate)
            selected_segment_arrays = collect_segments_for_indices(
                signals,
                sample_rate,
                file_indices,
            )
            file_features = extract_7d_features(
                selected_segment_arrays,
                channel_names,
                extractor,
                f"{patient_id}:{meta['file']}",
            )
            extracted_features.extend(file_features)
            extracted_labels.extend(selected_keys[(meta["file"], idx)] for idx in file_indices)
            del signals
            del selected_segment_arrays
            del file_features
            maybe_collect_garbage()

        balanced_features = np.array(extracted_features, dtype=np.float64)
        balanced_labels = np.array(extracted_labels, dtype=int)
        del extracted_features
        del extracted_labels
        shuffle_idx = np.arange(len(balanced_labels))
        rng.shuffle(shuffle_idx)
        balanced_features = balanced_features[shuffle_idx]
        balanced_labels = balanced_labels[shuffle_idx]

        hi1 = extract_hi_features(balanced_features, use_hi2=False)
        hi2 = extract_hi_features(balanced_features, use_hi2=True)
        del balanced_features
        maybe_collect_garbage()

        indices = np.arange(len(balanced_labels))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=patient_seed(patient_id) ^ 0xA5A5A5A5,
            stratify=balanced_labels,
        )

        x1_train, x1_test = hi1[train_idx], hi1[test_idx]
        x2_train, x2_test = hi2[train_idx], hi2[test_idx]
        y_train, y_test = balanced_labels[train_idx], balanced_labels[test_idx]

        clf_hi1 = fit_classifier(x1_train, y_train)
        clf_hi2 = fit_classifier(x2_train, y_train)

        y1_pred = clf_hi1.predict(x1_test)
        y2_pred = clf_hi2.predict(x2_test)

        hi1_metrics = metric_bundle(y_test, y1_pred, clf_hi1)
        hi2_metrics = metric_bundle(y_test, y2_pred, clf_hi2)
        anchor_selected_family, validation_scores, override_decision = select_override_family(
            hi2,
            balanced_labels,
            train_idx,
            patient_id,
        )
        stability = stability_lookup.get(patient_id)
        selected_family, policy_override_triggered, effective_min_trigger_rate = apply_regime_policy(
            anchor_selected_family,
            override_decision,
            stability=stability,
            regime_policy=regime_policy,
            min_trigger_rate=min_trigger_rate,
        )
        override_columns = OVERRIDE_FAMILIES[selected_family]
        clf_override = fit_classifier(hi2[train_idx][:, override_columns], y_train)
        y_override_pred = clf_override.predict(hi2[test_idx][:, override_columns])
        hi2_override_metrics = metric_bundle(y_test, y_override_pred, clf_override)
        deltas = {
            "accuracy_delta": float(hi2_metrics["accuracy"] - hi1_metrics["accuracy"]),
            "precision_delta": float(hi2_metrics["precision"] - hi1_metrics["precision"]),
            "recall_delta": float(hi2_metrics["recall"] - hi1_metrics["recall"]),
            "f1_delta": float(hi2_metrics["f1"] - hi1_metrics["f1"]),
        }
        override_deltas = {
            "accuracy_delta_vs_full": float(hi2_override_metrics["accuracy"] - hi2_metrics["accuracy"]),
            "precision_delta_vs_full": float(hi2_override_metrics["precision"] - hi2_metrics["precision"]),
            "recall_delta_vs_full": float(hi2_override_metrics["recall"] - hi2_metrics["recall"]),
            "f1_delta_vs_full": float(hi2_override_metrics["f1"] - hi2_metrics["f1"]),
            "accuracy_delta_vs_hi1": float(hi2_override_metrics["accuracy"] - hi1_metrics["accuracy"]),
            "precision_delta_vs_hi1": float(hi2_override_metrics["precision"] - hi1_metrics["precision"]),
            "recall_delta_vs_hi1": float(hi2_override_metrics["recall"] - hi1_metrics["recall"]),
            "f1_delta_vs_hi1": float(hi2_override_metrics["f1"] - hi1_metrics["f1"]),
        }
    except MemoryError:
        result["skip_reason"] = "memory_limit_exceeded"
        print(f"[{patient_id}] skipped: memory limit exceeded", flush=True)
        return result
    finally:
        maybe_collect_garbage()

    result.update(
        {
            "skipped": False,
            "skip_reason": None,
            "n_segments_balanced": int(len(balanced_labels)),
            "n_baseline": int((balanced_labels == 0).sum()),
            "n_seizure": int((balanced_labels == 1).sum()),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "HI_1.0": hi1_metrics,
            "HI_2.0": hi2_metrics,
            "HI_2.0_override_gated": hi2_override_metrics,
            "deltas": deltas,
            "override_deltas": override_deltas,
            "regime_detector": {
                "policy_mode": regime_policy,
                "policy_override_triggered": bool(policy_override_triggered),
                "min_trigger_rate": effective_min_trigger_rate,
                "selected_family": selected_family,
                "anchor_selected_family": anchor_selected_family,
                "stability": stability,
                "validation_scores": validation_scores,
                "decision": override_decision,
            },
        }
    )
    print(
        f"[{patient_id}] done: balanced={len(balanced_labels)} "
        f"F1Δ={deltas['f1_delta']:+.3f} "
        f"OverrideΔ={override_deltas['f1_delta_vs_full']:+.3f} "
        f"family={selected_family}",
        flush=True,
    )
    return result


def aggregate_results(patient_results: list[dict[str, object]]) -> dict[str, object]:
    evaluable = [result for result in patient_results if not result["skipped"]]
    skipped = [result for result in patient_results if result["skipped"]]

    total_segments = int(sum(result["n_segments_balanced"] for result in evaluable))
    total_segments_raw = int(sum(result["n_segments"] for result in evaluable))
    total_baseline = int(sum(result["n_baseline"] for result in evaluable))
    total_seizure = int(sum(result["n_seizure"] for result in evaluable))

    if evaluable:
        accuracy_deltas = [result["deltas"]["accuracy_delta"] for result in evaluable]
        f1_deltas = [result["deltas"]["f1_delta"] for result in evaluable]
        override_accuracy_deltas = [
            result["override_deltas"]["accuracy_delta_vs_full"] for result in evaluable
        ]
        override_f1_deltas = [
            result["override_deltas"]["f1_delta_vs_full"] for result in evaluable
        ]
        count_nonnegative_f1 = sum(1 for value in f1_deltas if value >= 0.0)
        count_positive_f1 = sum(1 for value in f1_deltas if value > 0.0)
        share_nonnegative = count_nonnegative_f1 / len(evaluable)
        mean_f1_delta = float(np.mean(f1_deltas))
        count_nonnegative_override_f1 = sum(1 for value in override_f1_deltas if value >= 0.0)
        share_nonnegative_override = count_nonnegative_override_f1 / len(evaluable)
        override_triggered_patients = [
            result["patient_id"]
            for result in evaluable
            if result["regime_detector"]["policy_override_triggered"]
        ]
    else:
        accuracy_deltas = []
        f1_deltas = []
        override_accuracy_deltas = []
        override_f1_deltas = []
        count_nonnegative_f1 = 0
        count_positive_f1 = 0
        count_nonnegative_override_f1 = 0
        share_nonnegative = 0.0
        share_nonnegative_override = 0.0
        mean_f1_delta = 0.0
        override_triggered_patients = []

    if len(evaluable) >= 5 and mean_f1_delta > 0.0 and share_nonnegative >= 0.6:
        verdict = "PASS"
        verdict_reason = (
            "At least five patients were evaluable, mean F1 delta was positive, "
            "and at least 60% of patients had non-negative F1 delta."
        )
    elif evaluable and (len(evaluable) < 5 or mean_f1_delta > 0.0 or count_positive_f1 > 0):
        verdict = "PARTIAL"
        verdict_reason = (
            "Evidence was mixed or underpowered: either fewer than five patients "
            "were evaluable or HI 2.0 gains were not consistent enough by patient."
        )
    else:
        verdict = "FAIL"
        verdict_reason = (
            "HI 2.0 did not improve mean F1 over HI 1.0 across evaluable patients."
        )

    return {
        "n_patients_evaluable": int(len(evaluable)),
        "n_patients_skipped": int(len(skipped)),
        "total_segments": total_segments,
        "total_segments_raw": total_segments_raw,
        "total_baseline": total_baseline,
        "total_seizure": total_seizure,
        "mean_accuracy_delta": float(np.mean(accuracy_deltas)) if accuracy_deltas else 0.0,
        "median_accuracy_delta": float(np.median(accuracy_deltas)) if accuracy_deltas else 0.0,
        "mean_f1_delta": float(np.mean(f1_deltas)) if f1_deltas else 0.0,
        "median_f1_delta": float(np.median(f1_deltas)) if f1_deltas else 0.0,
        "count_positive_f1_delta": int(count_positive_f1),
        "count_nonnegative_f1_delta": int(count_nonnegative_f1),
        "share_nonnegative_f1_delta": float(share_nonnegative),
        "override_gate": {
            "mean_accuracy_delta_vs_full": float(np.mean(override_accuracy_deltas)) if override_accuracy_deltas else 0.0,
            "median_accuracy_delta_vs_full": float(np.median(override_accuracy_deltas)) if override_accuracy_deltas else 0.0,
            "mean_f1_delta_vs_full": float(np.mean(override_f1_deltas)) if override_f1_deltas else 0.0,
            "median_f1_delta_vs_full": float(np.median(override_f1_deltas)) if override_f1_deltas else 0.0,
            "count_nonnegative_f1_delta_vs_full": int(count_nonnegative_override_f1),
            "share_nonnegative_f1_delta_vs_full": float(share_nonnegative_override),
            "override_triggered_patients": override_triggered_patients,
            "selected_family_counts": {
                "full": int(sum(1 for result in evaluable if result["regime_detector"]["selected_family"] == "full")),
                "coords_only": int(sum(1 for result in evaluable if result["regime_detector"]["selected_family"] == "coords_only")),
                "no_geometry": int(sum(1 for result in evaluable if result["regime_detector"]["selected_family"] == "no_geometry")),
            },
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def build_result_document(
    data_root: Path,
    patient_results: list[dict[str, object]],
    stability_audit: Path | None,
    regime_policy: str,
    min_trigger_rate: float,
    max_segments_per_class: int,
    memory_limit_mb: int | None,
) -> dict[str, object]:
    return {
        "experiment": {
            "id": "eeg_hi2_0_chbmit_scale_2026-03-29",
            "domain": "eeg",
            "script": "eeg_hi2_0_chbmit_scale.py",
            "hypothesis": (
                "HI 2.0 improves seizure-vs-baseline discrimination over HI 1.0 "
                "across the locally available CHB-MIT patient set, not only on balanced chb01."
            ),
            "success_criteria": (
                "PASS if at least 5 patients are evaluable locally, mean F1 delta is positive, "
                "and at least 60% of evaluable patients show non-negative F1 delta; "
                "PARTIAL if evidence is mixed or underpowered; FAIL otherwise."
            ),
            "data_root": str(data_root),
            "window_seconds": WINDOW_SEC,
            "overlap_seconds": OVERLAP_SEC,
            "min_segments_per_class": MIN_SEGMENTS_PER_CLASS,
            "max_segments_per_class": int(max_segments_per_class),
            "balance_method": "undersample",
            "test_size": TEST_SIZE,
            "hi2_config": HI2_CONFIG,
            "resource_profile": "local_resource_profile.json",
            "memory_limit_mb": memory_limit_mb,
            "regime_detector": {
                "override_families": ["coords_only", "no_geometry"],
                "override_margin": OVERRIDE_MARGIN,
                "full_weak_threshold": FULL_WEAK_THRESHOLD,
                "policy_mode": regime_policy,
                "min_trigger_rate": None if regime_policy == "anchor" else (1.0 if regime_policy == "robust_only" else float(min_trigger_rate)),
                "stability_audit_source": (
                    str(stability_audit)
                    if stability_audit is not None and stability_audit.exists()
                    else None
                ),
            },
            "open_brain_context": [
                "2026-03-28T10:41:47.905Z EEG nested model verdict: QA was predictive on chb01 but added nothing beyond delta in the threshold-fallback observer.",
                "2026-03-28T10:46:58.124Z EEG three-observer comparison: topographic k-means was the promising path, trending at ΔR²=+0.0447, p=0.087 on chb01.",
            ],
        },
        "patients": patient_results,
        "aggregate": aggregate_results(patient_results),
    }


def print_summary(doc: dict[str, object]) -> None:
    print("=" * 80)
    print("EEG HI 2.0 CHB-MIT SCALE RUN")
    print("=" * 80)
    print(f"Data root: {doc['experiment']['data_root']}")
    print()
    print("Patient-level results:")
    print(
        f"  {'Patient':<8} {'Status':<9} {'Files':>5} {'Segs':>6} "
        f"{'Base':>6} {'Seiz':>6} {'F1Δ':>8} {'GateΔ':>8} {'Family':>12}"
    )
    print(
        f"  {'-'*8} {'-'*9} {'-'*5} {'-'*6} "
        f"{'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*12}"
    )
    for patient in doc["patients"]:
        if patient["skipped"]:
            print(
                f"  {patient['patient_id']:<8} {'SKIPPED':<9} {patient['n_files']:>5} "
                f"{patient['n_segments']:>6} {patient['n_baseline']:>6} {patient['n_seizure']:>6} "
                f"{'--':>8} {'--':>8} {'--':>12}  {patient['skip_reason']}"
            )
            continue
        print(
            f"  {patient['patient_id']:<8} {'OK':<9} {patient['n_files']:>5} "
            f"{patient['n_segments_balanced']:>6} {patient['n_baseline']:>6} {patient['n_seizure']:>6} "
            f"{patient['deltas']['f1_delta']:>+8.3f} "
            f"{patient['override_deltas']['f1_delta_vs_full']:>+8.3f} "
            f"{patient['regime_detector']['selected_family']:>12}"
        )
    print()
    aggregate = doc["aggregate"]
    print("Aggregate:")
    print(f"  Evaluable patients: {aggregate['n_patients_evaluable']}")
    print(f"  Skipped patients:   {aggregate['n_patients_skipped']}")
    print(
        f"  Total balanced segments: {aggregate['total_segments']} "
        f"({aggregate['total_baseline']} baseline, {aggregate['total_seizure']} seizure)"
    )
    print(f"  Total raw candidate segments: {aggregate['total_segments_raw']}")
    print(f"  Mean F1 delta:      {aggregate['mean_f1_delta']:+.4f}")
    print(f"  Median F1 delta:    {aggregate['median_f1_delta']:+.4f}")
    print(
        f"  Non-negative F1:    {aggregate['count_nonnegative_f1_delta']}/"
        f"{aggregate['n_patients_evaluable']}"
    )
    print("Override gate:")
    print(f"  Mean F1 vs full:    {aggregate['override_gate']['mean_f1_delta_vs_full']:+.4f}")
    print(
        f"  Non-negative vs full: {aggregate['override_gate']['count_nonnegative_f1_delta_vs_full']}/"
        f"{aggregate['n_patients_evaluable']}"
    )
    print(f"  Triggered patients: {aggregate['override_gate']['override_triggered_patients']}")
    print(f"  Family counts:      {aggregate['override_gate']['selected_family_counts']}")
    print(f"  Verdict:            {aggregate['verdict']}")
    print(f"  Reason:             {aggregate['verdict_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale HI 1.0 vs HI 2.0 seizure detection across local CHB-MIT patients."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Local CHB-MIT root directory",
    )
    parser.add_argument(
        "--patient",
        nargs="*",
        help="Optional patient ids to restrict the run, e.g. chb01 chb02",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the result JSON",
    )
    parser.add_argument(
        "--stability-audit",
        type=Path,
        default=DEFAULT_STABILITY_AUDIT,
        help="Optional patient stability audit artifact",
    )
    parser.add_argument(
        "--regime-policy",
        choices=["anchor", "robust_only", "stability_threshold"],
        default=DEFAULT_REGIME_POLICY,
        help="Override policy applied after the raw anchor detector",
    )
    parser.add_argument(
        "--min-trigger-rate",
        type=float,
        default=DEFAULT_MIN_TRIGGER_RATE,
        help="Minimum stability trigger rate required when regime-policy=stability_threshold",
    )
    parser.add_argument(
        "--max-segments-per-class",
        type=int,
        default=MAX_SEGMENTS_PER_CLASS,
        help="Upper cap on balanced seizure and baseline windows per patient",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=DEFAULT_MEMORY_LIMIT_MB,
        help="Best-effort per-process memory ceiling; 0 disables the limit",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"Data root not found: {args.data_root}")

    configure_memory_limit(args.memory_limit_mb)
    extractor = EEGBrainFeatureExtractor(sample_rate=256)
    patient_dirs = sorted(path for path in args.data_root.glob("chb*/") if path.is_dir())
    if args.patient:
        wanted = set(args.patient)
        patient_dirs = [path for path in patient_dirs if path.name in wanted]

    stability_lookup = load_stability_lookup(args.stability_audit)
    patient_results = []
    for patient_dir in patient_dirs:
        patient_results.append(
            evaluate_patient(
                patient_dir,
                extractor,
                stability_lookup=stability_lookup,
                regime_policy=args.regime_policy,
                min_trigger_rate=args.min_trigger_rate,
                max_segments_per_class=args.max_segments_per_class,
            )
        )
        checkpoint_doc = build_result_document(
            args.data_root,
            patient_results,
            stability_audit=args.stability_audit,
            regime_policy=args.regime_policy,
            min_trigger_rate=args.min_trigger_rate,
            max_segments_per_class=args.max_segments_per_class,
            memory_limit_mb=(None if args.memory_limit_mb <= 0 else int(args.memory_limit_mb)),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty_json(checkpoint_doc), encoding="utf-8")
        maybe_collect_garbage()
    doc = build_result_document(
        args.data_root,
        patient_results,
        stability_audit=args.stability_audit,
        regime_policy=args.regime_policy,
        min_trigger_rate=args.min_trigger_rate,
        max_segments_per_class=args.max_segments_per_class,
        memory_limit_mb=(None if args.memory_limit_mb <= 0 else int(args.memory_limit_mb)),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pretty_json(doc), encoding="utf-8")
    print_summary(doc)
    print()
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
