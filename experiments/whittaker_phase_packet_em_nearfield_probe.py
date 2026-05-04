#!/usr/bin/env python3
"""Probe exact QA/Whittaker phase-packet residues on measured microwave EM data.

Dataset: University of Oviedo RUO 10.17811/ruo_datasets.75973, Escatt_meas.zip.
The files contain measured scattered S21 over a planar microwave imaging domain:
rows are x, y, real(S21), imag(S21), one text file per frequency.

This is an experiment, not a cert. The QA side computes exact rational phase
arguments and residue bins only; no trigonometric function is evaluated.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable


DATA_URL = (
    "https://digibuo.uniovi.es/dspace/bitstream/handle/10651/75973/"
    "Escatt_meas.zip?sequence=2&isAllowed=y"
)
DATA_PATH = Path("/tmp/Escatt_meas.zip")
RESULT_PATH = Path("results/whittaker_phase_packet_em_nearfield_probe.json")

# Exact S2 packets used in prior [273]/[275] work, represented as
# (x_num, y_num, z_num, den). The measured dataset is a planar x/y field slice,
# so z is set to 0 in the phase argument.
OMEGA_PACKETS = (
    (7, 24, 0, 25),
    (24, 7, 0, 25),
    (1200, 1200, -527, 1777),
    (3, 4, 0, 5),
    (4, 3, 0, 5),
)
MODULI = (9, 24, 72)


@dataclass(frozen=True)
class Sample:
    frequency_mhz: int
    x_mm: int
    y_mm: int
    real: float
    imag: float
    phase_quadrant: int
    amplitude: float


def ensure_data() -> None:
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 1_000_000:
        return
    print(f"downloading {DATA_URL}", file=sys.stderr)
    with urllib.request.urlopen(DATA_URL, timeout=180) as resp:
        DATA_PATH.write_bytes(resp.read())


def parse_frequency(name: str) -> int:
    stem = Path(name).stem
    return int(stem.split("_f_")[1].replace("MHz", ""))


def parse_decimal_mm(value: str) -> int:
    # Coordinates in the dataset are decimal meters on a 10 mm grid.
    return int(Fraction(value) * 1000)


def quadrant(real: float, imag: float) -> int:
    if real >= 0.0 and imag >= 0.0:
        return 0
    if real < 0.0 and imag >= 0.0:
        return 1
    if real < 0.0 and imag < 0.0:
        return 2
    return 3


def load_samples(max_frequencies: int = 67, spatial_stride: int = 3) -> list[Sample]:
    ensure_data()
    samples: list[Sample] = []
    with zipfile.ZipFile(DATA_PATH) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".txt"))
        # Use a deterministic spread across the full 12-18 GHz band.
        selected = names[:: max(1, len(names) // max_frequencies)]
        for name in selected[:max_frequencies]:
            freq_mhz = parse_frequency(name)
            with zf.open(name) as fh:
                for row_idx, raw in enumerate(fh):
                    if row_idx % spatial_stride:
                        continue
                    parts = raw.decode("ascii").split()
                    if len(parts) != 4:
                        continue
                    x_mm = parse_decimal_mm(parts[0])
                    y_mm = parse_decimal_mm(parts[1])
                    real = float(parts[2])
                    imag = float(parts[3])
                    samples.append(
                        Sample(
                            frequency_mhz=freq_mhz,
                            x_mm=x_mm,
                            y_mm=y_mm,
                            real=real,
                            imag=imag,
                            phase_quadrant=quadrant(real, imag),
                            amplitude=math.hypot(real, imag),
                        )
                    )
    return samples


def frac_bin(value: Fraction, modulus: int) -> int:
    floor = value.numerator // value.denominator
    frac = value - floor
    return int((frac.numerator * modulus) // frac.denominator)


def phase_features(sample: Sample, z_mm: int = 0, physical_cycles: bool = True) -> tuple[int, ...]:
    features: list[int] = []
    for ox, oy, oz, den in OMEGA_PACKETS:
        dot_mm = Fraction(ox * sample.x_mm + oy * sample.y_mm + oz * z_mm, den)
        if physical_cycles:
            # Cycles = f * distance / c. This avoids 2*pi and keeps the phase
            # coordinate exact over rationals while using a real wave convention.
            phase_arg = Fraction(sample.frequency_mhz * 1000, 299_792_458) * dot_mm
        else:
            phase_arg = Fraction(sample.frequency_mhz) * dot_mm
        for modulus in MODULI:
            features.append(frac_bin(phase_arg, modulus))
    return tuple(features)


def raw_xyf_features(sample: Sample) -> tuple[int, ...]:
    return (
        sample.x_mm,
        sample.y_mm,
        sample.frequency_mhz % 9,
        sample.frequency_mhz % 24,
        sample.frequency_mhz % 72,
    )


def raw_field_features(sample: Sample) -> tuple[int, ...]:
    return (
        sample.x_mm,
        sample.y_mm,
        sample.frequency_mhz % 24,
        int(sample.amplitude * 10_000),
    )


class CategoricalNB:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_counts: Counter[int] = Counter()
        self.feature_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.feature_values: dict[int, set[int]] = defaultdict(set)
        self.n_features = 0

    def fit(self, rows: Iterable[tuple[tuple[int, ...], int]]) -> None:
        for features, label in rows:
            self.class_counts[label] += 1
            self.n_features = len(features)
            for idx, value in enumerate(features):
                self.feature_counts[(label, idx)][value] += 1
                self.feature_values[idx].add(value)

    def predict_one(self, features: tuple[int, ...]) -> int:
        total = sum(self.class_counts.values())
        best_label = None
        best_score = -float("inf")
        labels = sorted(self.class_counts)
        for label in labels:
            class_count = self.class_counts[label]
            score = math.log((class_count + self.alpha) / (total + self.alpha * len(labels)))
            for idx, value in enumerate(features):
                value_count = self.feature_counts[(label, idx)][value]
                width = max(1, len(self.feature_values[idx]))
                score += math.log((value_count + self.alpha) / (class_count + self.alpha * width))
            if score > best_score:
                best_score = score
                best_label = label
        assert best_label is not None
        return best_label

    def accuracy(self, rows: Iterable[tuple[tuple[int, ...], int]]) -> float:
        total = 0
        correct = 0
        for features, label in rows:
            total += 1
            correct += int(self.predict_one(features) == label)
        return correct / total if total else 0.0


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    train: list[Sample] = []
    test: list[Sample] = []
    for sample in samples:
        if (sample.frequency_mhz // 30) % 5 == 0:
            test.append(sample)
        else:
            train.append(sample)
    return train, test


def phase_label(sample: Sample) -> int:
    return sample.phase_quadrant


def amplitude_threshold(samples: list[Sample]) -> float:
    values = sorted(sample.amplitude for sample in samples)
    return values[len(values) // 2]


def make_amplitude_label(threshold: float):
    return lambda sample: int(sample.amplitude >= threshold)


def evaluate(samples: list[Sample], feature_fn, label_fn, shuffle_seed: int | None = None) -> float:
    train, test = split_samples(samples)
    train_labels = [label_fn(s) for s in train]
    test_labels = [label_fn(s) for s in test]
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        train_labels = train_labels[:]
        test_labels = test_labels[:]
        rng.shuffle(train_labels)
        rng.shuffle(test_labels)
    model = CategoricalNB()
    model.fit((feature_fn(sample), label) for sample, label in zip(train, train_labels))
    return model.accuracy((feature_fn(sample), label) for sample, label in zip(test, test_labels))


def majority_accuracy(samples: list[Sample]) -> float:
    train, test = split_samples(samples)
    label = Counter(s.phase_quadrant for s in train).most_common(1)[0][0]
    return sum(1 for s in test if s.phase_quadrant == label) / len(test)


def majority_accuracy_for(samples: list[Sample], label_fn) -> float:
    train, test = split_samples(samples)
    label = Counter(label_fn(s) for s in train).most_common(1)[0][0]
    return sum(1 for s in test if label_fn(s) == label) / len(test)


def null_summary(samples: list[Sample], feature_fn, label_fn, observed: float) -> dict[str, float | int]:
    null = [evaluate(samples, feature_fn, label_fn, shuffle_seed=10_000 + i) for i in range(50)]
    return {
        "trials": len(null),
        "mean": sum(null) / len(null),
        "max": max(null),
        "min": min(null),
        "p_value_ge_observed": (1 + sum(1 for x in null if x >= observed)) / (1 + len(null)),
    }


def main() -> int:
    samples = load_samples()
    train, test = split_samples(samples)
    amp_threshold = amplitude_threshold(train)
    amp_label = make_amplitude_label(amp_threshold)
    target_counts = Counter(s.phase_quadrant for s in samples)

    feature_sets = {
        "qa_phase_cycles_z0": lambda s: phase_features(s, z_mm=0, physical_cycles=True),
        "qa_phase_cycles_z1000": lambda s: phase_features(s, z_mm=1000, physical_cycles=True),
        "qa_phase_unscaled_z0": lambda s: phase_features(s, z_mm=0, physical_cycles=False),
        "raw_xyf_modulus": raw_xyf_features,
        "raw_xyf_plus_amplitude": raw_field_features,
    }

    phase_results = {name: evaluate(samples, fn, phase_label) for name, fn in feature_sets.items()}
    amplitude_results = {name: evaluate(samples, fn, amp_label) for name, fn in feature_sets.items()}
    phase_majority = majority_accuracy_for(samples, phase_label)
    amplitude_majority = majority_accuracy_for(samples, amp_label)
    best_phase_name = max(
        (name for name in phase_results if name.startswith("qa_")),
        key=lambda name: phase_results[name],
    )
    best_amp_name = max(
        (name for name in amplitude_results if name.startswith("qa_")),
        key=lambda name: amplitude_results[name],
    )

    result = {
        "ok": True,
        "dataset": {
            "name": "RUO microwave imaging measurements Escatt_meas",
            "doi": "10.17811/ruo_datasets.75973",
            "source_url": DATA_URL,
            "measurement": "scattered microwave S21, proportional to measured electric field",
            "frequency_band_mhz": [12000, 18000],
            "sample_count": len(samples),
            "train_count": len(train),
            "test_count": len(test),
            "targets": [
                "measured complex S21 phase quadrant from signs of real/imag",
                "measured S21 amplitude above/below training median",
            ],
            "target_counts": dict(sorted(target_counts.items())),
        },
        "qa_phase_packet_boundary": {
            "omega_packets": [list(p) for p in OMEGA_PACKETS],
            "phase_arg": "cycles = frequency_hz * ((omega dot position_m) / c)",
            "features": "exact Fraction fractional residue bins",
            "moduli": list(MODULI),
            "no_trig_evaluated": True,
        },
        "phase_quadrant_accuracy": {
            **phase_results,
            "majority_class": phase_majority,
        },
        "amplitude_binary_accuracy": {
            **amplitude_results,
            "majority_class": amplitude_majority,
            "training_median_threshold": amp_threshold,
        },
        "label_shuffle_null": {
            "phase_quadrant": null_summary(
                samples, feature_sets[best_phase_name], phase_label, phase_results[best_phase_name]
            ),
            "amplitude_binary": null_summary(
                samples, feature_sets[best_amp_name], amp_label, amplitude_results[best_amp_name]
            ),
        },
        "interpretation": (
            "This is real measured EM S21 data. The QA phase-packet side computes exact "
            "rational wave-cycle residues from certified S2 directions and does not evaluate "
            "trig. Results should be judged against majority, raw-feature baselines, and "
            "label-shuffle nulls; weak or null performance is an empirical failure of this "
            "phase convention, not a reason to avoid the EM test."
        ),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
