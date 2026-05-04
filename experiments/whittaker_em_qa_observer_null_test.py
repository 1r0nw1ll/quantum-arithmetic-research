#!/usr/bin/env python3
"""Measured EM field test using the existing QA observer/null pattern.

This is the same style as the repo's real-data quantization tests:

1. take a measured scalar stream,
2. quantize to b in {1..m},
3. infer e_t = ((b_{t+1} - b_t - 1) % m) + 1,
4. compute QA orbit observables,
5. compare against permutation and block-bootstrap nulls.

The physical question here is not classifier accuracy. It is whether ordering
measured microwave S21 amplitudes by exact Whittaker/QA phase-packet coordinates
creates QA orbit structure beyond order-destroying nulls.
"""

from __future__ import annotations

import json
import math
import random
import sys
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qa_orbit_rules import norm_f, orbit_family  # noqa: E402


DATA_URL = (
    "https://digibuo.uniovi.es/dspace/bitstream/handle/10651/75973/"
    "Escatt_meas.zip?sequence=2&isAllowed=y"
)
DATA_PATH = Path("/tmp/Escatt_meas.zip")
OUT_PATH = Path("results/whittaker_em_qa_observer_null_test.json")

M = 24
N_PERMUTE = 60
N_BLOCK = 30
BLOCK_SIZE = 250
SEED = 42
CLASS_ORDER = ("cosmos", "satellite", "singularity")
OMEGA_PACKETS = (
    (7, 24, 0, 25),
    (24, 7, 0, 25),
    (1200, 1200, -527, 1777),
    (3, 4, 0, 5),
    (4, 3, 0, 5),
)


@dataclass(frozen=True)
class FieldSample:
    frequency_mhz: int
    x_mm: int
    y_mm: int
    real: float
    imag: float

    @property
    def amplitude(self) -> float:
        return math.hypot(self.real, self.imag)


def ensure_data() -> None:
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 1_000_000:
        return
    with urllib.request.urlopen(DATA_URL, timeout=180) as resp:
        DATA_PATH.write_bytes(resp.read())


def parse_frequency(name: str) -> int:
    return int(Path(name).stem.split("_f_")[1].replace("MHz", ""))


def parse_decimal_mm(value: str) -> int:
    return int(Fraction(value) * 1000)


def load_samples(max_frequencies: int = 67, spatial_stride: int = 3) -> list[FieldSample]:
    ensure_data()
    samples: list[FieldSample] = []
    with zipfile.ZipFile(DATA_PATH) as zf:
        names = sorted(name for name in zf.namelist() if name.endswith(".txt"))
        step = max(1, len(names) // max_frequencies)
        for name in names[::step][:max_frequencies]:
            freq = parse_frequency(name)
            with zf.open(name) as fh:
                for row_idx, raw in enumerate(fh):
                    if row_idx % spatial_stride:
                        continue
                    parts = raw.decode("ascii").split()
                    if len(parts) != 4:
                        continue
                    samples.append(
                        FieldSample(
                            frequency_mhz=freq,
                            x_mm=parse_decimal_mm(parts[0]),
                            y_mm=parse_decimal_mm(parts[1]),
                            real=float(parts[2]),
                            imag=float(parts[3]),
                        )
                    )
    return samples


def quantile_edges(values: list[float], m: int = M) -> list[float]:
    ordered = sorted(values)
    n = len(ordered)
    edges: list[float] = []
    for k in range(1, m):
        idx = (k / m) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        edges.append(ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo]))
    return edges


def q_fixed_edges(values: list[float], edges: list[float]) -> list[int]:
    out: list[int] = []
    for value in values:
        b = 1
        for edge in edges:
            if value > edge:
                b += 1
        out.append(b)
    return out


def infer_generators(b_seq: list[int], m: int = M) -> list[int]:
    return [((b_seq[t + 1] - b_seq[t] - 1) % m) + 1 for t in range(len(b_seq) - 1)]


def class_fracs(b_seq: list[int], e_seq: list[int], m: int = M) -> dict[str, float]:
    counts = {key: 0 for key in CLASS_ORDER}
    for b, e in zip(b_seq, e_seq):
        counts[orbit_family(int(b), int(e), m)] += 1
    total = max(1, sum(counts.values()))
    return {key: counts[key] / total for key in CLASS_ORDER}


def norm_f_entropy(b_seq: list[int], e_seq: list[int]) -> float:
    counts = Counter(norm_f(int(b), int(e)) for b, e in zip(b_seq, e_seq))
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def mean_return_time(b_seq: list[int], e_seq: list[int]) -> float:
    occur: dict[tuple[int, int], list[int]] = {}
    for idx, pair in enumerate(zip(b_seq, e_seq)):
        occur.setdefault(pair, []).append(idx)
    gaps: list[int] = []
    for indices in occur.values():
        for i in range(len(indices) - 1):
            gaps.append(indices[i + 1] - indices[i])
    return sum(gaps) / len(gaps) if gaps else 0.0


def observables(values: list[float], edges: list[float]) -> dict:
    b_seq = q_fixed_edges(values, edges)
    e_seq = infer_generators(b_seq)
    b_aligned = b_seq[:-1]
    return {
        "class_fracs": class_fracs(b_aligned, e_seq),
        "norm_f_entropy": norm_f_entropy(b_aligned, e_seq),
        "mean_return_time": mean_return_time(b_aligned, e_seq),
    }


def whittaker_phase_key(sample: FieldSample, omega_idx: int, z_mm: int = 0) -> Fraction:
    ox, oy, oz, den = OMEGA_PACKETS[omega_idx]
    dot_mm = Fraction(ox * sample.x_mm + oy * sample.y_mm + oz * z_mm, den)
    cycles = Fraction(sample.frequency_mhz * 1000, 299_792_458) * dot_mm
    return cycles - (cycles.numerator // cycles.denominator)


def orderings(samples: list[FieldSample]) -> dict[str, list[float]]:
    out = {
        "raster_frequency_order": [sample.amplitude for sample in samples],
    }
    for idx in range(len(OMEGA_PACKETS)):
        ordered = sorted(samples, key=lambda sample, i=idx: whittaker_phase_key(sample, i))
        out[f"whittaker_phase_order_omega{idx}"] = [sample.amplitude for sample in ordered]
    return out


def shuffle_permute(values: list[float], rng: random.Random) -> list[float]:
    out = values[:]
    rng.shuffle(out)
    return out


def shuffle_block(values: list[float], rng: random.Random, block_size: int = BLOCK_SIZE) -> list[float]:
    n = len(values)
    starts = [rng.randrange(0, max(1, n - block_size + 1)) for _ in range((n + block_size - 1) // block_size)]
    out: list[float] = []
    for start in starts:
        out.extend(values[start : start + block_size])
    return out[:n]


def chi2_vs_mean(fracs: dict[str, float], mean: dict[str, float]) -> float:
    return sum((fracs[key] - mean[key]) * (fracs[key] - mean[key]) / max(mean[key], 1e-12) for key in CLASS_ORDER)


def p_two_tailed(real: float, nulls: list[float]) -> float:
    mean = sum(nulls) / len(nulls)
    real_dev = abs(real - mean)
    return (1 + sum(1 for value in nulls if abs(value - mean) >= real_dev)) / (1 + len(nulls))


def null_test(values: list[float], null_fn: Callable[[list[float], random.Random], list[float]], n: int) -> dict:
    edges = quantile_edges(values)
    real = observables(values, edges)
    rng = random.Random(SEED)
    draws = [observables(null_fn(values, rng), edges) for _ in range(n)]
    mean_fracs = {
        key: sum(draw["class_fracs"][key] for draw in draws) / len(draws) for key in CLASS_ORDER
    }
    real_chi2 = chi2_vs_mean(real["class_fracs"], mean_fracs)
    null_chi2 = [chi2_vs_mean(draw["class_fracs"], mean_fracs) for draw in draws]
    ent_null = [draw["norm_f_entropy"] for draw in draws]
    rt_null = [draw["mean_return_time"] for draw in draws]
    return {
        "real": real,
        "null_mean": {
            "class_fracs": mean_fracs,
            "norm_f_entropy": sum(ent_null) / len(ent_null),
            "mean_return_time": sum(rt_null) / len(rt_null),
        },
        "p_values": {
            "class_chi2": (1 + sum(1 for value in null_chi2 if value >= real_chi2)) / (1 + len(null_chi2)),
            "norm_f_entropy": p_two_tailed(real["norm_f_entropy"], ent_null),
            "mean_return_time": p_two_tailed(real["mean_return_time"], rt_null),
        },
        "stats": {
            "class_chi2_real": real_chi2,
            "class_chi2_null_mean": sum(null_chi2) / len(null_chi2),
        },
    }


def main() -> int:
    samples = load_samples()
    ordered = orderings(samples)
    results = {}
    for name, values in ordered.items():
        results[name] = {
            "permute": null_test(values, shuffle_permute, N_PERMUTE),
            "block_bootstrap": null_test(values, shuffle_block, N_BLOCK),
        }
    payload = {
        "ok": True,
        "experiment": "whittaker_em_qa_observer_null_test",
        "dataset": {
            "name": "RUO microwave imaging measurements Escatt_meas",
            "doi": "10.17811/ruo_datasets.75973",
            "source_url": DATA_URL,
            "measurement": "measured scattered microwave S21 amplitudes",
            "sample_count": len(samples),
            "frequency_band_mhz": [12000, 18000],
        },
        "qa_observer_pattern": {
            "source_pattern": "phase2_5_quantization_compare real-data observer/null test",
            "m": M,
            "b": "fixed quantile bin of measured S21 amplitude, edges calibrated per ordering before null draws",
            "e": "((b_next - b - 1) % m) + 1",
            "observables": ["class_fracs", "norm_f_entropy", "mean_return_time"],
            "nulls": {
                "permute": N_PERMUTE,
                "block_bootstrap": {"trials": N_BLOCK, "block_size": BLOCK_SIZE},
            },
        },
        "tested_orderings": list(ordered.keys()),
        "omega_packets": [list(packet) for packet in OMEGA_PACKETS],
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
