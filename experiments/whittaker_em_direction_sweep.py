#!/usr/bin/env python3
"""Direction-sweep control for measured EM Whittaker/QA phase ordering.

The previous observer/null test showed QA recurrence structure in measured EM
phase, but raster order also carried it. This script asks the missing question:

Do the hand-selected Whittaker/[273] phase directions rank unusually against a
finite exact rational S2 direction population, or are they ordinary directions?

No trig is evaluated. Direction keys are exact rational wave-cycle residues.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qa_orbit_rules import norm_f, orbit_family  # noqa: E402


DATA_URL = (
    "https://digibuo.uniovi.es/dspace/bitstream/handle/10651/75973/"
    "Escatt_meas.zip?sequence=2&isAllowed=y"
)
DATA_PATH = Path("/tmp/Escatt_meas.zip")
OUT_PATH = Path("results/whittaker_em_direction_sweep.json")

M_QA = 24
S2_M = 3
CLASS_ORDER = ("cosmos", "satellite", "singularity")
OMEGA_PACKETS = (
    (7, 24, 0, 25, "omega0_7_24_0_25"),
    (24, 7, 0, 25, "omega1_24_7_0_25"),
    (1200, 1200, -527, 1777, "omega2_1200_1200_neg527_1777"),
    (3, 4, 0, 5, "omega3_3_4_0_5"),
    (4, 3, 0, 5, "omega4_4_3_0_5"),
)


@dataclass(frozen=True)
class FieldSample:
    frequency_mhz: int
    x_mm: int
    y_mm: int
    real: float
    imag: float

    @property
    def wrapped_phase(self) -> float:
        return math.atan2(self.imag, self.real)


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


def wrap_pi(value: float) -> float:
    while value <= -math.pi:
        value += 2.0 * math.pi
    while value > math.pi:
        value -= 2.0 * math.pi
    return value


def unwrap_ordered_phase(samples: list[FieldSample]) -> list[float]:
    if not samples:
        return []
    out = [samples[0].wrapped_phase]
    for sample in samples[1:]:
        out.append(out[-1] + wrap_pi(sample.wrapped_phase - out[-1]))
    return out


def q_fixed(values: list[float], m: int = M_QA) -> list[int]:
    ordered = sorted(values)
    edges = []
    for k in range(1, m):
        idx = (k / m) * (len(ordered) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        edges.append(ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo]))
    out = []
    for value in values:
        b = 1
        for edge in edges:
            if value > edge:
                b += 1
        out.append(b)
    return out


def infer_e(b_seq: list[int], m: int = M_QA) -> list[int]:
    return [((b_seq[t + 1] - b_seq[t] - 1) % m) + 1 for t in range(len(b_seq) - 1)]


def mean_return_time(b_seq: list[int], e_seq: list[int]) -> float:
    occur: dict[tuple[int, int], list[int]] = {}
    for idx, pair in enumerate(zip(b_seq, e_seq)):
        occur.setdefault(pair, []).append(idx)
    gaps = []
    for indices in occur.values():
        for i in range(len(indices) - 1):
            gaps.append(indices[i + 1] - indices[i])
    return sum(gaps) / len(gaps) if gaps else 0.0


def entropy(values: list[int]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def observables_for_ordered_samples(samples: list[FieldSample]) -> dict:
    values = unwrap_ordered_phase(samples)
    b_seq = q_fixed(values)
    e_seq = infer_e(b_seq)
    b_aligned = b_seq[:-1]
    class_counts = Counter(orbit_family(b, e, M_QA) for b, e in zip(b_aligned, e_seq))
    total = max(1, sum(class_counts.values()))
    return {
        "mean_return_time": mean_return_time(b_aligned, e_seq),
        "norm_f_entropy": entropy([norm_f(b, e) for b, e in zip(b_aligned, e_seq)]),
        "singularity_frac": class_counts["singularity"] / total,
        "satellite_frac": class_counts["satellite"] / total,
        "cosmos_frac": class_counts["cosmos"] / total,
    }


def qa_ratios(m: int) -> set[Fraction]:
    ratios: set[Fraction] = set()
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = b + 2 * e
            c = 2 * d * e
            f = a * b
            g = d * d + e * e
            ratios.add(Fraction(c, g))
            ratios.add(Fraction(f, g))
    return ratios


def s2_packet(r: Fraction, s: Fraction) -> tuple[int, int, int, int]:
    den = r.denominator * r.denominator * s.denominator * s.denominator
    rr = r.numerator * r.numerator * s.denominator * s.denominator
    ss = s.numerator * s.numerator * r.denominator * r.denominator
    q = r.denominator * r.denominator * s.denominator * s.denominator
    common = q + rr + ss
    x_num = 2 * r.numerator * r.denominator * s.denominator * s.denominator
    y_num = 2 * s.numerator * s.denominator * r.denominator * r.denominator
    z_num = q - rr - ss
    scale = gcd(gcd(abs(x_num), abs(y_num)), gcd(abs(z_num), abs(common)))
    del den
    return (x_num // scale, y_num // scale, z_num // scale, common // scale)


def direction_population(m: int) -> list[tuple[int, int, int, int, str]]:
    ratios = sorted(qa_ratios(m))
    packets = {}
    for r in ratios:
        for s in ratios:
            x, y, z, den = s2_packet(r, s)
            packets[(Fraction(x, den), Fraction(y, den), Fraction(z, den))] = (x, y, z, den)
    out = []
    for idx, packet in enumerate(sorted(packets.values())):
        out.append((*packet, f"s2_m{m}_{idx}"))
    return out


def direction_key(sample: FieldSample, packet: tuple[int, int, int, int, str]) -> Fraction:
    ox, oy, oz, den, _name = packet
    # The measured field plane has no per-sample z coordinate, so z=0 here.
    dot_mm = Fraction(ox * sample.x_mm + oy * sample.y_mm + oz * 0, den)
    cycles = Fraction(sample.frequency_mhz * 1000, 299_792_458) * dot_mm
    return cycles - (cycles.numerator // cycles.denominator)


def evaluate_direction(samples: list[FieldSample], packet: tuple[int, int, int, int, str]) -> dict:
    ordered = sorted(samples, key=lambda sample: direction_key(sample, packet))
    obs = observables_for_ordered_samples(ordered)
    return {"packet": list(packet[:4]), "name": packet[4], **obs}


def percentile_rank(value: float, population: list[float], lower_is_better: bool) -> float:
    if lower_is_better:
        count = sum(1 for item in population if item >= value)
    else:
        count = sum(1 for item in population if item <= value)
    return count / len(population)


def main() -> int:
    samples = load_samples()
    raster = observables_for_ordered_samples(samples)
    population_packets = direction_population(S2_M)
    population = [evaluate_direction(samples, packet) for packet in population_packets]
    named = [evaluate_direction(samples, packet) for packet in OMEGA_PACKETS]

    for row in named:
        row["rank_vs_s2_m3"] = {
            "mean_return_time_lower": percentile_rank(
                row["mean_return_time"], [p["mean_return_time"] for p in population], True
            ),
            "norm_f_entropy_lower": percentile_rank(
                row["norm_f_entropy"], [p["norm_f_entropy"] for p in population], True
            ),
            "singularity_frac_higher": percentile_rank(
                row["singularity_frac"], [p["singularity_frac"] for p in population], False
            ),
        }
        row["delta_vs_raster"] = {
            "mean_return_time": row["mean_return_time"] - raster["mean_return_time"],
            "norm_f_entropy": row["norm_f_entropy"] - raster["norm_f_entropy"],
            "singularity_frac": row["singularity_frac"] - raster["singularity_frac"],
        }

    top_return = sorted(population, key=lambda row: row["mean_return_time"])[:10]
    top_singularity = sorted(population, key=lambda row: row["singularity_frac"], reverse=True)[:10]

    payload = {
        "ok": True,
        "experiment": "whittaker_em_direction_sweep",
        "dataset": {
            "name": "RUO microwave imaging measurements Escatt_meas",
            "doi": "10.17811/ruo_datasets.75973",
            "source_url": DATA_URL,
            "sample_count": len(samples),
            "channel": "unwrapped measured S21 phase",
        },
        "question": "Do named Whittaker/[273] directions rank unusually against the exact S2 m=3 direction population?",
        "raster_observables": raster,
        "s2_population": {
            "m": S2_M,
            "count": len(population),
            "top_10_lowest_mean_return_time": top_return,
            "top_10_highest_singularity_frac": top_singularity,
        },
        "named_whittaker_directions": named,
        "interpretation": (
            "If named directions have weak percentile ranks or do not beat raster, "
            "the measured EM phase structure is real but not yet specifically tied "
            "to the current Whittaker direction choices."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
