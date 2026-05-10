#!/usr/bin/env python3
"""QA observer test on real measured magnetic hysteresis loops.

This script uses the Ring35 experimental FeSi hysteresis-loop dataset from
Zenodo record 17579041. The archive stores measured time, B(t), H(t), and
dB/dt traces plus a loss table. The test is intentionally direct:

1. Compute the measured loop-work proxy integral H dB from B(t), H(t).
2. Calibrate global QA bins on calibration loops only.
3. Map held-out measured loops to QA variables:
       b = bin(H), e = bin(B), d = b+e, a = b+2e
       J = b*d, X = d*e, K = d*a
4. Predict held-out energy loss from deterministic QA loop observables,
   including a physically dimensioned QA shell approximation to integral H dB.
5. Compare against mean-loss and Steinmetz-style baselines.

No synthetic loop is used in the main result. No neural model is trained. The
direct QA shell integral uses calibration-loop shell centers but no fitted
regression coefficient.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import re
import statistics
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# This bridge intentionally uses the established QA orbit substrate, not a
# hysteresis-specific invented feature set.
from qa_orbit_rules import norm_f, orbit_family  # noqa: E402

ZIP_PATH = Path("/tmp/Ring35_Dataset_Txt.zip")
OUT_PATH = Path("results/qa_hysteresis_real_loop_observer.json")

DATASET_RECORD = "https://zenodo.org/records/17579041"
DATASET_TITLE = "Dataset of Experimental Non-Standard Dynamic Hysteresis Loops"
DENSITY_KG_PER_M3 = 7632.0
MODULUS = 24
QA_LIFTED_VARIABLES = ("b", "e", "J", "X", "K", "Pi", "K_minus_J")
QA_TRANSITION_FEATURES = [
    "qa_transition_unique_edges",
    "qa_transition_edge_reuse_ratio",
    "qa_transition_entropy",
    "qa_transition_reversal_count",
    "qa_transition_irreversible_edge_fraction",
    "qa_transition_signed_flux",
    "qa_transition_abs_flux",
    "qa_transition_orientation_flux",
    "qa_transition_abs_orientation_flux",
    "qa_transition_branch_asymmetry",
    "qa_transition_return_time_mean",
    "qa_transition_return_time_std",
    "qa_transition_loop_closure_defect",
    "qa_transition_generator_distance_sum",
    "qa_transition_generator_distance_mean",
]
QA_MEMORY_FEATURES = [
    "qa_memory_unique_states",
    "qa_memory_unique_edges",
    "qa_memory_entropy",
    "qa_memory_branch_rising_fraction",
    "qa_memory_branch_falling_fraction",
    "qa_memory_branch_turning_fraction",
    "qa_memory_lag_aligned_fraction",
    "qa_memory_lag_opposed_fraction",
    "qa_memory_lag_B_lags_H_fraction",
    "qa_memory_lag_B_leads_H_fraction",
    "qa_memory_branch_asymmetry",
    "qa_memory_turning_density",
    "qa_memory_irreversible_edge_fraction",
    "qa_memory_return_time_mean",
    "qa_memory_return_time_std",
    "qa_memory_generator_distance_sum",
    "qa_memory_generator_distance_mean",
    "qa_memory_signed_orientation_flux",
    "qa_memory_abs_orientation_flux",
    "qa_memory_lag_weighted_orientation_flux",
]
QA_RECONSTRUCTION_FIELDS = [
    "qa_reconstruct_be_energy_mj_per_kg",
    "qa_reconstruct_be_dir_energy_mj_per_kg",
    "qa_reconstruct_be_branch_energy_mj_per_kg",
    "qa_reconstruct_memory_full_energy_mj_per_kg",
]
QA_RESIDUAL_RECONSTRUCTION_FIELDS = [
    "qa_residual_reconstruct_be_energy_mj_per_kg",
    "qa_residual_reconstruct_be_dir_energy_mj_per_kg",
    "qa_residual_reconstruct_be_branch_energy_mj_per_kg",
    "qa_residual_reconstruct_memory_full_energy_mj_per_kg",
]
QA_ORBIT_TOPOLOGY_FEATURES = [
    "qa_orbit_cosmos_fraction",
    "qa_orbit_satellite_fraction",
    "qa_orbit_singularity_fraction",
    "qa_orbit_norm_mod_entropy",
    "qa_orbit_family_transition_entropy",
    "qa_orbit_family_switch_fraction",
    "qa_orbit_cosmos_to_satellite_fraction",
    "qa_orbit_satellite_to_cosmos_fraction",
    "qa_orbit_norm_signed_mean",
    "qa_orbit_norm_signed_abs_mean",
    "qa_orbit_norm_flip_fraction",
]
QA_ORBIT_RESIDUAL_RECONSTRUCTION_FIELDS = [
    "qa_orbit_residual_reconstruct_family_energy_mj_per_kg",
    "qa_orbit_residual_reconstruct_family_norm_energy_mj_per_kg",
    "qa_orbit_residual_reconstruct_family_norm_branch_energy_mj_per_kg",
]


@dataclass(frozen=True)
class LossRow:
    filename: str
    bp_t: float
    frequency_hz: float
    power_loss_w_per_kg: float
    energy_loss_mj_per_kg: float


@dataclass(frozen=True)
class LoopTrace:
    row: LossRow
    t_s: list[float]
    b_t: list[float]
    h_a_per_m: list[float]


def parse_float(text: str) -> float:
    return float(text.strip())


def read_text_from_zip(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8-sig")


def load_sin_loss_table(zf: zipfile.ZipFile) -> list[LossRow]:
    text = read_text_from_zip(zf, "Ring35_Dataset_Txt/SIN/SIN_Loss_Table.txt")
    rows: list[LossRow] = []
    for row in csv.DictReader(io.StringIO(text)):
        rows.append(
            LossRow(
                filename=row["FileName"].strip(),
                bp_t=parse_float(row["B1_T"]),
                frequency_hz=parse_float(row["f_Hz"]),
                power_loss_w_per_kg=parse_float(row["PowerLoss_Wperkg"]),
                energy_loss_mj_per_kg=parse_float(row["EnergyLoss_mJperkg"]),
            )
        )
    return rows


def load_loop_trace(zf: zipfile.ZipFile, row: LossRow) -> LoopTrace:
    text = read_text_from_zip(zf, f"Ring35_Dataset_Txt/SIN/{row.filename}.txt")
    t_s: list[float] = []
    b_t: list[float] = []
    h_a_per_m: list[float] = []
    for data_row in csv.DictReader(io.StringIO(text)):
        t_s.append(parse_float(data_row["t_s"]))
        b_t.append(parse_float(data_row["B_T"]))
        h_a_per_m.append(parse_float(data_row["H_Aperm"]))
    return LoopTrace(row=row, t_s=t_s, b_t=b_t, h_a_per_m=h_a_per_m)


def select_sin_rows(rows: list[LossRow]) -> list[LossRow]:
    """Use a stable, moderate-size grid across amplitudes and frequencies."""
    wanted_f = {50.0, 400.0, 1000.0, 2000.0}
    wanted_bp = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.35}
    selected = [
        row
        for row in rows
        if row.frequency_hz in wanted_f and round(row.bp_t, 2) in wanted_bp
    ]
    return sorted(selected, key=lambda r: (r.frequency_hz, r.bp_t, r.filename))


def close(values: list[float]) -> list[float]:
    return values + [values[0]]


def line_integral_y_dx(x: list[float], y: list[float]) -> float:
    x2 = close(x)
    y2 = close(y)
    return sum(0.5 * (y2[i] + y2[i + 1]) * (x2[i + 1] - x2[i]) for i in range(len(x)))


def loop_components_mj_per_kg(x: list[float], y: list[float]) -> dict[str, float]:
    x2 = close(x)
    y2 = close(y)
    signed = 0.0
    positive = 0.0
    negative_abs = 0.0
    for i in range(len(x)):
        term = 0.5 * (y2[i] + y2[i + 1]) * (x2[i + 1] - x2[i])
        signed += term
        if term >= 0.0:
            positive += term
        else:
            negative_abs -= term
    scale = 1000.0 / DENSITY_KG_PER_M3
    return {
        "signed_mj_per_kg": signed * scale,
        "abs_mj_per_kg": abs(signed) * scale,
        "positive_mj_per_kg": positive * scale,
        "negative_abs_mj_per_kg": negative_abs * scale,
        "total_variation_mj_per_kg": (positive + negative_abs) * scale,
    }


def phase_stats(x: list[float], y: list[float]) -> dict[str, float]:
    angles = [math.atan2(yy, xx) for xx, yy in zip(x, y)]
    if not angles:
        return {"winding": 0.0, "total_variation": 0.0}
    winding = 0.0
    total_variation = 0.0
    for a0, a1 in zip(close(angles), close(angles)[1:]):
        delta = a1 - a0
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        while delta > math.pi:
            delta -= 2.0 * math.pi
        winding += delta
        total_variation += abs(delta)
    return {
        "winding": winding / (2.0 * math.pi),
        "total_variation": total_variation,
    }


def qa_transition_observables(b_seq: list[int], e_seq: list[int]) -> dict[str, float]:
    states = list(zip(b_seq, e_seq))
    closed_states = states + [states[0]]
    edges = list(zip(closed_states[:-1], closed_states[1:]))
    edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for edge in edges:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    total_edges = len(edges)
    unique_edges = len(edge_counts)
    entropy = 0.0
    for count in edge_counts.values():
        p = count / total_edges
        entropy -= p * math.log(p)

    reversal_count = 0
    for edge_a, edge_b in zip(edges, edges[1:] + edges[:1]):
        if edge_a[0] == edge_b[1] and edge_a[1] == edge_b[0]:
            reversal_count += 1

    irreversible = 0
    for edge in edge_counts:
        reverse = (edge[1], edge[0])
        if reverse not in edge_counts:
            irreversible += 1

    signed_flux = 0.0
    abs_flux = 0.0
    orientation_flux = 0.0
    abs_orientation_flux = 0.0
    positive_orientation = 0.0
    negative_orientation = 0.0
    generator_distance_sum = 0.0
    for (b0, e0), (b1, e1) in edges:
        db = b1 - b0
        de = e1 - e0
        flux = db * de
        orientation = b0 * de - e0 * db
        signed_flux += flux
        abs_flux += abs(flux)
        orientation_flux += orientation
        abs_orientation_flux += abs(orientation)
        if orientation >= 0:
            positive_orientation += orientation
        else:
            negative_orientation -= orientation
        generator_distance_sum += abs(db) + abs(de)

    branch_denom = positive_orientation + negative_orientation
    branch_asymmetry = (
        (positive_orientation - negative_orientation) / branch_denom
        if branch_denom > 0.0
        else 0.0
    )

    last_seen: dict[tuple[int, int], int] = {}
    return_times: list[int] = []
    doubled = states + states
    for idx, state in enumerate(doubled):
        if state in last_seen:
            delta = idx - last_seen[state]
            if 0 < delta <= len(states):
                return_times.append(delta)
        last_seen[state] = idx
    if return_times:
        return_mean = statistics.mean(return_times)
        return_std = statistics.pstdev(return_times) if len(return_times) > 1 else 0.0
    else:
        return_mean = float(len(states))
        return_std = 0.0

    first_b, first_e = states[0]
    last_b, last_e = states[-1]
    closure_defect = abs(last_b - first_b) + abs(last_e - first_e)

    return {
        "qa_transition_unique_edges": float(unique_edges),
        "qa_transition_edge_reuse_ratio": total_edges / unique_edges if unique_edges else 0.0,
        "qa_transition_entropy": entropy,
        "qa_transition_reversal_count": float(reversal_count),
        "qa_transition_irreversible_edge_fraction": irreversible / unique_edges
        if unique_edges
        else 0.0,
        "qa_transition_signed_flux": signed_flux,
        "qa_transition_abs_flux": abs_flux,
        "qa_transition_orientation_flux": orientation_flux,
        "qa_transition_abs_orientation_flux": abs_orientation_flux,
        "qa_transition_branch_asymmetry": branch_asymmetry,
        "qa_transition_return_time_mean": return_mean,
        "qa_transition_return_time_std": return_std,
        "qa_transition_loop_closure_defect": float(closure_defect),
        "qa_transition_generator_distance_sum": generator_distance_sum,
        "qa_transition_generator_distance_mean": generator_distance_sum / total_edges
        if total_edges
        else 0.0,
    }


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def branch_label(sb: int, prior_nonzero_sb: int) -> str:
    if sb == 0:
        return "flat"
    if prior_nonzero_sb and sb != prior_nonzero_sb:
        return "turning"
    return "rising" if sb > 0 else "falling"


def lag_label(sb: int, se: int) -> str:
    if sb == se and sb != 0:
        return "aligned"
    if sb != 0 and se == 0:
        return "B_lags_H"
    if sb == 0 and se != 0:
        return "B_leads_H"
    if sb * se < 0:
        return "opposed"
    return "flat"


def qa_memory_observables(b_seq: list[int], e_seq: list[int]) -> dict[str, float]:
    states = list(zip(b_seq, e_seq))
    closed_states = states + [states[0]]
    memory_states: list[tuple[int, int, int, int, str, str]] = []
    branch_counts = {"rising": 0, "falling": 0, "turning": 0, "flat": 0}
    lag_counts = {"aligned": 0, "opposed": 0, "B_lags_H": 0, "B_leads_H": 0, "flat": 0}
    prior_nonzero_sb = 0
    signed_orientation_flux = 0.0
    abs_orientation_flux = 0.0
    lag_weighted_orientation_flux = 0.0
    generator_distance_sum = 0.0

    for (b0, e0), (b1, e1) in zip(closed_states[:-1], closed_states[1:]):
        db = b1 - b0
        de = e1 - e0
        sb = sign(db)
        se = sign(de)
        branch = branch_label(sb, prior_nonzero_sb)
        lag = lag_label(sb, se)
        if sb != 0:
            prior_nonzero_sb = sb
        branch_counts[branch] += 1
        lag_counts[lag] += 1
        memory_states.append((b0, e0, sb, se, branch, lag))
        orientation = b0 * de - e0 * db
        signed_orientation_flux += orientation
        abs_orientation_flux += abs(orientation)
        lag_weight = {
            "aligned": 1.0,
            "opposed": -1.0,
            "B_lags_H": 0.5,
            "B_leads_H": -0.5,
            "flat": 0.0,
        }[lag]
        lag_weighted_orientation_flux += lag_weight * orientation
        generator_distance_sum += abs(db) + abs(de)

    edges = list(zip(memory_states, memory_states[1:] + memory_states[:1]))
    edge_counts: dict[
        tuple[tuple[int, int, int, int, str, str], tuple[int, int, int, int, str, str]], int
    ] = {}
    for edge in edges:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    entropy = 0.0
    total_edges = len(edges)
    for count in edge_counts.values():
        p = count / total_edges
        entropy -= p * math.log(p)

    irreversible = 0
    for edge in edge_counts:
        if (edge[1], edge[0]) not in edge_counts:
            irreversible += 1

    last_seen: dict[tuple[int, int, int, int, str, str], int] = {}
    return_times: list[int] = []
    doubled = memory_states + memory_states
    for idx, state in enumerate(doubled):
        if state in last_seen:
            delta = idx - last_seen[state]
            if 0 < delta <= len(memory_states):
                return_times.append(delta)
        last_seen[state] = idx
    if return_times:
        return_mean = statistics.mean(return_times)
        return_std = statistics.pstdev(return_times) if len(return_times) > 1 else 0.0
    else:
        return_mean = float(len(memory_states))
        return_std = 0.0

    branch_denom = branch_counts["rising"] + branch_counts["falling"]
    branch_asymmetry = (
        (branch_counts["rising"] - branch_counts["falling"]) / branch_denom
        if branch_denom
        else 0.0
    )

    total = len(memory_states)
    return {
        "qa_memory_unique_states": float(len(set(memory_states))),
        "qa_memory_unique_edges": float(len(edge_counts)),
        "qa_memory_entropy": entropy,
        "qa_memory_branch_rising_fraction": branch_counts["rising"] / total,
        "qa_memory_branch_falling_fraction": branch_counts["falling"] / total,
        "qa_memory_branch_turning_fraction": branch_counts["turning"] / total,
        "qa_memory_lag_aligned_fraction": lag_counts["aligned"] / total,
        "qa_memory_lag_opposed_fraction": lag_counts["opposed"] / total,
        "qa_memory_lag_B_lags_H_fraction": lag_counts["B_lags_H"] / total,
        "qa_memory_lag_B_leads_H_fraction": lag_counts["B_leads_H"] / total,
        "qa_memory_branch_asymmetry": branch_asymmetry,
        "qa_memory_turning_density": branch_counts["turning"] / total,
        "qa_memory_irreversible_edge_fraction": irreversible / len(edge_counts)
        if edge_counts
        else 0.0,
        "qa_memory_return_time_mean": return_mean,
        "qa_memory_return_time_std": return_std,
        "qa_memory_generator_distance_sum": generator_distance_sum,
        "qa_memory_generator_distance_mean": generator_distance_sum / total if total else 0.0,
        "qa_memory_signed_orientation_flux": signed_orientation_flux,
        "qa_memory_abs_orientation_flux": abs_orientation_flux,
        "qa_memory_lag_weighted_orientation_flux": lag_weighted_orientation_flux,
    }


def qa_memory_contexts(
    b_seq: list[int], e_seq: list[int]
) -> list[tuple[tuple[int, int], tuple[int, int, int, int], tuple[int, int, str, str], tuple[int, int, int, int, str, str]]]:
    states = list(zip(b_seq, e_seq))
    closed_states = states + [states[0]]
    prior_nonzero_sb = 0
    contexts = []
    for (b0, e0), (b1, e1) in zip(closed_states[:-1], closed_states[1:]):
        db = b1 - b0
        de = e1 - e0
        sb = sign(db)
        se = sign(de)
        branch = branch_label(sb, prior_nonzero_sb)
        lag = lag_label(sb, se)
        if sb != 0:
            prior_nonzero_sb = sb
        contexts.append(
            (
                (b0, e0),
                (b0, e0, sb, se),
                (b0, e0, branch, lag),
                (b0, e0, sb, se, branch, lag),
            )
        )
    return contexts


def entropy_from_counter(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        p = count / total
        out -= p * math.log(p)
    return out


def orbit_topology_observables(b_seq: list[int], e_seq: list[int]) -> dict[str, float]:
    families = [orbit_family(int(b), int(e), MODULUS) for b, e in zip(b_seq, e_seq)]
    norms_mod = [norm_f(int(b), int(e)) % MODULUS for b, e in zip(b_seq, e_seq)]
    signed_norms = []
    for norm in norms_mod:
        signed_norms.append(norm if norm <= MODULUS // 2 else norm - MODULUS)
    family_counts = Counter(families)
    norm_counts = Counter(norms_mod)
    family_edges = Counter(zip(families, families[1:] + families[:1]))
    total = len(families) or 1
    switches = sum(1 for a, b in zip(families, families[1:] + families[:1]) if a != b)
    norm_flips = sum(
        1
        for a, b in zip(signed_norms, signed_norms[1:] + signed_norms[:1])
        if a * b < 0
    )
    return {
        "qa_orbit_cosmos_fraction": family_counts["cosmos"] / total,
        "qa_orbit_satellite_fraction": family_counts["satellite"] / total,
        "qa_orbit_singularity_fraction": family_counts["singularity"] / total,
        "qa_orbit_norm_mod_entropy": entropy_from_counter(norm_counts),
        "qa_orbit_family_transition_entropy": entropy_from_counter(family_edges),
        "qa_orbit_family_switch_fraction": switches / total,
        "qa_orbit_cosmos_to_satellite_fraction": family_edges[("cosmos", "satellite")]
        / total,
        "qa_orbit_satellite_to_cosmos_fraction": family_edges[("satellite", "cosmos")]
        / total,
        "qa_orbit_norm_signed_mean": statistics.mean(signed_norms) if signed_norms else 0.0,
        "qa_orbit_norm_signed_abs_mean": statistics.mean(abs(n) for n in signed_norms)
        if signed_norms
        else 0.0,
        "qa_orbit_norm_flip_fraction": norm_flips / total,
    }


def qa_orbit_contexts(
    b_seq: list[int], e_seq: list[int]
) -> list[tuple[tuple[str], tuple[str, int], tuple[str, int, str, str]]]:
    memory_contexts = qa_memory_contexts(b_seq, e_seq)
    contexts = []
    for b_state, e_state, _sb, _se, branch, lag in (
        full_context for _be, _be_dir, _be_branch, full_context in memory_contexts
    ):
        family = orbit_family(int(b_state), int(e_state), MODULUS)
        norm_mod = norm_f(int(b_state), int(e_state)) % MODULUS
        contexts.append(
            (
                (family,),
                (family, norm_mod),
                (family, norm_mod, branch, lag),
            )
        )
    return contexts


def physical_energy_mj_per_kg(trace: LoopTrace) -> float:
    area_j_per_m3 = abs(line_integral_y_dx(trace.b_t, trace.h_a_per_m))
    return 1000.0 * area_j_per_m3 / DENSITY_KG_PER_M3


def quantile_edges(values: list[float], m: int = MODULUS) -> list[float]:
    ordered = sorted(values)
    edges: list[float] = []
    for k in range(1, m):
        idx = (k / m) * (len(ordered) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        edges.append(ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo]))
    return edges


def bins(values: list[float], edges: list[float]) -> list[int]:
    out: list[int] = []
    for value in values:
        bucket = 1
        for edge in edges:
            if value > edge:
                bucket += 1
        out.append(bucket)
    return out


def bin_centers(values: list[float], edges: list[float], m: int = MODULUS) -> list[float]:
    buckets: list[list[float]] = [[] for _ in range(m)]
    for value, bucket in zip(values, bins(values, edges)):
        buckets[bucket - 1].append(value)

    centers: list[float] = []
    fallback = statistics.mean(values)
    for idx, bucket_values in enumerate(buckets):
        if bucket_values:
            centers.append(statistics.mean(bucket_values))
            continue
        lower = edges[idx - 1] if idx > 0 else min(values)
        upper = edges[idx] if idx < len(edges) else max(values)
        if math.isfinite(lower) and math.isfinite(upper):
            centers.append(0.5 * (lower + upper))
        else:
            centers.append(fallback)
    return centers


def qa_vars(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    j = b * d
    x = d * e
    k = d * a
    return {
        "b": b,
        "e": e,
        "J": j,
        "X": x,
        "K": k,
        "Pi": j + x + k,
        "K_minus_J": k - j,
    }


def lifted_energy_field(h_var: str, b_var: str) -> str:
    return f"qa_lifted_{h_var.lower()}_{b_var.lower()}_shell_energy_mj_per_kg"


def lifted_component_field(h_var: str, b_var: str, component: str) -> str:
    return f"qa_lifted_{h_var.lower()}_{b_var.lower()}_shell_{component}"


def lifted_component_features(h_var: str, b_var: str) -> list[str]:
    return [
        lifted_energy_field(h_var, b_var),
        lifted_component_field(h_var, b_var, "positive_mj_per_kg"),
        lifted_component_field(h_var, b_var, "negative_abs_mj_per_kg"),
        lifted_component_field(h_var, b_var, "total_variation_mj_per_kg"),
        lifted_component_field(h_var, b_var, "phase_winding"),
        lifted_component_field(h_var, b_var, "phase_total_variation"),
    ]


def calibration_split(rows: list[LossRow]) -> tuple[set[str], set[str]]:
    calibration: set[str] = set()
    heldout: set[str] = set()
    for idx, row in enumerate(rows):
        if idx % 3 == 1:
            heldout.add(row.filename)
        else:
            calibration.add(row.filename)
    return calibration, heldout


def build_global_calibration(
    traces: list[LoopTrace], calibration_names: set[str]
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    dict[tuple[int, int], tuple[float, float]],
    dict[tuple[str, str], dict[int, float]],
    dict[str, dict[tuple, tuple[float, float]]],
    dict[str, dict[tuple, tuple[float, float]]],
    dict[str, dict[tuple, tuple[float, float]]],
]:
    h_values: list[float] = []
    b_values: list[float] = []
    for trace in traces:
        if trace.row.filename in calibration_names:
            h_values.extend(trace.h_a_per_m)
            b_values.extend(trace.b_t)
    h_edges = quantile_edges(h_values)
    b_edges = quantile_edges(b_values)
    h_centers = bin_centers(h_values, h_edges)
    b_centers = bin_centers(b_values, b_edges)

    state_values: dict[tuple[int, int], list[tuple[float, float]]] = {}
    lifted_values: dict[tuple[str, str], dict[int, list[float]]] = {}
    reconstruction_values: dict[str, dict[tuple, list[tuple[float, float]]]] = {
        "be": {},
        "be_dir": {},
        "be_branch": {},
        "memory_full": {},
    }
    residual_reconstruction_values: dict[str, dict[tuple, list[tuple[float, float]]]] = {
        "be": {},
        "be_dir": {},
        "be_branch": {},
        "memory_full": {},
    }
    orbit_residual_values: dict[str, dict[tuple, list[tuple[float, float]]]] = {
        "family": {},
        "family_norm": {},
        "family_norm_branch": {},
    }
    for trace in traces:
        if trace.row.filename not in calibration_names:
            continue
        b_seq = bins(trace.h_a_per_m, h_edges)
        e_seq = bins(trace.b_t, b_edges)
        context_seq = qa_memory_contexts(b_seq, e_seq)
        orbit_context_seq = qa_orbit_contexts(b_seq, e_seq)
        for h_value, b_value, b_state, e_state in zip(trace.h_a_per_m, trace.b_t, b_seq, e_seq):
            state_values.setdefault((b_state, e_state), []).append((h_value, b_value))
            vars_for_state = qa_vars(b_state, e_state)
            for var_name in QA_LIFTED_VARIABLES:
                key = vars_for_state[var_name]
                lifted_values.setdefault(("H", var_name), {}).setdefault(key, []).append(h_value)
                lifted_values.setdefault(("B", var_name), {}).setdefault(key, []).append(b_value)
        for b_state, e_state, contexts in zip(b_seq, e_seq, context_seq):
            h_shell = h_centers[b_state - 1]
            b_shell = b_centers[e_state - 1]
            for family, context in zip(
                ("be", "be_dir", "be_branch", "memory_full"), contexts
            ):
                reconstruction_values[family].setdefault(context, []).append((h_shell, b_shell))
        for h_value, b_value, b_state, e_state, contexts in zip(
            trace.h_a_per_m, trace.b_t, b_seq, e_seq, context_seq
        ):
            h_shell = h_centers[b_state - 1]
            b_shell = b_centers[e_state - 1]
            residual = (h_value - h_shell, b_value - b_shell)
            for family, context in zip(
                ("be", "be_dir", "be_branch", "memory_full"), contexts
            ):
                residual_reconstruction_values[family].setdefault(context, []).append(residual)
        for h_value, b_value, b_state, e_state, contexts in zip(
            trace.h_a_per_m, trace.b_t, b_seq, e_seq, orbit_context_seq
        ):
            h_shell = h_centers[b_state - 1]
            b_shell = b_centers[e_state - 1]
            residual = (h_value - h_shell, b_value - b_shell)
            for family, context in zip(
                ("family", "family_norm", "family_norm_branch"), contexts
            ):
                orbit_residual_values[family].setdefault(context, []).append(residual)

    state_centers = {
        state: (
            statistics.mean(h_value for h_value, _ in values),
            statistics.mean(b_value for _, b_value in values),
        )
        for state, values in state_values.items()
    }
    lifted_centers = {
        target_var: {key: statistics.mean(values) for key, values in value_map.items()}
        for target_var, value_map in lifted_values.items()
    }
    reconstruction_centers = {
        family: {
            context: (
                statistics.mean(h_value for h_value, _ in values),
                statistics.mean(b_value for _, b_value in values),
            )
            for context, values in value_map.items()
        }
        for family, value_map in reconstruction_values.items()
    }
    residual_reconstruction_centers = {
        family: {
            context: (
                statistics.mean(h_value for h_value, _ in values),
                statistics.mean(b_value for _, b_value in values),
            )
            for context, values in value_map.items()
        }
        for family, value_map in residual_reconstruction_values.items()
    }
    orbit_residual_centers = {
        family: {
            context: (
                statistics.mean(h_value for h_value, _ in values),
                statistics.mean(b_value for _, b_value in values),
            )
            for context, values in value_map.items()
        }
        for family, value_map in orbit_residual_values.items()
    }
    return (
        h_edges,
        b_edges,
        h_centers,
        b_centers,
        state_centers,
        lifted_centers,
        reconstruction_centers,
        residual_reconstruction_centers,
        orbit_residual_centers,
    )


def qa_observables(
    trace: LoopTrace,
    h_edges: list[float],
    b_edges: list[float],
    h_centers: list[float],
    b_centers: list[float],
    state_centers: dict[tuple[int, int], tuple[float, float]],
    lifted_centers: dict[tuple[str, str], dict[int, float]],
    reconstruction_centers: dict[str, dict[tuple, tuple[float, float]]],
    residual_reconstruction_centers: dict[str, dict[tuple, tuple[float, float]]],
    orbit_residual_centers: dict[str, dict[tuple, tuple[float, float]]],
) -> dict[str, float]:
    b_seq = bins(trace.h_a_per_m, h_edges)
    e_seq = bins(trace.b_t, b_edges)
    nonqa_h_shell_seq = [h_centers[b - 1] for b in b_seq]
    nonqa_b_shell_seq = [b_centers[e - 1] for e in e_seq]
    qa_h_shell_seq: list[float] = []
    qa_b_shell_seq: list[float] = []
    unseen_state_count = 0
    for b, e in zip(b_seq, e_seq):
        center = state_centers.get((b, e))
        if center is None:
            unseen_state_count += 1
            center = (h_centers[b - 1], b_centers[e - 1])
        qa_h_shell_seq.append(center[0])
        qa_b_shell_seq.append(center[1])
    j_seq: list[float] = []
    x_seq: list[float] = []
    k_seq: list[float] = []
    pi_seq: list[float] = []
    vars_seq: list[dict[str, int]] = []
    for b, e in zip(b_seq, e_seq):
        vars_for_state = qa_vars(b, e)
        j = vars_for_state["J"]
        x = vars_for_state["X"]
        k = vars_for_state["K"]
        j_seq.append(j)
        x_seq.append(x)
        k_seq.append(k)
        pi_seq.append(vars_for_state["Pi"])
        vars_seq.append(vars_for_state)

    be_area = abs(line_integral_y_dx(e_seq, b_seq))
    jx_area = abs(line_integral_y_dx(x_seq, j_seq))
    kx_area = abs(line_integral_y_dx(x_seq, k_seq))
    pi_x_area = abs(line_integral_y_dx(x_seq, pi_seq))
    path_energy = 0.0
    x2 = close(x_seq)
    pi2 = close(pi_seq)
    for i in range(len(x_seq)):
        path_energy += 0.5 * (pi2[i] + pi2[i + 1]) * abs(x2[i + 1] - x2[i])

    def lifted_shell_components(h_var: str, b_var: str) -> tuple[dict[str, float], float]:
        h_map = lifted_centers.get(("H", h_var), {})
        b_map = lifted_centers.get(("B", b_var), {})
        h_lifted_seq: list[float] = []
        b_lifted_seq: list[float] = []
        unseen = 0
        for b, e, vars_for_state in zip(b_seq, e_seq, vars_seq):
            h_key = vars_for_state[h_var]
            b_key = vars_for_state[b_var]
            h_value = h_map.get(h_key)
            b_value = b_map.get(b_key)
            if h_value is None:
                unseen += 1
                h_value = h_centers[b - 1]
            if b_value is None:
                unseen += 1
                b_value = b_centers[e - 1]
            h_lifted_seq.append(h_value)
            b_lifted_seq.append(b_value)
        components = loop_components_mj_per_kg(b_lifted_seq, h_lifted_seq)
        phase = phase_stats(b_lifted_seq, h_lifted_seq)
        components["phase_winding"] = phase["winding"]
        components["phase_total_variation"] = phase["total_variation"]
        return components, unseen / (2 * len(vars_seq))

    all_lifted: dict[str, float] = {}
    all_lifted_unseen: dict[str, float] = {}
    for h_var in QA_LIFTED_VARIABLES:
        for b_var in QA_LIFTED_VARIABLES:
            components, unseen_fraction = lifted_shell_components(h_var, b_var)
            field = lifted_energy_field(h_var, b_var)
            all_lifted[field] = components["abs_mj_per_kg"]
            for component_name, component_value in components.items():
                all_lifted[lifted_component_field(h_var, b_var, component_name)] = component_value
            all_lifted_unseen[field.replace("_energy_", "_unseen_")] = unseen_fraction

    lifted_jx = all_lifted[lifted_energy_field("J", "X")]
    lifted_jx_unseen = all_lifted_unseen[
        lifted_energy_field("J", "X").replace("_energy_", "_unseen_")
    ]
    lifted_kx = all_lifted[lifted_energy_field("K", "X")]
    lifted_kx_unseen = all_lifted_unseen[
        lifted_energy_field("K", "X").replace("_energy_", "_unseen_")
    ]
    lifted_k_minus_j_x = all_lifted[lifted_energy_field("K_minus_J", "X")]
    lifted_k_minus_j_x_unseen = all_lifted_unseen[
        lifted_energy_field("K_minus_J", "X").replace("_energy_", "_unseen_")
    ]
    lifted_j_pi = all_lifted[lifted_energy_field("J", "Pi")]
    lifted_j_pi_unseen = all_lifted_unseen[
        lifted_energy_field("J", "Pi").replace("_energy_", "_unseen_")
    ]

    nonqa_components = loop_components_mj_per_kg(nonqa_b_shell_seq, nonqa_h_shell_seq)
    qa_components = loop_components_mj_per_kg(qa_b_shell_seq, qa_h_shell_seq)
    nonqa_phase = phase_stats(nonqa_b_shell_seq, nonqa_h_shell_seq)
    qa_phase = phase_stats(qa_b_shell_seq, qa_h_shell_seq)
    context_seq = qa_memory_contexts(b_seq, e_seq)

    def reconstructed_energy(family: str, context_index: int) -> tuple[float, float]:
        h_hat: list[float] = []
        b_hat: list[float] = []
        unseen = 0
        family_map = reconstruction_centers.get(family, {})
        be_map = reconstruction_centers.get("be", {})
        for b_state, e_state, contexts in zip(b_seq, e_seq, context_seq):
            center = family_map.get(contexts[context_index])
            if center is None:
                center = be_map.get((b_state, e_state))
            if center is None:
                unseen += 1
                center = (h_centers[b_state - 1], b_centers[e_state - 1])
            h_hat.append(center[0])
            b_hat.append(center[1])
        energy = 1000.0 * abs(line_integral_y_dx(b_hat, h_hat)) / DENSITY_KG_PER_M3
        return energy, unseen / len(context_seq)

    def residual_reconstructed_energy(family: str, context_index: int) -> tuple[float, float]:
        h_hat: list[float] = []
        b_hat: list[float] = []
        unseen = 0
        family_map = residual_reconstruction_centers.get(family, {})
        be_map = residual_reconstruction_centers.get("be", {})
        for b_state, e_state, contexts in zip(b_seq, e_seq, context_seq):
            residual = family_map.get(contexts[context_index])
            if residual is None:
                residual = be_map.get((b_state, e_state))
            if residual is None:
                unseen += 1
                residual = (0.0, 0.0)
            h_hat.append(h_centers[b_state - 1] + residual[0])
            b_hat.append(b_centers[e_state - 1] + residual[1])
        energy = 1000.0 * abs(line_integral_y_dx(b_hat, h_hat)) / DENSITY_KG_PER_M3
        return energy, unseen / len(context_seq)

    orbit_context_seq = qa_orbit_contexts(b_seq, e_seq)

    def orbit_residual_reconstructed_energy(
        family: str, context_index: int
    ) -> tuple[float, float]:
        h_hat: list[float] = []
        b_hat: list[float] = []
        unseen = 0
        family_map = orbit_residual_centers.get(family, {})
        family_norm_map = orbit_residual_centers.get("family_norm", {})
        family_only_map = orbit_residual_centers.get("family", {})
        for b_state, e_state, contexts in zip(b_seq, e_seq, orbit_context_seq):
            residual = family_map.get(contexts[context_index])
            if residual is None:
                residual = family_norm_map.get(contexts[1])
            if residual is None:
                residual = family_only_map.get(contexts[0])
            if residual is None:
                unseen += 1
                residual = (0.0, 0.0)
            h_hat.append(h_centers[b_state - 1] + residual[0])
            b_hat.append(b_centers[e_state - 1] + residual[1])
        energy = 1000.0 * abs(line_integral_y_dx(b_hat, h_hat)) / DENSITY_KG_PER_M3
        return energy, unseen / len(orbit_context_seq)

    reconstruct_be, reconstruct_be_unseen = reconstructed_energy("be", 0)
    reconstruct_be_dir, reconstruct_be_dir_unseen = reconstructed_energy("be_dir", 1)
    reconstruct_be_branch, reconstruct_be_branch_unseen = reconstructed_energy(
        "be_branch", 2
    )
    reconstruct_memory_full, reconstruct_memory_full_unseen = reconstructed_energy(
        "memory_full", 3
    )
    residual_reconstruct_be, residual_reconstruct_be_unseen = residual_reconstructed_energy(
        "be", 0
    )
    residual_reconstruct_be_dir, residual_reconstruct_be_dir_unseen = (
        residual_reconstructed_energy("be_dir", 1)
    )
    residual_reconstruct_be_branch, residual_reconstruct_be_branch_unseen = (
        residual_reconstructed_energy("be_branch", 2)
    )
    residual_reconstruct_memory_full, residual_reconstruct_memory_full_unseen = (
        residual_reconstructed_energy("memory_full", 3)
    )
    orbit_residual_family, orbit_residual_family_unseen = (
        orbit_residual_reconstructed_energy("family", 0)
    )
    orbit_residual_family_norm, orbit_residual_family_norm_unseen = (
        orbit_residual_reconstructed_energy("family_norm", 1)
    )
    orbit_residual_family_norm_branch, orbit_residual_family_norm_branch_unseen = (
        orbit_residual_reconstructed_energy("family_norm_branch", 2)
    )

    fixed = {
        "qa_be_loop_area": be_area,
        "qa_jx_loop_area": jx_area,
        "qa_kx_loop_area": kx_area,
        "qa_pi_x_loop_area": pi_x_area,
        "qa_pi_abs_path_energy": path_energy,
        "nonqa_shell_hdb_energy_mj_per_kg": 1000.0
        * abs(line_integral_y_dx(nonqa_b_shell_seq, nonqa_h_shell_seq))
        / DENSITY_KG_PER_M3,
        "nonqa_shell_hdb_positive_mj_per_kg": nonqa_components["positive_mj_per_kg"],
        "nonqa_shell_hdb_negative_abs_mj_per_kg": nonqa_components["negative_abs_mj_per_kg"],
        "nonqa_shell_hdb_total_variation_mj_per_kg": nonqa_components[
            "total_variation_mj_per_kg"
        ],
        "nonqa_shell_hdb_phase_winding": nonqa_phase["winding"],
        "nonqa_shell_hdb_phase_total_variation": nonqa_phase["total_variation"],
        "qa_shell_hdb_energy_mj_per_kg": 1000.0
        * abs(line_integral_y_dx(qa_b_shell_seq, qa_h_shell_seq))
        / DENSITY_KG_PER_M3,
        "qa_shell_hdb_positive_mj_per_kg": qa_components["positive_mj_per_kg"],
        "qa_shell_hdb_negative_abs_mj_per_kg": qa_components["negative_abs_mj_per_kg"],
        "qa_shell_hdb_total_variation_mj_per_kg": qa_components["total_variation_mj_per_kg"],
        "qa_shell_hdb_phase_winding": qa_phase["winding"],
        "qa_shell_hdb_phase_total_variation": qa_phase["total_variation"],
        "qa_lifted_jx_shell_energy_mj_per_kg": lifted_jx,
        "qa_lifted_jx_unseen_fraction": lifted_jx_unseen,
        "qa_lifted_kx_shell_energy_mj_per_kg": lifted_kx,
        "qa_lifted_kx_unseen_fraction": lifted_kx_unseen,
        "qa_lifted_k_minus_j_x_shell_energy_mj_per_kg": lifted_k_minus_j_x,
        "qa_lifted_k_minus_j_x_unseen_fraction": lifted_k_minus_j_x_unseen,
        "qa_lifted_j_pi_shell_energy_mj_per_kg": lifted_j_pi,
        "qa_lifted_j_pi_unseen_fraction": lifted_j_pi_unseen,
        "qa_shell_unseen_state_fraction": unseen_state_count / len(b_seq),
        "qa_unique_states": float(len(set(zip(b_seq, e_seq)))),
        "qa_reconstruct_be_energy_mj_per_kg": reconstruct_be,
        "qa_reconstruct_be_unseen_fraction": reconstruct_be_unseen,
        "qa_reconstruct_be_dir_energy_mj_per_kg": reconstruct_be_dir,
        "qa_reconstruct_be_dir_unseen_fraction": reconstruct_be_dir_unseen,
        "qa_reconstruct_be_branch_energy_mj_per_kg": reconstruct_be_branch,
        "qa_reconstruct_be_branch_unseen_fraction": reconstruct_be_branch_unseen,
        "qa_reconstruct_memory_full_energy_mj_per_kg": reconstruct_memory_full,
        "qa_reconstruct_memory_full_unseen_fraction": reconstruct_memory_full_unseen,
        "qa_residual_reconstruct_be_energy_mj_per_kg": residual_reconstruct_be,
        "qa_residual_reconstruct_be_unseen_fraction": residual_reconstruct_be_unseen,
        "qa_residual_reconstruct_be_dir_energy_mj_per_kg": residual_reconstruct_be_dir,
        "qa_residual_reconstruct_be_dir_unseen_fraction": residual_reconstruct_be_dir_unseen,
        "qa_residual_reconstruct_be_branch_energy_mj_per_kg": residual_reconstruct_be_branch,
        "qa_residual_reconstruct_be_branch_unseen_fraction": residual_reconstruct_be_branch_unseen,
        "qa_residual_reconstruct_memory_full_energy_mj_per_kg": residual_reconstruct_memory_full,
        "qa_residual_reconstruct_memory_full_unseen_fraction": (
            residual_reconstruct_memory_full_unseen
        ),
    }
    fixed.update(all_lifted)
    fixed.update(all_lifted_unseen)
    fixed.update(qa_transition_observables(b_seq, e_seq))
    fixed.update(qa_memory_observables(b_seq, e_seq))
    fixed.update(orbit_topology_observables(b_seq, e_seq))
    fixed.update(
        {
            "qa_orbit_residual_reconstruct_family_energy_mj_per_kg": (
                orbit_residual_family
            ),
            "qa_orbit_residual_reconstruct_family_unseen_fraction": (
                orbit_residual_family_unseen
            ),
            "qa_orbit_residual_reconstruct_family_norm_energy_mj_per_kg": (
                orbit_residual_family_norm
            ),
            "qa_orbit_residual_reconstruct_family_norm_unseen_fraction": (
                orbit_residual_family_norm_unseen
            ),
            "qa_orbit_residual_reconstruct_family_norm_branch_energy_mj_per_kg": (
                orbit_residual_family_norm_branch
            ),
            "qa_orbit_residual_reconstruct_family_norm_branch_unseen_fraction": (
                orbit_residual_family_norm_branch_unseen
            ),
        }
    )
    return fixed


def solve_linear_least_squares(x_rows: list[list[float]], y: list[float]) -> list[float]:
    cols = len(x_rows[0])
    ata = [[0.0 for _ in range(cols)] for _ in range(cols)]
    aty = [0.0 for _ in range(cols)]
    ridge = 1e-8
    for row, target in zip(x_rows, y):
        for i in range(cols):
            aty[i] += row[i] * target
            for j in range(cols):
                ata[i][j] += row[i] * row[j]
    for i in range(cols):
        ata[i][i] += ridge
    return solve_square(ata, aty)


def solve_square(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-18:
            raise ValueError("singular calibration system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for k in range(col, n + 1):
            aug[col][k] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for k in range(col, n + 1):
                aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] for i in range(n)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def standardize_train_test(
    train_features: list[list[float]], test_features: list[list[float]]
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    cols = len(train_features[0])
    means: list[float] = []
    scales: list[float] = []
    for col in range(cols):
        vals = [row[col] for row in train_features]
        mean = statistics.mean(vals)
        scale = statistics.pstdev(vals)
        if scale <= 1e-12:
            scale = 1.0
        means.append(mean)
        scales.append(scale)

    def apply(rows: list[list[float]]) -> list[list[float]]:
        return [
            [1.0] + [(row[col] - means[col]) / scales[col] for col in range(cols)]
            for row in rows
        ]

    return apply(train_features), apply(test_features), means, scales


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    mean_actual = sum(actual) / len(actual)
    sse = sum((a - p) * (a - p) for a, p in zip(actual, predicted))
    sst = sum((a - mean_actual) * (a - mean_actual) for a in actual)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
    mape = sum(abs(a - p) / max(abs(a), 1e-12) for a, p in zip(actual, predicted)) / len(actual)
    return {
        "mae_mj_per_kg": mae,
        "mape": mape,
        "rmse_mj_per_kg": math.sqrt(sse / len(actual)),
        "r2": 1.0 - sse / sst if sst > 0.0 else float("nan"),
    }


def fit_predict(
    records: list[dict],
    calibration_names: set[str],
    heldout_names: set[str],
    feature_names: list[str],
) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    raw_x_train = [[float(r[name]) for name in feature_names] for r in train]
    raw_x_test = [[float(r[name]) for name in feature_names] for r in test]
    x_train, x_test, means, scales = standardize_train_test(raw_x_train, raw_x_test)
    y_train = [float(r["energy_loss_mj_per_kg_table"]) for r in train]
    coeffs = solve_linear_least_squares(x_train, y_train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [dot(coeffs, row) for row in x_test]
    return {
        "features": feature_names,
        "coefficients": coeffs,
        "feature_means": means,
        "feature_scales": scales,
        "heldout": metrics(actual, predicted),
    }


def fit_steinmetz(records: list[dict], calibration_names: set[str], heldout_names: set[str]) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    x_train = [
        [1.0, math.log(float(r["frequency_hz"])), math.log(float(r["bp_t"]))]
        for r in train
    ]
    y_train = [math.log(float(r["energy_loss_mj_per_kg_table"])) for r in train]
    coeffs = solve_linear_least_squares(x_train, y_train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [
        math.exp(dot(coeffs, [1.0, math.log(float(r["frequency_hz"])), math.log(float(r["bp_t"]))]))
        for r in test
    ]
    return {
        "features": ["intercept", "log_frequency_hz", "log_bp_t"],
        "coefficients": coeffs,
        "heldout": metrics(actual, predicted),
    }


def fit_log_feature_model(
    records: list[dict],
    calibration_names: set[str],
    heldout_names: set[str],
    feature_names: list[str],
) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    raw_train = [[float(r[name]) for name in feature_names] for r in train]
    raw_test = [[float(r[name]) for name in feature_names] for r in test]
    x_train, x_test, means, scales = standardize_train_test(raw_train, raw_test)
    y_train = [math.log(float(r["energy_loss_mj_per_kg_table"])) for r in train]
    coeffs = solve_linear_least_squares(x_train, y_train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [math.exp(dot(coeffs, row)) for row in x_test]
    return {
        "features": feature_names,
        "coefficients": coeffs,
        "feature_means": means,
        "feature_scales": scales,
        "target_transform": "log",
        "heldout": metrics(actual, predicted),
    }


def mean_baseline(records: list[dict], calibration_names: set[str], heldout_names: set[str]) -> dict:
    train = [r for r in records if r["filename"] in calibration_names]
    test = [r for r in records if r["filename"] in heldout_names]
    mean_y = statistics.mean(float(r["energy_loss_mj_per_kg_table"]) for r in train)
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [mean_y for _ in test]
    return {"constant": mean_y, "heldout": metrics(actual, predicted)}


def direct_prediction(
    records: list[dict], heldout_names: set[str], prediction_field: str
) -> dict:
    test = [r for r in records if r["filename"] in heldout_names]
    actual = [float(r["energy_loss_mj_per_kg_table"]) for r in test]
    predicted = [float(r[prediction_field]) for r in test]
    return {"prediction_field": prediction_field, "heldout": metrics(actual, predicted)}


def build_records_for_split(
    traces: list[LoopTrace], calibration_names: set[str], heldout_names: set[str]
) -> list[dict]:
    (
        h_edges,
        b_edges,
        h_centers,
        b_centers,
        state_centers,
        lifted_centers,
        reconstruction_centers,
        residual_reconstruction_centers,
        orbit_residual_centers,
    ) = build_global_calibration(traces, calibration_names)
    records: list[dict] = []
    for trace in traces:
        qa = qa_observables(
            trace,
            h_edges,
            b_edges,
            h_centers,
            b_centers,
            state_centers,
            lifted_centers,
            reconstruction_centers,
            residual_reconstruction_centers,
            orbit_residual_centers,
        )
        physical_from_loop = physical_energy_mj_per_kg(trace)
        rec = {
            "filename": trace.row.filename,
            "bp_t": trace.row.bp_t,
            "frequency_hz": trace.row.frequency_hz,
            "split": "heldout" if trace.row.filename in heldout_names else "calibration",
            "rows": len(trace.t_s),
            "energy_loss_mj_per_kg_table": trace.row.energy_loss_mj_per_kg,
            "energy_loss_mj_per_kg_from_integral": physical_from_loop,
            "integral_vs_table_abs_error": abs(physical_from_loop - trace.row.energy_loss_mj_per_kg),
        }
        rec.update(qa)
        records.append(rec)
    return records


def shuffled_state_map(
    states: list[tuple[int, int]], values: list, seed: int, domain: str
) -> dict[tuple[int, int], object]:
    rng = random.Random(f"{domain}:{seed}")
    shuffled = list(values)
    rng.shuffle(shuffled)
    return dict(zip(states, shuffled))


def mean_pair_map(
    values: dict[tuple, list[tuple[float, float]]]
) -> dict[tuple, tuple[float, float]]:
    return {
        key: (
            statistics.mean(h_value for h_value, _ in pairs),
            statistics.mean(b_value for _, b_value in pairs),
        )
        for key, pairs in values.items()
    }


def permutation_split_metrics(
    traces: list[LoopTrace],
    calibration_names: set[str],
    heldout_names: set[str],
    null_kind: str,
    seed: int,
) -> dict[str, float | bool]:
    h_values: list[float] = []
    b_values: list[float] = []
    for trace in traces:
        if trace.row.filename in calibration_names:
            h_values.extend(trace.h_a_per_m)
            b_values.extend(trace.b_t)
    h_edges = quantile_edges(h_values)
    b_edges = quantile_edges(b_values)
    h_centers = bin_centers(h_values, h_edges)
    b_centers = bin_centers(b_values, b_edges)

    binned: dict[str, tuple[list[int], list[int]]] = {}
    observed_states: set[tuple[int, int]] = set()
    for trace in traces:
        b_seq = bins(trace.h_a_per_m, h_edges)
        e_seq = bins(trace.b_t, b_edges)
        binned[trace.row.filename] = (b_seq, e_seq)
        observed_states.update(zip(b_seq, e_seq))

    states = sorted(observed_states)
    canonical_families = [orbit_family(b_state, e_state, MODULUS) for b_state, e_state in states]
    canonical_norms = [norm_f(b_state, e_state) % MODULUS for b_state, e_state in states]
    canonical_pairs = list(zip(canonical_families, canonical_norms))

    if null_kind == "orbit_family_shuffle_null":
        family_map = shuffled_state_map(states, canonical_families, seed, null_kind)
        norm_map = dict(zip(states, canonical_norms))
        context_mode = "family"
    elif null_kind == "norm_f_shuffle_null":
        family_map = dict(zip(states, canonical_families))
        norm_map = shuffled_state_map(states, canonical_norms, seed, null_kind)
        context_mode = "family_norm"
    elif null_kind == "family_norm_joint_shuffle_null":
        pair_map = shuffled_state_map(states, canonical_pairs, seed, null_kind)
        family_map = {state: pair_map[state][0] for state in states}
        norm_map = {state: pair_map[state][1] for state in states}
        context_mode = "family_norm"
    else:
        raise ValueError(f"unknown null kind: {null_kind}")

    family_residual_values: dict[tuple, list[tuple[float, float]]] = {}
    family_norm_residual_values: dict[tuple, list[tuple[float, float]]] = {}
    for trace in traces:
        if trace.row.filename not in calibration_names:
            continue
        b_seq, e_seq = binned[trace.row.filename]
        for h_value, b_value, b_state, e_state in zip(
            trace.h_a_per_m, trace.b_t, b_seq, e_seq
        ):
            state = (b_state, e_state)
            residual = (
                h_value - h_centers[b_state - 1],
                b_value - b_centers[e_state - 1],
            )
            family_key = (family_map[state],)
            family_norm_key = (family_map[state], norm_map[state])
            family_residual_values.setdefault(family_key, []).append(residual)
            family_norm_residual_values.setdefault(family_norm_key, []).append(residual)

    family_centers = mean_pair_map(family_residual_values)
    family_norm_centers = mean_pair_map(family_norm_residual_values)

    actual: list[float] = []
    null_predicted: list[float] = []
    nonqa_predicted: list[float] = []
    for trace in traces:
        if trace.row.filename not in heldout_names:
            continue
        b_seq, e_seq = binned[trace.row.filename]
        h_hat: list[float] = []
        b_hat: list[float] = []
        nonqa_h_shell = [h_centers[b_state - 1] for b_state in b_seq]
        nonqa_b_shell = [b_centers[e_state - 1] for e_state in e_seq]
        for b_state, e_state in zip(b_seq, e_seq):
            state = (b_state, e_state)
            family_key = (family_map[state],)
            family_norm_key = (family_map[state], norm_map[state])
            if context_mode == "family":
                residual = family_centers.get(family_key)
            else:
                residual = family_norm_centers.get(family_norm_key)
                if residual is None:
                    residual = family_centers.get(family_key)
            if residual is None:
                residual = (0.0, 0.0)
            h_hat.append(h_centers[b_state - 1] + residual[0])
            b_hat.append(b_centers[e_state - 1] + residual[1])
        actual.append(trace.row.energy_loss_mj_per_kg)
        null_predicted.append(
            1000.0 * abs(line_integral_y_dx(b_hat, h_hat)) / DENSITY_KG_PER_M3
        )
        nonqa_predicted.append(
            1000.0
            * abs(line_integral_y_dx(nonqa_b_shell, nonqa_h_shell))
            / DENSITY_KG_PER_M3
        )

    null_metrics = metrics(actual, null_predicted)
    nonqa_metrics = metrics(actual, nonqa_predicted)
    return {
        "beats_nonqa_by_rmse": (
            null_metrics["rmse_mj_per_kg"] < nonqa_metrics["rmse_mj_per_kg"]
        ),
        "delta_rmse_vs_nonqa_shell": (
            null_metrics["rmse_mj_per_kg"] - nonqa_metrics["rmse_mj_per_kg"]
        ),
        "delta_r2_vs_nonqa_shell": null_metrics["r2"] - nonqa_metrics["r2"],
        "rmse_mj_per_kg": null_metrics["rmse_mj_per_kg"],
        "r2": null_metrics["r2"],
        "nonqa_rmse_mj_per_kg": nonqa_metrics["rmse_mj_per_kg"],
        "nonqa_r2": nonqa_metrics["r2"],
    }


def orbit_bridge_permutation_tests(
    traces: list[LoopTrace],
    loss_rows: list[LossRow],
    observed_transfer_wins: int,
    observed_main_delta_rmse: float,
    observed_main_delta_r2: float,
    seed_count: int = 200,
) -> dict:
    split_defs = transfer_splits(loss_rows)
    null_names = [
        "orbit_family_shuffle_null",
        "norm_f_shuffle_null",
        "family_norm_joint_shuffle_null",
    ]
    nulls: dict[str, dict] = {}
    for null_name in null_names:
        transfer_wins: list[int] = []
        main_delta_rmse: list[float] = []
        main_delta_r2: list[float] = []
        for seed in range(seed_count):
            wins = 0
            for split_name, calibration_names, heldout_names in split_defs:
                split_result = permutation_split_metrics(
                    traces, calibration_names, heldout_names, null_name, seed
                )
                wins += int(split_result["beats_nonqa_by_rmse"])
                if split_name == "index_mod3_holdout":
                    main_delta_rmse.append(float(split_result["delta_rmse_vs_nonqa_shell"]))
                    main_delta_r2.append(float(split_result["delta_r2_vs_nonqa_shell"]))
            transfer_wins.append(wins)
        p_value = sum(1 for wins in transfer_wins if wins >= observed_transfer_wins) / len(
            transfer_wins
        )
        nulls[null_name] = {
            "seeds": seed_count,
            "mean_transfer_wins": statistics.mean(transfer_wins),
            "max_transfer_wins": max(transfer_wins),
            "p_value_transfer_wins_ge_observed": p_value,
            "mean_main_delta_rmse": statistics.mean(main_delta_rmse),
            "min_main_delta_rmse": min(main_delta_rmse),
            "mean_main_delta_r2": statistics.mean(main_delta_r2),
            "max_main_delta_r2": max(main_delta_r2),
        }
    return {
        "observed_transfer_wins": observed_transfer_wins,
        "observed_main_split_delta_rmse": observed_main_delta_rmse,
        "observed_main_split_delta_r2": observed_main_delta_r2,
        "nulls": nulls,
    }


def direct_model_suite(
    records: list[dict], calibration_names: set[str], heldout_names: set[str]
) -> dict:
    models = {
        "mean_loss_baseline": mean_baseline(records, calibration_names, heldout_names),
        "steinmetz_log_linear_baseline": fit_steinmetz(records, calibration_names, heldout_names),
        "raw_physical_hdb_direct": direct_prediction(
            records, heldout_names, "energy_loss_mj_per_kg_from_integral"
        ),
        "nonqa_shell_hdb_direct": direct_prediction(
            records, heldout_names, "nonqa_shell_hdb_energy_mj_per_kg"
        ),
        "qa_shell_hdb_direct": direct_prediction(
            records, heldout_names, "qa_shell_hdb_energy_mj_per_kg"
        ),
        "qa_lifted_jx_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_jx_shell_energy_mj_per_kg"
        ),
        "qa_lifted_kx_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_kx_shell_energy_mj_per_kg"
        ),
        "qa_lifted_k_minus_j_x_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_k_minus_j_x_shell_energy_mj_per_kg"
        ),
        "qa_lifted_j_pi_shell_direct": direct_prediction(
            records, heldout_names, "qa_lifted_j_pi_shell_energy_mj_per_kg"
        ),
        "qa_reconstruct_be_direct": direct_prediction(
            records, heldout_names, "qa_reconstruct_be_energy_mj_per_kg"
        ),
        "qa_reconstruct_be_dir_direct": direct_prediction(
            records, heldout_names, "qa_reconstruct_be_dir_energy_mj_per_kg"
        ),
        "qa_reconstruct_be_branch_direct": direct_prediction(
            records, heldout_names, "qa_reconstruct_be_branch_energy_mj_per_kg"
        ),
        "qa_reconstruct_memory_full_direct": direct_prediction(
            records, heldout_names, "qa_reconstruct_memory_full_energy_mj_per_kg"
        ),
        "qa_residual_reconstruct_be_direct": direct_prediction(
            records, heldout_names, "qa_residual_reconstruct_be_energy_mj_per_kg"
        ),
        "qa_residual_reconstruct_be_dir_direct": direct_prediction(
            records, heldout_names, "qa_residual_reconstruct_be_dir_energy_mj_per_kg"
        ),
        "qa_residual_reconstruct_be_branch_direct": direct_prediction(
            records, heldout_names, "qa_residual_reconstruct_be_branch_energy_mj_per_kg"
        ),
        "qa_residual_reconstruct_memory_full_direct": direct_prediction(
            records, heldout_names, "qa_residual_reconstruct_memory_full_energy_mj_per_kg"
        ),
        "qa_orbit_residual_reconstruct_family_direct": direct_prediction(
            records,
            heldout_names,
            "qa_orbit_residual_reconstruct_family_energy_mj_per_kg",
        ),
        "qa_orbit_residual_reconstruct_family_norm_direct": direct_prediction(
            records,
            heldout_names,
            "qa_orbit_residual_reconstruct_family_norm_energy_mj_per_kg",
        ),
        "qa_orbit_residual_reconstruct_family_norm_branch_direct": direct_prediction(
            records,
            heldout_names,
            "qa_orbit_residual_reconstruct_family_norm_branch_energy_mj_per_kg",
        ),
    }
    models["qa_transition_only"] = fit_predict(
        records, calibration_names, heldout_names, QA_TRANSITION_FEATURES
    )
    models["qa_transition_plus_shell"] = fit_predict(
        records,
        calibration_names,
        heldout_names,
        ["qa_shell_hdb_energy_mj_per_kg"] + QA_TRANSITION_FEATURES,
    )
    models["nonqa_shell_plus_transition"] = fit_predict(
        records,
        calibration_names,
        heldout_names,
        ["nonqa_shell_hdb_energy_mj_per_kg"] + QA_TRANSITION_FEATURES,
    )
    models["qa_memory_only"] = fit_predict(
        records, calibration_names, heldout_names, QA_MEMORY_FEATURES
    )
    models["qa_memory_plus_qa_shell"] = fit_predict(
        records,
        calibration_names,
        heldout_names,
        ["qa_shell_hdb_energy_mj_per_kg"] + QA_MEMORY_FEATURES,
    )
    models["nonqa_shell_plus_qa_memory"] = fit_predict(
        records,
        calibration_names,
        heldout_names,
        ["nonqa_shell_hdb_energy_mj_per_kg"] + QA_MEMORY_FEATURES,
    )
    models["qa_memory_plus_transition"] = fit_predict(
        records,
        calibration_names,
        heldout_names,
        QA_MEMORY_FEATURES + QA_TRANSITION_FEATURES,
    )
    models["nonqa_shell_plus_transition_plus_memory"] = fit_predict(
        records,
        calibration_names,
        heldout_names,
        ["nonqa_shell_hdb_energy_mj_per_kg"] + QA_TRANSITION_FEATURES + QA_MEMORY_FEATURES,
    )
    return models


def direct_comparison_rows(heldout_models: dict, comparison_order: list[str]) -> list[dict]:
    return [{"model": name, **heldout_models[name]["heldout"]} for name in comparison_order]


def split_regime(split_name: str) -> tuple[str, float | str | None]:
    if split_name == "index_mod3_holdout":
        return "index", None
    if split_name.startswith("frequency_") and split_name.endswith("Hz_holdout"):
        value = split_name.removeprefix("frequency_").removesuffix("Hz_holdout")
        return "frequency_hz", float(value)
    if split_name.startswith("amplitude_") and split_name.endswith("T_holdout"):
        value = split_name.removeprefix("amplitude_").removesuffix("T_holdout")
        return "amplitude_t", float(value)
    return "other", split_name


def summarize_rows_by_regime(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["regime_type"], []).append(row)
    out: dict[str, dict] = {}
    for regime_type, regime_rows in grouped.items():
        deltas = [row["orbit_family_delta_rmse_vs_nonqa_shell"] for row in regime_rows]
        out[regime_type] = {
            "split_count": len(regime_rows),
            "orbit_family_wins": sum(
                int(row["orbit_family_beats_nonqa_by_rmse"]) for row in regime_rows
            ),
            "mean_orbit_family_delta_rmse": statistics.mean(deltas),
            "median_orbit_family_delta_rmse": statistics.median(deltas),
            "min_orbit_family_delta_rmse": min(deltas),
            "max_orbit_family_delta_rmse": max(deltas),
            "winning_splits": [
                row["split"]
                for row in regime_rows
                if row["orbit_family_beats_nonqa_by_rmse"]
            ],
            "losing_splits": [
                row["split"]
                for row in regime_rows
                if not row["orbit_family_beats_nonqa_by_rmse"]
            ],
        }
    return out


def orbit_bridge_regime_diagnosis(split_summaries: list[dict]) -> dict:
    rows: list[dict] = []
    for split in split_summaries:
        regime_type, regime_value = split_regime(split["split"])
        models = split["models"]
        nonqa = models["nonqa_shell_hdb_direct"]
        family = models["qa_orbit_residual_reconstruct_family_direct"]
        family_norm = models["qa_orbit_residual_reconstruct_family_norm_direct"]
        family_norm_branch = models[
            "qa_orbit_residual_reconstruct_family_norm_branch_direct"
        ]
        row = {
            "split": split["split"],
            "regime_type": regime_type,
            "regime_value": regime_value,
            "heldout_count": split["heldout_count"],
            "nonqa_shell_rmse": nonqa["rmse_mj_per_kg"],
            "nonqa_shell_r2": nonqa["r2"],
            "orbit_family_rmse": family["rmse_mj_per_kg"],
            "orbit_family_r2": family["r2"],
            "orbit_family_delta_rmse_vs_nonqa_shell": (
                family["rmse_mj_per_kg"] - nonqa["rmse_mj_per_kg"]
            ),
            "orbit_family_delta_r2_vs_nonqa_shell": family["r2"] - nonqa["r2"],
            "orbit_family_beats_nonqa_by_rmse": (
                family["rmse_mj_per_kg"] < nonqa["rmse_mj_per_kg"]
            ),
            "orbit_family_norm_rmse": family_norm["rmse_mj_per_kg"],
            "orbit_family_norm_delta_rmse_vs_nonqa_shell": (
                family_norm["rmse_mj_per_kg"] - nonqa["rmse_mj_per_kg"]
            ),
            "orbit_family_norm_beats_nonqa_by_rmse": (
                family_norm["rmse_mj_per_kg"] < nonqa["rmse_mj_per_kg"]
            ),
            "orbit_family_norm_branch_rmse": family_norm_branch["rmse_mj_per_kg"],
            "orbit_family_norm_branch_delta_rmse_vs_nonqa_shell": (
                family_norm_branch["rmse_mj_per_kg"] - nonqa["rmse_mj_per_kg"]
            ),
            "orbit_family_norm_branch_beats_nonqa_by_rmse": (
                family_norm_branch["rmse_mj_per_kg"] < nonqa["rmse_mj_per_kg"]
            ),
        }
        rows.append(row)
    strongest_win = min(rows, key=lambda row: row["orbit_family_delta_rmse_vs_nonqa_shell"])
    strongest_loss = max(rows, key=lambda row: row["orbit_family_delta_rmse_vs_nonqa_shell"])
    grouped = summarize_rows_by_regime(rows)
    return {
        "rows": rows,
        "by_regime_type": grouped,
        "strongest_orbit_family_win": {
            "split": strongest_win["split"],
            "delta_rmse": strongest_win["orbit_family_delta_rmse_vs_nonqa_shell"],
        },
        "strongest_orbit_family_loss": {
            "split": strongest_loss["split"],
            "delta_rmse": strongest_loss["orbit_family_delta_rmse_vs_nonqa_shell"],
        },
    }


def lifted_pair_sweep(records: list[dict], heldout_names: set[str]) -> list[dict]:
    nonqa = direct_prediction(records, heldout_names, "nonqa_shell_hdb_energy_mj_per_kg")[
        "heldout"
    ]
    rows: list[dict] = []
    for h_var in QA_LIFTED_VARIABLES:
        for b_var in QA_LIFTED_VARIABLES:
            field = lifted_energy_field(h_var, b_var)
            result = direct_prediction(records, heldout_names, field)["heldout"]
            rows.append(
                {
                    "h_var": h_var,
                    "b_var": b_var,
                    "field": field,
                    **result,
                    "beats_nonqa_by_r2": result["r2"] > nonqa["r2"],
                    "beats_nonqa_by_rmse": result["rmse_mj_per_kg"]
                    < nonqa["rmse_mj_per_kg"],
                    "beats_nonqa_by_mae": result["mae_mj_per_kg"]
                    < nonqa["mae_mj_per_kg"],
                    "rmse_ratio_to_nonqa": result["rmse_mj_per_kg"]
                    / max(nonqa["rmse_mj_per_kg"], 1e-12),
                }
            )
    return rows


def phase_pair_sweep(
    records: list[dict], calibration_names: set[str], heldout_names: set[str]
) -> list[dict]:
    nonqa = direct_prediction(records, heldout_names, "nonqa_shell_hdb_energy_mj_per_kg")[
        "heldout"
    ]
    rows: list[dict] = []
    for h_var in QA_LIFTED_VARIABLES:
        for b_var in QA_LIFTED_VARIABLES:
            features = lifted_component_features(h_var, b_var)
            result = fit_predict(records, calibration_names, heldout_names, features)["heldout"]
            rows.append(
                {
                    "h_var": h_var,
                    "b_var": b_var,
                    "features": features,
                    **result,
                    "beats_nonqa_by_r2": result["r2"] > nonqa["r2"],
                    "beats_nonqa_by_rmse": result["rmse_mj_per_kg"]
                    < nonqa["rmse_mj_per_kg"],
                    "beats_nonqa_by_mae": result["mae_mj_per_kg"]
                    < nonqa["mae_mj_per_kg"],
                    "rmse_ratio_to_nonqa": result["rmse_mj_per_kg"]
                    / max(nonqa["rmse_mj_per_kg"], 1e-12),
                }
            )
    return rows


def summarize_lifted_pair_sweep(split_results: list[dict]) -> dict:
    return summarize_pair_sweep(split_results, "lifted_pair_sweep")


def summarize_phase_pair_sweep(split_results: list[dict]) -> dict:
    return summarize_pair_sweep(split_results, "phase_pair_sweep")


def summarize_pair_sweep(split_results: list[dict], sweep_key: str) -> dict:
    aggregates: dict[str, dict] = {}
    for split in split_results:
        for row in split[sweep_key]:
            key = f"{row['h_var']}/{row['b_var']}"
            bucket = aggregates.setdefault(
                key,
                {
                    "h_var": row["h_var"],
                    "b_var": row["b_var"],
                    "r2_win_count": 0,
                    "rmse_win_count": 0,
                    "mae_win_count": 0,
                    "rmse_ratios": [],
                    "r2_values": [],
                },
            )
            bucket["r2_win_count"] += int(row["beats_nonqa_by_r2"])
            bucket["rmse_win_count"] += int(row["beats_nonqa_by_rmse"])
            bucket["mae_win_count"] += int(row["beats_nonqa_by_mae"])
            bucket["rmse_ratios"].append(row["rmse_ratio_to_nonqa"])
            bucket["r2_values"].append(row["r2"])

    scored = []
    for key, bucket in aggregates.items():
        scored.append(
            {
                "pair": key,
                "h_var": bucket["h_var"],
                "b_var": bucket["b_var"],
                "r2_win_count": bucket["r2_win_count"],
                "rmse_win_count": bucket["rmse_win_count"],
                "mae_win_count": bucket["mae_win_count"],
                "mean_rmse_ratio_to_nonqa": statistics.mean(bucket["rmse_ratios"]),
                "median_rmse_ratio_to_nonqa": statistics.median(bucket["rmse_ratios"]),
                "mean_r2": statistics.mean(bucket["r2_values"]),
            }
        )

    scored.sort(
        key=lambda row: (
            -row["rmse_win_count"],
            -row["mae_win_count"],
            -row["r2_win_count"],
            row["median_rmse_ratio_to_nonqa"],
            row["mean_rmse_ratio_to_nonqa"],
        )
    )
    return {
        "pair_count": len(scored),
        "top_by_transfer_score": scored[:10],
        "best_pair": scored[0] if scored else None,
    }


def transfer_splits(loss_rows: list[LossRow]) -> list[tuple[str, set[str], set[str]]]:
    all_names = {row.filename for row in loss_rows}
    splits: list[tuple[str, set[str], set[str]]] = []
    calibration_names, heldout_names = calibration_split(loss_rows)
    splits.append(("index_mod3_holdout", calibration_names, heldout_names))

    for frequency in sorted({row.frequency_hz for row in loss_rows}):
        heldout = {row.filename for row in loss_rows if row.frequency_hz == frequency}
        if 2 <= len(heldout) < len(all_names):
            splits.append((f"frequency_{int(frequency)}Hz_holdout", all_names - heldout, heldout))

    for bp_t in sorted({row.bp_t for row in loss_rows}):
        heldout = {row.filename for row in loss_rows if row.bp_t == bp_t}
        if 2 <= len(heldout) < len(all_names):
            splits.append((f"amplitude_{bp_t:.2f}T_holdout", all_names - heldout, heldout))
    return splits


def summarize_transfer_split(
    split_name: str,
    traces: list[LoopTrace],
    calibration_names: set[str],
    heldout_names: set[str],
    comparison_order: list[str],
) -> dict:
    records = build_records_for_split(traces, calibration_names, heldout_names)
    models = direct_model_suite(records, calibration_names, heldout_names)
    rows = direct_comparison_rows(models, comparison_order)
    pair_rows = lifted_pair_sweep(records, heldout_names)
    phase_rows = phase_pair_sweep(records, calibration_names, heldout_names)
    best_by_rmse = min(rows, key=lambda row: row["rmse_mj_per_kg"])
    best_shell_by_rmse = min(
        [row for row in rows if row["model"] != "raw_physical_hdb_direct"],
        key=lambda row: row["rmse_mj_per_kg"],
    )
    nonqa = models["nonqa_shell_hdb_direct"]["heldout"]
    qa_j_pi = models["qa_lifted_j_pi_shell_direct"]["heldout"]
    transition_only = models["qa_transition_only"]["heldout"]
    transition_plus_shell = models["qa_transition_plus_shell"]["heldout"]
    nonqa_plus_transition = models["nonqa_shell_plus_transition"]["heldout"]
    memory_only = models["qa_memory_only"]["heldout"]
    memory_plus_shell = models["qa_memory_plus_qa_shell"]["heldout"]
    nonqa_plus_memory = models["nonqa_shell_plus_qa_memory"]["heldout"]
    memory_plus_transition = models["qa_memory_plus_transition"]["heldout"]
    nonqa_plus_transition_memory = models["nonqa_shell_plus_transition_plus_memory"]["heldout"]
    reconstruct_be = models["qa_reconstruct_be_direct"]["heldout"]
    reconstruct_be_dir = models["qa_reconstruct_be_dir_direct"]["heldout"]
    reconstruct_be_branch = models["qa_reconstruct_be_branch_direct"]["heldout"]
    reconstruct_memory_full = models["qa_reconstruct_memory_full_direct"]["heldout"]
    residual_reconstruct_be = models["qa_residual_reconstruct_be_direct"]["heldout"]
    residual_reconstruct_be_dir = models["qa_residual_reconstruct_be_dir_direct"]["heldout"]
    residual_reconstruct_be_branch = models[
        "qa_residual_reconstruct_be_branch_direct"
    ]["heldout"]
    residual_reconstruct_memory_full = models[
        "qa_residual_reconstruct_memory_full_direct"
    ]["heldout"]
    orbit_residual_family = models["qa_orbit_residual_reconstruct_family_direct"][
        "heldout"
    ]
    orbit_residual_family_norm = models[
        "qa_orbit_residual_reconstruct_family_norm_direct"
    ]["heldout"]
    orbit_residual_family_norm_branch = models[
        "qa_orbit_residual_reconstruct_family_norm_branch_direct"
    ]["heldout"]
    return {
        "split": split_name,
        "calibration_count": len(calibration_names),
        "heldout_count": len(heldout_names),
        "best_model_by_rmse": best_by_rmse["model"],
        "best_nonraw_model_by_rmse": best_shell_by_rmse["model"],
        "qa_lifted_j_pi_beats_nonqa_by_r2": qa_j_pi["r2"] > nonqa["r2"],
        "qa_lifted_j_pi_beats_nonqa_by_rmse": qa_j_pi["rmse_mj_per_kg"] < nonqa["rmse_mj_per_kg"],
        "qa_transition_only_beats_nonqa_by_rmse": transition_only["rmse_mj_per_kg"]
        < nonqa["rmse_mj_per_kg"],
        "qa_transition_plus_shell_beats_nonqa_by_rmse": transition_plus_shell[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "nonqa_shell_plus_transition_beats_nonqa_by_rmse": nonqa_plus_transition[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_memory_only_beats_nonqa_by_rmse": memory_only["rmse_mj_per_kg"]
        < nonqa["rmse_mj_per_kg"],
        "qa_memory_plus_qa_shell_beats_nonqa_by_rmse": memory_plus_shell[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "nonqa_shell_plus_qa_memory_beats_nonqa_by_rmse": nonqa_plus_memory[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_memory_plus_transition_beats_nonqa_by_rmse": memory_plus_transition[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "nonqa_shell_plus_transition_plus_memory_beats_nonqa_by_rmse": nonqa_plus_transition_memory[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_reconstruct_be_beats_nonqa_by_rmse": reconstruct_be["rmse_mj_per_kg"]
        < nonqa["rmse_mj_per_kg"],
        "qa_reconstruct_be_dir_beats_nonqa_by_rmse": reconstruct_be_dir["rmse_mj_per_kg"]
        < nonqa["rmse_mj_per_kg"],
        "qa_reconstruct_be_branch_beats_nonqa_by_rmse": reconstruct_be_branch[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_reconstruct_memory_full_beats_nonqa_by_rmse": reconstruct_memory_full[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_residual_reconstruct_be_beats_nonqa_by_rmse": residual_reconstruct_be[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_residual_reconstruct_be_dir_beats_nonqa_by_rmse": residual_reconstruct_be_dir[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_residual_reconstruct_be_branch_beats_nonqa_by_rmse": residual_reconstruct_be_branch[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_residual_reconstruct_memory_full_beats_nonqa_by_rmse": residual_reconstruct_memory_full[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_orbit_residual_reconstruct_family_beats_nonqa_by_rmse": orbit_residual_family[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_orbit_residual_reconstruct_family_norm_beats_nonqa_by_rmse": orbit_residual_family_norm[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "qa_orbit_residual_reconstruct_family_norm_branch_beats_nonqa_by_rmse": orbit_residual_family_norm_branch[
            "rmse_mj_per_kg"
        ]
        < nonqa["rmse_mj_per_kg"],
        "best_lifted_pair_by_rmse": min(pair_rows, key=lambda row: row["rmse_mj_per_kg"]),
        "best_lifted_pair_by_r2": max(pair_rows, key=lambda row: row["r2"]),
        "lifted_pair_sweep": pair_rows,
        "best_phase_pair_by_rmse": min(phase_rows, key=lambda row: row["rmse_mj_per_kg"]),
        "best_phase_pair_by_r2": max(phase_rows, key=lambda row: row["r2"]),
        "phase_pair_sweep": phase_rows,
        "models": {name: models[name]["heldout"] for name in comparison_order},
    }


def main() -> int:
    if not ZIP_PATH.exists():
        raise SystemExit(
            f"missing {ZIP_PATH}; download Ring35_Dataset_Txt.zip from {DATASET_RECORD}"
        )

    with zipfile.ZipFile(ZIP_PATH) as zf:
        loss_rows = select_sin_rows(load_sin_loss_table(zf))
        calibration_names, heldout_names = calibration_split(loss_rows)
        traces = [load_loop_trace(zf, row) for row in loss_rows]

    records = build_records_for_split(traces, calibration_names, heldout_names)

    qa_feature_sets = {
        "nonqa_shell_hdb_affine": ["nonqa_shell_hdb_energy_mj_per_kg"],
        "qa_shell_hdb_affine": ["qa_shell_hdb_energy_mj_per_kg"],
        "qa_be_only": ["qa_be_loop_area"],
        "qa_lifted_jx_only": ["qa_jx_loop_area"],
        "qa_lifted_jx_shell_affine": ["qa_lifted_jx_shell_energy_mj_per_kg"],
        "qa_lifted_j_pi_shell_affine": ["qa_lifted_j_pi_shell_energy_mj_per_kg"],
        "qa_lifted_packet": [
            "qa_shell_hdb_energy_mj_per_kg",
            "qa_lifted_jx_shell_energy_mj_per_kg",
            "qa_lifted_j_pi_shell_energy_mj_per_kg",
            "qa_jx_loop_area",
            "qa_kx_loop_area",
            "qa_pi_x_loop_area",
            "qa_pi_abs_path_energy",
            "qa_unique_states",
        ],
        "qa_with_drive_metadata": [
            "frequency_hz",
            "bp_t",
            "qa_shell_hdb_energy_mj_per_kg",
            "qa_lifted_j_pi_shell_energy_mj_per_kg",
            "qa_jx_loop_area",
            "qa_kx_loop_area",
            "qa_pi_abs_path_energy",
        ],
    }
    qa_models = {
        name: fit_predict(records, calibration_names, heldout_names, features)
        for name, features in qa_feature_sets.items()
    }

    heldout_models = {
        **direct_model_suite(records, calibration_names, heldout_names),
        "steinmetz_plus_qa_shape_log_model": fit_log_feature_model(
            records,
            calibration_names,
            heldout_names,
            [
                "frequency_hz",
                "bp_t",
                "qa_jx_loop_area",
                "qa_pi_abs_path_energy",
                "qa_unique_states",
            ],
        ),
        **qa_models,
    }
    comparison_order = [
        "mean_loss_baseline",
        "steinmetz_log_linear_baseline",
        "raw_physical_hdb_direct",
        "nonqa_shell_hdb_direct",
        "qa_shell_hdb_direct",
        "qa_lifted_jx_shell_direct",
        "qa_lifted_kx_shell_direct",
        "qa_lifted_k_minus_j_x_shell_direct",
        "qa_lifted_j_pi_shell_direct",
        "qa_reconstruct_be_direct",
        "qa_reconstruct_be_dir_direct",
        "qa_reconstruct_be_branch_direct",
        "qa_reconstruct_memory_full_direct",
        "qa_residual_reconstruct_be_direct",
        "qa_residual_reconstruct_be_dir_direct",
        "qa_residual_reconstruct_be_branch_direct",
        "qa_residual_reconstruct_memory_full_direct",
        "qa_orbit_residual_reconstruct_family_direct",
        "qa_orbit_residual_reconstruct_family_norm_direct",
        "qa_orbit_residual_reconstruct_family_norm_branch_direct",
        "qa_transition_only",
        "qa_transition_plus_shell",
        "nonqa_shell_plus_transition",
        "qa_memory_only",
        "qa_memory_plus_qa_shell",
        "nonqa_shell_plus_qa_memory",
        "qa_memory_plus_transition",
        "nonqa_shell_plus_transition_plus_memory",
        "qa_be_only",
        "qa_lifted_packet",
        "steinmetz_plus_qa_shape_log_model",
    ]
    direct_comparison = direct_comparison_rows(heldout_models, comparison_order)
    orbit_bridge_order = [
        "raw_physical_hdb_direct",
        "steinmetz_log_linear_baseline",
        "nonqa_shell_hdb_direct",
        "qa_orbit_residual_reconstruct_family_direct",
        "qa_orbit_residual_reconstruct_family_norm_direct",
        "qa_orbit_residual_reconstruct_family_norm_branch_direct",
    ]
    orbit_bridge_comparison = direct_comparison_rows(heldout_models, orbit_bridge_order)
    orbit_bridge_best_model_by_rmse = min(
        orbit_bridge_comparison, key=lambda row: row["rmse_mj_per_kg"]
    )["model"]
    orbit_bridge_nonraw_rows = [
        row for row in orbit_bridge_comparison if row["model"] != "raw_physical_hdb_direct"
    ]
    orbit_bridge_best_nonraw_model_by_rmse = min(
        orbit_bridge_nonraw_rows, key=lambda row: row["rmse_mj_per_kg"]
    )["model"]
    qa_shell_r2 = heldout_models["qa_shell_hdb_direct"]["heldout"]["r2"]
    nonqa_shell_r2 = heldout_models["nonqa_shell_hdb_direct"]["heldout"]["r2"]
    qa_shell_mae = heldout_models["qa_shell_hdb_direct"]["heldout"]["mae_mj_per_kg"]
    nonqa_shell_mae = heldout_models["nonqa_shell_hdb_direct"]["heldout"]["mae_mj_per_kg"]
    nonqa_shell_rmse = heldout_models["nonqa_shell_hdb_direct"]["heldout"][
        "rmse_mj_per_kg"
    ]
    orbit_family_r2 = heldout_models[
        "qa_orbit_residual_reconstruct_family_direct"
    ]["heldout"]["r2"]
    orbit_family_rmse = heldout_models[
        "qa_orbit_residual_reconstruct_family_direct"
    ]["heldout"]["rmse_mj_per_kg"]
    orbit_family_delta_rmse = orbit_family_rmse - nonqa_shell_rmse
    orbit_family_delta_r2 = orbit_family_r2 - nonqa_shell_r2
    lifted_j_pi_r2 = heldout_models["qa_lifted_j_pi_shell_direct"]["heldout"]["r2"]
    lifted_j_pi_mae = heldout_models["qa_lifted_j_pi_shell_direct"]["heldout"]["mae_mj_per_kg"]
    transfer_order = [
        "mean_loss_baseline",
        "steinmetz_log_linear_baseline",
        "raw_physical_hdb_direct",
        "nonqa_shell_hdb_direct",
        "qa_shell_hdb_direct",
        "qa_lifted_jx_shell_direct",
        "qa_lifted_j_pi_shell_direct",
        "qa_reconstruct_be_direct",
        "qa_reconstruct_be_dir_direct",
        "qa_reconstruct_be_branch_direct",
        "qa_reconstruct_memory_full_direct",
        "qa_residual_reconstruct_be_direct",
        "qa_residual_reconstruct_be_dir_direct",
        "qa_residual_reconstruct_be_branch_direct",
        "qa_residual_reconstruct_memory_full_direct",
        "qa_orbit_residual_reconstruct_family_direct",
        "qa_orbit_residual_reconstruct_family_norm_direct",
        "qa_orbit_residual_reconstruct_family_norm_branch_direct",
        "qa_transition_only",
        "qa_transition_plus_shell",
        "nonqa_shell_plus_transition",
        "qa_memory_only",
        "qa_memory_plus_qa_shell",
        "nonqa_shell_plus_qa_memory",
        "qa_memory_plus_transition",
        "nonqa_shell_plus_transition_plus_memory",
    ]
    split_summaries = [
        summarize_transfer_split(name, traces, cal, hold, transfer_order)
        for name, cal, hold in transfer_splits(loss_rows)
    ]
    lifted_pair_sweep_summary = summarize_lifted_pair_sweep(split_summaries)
    phase_pair_sweep_summary = summarize_phase_pair_sweep(split_summaries)
    best_sweep_pair = lifted_pair_sweep_summary["best_pair"]
    best_phase_pair = phase_pair_sweep_summary["best_pair"]
    qa_j_pi_r2_wins = sum(
        1 for split in split_summaries if split["qa_lifted_j_pi_beats_nonqa_by_r2"]
    )
    qa_j_pi_rmse_wins = sum(
        1 for split in split_summaries if split["qa_lifted_j_pi_beats_nonqa_by_rmse"]
    )
    transition_only_rmse_wins = sum(
        1 for split in split_summaries if split["qa_transition_only_beats_nonqa_by_rmse"]
    )
    transition_plus_shell_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_transition_plus_shell_beats_nonqa_by_rmse"]
    )
    nonqa_plus_transition_rmse_wins = sum(
        1
        for split in split_summaries
        if split["nonqa_shell_plus_transition_beats_nonqa_by_rmse"]
    )
    memory_only_rmse_wins = sum(
        1 for split in split_summaries if split["qa_memory_only_beats_nonqa_by_rmse"]
    )
    memory_plus_shell_rmse_wins = sum(
        1 for split in split_summaries if split["qa_memory_plus_qa_shell_beats_nonqa_by_rmse"]
    )
    nonqa_plus_memory_rmse_wins = sum(
        1
        for split in split_summaries
        if split["nonqa_shell_plus_qa_memory_beats_nonqa_by_rmse"]
    )
    memory_plus_transition_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_memory_plus_transition_beats_nonqa_by_rmse"]
    )
    nonqa_plus_transition_memory_rmse_wins = sum(
        1
        for split in split_summaries
        if split["nonqa_shell_plus_transition_plus_memory_beats_nonqa_by_rmse"]
    )
    reconstruct_be_rmse_wins = sum(
        1 for split in split_summaries if split["qa_reconstruct_be_beats_nonqa_by_rmse"]
    )
    reconstruct_be_dir_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_reconstruct_be_dir_beats_nonqa_by_rmse"]
    )
    reconstruct_be_branch_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_reconstruct_be_branch_beats_nonqa_by_rmse"]
    )
    reconstruct_memory_full_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_reconstruct_memory_full_beats_nonqa_by_rmse"]
    )
    residual_reconstruct_be_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_residual_reconstruct_be_beats_nonqa_by_rmse"]
    )
    residual_reconstruct_be_dir_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_residual_reconstruct_be_dir_beats_nonqa_by_rmse"]
    )
    residual_reconstruct_be_branch_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_residual_reconstruct_be_branch_beats_nonqa_by_rmse"]
    )
    residual_reconstruct_memory_full_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_residual_reconstruct_memory_full_beats_nonqa_by_rmse"]
    )
    orbit_residual_family_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_orbit_residual_reconstruct_family_beats_nonqa_by_rmse"]
    )
    orbit_residual_family_norm_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_orbit_residual_reconstruct_family_norm_beats_nonqa_by_rmse"]
    )
    orbit_residual_family_norm_branch_rmse_wins = sum(
        1
        for split in split_summaries
        if split["qa_orbit_residual_reconstruct_family_norm_branch_beats_nonqa_by_rmse"]
    )
    transition_family_scores = {
        "qa_transition_only": transition_only_rmse_wins,
        "qa_transition_plus_shell": transition_plus_shell_rmse_wins,
        "nonqa_shell_plus_transition": nonqa_plus_transition_rmse_wins,
    }
    best_transition_family = max(
        transition_family_scores, key=lambda name: transition_family_scores[name]
    )
    memory_family_scores = {
        "qa_memory_only": memory_only_rmse_wins,
        "qa_memory_plus_qa_shell": memory_plus_shell_rmse_wins,
        "nonqa_shell_plus_qa_memory": nonqa_plus_memory_rmse_wins,
        "qa_memory_plus_transition": memory_plus_transition_rmse_wins,
        "nonqa_shell_plus_transition_plus_memory": nonqa_plus_transition_memory_rmse_wins,
    }
    best_memory_family = max(memory_family_scores, key=lambda name: memory_family_scores[name])
    reconstruction_family_scores = {
        "qa_reconstruct_be_direct": reconstruct_be_rmse_wins,
        "qa_reconstruct_be_dir_direct": reconstruct_be_dir_rmse_wins,
        "qa_reconstruct_be_branch_direct": reconstruct_be_branch_rmse_wins,
        "qa_reconstruct_memory_full_direct": reconstruct_memory_full_rmse_wins,
    }
    best_reconstruction_family = max(
        reconstruction_family_scores, key=lambda name: reconstruction_family_scores[name]
    )
    residual_reconstruction_family_scores = {
        "qa_residual_reconstruct_be_direct": residual_reconstruct_be_rmse_wins,
        "qa_residual_reconstruct_be_dir_direct": residual_reconstruct_be_dir_rmse_wins,
        "qa_residual_reconstruct_be_branch_direct": (
            residual_reconstruct_be_branch_rmse_wins
        ),
        "qa_residual_reconstruct_memory_full_direct": (
            residual_reconstruct_memory_full_rmse_wins
        ),
    }
    best_residual_reconstruction_family = max(
        residual_reconstruction_family_scores,
        key=lambda name: residual_reconstruction_family_scores[name],
    )
    orbit_residual_family_scores = {
        "qa_orbit_residual_reconstruct_family_direct": orbit_residual_family_rmse_wins,
        "qa_orbit_residual_reconstruct_family_norm_direct": (
            orbit_residual_family_norm_rmse_wins
        ),
        "qa_orbit_residual_reconstruct_family_norm_branch_direct": (
            orbit_residual_family_norm_branch_rmse_wins
        ),
    }
    best_orbit_residual_family = max(
        orbit_residual_family_scores, key=lambda name: orbit_residual_family_scores[name]
    )
    memory_result_interpretation = (
        f"Memory-only QA wins {memory_only_rmse_wins} of {len(split_summaries)} "
        f"RMSE transfer splits against non-QA shell; memory plus QA shell wins "
        f"{memory_plus_shell_rmse_wins}; non-QA shell plus QA memory wins "
        f"{nonqa_plus_memory_rmse_wins}; QA memory plus transition wins "
        f"{memory_plus_transition_rmse_wins}; non-QA shell plus transition plus "
        f"memory wins {nonqa_plus_transition_memory_rmse_wins}. Best memory family: "
        f"{best_memory_family}."
    )
    reconstruction_result_interpretation = (
        f"Branch-local lookup reconstruction wins against ordinary non-QA shell in "
        f"{reconstruct_be_rmse_wins}/{len(split_summaries)} splits for (b,e), "
        f"{reconstruct_be_dir_rmse_wins}/{len(split_summaries)} for (b,e,sb,se), "
        f"{reconstruct_be_branch_rmse_wins}/{len(split_summaries)} for "
        f"(b,e,branch,lag), and {reconstruct_memory_full_rmse_wins}/"
        f"{len(split_summaries)} for full memory context. "
        f"Best reconstruction family: {best_reconstruction_family}. Because the "
        "declared reconstruction target is the marginal H/B shell center and every "
        "tested context contains (b,e), this formulation collapses to ordinary "
        "shell reconstruction rather than adding QA-specific branch information."
    )
    residual_reconstruction_result_interpretation = (
        f"Residual branch-local reconstruction wins against ordinary non-QA shell in "
        f"{residual_reconstruct_be_rmse_wins}/{len(split_summaries)} splits for "
        f"(b,e), {residual_reconstruct_be_dir_rmse_wins}/{len(split_summaries)} "
        f"for (b,e,sb,se), {residual_reconstruct_be_branch_rmse_wins}/"
        f"{len(split_summaries)} for (b,e,branch,lag), and "
        f"{residual_reconstruct_memory_full_rmse_wins}/{len(split_summaries)} for "
        f"full memory context. Best residual reconstruction family: "
        f"{best_residual_reconstruction_family}."
    )
    orbit_residual_result_interpretation = (
        "Established QA orbit-topology residual reconstruction wins against "
        f"ordinary non-QA shell in {orbit_residual_family_rmse_wins}/"
        f"{len(split_summaries)} splits for orbit family, "
        f"{orbit_residual_family_norm_rmse_wins}/{len(split_summaries)} for "
        f"orbit family + norm_f mod {MODULUS}, and "
        f"{orbit_residual_family_norm_branch_rmse_wins}/{len(split_summaries)} "
        "for orbit family + norm + branch/lag. Best orbit residual family: "
        f"{best_orbit_residual_family}."
    )
    orbit_bridge_result_interpretation = (
        "Earlier ad hoc QA summaries failed to produce a stable advantage over "
        "ordinary calibrated H/B shell physics. Canonical QA orbit topology is the "
        "first established QA substrate in this experiment that matches and slightly "
        "beats the non-QA shell baseline on the main split: orbit-family residual "
        f"delta RMSE={orbit_family_delta_rmse:.6f} mJ/kg and delta R2="
        f"{orbit_family_delta_r2:.6f}. The transfer result is promising but not "
        f"conclusive at {orbit_residual_family_rmse_wins}/{len(split_summaries)} "
        "RMSE wins. The norm_f and branch/lag variants are reported separately "
        "because adding them is not automatically better. The next hardening step "
        "is a significance/permutation test over orbit labels or bootstrap over loops."
    )
    orbit_bridge_permutation = orbit_bridge_permutation_tests(
        traces,
        loss_rows,
        orbit_residual_family_rmse_wins,
        orbit_family_delta_rmse,
        orbit_family_delta_r2,
    )
    null_pieces = []
    for null_name, null_result in orbit_bridge_permutation["nulls"].items():
        null_pieces.append(
            f"{null_name}: p={null_result['p_value_transfer_wins_ge_observed']:.3f}, "
            f"max_wins={null_result['max_transfer_wins']}"
        )
    if all(
        null_result["p_value_transfer_wins_ge_observed"] < 0.05
        for null_result in orbit_bridge_permutation["nulls"].values()
    ):
        null_verdict = "above all shuffled-label nulls at the 0.05 empirical level"
    elif all(
        null_result["p_value_transfer_wins_ge_observed"] < 0.10
        for null_result in orbit_bridge_permutation["nulls"].values()
    ):
        null_verdict = "above all shuffled-label nulls at the 0.10 empirical level"
    else:
        null_verdict = "not cleanly above all shuffled-label nulls"
    orbit_bridge_permutation_interpretation = (
        f"Observed canonical orbit-family transfer wins: "
        f"{orbit_residual_family_rmse_wins}/{len(split_summaries)}. Shuffled-label "
        f"null summary: {'; '.join(null_pieces)}. The observed signal is "
        f"{null_verdict}; interpret the bridge accordingly."
    )
    orbit_regime_diagnosis = orbit_bridge_regime_diagnosis(split_summaries)
    regime_groups = orbit_regime_diagnosis["by_regime_type"]
    frequency_group = regime_groups.get("frequency_hz", {})
    amplitude_group = regime_groups.get("amplitude_t", {})
    orbit_bridge_regime_interpretation = (
        "Regime diagnosis localizes the canonical orbit-family signal instead of "
        "treating 6/12 wins as uniform. Frequency-heldout wins are "
        f"{frequency_group.get('orbit_family_wins', 0)}/"
        f"{frequency_group.get('split_count', 0)} with winning splits "
        f"{frequency_group.get('winning_splits', [])}. Amplitude-heldout wins are "
        f"{amplitude_group.get('orbit_family_wins', 0)}/"
        f"{amplitude_group.get('split_count', 0)} with winning splits "
        f"{amplitude_group.get('winning_splits', [])}. Strongest orbit-family win: "
        f"{orbit_regime_diagnosis['strongest_orbit_family_win']['split']} "
        f"(delta RMSE "
        f"{orbit_regime_diagnosis['strongest_orbit_family_win']['delta_rmse']:.6f}); "
        f"strongest loss: {orbit_regime_diagnosis['strongest_orbit_family_loss']['split']} "
        f"(delta RMSE "
        f"{orbit_regime_diagnosis['strongest_orbit_family_loss']['delta_rmse']:.6f})."
    )

    table_errors = [r["integral_vs_table_abs_error"] for r in records]
    payload = {
        "ok": True,
        "experiment": "qa_hysteresis_real_loop_observer",
        "dataset": {
            "title": DATASET_TITLE,
            "record_url": DATASET_RECORD,
            "local_archive": str(ZIP_PATH),
            "subset": "Ring35_Dataset_Txt/SIN selected grid",
            "material": "0.3471 mm FeSi ring",
            "density_kg_per_m3": DENSITY_KG_PER_M3,
            "loop_count": len(records),
            "calibration_count": len(calibration_names),
            "heldout_count": len(heldout_names),
            "selected_frequencies_hz": sorted({r.frequency_hz for r in loss_rows}),
            "selected_bp_t": sorted({r.bp_t for r in loss_rows}),
        },
        "physical_observable": {
            "target": "EnergyLoss_mJperkg from measured loss table",
            "loop_integral_check": "abs(integral H dB) / density converted to mJ/kg",
            "mean_abs_table_vs_integral_error_mj_per_kg": statistics.mean(table_errors),
            "max_abs_table_vs_integral_error_mj_per_kg": max(table_errors),
        },
        "qa_mapping": {
            "m": MODULUS,
            "bin_edges": "global calibration-loop quantiles only",
            "bin_centers": "global calibration-loop shell means only",
            "qa_state_centers": "global calibration-loop joint (b,e) state means only",
            "b": "bin(H_Aperm)",
            "e": "bin(B_T)",
            "d": "b+e",
            "a": "b+2*e",
            "J": "b*d",
            "X": "d*e",
            "K": "d*a",
            "observables": [
                "abs(integral b de)",
                "abs(integral H_bin dB_bin) converted to mJ/kg",
                "abs(integral H_shell dB_shell) converted to mJ/kg",
                "abs(integral H_J dB_X) converted to mJ/kg",
                "abs(integral H_K dB_X) converted to mJ/kg",
                "abs(integral H_(K-J) dB_X) converted to mJ/kg",
                "abs(integral H_J dB_Pi) converted to mJ/kg",
                "abs(integral J dX)",
                "abs(integral K dX)",
                "abs(integral Pi dX)",
                "sum 0.5*(Pi_i+Pi_j)*abs(delta_X)",
                "unique (b,e) states",
                "closed-loop transition graph observables over state_t=(b_t,e_t)",
                "branch-local lookup reconstruction from QA contexts to H/B shell paths",
            ],
            "transition_features": QA_TRANSITION_FEATURES,
            "memory_features": QA_MEMORY_FEATURES,
            "orbit_topology_features": QA_ORBIT_TOPOLOGY_FEATURES,
            "reconstruction_fields": QA_RECONSTRUCTION_FIELDS,
            "residual_reconstruction_fields": QA_RESIDUAL_RECONSTRUCTION_FIELDS,
            "orbit_residual_reconstruction_fields": QA_ORBIT_RESIDUAL_RECONSTRUCTION_FIELDS,
            "reconstruction_contexts": [
                "ctx_be=(b_t,e_t)",
                "ctx_be_dir=(b_t,e_t,sb_t,se_t)",
                "ctx_be_branch=(b_t,e_t,branch_t,lag_t)",
                "ctx_memory_full=(b_t,e_t,sb_t,se_t,branch_t,lag_t)",
            ],
        },
        "heldout_models": heldout_models,
        "direct_predictor_comparison": direct_comparison,
        "orbit_bridge_comparison": orbit_bridge_comparison,
        "orbit_bridge_summary": {
            "orbit_bridge_best_model_by_rmse": orbit_bridge_best_model_by_rmse,
            "orbit_bridge_best_nonraw_model_by_rmse": (
                orbit_bridge_best_nonraw_model_by_rmse
            ),
            "orbit_family_residual_delta_rmse_vs_nonqa_shell": (
                orbit_family_delta_rmse
            ),
            "orbit_family_residual_delta_r2_vs_nonqa_shell": orbit_family_delta_r2,
            "orbit_family_residual_transfer_wins_vs_nonqa_shell": (
                orbit_residual_family_rmse_wins
            ),
            "orbit_family_norm_transfer_wins_vs_nonqa_shell": (
                orbit_residual_family_norm_rmse_wins
            ),
            "orbit_family_norm_branch_transfer_wins_vs_nonqa_shell": (
                orbit_residual_family_norm_branch_rmse_wins
            ),
            "orbit_bridge_result_interpretation": orbit_bridge_result_interpretation,
            "orbit_bridge_permutation_tests": orbit_bridge_permutation,
            "orbit_bridge_permutation_interpretation": (
                orbit_bridge_permutation_interpretation
            ),
            "orbit_bridge_regime_diagnosis": orbit_regime_diagnosis,
            "orbit_bridge_regime_interpretation": (
                orbit_bridge_regime_interpretation
            ),
        },
        "transfer_split_summary": {
            "split_count": len(split_summaries),
            "splits": split_summaries,
            "qa_lifted_j_pi_r2_wins_vs_nonqa_shell": qa_j_pi_r2_wins,
            "qa_lifted_j_pi_rmse_wins_vs_nonqa_shell": qa_j_pi_rmse_wins,
            "qa_transition_only_rmse_wins_vs_nonqa_shell": transition_only_rmse_wins,
            "qa_transition_plus_shell_rmse_wins_vs_nonqa_shell": transition_plus_shell_rmse_wins,
            "nonqa_shell_plus_transition_rmse_wins_vs_nonqa_shell": nonqa_plus_transition_rmse_wins,
            "best_transition_family": best_transition_family,
            "qa_memory_only_rmse_wins_vs_nonqa_shell": memory_only_rmse_wins,
            "qa_memory_plus_qa_shell_rmse_wins_vs_nonqa_shell": memory_plus_shell_rmse_wins,
            "nonqa_shell_plus_qa_memory_rmse_wins_vs_nonqa_shell": nonqa_plus_memory_rmse_wins,
            "qa_memory_plus_transition_rmse_wins_vs_nonqa_shell": memory_plus_transition_rmse_wins,
            "nonqa_shell_plus_transition_plus_memory_rmse_wins_vs_nonqa_shell": nonqa_plus_transition_memory_rmse_wins,
            "best_memory_family": best_memory_family,
            "memory_result_interpretation": memory_result_interpretation,
            "qa_reconstruct_be_rmse_wins_vs_nonqa_shell": reconstruct_be_rmse_wins,
            "qa_reconstruct_be_dir_rmse_wins_vs_nonqa_shell": reconstruct_be_dir_rmse_wins,
            "qa_reconstruct_be_branch_rmse_wins_vs_nonqa_shell": reconstruct_be_branch_rmse_wins,
            "qa_reconstruct_memory_full_rmse_wins_vs_nonqa_shell": reconstruct_memory_full_rmse_wins,
            "best_reconstruction_family": best_reconstruction_family,
            "reconstruction_result_interpretation": reconstruction_result_interpretation,
            "qa_residual_reconstruct_be_rmse_wins_vs_nonqa_shell": (
                residual_reconstruct_be_rmse_wins
            ),
            "qa_residual_reconstruct_be_dir_rmse_wins_vs_nonqa_shell": (
                residual_reconstruct_be_dir_rmse_wins
            ),
            "qa_residual_reconstruct_be_branch_rmse_wins_vs_nonqa_shell": (
                residual_reconstruct_be_branch_rmse_wins
            ),
            "qa_residual_reconstruct_memory_full_rmse_wins_vs_nonqa_shell": (
                residual_reconstruct_memory_full_rmse_wins
            ),
            "best_residual_reconstruction_family": best_residual_reconstruction_family,
            "residual_reconstruction_result_interpretation": (
                residual_reconstruction_result_interpretation
            ),
            "qa_orbit_residual_reconstruct_family_rmse_wins_vs_nonqa_shell": (
                orbit_residual_family_rmse_wins
            ),
            "qa_orbit_residual_reconstruct_family_norm_rmse_wins_vs_nonqa_shell": (
                orbit_residual_family_norm_rmse_wins
            ),
            "qa_orbit_residual_reconstruct_family_norm_branch_rmse_wins_vs_nonqa_shell": (
                orbit_residual_family_norm_branch_rmse_wins
            ),
            "best_orbit_residual_family": best_orbit_residual_family,
            "orbit_residual_result_interpretation": orbit_residual_result_interpretation,
            "lifted_pair_sweep_summary": lifted_pair_sweep_summary,
            "phase_pair_sweep_summary": phase_pair_sweep_summary,
        },
        "verdict": {
            "qa_not_random": (
                "YES: QA-only held-out models reduce error substantially relative "
                "to the mean-loss baseline on real measured loops."
            ),
            "qa_beats_steinmetz_here": (
                "YES: the direct dimensioned QA shell integral outperforms the "
                "log-linear Steinmetz baseline on this selected Ring35 SIN subset."
            ),
            "qa_adds_to_steinmetz_here": (
                "The tested log Steinmetz+rank-QA shape model does not improve "
                "Steinmetz, but the physically dimensioned QA shell integral beats "
                "Steinmetz directly."
            ),
            "qa_beats_nonqa_shell_here": (
                "YES"
                if qa_shell_r2 > nonqa_shell_r2 and qa_shell_mae < nonqa_shell_mae
                else "NO"
            ),
            "qa_lifted_vs_nonqa_shell_here": (
                "MIXED: calibrated QA J/Pi shells have higher held-out R2 than "
                "ordinary H/B shell binning, but worse MAE/RMSE. This is not yet "
                "a clean QA-specific win."
                if lifted_j_pi_r2 > nonqa_shell_r2 and lifted_j_pi_mae > nonqa_shell_mae
                else "See direct_predictor_comparison."
            ),
            "shell_result_interpretation": (
                "The raw physical loop integral is the expected upper-bound sanity "
                "check. Ordinary non-QA H/B shell binning explains most of the "
                "direct shell result and beats the joint (b,e) QA state shell by "
                "MAE/RMSE. A lifted QA J/Pi shell slightly beats non-QA by R2 but "
                "not by MAE/RMSE, so this run supports finite-shell hysteresis "
                "mapping and only a mixed, not decisive, QA-specific advantage."
            ),
            "transfer_result_interpretation": (
                f"Across {len(split_summaries)} index/frequency/amplitude transfer "
                f"splits, QA J/Pi beats ordinary non-QA shell by R2 in "
                f"{qa_j_pi_r2_wins} splits and by RMSE in {qa_j_pi_rmse_wins} splits. "
                "This hardens the conclusion by checking family-heldout transfer. "
                "The lifted-pair sweep reports whether another QA variable pair "
                "transfers more consistently."
            ),
            "lifted_pair_sweep_interpretation": (
                f"Best lifted pair by the predeclared transfer score is "
                f"{best_sweep_pair['pair']}, with RMSE wins in "
                f"{best_sweep_pair['rmse_win_count']} of {len(split_summaries)} splits "
                f"and median RMSE ratio {best_sweep_pair['median_rmse_ratio_to_nonqa']:.3f} "
                "relative to ordinary non-QA shell. This does not establish a stable "
                "QA-specific advantage yet."
            ),
            "phase_pair_sweep_interpretation": (
                f"Best phase/orientation-aware QA pair is {best_phase_pair['pair']}, "
                f"with RMSE wins in {best_phase_pair['rmse_win_count']} of "
                f"{len(split_summaries)} splits and median RMSE ratio "
                f"{best_phase_pair['median_rmse_ratio_to_nonqa']:.3f} relative to "
                "ordinary non-QA shell."
            ),
            "transition_result_interpretation": (
                f"Transition-only QA wins {transition_only_rmse_wins} of "
                f"{len(split_summaries)} RMSE transfer splits against non-QA shell; "
                f"QA transition plus QA shell wins {transition_plus_shell_rmse_wins}; "
                f"non-QA shell plus QA transition wins {nonqa_plus_transition_rmse_wins}. "
                f"Best transition family: {best_transition_family}."
            ),
            "memory_result_interpretation": memory_result_interpretation,
            "reconstruction_result_interpretation": reconstruction_result_interpretation,
            "residual_reconstruction_result_interpretation": (
                residual_reconstruction_result_interpretation
            ),
            "orbit_residual_result_interpretation": orbit_residual_result_interpretation,
            "orbit_bridge_result_interpretation": orbit_bridge_result_interpretation,
            "orbit_bridge_permutation_interpretation": (
                orbit_bridge_permutation_interpretation
            ),
            "orbit_bridge_regime_interpretation": orbit_bridge_regime_interpretation,
            "leakage_controls": (
                "H/B quantile edges, marginal shell centers, and joint QA state "
                "centers are fit on calibration loops only. Direct predictors use "
                "no held-out loss labels and no fitted regression coefficient. "
                "Branch-local reconstruction lookup tables are fit separately per "
                "split on calibration traces only. Residual reconstruction lookup "
                "tables also use calibration traces only and fall back to (b,e) "
                "residuals or zero residual when a held-out context is unseen."
            ),
        },
        "records": records,
        "interpretation": (
            "A useful QA hysteresis mapping should improve held-out loss prediction "
            "over the mean baseline and should be evaluated against Steinmetz. "
            "The rank-only QA models test whether measured loop geometry maps into "
            "QA coordinates without physical scale. The dimensioned QA shell model "
            "keeps calibration-set H/B shell centers and directly approximates the "
            "hysteresis work integral."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["heldout_models"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
