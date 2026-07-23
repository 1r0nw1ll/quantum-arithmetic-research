#!/usr/bin/env python3
"""Stage 32 global orbit-path invariant mining with order controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_orbit_rules import orbit_family, orbit_period, qa_step


DOMAIN = "QA_QUANTUM_ARITHMETIC_ORBIT_PATH_INVARIANT_STAGE32.v1"
MODULI = (9, 24)
ORBIT_FAMILY_CODE = {"cosmos": 0, "satellite": 1, "singularity": 2}
DEFAULT_TARGETS = (
    "path_any_X_semiprime,"
    "path_count_X_semiprime_ge2,"
    "path_any_F_semiprime,"
    "path_count_F_squarefree_ge_half,"
    "path_X_omega_range_ge3,"
    "path_F_omega_range_ge3,"
    "path_X_squarefree_flip_count_ge2,"
    "path_F_squarefree_flip_count_ge2,"
    "path_any_G_square,"
    "path_any_h_integer,"
    "path_any_DplusF_square,"
    "path_contains_G_square_and_h_integer"
)


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def qa_mod_coord(value: int, modulus: int) -> int:
    return ((value - 1) % modulus) + 1


FACTOR_CACHE: dict[int, list[int]] = {}


def prime_factors(n: int) -> list[int]:
    if n in FACTOR_CACHE:
        return FACTOR_CACHE[n]
    factors: list[int] = []
    remaining = n
    while remaining > 1 and remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    p = 3
    while p * p <= remaining:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p
        p += 2
    if remaining > 1:
        factors.append(remaining)
    FACTOR_CACHE[n] = factors
    return factors


def factor_count(n: int) -> int:
    return len(prime_factors(n))


def is_semiprime(n: int) -> bool:
    return factor_count(n) == 2


def is_squarefree(n: int) -> bool:
    factors = prime_factors(n)
    return len(factors) == len(set(factors))


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def cap(n: int, high: int = 8) -> int:
    return min(high, n)


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    D = d * d
    F = a * b
    G = D + e * e
    X = e * d
    D_plus_F = D + F
    return {"b": b, "e": e, "d": d, "a": a, "D": D, "X": X, "F": F, "G": G, "D_plus_F": D_plus_F}


def product_factor_count(left: int, right: int) -> int:
    return factor_count(left) + factor_count(right)


def product_semiprime(left: int, right: int) -> bool:
    return product_factor_count(left, right) == 2


def product_squarefree(left: int, right: int) -> bool:
    return is_squarefree(left) and is_squarefree(right) and math.gcd(left, right) == 1


def x_factor_count(row: dict[str, int]) -> int:
    return product_factor_count(row["e"], row["d"])


def f_factor_count(row: dict[str, int]) -> int:
    return product_factor_count(row["a"], row["b"])


def x_semiprime(row: dict[str, int]) -> bool:
    return product_semiprime(row["e"], row["d"])


def f_semiprime(row: dict[str, int]) -> bool:
    return product_semiprime(row["a"], row["b"])


def x_squarefree(row: dict[str, int]) -> bool:
    return product_squarefree(row["e"], row["d"])


def f_squarefree(row: dict[str, int]) -> bool:
    return product_squarefree(row["a"], row["b"])


def t_integer_state(b: int, e: int, steps: int) -> tuple[int, int]:
    cur_b = b
    cur_e = e
    for _ in range(steps):
        cur_b, cur_e = cur_e, cur_b + cur_e
    return cur_b, cur_e


def integer_path_rows(row: dict[str, int], length: int) -> list[dict[str, int]]:
    out = []
    for step in range(length):
        b, e = t_integer_state(row["b"], row["e"], step)
        out.append(qa_values(b, e))
    return out


def residue_path(row: dict[str, int], modulus: int, length: int) -> tuple[tuple[int, int], ...]:
    b = qa_mod_coord(row["b"], modulus)
    e = qa_mod_coord(row["e"], modulus)
    out = []
    for _ in range(length):
        out.append((b, e))
        b, e = qa_step(b, e, modulus)
    return tuple(out)


def stable_shuffle_path(path: tuple[tuple[int, int], ...], seed: int) -> tuple[tuple[int, int], ...]:
    payload = canonical_json({"path": path, "seed": seed})
    local_seed = int(domain_sha256(f"{DOMAIN}.path_shuffle", payload)[:16], 16)
    rng = random.Random(local_seed)
    items = list(path)
    rng.shuffle(items)
    return tuple(items)


def orbit_index_map(m: int) -> dict[tuple[int, int], int]:
    seen: set[tuple[int, int]] = set()
    out: dict[tuple[int, int], int] = {}
    index = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen:
                continue
            cur = (b, e)
            while cur not in seen:
                seen.add(cur)
                out[cur] = index
                cur = qa_step(cur[0], cur[1], m)
            index += 1
    return out


ORBIT_INDEX = {9: orbit_index_map(9), 24: orbit_index_map(24)}


def orbit_family_code(row: dict[str, int], modulus: int) -> int:
    b = qa_mod_coord(row["b"], modulus)
    e = qa_mod_coord(row["e"], modulus)
    return ORBIT_FAMILY_CODE[orbit_family(int(b), int(e), modulus)]


def orbit_id(row: dict[str, int], modulus: int) -> int:
    b = qa_mod_coord(row["b"], modulus)
    e = qa_mod_coord(row["e"], modulus)
    return ORBIT_INDEX[modulus][(b, e)]


PATH_STATS_CACHE: dict[tuple[int, int, int], dict[str, object]] = {}


def path_stats(row: dict[str, int], length: int) -> dict[str, object]:
    key = (row["b"], row["e"], length)
    if key in PATH_STATS_CACHE:
        return PATH_STATS_CACHE[key]
    rows = integer_path_rows(row, length)
    x_omegas = [x_factor_count(item) for item in rows]
    f_omegas = [f_factor_count(item) for item in rows]
    x_squarefree_flags = [x_squarefree(item) for item in rows]
    f_squarefree_flags = [f_squarefree(item) for item in rows]
    stats = {
        "x_semiprime_count": sum(1 for item in rows if x_semiprime(item)),
        "f_semiprime_count": sum(1 for item in rows if f_semiprime(item)),
        "f_squarefree_count": sum(1 for item in rows if f_squarefree(item)),
        "x_omega_min": min(x_omegas),
        "x_omega_max": max(x_omegas),
        "f_omega_min": min(f_omegas),
        "f_omega_max": max(f_omegas),
        "x_squarefree_flips": sum(1 for left, right in zip(x_squarefree_flags, x_squarefree_flags[1:]) if left != right),
        "f_squarefree_flips": sum(1 for left, right in zip(f_squarefree_flags, f_squarefree_flags[1:]) if left != right),
        "any_G_square": any(is_square(item["G"]) for item in rows),
        "any_h_integer": any(is_square(item["F"]) for item in rows),
        "any_DplusF_square": any(is_square(item["D_plus_F"]) for item in rows),
    }
    PATH_STATS_CACHE[key] = stats
    return stats


def label_for(row: dict[str, int], target: str, path_length: int) -> int:
    stats = path_stats(row, path_length)
    if target == "path_any_X_semiprime":
        return int(int(stats["x_semiprime_count"]) > 0)
    if target == "path_count_X_semiprime_ge2":
        return int(int(stats["x_semiprime_count"]) >= 2)
    if target == "path_any_F_semiprime":
        return int(int(stats["f_semiprime_count"]) > 0)
    if target == "path_count_F_squarefree_ge_half":
        return int(int(stats["f_squarefree_count"]) * 2 >= path_length)
    if target == "path_X_omega_range_ge3":
        return int(int(stats["x_omega_max"]) - int(stats["x_omega_min"]) >= 3)
    if target == "path_F_omega_range_ge3":
        return int(int(stats["f_omega_max"]) - int(stats["f_omega_min"]) >= 3)
    if target == "path_X_squarefree_flip_count_ge2":
        return int(int(stats["x_squarefree_flips"]) >= 2)
    if target == "path_F_squarefree_flip_count_ge2":
        return int(int(stats["f_squarefree_flips"]) >= 2)
    if target == "path_any_G_square":
        return int(bool(stats["any_G_square"]))
    if target == "path_any_h_integer":
        return int(bool(stats["any_h_integer"]))
    if target == "path_any_DplusF_square":
        return int(bool(stats["any_DplusF_square"]))
    if target == "path_contains_G_square_and_h_integer":
        return int(bool(stats["any_G_square"]) and bool(stats["any_h_integer"]))
    raise ValueError(f"unknown target: {target}")


FeatureFunc = Callable[[dict[str, int]], object]


def path_factor_signature(row: dict[str, int], path_length: int) -> tuple[int, ...]:
    stats = path_stats(row, path_length)
    return (
        cap(int(stats["x_semiprime_count"])),
        cap(int(stats["f_semiprime_count"])),
        cap(int(stats["f_squarefree_count"])),
        cap(int(stats["x_omega_min"])),
        cap(int(stats["x_omega_max"])),
        cap(int(stats["f_omega_min"])),
        cap(int(stats["f_omega_max"])),
        cap(int(stats["x_squarefree_flips"])),
        cap(int(stats["f_squarefree_flips"])),
        int(bool(stats["any_G_square"])),
        int(bool(stats["any_h_integer"])),
        int(bool(stats["any_DplusF_square"])),
    )


def feature_sets(path_length: int, shuffle_seed: int) -> dict[str, tuple[str, FeatureFunc]]:
    return {
        "ordered_path9": ("ordered_path", lambda row: residue_path(row, 9, path_length)),
        "ordered_path24": ("ordered_path", lambda row: residue_path(row, 24, path_length)),
        "ordered_path9_24": (
            "ordered_path",
            lambda row: (residue_path(row, 9, path_length), residue_path(row, 24, path_length)),
        ),
        "path_family_sequence9": (
            "ordered_path",
            lambda row: tuple(orbit_family_code({"b": b, "e": e}, 9) for b, e in residue_path(row, 9, path_length)),
        ),
        "path_family_sequence24": (
            "ordered_path",
            lambda row: tuple(orbit_family_code({"b": b, "e": e}, 24) for b, e in residue_path(row, 24, path_length)),
        ),
        "unordered_path9": ("unordered_path_control", lambda row: tuple(sorted(residue_path(row, 9, path_length)))),
        "unordered_path24": ("unordered_path_control", lambda row: tuple(sorted(residue_path(row, 24, path_length)))),
        "unordered_path9_24": (
            "unordered_path_control",
            lambda row: (tuple(sorted(residue_path(row, 9, path_length))), tuple(sorted(residue_path(row, 24, path_length)))),
        ),
        "shuffled_path9": (
            "shuffled_path_control",
            lambda row: stable_shuffle_path(residue_path(row, 9, path_length), shuffle_seed),
        ),
        "shuffled_path24": (
            "shuffled_path_control",
            lambda row: stable_shuffle_path(residue_path(row, 24, path_length), shuffle_seed),
        ),
        "shuffled_path9_24": (
            "shuffled_path_control",
            lambda row: (
                stable_shuffle_path(residue_path(row, 9, path_length), shuffle_seed),
                stable_shuffle_path(residue_path(row, 24, path_length), shuffle_seed),
            ),
        ),
        "be_pair9": ("current_cell_control", lambda row: (qa_mod_coord(row["b"], 9), qa_mod_coord(row["e"], 9))),
        "be_pair24": ("current_cell_control", lambda row: (qa_mod_coord(row["b"], 24), qa_mod_coord(row["e"], 24))),
        "be_pair9_24": (
            "current_cell_control",
            lambda row: (
                qa_mod_coord(row["b"], 9),
                qa_mod_coord(row["e"], 9),
                qa_mod_coord(row["b"], 24),
                qa_mod_coord(row["e"], 24),
            ),
        ),
        "current_factor_signature": (
            "factor_control",
            lambda row: (
                cap(x_factor_count(row)),
                cap(f_factor_count(row)),
                int(x_squarefree(row)),
                int(f_squarefree(row)),
                int(x_semiprime(row)),
                int(f_semiprime(row)),
                int(is_square(row["G"])),
                int(is_square(row["F"])),
                int(is_square(row["D_plus_F"])),
            ),
        ),
        "path_factor_signature": ("factor_control", lambda row: path_factor_signature(row, path_length)),
        "qa_orbit_family9": ("static_orbit_control", lambda row: orbit_family_code(row, 9)),
        "qa_orbit_id9": ("static_orbit_control", lambda row: orbit_id(row, 9)),
        "qa_orbit_family24": ("static_orbit_control", lambda row: orbit_family_code(row, 24)),
        "qa_orbit_id24": ("static_orbit_control", lambda row: orbit_id(row, 24)),
        "orbit_period9_24": (
            "static_orbit_control",
            lambda row: (
                orbit_period(qa_mod_coord(row["b"], 9), qa_mod_coord(row["e"], 9), 9),
                orbit_period(qa_mod_coord(row["b"], 24), qa_mod_coord(row["e"], 24), 24),
            ),
        ),
    }


def square_rows(start: int, end: int) -> list[dict[str, int]]:
    return [qa_values(b, e) for b in range(start, end + 1) for e in range(start, end + 1)]


def score_predictions(predictions: list[int], labels: list[int]) -> dict[str, float | int]:
    tp = sum(1 for pred, truth in zip(predictions, labels) if pred == 1 and truth == 1)
    fp = sum(1 for pred, truth in zip(predictions, labels) if pred == 1 and truth == 0)
    fn = sum(1 for pred, truth in zip(predictions, labels) if pred == 0 and truth == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    base_rate = sum(labels) / len(labels) if labels else 0.0
    return {
        "predicted_positive": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
        "base_rate": base_rate,
    }


def best_threshold(scores: list[float], labels: list[int]) -> float:
    grouped: list[tuple[float, int, int]] = []
    for score, label in sorted(zip(scores, labels), reverse=True):
        if grouped and grouped[-1][0] == score:
            prior_score, prior_total, prior_positive = grouped[-1]
            grouped[-1] = (prior_score, prior_total + 1, prior_positive + label)
        else:
            grouped.append((score, 1, label))
    total_positive = sum(labels)
    predicted_positive = 0
    true_positive = 0
    best_f1 = -1.0
    selected = grouped[0][0] if grouped else 0.0
    for threshold, total, positive in grouped:
        predicted_positive += total
        true_positive += positive
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / total_positive if total_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            selected = threshold
    return selected


def train_category_rate(categories: list[str], labels: list[int]) -> dict[str, object]:
    base = sum(labels) / len(labels) if labels else 0.0
    counts: dict[str, list[int]] = {}
    for category, label in zip(categories, labels):
        key = category
        if key not in counts:
            counts[key] = [0, 0]
        counts[key][0] += 1
        counts[key][1] += label
    rates = {key: positive / total for key, (total, positive) in counts.items()}
    scores = [rates.get(category, base) for category in categories]
    return {"base_rate": base, "rates": rates, "threshold": best_threshold(scores, labels)}


def evaluate_feature(
    train_categories: list[str],
    train_labels: list[int],
    test_categories: list[str],
    test_labels: list[int],
) -> dict[str, float | int]:
    model = train_category_rate(train_categories, train_labels)
    rates = model["rates"]
    base = float(model["base_rate"])
    threshold = float(model["threshold"])
    scores = [rates.get(category, base) for category in test_categories]
    predictions = [int(score >= threshold) for score in scores]
    return score_predictions(predictions, test_labels)


def null_summary(
    train_categories: list[str],
    train_labels: list[int],
    test_categories: list[str],
    test_labels: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    lifts = []
    f1s = []
    for _ in range(iterations):
        shuffled = list(train_labels)
        rng.shuffle(shuffled)
        row = evaluate_feature(train_categories, shuffled, test_categories, test_labels)
        lifts.append(float(row["lift"]))
        f1s.append(float(row["f1"]))
    return {
        "null_lift_mean": sum(lifts) / len(lifts) if lifts else 0.0,
        "null_lift_max": max(lifts) if lifts else 0.0,
        "null_f1_mean": sum(f1s) / len(f1s) if f1s else 0.0,
        "null_f1_max": max(f1s) if f1s else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def verdict_for(row: dict[str, object], margin: float) -> str:
    if row["train_positive_rows"] < row["min_positive"] or row["train_negative_rows"] < row["min_positive"]:
        return "DEGENERATE_OR_LOW_TRAIN_SUPPORT"
    if row["test_positive_rows"] < row["min_positive"] or row["test_negative_rows"] < row["min_positive"]:
        return "DEGENERATE_OR_LOW_TEST_SUPPORT"
    if float(row["best_ordered_path_lift"]) <= float(row["best_ordered_path_null_lift_max"]):
        return "ORDERED_PATH_NULL_COMPETITIVE"
    if float(row["ordered_path_margin"]) >= margin:
        return "ORBIT_PATH_CANDIDATE"
    return "CONTROL_BASELINE_COMPETITIVE"


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [piece.strip() for piece in args.targets.split(",") if piece.strip()]
    train_rows = square_rows(args.train_start, args.train_end)
    test_rows = square_rows(args.test_start, args.test_end)
    features = feature_sets(args.path_length, args.shuffle_seed)
    train_categories = {name: [repr(func(row)) for row in train_rows] for name, (_kind, func) in features.items()}
    test_categories = {name: [repr(func(row)) for row in test_rows] for name, (_kind, func) in features.items()}
    feature_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        train_labels = [label_for(row, target, args.path_length) for row in train_rows]
        test_labels = [label_for(row, target, args.path_length) for row in test_rows]
        train_positive = sum(train_labels)
        test_positive = sum(test_labels)
        best_ordered: dict[str, object] | None = None
        best_control: dict[str, object] | None = None
        for feature_index, (feature_name, (feature_kind, _func)) in enumerate(features.items()):
            if (
                train_positive < args.min_positive
                or len(train_labels) - train_positive < args.min_positive
                or test_positive < args.min_positive
                or len(test_labels) - test_positive < args.min_positive
            ):
                observed = {
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "lift": None,
                    "base_rate": test_positive / len(test_labels),
                    "predicted_positive": None,
                }
                controls = {"null_lift_mean": None, "null_lift_max": None, "null_f1_mean": None, "null_f1_max": None}
            else:
                observed = evaluate_feature(train_categories[feature_name], train_labels, test_categories[feature_name], test_labels)
                if feature_kind == "ordered_path":
                    controls = null_summary(
                        train_categories[feature_name],
                        train_labels,
                        test_categories[feature_name],
                        test_labels,
                        args.null_iterations,
                        args.seed + target_index * 1777 + feature_index * 919,
                    )
                else:
                    controls = {"null_lift_mean": None, "null_lift_max": None, "null_f1_mean": None, "null_f1_max": None}
            payload = {
                "target": target,
                "feature_set": feature_name,
                "feature_kind": feature_kind,
                "train_window": f"square_{args.train_start}_{args.train_end}",
                "test_window": f"square_{args.test_start}_{args.test_end}",
                "path_length": args.path_length,
                "model": "category_rate_threshold",
                "train_positive_rows": train_positive,
                "train_negative_rows": len(train_labels) - train_positive,
                "test_positive_rows": test_positive,
                "test_negative_rows": len(test_labels) - test_positive,
                "test_base_rate": observed["base_rate"],
                "precision": observed["precision"],
                "recall": observed["recall"],
                "f1": observed["f1"],
                "observed_lift": observed["lift"],
                "predicted_positive": observed["predicted_positive"],
                "null_lift_mean": controls["null_lift_mean"],
                "null_lift_max": controls["null_lift_max"],
                "null_f1_mean": controls["null_f1_mean"],
                "null_f1_max": controls["null_f1_max"],
            }
            feature_rows.append({**payload, "hash": domain_sha256(f"{DOMAIN}.feature", canonical_json(payload))})
            if observed["lift"] is not None:
                if feature_kind == "ordered_path":
                    if best_ordered is None or float(observed["lift"]) > float(best_ordered["observed_lift"]):
                        best_ordered = {**payload}
                else:
                    if best_control is None or float(observed["lift"]) > float(best_control["observed_lift"]):
                        best_control = {**payload}
        ordered_lift = 0.0 if best_ordered is None else float(best_ordered["observed_lift"])
        control_lift = 0.0 if best_control is None else float(best_control["observed_lift"])
        ordered_null = 0.0 if best_ordered is None else float(best_ordered["null_lift_max"] or 0.0)
        summary = {
            "target": target,
            "train_positive_rows": train_positive,
            "train_negative_rows": len(train_labels) - train_positive,
            "test_positive_rows": test_positive,
            "test_negative_rows": len(test_labels) - test_positive,
            "min_positive": args.min_positive,
            "test_base_rate": test_positive / len(test_labels),
            "best_ordered_path_feature": "" if best_ordered is None else best_ordered["feature_set"],
            "best_ordered_path_lift": ordered_lift,
            "best_ordered_path_null_lift_max": ordered_null,
            "best_control_feature": "" if best_control is None else best_control["feature_set"],
            "best_control_kind": "" if best_control is None else best_control["feature_kind"],
            "best_control_lift": control_lift,
            "ordered_path_margin": ordered_lift - control_lift,
        }
        summary["verdict"] = verdict_for(summary, args.orbit_margin)
        summary["hash"] = domain_sha256(f"{DOMAIN}.target", canonical_json(summary))
        target_rows.append(summary)
    target_rows.sort(
        key=lambda row: (
            str(row["verdict"]) == "ORBIT_PATH_CANDIDATE",
            float(row["ordered_path_margin"]),
            float(row["best_ordered_path_lift"]),
        ),
        reverse=True,
    )
    feature_rows.sort(key=lambda row: (str(row["target"]), float(row["observed_lift"] or 0.0)), reverse=True)
    target_csv = out_dir / args.target_summary_csv
    feature_csv = out_dir / args.feature_leaderboard_csv
    write_csv(target_csv, target_rows)
    write_csv(feature_csv, feature_rows)
    verdict_counts: dict[str, int] = {}
    for row in target_rows:
        verdict = str(row["verdict"])
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    payload = {
        "stage_id": "qa_quantum_arithmetic_stage32_orbit_path_invariant",
        "hypothesis": (
            "Ordered global orbit-path features should only count as QA-dynamic evidence if they beat shuffled-path, "
            "unordered-path, current-cell, static-orbit, and factor-signature controls on orbit-integrated labels."
        ),
        "parameters": {
            "train_window": f"square_{args.train_start}_{args.train_end}",
            "test_window": f"square_{args.test_start}_{args.test_end}",
            "targets": targets,
            "feature_sets": {name: kind for name, (kind, _func) in features.items()},
            "path_length": args.path_length,
            "orbit_moduli": list(MODULI),
            "null_iterations": args.null_iterations,
            "min_positive": args.min_positive,
            "orbit_margin": args.orbit_margin,
            "seed": args.seed,
            "shuffle_seed": args.shuffle_seed,
        },
        "artifacts": {
            "target_summary_csv": str(target_csv),
            "feature_leaderboard_csv": str(feature_csv),
        },
        "target_rows": len(target_rows),
        "feature_rows": len(feature_rows),
        "verdict_counts": verdict_counts,
        "top_targets": target_rows[:20],
        "honest_interpretation": (
            "Ordered path features are candidate evidence only when they beat all non-ordered controls. "
            "A result where current-cell or path-factor controls dominate means the label is still local arithmetic, "
            "not evidence for path-order-specific QA dynamics."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    json_path = out_dir / args.summary_json
    json_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            train_start=1,
            train_end=12,
            test_start=13,
            test_end=24,
            targets="path_any_X_semiprime,path_X_omega_range_ge3,path_any_G_square",
            path_length=8,
            null_iterations=2,
            min_positive=1,
            orbit_margin=0.05,
            seed=3232,
            shuffle_seed=13232,
            target_summary_csv="stage32_target_summary.csv",
            feature_leaderboard_csv="stage32_feature_leaderboard.csv",
            summary_json="stage32_summary.json",
        )
        payload = run(args)
        ok = (
            payload["target_rows"] == 3
            and payload["feature_rows"] == 3 * len(feature_sets(args.path_length, args.shuffle_seed))
            and Path(tmp, "stage32_target_summary.csv").exists()
            and Path(tmp, "stage32_feature_leaderboard.csv").exists()
            and Path(tmp, "stage32_summary.json").exists()
            and len(payload["canonical_hash"]) == 64
        )
        return {"ok": ok, "target_rows": payload["target_rows"], "feature_rows": payload["feature_rows"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=100)
    parser.add_argument("--test-start", type=int, default=101)
    parser.add_argument("--test-end", type=int, default=300)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--path-length", type=int, default=24)
    parser.add_argument("--null-iterations", type=int, default=50)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--orbit-margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3232)
    parser.add_argument("--shuffle-seed", type=int, default=13232)
    parser.add_argument("--target-summary-csv", default="qa_quantum_arithmetic_stage32_orbit_path_target_summary.csv")
    parser.add_argument("--feature-leaderboard-csv", default="qa_quantum_arithmetic_stage32_orbit_path_feature_leaderboard.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage32_orbit_path_invariant.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(
        canonical_json(
            {
                "ok": True,
                "stage_id": payload["stage_id"],
                "target_rows": payload["target_rows"],
                "feature_rows": payload["feature_rows"],
                "verdict_counts": payload["verdict_counts"],
                "artifacts": payload["artifacts"],
                "canonical_hash": payload["canonical_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
