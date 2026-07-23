#!/usr/bin/env python3
"""Stage 9 cross-scale confirmation for QA arithmetic residue mining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from functools import lru_cache
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_CROSS_SCALE_CONFIRMATION_STAGE9.v1"


DEFAULT_TARGETS = (
    "G_square,h_integer,X_semiprime,F_semiprime,W_semiprime,"
    "R_inradius_semiprime,J_exradius_semiprime,K_semiperimeter_semiprime,EA_exradius_semiprime,"
    "directrix_distance_integer,primitive_triangle_condition,eccentricity_den_smooth_13,"
    "X_smooth_7,K_smooth_13"
)


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


@lru_cache(maxsize=None)
def factor_tuple(n: int) -> tuple[int, ...]:
    factors: list[int] = []
    temp = n
    while temp > 1 and temp % 2 == 0:
        factors.append(2)
        temp //= 2
    divisor = 3
    while divisor <= math.isqrt(temp):
        while temp % divisor == 0:
            factors.append(divisor)
            temp //= divisor
        divisor += 2
    if temp > 1:
        factors.append(temp)
    return tuple(factors)


def omega_product(*parts: int) -> int:
    return sum(len(factor_tuple(part)) for part in parts)


def distinct_omega_product(*parts: int) -> int:
    factors: set[int] = set()
    for part in parts:
        factors.update(factor_tuple(part))
    return len(factors)


def is_semiprime_product(*parts: int) -> bool:
    return omega_product(*parts) == 2


def is_smooth_product(bound: int, *parts: int) -> bool:
    factors: list[int] = []
    for part in parts:
        factors.extend(factor_tuple(part))
    return bool(factors) and max(factors) <= bound


def is_squarefree_product(*parts: int) -> bool:
    factors: list[int] = []
    for part in parts:
        factors.extend(factor_tuple(part))
    return len(factors) == len(set(factors))


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = e + d
    D = d * d
    q = e + a
    return {
        "b": b,
        "e": e,
        "d": d,
        "a": a,
        "D": D,
        "X": e * d,
        "F": b * a,
        "G": D + e * e,
        "W": d * q,
        "R": b * e,
        "J": d * b,
        "K": d * a,
        "EA": e * a,
        "q": q,
    }


def directrix_distance(row: dict[str, int]) -> int | None:
    numerator = row["d"] * row["d"] * row["d"]
    if numerator % row["e"] != 0:
        return None
    return numerator // row["e"]


def primitive_triangle(row: dict[str, int]) -> bool:
    return math.gcd(row["d"], row["e"]) == 1 and (row["d"] - row["e"]) % 2 != 0


def reduced_eccentricity_parts(row: dict[str, int]) -> tuple[int, int]:
    divisor = math.gcd(row["e"], row["d"])
    return row["e"] // divisor, row["d"] // divisor


def label_for(row: dict[str, int], target: str) -> int:
    if target == "G_square":
        return int(is_square(row["G"]))
    if target == "h_integer":
        return int(is_square(row["F"]))
    if target == "X_semiprime":
        return int(is_semiprime_product(row["e"], row["d"]))
    if target == "F_semiprime":
        return int(is_semiprime_product(row["b"], row["a"]))
    if target == "W_semiprime":
        return int(is_semiprime_product(row["d"], row["q"]))
    if target == "R_inradius_semiprime":
        return int(is_semiprime_product(row["b"], row["e"]))
    if target == "J_exradius_semiprime":
        return int(is_semiprime_product(row["d"], row["b"]))
    if target == "K_semiperimeter_semiprime":
        return int(is_semiprime_product(row["d"], row["a"]))
    if target == "EA_exradius_semiprime":
        return int(is_semiprime_product(row["e"], row["a"]))
    if target == "directrix_distance_integer":
        return int(directrix_distance(row) is not None)
    if target == "primitive_triangle_condition":
        return int(primitive_triangle(row))
    if target == "eccentricity_den_smooth_13":
        return int(is_smooth_product(13, reduced_eccentricity_parts(row)[1]))
    if target == "ecc_den_smooth_13":
        return int(is_smooth_product(13, reduced_eccentricity_parts(row)[1]))
    if target in {"eccentricity_reduced_den_semiprime", "ecc_den_semiprime"}:
        return int(is_semiprime_product(reduced_eccentricity_parts(row)[1]))
    if target in {"semi_latus_squarefree", "F_latus_semirectum_squarefree"}:
        return int(is_squarefree_product(row["b"], row["a"]))
    if target in {"full_latus_squarefree", "F_latus_rectum_squarefree"}:
        return int(is_squarefree_product(2, row["b"], row["a"]))
    if target in {"semi_latus_distinct_omega_2", "F_latus_semirectum_distinct_omega_2"}:
        return int(distinct_omega_product(row["b"], row["a"]) == 2)
    if target in {"director_factor_D_plus_F_semiprime", "polar_scale_D_plus_F_semiprime"}:
        return int(omega_product(row["D"] + row["F"]) == 2)
    if target == "director_factor_D_plus_F_squarefree":
        return int(is_squarefree_product(row["D"] + row["F"]))
    if target == "polar_scale_X_plus_F_semiprime":
        return int(omega_product(row["X"] + row["F"]) == 2)
    if target == "X_smooth_7":
        return int(is_smooth_product(7, row["e"], row["d"]))
    if target == "K_smooth_13":
        return int(is_smooth_product(13, row["d"], row["a"]))
    raise ValueError(f"unknown target: {target}")


def square_pair_iter(start: int, end: int):
    for b in range(start, end + 1):
        for e in range(start, end + 1):
            yield b, e


def sample_pair_iter(pair_iter, total: int, cap: int, seed: int) -> tuple[list[tuple[int, int]], bool]:
    if cap <= 0 or total <= cap:
        return list(pair_iter), False
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total), cap))
    out: list[tuple[int, int]] = []
    cursor = 0
    wanted_index = 0
    for pair in pair_iter:
        if wanted_index >= len(selected):
            break
        if cursor == selected[wanted_index]:
            out.append(pair)
            wanted_index += 1
        cursor += 1
    return out, True


def random_sparse_pairs(limit: int, count: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < count:
        seen.add((rng.randint(1, limit), rng.randint(1, limit)))
    return sorted(seen)


def build_windows(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {"name": "square_101_300", "pairs": lambda: square_pair_iter(101, 300), "total": 200 * 200, "cap": 0},
        {"name": "square_301_1000", "pairs": lambda: square_pair_iter(301, 1000), "total": 700 * 700, "cap": args.sample_cap},
        {"name": "square_1001_3000", "pairs": lambda: square_pair_iter(1001, 3000), "total": 2000 * 2000, "cap": args.sample_cap},
        {"name": "square_3001_10000", "pairs": lambda: square_pair_iter(3001, 10000), "total": 7000 * 7000, "cap": args.sample_cap},
        {"name": "random_sparse_1e6", "pairs": None, "total": args.random_count, "cap": args.random_count},
    ]


def feature_sets(base_moduli: list[int], requested: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    names = {"full_residue", "no_parity", "parity_only"} if requested == "all" else {
        piece.strip() for piece in requested.split(",") if piece.strip()
    }
    if "full_residue" in names:
        out["full_residue"] = base_moduli
    if "no_parity" in names:
        out["no_parity"] = [modulus for modulus in base_moduli if modulus != 2]
    if "parity_only" in names:
        out["parity_only"] = [2]
    return out


def feature_offsets(fields: list[str], moduli: list[int]) -> tuple[dict[tuple[str, int], int], int]:
    offsets: dict[tuple[str, int], int] = {}
    cursor = 0
    for field in fields:
        for modulus in moduli:
            offsets[(field, modulus)] = cursor
            cursor += modulus
    return offsets, cursor


def active_features(row: dict[str, int], fields: list[str], moduli: list[int], offsets: dict[tuple[str, int], int]) -> list[int]:
    active = []
    for field in fields:
        for modulus in moduli:
            active.append(offsets[(field, modulus)] + (row[field] % modulus))
    return active


def feature_matrix(rows: list[dict[str, int]], fields: list[str], moduli: list[int]) -> tuple[list[list[int]], int]:
    offsets, feature_count = feature_offsets(fields, moduli)
    return [active_features(row, fields, moduli, offsets) for row in rows], feature_count


def score_active(contrast: list[float], contrast_sum: float, active: list[int]) -> float:
    return (2.0 * sum(contrast[index] for index in active)) - contrast_sum


def best_threshold(scores: list[float], labels: list[int]) -> float:
    total_positive = sum(labels)
    grouped: list[tuple[float, int, int]] = []
    for score, label in sorted(zip(scores, labels), reverse=True):
        if grouped and grouped[-1][0] == score:
            old_score, old_total, old_positive = grouped[-1]
            grouped[-1] = (old_score, old_total + 1, old_positive + label)
        else:
            grouped.append((score, 1, label))
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


def train_hebbian(vectors: list[list[int]], labels: list[int], feature_count: int) -> dict[str, object]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("training labels must contain positive and negative rows")
    pos_counts = [0] * feature_count
    neg_counts = [0] * feature_count
    for active, label in zip(vectors, labels):
        counts = pos_counts if label else neg_counts
        for index in active:
            counts[index] += 1
    contrast = [(pos_counts[index] / positives) - (neg_counts[index] / negatives) for index in range(feature_count)]
    contrast_sum = sum(contrast)
    scores = [score_active(contrast, contrast_sum, active) for active in vectors]
    return {
        "contrast": contrast,
        "contrast_sum": contrast_sum,
        "threshold": best_threshold(scores, labels),
        "train_positive_rows": positives,
        "train_negative_rows": negatives,
    }


def threshold_metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        pred = int(score >= threshold)
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    base_rate = (tp + fn) / len(labels) if labels else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "predicted_positive": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
        "base_rate": base_rate,
    }


def average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(sorted(zip(scores, labels), reverse=True), start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def top_fraction_metrics(scores: list[float], labels: list[int], fraction: float) -> dict[str, float | int]:
    total = len(labels)
    positives = sum(labels)
    base_rate = positives / total if total else 0.0
    k = max(1, int(round(total * fraction)))
    selected = sorted(zip(scores, labels), reverse=True)[:k]
    hits = sum(label for _, label in selected)
    precision = hits / k
    return {
        "fraction": fraction,
        "k": k,
        "hits": hits,
        "precision": precision,
        "recall": hits / positives if positives else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
    }


def model_scores(model: dict[str, object], vectors: list[list[int]]) -> list[float]:
    contrast = model["contrast"]
    return [score_active(contrast, float(model["contrast_sum"]), active) for active in vectors]


def label_shuffle_nulls(
    train_vectors: list[list[int]],
    train_labels: list[int],
    feature_count: int,
    test_vectors: list[list[int]],
    test_labels: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    max_lift = 0.0
    max_f1 = 0.0
    for _ in range(iterations):
        labels = list(train_labels)
        rng.shuffle(labels)
        model = train_hebbian(train_vectors, labels, feature_count)
        scores = model_scores(model, test_vectors)
        metrics = threshold_metrics(scores, test_labels, float(model["threshold"]))
        max_lift = max(max_lift, float(metrics["lift"]))
        max_f1 = max(max_f1, float(metrics["f1"]))
    return {"label_shuffle_null_max_lift": max_lift, "label_shuffle_null_max_f1": max_f1}


def same_density_topk_null(labels: list[int], fraction: float, iterations: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    total = len(labels)
    positives = sum(labels)
    base_rate = positives / total if total else 0.0
    k = max(1, int(round(total * fraction)))
    positive_indices = {index for index, label in enumerate(labels) if label}
    max_lift = 0.0
    max_precision = 0.0
    for _ in range(iterations):
        selected = set(rng.sample(range(total), k))
        hits = len(selected & positive_indices)
        precision = hits / k
        lift = precision / base_rate if base_rate else 0.0
        max_lift = max(max_lift, lift)
        max_precision = max(max_precision, precision)
    return {f"same_density_top{int(fraction * 100)}pct_null_max_lift": max_lift, f"same_density_top{int(fraction * 100)}pct_null_max_precision": max_precision}


def verdict_for(
    train_positive: int,
    test_positive: int,
    observed: dict[str, float | int],
    top1: dict[str, float | int],
    label_null: dict[str, float],
    density_null: dict[str, float],
    min_positive: int,
) -> str:
    if train_positive < min_positive:
        return "LOW_TRAIN_SUPPORT"
    if test_positive < min_positive:
        return "LOW_TEST_SUPPORT"
    if (
        float(observed["lift"]) > label_null["label_shuffle_null_max_lift"]
        and float(observed["f1"]) > label_null["label_shuffle_null_max_f1"]
        and float(top1["lift"]) > density_null["same_density_top1pct_null_max_lift"]
        and int(top1["hits"]) >= min_positive
    ):
        return "PERSISTENT_SCALE_SIGNAL"
    if float(top1["lift"]) > density_null["same_density_top1pct_null_max_lift"] and int(top1["hits"]) >= min_positive:
        return "RANK_ONLY_SCALE_SIGNAL"
    return "WEAK_OR_NULL"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [piece.strip() for piece in args.fields.split(",") if piece.strip()]
    moduli = [int(piece.strip()) for piece in args.moduli.split(",") if piece.strip()]
    targets = [piece.strip() for piece in args.targets.split(",") if piece.strip()]
    selected_feature_sets = feature_sets(moduli, args.feature_sets)

    train_rows = [qa_values(b, e) for b, e in square_pair_iter(1, 100)]
    train_labels_by_target = {target: [label_for(row, target) for row in train_rows] for target in targets}
    train_positive_by_target = {target: sum(labels) for target, labels in train_labels_by_target.items()}

    leaderboard: list[dict[str, object]] = []
    for feature_set_index, (feature_set_name, feature_moduli) in enumerate(selected_feature_sets.items()):
        train_vectors, feature_count = feature_matrix(train_rows, fields, feature_moduli)
        models: dict[str, dict[str, object] | None] = {}
        for target in targets:
            labels = train_labels_by_target[target]
            positive = train_positive_by_target[target]
            if positive < args.min_positive or len(labels) - positive < args.min_positive:
                models[target] = None
            else:
                models[target] = train_hebbian(train_vectors, labels, feature_count)

        for window_index, window in enumerate(build_windows(args)):
            if window["name"] == "random_sparse_1e6":
                pairs = random_sparse_pairs(1_000_000, args.random_count, args.seed + 100000)
                sampled = True
            else:
                pairs, sampled = sample_pair_iter(
                    window["pairs"](),
                    int(window["total"]),
                    int(window["cap"]),
                    args.seed + window_index,
                )
            test_rows = [qa_values(b, e) for b, e in pairs]
            test_vectors, _ = feature_matrix(test_rows, fields, feature_moduli)

            for target_index, target in enumerate(targets):
                train_positive = train_positive_by_target[target]
                test_labels = [label_for(row, target) for row in test_rows]
                test_positive = sum(test_labels)
                model = models[target]
                if model is None or train_positive < args.min_positive:
                    observed = {
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "lift": None,
                        "base_rate": None,
                        "predicted_positive": None,
                    }
                    ap = None
                    top1 = {"precision": None, "lift": None, "hits": None}
                    top5 = {"precision": None, "lift": None, "hits": None}
                    label_null = {"label_shuffle_null_max_lift": None, "label_shuffle_null_max_f1": None}
                    density_top1 = {"same_density_top1pct_null_max_lift": None, "same_density_top1pct_null_max_precision": None}
                    density_top5 = {"same_density_top5pct_null_max_lift": None, "same_density_top5pct_null_max_precision": None}
                    verdict = "LOW_TRAIN_SUPPORT"
                elif test_positive < args.min_positive:
                    observed = {
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "lift": None,
                        "base_rate": test_positive / len(test_labels) if test_labels else 0.0,
                        "predicted_positive": None,
                    }
                    ap = None
                    top1 = {"precision": None, "lift": None, "hits": None}
                    top5 = {"precision": None, "lift": None, "hits": None}
                    label_null = {"label_shuffle_null_max_lift": None, "label_shuffle_null_max_f1": None}
                    density_top1 = {"same_density_top1pct_null_max_lift": None, "same_density_top1pct_null_max_precision": None}
                    density_top5 = {"same_density_top5pct_null_max_lift": None, "same_density_top5pct_null_max_precision": None}
                    verdict = "LOW_TEST_SUPPORT"
                else:
                    scores = model_scores(model, test_vectors)
                    observed = threshold_metrics(scores, test_labels, float(model["threshold"]))
                    ap = average_precision(scores, test_labels)
                    top1 = top_fraction_metrics(scores, test_labels, 0.01)
                    top5 = top_fraction_metrics(scores, test_labels, 0.05)
                    seed_base = args.seed + feature_set_index * 100000 + window_index * 10000 + target_index * 100
                    label_null = label_shuffle_nulls(
                        train_vectors,
                        train_labels_by_target[target],
                        feature_count,
                        test_vectors,
                        test_labels,
                        args.null_iterations,
                        seed_base,
                    )
                    density_top1 = same_density_topk_null(test_labels, 0.01, args.null_iterations, seed_base + 1)
                    density_top5 = same_density_topk_null(test_labels, 0.05, args.null_iterations, seed_base + 2)
                    verdict = verdict_for(
                        train_positive,
                        test_positive,
                        observed,
                        top1,
                        label_null,
                        density_top1,
                        args.min_positive,
                    )
                row_payload = {
                    "target": target,
                    "train_window": "square_1_100",
                    "test_window": window["name"],
                    "feature_set": feature_set_name,
                    "model": "hebbian_residue_prototype",
                    "observed_lift": observed["lift"],
                    "null_max_lift": label_null["label_shuffle_null_max_lift"],
                    "top1pct_lift": top1["lift"],
                    "same_density_top1pct_null_max_lift": density_top1["same_density_top1pct_null_max_lift"],
                    "verdict": verdict,
                }
                leaderboard.append(
                    {
                        **row_payload,
                        "hash": domain_sha256(f"{DOMAIN}.leaderboard", canonical_json(row_payload)),
                        "train_rows": len(train_rows),
                        "test_rows": len(test_rows),
                        "sampled": sampled,
                        "train_positive_rows": train_positive,
                        "test_positive_rows": test_positive,
                        "test_base_rate": observed["base_rate"],
                        "precision": observed["precision"],
                        "recall": observed["recall"],
                        "f1": observed["f1"],
                        "average_precision": ap,
                        "predicted_positive_rows": observed["predicted_positive"],
                        "top1pct_precision": top1["precision"],
                        "top1pct_hits": top1["hits"],
                        "top5pct_precision": top5["precision"],
                        "top5pct_lift": top5["lift"],
                        "top5pct_hits": top5["hits"],
                        "label_shuffle_null_max_f1": label_null["label_shuffle_null_max_f1"],
                        "same_density_top1pct_null_max_precision": density_top1[
                            "same_density_top1pct_null_max_precision"
                        ],
                        "same_density_top5pct_null_max_lift": density_top5[
                            "same_density_top5pct_null_max_lift"
                        ],
                        "same_density_top5pct_null_max_precision": density_top5[
                            "same_density_top5pct_null_max_precision"
                        ],
                    }
                )

    leaderboard.sort(
        key=lambda row: (
            1 if row["verdict"] == "PERSISTENT_SCALE_SIGNAL" else 0,
            float(row["top1pct_lift"] or 0.0) - float(row["same_density_top1pct_null_max_lift"] or 0.0),
            float(row["observed_lift"] or 0.0) - float(row["null_max_lift"] or 0.0),
        ),
        reverse=True,
    )
    csv_path = out_dir / args.leaderboard_csv
    write_csv(csv_path, leaderboard)
    verdict_counts: dict[str, int] = {}
    for row in leaderboard:
        verdict_counts[str(row["verdict"])] = verdict_counts.get(str(row["verdict"]), 0) + 1
    payload = {
        "stage_id": args.stage_id,
        "hypothesis": (
            "Strong Stage 8 QA arithmetic target families should retain above-null enrichment across larger "
            "unseen windows, while fragile conic and smoothness targets may degrade with scale."
        ),
        "parameters": {
            "fields": fields,
            "moduli": moduli,
            "targets": targets,
            "feature_sets": list(selected_feature_sets.keys()),
            "sample_cap": args.sample_cap,
            "random_count": args.random_count,
            "null_iterations": args.null_iterations,
            "min_positive": args.min_positive,
            "seed": args.seed,
        },
        "train": {
            "window": "square_1_100",
            "rows": len(train_rows),
            "positive_rows_by_target": train_positive_by_target,
        },
        "windows": [window["name"] for window in build_windows(args)],
        "artifacts": {"leaderboard_csv": str(csv_path)},
        "verdict_counts": verdict_counts,
        "leaderboard": leaderboard,
        "honest_interpretation": (
            "Stage 9 is a cross-scale empirical confirmation pass. PERSISTENT_SCALE_SIGNAL requires threshold lift "
            "and F1 to beat shuffled-label nulls and top-1-percent lift to beat same-density random top-k controls."
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
            fields="b,e,d,a",
            moduli="2,3,4,5",
            targets="X_semiprime,F_semiprime,G_square,h_integer,directrix_distance_integer",
            feature_sets="full_residue,no_parity",
            sample_cap=200,
            random_count=200,
            null_iterations=2,
            min_positive=1,
            seed=91,
            summary_json="stage9_selftest.json",
            leaderboard_csv="stage9_selftest.csv",
            stage_id="qa_quantum_arithmetic_cross_scale_confirmation_stage9_selftest",
        )
        payload = run(args)
        csv_path = Path(tmp) / "stage9_selftest.csv"
        json_path = Path(tmp) / "stage9_selftest.json"
        assert csv_path.exists()
        assert json_path.exists()
        assert payload["verdict_counts"]
        assert payload["canonical_hash"] == domain_sha256(
            DOMAIN, canonical_json({key: value for key, value in payload.items() if key != "canonical_hash"})
        )
        return {"ok": True, "rows": len(payload["leaderboard"]), "verdict_counts": payload["verdict_counts"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--fields", default="b,e,d,a")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13,16,17,19,24")
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--feature-sets", default="all")
    parser.add_argument("--sample-cap", type=int, default=50000)
    parser.add_argument("--random-count", type=int, default=50000)
    parser.add_argument("--null-iterations", type=int, default=20)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage9_cross_scale.json")
    parser.add_argument("--leaderboard-csv", default="qa_quantum_arithmetic_stage9_cross_scale_leaderboard.csv")
    parser.add_argument("--stage-id", default="qa_quantum_arithmetic_cross_scale_confirmation_stage9")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(canonical_json(self_test()))
        return
    payload = run(args)
    print(
        canonical_json(
            {
                "ok": True,
                "stage_id": payload["stage_id"],
                "verdict_counts": payload["verdict_counts"],
                "artifacts": payload["artifacts"],
                "canonical_hash": payload["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
