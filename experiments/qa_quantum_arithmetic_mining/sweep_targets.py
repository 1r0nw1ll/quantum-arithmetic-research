#!/usr/bin/env python3
"""Stage 7 target sweep for QA arithmetic coordinate-residue structure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Iterator


DOMAIN = "QA_QUANTUM_ARITHMETIC_SWEEP_TARGETS_STAGE7.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def factor_count(n: int) -> int:
    count = 0
    temp = n
    while temp % 2 == 0 and temp > 1:
        count += 1
        temp //= 2
    divisor = 3
    while divisor <= math.isqrt(temp):
        while temp % divisor == 0:
            count += 1
            temp //= divisor
        divisor += 2
    if temp > 1:
        count += 1
    return count


def prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    temp = n
    while temp % 2 == 0 and temp > 1:
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
    return factors


def distinct_factor_count(n: int) -> int:
    return len(set(prime_factors(n)))


def is_semiprime(n: int) -> bool:
    return factor_count(n) == 2


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    divisor = 3
    limit = math.isqrt(n)
    while divisor <= limit:
        if n % divisor == 0:
            return False
        divisor += 2
    return True


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def is_squarefree(n: int) -> bool:
    factors = prime_factors(n)
    return len(factors) == len(set(factors))


def is_smooth(n: int, bound: int) -> bool:
    factors = prime_factors(n)
    return bool(factors) and max(factors) <= bound


def mobius(n: int) -> int:
    factors = prime_factors(n)
    if len(factors) != len(set(factors)):
        return 0
    return 1 if len(factors) % 2 == 0 else -1


def liouville(n: int) -> int:
    return 1 if factor_count(n) % 2 == 0 else -1


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = e + d
    D = d * d
    X = e * d
    F = a * b
    G = D + e * e
    W = d * (e + a)
    R = b * e
    J = d * b
    K = d * a
    EA = e * a
    P = 2 * K
    F_gap = d - math.isqrt(F)
    X_gap2 = a - math.isqrt(4 * X)
    W_gap = a - math.isqrt(W)
    R_gap2 = d - math.isqrt(4 * R)
    J_gap2 = d + b - math.isqrt(4 * J)
    K_gap2 = d + a - math.isqrt(4 * K)
    EA_gap2 = e + a - math.isqrt(4 * EA)
    return {
        "b": b,
        "e": e,
        "d": d,
        "a": a,
        "D": D,
        "X": X,
        "F": F,
        "G": G,
        "W": W,
        "R": R,
        "J": J,
        "K": K,
        "EA": EA,
        "P": P,
        "F_gap": F_gap,
        "X_gap2": X_gap2,
        "W_gap": W_gap,
        "R_gap2": R_gap2,
        "J_gap2": J_gap2,
        "K_gap2": K_gap2,
        "EA_gap2": EA_gap2,
    }


def primitive_triangle(row: dict[str, int]) -> bool:
    return math.gcd(row["d"], row["e"]) == 1 and (row["d"] - row["e"]) % 2 != 0


def reduced_eccentricity_parts(row: dict[str, int]) -> tuple[int, int]:
    divisor = math.gcd(row["e"], row["d"])
    return row["e"] // divisor, row["d"] // divisor


def reduced_axis_ratio_parts(row: dict[str, int]) -> tuple[int, int]:
    divisor = math.gcd(row["F"], row["D"])
    return row["F"] // divisor, row["D"] // divisor


def reduced_eccentricity_square_parts(row: dict[str, int]) -> tuple[int, int]:
    numerator = row["e"] * row["e"]
    denominator = row["D"]
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def same_focus_distance_bucket(row: dict[str, int]) -> int:
    return min(5, factor_count(row["X"]))


def director_radius_sq(row: dict[str, int]) -> int:
    return row["D"] * (row["D"] + row["F"])


def evolute_gap(row: dict[str, int]) -> int:
    return row["D"] - row["F"]


def directrix_distance(row: dict[str, int]) -> int | None:
    numerator = row["d"] * row["d"] * row["d"]
    if numerator % row["e"] != 0:
        return None
    return numerator // row["e"]


def focus_to_directrix_distance(row: dict[str, int]) -> int | None:
    distance = directrix_distance(row)
    if distance is None:
        return None
    return distance - row["X"]


def label_for(row: dict[str, int], target: str) -> int:
    X = row["X"]
    if target in {"full_latus_smooth_13", "F_latus_rectum_smooth_13"}:
        return int(is_smooth(2 * row["F"], 13))
    if target in {"full_latus_squarefree", "F_latus_rectum_squarefree"}:
        return int(is_squarefree(2 * row["F"]))
    if target in {"semi_latus_squarefree", "F_latus_semirectum_squarefree"}:
        return int(is_squarefree(row["F"]))
    if target in {"semi_latus_distinct_omega_2", "F_latus_semirectum_distinct_omega_2"}:
        return int(distinct_factor_count(row["F"]) == 2)
    if target in {"full_latus_omega_4", "F_latus_rectum_omega_4"}:
        return int(factor_count(2 * row["F"]) == 4)
    if target == "axis_ratio_reduced_den_semiprime":
        return int(is_semiprime(reduced_axis_ratio_parts(row)[1]))
    if target == "axis_ratio_reduced_num_semiprime":
        return int(is_semiprime(reduced_axis_ratio_parts(row)[0]))
    if target == "eccentricity_square_num_squarefree":
        return int(is_squarefree(reduced_eccentricity_square_parts(row)[0]))
    if target == "eccentricity_square_den_squarefree":
        return int(is_squarefree(reduced_eccentricity_square_parts(row)[1]))
    if target == "F_divides_D":
        return int(row["D"] % row["F"] == 0)
    if target == "D_divides_F":
        return int(row["F"] % row["D"] == 0)
    if target == "same_focus_distance_bucket":
        return int(same_focus_distance_bucket(row) == 2)
    if target == "gcd_X_D_eq_D":
        return int(math.gcd(row["X"], row["D"]) == row["D"])
    if target == "gcd_X_D_eq_d":
        return int(math.gcd(row["X"], row["D"]) == row["d"])
    if target == "gcd_X_D_gt_d":
        return int(math.gcd(row["X"], row["D"]) > row["d"])
    if target == "gcd_X_F_eq_1":
        return int(math.gcd(row["X"], row["F"]) == 1)
    if target == "gcd_X_F_gt_1":
        return int(math.gcd(row["X"], row["F"]) > 1)
    if target == "director_radius_sq_square":
        return int(is_square(director_radius_sq(row)))
    if target == "director_radius_sq_semiprime":
        return int(is_semiprime(director_radius_sq(row)))
    if target == "director_radius_sq_squarefree":
        return int(is_squarefree(director_radius_sq(row)))
    if target == "director_radius_factor_D_plus_F_semiprime":
        return int(is_semiprime(row["D"] + row["F"]))
    if target == "director_factor_D_plus_F_semiprime":
        return int(is_semiprime(row["D"] + row["F"]))
    if target == "director_factor_D_plus_F_squarefree":
        return int(is_squarefree(row["D"] + row["F"]))
    if target == "D_plus_F_semiprime":
        return int(is_semiprime(row["D"] + row["F"]))
    if target == "D_plus_F_squarefree":
        return int(is_squarefree(row["D"] + row["F"]))
    if target == "polar_scale_D_plus_F_semiprime":
        return int(is_semiprime(row["D"] + row["F"]))
    if target == "polar_scale_D_minus_F_square":
        return int(is_square(evolute_gap(row)))
    if target == "polar_scale_X_plus_F_semiprime":
        return int(is_semiprime(row["X"] + row["F"]))
    if target == "evolute_gap_E_semiprime":
        return int(is_semiprime(row["e"] * row["e"]))
    if target == "evolute_gap_E_square":
        return int(is_square(row["e"] * row["e"]))
    if target == "evolute_gap_E_squarefree":
        return int(is_squarefree(row["e"] * row["e"]))
    if target == "D_minus_F_square":
        return int(is_square(evolute_gap(row)))
    if target == "D_minus_F_semiprime":
        return int(is_semiprime(evolute_gap(row)))
    if target == "eccentricity_den_smooth_13":
        return int(is_smooth(reduced_eccentricity_parts(row)[1], 13))
    if target == "ecc_den_smooth_13":
        return int(is_smooth(reduced_eccentricity_parts(row)[1], 13))
    if target == "eccentricity_num_smooth_13":
        return int(is_smooth(reduced_eccentricity_parts(row)[0], 13))
    if target == "directrix_distance_integer":
        return int(directrix_distance(row) is not None)
    if target == "directrix_distance_semiprime":
        distance = directrix_distance(row)
        return int(distance is not None and is_semiprime(distance))
    if target == "directrix_gap_integer":
        return int(focus_to_directrix_distance(row) is not None)
    if target == "focus_to_directrix_distance_integer":
        return int(focus_to_directrix_distance(row) is not None)
    if target == "focus_to_directrix_distance_semiprime":
        distance = focus_to_directrix_distance(row)
        return int(distance is not None and distance > 0 and is_semiprime(distance))
    if "_smooth_" in target:
        field, raw_bound = target.split("_smooth_", 1)
        aliases = {
            "F_latus_semirectum": "F",
            "R": "R",
            "R_inradius": "R",
            "K": "K",
            "K_semiperimeter": "K",
        }
        field_name = aliases.get(field, field)
        if field_name not in row:
            raise ValueError(f"unknown smoothness field in target: {target}")
        return int(is_smooth(row[field_name], int(raw_bound)))
    if target == "X_semiprime":
        return int(is_semiprime(X))
    if target == "X_focus_distance_semiprime":
        return int(is_semiprime(X))
    if target == "F_semiprime":
        return int(is_semiprime(row["F"]))
    if target == "F_latus_semirectum_semiprime":
        return int(is_semiprime(row["F"]))
    if target == "F_latus_semirectum_omega_3":
        return int(factor_count(row["F"]) == 3)
    if target == "F_latus_semirectum_smooth_13":
        return int(is_smooth(row["F"], 13))
    if target == "F_latus_rectum_omega_3":
        return int(factor_count(2 * row["F"]) == 3)
    if target == "F_omega_2":
        return int(factor_count(row["F"]) == 2)
    if target == "F_omega_3":
        return int(factor_count(row["F"]) == 3)
    if target == "F_omega_4":
        return int(factor_count(row["F"]) == 4)
    if target == "W_semiprime":
        return int(is_semiprime(row["W"]))
    if target == "W_omega_2":
        return int(factor_count(row["W"]) == 2)
    if target == "W_omega_3":
        return int(factor_count(row["W"]) == 3)
    if target == "W_omega_4":
        return int(factor_count(row["W"]) == 4)
    if target == "G_square":
        return int(is_square(row["G"]))
    if target == "D_apex_distance_square":
        return int(is_square(row["D"]))
    if target == "D_apex_distance_semiprime":
        return int(is_semiprime(row["D"]))
    if target == "D_plus_X_semiprime":
        return int(is_semiprime(row["D"] + row["X"]))
    if target == "D_minus_X_semiprime":
        return int(is_semiprime(row["D"] - row["X"]))
    if target == "gcd_D_X_eq_1":
        return int(math.gcd(row["D"], row["X"]) == 1)
    if target == "gcd_b_e_eq_1":
        return int(math.gcd(row["b"], row["e"]) == 1)
    if target == "gcd_e_d_eq_1":
        return int(math.gcd(row["e"], row["d"]) == 1)
    if target == "gcd_d_e_eq_1":
        return int(math.gcd(row["d"], row["e"]) == 1)
    if target == "gcd_d_a_eq_1":
        return int(math.gcd(row["d"], row["a"]) == 1)
    if target == "primitive_triangle_condition":
        return int(primitive_triangle(row))
    if target == "G_square_and_primitive":
        return int(is_square(row["G"]) and primitive_triangle(row))
    if target == "G_hypotenuse_of_primitive_triple":
        return int(is_square(row["G"]) and primitive_triangle(row))
    if target == "h_integer":
        return int(is_square(row["F"]))
    if target == "h_integer_and_F_semiprime":
        return int(is_square(row["F"]) and is_semiprime(row["F"]))
    if target == "G_square_and_h_integer":
        return int(is_square(row["G"]) and is_square(row["F"]))
    if target == "eccentricity_reduced_den_prime":
        return int(is_prime(reduced_eccentricity_parts(row)[1]))
    if target == "eccentricity_reduced_den_semiprime":
        return int(is_semiprime(reduced_eccentricity_parts(row)[1]))
    if target == "ecc_den_semiprime":
        return int(is_semiprime(reduced_eccentricity_parts(row)[1]))
    if target == "eccentricity_reduced_num_prime":
        return int(is_prime(reduced_eccentricity_parts(row)[0]))
    if target == "eccentricity_reduced_num_semiprime":
        return int(is_semiprime(reduced_eccentricity_parts(row)[0]))
    if target == "ecc_num_semiprime":
        return int(is_semiprime(reduced_eccentricity_parts(row)[0]))
    if target == "h_integer_and_G_square":
        return int(is_square(row["F"]) and is_square(row["G"]))
    if target == "X_squarefree":
        return int(is_squarefree(X))
    if target == "X_distinct_omega_1":
        return int(distinct_factor_count(X) == 1)
    if target == "X_distinct_omega_2":
        return int(distinct_factor_count(X) == 2)
    if target == "X_distinct_omega_3":
        return int(distinct_factor_count(X) == 3)
    if target == "X_mobius_zero":
        return int(mobius(X) == 0)
    if target == "X_mobius_positive":
        return int(mobius(X) == 1)
    if target == "X_mobius_negative":
        return int(mobius(X) == -1)
    if target == "X_liouville_positive":
        return int(liouville(X) == 1)
    if target == "F_distinct_omega_2":
        return int(distinct_factor_count(row["F"]) == 2)
    if target == "F_mobius_zero":
        return int(mobius(row["F"]) == 0)
    if target == "F_liouville_positive":
        return int(liouville(row["F"]) == 1)
    if target == "W_distinct_omega_2":
        return int(distinct_factor_count(row["W"]) == 2)
    if target == "W_mobius_zero":
        return int(mobius(row["W"]) == 0)
    if target == "W_liouville_positive":
        return int(liouville(row["W"]) == 1)
    if target == "X_omega_2":
        return int(factor_count(X) == 2)
    if target == "X_omega_3":
        return int(factor_count(X) == 3)
    if target == "X_omega_4":
        return int(factor_count(X) == 4)
    if target == "R_inradius_semiprime":
        return int(is_semiprime(row["R"]))
    if target == "R_squarefree":
        return int(is_squarefree(row["R"]))
    if target == "R_inradius_omega_2":
        return int(factor_count(row["R"]) == 2)
    if target == "R_inradius_omega_3":
        return int(factor_count(row["R"]) == 3)
    if target == "J_semiprime" or target == "J_exradius_semiprime":
        return int(is_semiprime(row["J"]))
    if target == "J_squarefree":
        return int(is_squarefree(row["J"]))
    if target == "K_semiperimeter_semiprime":
        return int(is_semiprime(row["K"]))
    if target == "K_squarefree":
        return int(is_squarefree(row["K"]))
    if target == "EA_semiprime":
        return int(is_semiprime(row["EA"]))
    if target == "K_omega_2":
        return int(factor_count(row["K"]) == 2)
    if target == "K_omega_3":
        return int(factor_count(row["K"]) == 3)
    if target == "K_omega_4":
        return int(factor_count(row["K"]) == 4)
    if target == "K_distinct_omega_2":
        return int(distinct_factor_count(row["K"]) == 2)
    if target == "K_mobius_zero":
        return int(mobius(row["K"]) == 0)
    if target == "K_liouville_positive":
        return int(liouville(row["K"]) == 1)
    if target == "EA_exradius_semiprime":
        return int(is_semiprime(row["EA"]))
    if target == "EA_squarefree":
        return int(is_squarefree(row["EA"]))
    if target == "EA_omega_2":
        return int(factor_count(row["EA"]) == 2)
    if target == "EA_omega_3":
        return int(factor_count(row["EA"]) == 3)
    if target == "EA_distinct_omega_2":
        return int(distinct_factor_count(row["EA"]) == 2)
    if target == "EA_mobius_zero":
        return int(mobius(row["EA"]) == 0)
    if target == "EA_liouville_positive":
        return int(liouville(row["EA"]) == 1)
    if target == "R_distinct_omega_2":
        return int(distinct_factor_count(row["R"]) == 2)
    if target == "R_mobius_zero":
        return int(mobius(row["R"]) == 0)
    if target == "R_liouville_positive":
        return int(liouville(row["R"]) == 1)
    if target == "P_perimeter_omega_3":
        return int(factor_count(row["P"]) == 3)
    raise ValueError(f"unknown target: {target}")


def square_rows(start: int, end: int) -> Iterator[dict[str, int]]:
    for b in range(start, end + 1):
        for e in range(start, end + 1):
            yield qa_values(b, e)


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
        value = row[field]
        for modulus in moduli:
            active.append(offsets[(field, modulus)] + (value % modulus))
    return active


def feature_matrix(rows: list[dict[str, int]], fields: list[str], moduli: list[int]) -> tuple[list[list[int]], int]:
    offsets, feature_count = feature_offsets(fields, moduli)
    return [active_features(row, fields, moduli, offsets) for row in rows], feature_count


def score_predictions(predictions: list[int], truths: list[int]) -> dict[str, float | int]:
    tp = sum(1 for pred, truth in zip(predictions, truths) if pred == 1 and truth == 1)
    fp = sum(1 for pred, truth in zip(predictions, truths) if pred == 1 and truth == 0)
    fn = sum(1 for pred, truth in zip(predictions, truths) if pred == 0 and truth == 1)
    tn = sum(1 for pred, truth in zip(predictions, truths) if pred == 0 and truth == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    base_rate = (tp + fn) / len(truths) if truths else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "support": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
        "base_rate": base_rate,
    }


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


def train_hebbian_active(vectors: list[list[int]], labels: list[int], feature_count: int) -> dict[str, object]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("training labels must contain positive and negative rows")
    pos_counts = [0] * feature_count
    neg_counts = [0] * feature_count
    for vector, label in zip(vectors, labels):
        counts = pos_counts if label else neg_counts
        for index in vector:
            counts[index] += 1
    contrast = [(pos_counts[index] / positives) - (neg_counts[index] / negatives) for index in range(feature_count)]
    contrast_sum = sum(contrast)
    scores = [score_active(contrast, contrast_sum, vector) for vector in vectors]
    return {
        "contrast": contrast,
        "contrast_sum": contrast_sum,
        "threshold": best_threshold(scores, labels),
        "train_positive_rows": positives,
        "train_negative_rows": negatives,
        "feature_count": feature_count,
    }


def score_active(contrast: list[float], contrast_sum: float, vector: list[int]) -> float:
    active_sum = sum(contrast[index] for index in vector)
    return (2.0 * active_sum) - contrast_sum


def evaluate_model(model: dict[str, object], vectors: list[list[int]], labels: list[int]) -> dict[str, float | int]:
    contrast = model["contrast"]
    contrast_sum = float(model["contrast_sum"])
    threshold = float(model["threshold"])
    predictions = [int(score_active(contrast, contrast_sum, vector) >= threshold) for vector in vectors]
    return score_predictions(predictions, labels)


def null_summary(
    train_vectors: list[list[int]],
    train_labels: list[int],
    feature_count: int,
    test_vectors: list[list[int]],
    test_labels: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    rows = []
    for _ in range(iterations):
        shuffled = list(train_labels)
        rng.shuffle(shuffled)
        model = train_hebbian_active(train_vectors, shuffled, feature_count)
        rows.append(evaluate_model(model, test_vectors, test_labels))
    out = {}
    for key in ["precision", "recall", "f1", "lift"]:
        values = [float(row[key]) for row in rows]
        out[f"null_{key}_mean"] = sum(values) / len(values) if values else 0.0
        out[f"null_{key}_max"] = max(values) if values else 0.0
    return out


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
    feature_sets = {}
    if args.feature_sets in {"full_residue", "all"}:
        feature_sets["full_residue"] = moduli
    if args.feature_sets in {"no_parity", "all"}:
        feature_sets["no_parity"] = [modulus for modulus in moduli if modulus != 2]
    if args.feature_sets in {"parity_only", "all"}:
        feature_sets["parity_only"] = [2]
    if args.feature_sets in {"fermat_only", "fermat"}:
        feature_sets["fermat_only"] = moduli
    if args.feature_sets in {"residue_plus_fermat", "fermat"}:
        feature_sets["residue_plus_fermat"] = moduli

    train_rows = list(square_rows(args.train_start, args.train_end))
    test_rows = list(square_rows(args.test_start, args.test_end))
    leaderboard = []
    for feature_set_index, (feature_set_name, feature_moduli) in enumerate(feature_sets.items()):
        if feature_set_name == "fermat_only":
            feature_fields = ["F_gap", "X_gap2", "W_gap", "R_gap2", "J_gap2", "K_gap2", "EA_gap2"]
        elif feature_set_name == "residue_plus_fermat":
            feature_fields = fields + ["F_gap", "X_gap2", "W_gap", "R_gap2", "J_gap2", "K_gap2", "EA_gap2"]
        else:
            feature_fields = fields
        train_vectors, feature_count = feature_matrix(train_rows, feature_fields, feature_moduli)
        test_vectors, _ = feature_matrix(test_rows, feature_fields, feature_moduli)
        for target_index, target in enumerate(targets):
            train_labels = [label_for(row, target) for row in train_rows]
            test_labels = [label_for(row, target) for row in test_rows]
            train_positive = sum(train_labels)
            test_positive = sum(test_labels)
            if train_positive < args.min_positive or len(train_labels) - train_positive < args.min_positive:
                observed = {"precision": None, "recall": None, "f1": None, "lift": None, "base_rate": None}
                controls = {"null_f1_max": None, "null_lift_max": None, "null_f1_mean": None, "null_lift_mean": None}
                verdict = "LOW_TRAIN_SUPPORT"
            elif test_positive < args.min_positive:
                observed = {"precision": None, "recall": None, "f1": None, "lift": None, "base_rate": test_positive / len(test_labels)}
                controls = {"null_f1_max": None, "null_lift_max": None, "null_f1_mean": None, "null_lift_mean": None}
                verdict = "LOW_TEST_SUPPORT"
            else:
                model = train_hebbian_active(train_vectors, train_labels, feature_count)
                observed = evaluate_model(model, test_vectors, test_labels)
                controls = null_summary(
                    train_vectors,
                    train_labels,
                    feature_count,
                    test_vectors,
                    test_labels,
                    args.null_iterations,
                    args.seed + feature_set_index * 10000 + target_index * 1000,
                )
                verdict = (
                    "PERSISTENT_LIFT"
                    if float(observed["lift"]) > float(controls["null_lift_max"])
                    and float(observed["f1"]) > float(controls["null_f1_max"])
                    else "NULL_COMPETITIVE"
                )
            row_payload = {
                "target": target,
                "train_window": f"square_{args.train_start}_{args.train_end}",
                "test_window": f"square_{args.test_start}_{args.test_end}",
                "feature_set": feature_set_name,
                "model": "hebbian_prototype",
                "observed_lift": observed["lift"],
                "null_max_lift": controls["null_lift_max"],
                "verdict": verdict,
            }
            row_hash = domain_sha256(f"{DOMAIN}.leaderboard", canonical_json(row_payload))
            leaderboard.append(
                {
                    **row_payload,
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    "train_positive_rows": train_positive,
                    "test_positive_rows": test_positive,
                    "test_base_rate": observed["base_rate"],
                    "precision": observed["precision"],
                    "recall": observed["recall"],
                    "f1": observed["f1"],
                    "null_f1_mean": controls["null_f1_mean"],
                    "null_f1_max": controls["null_f1_max"],
                    "null_lift_mean": controls["null_lift_mean"],
                    "hash": row_hash,
                }
            )

    leaderboard.sort(
        key=lambda row: (
            float(row["observed_lift"] or 0.0) - float(row["null_max_lift"] or 0.0),
            float(row["observed_lift"] or 0.0),
        ),
        reverse=True,
    )
    csv_path = out_dir / args.leaderboard_csv
    write_csv(csv_path, leaderboard)
    payload = {
        "stage_id": "qa_quantum_arithmetic_sweep_targets_stage7",
        "hypothesis": (
            "A first serious multi-target sweep should show whether X_semiprime is special or whether coordinate "
            "residue Hebbian prototypes capture broader QA arithmetic structure."
        ),
        "parameters": {
            "train_window": f"square_{args.train_start}_{args.train_end}",
            "test_window": f"square_{args.test_start}_{args.test_end}",
            "fields": fields,
            "moduli": moduli,
            "feature_sets": list(feature_sets.keys()),
            "targets": targets,
            "null_iterations": args.null_iterations,
            "min_positive": args.min_positive,
            "seed": args.seed,
        },
        "artifacts": {"leaderboard_csv": str(csv_path)},
        "leaderboard": leaderboard,
        "persistent_lift_count": sum(1 for row in leaderboard if row["verdict"] == "PERSISTENT_LIFT"),
        "honest_interpretation": (
            "This sweep compares target enrichment against shuffled-label nulls on one strict external square. "
            "It ranks empirical persistence; it is not a theorem or factorization shortcut."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    summary_path = out_dir / args.summary_json
    summary_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            train_start=1,
            train_end=20,
            test_start=21,
            test_end=35,
            fields="b,e,d,a",
            moduli="2,3,4,5",
            targets=(
                "X_semiprime,F_semiprime,G_square,h_integer,X_squarefree,"
                "full_latus_smooth_13,semi_latus_squarefree,axis_ratio_reduced_den_semiprime,gcd_X_D_eq_d"
                ",director_radius_sq_square,D_minus_F_square"
                ",polar_scale_X_plus_F_semiprime"
            ),
            feature_sets="all",
            null_iterations=2,
            min_positive=1,
            seed=89,
            leaderboard_csv="qa_quantum_arithmetic_sweep_targets_leaderboard.csv",
            summary_json="qa_quantum_arithmetic_sweep_targets.json",
        )
        payload = run(args)
        ok = (
            len(payload["leaderboard"]) == 36
            and Path(payload["artifacts"]["leaderboard_csv"]).exists()
            and all(row["hash"] for row in payload["leaderboard"])
        )
        return {"ok": ok, "rows": len(payload["leaderboard"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 7 QA arithmetic multi-target Hebbian/null sweep.")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=100)
    parser.add_argument("--test-start", type=int, default=101)
    parser.add_argument("--test-end", type=int, default=300)
    parser.add_argument("--fields", default="b,e,d,a")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13,16,17,19,24")
    parser.add_argument(
        "--feature-sets",
        choices=["full_residue", "no_parity", "parity_only", "fermat_only", "residue_plus_fermat", "fermat", "all"],
        default="all",
    )
    parser.add_argument(
        "--targets",
        default=(
            "X_semiprime,F_semiprime,W_semiprime,G_square,h_integer,X_squarefree,X_omega_2,X_omega_3,"
            "R_inradius_semiprime,J_semiprime,K_semiperimeter_semiprime,EA_semiprime,K_omega_2,K_omega_3,"
            "P_perimeter_omega_3"
        ),
    )
    parser.add_argument("--null-iterations", type=int, default=20)
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--leaderboard-csv", default="qa_quantum_arithmetic_sweep_targets_stage7_leaderboard.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_sweep_targets_stage7.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(f"[qa_quantum_arithmetic_sweep_targets_stage7] wrote {payload['artifacts']['leaderboard_csv']}")
    print(f"[qa_quantum_arithmetic_sweep_targets_stage7] leaderboard_rows={len(payload['leaderboard'])}")
    print(f"[qa_quantum_arithmetic_sweep_targets_stage7] persistent_lift_count={payload['persistent_lift_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
