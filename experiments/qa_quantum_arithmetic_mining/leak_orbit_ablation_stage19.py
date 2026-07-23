#!/usr/bin/env python3
"""Stage 19 leak-corrected field and QA-orbit ablation for QA mining."""

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
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_orbit_rules import orbit_family, qa_step


DOMAIN = "QA_QUANTUM_ARITHMETIC_LEAK_ORBIT_ABLATION_STAGE19.v1"

DEFAULT_TARGETS = (
    "X_semiprime,F_semiprime,W_semiprime,R_inradius_semiprime,J_exradius_semiprime,"
    "K_semiperimeter_semiprime,EA_exradius_semiprime,G_square,h_integer,"
    "director_radius_sq_square,D_plus_F_square,D_plus_F_semiprime,D_plus_F_squarefree,"
    "semi_latus_squarefree,semi_latus_distinct_omega_2,directrix_distance_integer,"
    "ecc_den_smooth_13,polar_scale_X_plus_F_semiprime"
)

MODULI = [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 24]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def prime_factors(n: int) -> list[int]:
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
    return factors


def factor_count(n: int) -> int:
    return len(prime_factors(n))


def is_semiprime(n: int) -> bool:
    return factor_count(n) == 2


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def is_squarefree(n: int) -> bool:
    factors = prime_factors(n)
    return len(factors) == len(set(factors))


def is_smooth(n: int, bound: int) -> bool:
    factors = prime_factors(n)
    return bool(factors) and max(factors) <= bound


def distinct_factor_count(n: int) -> int:
    return len(set(prime_factors(n)))


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
    row = {
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
        "D_plus_F": D + F,
        "X_plus_F": X + F,
    }
    return row


def directrix_distance(row: dict[str, int]) -> int | None:
    numerator = row["d"] * row["d"] * row["d"]
    if numerator % row["e"] != 0:
        return None
    return numerator // row["e"]


def reduced_eccentricity_parts(row: dict[str, int]) -> tuple[int, int]:
    divisor = math.gcd(row["e"], row["d"])
    return row["e"] // divisor, row["d"] // divisor


def director_radius_sq(row: dict[str, int]) -> int:
    return row["D"] * row["D_plus_F"]


def label_for(row: dict[str, int], target: str) -> int:
    if target == "X_semiprime":
        return int(is_semiprime(row["X"]))
    if target == "F_semiprime":
        return int(is_semiprime(row["F"]))
    if target == "W_semiprime":
        return int(is_semiprime(row["W"]))
    if target == "R_inradius_semiprime":
        return int(is_semiprime(row["R"]))
    if target in {"J_semiprime", "J_exradius_semiprime"}:
        return int(is_semiprime(row["J"]))
    if target == "K_semiperimeter_semiprime":
        return int(is_semiprime(row["K"]))
    if target in {"EA_semiprime", "EA_exradius_semiprime"}:
        return int(is_semiprime(row["EA"]))
    if target == "G_square":
        return int(is_square(row["G"]))
    if target == "h_integer":
        return int(is_square(row["F"]))
    if target == "director_radius_sq_square":
        return int(is_square(director_radius_sq(row)))
    if target == "D_plus_F_square":
        return int(is_square(row["D_plus_F"]))
    if target in {"D_plus_F_semiprime", "director_factor_D_plus_F_semiprime"}:
        return int(is_semiprime(row["D_plus_F"]))
    if target in {"D_plus_F_squarefree", "director_factor_D_plus_F_squarefree"}:
        return int(is_squarefree(row["D_plus_F"]))
    if target == "semi_latus_squarefree":
        return int(is_squarefree(row["F"]))
    if target == "semi_latus_distinct_omega_2":
        return int(distinct_factor_count(row["F"]) == 2)
    if target == "directrix_distance_integer":
        return int(directrix_distance(row) is not None)
    if target == "ecc_den_smooth_13":
        return int(is_smooth(reduced_eccentricity_parts(row)[1], 13))
    if target == "polar_scale_X_plus_F_semiprime":
        return int(is_semiprime(row["X_plus_F"]))
    raise ValueError(f"unknown target: {target}")


def component_omega(row: dict[str, int], target: str) -> tuple[int, ...] | None:
    if target == "X_semiprime":
        return factor_count(row["e"]), factor_count(row["d"])
    if target == "F_semiprime":
        return factor_count(row["b"]), factor_count(row["a"])
    if target == "W_semiprime":
        return factor_count(row["d"]), factor_count(row["e"] + row["a"])
    if target == "R_inradius_semiprime":
        return factor_count(row["b"]), factor_count(row["e"])
    if target in {"J_semiprime", "J_exradius_semiprime"}:
        return factor_count(row["d"]), factor_count(row["b"])
    if target == "K_semiperimeter_semiprime":
        return factor_count(row["d"]), factor_count(row["a"])
    if target in {"EA_semiprime", "EA_exradius_semiprime"}:
        return factor_count(row["e"]), factor_count(row["a"])
    return None


def trivial_product_predict(row: dict[str, int], target: str) -> int | None:
    omegas = component_omega(row, target)
    if omegas is None:
        return None
    return int(sum(omegas) == 2)


def trivial_obstruction_predict(row: dict[str, int], target: str) -> int | None:
    if target in {"D_plus_F_square", "director_radius_sq_square"}:
        return int(row["b"] % 2 == 0)
    return None


def square_rows(start: int, end: int) -> Iterator[dict[str, int]]:
    for b in range(start, end + 1):
        for e in range(start, end + 1):
            yield qa_values(b, e)


def qa_mod_coord(value: int, modulus: int) -> int:
    return ((value - 1) % modulus) + 1


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


ORBIT_INDEX_9 = orbit_index_map(9)
ORBIT_FAMILY_CODE = {"cosmos": 0, "satellite": 1, "singularity": 2}


def row_feature_value(row: dict[str, int], field: str) -> int:
    if field == "orbit_family9":
        b9 = qa_mod_coord(row["b"], 9)
        e9 = qa_mod_coord(row["e"], 9)
        return ORBIT_FAMILY_CODE[orbit_family(int(b9), int(e9), 9)]
    if field == "orbit_id9":
        return ORBIT_INDEX_9[(qa_mod_coord(row["b"], 9), qa_mod_coord(row["e"], 9))]
    if field == "orbit_family24":
        b24 = qa_mod_coord(row["b"], 24)
        e24 = qa_mod_coord(row["e"], 24)
        return ORBIT_FAMILY_CODE[orbit_family(int(b24), int(e24), 24)]
    return row[field]


def feature_offsets(fields: list[str], moduli: list[int]) -> tuple[dict[tuple[str, int], int], int]:
    offsets: dict[tuple[str, int], int] = {}
    cursor = 0
    for field in fields:
        if field == "orbit_family9" or field == "orbit_family24":
            width = 3
        elif field == "orbit_id9":
            width = len(set(ORBIT_INDEX_9.values()))
        else:
            width = 0
        if width:
            offsets[(field, 0)] = cursor
            cursor += width
        else:
            for modulus in moduli:
                offsets[(field, modulus)] = cursor
                cursor += modulus
    return offsets, cursor


def active_features(row: dict[str, int], fields: list[str], moduli: list[int], offsets: dict[tuple[str, int], int]) -> list[int]:
    active = []
    for field in fields:
        value = row_feature_value(row, field)
        if field in {"orbit_family9", "orbit_family24", "orbit_id9"}:
            active.append(offsets[(field, 0)] + value)
        else:
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
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    base_rate = (tp + fn) / len(truths) if truths else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
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


def score_active(contrast: list[float], contrast_sum: float, vector: list[int]) -> float:
    active_sum = sum(contrast[index] for index in vector)
    return (2.0 * active_sum) - contrast_sum


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
        "feature_count": feature_count,
    }


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


def evaluate_trivial_product(test_rows: list[dict[str, int]], target: str, test_labels: list[int]) -> dict[str, float | int] | None:
    predictions = [trivial_product_predict(row, target) for row in test_rows]
    if any(prediction is None for prediction in predictions):
        return None
    return score_predictions([int(prediction) for prediction in predictions], test_labels)


def evaluate_trivial_obstruction(test_rows: list[dict[str, int]], target: str, test_labels: list[int]) -> dict[str, float | int] | None:
    predictions = [trivial_obstruction_predict(row, target) for row in test_rows]
    if any(prediction is None for prediction in predictions):
        return None
    return score_predictions([int(prediction) for prediction in predictions], test_labels)


def feature_set_fields() -> dict[str, list[str]]:
    return {
        "coordinate_full": ["b", "e", "d", "a"],
        "coordinate_only": ["b", "e"],
        "b_only": ["b"],
        "e_only": ["e"],
        "drop_raw_be": ["d", "a"],
        "derived_products": ["X", "F", "W", "R", "J", "K", "EA", "D_plus_F", "X_plus_F"],
        "qa_orbit_family9": ["orbit_family9"],
        "qa_orbit_id9": ["orbit_id9"],
        "qa_orbit_family24": ["orbit_family24"],
        "orbit9_plus_drop_be": ["orbit_id9", "d", "a"],
    }


def verdict_for(
    observed: dict[str, float | int],
    controls: dict[str, float | None],
    product_trivial: dict[str, float | int] | None,
    obstruction_trivial: dict[str, float | int] | None,
) -> str:
    lift = float(observed["lift"])
    f1 = float(observed["f1"])
    null_lift = float(controls["null_lift_max"] or 0.0)
    null_f1 = float(controls["null_f1_max"] or 0.0)
    if lift <= null_lift or f1 <= null_f1:
        return "NULL_COMPETITIVE"
    if product_trivial is not None and lift <= float(product_trivial["lift"]):
        return "BEATS_NULL_BUT_NOT_TRIVIAL_PRODUCT"
    if obstruction_trivial is not None and lift <= float(obstruction_trivial["lift"]):
        return "BEATS_NULL_BUT_NOT_TRIVIAL_OBSTRUCTION"
    return "PERSISTENT_AFTER_LEAK_BASELINE"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [piece.strip() for piece in args.targets.split(",") if piece.strip()]
    train_rows = list(square_rows(args.train_start, args.train_end))
    test_rows = list(square_rows(args.test_start, args.test_end))
    selected_feature_sets = feature_set_fields()
    rows = []
    for feature_set_index, (feature_set, fields) in enumerate(selected_feature_sets.items()):
        train_vectors, feature_count = feature_matrix(train_rows, fields, MODULI)
        test_vectors, _ = feature_matrix(test_rows, fields, MODULI)
        for target_index, target in enumerate(targets):
            train_labels = [label_for(row, target) for row in train_rows]
            test_labels = [label_for(row, target) for row in test_rows]
            train_positive = sum(train_labels)
            test_positive = sum(test_labels)
            product_trivial = evaluate_trivial_product(test_rows, target, test_labels)
            obstruction_trivial = evaluate_trivial_obstruction(test_rows, target, test_labels)
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
                verdict = verdict_for(observed, controls, product_trivial, obstruction_trivial)
            product_trivial_lift = None if product_trivial is None else product_trivial["lift"]
            obstruction_trivial_lift = None if obstruction_trivial is None else obstruction_trivial["lift"]
            strongest_trivial_lift = max(
                [float(value) for value in [product_trivial_lift, obstruction_trivial_lift] if value is not None],
                default=None,
            )
            row_payload = {
                "target": target,
                "train_window": f"square_{args.train_start}_{args.train_end}",
                "test_window": f"square_{args.test_start}_{args.test_end}",
                "feature_set": feature_set,
                "feature_fields": ",".join(fields),
                "model": "hebbian_prototype",
                "observed_lift": observed["lift"],
                "null_max_lift": controls["null_lift_max"],
                "trivial_product_lift": product_trivial_lift,
                "trivial_obstruction": "b_even" if obstruction_trivial is not None else "",
                "trivial_obstruction_lift": obstruction_trivial_lift,
                "strongest_trivial_lift": strongest_trivial_lift,
                "lift_over_trivial_product": (
                    None
                    if observed["lift"] is None or product_trivial_lift is None
                    else float(observed["lift"]) - float(product_trivial_lift)
                ),
                "lift_over_trivial_obstruction": (
                    None
                    if observed["lift"] is None or obstruction_trivial_lift is None
                    else float(observed["lift"]) - float(obstruction_trivial_lift)
                ),
                "lift_over_strongest_trivial": (
                    None
                    if observed["lift"] is None or strongest_trivial_lift is None
                    else float(observed["lift"]) - float(strongest_trivial_lift)
                ),
                "verdict": verdict,
            }
            row_hash = domain_sha256(f"{DOMAIN}.leaderboard", canonical_json(row_payload))
            rows.append(
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
                    "trivial_product_precision": None if product_trivial is None else product_trivial["precision"],
                    "trivial_product_recall": None if product_trivial is None else product_trivial["recall"],
                    "trivial_product_f1": None if product_trivial is None else product_trivial["f1"],
                    "trivial_obstruction_precision": None if obstruction_trivial is None else obstruction_trivial["precision"],
                    "trivial_obstruction_recall": None if obstruction_trivial is None else obstruction_trivial["recall"],
                    "trivial_obstruction_f1": None if obstruction_trivial is None else obstruction_trivial["f1"],
                    "hash": row_hash,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["target"]),
            float(row["observed_lift"] or 0.0) - float(row["null_max_lift"] or 0.0),
            float(row["observed_lift"] or 0.0),
        ),
        reverse=True,
    )
    csv_path = out_dir / args.leaderboard_csv
    write_csv(csv_path, rows)
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row["verdict"])
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    payload = {
        "stage_id": args.stage_id,
        "hypothesis": (
            "Dropping raw b/e coordinate residues and adding canonical QA orbit features can distinguish generic "
            "small-modulus sieve leakage from QA-specific orbit signal."
        ),
        "parameters": {
            "train_window": f"square_{args.train_start}_{args.train_end}",
            "test_window": f"square_{args.test_start}_{args.test_end}",
            "targets": targets,
            "feature_sets": selected_feature_sets,
            "moduli": MODULI,
            "null_iterations": args.null_iterations,
            "min_positive": args.min_positive,
            "seed": args.seed,
            "trivial_obstruction_baselines": {
                "D_plus_F_square": "b_even",
                "director_radius_sq_square": "b_even",
            },
        },
        "artifacts": {"leaderboard_csv": str(csv_path)},
        "leaderboard_rows": len(rows),
        "verdict_counts": verdict_counts,
        "top_orbit_only": [
            row
            for row in rows
            if str(row["feature_set"]).startswith("qa_orbit")
            and row["observed_lift"] is not None
        ][:20],
        "honest_interpretation": (
            "This is a leakage diagnostic. Product targets whose trivial component-Omega baseline matches or beats "
            "Hebbian lift should not be treated as evidence of new QA structure."
        ),
    }
    json_path = out_dir / args.summary_json
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    json_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            train_start=1,
            train_end=18,
            test_start=19,
            test_end=32,
            targets="X_semiprime,F_semiprime,D_plus_F_square,director_radius_sq_square,G_square",
            null_iterations=2,
            min_positive=1,
            seed=191,
            stage_id="qa_quantum_arithmetic_leak_orbit_ablation_stage19_selftest",
            leaderboard_csv="stage19_selftest_leaderboard.csv",
            summary_json="stage19_selftest.json",
        )
        payload = run(args)
        ok = (
            payload["leaderboard_rows"] == 50
            and Path(tmp, "stage19_selftest_leaderboard.csv").exists()
            and Path(tmp, "stage19_selftest.json").exists()
        )
        return {"ok": ok, "rows": payload["leaderboard_rows"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=100)
    parser.add_argument("--test-start", type=int, default=101)
    parser.add_argument("--test-end", type=int, default=300)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--null-iterations", type=int, default=20)
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--stage-id", default="qa_quantum_arithmetic_leak_orbit_ablation_stage19")
    parser.add_argument("--leaderboard-csv", default="qa_quantum_arithmetic_stage19_leak_orbit_ablation_leaderboard.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage19_leak_orbit_ablation.json")
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
                "leaderboard_rows": payload["leaderboard_rows"],
                "verdict_counts": payload["verdict_counts"],
                "artifacts": payload["artifacts"],
                "canonical_hash": payload["canonical_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
