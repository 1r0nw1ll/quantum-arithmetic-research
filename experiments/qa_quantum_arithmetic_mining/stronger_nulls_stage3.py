#!/usr/bin/env python3
"""Stage 3 stronger null controls for QA arithmetic residue mining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

from pattern_targets_stage2 import label_for, target_names
from scale_grid_stage1 import (
    build_windows,
    canonical_json,
    domain_sha256,
    qa_values,
    sample_pairs,
    square_pairs,
)


DOMAIN = "QA_QUANTUM_ARITHMETIC_STRONGER_NULLS_STAGE3.v1"


def metric_keys() -> list[str]:
    return ["precision", "recall", "f1", "lift"]


def summarize_runs(runs: list[dict[str, float | int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_keys():
        values = [float(run[key]) for run in runs]
        out[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
        out[f"{key}_max"] = max(values) if values else 0.0
    return out


def random_labels(count: int, positives: int, rng: random.Random) -> list[int]:
    labels = [1] * positives + [0] * (count - positives)
    rng.shuffle(labels)
    return labels


def coordinate_shuffle_rows(rows: list[dict[str, int]], rng: random.Random) -> list[dict[str, int]]:
    b_values = [row["b"] for row in rows]
    e_values = [row["e"] for row in rows]
    rng.shuffle(b_values)
    rng.shuffle(e_values)
    return [qa_values(b, e) for b, e in zip(b_values, e_values)]


def polynomial_labels(rows: list[dict[str, int]], rng: random.Random) -> list[int]:
    modulus = rng.choice([5, 7, 8, 9, 11, 13, 16, 17, 19, 24])
    residue = rng.randrange(modulus)
    coeffs = [rng.randint(1, modulus - 1) for _ in range(5)]
    labels = []
    for row in rows:
        value = (
            coeffs[0] * row["b"]
            + coeffs[1] * row["e"]
            + coeffs[2] * row["d"]
            + coeffs[3] * row["a"]
            + coeffs[4] * row["b"] * row["e"]
        )
        labels.append(int(value % modulus == residue))
    return labels


def block_offsets(fields: list[str], moduli: list[int]) -> tuple[dict[tuple[str, int], int], int]:
    offsets = {}
    cursor = 0
    for field in fields:
        for modulus in moduli:
            offsets[(field, modulus)] = cursor
            cursor += modulus
    return offsets, cursor


def active_feature_indices(row: dict[str, int], fields: list[str], moduli: list[int], offsets: dict[tuple[str, int], int]) -> list[int]:
    active = []
    for field in fields:
        value = row[field]
        for modulus in moduli:
            active.append(offsets[(field, modulus)] + (value % modulus))
    return active


def build_feature_matrix(rows: list[dict[str, int]], fields: list[str], moduli: list[int]) -> tuple[list[list[int]], int]:
    offsets, feature_count = block_offsets(fields, moduli)
    return [active_feature_indices(row, fields, moduli, offsets) for row in rows], feature_count


def train_hebbian_active(vectors: list[list[int]], labels: list[int], feature_count: int) -> dict[str, object]:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_sums = [-positive_count for _ in range(feature_count)]
    negative_sums = [-negative_count for _ in range(feature_count)]
    for active, label in zip(vectors, labels):
        sums = positive_sums if label == 1 else negative_sums
        for index in active:
            sums[index] += 2
    contrast = [
        (positive_sums[index] / positive_count) - (negative_sums[index] / negative_count)
        for index in range(feature_count)
    ]
    contrast_sum = sum(contrast)
    scores = [score_active(contrast, contrast_sum, active) for active in vectors]
    return {
        "contrast": contrast,
        "contrast_sum": contrast_sum,
        "threshold": best_threshold(scores, labels),
        "train_positive_rows": positive_count,
        "train_negative_rows": negative_count,
        "feature_count": feature_count,
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


def score_active(contrast: list[float], contrast_sum: float, active: list[int]) -> float:
    return -contrast_sum + 2 * sum(contrast[index] for index in active)


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


def evaluate_model_active(model: dict[str, object], vectors: list[list[int]], labels: list[int]) -> dict[str, float | int]:
    contrast = model["contrast"]
    contrast_sum = float(model["contrast_sum"])
    threshold = float(model["threshold"])
    predictions = [int(score_active(contrast, contrast_sum, active) >= threshold) for active in vectors]
    return score_predictions(predictions, labels)


def residue_column_shuffle(vectors: list[list[int]], block_count: int, rng: random.Random) -> list[list[int]]:
    if not vectors:
        return []
    columns = list(zip(*vectors))
    shuffled_columns = []
    for column in columns[:block_count]:
        values = list(column)
        rng.shuffle(values)
        shuffled_columns.append(values)
    return [list(row) for row in zip(*shuffled_columns)]


def null_family_metrics(
    family: str,
    train_rows: list[dict[str, int]],
    train_vectors: list[list[int]],
    feature_count: int,
    train_labels: list[int],
    test_rows: list[dict[str, int]],
    test_vectors: list[list[int]],
    test_labels: list[int],
    fields: list[str],
    moduli: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    runs = []
    for _ in range(iterations):
        if family == "label_shuffle":
            labels = list(train_labels)
            rng.shuffle(labels)
            model = train_hebbian_active(train_vectors, labels, feature_count)
            runs.append(evaluate_model_active(model, test_vectors, test_labels))
        elif family == "coordinate_shuffle":
            shuffled_rows = coordinate_shuffle_rows(train_rows, rng)
            shuffled_vectors, _ = build_feature_matrix(shuffled_rows, fields, moduli)
            model = train_hebbian_active(shuffled_vectors, train_labels, feature_count)
            runs.append(evaluate_model_active(model, test_vectors, test_labels))
        elif family == "residue_column_shuffle":
            shuffled_train_vectors = residue_column_shuffle(train_vectors, len(fields) * len(moduli), rng)
            model = train_hebbian_active(shuffled_train_vectors, train_labels, feature_count)
            runs.append(evaluate_model_active(model, test_vectors, test_labels))
        elif family == "same_density_random_positive":
            labels = random_labels(len(train_labels), sum(train_labels), rng)
            test_random = random_labels(len(test_labels), sum(test_labels), rng)
            model = train_hebbian_active(train_vectors, labels, feature_count)
            runs.append(evaluate_model_active(model, test_vectors, test_random))
        elif family == "random_polynomial_target":
            train_poly = polynomial_labels(train_rows, rng)
            test_poly = polynomial_labels(test_rows, rng)
            if min(sum(train_poly), len(train_poly) - sum(train_poly), sum(test_poly), len(test_poly) - sum(test_poly)) <= 0:
                continue
            model = train_hebbian_active(train_vectors, train_poly, feature_count)
            runs.append(evaluate_model_active(model, test_vectors, test_poly))
        else:
            raise ValueError(f"unknown null family: {family}")
    return summarize_runs(runs)


def evaluate_reversed_window(
    train_rows: list[dict[str, int]],
    test_rows: list[dict[str, int]],
    target: str,
    fields: list[str],
    moduli: list[int],
) -> dict[str, float | int] | None:
    reverse_train_labels = [label_for(row, target) for row in test_rows]
    if min(sum(reverse_train_labels), len(reverse_train_labels) - sum(reverse_train_labels)) <= 0:
        return None
    train_vectors, feature_count = build_feature_matrix(test_rows, fields, moduli)
    test_vectors, _ = build_feature_matrix(train_rows, fields, moduli)
    model = train_hebbian_active(train_vectors, reverse_train_labels, feature_count)
    return evaluate_model_active(model, test_vectors, [label_for(row, target) for row in train_rows])


def ablation_metrics(
    train_rows: list[dict[str, int]],
    test_rows: list[dict[str, int]],
    target: str,
    fields: list[str],
    ablation_sets: dict[str, list[int]],
) -> dict[str, dict[str, float | int] | None]:
    out = {}
    train_labels = [label_for(row, target) for row in train_rows]
    test_labels = [label_for(row, target) for row in test_rows]
    for name, moduli in ablation_sets.items():
        if min(sum(train_labels), len(train_labels) - sum(train_labels), sum(test_labels), len(test_labels) - sum(test_labels)) <= 0:
            out[name] = None
            continue
        train_vectors, feature_count = build_feature_matrix(train_rows, fields, moduli)
        test_vectors, _ = build_feature_matrix(test_rows, fields, moduli)
        model = train_hebbian_active(train_vectors, train_labels, feature_count)
        out[name] = evaluate_model_active(model, test_vectors, test_labels)
    return out


def verdict(observed: dict[str, float | int], nulls: dict[str, dict[str, float]], ablations: dict[str, dict[str, float | int] | None]) -> str:
    obs_f1 = float(observed["f1"])
    obs_lift = float(observed["lift"])
    null_f1_max = max(null.get("f1_max", 0.0) for null in nulls.values())
    null_lift_max = max(null.get("lift_max", 0.0) for null in nulls.values())
    ablation_persistent = any(
        metrics is not None and float(metrics["f1"]) >= obs_f1 * 0.5 and float(metrics["lift"]) > 1.5
        for name, metrics in ablations.items()
        if name != "full"
    )
    if obs_f1 > null_f1_max and obs_lift > null_lift_max and ablation_persistent:
        return "STRONG_PERSISTENT_SIGNAL"
    if obs_f1 > null_f1_max and obs_lift > null_lift_max:
        return "QUALIFIED_PERSISTENT_SIGNAL"
    return "WEAK_OR_NULL_SIGNAL"


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [piece.strip() for piece in args.fields.split(",") if piece.strip()]
    moduli = [int(piece.strip()) for piece in args.moduli.split(",") if piece.strip()]
    targets = [piece.strip() for piece in args.targets.split(",") if piece.strip()]
    windows_all = build_windows(args)
    wanted = {piece.strip() for piece in args.windows.split(",") if piece.strip()}
    windows = [window for window in windows_all if window["name"] in wanted]

    train_rows = [qa_values(b, e) for b, e in square_pairs(1, 100)]
    train_vectors, feature_count = build_feature_matrix(train_rows, fields, moduli)
    null_families = [piece.strip() for piece in args.null_families.split(",") if piece.strip()]
    ablation_sets = {
        "small_moduli": [2, 3, 4, 5],
        "odd_prime_moduli": [3, 5, 7, 11, 13, 17, 19],
        "power_composite_moduli": [2, 4, 8, 9, 16, 24],
        "full": moduli,
    }

    rows_out = []
    details = []
    for window_index, window in enumerate(windows):
        test_rows, sampled = sample_pairs(window["pairs"](), window["total"], args.sample_cap, args.seed + window_index)
        test_vectors, _ = build_feature_matrix(test_rows, fields, moduli)
        for target_index, target in enumerate(targets):
            train_labels = [label_for(row, target) for row in train_rows]
            test_labels = [label_for(row, target) for row in test_rows]
            train_pos = sum(train_labels)
            test_pos = sum(test_labels)
            if min(train_pos, len(train_labels) - train_pos, test_pos, len(test_labels) - test_pos) < args.min_positive:
                rows_out.append(
                    {
                        "target": target,
                        "window": window["name"],
                        "rows_evaluated": len(test_rows),
                        "sampled": sampled,
                        "positive_rows": test_pos,
                        "base_rate": test_pos / len(test_rows) if test_rows else 0.0,
                        "observed_f1": None,
                        "observed_lift": None,
                        "max_null_f1": None,
                        "max_null_lift": None,
                        "reversed_f1": None,
                        "verdict": "LOW_SUPPORT",
                    }
                )
                continue

            model = train_hebbian_active(train_vectors, train_labels, feature_count)
            observed = evaluate_model_active(model, test_vectors, test_labels)
            nulls = {}
            for family_index, family in enumerate(null_families):
                nulls[family] = null_family_metrics(
                    family,
                    train_rows,
                    train_vectors,
                    feature_count,
                    train_labels,
                    test_rows,
                    test_vectors,
                    test_labels,
                    fields,
                    moduli,
                    args.null_iterations,
                    args.seed + 100000 + window_index * 1000 + target_index * 10 + family_index,
                )
            reversed_metrics = evaluate_reversed_window(train_rows=train_rows, test_rows=test_rows, target=target, fields=fields, moduli=moduli)
            ablations = ablation_metrics(train_rows, test_rows, target, fields, ablation_sets)
            row_verdict = verdict(observed, nulls, ablations)
            max_null_f1 = max(null["f1_max"] for null in nulls.values())
            max_null_lift = max(null["lift_max"] for null in nulls.values())
            rows_out.append(
                {
                    "target": target,
                    "window": window["name"],
                    "rows_evaluated": len(test_rows),
                    "sampled": sampled,
                    "positive_rows": test_pos,
                    "base_rate": observed["base_rate"],
                    "observed_f1": observed["f1"],
                    "observed_lift": observed["lift"],
                    "max_null_f1": max_null_f1,
                    "max_null_lift": max_null_lift,
                    "reversed_f1": None if reversed_metrics is None else reversed_metrics["f1"],
                    "verdict": row_verdict,
                }
            )
            details.append(
                {
                    "target": target,
                    "window": window["name"],
                    "observed": observed,
                    "nulls": nulls,
                    "reversed_window": reversed_metrics,
                    "ablation": ablations,
                    "verdict": row_verdict,
                }
            )

    csv_path = out_dir / args.leaderboard_csv
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    verdict_counts: dict[str, int] = {}
    for row in rows_out:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    payload = {
        "stage_id": "qa_quantum_arithmetic_stronger_nulls_stage3",
        "hypothesis": (
            "If QA coordinate-residue Hebbian prototypes capture target-specific QA structure, observed metrics should "
            "beat label, coordinate, residue-column, random-polynomial, and same-density random nulls, and should not "
            "depend exclusively on a single modulus family."
        ),
        "parameters": {
            "fields": fields,
            "moduli": moduli,
            "targets": targets,
            "windows": [window["name"] for window in windows],
            "sample_cap": args.sample_cap,
            "null_iterations": args.null_iterations,
            "null_families": null_families,
            "min_positive": args.min_positive,
            "seed": args.seed,
        },
        "artifacts": {"leaderboard_csv": str(csv_path)},
        "verdict_counts": verdict_counts,
        "leaderboard": rows_out,
        "details": details,
        "honest_interpretation": (
            "This is a stronger empirical control harness. STRONG means the observed model beat all configured null "
            "families on F1 and lift and retained nontrivial signal in a modulus ablation. It still does not prove "
            "a theorem or establish factorization capability."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    out_path = out_dir / args.summary_json
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            fields="b,e,d,a",
            moduli="2,3,4,5",
            targets="X_semiprime,F_semiprime",
            windows="square_101_300",
            sample_cap=300,
            null_iterations=1,
            null_families="label_shuffle,coordinate_shuffle,residue_column_shuffle,same_density_random_positive,random_polynomial_target",
            min_positive=2,
            seed=41,
            prime_max=50,
            prime_radius=1,
            fibonacci_limit=100,
            fibonacci_radius=1,
            special_cap=100,
            random_count=100,
            leaderboard_csv="qa_quantum_arithmetic_stronger_nulls_stage3_leaderboard.csv",
            summary_json="qa_quantum_arithmetic_stronger_nulls_stage3.json",
        )
        payload = run(args)
        ok = len(payload["leaderboard"]) == 2 and Path(payload["artifacts"]["leaderboard_csv"]).exists()
        return {"ok": ok, "verdict_counts": payload["verdict_counts"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3 stronger null controls for QA arithmetic mining.")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--fields", default="b,e,d,a")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13,16,17,19,24")
    parser.add_argument("--targets", default="X_semiprime,F_semiprime,W_semiprime,G_prime,squarefree_X,X_omega_3,X_lpf_ge_sqrt")
    parser.add_argument("--windows", default="square_101_300,square_3001_10000,band_b1_1000_e1_100,random_sparse_1e6")
    parser.add_argument("--sample-cap", type=int, default=50000)
    parser.add_argument("--null-iterations", type=int, default=5)
    parser.add_argument(
        "--null-families",
        default="label_shuffle,coordinate_shuffle,residue_column_shuffle,same_density_random_positive,random_polynomial_target",
    )
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--prime-max", type=int, default=5000)
    parser.add_argument("--prime-radius", type=int, default=2)
    parser.add_argument("--fibonacci-limit", type=int, default=10000)
    parser.add_argument("--fibonacci-radius", type=int, default=2)
    parser.add_argument("--special-cap", type=int, default=50000)
    parser.add_argument("--random-count", type=int, default=50000)
    parser.add_argument("--leaderboard-csv", default="qa_quantum_arithmetic_stronger_nulls_stage3_leaderboard.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stronger_nulls_stage3.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(f"[qa_quantum_arithmetic_stronger_nulls_stage3] wrote {payload['artifacts']['leaderboard_csv']}")
    print(f"[qa_quantum_arithmetic_stronger_nulls_stage3] leaderboard_rows={len(payload['leaderboard'])}")
    print(f"[qa_quantum_arithmetic_stronger_nulls_stage3] verdict_counts={canonical_json(payload['verdict_counts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
