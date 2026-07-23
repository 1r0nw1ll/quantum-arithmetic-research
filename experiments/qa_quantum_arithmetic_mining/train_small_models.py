#!/usr/bin/env python3
"""Run tiny no-dependency model probes on QA arithmetic mining CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tempfile
from pathlib import Path

from generate_dataset import canonical_json, domain_sha256, run as generate_run


DOMAIN = "QA_QUANTUM_ARITHMETIC_TINY_MODEL_PROBE.v1"


def read_core_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def add_bias(features: list[float]) -> list[float]:
    return [1.0] + features


def feature_vector(row: dict[str, float]) -> list[float]:
    b = row["b"]
    e = row["e"]
    return add_bias([b, e, b * e, b * b, e * e])


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def train_linear_gd(rows: list[dict[str, float]], target: str, epochs: int, learning_rate: float) -> list[float]:
    weights = [0.0 for _ in feature_vector(rows[0])]
    scale = max(abs(row[target]) for row in rows) or 1.0
    for _ in range(epochs):
        for row in rows:
            features = feature_vector(row)
            prediction = dot(weights, features)
            truth = row[target] / scale
            error = prediction - truth
            for index, value in enumerate(features):
                weights[index] -= learning_rate * error * value
    return [weight * scale for weight in weights]


def evaluate_regression(rows: list[dict[str, float]], target: str, weights: list[float]) -> dict[str, float]:
    errors = []
    baseline = sum(row[target] for row in rows) / len(rows)
    baseline_errors = []
    for row in rows:
        prediction = dot(weights, feature_vector(row))
        truth = row[target]
        errors.append(abs(prediction - truth))
        baseline_errors.append(abs(baseline - truth))
    mae = sum(errors) / len(errors)
    baseline_mae = sum(baseline_errors) / len(baseline_errors)
    return {
        "mae": mae,
        "baseline_mae": baseline_mae,
        "improvement_ratio": baseline_mae / mae if mae else float("inf"),
    }


def nearest_neighbor_semiprime(train_rows: list[dict[str, float]], row: dict[str, float]) -> int:
    best_distance = None
    best_label = 0
    for candidate in train_rows:
        distance = (
            (candidate["b"] - row["b"]) * (candidate["b"] - row["b"])
            + (candidate["e"] - row["e"]) * (candidate["e"] - row["e"])
        )
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = int(candidate["x_is_semiprime"])
    return best_label


def attach_semiprime_labels(core_rows: list[dict[str, float]]) -> None:
    for row in core_rows:
        x = int(row["X"])
        factors = 0
        temp = x
        while temp % 2 == 0 and temp > 1:
            factors += 1
            temp //= 2
        divisor = 3
        while divisor <= math.isqrt(temp):
            while temp % divisor == 0:
                factors += 1
                temp //= divisor
            divisor += 2
        if temp > 1:
            factors += 1
        row["x_is_semiprime"] = 1.0 if factors == 2 else 0.0


def classification_metrics(train_rows: list[dict[str, float]], test_rows: list[dict[str, float]]) -> dict[str, float]:
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for row in test_rows:
        predicted = nearest_neighbor_semiprime(train_rows, row)
        truth = int(row["x_is_semiprime"])
        correct += int(predicted == truth)
        true_positive += int(predicted == 1 and truth == 1)
        false_positive += int(predicted == 1 and truth == 0)
        false_negative += int(predicted == 0 and truth == 1)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "accuracy": correct / len(test_rows),
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = read_core_csv(Path(args.core_csv))
    attach_semiprime_labels(rows)
    random.Random(args.seed).shuffle(rows)
    split = max(1, int(len(rows) * args.train_fraction))
    train_rows = rows[:split]
    test_rows = rows[split:]
    if not test_rows:
        raise ValueError("not enough rows for a held-out test split")

    regression = {}
    for target in ["D", "W", "h"]:
        weights = train_linear_gd(train_rows, target, args.epochs, args.learning_rate)
        regression[target] = evaluate_regression(test_rows, target, weights)
    semiprime = classification_metrics(train_rows, test_rows)
    payload = {
        "probe_id": "qa_quantum_arithmetic_tiny_model_probe_001",
        "source_core_csv": args.core_csv,
        "model_scope": (
            "No-dependency smoke probe: polynomial-feature linear regression for smooth QA variables and "
            "nearest-neighbor classification for X semiprime labels."
        ),
        "parameters": {
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        "summary": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "regression": regression,
            "x_semiprime_nearest_neighbor": semiprime,
        },
        "honest_interpretation": (
            "Smooth generated variables should be easy to approximate from b,e. Semiprime classification is a "
            "discrete probe and should be read as a baseline, not as evidence of factorization shortcut learning."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        gen_args = argparse.Namespace(
            b_min=1,
            b_max=12,
            e_min=1,
            e_max=12,
            origin_b=1,
            origin_e=2,
            out_dir=tmp,
            db="qa_quantum_arithmetic_mining.sqlite",
            core_csv="qa_quantum_arithmetic_core.csv",
            semiprime_csv="qa_quantum_arithmetic_x_semiprime.csv",
            summary_json="qa_quantum_arithmetic_summary.json",
        )
        generate_run(gen_args)
        args = argparse.Namespace(
            core_csv=str(Path(tmp) / "qa_quantum_arithmetic_core.csv"),
            out=str(Path(tmp) / "qa_quantum_arithmetic_model_probe.json"),
            seed=7,
            train_fraction=0.75,
            epochs=10,
            learning_rate=0.00000001,
        )
        payload = run(args)
        ok = payload["summary"]["test_rows"] > 0 and "h" in payload["summary"]["regression"]
        return {"ok": ok, "test_rows": payload["summary"]["test_rows"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tiny model probes on QA arithmetic mining data.")
    parser.add_argument("--core-csv", default="results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_core.csv")
    parser.add_argument("--out", default="results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_model_probe.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.00000000001)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(f"[qa_quantum_arithmetic_tiny_model_probe] wrote {args.out}")
    print(f"[qa_quantum_arithmetic_tiny_model_probe] test_rows={payload['summary']['test_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
