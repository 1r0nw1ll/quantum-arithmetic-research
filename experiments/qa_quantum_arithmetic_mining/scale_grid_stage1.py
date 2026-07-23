#!/usr/bin/env python3
"""Stage 1 scale sweep for QA arithmetic semiprime residue mining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Iterator


DOMAIN = "QA_QUANTUM_ARITHMETIC_SCALE_GRID_STAGE1.v1"


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


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = e + d
    D = d * d
    G = D + e * e
    W = d * (e + a)
    return {
        "b": b,
        "e": e,
        "d": d,
        "a": a,
        "D": D,
        "X": e * d,
        "F": a * b,
        "G": G,
        "W": W,
    }


def label_for(row: dict[str, int], target: str) -> int:
    if target == "X_semiprime":
        return int(is_semiprime(row["X"]))
    if target == "F_semiprime":
        return int(is_semiprime(row["F"]))
    raise ValueError(f"unknown target: {target}")


def feature_vector(row: dict[str, int], fields: list[str], moduli: list[int]) -> list[int]:
    vector: list[int] = []
    for field in fields:
        value = row[field]
        for modulus in moduli:
            residue = value % modulus
            for candidate in range(modulus):
                vector.append(1 if residue == candidate else -1)
    return vector


def dot(left: list[float], right: list[int]) -> float:
    return sum(a * b for a, b in zip(left, right))


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


def train_hebbian(
    vectors: list[list[int]],
    labels: list[int],
) -> dict[str, object]:
    positive = [vector for vector, label in zip(vectors, labels) if label == 1]
    negative = [vector for vector, label in zip(vectors, labels) if label == 0]
    if not positive or not negative:
        raise ValueError("training labels must contain positive and negative rows")
    feature_count = len(positive[0])
    positive_proto = [sum(vector[index] for vector in positive) / len(positive) for index in range(feature_count)]
    negative_proto = [sum(vector[index] for vector in negative) / len(negative) for index in range(feature_count)]
    contrast = [pos - neg for pos, neg in zip(positive_proto, negative_proto)]
    scores = [dot(contrast, vector) for vector in vectors]
    threshold = best_threshold(scores, labels)
    return {
        "contrast": contrast,
        "threshold": threshold,
        "train_positive_rows": sum(labels),
        "train_negative_rows": len(labels) - sum(labels),
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


def evaluate_model(
    model: dict[str, object],
    vectors: list[list[int]],
    labels: list[int],
) -> dict[str, float | int]:
    contrast = model["contrast"]
    threshold = float(model["threshold"])
    predictions = [int(dot(contrast, vector) >= threshold) for vector in vectors]
    return score_predictions(predictions, labels)


def square_pairs(start: int, end: int) -> Iterator[tuple[int, int]]:
    for b in range(start, end + 1):
        for e in range(start, end + 1):
            yield b, e


def band_pairs(b_start: int, b_end: int, e_start: int, e_end: int) -> Iterator[tuple[int, int]]:
    for b in range(b_start, b_end + 1):
        for e in range(e_start, e_end + 1):
            yield b, e


def random_pairs(limit: int, count: int, seed: int) -> Iterator[tuple[int, int]]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < count:
        pair = (rng.randint(1, limit), rng.randint(1, limit))
        if pair not in seen:
            seen.add(pair)
            yield pair


def prime_center_pairs(max_prime: int, radius: int, cap: int) -> Iterator[tuple[int, int]]:
    primes = [value for value in range(2, max_prime + 1) if is_prime(value)]
    emitted = 0
    seen: set[tuple[int, int]] = set()
    for prime in primes:
        for db in range(-radius, radius + 1):
            for de in range(-radius, radius + 1):
                b = prime + db
                e = prime + de
                if b < 1 or e < 1:
                    continue
                pair = (b, e)
                if pair in seen:
                    continue
                seen.add(pair)
                yield pair
                emitted += 1
                if emitted >= cap:
                    return


def fibonacci_numbers(limit: int) -> list[int]:
    values = [1, 2]
    while values[-1] + values[-2] <= limit:
        values.append(values[-1] + values[-2])
    return values


def fibonacci_pairs(limit: int, radius: int) -> Iterator[tuple[int, int]]:
    values = fibonacci_numbers(limit)
    seen: set[tuple[int, int]] = set()
    for index in range(len(values) - 1):
        bases = [(values[index], values[index + 1]), (values[index + 1], values[index])]
        for base_b, base_e in bases:
            for db in range(-radius, radius + 1):
                for de in range(-radius, radius + 1):
                    b = base_b + db
                    e = base_e + de
                    if b < 1 or e < 1:
                        continue
                    pair = (b, e)
                    if pair not in seen:
                        seen.add(pair)
                        yield pair


def sample_pairs(
    pairs: Iterable[tuple[int, int]],
    total_pairs: int | None,
    cap: int,
    seed: int,
) -> tuple[list[dict[str, int]], bool]:
    if total_pairs is not None and total_pairs <= cap:
        return [qa_values(b, e) for b, e in pairs], False
    if total_pairs is None:
        rows = []
        for index, (b, e) in enumerate(pairs):
            if index >= cap:
                break
            rows.append(qa_values(b, e))
        return rows, True

    rng = random.Random(seed)
    selected_indices = set(rng.sample(range(total_pairs), cap))
    rows = []
    for index, (b, e) in enumerate(pairs):
        if index in selected_indices:
            rows.append(qa_values(b, e))
            if len(rows) == cap:
                break
    return rows, True


def build_windows(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {"name": "square_101_300", "pairs": lambda: square_pairs(101, 300), "total": 200 * 200},
        {"name": "square_301_1000", "pairs": lambda: square_pairs(301, 1000), "total": 700 * 700},
        {"name": "square_1001_3000", "pairs": lambda: square_pairs(1001, 3000), "total": 2000 * 2000},
        {"name": "square_3001_10000", "pairs": lambda: square_pairs(3001, 10000), "total": 7000 * 7000},
        {"name": "band_b1_1000_e1_100", "pairs": lambda: band_pairs(1, 1000, 1, 100), "total": 1000 * 100},
        {"name": "band_b1_100_e1_1000", "pairs": lambda: band_pairs(1, 100, 1, 1000), "total": 100 * 1000},
        {
            "name": "prime_centered_radius_2",
            "pairs": lambda: prime_center_pairs(args.prime_max, args.prime_radius, args.special_cap),
            "total": None,
        },
        {
            "name": "fibonacci_radius_2",
            "pairs": lambda: fibonacci_pairs(args.fibonacci_limit, args.fibonacci_radius),
            "total": None,
        },
        {
            "name": "random_sparse_1e6",
            "pairs": lambda: random_pairs(1_000_000, args.random_count, args.seed + 1000),
            "total": args.random_count,
        },
    ]


def null_metrics(
    train_vectors: list[list[int]],
    train_labels: list[int],
    test_vectors: list[list[int]],
    test_labels: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    runs = []
    for _ in range(iterations):
        shuffled = list(train_labels)
        rng.shuffle(shuffled)
        model = train_hebbian(train_vectors, shuffled)
        runs.append(evaluate_model(model, test_vectors, test_labels))
    summary: dict[str, float] = {}
    for key in ["precision", "recall", "f1", "lift"]:
        values = [float(run[key]) for run in runs]
        summary[f"{key}_mean"] = statistics.mean(values) if values else 0.0
        summary[f"{key}_max"] = max(values) if values else 0.0
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    moduli = [int(piece.strip()) for piece in args.moduli.split(",") if piece.strip()]
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]

    train_rows = [qa_values(b, e) for b, e in square_pairs(1, 100)]
    train_vectors = [feature_vector(row, fields, moduli) for row in train_rows]
    models = {}
    train_label_counts = {}
    for target in targets:
        train_labels = [label_for(row, target) for row in train_rows]
        train_label_counts[target] = sum(train_labels)
        models[target] = train_hebbian(train_vectors, train_labels)

    leaderboard = []
    windows = build_windows(args)
    for window_index, window in enumerate(windows):
        rows, sampled = sample_pairs(
            window["pairs"](),
            window["total"],
            args.sample_cap,
            args.seed + window_index,
        )
        vectors = [feature_vector(row, fields, moduli) for row in rows]
        for target in targets:
            labels = [label_for(row, target) for row in rows]
            observed = evaluate_model(models[target], vectors, labels)
            controls = null_metrics(
                train_vectors,
                [label_for(row, target) for row in train_rows],
                vectors,
                labels,
                args.null_iterations,
                args.seed + 10000 + window_index,
            )
            verdict = (
                "PERSISTENT_SIGNAL"
                if float(observed["f1"]) > controls["f1_max"] and float(observed["lift"]) > controls["lift_max"]
                else "WEAK_OR_NULL_SIGNAL"
            )
            leaderboard.append(
                {
                    "target": target,
                    "window": window["name"],
                    "rows_evaluated": len(rows),
                    "window_total_pairs": window["total"],
                    "sampled": sampled,
                    "positive_rows": sum(labels),
                    "base_rate": observed["base_rate"],
                    "precision": observed["precision"],
                    "recall": observed["recall"],
                    "f1": observed["f1"],
                    "lift": observed["lift"],
                    "null_f1_max": controls["f1_max"],
                    "null_lift_max": controls["lift_max"],
                    "verdict": verdict,
                }
            )

    csv_path = out_dir / args.leaderboard_csv
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)

    payload = {
        "stage_id": "qa_quantum_arithmetic_scale_grid_stage1",
        "hypothesis": (
            "Coordinate-only residue Hebbian prototypes trained on b,e=1..100 should retain above-null signal for "
            "semiprime targets on larger and structurally different coordinate samples if the signal is not only a "
            "low-range artifact."
        ),
        "parameters": {
            "fields": fields,
            "moduli": moduli,
            "targets": targets,
            "sample_cap": args.sample_cap,
            "null_iterations": args.null_iterations,
            "seed": args.seed,
        },
        "train": {
            "window": "square_1_100",
            "rows": len(train_rows),
            "positive_rows_by_target": train_label_counts,
        },
        "artifacts": {"leaderboard_csv": str(csv_path)},
        "leaderboard": leaderboard,
        "honest_interpretation": (
            "Large windows are deterministic samples when sampled=true. This sweep ranks empirical persistence "
            "against shuffled-label controls; it does not prove semiprime prediction or factorization capability."
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
            sample_cap=200,
            null_iterations=2,
            seed=23,
            prime_max=50,
            prime_radius=1,
            fibonacci_limit=100,
            fibonacci_radius=1,
            special_cap=100,
            random_count=100,
            leaderboard_csv="qa_quantum_arithmetic_scale_stage1_leaderboard.csv",
            summary_json="qa_quantum_arithmetic_scale_stage1.json",
        )
        payload = run(args)
        targets = {row["target"] for row in payload["leaderboard"]}
        ok = (
            targets == {"X_semiprime", "F_semiprime"}
            and len(payload["leaderboard"]) == 18
            and Path(payload["artifacts"]["leaderboard_csv"]).exists()
        )
        return {"ok": ok, "rows": len(payload["leaderboard"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 QA arithmetic large-grid semiprime sweep.")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--fields", default="b,e,d,a")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13,16,17,19,24")
    parser.add_argument("--targets", default="X_semiprime,F_semiprime")
    parser.add_argument("--sample-cap", type=int, default=50000)
    parser.add_argument("--null-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--prime-max", type=int, default=5000)
    parser.add_argument("--prime-radius", type=int, default=2)
    parser.add_argument("--fibonacci-limit", type=int, default=10000)
    parser.add_argument("--fibonacci-radius", type=int, default=2)
    parser.add_argument("--special-cap", type=int, default=50000)
    parser.add_argument("--random-count", type=int, default=50000)
    parser.add_argument("--leaderboard-csv", default="qa_quantum_arithmetic_scale_stage1_leaderboard.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_scale_stage1.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    persistent = sum(1 for row in payload["leaderboard"] if row["verdict"] == "PERSISTENT_SIGNAL")
    print(f"[qa_quantum_arithmetic_scale_stage1] wrote {payload['artifacts']['leaderboard_csv']}")
    print(f"[qa_quantum_arithmetic_scale_stage1] leaderboard_rows={len(payload['leaderboard'])}")
    print(f"[qa_quantum_arithmetic_scale_stage1] persistent_rows={persistent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
