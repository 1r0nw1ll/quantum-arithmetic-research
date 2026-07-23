#!/usr/bin/env python3
"""Stage 2 target expansion for QA arithmetic pattern mining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

from scale_grid_stage1 import (
    best_threshold,
    build_windows,
    canonical_json,
    domain_sha256,
    dot,
    evaluate_model,
    factor_count,
    feature_vector,
    is_prime,
    is_semiprime,
    qa_values,
    sample_pairs,
    score_predictions,
    square_pairs,
    train_hebbian,
)


DOMAIN = "QA_QUANTUM_ARITHMETIC_PATTERN_TARGETS_STAGE2.v1"


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


def is_square(n: int) -> bool:
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def is_squarefree(n: int) -> bool:
    factors = prime_factors(n)
    return len(factors) == len(set(factors))


def largest_prime_factor(n: int) -> int:
    factors = prime_factors(n)
    return max(factors) if factors else 1


def target_names() -> list[str]:
    return [
        "X_semiprime",
        "F_semiprime",
        "W_semiprime",
        "G_prime",
        "G_square",
        "h_integer",
        "D_plus_X_prime",
        "W_minus_D_prime",
        "gcd_X_W_gt_1",
        "gcd_X_W_gt_D",
        "squarefree_X",
        "X_omega_2",
        "X_omega_3",
        "X_lpf_ge_sqrt",
        "X_lpf_ge_100",
    ]


def label_for(row: dict[str, int], target: str) -> int:
    X = row["X"]
    F = row["F"]
    W = row["W"]
    G = row["G"]
    D = row["D"]
    if target == "X_semiprime":
        return int(is_semiprime(X))
    if target == "F_semiprime":
        return int(is_semiprime(F))
    if target == "W_semiprime":
        return int(is_semiprime(W))
    if target == "G_prime":
        return int(is_prime(G))
    if target == "G_square":
        return int(is_square(G))
    if target == "h_integer":
        return int(is_square(F))
    if target == "D_plus_X_prime":
        return int(is_prime(D + X))
    if target == "W_minus_D_prime":
        return int(W > D and is_prime(W - D))
    if target == "gcd_X_W_gt_1":
        return int(math.gcd(X, W) > 1)
    if target == "gcd_X_W_gt_D":
        return int(math.gcd(X, W) > D)
    if target == "squarefree_X":
        return int(is_squarefree(X))
    if target == "X_omega_2":
        return int(factor_count(X) == 2)
    if target == "X_omega_3":
        return int(factor_count(X) == 3)
    if target == "X_lpf_ge_sqrt":
        return int(largest_prime_factor(X) >= math.isqrt(X))
    if target == "X_lpf_ge_100":
        return int(largest_prime_factor(X) >= 100)
    raise ValueError(f"unknown target: {target}")


def null_summary(
    train_vectors: list[list[int]],
    train_labels: list[int],
    test_vectors: list[list[int]],
    test_labels: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    runs = []
    for _ in range(iterations):
        shuffled = list(train_labels)
        rng.shuffle(shuffled)
        model = train_hebbian(train_vectors, shuffled)
        runs.append(evaluate_model(model, test_vectors, test_labels))
    out = {}
    for key in ["precision", "recall", "f1", "lift"]:
        values = [float(run[key]) for run in runs]
        out[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
        out[f"{key}_max"] = max(values) if values else 0.0
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

    train_rows = [qa_values(b, e) for b, e in square_pairs(1, 100)]
    train_vectors = [feature_vector(row, fields, moduli) for row in train_rows]
    models = {}
    train_counts = {}
    skipped_train = []
    for target in targets:
        labels = [label_for(row, target) for row in train_rows]
        positives = sum(labels)
        train_counts[target] = positives
        if positives < args.min_positive or len(labels) - positives < args.min_positive:
            skipped_train.append(target)
            continue
        models[target] = train_hebbian(train_vectors, labels)

    windows = build_windows(args)
    if args.windows:
        wanted = {piece.strip() for piece in args.windows.split(",") if piece.strip()}
        windows = [window for window in windows if window["name"] in wanted]

    rows_out = []
    for window_index, window in enumerate(windows):
        test_rows, sampled = sample_pairs(
            window["pairs"](),
            window["total"],
            args.sample_cap,
            args.seed + window_index,
        )
        test_vectors = [feature_vector(row, fields, moduli) for row in test_rows]
        for target in targets:
            labels = [label_for(row, target) for row in test_rows]
            positives = sum(labels)
            base = positives / len(labels) if labels else 0.0
            if target not in models or positives < args.min_positive:
                rows_out.append(
                    {
                        "target": target,
                        "window": window["name"],
                        "rows_evaluated": len(test_rows),
                        "sampled": sampled,
                        "positive_rows": positives,
                        "base_rate": base,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "lift": None,
                        "null_f1_max": None,
                        "null_lift_max": None,
                        "verdict": "LOW_SUPPORT",
                    }
                )
                continue
            observed = evaluate_model(models[target], test_vectors, labels)
            controls = null_summary(
                train_vectors,
                [label_for(row, target) for row in train_rows],
                test_vectors,
                labels,
                args.null_iterations,
                args.seed + 10000 + window_index,
            )
            verdict = (
                "PERSISTENT_SIGNAL"
                if float(observed["f1"]) > controls["f1_max"] and float(observed["lift"]) > controls["lift_max"]
                else "WEAK_OR_NULL_SIGNAL"
            )
            rows_out.append(
                {
                    "target": target,
                    "window": window["name"],
                    "rows_evaluated": len(test_rows),
                    "sampled": sampled,
                    "positive_rows": positives,
                    "base_rate": base,
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
    write_csv(csv_path, rows_out)
    target_summary = []
    for target in targets:
        target_rows = [row for row in rows_out if row["target"] == target and row["verdict"] != "LOW_SUPPORT"]
        persistent = sum(1 for row in target_rows if row["verdict"] == "PERSISTENT_SIGNAL")
        lifts = [float(row["lift"]) for row in target_rows if row["lift"] is not None]
        f1s = [float(row["f1"]) for row in target_rows if row["f1"] is not None]
        target_summary.append(
            {
                "target": target,
                "train_positive_rows": train_counts.get(target, 0),
                "evaluated_windows": len(target_rows),
                "persistent_windows": persistent,
                "mean_lift": sum(lifts) / len(lifts) if lifts else None,
                "mean_f1": sum(f1s) / len(f1s) if f1s else None,
            }
        )

    payload = {
        "stage_id": "qa_quantum_arithmetic_pattern_targets_stage2",
        "hypothesis": (
            "If QA coordinate residues carry general arithmetic structure, multiple QA-derived targets beyond "
            "X_semiprime should retain above-null Hebbian signal on out-of-window samples."
        ),
        "parameters": {
            "fields": fields,
            "moduli": moduli,
            "targets": targets,
            "windows": [window["name"] for window in windows],
            "sample_cap": args.sample_cap,
            "null_iterations": args.null_iterations,
            "min_positive": args.min_positive,
            "seed": args.seed,
        },
        "train": {
            "window": "square_1_100",
            "rows": len(train_rows),
            "positive_rows_by_target": train_counts,
            "skipped_low_support_targets": skipped_train,
        },
        "artifacts": {"leaderboard_csv": str(csv_path)},
        "target_summary": target_summary,
        "leaderboard": rows_out,
        "honest_interpretation": (
            "This expands targets for empirical mining. LOW_SUPPORT rows are not interpreted. Persistent signal "
            "means above shuffled-label nulls on the sampled windows, not a theorem or factorization method."
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
            targets="X_semiprime,F_semiprime,W_semiprime,G_prime,G_square,h_integer",
            windows="square_101_300,fibonacci_radius_2",
            sample_cap=300,
            null_iterations=1,
            min_positive=2,
            seed=31,
            prime_max=50,
            prime_radius=1,
            fibonacci_limit=100,
            fibonacci_radius=1,
            special_cap=100,
            random_count=100,
            leaderboard_csv="qa_quantum_arithmetic_pattern_targets_stage2_leaderboard.csv",
            summary_json="qa_quantum_arithmetic_pattern_targets_stage2.json",
        )
        payload = run(args)
        ok = (
            len(payload["leaderboard"]) == 12
            and Path(payload["artifacts"]["leaderboard_csv"]).exists()
            and "X_semiprime" in payload["train"]["positive_rows_by_target"]
        )
        return {"ok": ok, "rows": len(payload["leaderboard"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 QA arithmetic expanded target sweep.")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--fields", default="b,e,d,a")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13,16,17,19,24")
    parser.add_argument("--targets", default=",".join(target_names()))
    parser.add_argument(
        "--windows",
        default="square_101_300,square_3001_10000,band_b1_1000_e1_100,random_sparse_1e6",
    )
    parser.add_argument("--sample-cap", type=int, default=12000)
    parser.add_argument("--null-iterations", type=int, default=1)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--prime-max", type=int, default=5000)
    parser.add_argument("--prime-radius", type=int, default=2)
    parser.add_argument("--fibonacci-limit", type=int, default=10000)
    parser.add_argument("--fibonacci-radius", type=int, default=2)
    parser.add_argument("--special-cap", type=int, default=12000)
    parser.add_argument("--random-count", type=int, default=12000)
    parser.add_argument("--leaderboard-csv", default="qa_quantum_arithmetic_pattern_targets_stage2_leaderboard.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_pattern_targets_stage2.json")
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
    low_support = sum(1 for row in payload["leaderboard"] if row["verdict"] == "LOW_SUPPORT")
    print(f"[qa_quantum_arithmetic_pattern_targets_stage2] wrote {payload['artifacts']['leaderboard_csv']}")
    print(f"[qa_quantum_arithmetic_pattern_targets_stage2] leaderboard_rows={len(payload['leaderboard'])}")
    print(f"[qa_quantum_arithmetic_pattern_targets_stage2] persistent_rows={persistent} low_support_rows={low_support}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
