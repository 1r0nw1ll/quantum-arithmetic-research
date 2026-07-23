#!/usr/bin/env python3
"""Stage 4 ranking/generalization metrics for QA arithmetic residue mining."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

from pattern_targets_stage2 import label_for
from scale_grid_stage1 import build_windows, canonical_json, domain_sha256, qa_values, sample_pairs, square_pairs
from stronger_nulls_stage3 import (
    build_feature_matrix,
    score_active,
    score_predictions,
    train_hebbian_active,
)


DOMAIN = "QA_QUANTUM_ARITHMETIC_GENERALIZATION_METRICS_STAGE4.v1"


def model_scores(model: dict[str, object], vectors: list[list[int]]) -> list[float]:
    contrast = model["contrast"]
    contrast_sum = float(model["contrast_sum"])
    return [score_active(contrast, contrast_sum, vector) for vector in vectors]


def threshold_metrics(model: dict[str, object], scores: list[float], labels: list[int]) -> dict[str, float | int]:
    threshold = float(model["threshold"])
    predictions = [int(score >= threshold) for score in scores]
    return score_predictions(predictions, labels)


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


def topk_metrics(scores: list[float], labels: list[int], fractions: list[float]) -> dict[str, dict[str, float | int]]:
    ordered = sorted(zip(scores, labels), reverse=True)
    total = len(labels)
    positives = sum(labels)
    base_rate = positives / total if total else 0.0
    out = {}
    for fraction in fractions:
        k = max(1, int(round(total * fraction)))
        selected = ordered[:k]
        hits = sum(label for _, label in selected)
        precision = hits / k
        recall = hits / positives if positives else 0.0
        out[f"top_{fraction:g}"] = {
            "fraction": fraction,
            "k": k,
            "hits": hits,
            "precision": precision,
            "recall": recall,
            "lift": precision / base_rate if base_rate else 0.0,
        }
    return out


def calibration_buckets(
    target: str,
    window: str,
    scores: list[float],
    labels: list[int],
    bucket_count: int,
) -> list[dict[str, object]]:
    ordered = sorted(zip(scores, labels), reverse=True)
    total = len(ordered)
    positives = sum(labels)
    base_rate = positives / total if total else 0.0
    rows = []
    for bucket_index in range(bucket_count):
        start = int(total * bucket_index / bucket_count)
        end = int(total * (bucket_index + 1) / bucket_count)
        bucket = ordered[start:end]
        if not bucket:
            continue
        hits = sum(label for _, label in bucket)
        precision = hits / len(bucket)
        rows.append(
            {
                "target": target,
                "window": window,
                "bucket": bucket_index + 1,
                "bucket_count": bucket_count,
                "rank_start": start + 1,
                "rank_end": end,
                "rows": len(bucket),
                "positive_rows": hits,
                "precision": precision,
                "lift": precision / base_rate if base_rate else 0.0,
                "score_min": min(score for score, _ in bucket),
                "score_max": max(score for score, _ in bucket),
            }
        )
    return rows


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
    fractions = [float(piece.strip()) for piece in args.topk_fractions.split(",") if piece.strip()]
    wanted_windows = {piece.strip() for piece in args.windows.split(",") if piece.strip()}
    windows = [window for window in build_windows(args) if window["name"] in wanted_windows]

    train_rows = [qa_values(b, e) for b, e in square_pairs(1, 100)]
    train_vectors, feature_count = build_feature_matrix(train_rows, fields, moduli)
    models = {}
    train_counts = {}
    for target in targets:
        labels = [label_for(row, target) for row in train_rows]
        train_counts[target] = sum(labels)
        models[target] = train_hebbian_active(train_vectors, labels, feature_count)

    leaderboard = []
    calibration_rows = []
    for window_index, window in enumerate(windows):
        rows, sampled = sample_pairs(window["pairs"](), window["total"], args.sample_cap, args.seed + window_index)
        vectors, _ = build_feature_matrix(rows, fields, moduli)
        for target in targets:
            labels = [label_for(row, target) for row in rows]
            positives = sum(labels)
            if positives < args.min_positive:
                leaderboard.append(
                    {
                        "target": target,
                        "window": window["name"],
                        "rows_evaluated": len(rows),
                        "sampled": sampled,
                        "positive_rows": positives,
                        "base_rate": positives / len(rows) if rows else 0.0,
                        "average_precision": None,
                        "threshold_f1": None,
                        "threshold_lift": None,
                        "top_1pct_precision": None,
                        "top_1pct_lift": None,
                        "top_1pct_hits": None,
                        "serious_result": False,
                    }
                )
                continue
            scores = model_scores(models[target], vectors)
            threshold = threshold_metrics(models[target], scores, labels)
            ap = average_precision(scores, labels)
            topk = topk_metrics(scores, labels, fractions)
            top_1 = topk.get("top_0.01") or topk.get("top_0.01".replace("0.01", "0.01"))
            if top_1 is None:
                top_1 = topk[min(topk.keys(), key=lambda key: abs(topk[key]["fraction"] - 0.01))]
            base_rate = positives / len(rows)
            serious = bool(top_1["lift"] >= args.serious_lift and top_1["hits"] >= args.min_positive)
            leaderboard.append(
                {
                    "target": target,
                    "window": window["name"],
                    "rows_evaluated": len(rows),
                    "sampled": sampled,
                    "positive_rows": positives,
                    "base_rate": base_rate,
                    "average_precision": ap,
                    "threshold_f1": threshold["f1"],
                    "threshold_lift": threshold["lift"],
                    "top_1pct_precision": top_1["precision"],
                    "top_1pct_lift": top_1["lift"],
                    "top_1pct_hits": top_1["hits"],
                    "serious_result": serious,
                }
            )
            calibration_rows.extend(calibration_buckets(target, window["name"], scores, labels, args.bucket_count))
            for name, metrics in topk.items():
                leaderboard[-1][f"{name}_precision"] = metrics["precision"]
                leaderboard[-1][f"{name}_lift"] = metrics["lift"]
                leaderboard[-1][f"{name}_hits"] = metrics["hits"]

    leaderboard_csv = out_dir / args.leaderboard_csv
    calibration_csv = out_dir / args.calibration_csv
    write_csv(leaderboard_csv, leaderboard)
    write_csv(calibration_csv, calibration_rows)
    payload = {
        "stage_id": "qa_quantum_arithmetic_generalization_metrics_stage4",
        "hypothesis": (
            "For sparse QA arithmetic targets, Hebbian score ranking should show stronger generalization evidence "
            "through average precision, top-k enrichment, and calibration than through accuracy alone."
        ),
        "parameters": {
            "fields": fields,
            "moduli": moduli,
            "targets": targets,
            "windows": [window["name"] for window in windows],
            "sample_cap": args.sample_cap,
            "topk_fractions": fractions,
            "bucket_count": args.bucket_count,
            "serious_lift": args.serious_lift,
            "seed": args.seed,
        },
        "train": {
            "window": "square_1_100",
            "rows": len(train_rows),
            "positive_rows_by_target": train_counts,
        },
        "artifacts": {
            "leaderboard_csv": str(leaderboard_csv),
            "calibration_csv": str(calibration_csv),
        },
        "serious_result_count": sum(1 for row in leaderboard if row["serious_result"]),
        "leaderboard": leaderboard,
        "honest_interpretation": (
            "Stage 4 evaluates ranking enrichment. A serious_result flag means top 1% lift crossed the configured "
            "threshold with enough positive hits; it is still an empirical result rather than a theorem."
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
            topk_fractions="0.01,0.05,0.1",
            bucket_count=5,
            min_positive=2,
            serious_lift=2.0,
            seed=53,
            prime_max=50,
            prime_radius=1,
            fibonacci_limit=100,
            fibonacci_radius=1,
            special_cap=100,
            random_count=100,
            leaderboard_csv="qa_quantum_arithmetic_generalization_stage4_leaderboard.csv",
            calibration_csv="qa_quantum_arithmetic_generalization_stage4_calibration.csv",
            summary_json="qa_quantum_arithmetic_generalization_stage4.json",
        )
        payload = run(args)
        ok = (
            len(payload["leaderboard"]) == 2
            and Path(payload["artifacts"]["leaderboard_csv"]).exists()
            and Path(payload["artifacts"]["calibration_csv"]).exists()
        )
        return {"ok": ok, "rows": len(payload["leaderboard"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4 ranking/generalization metrics for QA arithmetic mining.")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--fields", default="b,e,d,a")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13,16,17,19,24")
    parser.add_argument("--targets", default="X_semiprime,F_semiprime,W_semiprime,X_omega_3,squarefree_X")
    parser.add_argument("--windows", default="square_101_300,square_3001_10000,band_b1_1000_e1_100,random_sparse_1e6")
    parser.add_argument("--sample-cap", type=int, default=50000)
    parser.add_argument("--topk-fractions", default="0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--serious-lift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--prime-max", type=int, default=5000)
    parser.add_argument("--prime-radius", type=int, default=2)
    parser.add_argument("--fibonacci-limit", type=int, default=10000)
    parser.add_argument("--fibonacci-radius", type=int, default=2)
    parser.add_argument("--special-cap", type=int, default=50000)
    parser.add_argument("--random-count", type=int, default=50000)
    parser.add_argument("--leaderboard-csv", default="qa_quantum_arithmetic_generalization_stage4_leaderboard.csv")
    parser.add_argument("--calibration-csv", default="qa_quantum_arithmetic_generalization_stage4_calibration.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_generalization_stage4.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(f"[qa_quantum_arithmetic_generalization_stage4] wrote {payload['artifacts']['leaderboard_csv']}")
    print(f"[qa_quantum_arithmetic_generalization_stage4] leaderboard_rows={len(payload['leaderboard'])}")
    print(f"[qa_quantum_arithmetic_generalization_stage4] serious_result_count={payload['serious_result_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
