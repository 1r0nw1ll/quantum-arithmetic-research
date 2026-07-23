#!/usr/bin/env python3
"""Mine symbolic coordinate rules and Hebbian prototypes for QA semiprime rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tempfile
from pathlib import Path

from generate_dataset import canonical_json, domain_sha256, run as generate_run


DOMAIN = "QA_QUANTUM_ARITHMETIC_SYMBOLIC_HEBBIAN_PROBE.v1"


def read_core_csv(path: Path) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, int | float] = {}
            for key, value in raw.items():
                row[key] = float(value) if key in {"L", "h"} else int(value)
            rows.append(row)
    return rows


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


def attach_labels(rows: list[dict[str, int | float]]) -> None:
    for row in rows:
        row["x_is_semiprime"] = 1 if factor_count(int(row["X"])) == 2 else 0


def train_test_split(rows: list[dict[str, int | float]], seed: int, train_fraction: float) -> tuple[list[dict[str, int | float]], list[dict[str, int | float]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * train_fraction))
    return shuffled[:split], shuffled[split:]


def literal_value(row: dict[str, int | float], literal: tuple[str, str, int, int]) -> bool:
    kind, field, modulus, residue = literal
    value = int(row[field])
    if kind == "mod_eq":
        return value % modulus == residue
    if kind == "mod_ne":
        return value % modulus != residue
    raise ValueError(f"unknown literal kind: {kind}")


def literal_label(literal: tuple[str, str, int, int]) -> str:
    kind, field, modulus, residue = literal
    op = "==" if kind == "mod_eq" else "!="
    return f"{field} mod {modulus} {op} {residue}"


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
    }


def mine_literals(rows: list[dict[str, int | float]], fields: list[str], moduli: list[int]) -> list[tuple[str, str, int, int]]:
    literals: list[tuple[str, str, int, int]] = []
    for field in fields:
        for modulus in moduli:
            for residue in range(modulus):
                literals.append(("mod_eq", field, modulus, residue))
    return literals


def evaluate_rule(rows: list[dict[str, int | float]], literals: list[tuple[str, str, int, int]]) -> dict[str, float | int | str]:
    truths = [int(row["x_is_semiprime"]) for row in rows]
    predictions = [int(all(literal_value(row, literal) for literal in literals)) for row in rows]
    score = score_predictions(predictions, truths)
    score["rule"] = " AND ".join(literal_label(literal) for literal in literals)
    return score


def symbolic_search(
    train_rows: list[dict[str, int | float]],
    test_rows: list[dict[str, int | float]],
    fields: list[str],
    moduli: list[int],
    min_support: int,
    top_k: int,
) -> list[dict[str, object]]:
    literals = mine_literals(train_rows, fields, moduli)
    candidates: list[tuple[float, float, int, list[tuple[str, str, int, int]]]] = []
    for literal in literals:
        train_score = evaluate_rule(train_rows, [literal])
        if int(train_score["support"]) >= min_support:
            candidates.append((float(train_score["lift"]), float(train_score["precision"]), int(train_score["support"]), [literal]))

    top_literals = [candidate[3][0] for candidate in sorted(candidates, reverse=True)[: min(80, len(candidates))]]
    for index, left in enumerate(top_literals):
        for right in top_literals[index + 1 :]:
            if left[1:] == right[1:]:
                continue
            train_score = evaluate_rule(train_rows, [left, right])
            if int(train_score["support"]) >= min_support:
                candidates.append((float(train_score["lift"]), float(train_score["precision"]), int(train_score["support"]), [left, right]))

    selected = sorted(candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True)[:top_k]
    rules: list[dict[str, object]] = []
    for _, _, _, rule_literals in selected:
        train_score = evaluate_rule(train_rows, rule_literals)
        test_score = evaluate_rule(test_rows, rule_literals)
        rules.append({"rule": train_score["rule"], "train": train_score, "test": test_score})
    return rules


def hebbian_features(row: dict[str, int | float], fields: list[str], moduli: list[int]) -> list[int]:
    features: list[int] = []
    for field in fields:
        value = int(row[field])
        for modulus in moduli:
            residue = value % modulus
            for candidate in range(modulus):
                features.append(1 if residue == candidate else -1)
    return features


def mean_vector(vectors: list[list[int]]) -> list[float]:
    if not vectors:
        return []
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))]


def dot(left: list[float], right: list[int]) -> float:
    return sum(a * b for a, b in zip(left, right))


def hebbian_probe(
    train_rows: list[dict[str, int | float]],
    test_rows: list[dict[str, int | float]],
    fields: list[str],
    moduli: list[int],
) -> dict[str, object]:
    positive_vectors = [
        hebbian_features(row, fields, moduli)
        for row in train_rows
        if int(row["x_is_semiprime"]) == 1
    ]
    negative_vectors = [
        hebbian_features(row, fields, moduli)
        for row in train_rows
        if int(row["x_is_semiprime"]) == 0
    ]
    positive_proto = mean_vector(positive_vectors)
    negative_proto = mean_vector(negative_vectors)
    contrast = [pos - neg for pos, neg in zip(positive_proto, negative_proto)]

    scores = [dot(contrast, hebbian_features(row, fields, moduli)) for row in train_rows]
    truths = [int(row["x_is_semiprime"]) for row in train_rows]
    best_threshold = 0.0
    best_f1 = -1.0
    total_positive = sum(truths)
    grouped_scores: list[tuple[float, int, int]] = []
    for score, truth in sorted(zip(scores, truths), reverse=True):
        if grouped_scores and grouped_scores[-1][0] == score:
            old_score, old_total, old_positive = grouped_scores[-1]
            grouped_scores[-1] = (old_score, old_total + 1, old_positive + truth)
        else:
            grouped_scores.append((score, 1, truth))
    predicted_positive = 0
    true_positive = 0
    for threshold, group_total, group_positive in grouped_scores:
        predicted_positive += group_total
        true_positive += group_positive
        false_positive = predicted_positive - true_positive
        false_negative = total_positive - true_positive
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / total_positive if total_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    test_scores = [dot(contrast, hebbian_features(row, fields, moduli)) for row in test_rows]
    test_truths = [int(row["x_is_semiprime"]) for row in test_rows]
    test_predictions = [int(score >= best_threshold) for score in test_scores]
    metrics = score_predictions(test_predictions, test_truths)
    strongest = sorted(
        enumerate(contrast),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:20]
    labels = feature_labels(fields, moduli)
    return {
        "train_positive_rows": len(positive_vectors),
        "train_negative_rows": len(negative_vectors),
        "feature_count": len(contrast),
        "threshold": best_threshold,
        "test_metrics": metrics,
        "strongest_associations": [
            {"feature": labels[index], "weight": weight}
            for index, weight in strongest
        ],
    }


def feature_labels(fields: list[str], moduli: list[int]) -> list[str]:
    labels: list[str] = []
    for field in fields:
        for modulus in moduli:
            for residue in range(modulus):
                labels.append(f"{field} mod {modulus} == {residue}")
    return labels


def write_rule_csv(path: Path, rules: list[dict[str, object]]) -> None:
    rows = []
    for rule in rules:
        train = rule["train"]
        test = rule["test"]
        rows.append(
            {
                "rule": rule["rule"],
                "train_precision": train["precision"],
                "train_recall": train["recall"],
                "train_lift": train["lift"],
                "train_support": train["support"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_lift": test["lift"],
                "test_support": test["support"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["rule"])
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = read_core_csv(Path(args.core_csv))
    attach_labels(rows)
    train_rows, test_rows = train_test_split(rows, args.seed, args.train_fraction)
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    moduli = [int(piece.strip()) for piece in args.moduli.split(",") if piece.strip()]
    rules = symbolic_search(train_rows, test_rows, fields, moduli, args.min_support, args.top_k)
    hebbian = hebbian_probe(train_rows, test_rows, fields, moduli)

    out_path = Path(args.out)
    rule_csv = Path(args.rule_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_rule_csv(rule_csv, rules)
    payload = {
        "probe_id": "qa_quantum_arithmetic_symbolic_hebbian_probe_001",
        "source_core_csv": args.core_csv,
        "artifacts": {"rule_csv": str(rule_csv)},
        "parameters": {
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "fields": fields,
            "moduli": moduli,
            "min_support": args.min_support,
            "top_k": args.top_k,
        },
        "summary": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_semiprime_rows": sum(int(row["x_is_semiprime"]) for row in train_rows),
            "test_semiprime_rows": sum(int(row["x_is_semiprime"]) for row in test_rows),
            "best_symbolic_rules": rules,
            "hebbian": hebbian,
        },
        "honest_interpretation": (
            "Symbolic rules are ranked by held-out descriptive lift and support; Hebbian weights are residue "
            "associations, not causal laws. Any promising rule still needs null controls and larger grids."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        gen_args = argparse.Namespace(
            b_min=1,
            b_max=14,
            e_min=1,
            e_max=14,
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
            out=str(Path(tmp) / "qa_quantum_arithmetic_symbolic_hebbian_probe.json"),
            rule_csv=str(Path(tmp) / "qa_quantum_arithmetic_symbolic_rules.csv"),
            seed=11,
            train_fraction=0.75,
            fields="b,e,d,a,X,D,F,G,W",
            moduli="2,3,4,5",
            min_support=3,
            top_k=8,
        )
        payload = run(args)
        ok = (
            payload["summary"]["test_rows"] > 0
            and len(payload["summary"]["best_symbolic_rules"]) > 0
            and payload["summary"]["hebbian"]["feature_count"] > 0
        )
        return {"ok": ok, "rules": len(payload["summary"]["best_symbolic_rules"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run symbolic and Hebbian QA semiprime-coordinate probes.")
    parser.add_argument("--core-csv", default="results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_core.csv")
    parser.add_argument("--out", default="results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_symbolic_hebbian_probe.json")
    parser.add_argument("--rule-csv", default="results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_symbolic_rules.csv")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--fields", default="b,e,d,a,X,D,F,G,W")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    best = payload["summary"]["best_symbolic_rules"][0]
    hebbian = payload["summary"]["hebbian"]["test_metrics"]
    print(f"[qa_quantum_arithmetic_symbolic_hebbian_probe] wrote {args.out}")
    print(f"[qa_quantum_arithmetic_symbolic_hebbian_probe] best_rule={best['rule']}")
    print(
        "[qa_quantum_arithmetic_symbolic_hebbian_probe] "
        f"hebbian_f1={hebbian['f1']:.6f} precision={hebbian['precision']:.6f} recall={hebbian['recall']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
