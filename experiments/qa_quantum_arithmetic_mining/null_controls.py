#!/usr/bin/env python3
"""Run persistence and shuffled-label controls for QA semiprime residue maps."""

from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path

from generate_dataset import canonical_json, domain_sha256, run as generate_run
from symbolic_hebbian_probe import (
    attach_labels,
    hebbian_probe,
    read_core_csv,
    symbolic_search,
)


DOMAIN = "QA_QUANTUM_ARITHMETIC_MINING_NULL_CONTROLS.v1"


def generate_grid_csv(
    out_dir: Path,
    b_min: int,
    b_max: int,
    e_min: int,
    e_max: int,
    origin_b: int,
    origin_e: int,
) -> Path:
    args = argparse.Namespace(
        b_min=b_min,
        b_max=b_max,
        e_min=e_min,
        e_max=e_max,
        origin_b=origin_b,
        origin_e=origin_e,
        out_dir=str(out_dir),
        db="qa_quantum_arithmetic_mining.sqlite",
        core_csv="qa_quantum_arithmetic_core.csv",
        semiprime_csv="qa_quantum_arithmetic_x_semiprime.csv",
        summary_json="qa_quantum_arithmetic_summary.json",
    )
    generate_run(args)
    return out_dir / "qa_quantum_arithmetic_core.csv"


def clone_rows(rows: list[dict[str, int | float]]) -> list[dict[str, int | float]]:
    return [dict(row) for row in rows]


def shuffled_label_rows(rows: list[dict[str, int | float]], rng: random.Random) -> list[dict[str, int | float]]:
    copied = clone_rows(rows)
    labels = [row["x_is_semiprime"] for row in copied]
    rng.shuffle(labels)
    for row, label in zip(copied, labels):
        row["x_is_semiprime"] = label
    return copied


def metric_summary(metrics: list[dict[str, float | int]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = ["precision", "recall", "f1", "lift"]
    summary = {}
    for key in keys:
        values = [float(metric[key]) for metric in metrics]
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_max"] = max(values)
    return summary


def write_null_csv(path: Path, null_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["iteration", "precision", "recall", "f1", "lift", "support"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in null_rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(args.train_core_csv)
    external_dir = out_dir / f"external_b{args.external_b_min}_{args.external_b_max}_e{args.external_e_min}_{args.external_e_max}"
    external_csv = generate_grid_csv(
        external_dir,
        args.external_b_min,
        args.external_b_max,
        args.external_e_min,
        args.external_e_max,
        args.origin_b,
        args.origin_e,
    )

    train_rows = read_core_csv(train_csv)
    external_rows = read_core_csv(external_csv)
    attach_labels(train_rows)
    attach_labels(external_rows)

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    moduli = [int(piece.strip()) for piece in args.moduli.split(",") if piece.strip()]

    symbolic_rules = symbolic_search(
        train_rows,
        external_rows,
        fields,
        moduli,
        args.min_support,
        args.top_k,
    )
    observed_hebbian = hebbian_probe(train_rows, external_rows, fields, moduli)
    observed_metrics = observed_hebbian["test_metrics"]

    rng = random.Random(args.seed)
    null_rows = []
    null_metrics = []
    for iteration in range(args.null_iterations):
        shuffled_train = shuffled_label_rows(train_rows, rng)
        null_hebbian = hebbian_probe(shuffled_train, external_rows, fields, moduli)
        metrics = null_hebbian["test_metrics"]
        null_metrics.append(metrics)
        null_rows.append(
            {
                "iteration": iteration,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "lift": metrics["lift"],
                "support": metrics["support"],
            }
        )

    null_csv = out_dir / args.null_csv
    write_null_csv(null_csv, null_rows)
    null_summary = metric_summary(null_metrics)
    verdict = (
        "PERSISTENT_SIGNAL"
        if float(observed_metrics["f1"]) > null_summary.get("f1_max", 0.0)
        and float(observed_metrics["lift"]) > null_summary.get("lift_max", 0.0)
        else "WEAK_OR_NULL_SIGNAL"
    )
    payload = {
        "control_id": "qa_quantum_arithmetic_mining_null_controls_001",
        "source_train_core_csv": str(train_csv),
        "source_external_core_csv": str(external_csv),
        "artifacts": {"null_csv": str(null_csv)},
        "parameters": {
            "fields": fields,
            "moduli": moduli,
            "origin_b": args.origin_b,
            "origin_e": args.origin_e,
            "external_grid": {
                "b_min": args.external_b_min,
                "b_max": args.external_b_max,
                "e_min": args.external_e_min,
                "e_max": args.external_e_max,
            },
            "null_iterations": args.null_iterations,
            "seed": args.seed,
            "min_support": args.min_support,
            "top_k": args.top_k,
        },
        "summary": {
            "train_rows": len(train_rows),
            "external_rows": len(external_rows),
            "train_semiprime_rows": sum(int(row["x_is_semiprime"]) for row in train_rows),
            "external_semiprime_rows": sum(int(row["x_is_semiprime"]) for row in external_rows),
            "observed_hebbian": observed_hebbian,
            "null_hebbian_summary": null_summary,
            "best_external_symbolic_rules": symbolic_rules[:10],
            "verdict": verdict,
        },
        "honest_interpretation": (
            "This control tests whether residue associations trained on the original grid persist on an out-of-window "
            "grid and beat shuffled-label Hebbian nulls. A persistent signal is still empirical and must be retested "
            "across larger ranges and alternative external windows."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    out_path = out_dir / args.summary_json
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_dir = root / "train"
        train_csv = generate_grid_csv(train_dir, 1, 14, 1, 14, 1, 2)
        args = argparse.Namespace(
            train_core_csv=str(train_csv),
            out_dir=str(root / "controls"),
            external_b_min=15,
            external_b_max=22,
            external_e_min=15,
            external_e_max=22,
            origin_b=1,
            origin_e=2,
            fields="b,e,d,a",
            moduli="2,3,4,5",
            min_support=3,
            top_k=5,
            null_iterations=3,
            seed=17,
            null_csv="qa_quantum_arithmetic_hebbian_null_controls.csv",
            summary_json="qa_quantum_arithmetic_null_controls.json",
        )
        payload = run(args)
        ok = (
            payload["summary"]["external_rows"] == 64
            and len(payload["summary"]["best_external_symbolic_rules"]) > 0
            and len(payload["summary"]["null_hebbian_summary"]) > 0
        )
        return {"ok": ok, "verdict": payload["summary"]["verdict"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QA arithmetic semiprime null controls.")
    parser.add_argument("--train-core-csv", default="results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_core.csv")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--external-b-min", type=int, default=101)
    parser.add_argument("--external-b-max", type=int, default=200)
    parser.add_argument("--external-e-min", type=int, default=101)
    parser.add_argument("--external-e-max", type=int, default=200)
    parser.add_argument("--origin-b", type=int, default=1)
    parser.add_argument("--origin-e", type=int, default=2)
    parser.add_argument("--fields", default="b,e,d,a")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--null-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--null-csv", default="qa_quantum_arithmetic_hebbian_null_controls.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_null_controls.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    observed = payload["summary"]["observed_hebbian"]["test_metrics"]
    null_summary = payload["summary"]["null_hebbian_summary"]
    print(f"[qa_quantum_arithmetic_null_controls] verdict={payload['summary']['verdict']}")
    print(
        "[qa_quantum_arithmetic_null_controls] "
        f"observed_f1={observed['f1']:.6f} null_f1_max={null_summary['f1_max']:.6f}"
    )
    print(
        "[qa_quantum_arithmetic_null_controls] "
        f"observed_lift={observed['lift']:.6f} null_lift_max={null_summary['lift_max']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
