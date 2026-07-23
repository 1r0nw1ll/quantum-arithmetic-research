#!/usr/bin/env python3
"""Analyze coordinate-ray pattern cuts from a generated QA mining database."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from pathlib import Path

from generate_dataset import canonical_json, domain_sha256, run as generate_run


DOMAIN = "QA_QUANTUM_ARITHMETIC_PATTERN_ANALYSIS.v1"


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["slope_label", "semiprime_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ray_csv = out_dir / args.ray_csv
    near_csv = out_dir / args.near_csv
    summary_json = out_dir / args.summary_json

    conn = connect(db_path)
    try:
        ray_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    v.slope_label,
                    COUNT(*) AS semiprime_count,
                    MIN(v.distance_from_origin) AS nearest_distance,
                    AVG(c.X) AS mean_X,
                    MIN(c.X) AS min_X,
                    MAX(c.X) AS max_X
                FROM validation_matrix v
                JOIN core_matrix c ON c.config_id = v.config_id
                WHERE v.x_is_semiprime = 1
                GROUP BY v.slope_label
                ORDER BY semiprime_count DESC, nearest_distance ASC, slope_label ASC
                LIMIT ?
                """,
                (args.top_rays,),
            ).fetchall()
        ]
        near_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    c.b_seed AS b, c.e_seed AS e, c.X, v.x_prime_factors, v.slope_label,
                    v.distance_from_origin, c.D_area AS D, c.W, c.h_height AS h, v.apex_focus_residual
                FROM validation_matrix v
                JOIN core_matrix c ON c.config_id = v.config_id
                WHERE v.x_is_semiprime = 1
                ORDER BY v.distance_from_origin ASC, c.b_seed ASC, c.e_seed ASC
                LIMIT ?
                """,
                (args.nearest,),
            ).fetchall()
        ]
        total_rows = conn.execute("SELECT COUNT(*) FROM core_matrix").fetchone()[0]
        semiprime_rows = conn.execute(
            "SELECT COUNT(*) FROM validation_matrix WHERE x_is_semiprime = 1"
        ).fetchone()[0]
        ray_count = conn.execute(
            "SELECT COUNT(DISTINCT slope_label) FROM validation_matrix WHERE x_is_semiprime = 1"
        ).fetchone()[0]
    finally:
        conn.close()

    write_csv(ray_csv, ray_rows)
    write_csv(near_csv, near_rows)
    payload = {
        "analysis_id": "qa_quantum_arithmetic_pattern_analysis_001",
        "source_db": str(db_path),
        "artifacts": {
            "ray_csv": str(ray_csv),
            "nearest_semiprime_csv": str(near_csv),
        },
        "summary": {
            "total_rows": total_rows,
            "x_semiprime_rows": semiprime_rows,
            "semiprime_slope_label_count": ray_count,
            "top_rays": ray_rows,
            "nearest_semiprime_examples": near_rows[:10],
        },
        "honest_interpretation": (
            "Ray counts are descriptive coordinates for inspection. They are not evidence of a prime classifier "
            "unless tested against held-out grids and null/permutation controls."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    summary_json.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        gen_args = argparse.Namespace(
            b_min=1,
            b_max=10,
            e_min=1,
            e_max=10,
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
            db=str(Path(tmp) / "qa_quantum_arithmetic_mining.sqlite"),
            out_dir=tmp,
            ray_csv="qa_quantum_arithmetic_semiprime_rays.csv",
            near_csv="qa_quantum_arithmetic_nearest_semiprimes.csv",
            summary_json="qa_quantum_arithmetic_pattern_summary.json",
            top_rays=8,
            nearest=12,
        )
        payload = run(args)
        ok = payload["summary"]["x_semiprime_rows"] > 0 and Path(payload["artifacts"]["ray_csv"]).exists()
        return {"ok": ok, "x_semiprime_rows": payload["summary"]["x_semiprime_rows"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze QA arithmetic mining pattern cuts.")
    parser.add_argument("--db", default="results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_mining.sqlite")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--ray-csv", default="qa_quantum_arithmetic_semiprime_rays.csv")
    parser.add_argument("--near-csv", default="qa_quantum_arithmetic_nearest_semiprimes.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_pattern_summary.json")
    parser.add_argument("--top-rays", type=int, default=25)
    parser.add_argument("--nearest", type=int, default=50)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(f"[qa_quantum_arithmetic_pattern_analysis] semiprime rays={payload['summary']['semiprime_slope_label_count']}")
    print(f"[qa_quantum_arithmetic_pattern_analysis] wrote {payload['artifacts']['ray_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
