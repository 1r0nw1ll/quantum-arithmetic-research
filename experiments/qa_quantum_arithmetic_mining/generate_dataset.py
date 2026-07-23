#!/usr/bin/env python3
"""Generate QA arithmetic mining tables as SQLite, CSV, and JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_MINING.v1"


CORE_COLUMNS = [
    "config_id",
    "b",
    "e",
    "d",
    "a",
    "B",
    "E",
    "D",
    "A",
    "X",
    "C",
    "F",
    "G",
    "L",
    "H",
    "I",
    "J",
    "K",
    "W",
    "Y",
    "Z",
    "h",
]

CORE_DB_COLUMNS = [
    "config_id",
    "b_seed",
    "e_seed",
    "d_len",
    "a_len",
    "B_area",
    "E_area",
    "D_area",
    "A_area",
    "X",
    "C",
    "F",
    "G",
    "L",
    "H_sum",
    "I",
    "J",
    "K",
    "W",
    "Y",
    "Z",
    "h_height",
]


SEMIPRIME_COLUMNS = [
    "config_id",
    "b",
    "e",
    "X",
    "prime_factors",
    "slope_label",
    "slope_value",
    "distance_from_origin",
    "D",
    "W",
    "h",
    "apex_focus_distance",
]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = math.isqrt(n)
    divisor = 3
    while divisor <= limit:
        if n % divisor == 0:
            return False
        divisor += 2
    return True


def factorization(n: int) -> list[int]:
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


def semiprime_factors(n: int) -> tuple[bool, str | None]:
    factors = factorization(n)
    if len(factors) == 2 and all(is_prime(factor) for factor in factors):
        return True, f"{factors[0]} x {factors[1]}"
    return False, None


def is_square(n: int) -> bool:
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def slope_parts(b: int, e: int, origin_b: int, origin_e: int) -> tuple[str, float | None]:
    delta_b = b - origin_b
    delta_e = e - origin_e
    if delta_b == 0 and delta_e == 0:
        return "origin", 0.0
    if delta_b == 0:
        return "vertical", None
    divisor = math.gcd(delta_e, delta_b)
    numerator = delta_e // divisor
    denominator = delta_b // divisor
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    return f"{numerator}/{denominator}", delta_e / delta_b


def qa_row(config_id: int, b: int, e: int) -> dict[str, object]:
    d = b + e
    a = e + d
    B = b * b
    E = e * e
    D = d * d
    A = a * a
    X = e * d
    C = e * d * 2
    F = b * a
    G = D + E
    L = (C * F) / 12
    H = C + F
    I = abs(C - F)
    J = d * b
    K = d * a
    W = d * (e + a)
    Y = A - D
    Z = E + K
    h = math.sqrt(F) * d
    return {
        "config_id": config_id,
        "b": b,
        "e": e,
        "d": d,
        "a": a,
        "B": B,
        "E": E,
        "D": D,
        "A": A,
        "X": X,
        "C": C,
        "F": F,
        "G": G,
        "L": L,
        "H": H,
        "I": I,
        "J": J,
        "K": K,
        "W": W,
        "Y": Y,
        "Z": Z,
        "h": h,
    }


def validation_row(row: dict[str, object], origin_b: int, origin_e: int) -> dict[str, object]:
    b = int(row["b"])
    e = int(row["e"])
    X = int(row["X"])
    F = int(row["F"])
    C = int(row["C"])
    D = int(row["D"])
    G = int(row["G"])
    h = float(row["h"])
    is_x_semiprime, factors = semiprime_factors(X)
    slope_label, slope_value = slope_parts(b, e, origin_b, origin_e)
    distance = math.sqrt((b - origin_b) * (b - origin_b) + (e - origin_e) * (e - origin_e))
    apex_focus_distance = math.sqrt(X * X + h * h)
    triangle_area_cf = (C * F) / 2
    return {
        "config_id": int(row["config_id"]),
        "slope_label": slope_label,
        "slope_value": slope_value,
        "distance_from_origin": distance,
        "x_is_semiprime": int(is_x_semiprime),
        "x_prime_factors": factors,
        "x_factor_count": len(factorization(X)),
        "g_is_square": int(is_square(G)),
        "f_is_square": int(is_square(F)),
        "h_is_integer": int(is_square(F)),
        "triangle_area_CF": triangle_area_cf,
        "L_times_6": float(row["L"]) * 6,
        "apex_focus_distance": apex_focus_distance,
        "apex_focus_residual": abs(apex_focus_distance - D),
    }


def connect_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS validation_matrix;
        DROP TABLE IF EXISTS core_matrix;

        CREATE TABLE core_matrix (
            config_id INTEGER PRIMARY KEY,
            b_seed INTEGER NOT NULL,
            e_seed INTEGER NOT NULL,
            d_len INTEGER NOT NULL,
            a_len INTEGER NOT NULL,
            B_area INTEGER NOT NULL,
            E_area INTEGER NOT NULL,
            D_area INTEGER NOT NULL,
            A_area INTEGER NOT NULL,
            X INTEGER NOT NULL,
            C INTEGER NOT NULL,
            F INTEGER NOT NULL,
            G INTEGER NOT NULL,
            L REAL NOT NULL,
            H_sum INTEGER NOT NULL,
            I INTEGER NOT NULL,
            J INTEGER NOT NULL,
            K INTEGER NOT NULL,
            W INTEGER NOT NULL,
            Y INTEGER NOT NULL,
            Z INTEGER NOT NULL,
            h_height REAL NOT NULL
        );

        CREATE TABLE validation_matrix (
            validation_id INTEGER PRIMARY KEY,
            config_id INTEGER NOT NULL,
            slope_label TEXT NOT NULL,
            slope_value REAL,
            distance_from_origin REAL NOT NULL,
            x_is_semiprime INTEGER NOT NULL,
            x_prime_factors TEXT,
            x_factor_count INTEGER NOT NULL,
            g_is_square INTEGER NOT NULL,
            f_is_square INTEGER NOT NULL,
            h_is_integer INTEGER NOT NULL,
            triangle_area_CF REAL NOT NULL,
            L_times_6 REAL NOT NULL,
            apex_focus_distance REAL NOT NULL,
            apex_focus_residual REAL NOT NULL,
            FOREIGN KEY(config_id) REFERENCES core_matrix(config_id)
        );

        CREATE INDEX idx_core_be ON core_matrix(b_seed, e_seed);
        CREATE INDEX idx_core_x ON core_matrix(X);
        CREATE INDEX idx_validation_semiprime ON validation_matrix(x_is_semiprime);
        CREATE INDEX idx_validation_slope ON validation_matrix(slope_label);
        """
    )


def insert_rows(conn: sqlite3.Connection, rows: list[dict[str, object]], validations: list[dict[str, object]]) -> None:
    core_insert = f"""
        INSERT INTO core_matrix ({", ".join(CORE_DB_COLUMNS)})
        VALUES ({", ".join("?" for _ in CORE_DB_COLUMNS)})
    """
    conn.executemany(core_insert, [[row[column] for column in CORE_COLUMNS] for row in rows])

    validation_columns = [
        "config_id",
        "slope_label",
        "slope_value",
        "distance_from_origin",
        "x_is_semiprime",
        "x_prime_factors",
        "x_factor_count",
        "g_is_square",
        "f_is_square",
        "h_is_integer",
        "triangle_area_CF",
        "L_times_6",
        "apex_focus_distance",
        "apex_focus_residual",
    ]
    validation_insert = f"""
        INSERT INTO validation_matrix ({", ".join(validation_columns)})
        VALUES ({", ".join("?" for _ in validation_columns)})
    """
    conn.executemany(validation_insert, [[row[column] for column in validation_columns] for row in validations])
    conn.commit()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_semiprime_rows(conn: sqlite3.Connection, limit: int | None = None) -> list[dict[str, object]]:
    query = """
        SELECT
            c.config_id, c.b_seed AS b, c.e_seed AS e, c.X, v.x_prime_factors AS prime_factors,
            v.slope_label, v.slope_value, v.distance_from_origin,
            c.D_area AS D, c.W, c.h_height AS h, v.apex_focus_distance
        FROM core_matrix c
        JOIN validation_matrix v ON c.config_id = v.config_id
        WHERE v.x_is_semiprime = 1
        ORDER BY v.distance_from_origin ASC, c.b_seed ASC, c.e_seed ASC
    """
    if limit is not None:
        query += " LIMIT ?"
        cursor = conn.execute(query, (limit,))
    else:
        cursor = conn.execute(query)
    return [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()]


def summarize(conn: sqlite3.Connection, args: argparse.Namespace, db_path: Path, semiprime_csv: Path) -> dict[str, object]:
    total_rows = conn.execute("SELECT COUNT(*) FROM core_matrix").fetchone()[0]
    semiprime_rows = conn.execute(
        "SELECT COUNT(*) FROM validation_matrix WHERE x_is_semiprime = 1"
    ).fetchone()[0]
    square_g_rows = conn.execute("SELECT COUNT(*) FROM validation_matrix WHERE g_is_square = 1").fetchone()[0]
    integer_h_rows = conn.execute("SELECT COUNT(*) FROM validation_matrix WHERE h_is_integer = 1").fetchone()[0]
    max_apex_residual = conn.execute("SELECT MAX(apex_focus_residual) FROM validation_matrix").fetchone()[0]
    top_slopes = [
        {"slope_label": row[0], "semiprime_count": row[1]}
        for row in conn.execute(
            """
            SELECT slope_label, COUNT(*) AS count
            FROM validation_matrix
            WHERE x_is_semiprime = 1
            GROUP BY slope_label
            ORDER BY count DESC, slope_label ASC
            LIMIT 12
            """
        ).fetchall()
    ]
    payload = {
        "experiment_id": "qa_quantum_arithmetic_mining_001",
        "domain": "number_theory_geometry",
        "hypothesis": (
            "A relational QA arithmetic table can expose coordinate rays and distance-ranked cuts where generated "
            "variables such as X=e*d have semiprime or square structure, without claiming a prime-prediction law."
        ),
        "success_criteria": (
            "PASS if the standalone generator emits SQLite, semiprime CSV, core CSV, and summary JSON artifacts; "
            "semiprime factors are exact; and the apex-to-focus validation residual is numerically negligible."
        ),
        "parameters": {
            "b_min": args.b_min,
            "b_max": args.b_max,
            "e_min": args.e_min,
            "e_max": args.e_max,
            "origin_b": args.origin_b,
            "origin_e": args.origin_e,
        },
        "artifacts": {
            "sqlite": str(db_path),
            "core_csv": str(args.core_csv),
            "semiprime_csv": str(semiprime_csv),
        },
        "summary": {
            "total_rows": total_rows,
            "x_semiprime_rows": semiprime_rows,
            "x_semiprime_fraction": semiprime_rows / total_rows if total_rows else 0.0,
            "g_square_rows": square_g_rows,
            "h_integer_rows": integer_h_rows,
            "max_apex_focus_residual": max_apex_residual,
            "top_semiprime_slope_labels": top_slopes,
        },
        "honest_interpretation": (
            "This is an exploratory data-mining scaffold. It validates generated identities and ranks coordinate "
            "patterns for inspection; it does not establish a new theorem about semiprimes."
        ),
    }
    payload["result"] = "PASS" if max_apex_residual is not None and max_apex_residual < 0.000000001 else "FAIL"
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    return payload


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.b_min < 1 or args.e_min < 1:
        raise ValueError("b_min and e_min must be >= 1")
    if args.b_max < args.b_min or args.e_max < args.e_min:
        raise ValueError("max bounds must be >= min bounds")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = out_dir / db_path
    core_csv = Path(args.core_csv)
    if not core_csv.is_absolute():
        core_csv = out_dir / core_csv
    semiprime_csv = Path(args.semiprime_csv)
    if not semiprime_csv.is_absolute():
        semiprime_csv = out_dir / semiprime_csv
    summary_json = Path(args.summary_json)
    if not summary_json.is_absolute():
        summary_json = out_dir / summary_json

    rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    config_id = 1
    for b in range(args.b_min, args.b_max + 1):
        for e in range(args.e_min, args.e_max + 1):
            row = qa_row(config_id, b, e)
            rows.append(row)
            validations.append(validation_row(row, args.origin_b, args.origin_e))
            config_id += 1

    conn = connect_database(db_path)
    try:
        create_schema(conn)
        insert_rows(conn, rows, validations)
        write_csv(core_csv, CORE_COLUMNS, rows)
        write_csv(semiprime_csv, SEMIPRIME_COLUMNS, fetch_semiprime_rows(conn))
        args.core_csv = core_csv
        payload = summarize(conn, args, db_path, semiprime_csv)
    finally:
        conn.close()

    summary_json.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            b_min=1,
            b_max=8,
            e_min=1,
            e_max=8,
            origin_b=1,
            origin_e=2,
            out_dir=tmp,
            db="qa_quantum_arithmetic_mining.sqlite",
            core_csv="qa_quantum_arithmetic_core.csv",
            semiprime_csv="qa_quantum_arithmetic_x_semiprime.csv",
            summary_json="qa_quantum_arithmetic_summary.json",
        )
        payload = run(args)
        db_path = Path(tmp) / "qa_quantum_arithmetic_mining.sqlite"
        conn = connect_database(db_path)
        try:
            row = conn.execute(
                """
                SELECT b_seed, e_seed, d_len, a_len, D_area, X, C, F, W
                FROM core_matrix
                WHERE b_seed=1 AND e_seed=2
                """
            ).fetchone()
            semiprime_count = conn.execute(
                "SELECT COUNT(*) FROM validation_matrix WHERE x_is_semiprime = 1"
            ).fetchone()[0]
            max_residual = conn.execute("SELECT MAX(apex_focus_residual) FROM validation_matrix").fetchone()[0]
        finally:
            conn.close()
        ok = (
            payload["result"] == "PASS"
            and row == (1, 2, 3, 5, 9, 6, 12, 5, 21)
            and semiprime_count > 0
            and max_residual < 0.000000001
        )
        return {
            "ok": ok,
            "payload_result": payload["result"],
            "semiprime_count": semiprime_count,
            "checked_origin_like_row": list(row),
            "max_apex_focus_residual": max_residual,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate QA arithmetic mining data tables.")
    parser.add_argument("--b-min", type=int, default=1)
    parser.add_argument("--b-max", type=int, default=100)
    parser.add_argument("--e-min", type=int, default=1)
    parser.add_argument("--e-max", type=int, default=100)
    parser.add_argument("--origin-b", type=int, default=1)
    parser.add_argument("--origin-e", type=int, default=2)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--db", default="qa_quantum_arithmetic_mining.sqlite")
    parser.add_argument("--core-csv", default="qa_quantum_arithmetic_core.csv")
    parser.add_argument("--semiprime-csv", default="qa_quantum_arithmetic_x_semiprime.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_summary.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(f"[qa_quantum_arithmetic_mining] result={payload['result']}")
    print(f"[qa_quantum_arithmetic_mining] wrote {payload['artifacts']['sqlite']}")
    print(f"[qa_quantum_arithmetic_mining] semiprime rows={payload['summary']['x_semiprime_rows']}")
    return 0 if payload["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
