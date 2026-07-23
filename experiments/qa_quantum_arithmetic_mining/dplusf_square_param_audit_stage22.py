#!/usr/bin/env python3
"""Stage 22 parametrization audit for D_plus_F_square."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_DPLUSF_SQUARE_PARAM_AUDIT_STAGE22.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def d_plus_f(b: int, e: int) -> int:
    d = b + e
    a = b + 2 * e
    return d * d + a * b


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def square_witness(b: int, e: int) -> int | None:
    value = d_plus_f(b, e)
    root = math.isqrt(value)
    if root * root == value:
        return root
    return None


def brute_solutions(b_max: int, e_max: int) -> list[dict[str, int]]:
    rows = []
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            root = square_witness(b, e)
            if root is not None:
                rows.append({"b": b, "e": e, "u": e + 2 * b, "k": root, "D_plus_F": root * root})
    return rows


def generated_solutions(b_max: int, e_max: int, m_max: int, n_max: int, t_max: int) -> dict[tuple[int, int], dict[str, int]]:
    found: dict[tuple[int, int], dict[str, int]] = {}
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            base_b = 2 * m * n
            base_u = m * m + 2 * n * n
            base_k = abs(m * m - 2 * n * n)
            if base_k == 0:
                continue
            base_e = base_u - 2 * base_b
            if base_e <= 0:
                continue
            for t in range(1, t_max + 1):
                b = t * base_b
                e = t * base_e
                if b > b_max or e > e_max:
                    continue
                key = (b, e)
                found.setdefault(
                    key,
                    {
                        "b": b,
                        "e": e,
                        "m": m,
                        "n": n,
                        "t": t,
                        "u": t * base_u,
                        "k": t * base_k,
                    },
                )
    return found


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    brute = brute_solutions(args.b_max, args.e_max)
    generated = generated_solutions(args.b_max, args.e_max, args.m_max, args.n_max, args.t_max)
    generated_keys = set(generated)
    brute_keys = {(row["b"], row["e"]) for row in brute}
    misses = [row for row in brute if (row["b"], row["e"]) not in generated_keys]
    extras = [
        generated[key]
        for key in sorted(generated_keys - brute_keys)
        if square_witness(key[0], key[1]) is None
    ]
    parity_violations = [row for row in brute if row["b"] % 2 != 0]
    witness_rows = []
    for row in brute:
        key = (row["b"], row["e"])
        gen = generated.get(key, {})
        witness_rows.append(
            {
                **row,
                "b_even": row["b"] % 2 == 0,
                "param_hit": key in generated,
                "m": gen.get("m"),
                "n": gen.get("n"),
                "t": gen.get("t"),
                "sign_branch": gen.get("sign_branch"),
            }
        )
    witness_path = out_dir / args.witness_csv
    miss_path = out_dir / args.miss_csv
    write_csv(witness_path, witness_rows)
    write_csv(miss_path, misses)
    payload = {
        "stage_id": "qa_quantum_arithmetic_stage22_dplusf_square_param_audit",
        "hypothesis": (
            "D_plus_F_square is governed by the quadratic identity "
            "(e+2*b)^2 - 2*b*b = k*k and a Pell-style parametrization."
        ),
        "parameters": {
            "b_max": args.b_max,
            "e_max": args.e_max,
            "m_max": args.m_max,
            "n_max": args.n_max,
            "t_max": args.t_max,
        },
        "proved_identities": [
            "D_plus_F = 2*b*b + 4*b*e + e*e",
            "D_plus_F = (e + 2*b)*(e + 2*b) - 2*b*b",
            "D_plus_F_square implies b_even by mod-4 obstruction",
            "Param family: b=t*2*m*n, u=t*(m*m+2*n*n), k=t*abs(m*m-2*n*n), e=u-2*b",
        ],
        "artifacts": {"witness_csv": str(witness_path), "miss_csv": str(miss_path)},
        "brute_solution_count": len(brute),
        "param_hit_count": sum(1 for row in witness_rows if row["param_hit"]),
        "miss_count": len(misses),
        "parity_violation_count": len(parity_violations),
        "param_extra_invalid_count": len(extras),
        "first_misses": misses[:20],
        "honest_interpretation": (
            "This is a bounded exhaustiveness audit for a proposed parametrization. A zero miss count here supports "
            "but does not by itself prove global exhaustiveness."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    summary_path = out_dir / args.summary_json
    summary_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            b_max=30,
            e_max=40,
            m_max=40,
            n_max=40,
            t_max=40,
            witness_csv="stage22_selftest_witness.csv",
            miss_csv="stage22_selftest_miss.csv",
            summary_json="stage22_selftest.json",
        )
        payload = run(args)
        ok = (
            payload["brute_solution_count"] > 0
            and payload["parity_violation_count"] == 0
            and payload["param_extra_invalid_count"] == 0
            and Path(tmp, "stage22_selftest_witness.csv").exists()
            and Path(tmp, "stage22_selftest.json").exists()
        )
        return {"ok": ok, "solutions": payload["brute_solution_count"], "misses": payload["miss_count"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--b-max", type=int, default=300)
    parser.add_argument("--e-max", type=int, default=300)
    parser.add_argument("--m-max", type=int, default=400)
    parser.add_argument("--n-max", type=int, default=400)
    parser.add_argument("--t-max", type=int, default=400)
    parser.add_argument("--witness-csv", default="qa_quantum_arithmetic_stage22_dplusf_square_param_witnesses.csv")
    parser.add_argument("--miss-csv", default="qa_quantum_arithmetic_stage22_dplusf_square_param_misses.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage22_dplusf_square_param_audit.json")
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
                "brute_solution_count": payload["brute_solution_count"],
                "miss_count": payload["miss_count"],
                "parity_violation_count": payload["parity_violation_count"],
                "canonical_hash": payload["canonical_hash"],
                "artifacts": payload["artifacts"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
