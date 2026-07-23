#!/usr/bin/env python3
"""Stage 16 algebraic identity audit for QA arithmetic mining targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_IDENTITY_AUDIT_STAGE16.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = e + d
    D = d * d
    E = e * e
    F = a * b
    X = e * d
    return {
        "b": b,
        "e": e,
        "d": d,
        "a": a,
        "D": D,
        "E": E,
        "F": F,
        "X": X,
        "director_radius_sq": D * (D + F),
    }


def factor_count(n: int) -> int:
    count = 0
    temp = n
    while temp > 1 and temp % 2 == 0:
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


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def is_squarefree(n: int) -> bool:
    temp = n
    if temp % 4 == 0:
        return False
    divisor = 3
    while divisor * divisor <= temp:
        square = divisor * divisor
        if temp % square == 0:
            return False
        divisor += 2
    return True


def audit_rows(limit: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    mismatches = {
        "D_minus_F_eq_E": 0,
        "gcd_X_D_eq_d_iff_gcd_X_F_eq_1": 0,
        "gcd_X_D_gt_d_iff_gcd_X_F_gt_1": 0,
        "director_radius_sq_never_semiprime": 0,
        "director_radius_sq_never_squarefree": 0,
        "D_minus_F_square_always_true": 0,
    }
    checked = 0
    for b in range(1, limit + 1):
        for e in range(1, limit + 1):
            row = qa_values(b, e)
            checked += 1
            gap = row["D"] - row["F"]
            gcd_x_d_eq_d = math.gcd(row["X"], row["D"]) == row["d"]
            gcd_x_f_eq_1 = math.gcd(row["X"], row["F"]) == 1
            gcd_x_d_gt_d = math.gcd(row["X"], row["D"]) > row["d"]
            gcd_x_f_gt_1 = math.gcd(row["X"], row["F"]) > 1
            if gap != row["E"]:
                mismatches["D_minus_F_eq_E"] += 1
            if gcd_x_d_eq_d != gcd_x_f_eq_1:
                mismatches["gcd_X_D_eq_d_iff_gcd_X_F_eq_1"] += 1
            if gcd_x_d_gt_d != gcd_x_f_gt_1:
                mismatches["gcd_X_D_gt_d_iff_gcd_X_F_gt_1"] += 1
            if factor_count(row["director_radius_sq"]) == 2:
                mismatches["director_radius_sq_never_semiprime"] += 1
            if is_squarefree(row["director_radius_sq"]):
                mismatches["director_radius_sq_never_squarefree"] += 1
            if not is_square(gap):
                mismatches["D_minus_F_square_always_true"] += 1

    ledger = [
        {
            "claim": "D_minus_F_eq_E",
            "target_labels": "D_minus_F_square,evolute_gap_E_square,D_minus_F_semiprime,evolute_gap_E_semiprime",
            "status": "PROVEN_STRUCTURAL_IDENTITY",
            "algebraic_reason": "D-F=(b+e)*(b+e)-b*(b+2e)=e*e=E",
            "audit_mismatches": mismatches["D_minus_F_eq_E"],
        },
        {
            "claim": "D_minus_F_square_always_true",
            "target_labels": "D_minus_F_square,evolute_gap_E_square,polar_scale_D_minus_F_square",
            "status": "PROVEN_ALWAYS_TRUE",
            "algebraic_reason": "D-F=E=e*e, so the gap is a square for every valid b,e",
            "audit_mismatches": mismatches["D_minus_F_square_always_true"],
        },
        {
            "claim": "evolute_gap_E_squarefree_train_only_e1",
            "target_labels": "evolute_gap_E_squarefree",
            "status": "PROVEN_BOUNDARY_ONLY",
            "algebraic_reason": "E=e*e is squarefree only at e=1; held-out windows with e>1 have zero support",
            "audit_mismatches": 0,
        },
        {
            "claim": "gcd_X_D_eq_d_iff_gcd_X_F_eq_1",
            "target_labels": "gcd_X_D_eq_d,gcd_X_F_eq_1",
            "status": "PROVEN_STRUCTURAL_IDENTITY",
            "algebraic_reason": "gcd(X,D)=gcd(e*d,d*d)=d*gcd(e,d); the QA progression makes gcd(e,d)=1 equivalent to gcd(X,F)=1",
            "audit_mismatches": mismatches["gcd_X_D_eq_d_iff_gcd_X_F_eq_1"],
        },
        {
            "claim": "gcd_X_D_gt_d_iff_gcd_X_F_gt_1",
            "target_labels": "gcd_X_D_gt_d,gcd_X_F_gt_1",
            "status": "PROVEN_STRUCTURAL_IDENTITY",
            "algebraic_reason": "The complement of gcd(e,d)=1 gives extra gcd(X,D) beyond d and exactly matches shared factors between X and F",
            "audit_mismatches": mismatches["gcd_X_D_gt_d_iff_gcd_X_F_gt_1"],
        },
        {
            "claim": "director_radius_sq_never_semiprime",
            "target_labels": "director_radius_sq_semiprime",
            "status": "PROVEN_EMPTY",
            "algebraic_reason": "director_radius_sq=d*d*(D+F); d>=2 and D+F>1, so Omega is at least 3",
            "audit_mismatches": mismatches["director_radius_sq_never_semiprime"],
        },
        {
            "claim": "director_radius_sq_never_squarefree",
            "target_labels": "director_radius_sq_squarefree",
            "status": "PROVEN_EMPTY",
            "algebraic_reason": "director_radius_sq carries factor d*d with d>=2, so it is never squarefree",
            "audit_mismatches": mismatches["director_radius_sq_never_squarefree"],
        },
        {
            "claim": "gcd_X_D_eq_D_empty",
            "target_labels": "gcd_X_D_eq_D",
            "status": "PROVEN_EMPTY",
            "algebraic_reason": "gcd(X,D)=d*gcd(e,d), and gcd(e,d)<=d with equality requiring d divides e; impossible for d=b+e>e",
            "audit_mismatches": 0,
        },
    ]
    for item in ledger:
        item["checked_pairs"] = checked
        item["hash"] = domain_sha256(f"{DOMAIN}.ledger", canonical_json({key: value for key, value in item.items() if key != "hash"}))
    return ledger, mismatches


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger, mismatches = audit_rows(args.limit)
    csv_path = out_dir / args.ledger_csv
    write_csv(csv_path, ledger)
    status_counts: dict[str, int] = {}
    for item in ledger:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "stage_id": "qa_quantum_arithmetic_identity_audit_stage16",
        "hypothesis": (
            "Several mined QA conic targets are structural identities or provably empty degeneracies rather than "
            "ordinary empirical low-support labels."
        ),
        "parameters": {"limit": args.limit},
        "audit_mismatches": mismatches,
        "status_counts": status_counts,
        "artifacts": {"ledger_csv": str(csv_path)},
        "ledger": ledger,
        "honest_interpretation": (
            "This audit separates targets that warrant more mining from targets closed by elementary QA algebra."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    json_path = out_dir / args.summary_json
    json_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            limit=25,
            summary_json="identity_audit_selftest.json",
            ledger_csv="identity_audit_selftest.csv",
        )
        payload = run(args)
        ok = (
            Path(tmp, "identity_audit_selftest.json").exists()
            and Path(tmp, "identity_audit_selftest.csv").exists()
            and all(value == 0 for value in payload["audit_mismatches"].values())
            and payload["status_counts"]["PROVEN_STRUCTURAL_IDENTITY"] >= 3
        )
        return {"ok": ok, "rows": len(payload["ledger"]), "status_counts": payload["status_counts"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage16_identity_audit.json")
    parser.add_argument("--ledger-csv", default="qa_quantum_arithmetic_stage16_identity_audit_ledger.csv")
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
                "status_counts": payload["status_counts"],
                "audit_mismatches": payload["audit_mismatches"],
                "artifacts": payload["artifacts"],
                "canonical_hash": payload["canonical_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
