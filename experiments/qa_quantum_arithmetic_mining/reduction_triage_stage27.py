#!/usr/bin/env python3
"""Stage 27 reduction triage for remaining QA arithmetic mining survivors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_REDUCTION_TRIAGE_STAGE27.v1"

TARGETS = (
    "D_plus_F_semiprime",
    "D_plus_F_squarefree",
    "semi_latus_squarefree",
    "semi_latus_distinct_omega_2",
    "ecc_den_smooth_13",
    "polar_scale_X_plus_F_semiprime",
)


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def prime_factor_multiset(n: int) -> list[int]:
    factors: list[int] = []
    remaining = n
    while remaining > 1 and remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    p = 3
    while p * p <= remaining:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p
        p += 2
    if remaining > 1:
        factors.append(remaining)
    return factors


def factor_count(n: int) -> int:
    return len(prime_factor_multiset(n))


def distinct_factor_count(n: int) -> int:
    return len(set(prime_factor_multiset(n)))


def is_semiprime(n: int) -> bool:
    return factor_count(n) == 2


def is_squarefree(n: int) -> bool:
    factors = prime_factor_multiset(n)
    return len(factors) == len(set(factors))


def is_smooth(n: int, bound: int) -> bool:
    factors = prime_factor_multiset(n)
    return bool(factors) and max(factors) <= bound


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    D = d * d
    F = a * b
    X = e * d
    return {
        "b": b,
        "e": e,
        "d": d,
        "a": a,
        "D": D,
        "F": F,
        "X": X,
        "D_plus_F": D + F,
        "X_plus_F": X + F,
    }


def label_for(row: dict[str, int], target: str) -> bool:
    if target == "D_plus_F_semiprime":
        return is_semiprime(row["D_plus_F"])
    if target == "D_plus_F_squarefree":
        return is_squarefree(row["D_plus_F"])
    if target == "semi_latus_squarefree":
        return is_squarefree(row["F"])
    if target == "semi_latus_distinct_omega_2":
        return distinct_factor_count(row["F"]) == 2
    if target == "ecc_den_smooth_13":
        reduced_den = row["d"] // math.gcd(row["e"], row["d"])
        return is_smooth(reduced_den, 13)
    if target == "polar_scale_X_plus_F_semiprime":
        return is_semiprime(row["X_plus_F"])
    raise ValueError(f"unknown target: {target}")


def reduced_condition(row: dict[str, int], target: str) -> bool | None:
    b = row["b"]
    e = row["e"]
    d = row["d"]
    a = row["a"]
    if target == "semi_latus_squarefree":
        return is_squarefree(b) and is_squarefree(a) and math.gcd(b, a) == 1
    if target == "semi_latus_distinct_omega_2":
        return len(set(prime_factor_multiset(b)) | set(prime_factor_multiset(a))) == 2
    if target == "ecc_den_smooth_13":
        reduced_den = d // math.gcd(e, b)
        return is_smooth(reduced_den, 13)
    return None


def target_claim(target: str) -> dict[str, str]:
    claims = {
        "D_plus_F_semiprime": {
            "classification": "EMPIRICAL_ONLY",
            "reduction": "D+F = 2*b*b + 4*b*e + e*e; semiprime status remains an irreducible quadratic-form target.",
            "next_action": "Deprioritize until G_square and h_integer are closed; do not certify without a separate number-theory theorem.",
        },
        "D_plus_F_squarefree": {
            "classification": "EMPIRICAL_ONLY",
            "reduction": "D+F = 2*b*b + 4*b*e + e*e; squarefree status remains an irreducible quadratic-form target.",
            "next_action": "Keep as an empirical sieve target; not cert-ready from this triage alone.",
        },
        "semi_latus_squarefree": {
            "classification": "REDUCIBLE_TO_COMPONENT_FACTORIZATION",
            "reduction": "F=a*b is squarefree iff b and a are squarefree and gcd(a,b)=1.",
            "next_action": "Document as a component-factorization reduction, not as new QA geometry.",
        },
        "semi_latus_distinct_omega_2": {
            "classification": "REDUCIBLE_TO_COMPONENT_FACTORIZATION",
            "reduction": "distinct_omega(F)=2 iff the union of prime divisors of b and a has size 2.",
            "next_action": "Document as a component-factorization reduction; no independent cert priority.",
        },
        "ecc_den_smooth_13": {
            "classification": "REDUCIBLE_TO_COMPONENT_FACTORIZATION",
            "reduction": "reduced denominator of e/d is d/gcd(e,d)=d/gcd(e,b); target is 13-smoothness of that reduced denominator.",
            "next_action": "Document as an exact reduced-ratio identity; no proof machinery needed.",
        },
        "polar_scale_X_plus_F_semiprime": {
            "classification": "EMPIRICAL_ONLY",
            "reduction": "X+F = b*b + 3*b*e + e*e; semiprime status remains a discriminant-5 quadratic-form target.",
            "next_action": "Keep as a later quadratic-form candidate after G_square/h_integer; not a product leakage closure.",
        },
    }
    return claims[target]


def audit_target(target: str, b_max: int, e_max: int) -> dict[str, object]:
    support = 0
    mismatch_count = 0
    mismatches: list[dict[str, int]] = []
    sample_positives: list[dict[str, int]] = []
    formula_mismatch_count = 0
    formula_mismatches: list[dict[str, int]] = []
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            row = qa_values(b, e)
            if row["D_plus_F"] != 2 * b * b + 4 * b * e + e * e:
                formula_mismatch_count += 1
                if len(formula_mismatches) < 10:
                    formula_mismatches.append({"b": b, "e": e})
            if row["X_plus_F"] != b * b + 3 * b * e + e * e:
                formula_mismatch_count += 1
                if len(formula_mismatches) < 10:
                    formula_mismatches.append({"b": b, "e": e})
            actual = label_for(row, target)
            if actual:
                support += 1
                if len(sample_positives) < 8:
                    sample_positives.append(
                        {
                            "b": b,
                            "e": e,
                            "d": row["d"],
                            "a": row["a"],
                            "F": row["F"],
                            "D_plus_F": row["D_plus_F"],
                            "X_plus_F": row["X_plus_F"],
                        }
                    )
            reduced = reduced_condition(row, target)
            if reduced is not None and actual != reduced:
                mismatch_count += 1
                if len(mismatches) < 10:
                    mismatches.append({"b": b, "e": e})
    claim = target_claim(target)
    pair_count = b_max * e_max
    has_reduction = reduced_condition(qa_values(1, 1), target) is not None
    verification = "exact_reduction_verified" if has_reduction else "formula_and_support_audit_only"
    if has_reduction and mismatch_count:
        verification = "reduction_failed"
    return {
        "target": target,
        "classification": claim["classification"],
        "reduction": claim["reduction"],
        "next_action": claim["next_action"],
        "verification": verification,
        "b_max": b_max,
        "e_max": e_max,
        "pair_count": pair_count,
        "support": support,
        "base_rate": support / pair_count,
        "reduction_mismatch_count": mismatch_count,
        "first_reduction_mismatches": mismatches,
        "formula_mismatch_count": formula_mismatch_count,
        "first_formula_mismatches": formula_mismatches,
        "sample_positive_rows": sample_positives,
    }


def build_ledger(args: argparse.Namespace) -> dict[str, object]:
    rows = [audit_target(target, args.b_max, args.e_max) for target in TARGETS]
    verdict_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["classification"])
        verdict_counts[key] = verdict_counts.get(key, 0) + 1
    ledger: dict[str, object] = {
        "stage_id": "qa_quantum_arithmetic_stage27_reduction_triage",
        "purpose": "Classify six remaining Stage 17/18 survivors before opening new mining tiers or cert work.",
        "targets": list(TARGETS),
        "parameters": {"b_max": args.b_max, "e_max": args.e_max},
        "classification_counts": verdict_counts,
        "rows": rows,
        "honest_interpretation": (
            "Three targets reduce exactly to component or reduced-ratio factorization checks over the audited window. "
            "The two D_plus_F targets and polar_scale_X_plus_F_semiprime reduce only to irreducible quadratic forms here; "
            "their support is empirical and should not be promoted to theorem language without a separate proof."
        ),
    }
    payload = canonical_json({k: v for k, v in ledger.items() if k != "canonical_hash"})
    ledger["canonical_hash"] = domain_sha256(DOMAIN, payload)
    return ledger


def write_outputs(ledger: dict[str, object], results_dir: Path, summary_json: str, ledger_csv: str) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / summary_json
    csv_path = results_dir / ledger_csv
    json_path.write_text(canonical_json(ledger) + "\n")
    fieldnames = [
        "target",
        "classification",
        "verification",
        "support",
        "base_rate",
        "reduction_mismatch_count",
        "formula_mismatch_count",
        "reduction",
        "next_action",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ledger["rows"]:
            writer.writerow({key: row[key] for key in fieldnames})
    return json_path, csv_path


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            b_max=30,
            e_max=30,
            results_dir=Path(tmp),
            summary_json="stage27_selftest.json",
            ledger_csv="stage27_selftest.csv",
        )
        ledger = build_ledger(args)
        json_path, csv_path = write_outputs(ledger, args.results_dir, args.summary_json, args.ledger_csv)
        rows = ledger["rows"]
        ok = (
            json_path.exists()
            and csv_path.exists()
            and len(rows) == len(TARGETS)
            and ledger["classification_counts"].get("REDUCIBLE_TO_COMPONENT_FACTORIZATION") == 3
            and ledger["classification_counts"].get("EMPIRICAL_ONLY") == 3
            and all(row["formula_mismatch_count"] == 0 for row in rows)
            and all(
                row["reduction_mismatch_count"] == 0
                for row in rows
                if row["verification"] == "exact_reduction_verified"
            )
            and len(ledger["canonical_hash"]) == 64
        )
        print(
            canonical_json(
                {
                    "ok": ok,
                    "rows": len(rows),
                    "classification_counts": ledger["classification_counts"],
                    "canonical_hash": ledger["canonical_hash"],
                }
            )
        )
        raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-max", type=int, default=400)
    parser.add_argument("--e-max", type=int, default=400)
    parser.add_argument("--results-dir", type=Path, default=Path("results/qa_quantum_arithmetic_mining_001"))
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage27_reduction_triage.json")
    parser.add_argument("--ledger-csv", default="qa_quantum_arithmetic_stage27_reduction_triage.csv")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    ledger = build_ledger(args)
    json_path, csv_path = write_outputs(ledger, args.results_dir, args.summary_json, args.ledger_csv)
    print(
        canonical_json(
            {
                "ok": True,
                "summary_json": str(json_path),
                "ledger_csv": str(csv_path),
                "classification_counts": ledger["classification_counts"],
                "canonical_hash": ledger["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
