#!/usr/bin/env python3
"""Stage 25 proof closure for directrix_distance_integer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_DIRECTRIX_DIVISIBILITY_CLOSURE_STAGE25.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def directrix_distance_integer(b: int, e: int) -> bool:
    d = b + e
    return (d * d * d) % e == 0


def reduced_directrix_integer(b: int, e: int) -> bool:
    return (b * b * b) % e == 0


def factor(n: int) -> dict[int, int]:
    remaining = n
    out: dict[int, int] = {}
    while remaining % 2 == 0:
        out[2] = out.get(2, 0) + 1
        remaining //= 2
    p = 3
    while p * p <= remaining:
        while remaining % p == 0:
            out[p] = out.get(p, 0) + 1
            remaining //= p
        p += 2
    if remaining > 1:
        out[remaining] = out.get(remaining, 0) + 1
    return out


def kernel3(e: int) -> int:
    out = 1
    for prime, exponent in factor(e).items():
        out *= prime ** math.ceil(exponent / 3)
    return out


def kernel_classifier(b: int, e: int) -> bool:
    return b % kernel3(e) == 0


def audit_window(b_max: int, e_max: int) -> dict[str, object]:
    directrix_support = 0
    reduction_mismatches = []
    kernel_mismatches = []
    kernel_histogram: dict[str, int] = {}
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            direct = directrix_distance_integer(b, e)
            reduced = reduced_directrix_integer(b, e)
            kernel = kernel_classifier(b, e)
            if direct:
                directrix_support += 1
            if direct != reduced:
                reduction_mismatches.append({"b": b, "e": e})
            if reduced != kernel:
                kernel_mismatches.append({"b": b, "e": e, "kernel3_e": kernel3(e)})
            if b == 1:
                k = str(kernel3(e))
                kernel_histogram[k] = kernel_histogram.get(k, 0) + 1
    return {
        "b_max": b_max,
        "e_max": e_max,
        "pair_count": b_max * e_max,
        "directrix_support": directrix_support,
        "directrix_base_rate": directrix_support / (b_max * e_max),
        "reduction_mismatch_count": len(reduction_mismatches),
        "first_reduction_mismatches": reduction_mismatches[:20],
        "kernel_mismatch_count": len(kernel_mismatches),
        "first_kernel_mismatches": kernel_mismatches[:20],
        "distinct_kernel_count_for_e_window": len(kernel_histogram),
    }


def sample_rows(b_max: int, e_max: int, limit: int) -> list[dict[str, object]]:
    rows = []
    for e in range(1, e_max + 1):
        k = kernel3(e)
        for b in range(1, b_max + 1):
            if directrix_distance_integer(b, e):
                rows.append(
                    {
                        "b": b,
                        "e": e,
                        "d": b + e,
                        "kernel3_e": k,
                        "directrix_integer": True,
                        "minimal_b_condition": f"b divisible by {k}",
                    }
                )
                if len(rows) >= limit:
                    return rows
    return rows


def build_ledger(args: argparse.Namespace) -> dict[str, object]:
    audit = audit_window(args.b_max, args.e_max)
    rows = sample_rows(args.b_max, args.e_max, args.sample_limit)
    ledger: dict[str, object] = {
        "stage_id": "qa_quantum_arithmetic_stage25_directrix_divisibility_closure",
        "theorem_status": "PROVEN_STRUCTURAL_DIVISIBILITY_REDUCTION",
        "theorem_statement": (
            "For integers b,e >= 1 and d=b+e, the exact directrix integrality condition "
            "e | d*d*d is equivalent to e | b*b*b. Equivalently, if "
            "kernel3(e)=product p^ceil(v_p(e)/3), then directrix_distance_integer holds "
            "exactly when kernel3(e) divides b."
        ),
        "proof": [
            "The directrix target tests whether d*d*d/e is an integer, i.e. whether e divides d*d*d.",
            "Since d=b+e, d is congruent to b modulo e.",
            "Congruence is compatible with multiplication, so d*d*d is congruent to b*b*b modulo e.",
            "Therefore e divides d*d*d iff e divides b*b*b.",
            "For e=product p^r, the condition e|b*b*b is equivalent prime-by-prime to r <= 3*v_p(b).",
            "Thus v_p(b) >= ceil(r/3) for every prime p|e, which is exactly kernel3(e)|b.",
        ],
        "relation_to_stage21": (
            "Stage 21 already ran the b_only/e_only controls. e_only lift was 2.43, b_only lift was 2.61, "
            "qa_orbit_family9 and qa_orbit_id9 lifted 3.93, drop_raw_be lifted 6.57, and derived_products lifted "
            "7.17. This closure explains the target as generator divisibility e|b*b*b; orbit features can remain "
            "predictive, but the target is not an unexplained conic invariant."
        ),
        "audit": audit,
        "sample_positive_rows": rows,
        "parameters": {
            "b_max": args.b_max,
            "e_max": args.e_max,
            "sample_limit": args.sample_limit,
        },
        "cert_readiness": (
            "Ready for a small structural theorem cert if desired; otherwise this should be merged into the "
            "algebraic-status ledger as PROVEN_STRUCTURAL_DIVISIBILITY_REDUCTION."
        ),
    }
    payload = canonical_json({k: v for k, v in ledger.items() if k != "canonical_hash"})
    ledger["canonical_hash"] = domain_sha256(DOMAIN, payload)
    return ledger


def write_ledger(ledger: dict[str, object], results_dir: Path, summary_json: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / summary_json
    out.write_text(canonical_json(ledger) + "\n")
    return out


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            b_max=40,
            e_max=40,
            sample_limit=10,
            results_dir=Path(tmp),
            summary_json="stage25_self_test.json",
        )
        ledger = build_ledger(args)
        out = write_ledger(ledger, args.results_dir, args.summary_json)
        reread = json.loads(out.read_text())
        audit = reread["audit"]
        ok = (
            audit["directrix_support"] > 0
            and audit["reduction_mismatch_count"] == 0
            and audit["kernel_mismatch_count"] == 0
            and reread["theorem_status"] == "PROVEN_STRUCTURAL_DIVISIBILITY_REDUCTION"
            and len(reread["canonical_hash"]) == 64
        )
        print(
            canonical_json(
                {
                    "ok": ok,
                    "support": audit["directrix_support"],
                    "reduction_mismatches": audit["reduction_mismatch_count"],
                    "kernel_mismatches": audit["kernel_mismatch_count"],
                }
            )
        )
        raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-max", type=int, default=400)
    parser.add_argument("--e-max", type=int, default=400)
    parser.add_argument("--sample-limit", type=int, default=25)
    parser.add_argument("--results-dir", type=Path, default=Path("results/qa_quantum_arithmetic_mining_001"))
    parser.add_argument(
        "--summary-json",
        default="qa_quantum_arithmetic_stage25_directrix_divisibility_closure.json",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    ledger = build_ledger(args)
    out = write_ledger(ledger, args.results_dir, args.summary_json)
    audit = ledger["audit"]
    print(
        canonical_json(
            {
                "ok": True,
                "summary_json": str(out),
                "theorem_status": ledger["theorem_status"],
                "directrix_support": audit["directrix_support"],
                "reduction_mismatch_count": audit["reduction_mismatch_count"],
                "kernel_mismatch_count": audit["kernel_mismatch_count"],
                "canonical_hash": ledger["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
