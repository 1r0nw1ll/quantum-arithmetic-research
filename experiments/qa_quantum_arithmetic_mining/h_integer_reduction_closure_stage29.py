#!/usr/bin/env python3
"""Stage 29 proof closure for h_integer via coprime square parts of F=a*b."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_H_INTEGER_REDUCTION_CLOSURE_STAGE29.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    F = a * b
    return {"b": b, "e": e, "d": d, "a": a, "F": F}


def h_integer_target(b: int, e: int) -> bool:
    return is_square(qa_values(b, e)["F"])


def reduced_square_condition(b: int, e: int) -> bool:
    row = qa_values(b, e)
    divisor = math.gcd(row["a"], row["b"])
    a_reduced = row["a"] // divisor
    b_reduced = row["b"] // divisor
    return is_square(a_reduced) and is_square(b_reduced)


def brute_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], dict[str, int]]:
    out: dict[tuple[int, int], dict[str, int]] = {}
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            row = qa_values(b, e)
            root = math.isqrt(row["F"])
            if root * root == row["F"]:
                divisor = math.gcd(row["a"], row["b"])
                out[(b, e)] = {
                    **row,
                    "sqrt_F": root,
                    "g": divisor,
                    "a_reduced": row["a"] // divisor,
                    "b_reduced": row["b"] // divisor,
                }
    return out


def generated_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], dict[str, int]]:
    out: dict[tuple[int, int], dict[str, int]] = {}
    s_max = math.isqrt(b_max + 2 * e_max) + 2
    for g in range(1, b_max + 2 * e_max + 1):
        for r in range(1, math.isqrt(b_max // g) + 1):
            b = g * r * r
            if b > b_max:
                continue
            for s in range(r + 1, s_max + 1):
                delta = g * (s * s - r * r)
                if delta % 2 != 0:
                    continue
                e = delta // 2
                if e < 1 or e > e_max:
                    continue
                a = g * s * s
                if a != b + 2 * e:
                    continue
                key = (b, e)
                prior = out.get(key)
                if prior is None or g < prior["g"]:
                    out[key] = {
                        "b": b,
                        "e": e,
                        "a": a,
                        "d": b + e,
                        "F": a * b,
                        "sqrt_F": g * r * s,
                        "g": g,
                        "r": r,
                        "s": s,
                    }
    return out


def reduction_audit(b_max: int, e_max: int) -> dict[str, object]:
    support = 0
    reduction_mismatches = []
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            actual = h_integer_target(b, e)
            reduced = reduced_square_condition(b, e)
            if actual:
                support += 1
            if actual != reduced:
                reduction_mismatches.append({"b": b, "e": e})
    return {
        "b_max": b_max,
        "e_max": e_max,
        "pair_count": b_max * e_max,
        "support": support,
        "base_rate": support / (b_max * e_max),
        "reduction_mismatch_count": len(reduction_mismatches),
        "first_reduction_mismatches": reduction_mismatches[:20],
    }


def generation_audit(b_max: int, e_max: int) -> dict[str, object]:
    brute = brute_solutions(b_max, e_max)
    generated = generated_solutions(b_max, e_max)
    misses = sorted(key for key in brute if key not in generated)
    invalid = []
    for (b, e), witness in sorted(generated.items()):
        sqrt_F = witness["sqrt_F"]
        if qa_values(b, e)["F"] != sqrt_F * sqrt_F:
            invalid.append({"b": b, "e": e, "sqrt_F": sqrt_F})
    sample_witnesses = [
        generated[key]
        for key in sorted(generated)[:20]
    ]
    return {
        "b_max": b_max,
        "e_max": e_max,
        "brute_solution_count": len(brute),
        "generated_solution_count": len(generated),
        "generated_hit_count": len(brute) - len(misses),
        "miss_count": len(misses),
        "invalid_generated_count": len(invalid),
        "first_misses": [{"b": b, "e": e} for b, e in misses[:20]],
        "first_invalid_generated": invalid[:20],
        "sample_witnesses": sample_witnesses,
    }


def build_ledger(args: argparse.Namespace) -> dict[str, object]:
    reduction = reduction_audit(args.b_max, args.e_max)
    generation = generation_audit(args.b_max, args.e_max)
    ledger: dict[str, object] = {
        "stage_id": "qa_quantum_arithmetic_stage29_h_integer_reduction_closure",
        "theorem_status": "PROVEN_STRUCTURAL_SQUARE_PART_REDUCTION",
        "theorem_statement": (
            "For integers b,e >= 1, define d=b+e, a=b+2*e, F=a*b, and h=sqrt(F)*d. "
            "Then h is an integer iff, for g=gcd(a,b), both a/g and b/g are perfect squares."
        ),
        "qa_reduction": [
            "Since d is an integer, h=sqrt(F)*d is an integer exactly when F is a square.",
            "The target is therefore is_square(a*b), with a=b+2*e.",
        ],
        "proof": [
            "Let g=gcd(a,b), a=g*A, and b=g*B with gcd(A,B)=1.",
            "Then a*b=g*g*A*B.",
            "The factor g*g is already a square.",
            "Because A and B are coprime, A*B is a square iff A and B are each perfect squares.",
            "Therefore F=a*b is square iff a/g and b/g are both squares.",
        ],
        "derived_enumeration": [
            "The reduction gives a bounded witness enumeration: choose g,r,s >= 1 with s>r.",
            "Set b=g*r*r and a=g*s*s.",
            "The QA constraint a=b+2*e requires e=g*(s*s-r*r)/2, so e is retained only when that numerator is even and positive.",
            "This enumeration is included as an audit cross-check, but the theorem is framed as a structural reduction rather than a geometry-parametrization family.",
        ],
        "reduction_audit": reduction,
        "generation_audit": generation,
        "parameters": {"b_max": args.b_max, "e_max": args.e_max},
        "cert_readiness": (
            "Ready for a structural reduction cert if promoted. Strength class matches directrix divisibility [530], "
            "not D_plus_F/G_square parametrization families."
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
            b_max=60,
            e_max=60,
            results_dir=Path(tmp),
            summary_json="stage29_selftest.json",
        )
        ledger = build_ledger(args)
        out = write_ledger(ledger, args.results_dir, args.summary_json)
        reread = json.loads(out.read_text())
        reduction = reread["reduction_audit"]
        generation = reread["generation_audit"]
        ok = (
            out.exists()
            and reduction["support"] > 0
            and reduction["reduction_mismatch_count"] == 0
            and generation["brute_solution_count"] == generation["generated_solution_count"]
            and generation["miss_count"] == 0
            and generation["invalid_generated_count"] == 0
            and reread["theorem_status"] == "PROVEN_STRUCTURAL_SQUARE_PART_REDUCTION"
            and len(reread["canonical_hash"]) == 64
        )
        print(
            canonical_json(
                {
                    "ok": ok,
                    "support": reduction["support"],
                    "reduction_mismatches": reduction["reduction_mismatch_count"],
                    "generation_misses": generation["miss_count"],
                    "invalid_generated": generation["invalid_generated_count"],
                }
            )
        )
        raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-max", type=int, default=400)
    parser.add_argument("--e-max", type=int, default=400)
    parser.add_argument("--results-dir", type=Path, default=Path("results/qa_quantum_arithmetic_mining_001"))
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage29_h_integer_reduction_closure.json")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    ledger = build_ledger(args)
    out = write_ledger(ledger, args.results_dir, args.summary_json)
    reduction = ledger["reduction_audit"]
    generation = ledger["generation_audit"]
    print(
        canonical_json(
            {
                "ok": True,
                "summary_json": str(out),
                "theorem_status": ledger["theorem_status"],
                "support": reduction["support"],
                "reduction_mismatch_count": reduction["reduction_mismatch_count"],
                "brute_solution_count": generation["brute_solution_count"],
                "generated_solution_count": generation["generated_solution_count"],
                "miss_count": generation["miss_count"],
                "invalid_generated_count": generation["invalid_generated_count"],
                "canonical_hash": ledger["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
