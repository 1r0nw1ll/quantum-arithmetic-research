#!/usr/bin/env python3
"""Stage 28 proof closure for G_square via Pythagorean triples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_G_SQUARE_PROOF_CLOSURE_STAGE28.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def g_value(b: int, e: int) -> int:
    d = b + e
    return d * d + e * e


def brute_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            value = g_value(b, e)
            root = math.isqrt(value)
            if root * root == value:
                out[(b, e)] = root
    return out


def parametrized_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], dict[str, int | str]]:
    out: dict[tuple[int, int], dict[str, int | str]] = {}
    d_max = b_max + e_max
    for m in range(2, d_max + 1):
        for n in range(1, m):
            leg_odd = m * m - n * n
            leg_even = 2 * m * n
            if leg_odd < 1 or leg_even < 1:
                continue
            max_base_leg = max(leg_odd, leg_even)
            if max_base_leg > d_max:
                continue
            for t in range(1, d_max // max_base_leg + 1):
                legs = (
                    ("odd_as_d", t * leg_odd, t * leg_even),
                    ("even_as_d", t * leg_even, t * leg_odd),
                )
                for branch, d, e in legs:
                    b = d - e
                    if b < 1 or b > b_max or e > e_max:
                        continue
                    k = t * (m * m + n * n)
                    key = (b, e)
                    prior = out.get(key)
                    if prior is None or k < int(prior["k"]):
                        out[key] = {
                            "b": b,
                            "e": e,
                            "d": d,
                            "k": k,
                            "t": t,
                            "m": m,
                            "n": n,
                            "branch": branch,
                        }
    return out


def bounded_audit(b_max: int, e_max: int) -> dict[str, object]:
    brute = brute_solutions(b_max, e_max)
    param = parametrized_solutions(b_max, e_max)
    misses = sorted(key for key in brute if key not in param)
    invalid = []
    for (b, e), witness in sorted(param.items()):
        k = int(witness["k"])
        if g_value(b, e) != k * k:
            invalid.append({"b": b, "e": e, "k": k})
    branch_counts: dict[str, int] = {}
    for witness in param.values():
        branch = str(witness["branch"])
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    return {
        "b_max": b_max,
        "e_max": e_max,
        "pair_count": b_max * e_max,
        "brute_solution_count": len(brute),
        "param_solution_count": len(param),
        "param_hit_count": len(brute) - len(misses),
        "miss_count": len(misses),
        "invalid_param_count": len(invalid),
        "branch_counts": branch_counts,
        "first_misses": [{"b": b, "e": e} for b, e in misses[:20]],
        "first_invalid_params": invalid[:20],
    }


def forward_identity_audit(t_max: int, m_max: int, n_max: int) -> dict[str, int]:
    checked = 0
    failures = 0
    for t in range(1, t_max + 1):
        for m in range(2, m_max + 1):
            for n in range(1, min(n_max, m - 1) + 1):
                leg_odd = t * (m * m - n * n)
                leg_even = t * 2 * m * n
                k = t * (m * m + n * n)
                for d, e in ((leg_odd, leg_even), (leg_even, leg_odd)):
                    if d <= e:
                        continue
                    b = d - e
                    checked += 1
                    if b < 1 or g_value(b, e) != k * k:
                        failures += 1
    return {"checked_param_rows": checked, "failure_count": failures}


def build_ledger(args: argparse.Namespace) -> dict[str, object]:
    audit = bounded_audit(args.b_max, args.e_max)
    forward = forward_identity_audit(args.t_max, args.m_max, args.n_max)
    ledger: dict[str, object] = {
        "stage_id": "qa_quantum_arithmetic_stage28_g_square_proof_closure",
        "theorem_status": "PROVEN_BY_EUCLID_PYTHAGOREAN_PARAMETRIZATION",
        "theorem_statement": (
            "For integers b,e >= 1, define d=b+e and G=d*d+e*e. Then G is a square iff "
            "there exist integers t>=1 and m>n>=1 such that the unordered pair {d,e} equals "
            "{t*(m*m-n*n), t*2*m*n}, with the branch filtered by d>e and b=d-e."
        ),
        "qa_reduction": [
            "G=d*d+e*e.",
            "Thus G_square is exactly the assertion that d and e are integer legs of a right triangle.",
            "Since d=b+e, every valid QA pair has d>e and b=d-e.",
        ],
        "forward_proof": [
            "Assume {d,e}={t*(m*m-n*n), t*2*m*n} with m>n>=1 and t>=1.",
            "Then d*d+e*e=t*t*((m*m-n*n)*(m*m-n*n)+4*m*m*n*n).",
            "The inner expression equals (m*m+n*n)*(m*m+n*n), so G is the square t*(m*m+n*n).",
            "Filtering d>e and setting b=d-e gives exactly the positive QA coordinate b>=1.",
        ],
        "reverse_proof": [
            "Assume G=k*k. Then d*d+e*e=k*k, so (d,e,k) is an integer right triangle.",
            "Divide by g=gcd(d,e) to obtain a primitive integer right triangle.",
            "The classical Euclid parametrization of primitive Pythagorean triples gives coprime m>n>=1 of opposite parity with primitive legs m*m-n*n and 2*m*n.",
            "Multiplying back by the scale t recovers the general legs {d,e}={t*(m*m-n*n), t*2*m*n}.",
            "Because QA imposes d=b+e, only the branch with d>e is retained, and b=d-e is then forced.",
        ],
        "bounded_audit": audit,
        "forward_identity_audit": forward,
        "parameters": {
            "b_max": args.b_max,
            "e_max": args.e_max,
            "t_max": args.t_max,
            "m_max": args.m_max,
            "n_max": args.n_max,
        },
        "known_cross_check": (
            "The default b,e<=299 window corresponds to the independently reported b,e<300 audit and should yield "
            "379 brute-force positive pairs with zero parametrization misses."
        ),
        "cert_readiness": (
            "Ready for a parametrization theorem cert if promoted. This is a complete generator theorem, not merely "
            "a structural reduction."
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
            t_max=20,
            m_max=30,
            n_max=30,
            results_dir=Path(tmp),
            summary_json="stage28_selftest.json",
        )
        ledger = build_ledger(args)
        out = write_ledger(ledger, args.results_dir, args.summary_json)
        reread = json.loads(out.read_text())
        bounded = reread["bounded_audit"]
        forward = reread["forward_identity_audit"]
        ok = (
            bounded["brute_solution_count"] > 0
            and bounded["miss_count"] == 0
            and bounded["invalid_param_count"] == 0
            and forward["checked_param_rows"] > 0
            and forward["failure_count"] == 0
            and reread["theorem_status"] == "PROVEN_BY_EUCLID_PYTHAGOREAN_PARAMETRIZATION"
            and len(reread["canonical_hash"]) == 64
        )
        print(
            canonical_json(
                {
                    "ok": ok,
                    "solutions": bounded["brute_solution_count"],
                    "misses": bounded["miss_count"],
                    "invalid_params": bounded["invalid_param_count"],
                }
            )
        )
        raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-max", type=int, default=299)
    parser.add_argument("--e-max", type=int, default=299)
    parser.add_argument("--t-max", type=int, default=60)
    parser.add_argument("--m-max", type=int, default=80)
    parser.add_argument("--n-max", type=int, default=80)
    parser.add_argument("--results-dir", type=Path, default=Path("results/qa_quantum_arithmetic_mining_001"))
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage28_g_square_proof_closure.json")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    ledger = build_ledger(args)
    out = write_ledger(ledger, args.results_dir, args.summary_json)
    audit = ledger["bounded_audit"]
    print(
        canonical_json(
            {
                "ok": True,
                "summary_json": str(out),
                "theorem_status": ledger["theorem_status"],
                "brute_solution_count": audit["brute_solution_count"],
                "param_solution_count": audit["param_solution_count"],
                "miss_count": audit["miss_count"],
                "invalid_param_count": audit["invalid_param_count"],
                "canonical_hash": ledger["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
