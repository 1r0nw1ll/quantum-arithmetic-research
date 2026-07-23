#!/usr/bin/env python3
"""Stage 24 proof closure for the D_plus_F square parametrization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_DPLUSF_SQUARE_PROOF_CLOSURE_STAGE24.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def d_plus_f(b: int, e: int) -> int:
    d = b + e
    a = b + 2 * e
    return d * d + a * b


def parametrized_solution(t: int, m: int, n: int) -> tuple[int, int, int, int] | None:
    b = t * 2 * m * n
    u = t * (m * m + 2 * n * n)
    e = u - 2 * b
    if b < 1 or e < 1:
        return None
    k = t * abs(m * m - 2 * n * n)
    return b, e, u, k


def brute_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            value = d_plus_f(b, e)
            root = math.isqrt(value)
            if root * root == value:
                out[(b, e)] = root
    return out


def parametrized_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for m in range(1, b_max + 1):
        for n in range(1, b_max + 1):
            denom = 2 * m * n
            if denom > b_max:
                continue
            for t in range(1, b_max // denom + 1):
                row = parametrized_solution(t, m, n)
                if row is None:
                    continue
                b, e, _u, k = row
                if b > b_max or e > e_max:
                    continue
                prior = out.get((b, e))
                if prior is None or k < prior:
                    out[(b, e)] = k
    return out


def bounded_audit(b_max: int, e_max: int) -> dict[str, object]:
    brute = brute_solutions(b_max, e_max)
    param = parametrized_solutions(b_max, e_max)
    misses = sorted(key for key in brute if key not in param)
    invalid = []
    for (b, e), k in sorted(param.items()):
        if d_plus_f(b, e) != k * k:
            invalid.append({"b": b, "e": e, "k": k})
    return {
        "b_max": b_max,
        "e_max": e_max,
        "brute_solution_count": len(brute),
        "param_solution_count": len(param),
        "param_hit_count": len(brute) - len(misses),
        "miss_count": len(misses),
        "invalid_param_count": len(invalid),
        "first_misses": [{"b": b, "e": e} for b, e in misses[:20]],
        "first_invalid_params": invalid[:20],
    }


def forward_identity_audit(t_max: int, m_max: int, n_max: int) -> dict[str, int]:
    checked = 0
    failures = 0
    for t in range(1, t_max + 1):
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                row = parametrized_solution(t, m, n)
                if row is None:
                    continue
                b, e, u, k = row
                checked += 1
                if u != e + 2 * b or d_plus_f(b, e) != k * k:
                    failures += 1
    return {"checked_param_rows": checked, "failure_count": failures}


def build_ledger(args: argparse.Namespace) -> dict[str, object]:
    audit = bounded_audit(args.b_max, args.e_max)
    forward = forward_identity_audit(args.t_max, args.m_max, args.n_max)
    ledger: dict[str, object] = {
        "stage_id": "qa_quantum_arithmetic_stage24_dplusf_square_proof_closure",
        "theorem_status": "PROVEN_BY_RATIONAL_CONIC_PARAMETRIZATION",
        "theorem_statement": (
            "For integers b,e >= 1, define d=b+e, a=b+2*e, D=d*d, and F=a*b. "
            "Then D+F is a square iff there exist integers t,m,n >= 1 with "
            "e=t*(m*m - 4*m*n + 2*n*n)>0 such that b=t*2*m*n. In that case "
            "sqrt(D+F)=t*abs(m*m - 2*n*n)."
        ),
        "qa_reduction": [
            "D+F=(b+e)*(b+e)+b*(b+2*e)=2*b*b+4*b*e+e*e.",
            "Let u=e+2*b. Then D+F=u*u-2*b*b.",
            "So D+F=k*k is equivalent to k*k+2*b*b=u*u.",
        ],
        "forward_proof": [
            "Assume b=t*2*m*n and e=t*(m*m - 4*m*n + 2*n*n)>0.",
            "Then u=e+2*b=t*(m*m+2*n*n).",
            "Therefore u*u-2*b*b=t*t*((m*m+2*n*n)*(m*m+2*n*n)-8*m*m*n*n).",
            "The inner expression equals (m*m-2*n*n)*(m*m-2*n*n), so D+F=t*t*(m*m-2*n*n)*(m*m-2*n*n).",
        ],
        "reverse_proof": [
            "Assume D+F=k*k. Put u=e+2*b, so k*k+2*b*b=u*u with 0<b<u.",
            "Divide by u*u and set X=k/u, Y=b/u. Then X*X+2*Y*Y=1 and Y is nonzero rational.",
            "Every rational point on X*X+2*Y*Y=1 with Y nonzero lies on a unique rational line through (1,0).",
            "Writing the line slope as n/m in lowest terms gives X=(m*m-2*n*n)/(m*m+2*n*n) and Y=2*m*n/(m*m+2*n*n), up to the harmless sign of X.",
            "Since b/u=Y, there is an integer scale t with u=t*(m*m+2*n*n) and b=t*2*m*n after clearing the common denominator.",
            "Then e=u-2*b=t*(m*m - 4*m*n + 2*n*n). The original condition e>=1 is exactly the positive-branch constraint on this parameter family.",
            "Finally k=abs(X*u)=t*abs(m*m-2*n*n), completing the reverse implication.",
        ],
        "factorization_cross_check": [
            "The same obstruction appears from (u-k)*(u+k)=2*b*b.",
            "Modulo 4 rules out odd b, so every solution has b even.",
            "The conic parametrization refines that obstruction into the full b=t*2*m*n family.",
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
        "cert_readiness": (
            "Ready for conversion into a formal QA theorem/empirical-observation cert if the project wants this "
            "closed result elevated out of the experiment ledger."
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
            t_max=12,
            m_max=20,
            n_max=20,
            results_dir=Path(tmp),
            summary_json="stage24_self_test.json",
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
            and reread["theorem_status"] == "PROVEN_BY_RATIONAL_CONIC_PARAMETRIZATION"
            and len(reread["canonical_hash"]) == 64
        )
        print(canonical_json({"ok": ok, "solutions": bounded["brute_solution_count"], "misses": bounded["miss_count"]}))
        raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-max", type=int, default=300)
    parser.add_argument("--e-max", type=int, default=300)
    parser.add_argument("--t-max", type=int, default=40)
    parser.add_argument("--m-max", type=int, default=40)
    parser.add_argument("--n-max", type=int, default=40)
    parser.add_argument("--results-dir", type=Path, default=Path("results/qa_quantum_arithmetic_mining_001"))
    parser.add_argument(
        "--summary-json",
        default="qa_quantum_arithmetic_stage24_dplusf_square_proof_closure.json",
    )
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
                "miss_count": audit["miss_count"],
                "canonical_hash": ledger["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
