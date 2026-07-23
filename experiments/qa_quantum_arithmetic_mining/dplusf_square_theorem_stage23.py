#!/usr/bin/env python3
"""Stage 23 proof ledger for the D_plus_F square parametrization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_DPLUSF_SQUARE_THEOREM_STAGE23.v1"


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


def solution_key(b: int, e: int) -> tuple[int, int]:
    return (b, e)


def parametrized_solution(t: int, m: int, n: int) -> tuple[int, int, int] | None:
    b = t * 2 * m * n
    e = t * (m * m - 4 * m * n + 2 * n * n)
    if b < 1 or e < 1:
        return None
    k = t * abs(m * m - 2 * n * n)
    return b, e, k


def brute_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            value = d_plus_f(b, e)
            root = math.isqrt(value)
            if root * root == value:
                out[solution_key(b, e)] = root
    return out


def parametrized_solutions(b_max: int, e_max: int, m_max: int, n_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            denom = 2 * m * n
            if denom > b_max:
                continue
            max_t = b_max // denom
            for t in range(1, max_t + 1):
                triple = parametrized_solution(t, m, n)
                if triple is None:
                    continue
                b, e, k = triple
                if b > b_max or e > e_max:
                    continue
                key = solution_key(b, e)
                prior = out.get(key)
                if prior is None or k < prior:
                    out[key] = k
    return out


def audit_param_identity(t_max: int, m_max: int, n_max: int) -> dict[str, int]:
    valid = 0
    invalid_identity = 0
    for t in range(1, t_max + 1):
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                triple = parametrized_solution(t, m, n)
                if triple is None:
                    continue
                b, e, k = triple
                valid += 1
                if d_plus_f(b, e) != k * k:
                    invalid_identity += 1
    return {"valid_param_rows": valid, "invalid_identity_rows": invalid_identity}


def bounded_comparison(b_max: int, e_max: int) -> dict[str, object]:
    brute = brute_solutions(b_max, e_max)
    param = parametrized_solutions(b_max, e_max, b_max, b_max)
    misses = sorted(key for key in brute if key not in param)
    invalid_extras = []
    for key, k in sorted(param.items()):
        b, e = key
        if b <= b_max and e <= e_max and d_plus_f(b, e) != k * k:
            invalid_extras.append({"b": b, "e": e, "k": k})
    return {
        "b_max": b_max,
        "e_max": e_max,
        "brute_solution_count": len(brute),
        "param_hit_count": len(brute) - len(misses),
        "miss_count": len(misses),
        "first_misses": [{"b": b, "e": e} for b, e in misses[:20]],
        "invalid_extra_count": len(invalid_extras),
        "first_invalid_extras": invalid_extras[:20],
    }


def build_proof_ledger(args: argparse.Namespace) -> dict[str, object]:
    identity_audit = audit_param_identity(args.t_max, args.m_max, args.n_max)
    bounded = bounded_comparison(args.b_max, args.e_max)
    ledger: dict[str, object] = {
        "stage_id": "qa_quantum_arithmetic_stage23_dplusf_square_theorem",
        "theorem_status": "PROOF_LEDGER_PLUS_BOUNDED_AUDIT",
        "theorem_statement": (
            "For b,e >= 1, with D=(b+e)*(b+e), F=b*(b+2*e), u=e+2*b, "
            "D+F is a square iff there exist integers t,m,n >= 1 such that "
            "b=t*2*m*n and e=t*(m*m - 4*m*n + 2*n*n)>0. The square root is "
            "k=t*abs(m*m - 2*n*n)."
        ),
        "qa_identity": "D+F = 2*b*b + 4*b*e + e*e = (e+2*b)*(e+2*b) - 2*b*b",
        "forward_param_identity": (
            "Substitute b=t*2*m*n and e=t*(m*m - 4*m*n + 2*n*n). Then "
            "u=e+2*b=t*(m*m+2*n*n), so D+F=u*u-2*b*b="
            "t*t*(m*m-2*n*n)*(m*m-2*n*n)."
        ),
        "reverse_proof_skeleton": [
            "Let k*k=D+F and u=e+2*b. Then u*u-k*k=2*b*b.",
            "Modulo 4 gives b even, so write b=2*c.",
            "Then (u-k)*(u+k)=8*c*c. Since u and k have the same parity, set u-k=2*p and u+k=2*q.",
            "The equation becomes p*q=2*c*c with gcd(p,q) controlled by the common divisor of u-k and u+k.",
            "After extracting the common scale t and applying the standard coprime-square factor casework used for Euclid-style parametrizations, the primitive coprime factors are m*m and 2*n*n in either order.",
            "The positive-e branch gives u=t*(m*m+2*n*n), b=t*2*m*n, and e=u-2*b=t*(m*m - 4*m*n + 2*n*n)>0.",
        ],
        "audit_note": (
            "The reverse proof skeleton records the intended universal casework, while the executable checks verify "
            "the forward identity and bounded zero-miss consistency. This ledger should be reviewed as a theorem "
            "draft before being promoted to a formal cert."
        ),
        "identity_audit": identity_audit,
        "bounded_comparison": bounded,
        "parameters": {
            "b_max": args.b_max,
            "e_max": args.e_max,
            "t_max": args.t_max,
            "m_max": args.m_max,
            "n_max": args.n_max,
        },
    }
    payload = canonical_json({k: v for k, v in ledger.items() if k != "canonical_hash"})
    ledger["canonical_hash"] = domain_sha256(DOMAIN, payload)
    return ledger


def write_outputs(ledger: dict[str, object], results_dir: Path, summary_json: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / summary_json
    out.write_text(canonical_json(ledger) + "\n")
    return out


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            b_max=40,
            e_max=40,
            t_max=80,
            m_max=80,
            n_max=80,
            results_dir=Path(tmp),
            summary_json="self_test.json",
        )
        ledger = build_proof_ledger(args)
        out = write_outputs(ledger, args.results_dir, args.summary_json)
        reread = json.loads(out.read_text())
        bounded = reread["bounded_comparison"]
        identity = reread["identity_audit"]
        ok = (
            bounded["brute_solution_count"] > 0
            and bounded["miss_count"] == 0
            and bounded["invalid_extra_count"] == 0
            and identity["valid_param_rows"] > 0
            and identity["invalid_identity_rows"] == 0
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
        default="qa_quantum_arithmetic_stage23_dplusf_square_theorem.json",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    ledger = build_proof_ledger(args)
    out = write_outputs(ledger, args.results_dir, args.summary_json)
    bounded = ledger["bounded_comparison"]
    print(
        canonical_json(
            {
                "ok": True,
                "summary_json": str(out),
                "brute_solution_count": bounded["brute_solution_count"],
                "miss_count": bounded["miss_count"],
                "canonical_hash": ledger["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
