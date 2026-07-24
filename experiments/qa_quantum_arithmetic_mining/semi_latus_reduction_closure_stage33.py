#!/usr/bin/env python3
"""Stage 33 proof closure for semi_latus_squarefree and semi_latus_distinct_omega_2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_SEMI_LATUS_REDUCTION_CLOSURE_STAGE33.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


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


def prime_set(n: int) -> set[int]:
    return set(factor(n).keys())


def is_squarefree(n: int) -> bool:
    return all(exponent == 1 for exponent in factor(n).values())


def distinct_factor_count(n: int) -> int:
    return len(prime_set(n))


def qa_ab(b: int, e: int) -> tuple[int, int, int]:
    a = b + 2 * e
    return a, b, a * b


# --- semi_latus_squarefree ---------------------------------------------------

def semi_latus_squarefree(b: int, e: int) -> bool:
    _a, _b, F = qa_ab(b, e)
    return is_squarefree(F)


def reduced_squarefree(b: int, e: int) -> bool:
    a, b_, _F = qa_ab(b, e)
    return is_squarefree(a) and is_squarefree(b_) and math.gcd(a, b_) == 1


# --- semi_latus_distinct_omega_2 --------------------------------------------

def semi_latus_distinct_omega_2(b: int, e: int) -> bool:
    _a, _b, F = qa_ab(b, e)
    return distinct_factor_count(F) == 2


def reduced_distinct_omega_2(b: int, e: int) -> bool:
    a, b_, _F = qa_ab(b, e)
    return len(prime_set(a) | prime_set(b_)) == 2


def rad_union_matches(b: int, e: int) -> bool:
    a, b_, F = qa_ab(b, e)
    return prime_set(F) == (prime_set(a) | prime_set(b_))


def audit_window(b_max: int, e_max: int) -> dict[str, object]:
    sf_support = 0
    sf_mismatches: list[dict[str, int]] = []
    omega2_support = 0
    omega2_mismatches: list[dict[str, int]] = []
    rad_mismatches: list[dict[str, int]] = []
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            direct_sf = semi_latus_squarefree(b, e)
            reduced_sf = reduced_squarefree(b, e)
            direct_o2 = semi_latus_distinct_omega_2(b, e)
            reduced_o2 = reduced_distinct_omega_2(b, e)
            if direct_sf:
                sf_support += 1
            if direct_sf != reduced_sf:
                sf_mismatches.append({"b": b, "e": e})
            if direct_o2:
                omega2_support += 1
            if direct_o2 != reduced_o2:
                omega2_mismatches.append({"b": b, "e": e})
            if not rad_union_matches(b, e):
                rad_mismatches.append({"b": b, "e": e})
    pair_count = b_max * e_max
    return {
        "b_max": b_max,
        "e_max": e_max,
        "pair_count": pair_count,
        "rad_union_mismatch_count": len(rad_mismatches),
        "first_rad_union_mismatches": rad_mismatches[:20],
        "squarefree_support": sf_support,
        "squarefree_base_rate": sf_support / pair_count,
        "squarefree_reduction_mismatch_count": len(sf_mismatches),
        "first_squarefree_reduction_mismatches": sf_mismatches[:20],
        "omega2_support": omega2_support,
        "omega2_base_rate": omega2_support / pair_count,
        "omega2_reduction_mismatch_count": len(omega2_mismatches),
        "first_omega2_reduction_mismatches": omega2_mismatches[:20],
    }


def sample_rows(b_max: int, e_max: int, limit: int) -> list[dict[str, object]]:
    rows = []
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            a, b_, F = qa_ab(b, e)
            if semi_latus_squarefree(b, e) or semi_latus_distinct_omega_2(b, e):
                rows.append(
                    {
                        "b": b,
                        "e": e,
                        "a": a,
                        "F": F,
                        "squarefree_F": semi_latus_squarefree(b, e),
                        "distinct_omega_2_F": semi_latus_distinct_omega_2(b, e),
                        "gcd_a_b": math.gcd(a, b_),
                        "primes_a": sorted(prime_set(a)),
                        "primes_b": sorted(prime_set(b_)),
                    }
                )
                if len(rows) >= limit:
                    return rows
    return rows


def build_ledger(args: argparse.Namespace) -> dict[str, object]:
    audit = audit_window(args.b_max, args.e_max)
    rows = sample_rows(args.b_max, args.e_max, args.sample_limit)
    ledger: dict[str, object] = {
        "stage_id": "qa_quantum_arithmetic_stage33_semi_latus_reduction_closure",
        "theorem_status": "PROVEN_STRUCTURAL_FACTORIZATION_REDUCTION",
        "theorem_statement": (
            "For integers b,e >= 1 with a=b+2*e and F=a*b (the semi-latus-rectum quantity): "
            "(1) F is squarefree iff a and b are each squarefree AND gcd(a,b)=1; "
            "(2) F has exactly 2 distinct prime factors iff the UNION of a's and b's distinct "
            "prime factors has exactly 2 elements. Both reduce the target to elementary "
            "factorization facts about the two QA generators a,b, computed directly, with no "
            "dependence on residues or orbit structure."
        ),
        "proof": [
            "rad(F) = rad(a*b) = rad(a) union rad(b) always, for any positive integers a,b "
            "(the set of prime divisors of a product is the union of the prime-divisor sets "
            "of its factors) -- this holds regardless of gcd(a,b).",
            "(2) follows immediately: distinct_factor_count(F) = |rad(F)| = |rad(a) union rad(b)|, "
            "so F has exactly 2 distinct prime factors iff that union has exactly 2 elements.",
            "(1) needs the coprime case: if gcd(a,b)=1, rad(a) and rad(b) are disjoint, so F is "
            "squarefree iff every prime in rad(a) union rad(b) appears to exponent 1 in F, which "
            "(by disjointness) happens iff it appears to exponent 1 in whichever of a,b it divides "
            "-- i.e. iff a and b are each squarefree.",
            "If gcd(a,b)>1, some prime p divides both a and b, so p appears in F=a*b with exponent "
            "v_p(a)+v_p(b) >= 2, so F is not squarefree. Hence squarefree(F) requires gcd(a,b)=1 "
            "as well as squarefree(a) and squarefree(b); all three together are also sufficient by "
            "the disjointness argument above.",
        ],
        "relation_to_stage27": (
            "Stage 27 classified both targets REDUCIBLE_TO_COMPONENT_FACTORIZATION with 0 "
            "reduction mismatches on b,e<=400 (160000 pairs). This stage re-derives and re-verifies "
            "both reductions from the single rad(F)=rad(a) union rad(b) identity, and adds the "
            "rad-union check as its own audited claim rather than leaving it implicit."
        ),
        "audit": audit,
        "sample_positive_rows": rows,
        "parameters": {
            "b_max": args.b_max,
            "e_max": args.e_max,
            "sample_limit": args.sample_limit,
        },
        "cert_readiness": (
            "Both reductions are elementary consequences of rad(xy)=rad(x) union rad(y) and do not "
            "individually warrant separate theorem certs the way the conic-parametrization targets "
            "did; if certified, they belong in the [530]/[532]-style structural-reduction class, "
            "not the [529]/[531]-style complete-parametrization class."
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
            summary_json="stage33_self_test.json",
        )
        ledger = build_ledger(args)
        out = write_ledger(ledger, args.results_dir, args.summary_json)
        reread = json.loads(out.read_text())
        audit = reread["audit"]
        ok = (
            audit["squarefree_support"] > 0
            and audit["omega2_support"] > 0
            and audit["rad_union_mismatch_count"] == 0
            and audit["squarefree_reduction_mismatch_count"] == 0
            and audit["omega2_reduction_mismatch_count"] == 0
            and reread["theorem_status"] == "PROVEN_STRUCTURAL_FACTORIZATION_REDUCTION"
            and len(reread["canonical_hash"]) == 64
        )
        print(
            canonical_json(
                {
                    "ok": ok,
                    "squarefree_support": audit["squarefree_support"],
                    "omega2_support": audit["omega2_support"],
                    "rad_union_mismatches": audit["rad_union_mismatch_count"],
                    "squarefree_mismatches": audit["squarefree_reduction_mismatch_count"],
                    "omega2_mismatches": audit["omega2_reduction_mismatch_count"],
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
        default="qa_quantum_arithmetic_stage33_semi_latus_reduction_closure.json",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    ledger = build_ledger(args)
    out = write_ledger(ledger, args.results_dir, args.summary_json)
    audit = ledger["audit"]
    print(
        canonical_json(
            {
                "ok": True,
                "summary_json": str(out),
                "theorem_status": ledger["theorem_status"],
                "squarefree_support": audit["squarefree_support"],
                "omega2_support": audit["omega2_support"],
                "rad_union_mismatch_count": audit["rad_union_mismatch_count"],
                "squarefree_reduction_mismatch_count": audit["squarefree_reduction_mismatch_count"],
                "omega2_reduction_mismatch_count": audit["omega2_reduction_mismatch_count"],
                "canonical_hash": ledger["canonical_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
