#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=paired_pisano_fixtures"
"""QA Paired Pisano Cert family [179] — certifies that Fibonacci coprime
pairs have higher paired Pisano divisibility than non-Fibonacci pairs.

TIER 3 — STATISTICAL (p=0.0017):
  For coprime (p,q), "both-divide" means both p and q divide pi(m).
  Fibonacci pairs: mean both-divide rate 0.526.
  Non-Fibonacci pairs: mean both-divide rate 0.234.
  Ratio: 2.25x. Mann-Whitney U p=0.0017.

  Mechanism: lcm(p,q)=p*q for coprime pairs. Fibonacci pairs have
  smaller products AND Fibonacci numbers individually divide pi(m)
  more often (3.70x from [163]).

  Honest: product-matched control is only 56% (5/9).
  Exception: 4:1 beats Fibonacci at order-3 (lcm=4 is small).

SOURCE: qa_pisano_paired_divisibility.py — Will Dale conjecture,
Claude computation (2026-04-02). Wall (1960), Renault (1996).

Checks
------
PP_1       schema_version == 'QA_PAIRED_PISANO_CERT.v1'
PP_PAIRS   at least 4 Fibonacci pairs with both_divide_rate reported
PP_STAT    overall Mann-Whitney p < 0.05
PP_RATIO   overall Fibonacci/non-Fibonacci ratio >= 1.5
PP_ORDER   order_analysis present with at least 2 orders, AND each
           entry's order label matches |p-q| for its listed pairs, AND
           its fib_mean/nonfib_mean/ratio are genuinely recomputed from
           real Pisano periods (hardened 2026-07-06)
PP_MECH    mechanism section with lcm_principle
PP_HONEST  caveats section present
PP_W       at least 2 sources
PP_RECOMPUTE  every declared fibonacci_pairs[].both_divide_rate is
              genuinely recomputed from real Pisano periods m=2..500
              (added 2026-07-06 -- see docs/families/179_qa_paired_pisano_cert.md
              Verification Note for what this found: all 9 shipped
              per-pair rates were wrong, and the order_analysis table had
              wrong pair-to-order groupings and wrong per-order stats,
              even reversing the sign of the order-3 comparison)
PP_F       fail detection

Primary source: qa_pisano_paired_divisibility.py (Dale conjecture,
Claude computation, 2026-04-02); Pisano period theory per Wall (1960)
and Renault (1996).
"""

import json
import os
import sys
from collections import defaultdict
from math import gcd

SCHEMA = "QA_PAIRED_PISANO_CERT.v1"
RECOMPUTE_TOL = 0.001


def pisano_period(m):
    """Compute Pisano period pi(m) by Fibonacci sequence mod m."""
    if m <= 1:
        return 1
    prev, curr = 0, 1
    for k in range(1, 6 * m * m + 10):
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return k
    return -1


def _both_divide_rate(p, q, pisano, n_moduli):
    count = sum(1 for pi_m in pisano.values() if pi_m % p == 0 and pi_m % q == 0)
    return count / n_moduli


def _fibonacci_set(max_p):
    fibs = set()
    a, b = 1, 1
    while a <= max_p:
        fibs.add(a)
        a, b = b, a + b
    return fibs


def _all_coprime_pairs(max_p):
    pairs = []
    for p in range(2, max_p + 1):
        for q in range(1, p):
            if gcd(p, q) == 1:
                pairs.append((p, q))
    return pairs


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # PP_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("PP_1", f"schema_version must be {SCHEMA}")

    # PP_PAIRS — Fibonacci pairs reported
    fib_pairs = cert.get("fibonacci_pairs", [])
    if len(fib_pairs) < 4:
        err("PP_PAIRS", f"need >= 4 fibonacci_pairs entries, got {len(fib_pairs)}")

    for fp in fib_pairs:
        rate = fp.get("both_divide_rate")
        if rate is None or not isinstance(rate, (int, float)):
            err("PP_PAIRS", f"pair ({fp.get('p')},{fp.get('q')}) missing valid both_divide_rate")
        p = fp.get("p", 0)
        q = fp.get("q", 0)
        if p <= 0 or q <= 0 or p <= q:
            err("PP_PAIRS", f"invalid pair ({p},{q}): need p > q > 0")

    # PP_RECOMPUTE — genuinely recompute every declared both_divide_rate
    # from real Pisano periods, not just trust the declared number.
    params = cert.get("parameters", {})
    max_m = params.get("max_modulus", 500)
    max_p = params.get("max_p", 10)
    pisano = {m: pisano_period(m) for m in range(2, max_m + 1)}
    n_moduli = len(pisano)
    fibs = _fibonacci_set(max_p)
    all_pairs = _all_coprime_pairs(max_p)
    rate_cache = {(p, q): _both_divide_rate(p, q, pisano, n_moduli) for p, q in all_pairs}

    for fp in fib_pairs:
        p, q = fp.get("p"), fp.get("q")
        declared_rate = fp.get("both_divide_rate")
        if p is None or q is None or declared_rate is None:
            continue
        actual_rate = rate_cache.get((p, q))
        if actual_rate is None:
            continue
        if abs(actual_rate - declared_rate) > RECOMPUTE_TOL:
            err("PP_RECOMPUTE",
                f"pair ({p},{q}): declared both_divide_rate={declared_rate}, "
                f"genuine recompute over m=2..{max_m} gives {actual_rate:.4f}")

    # PP_STAT — statistical significance
    stats = cert.get("statistics", {})
    p_value = stats.get("overall_mann_whitney_p")
    if p_value is not None and p_value >= 0.05:
        err("PP_STAT", f"overall_mann_whitney_p={p_value} >= 0.05 — not significant")
    if p_value is None:
        warnings.append("PP_STAT: no overall_mann_whitney_p declared")

    # PP_RATIO — meaningful advantage
    ratio = stats.get("overall_ratio")
    if ratio is not None and ratio < 1.5:
        err("PP_RATIO", f"overall_ratio={ratio} < 1.5 — no meaningful advantage")
    if ratio is None:
        warnings.append("PP_RATIO: no overall_ratio declared")

    # PP_ORDER — order-stratified analysis: entries present, each pair's
    # order label genuinely matches |p-q|, and fib_mean/nonfib_mean/ratio
    # are genuinely recomputed from real Pisano periods (hardened
    # 2026-07-06 -- previously only checked entry count).
    order_analysis = cert.get("order_analysis", [])
    if len(order_analysis) < 2:
        err("PP_ORDER", f"need >= 2 order_analysis entries, got {len(order_analysis)}")

    nonfib_by_order = defaultdict(list)
    for p, q in all_pairs:
        if not (p in fibs and q in fibs):
            nonfib_by_order[p - q].append((p, q))

    for entry in order_analysis:
        declared_order = entry.get("order")
        fib_pair_strs = entry.get("fib_pairs", [])
        parsed_pairs = []
        for s in fib_pair_strs:
            try:
                p_str, q_str = s.split(":")
                parsed_pairs.append((int(p_str), int(q_str)))
            except (ValueError, AttributeError):
                err("PP_ORDER", f"order {declared_order}: could not parse fib_pair {s!r}")
                continue
        for p, q in parsed_pairs:
            if p - q != declared_order:
                err("PP_ORDER",
                    f"order {declared_order}: pair ({p},{q}) has |p-q|={p - q}, "
                    f"does not match declared order {declared_order}")

        if not parsed_pairs:
            continue
        actual_rates = [rate_cache.get(pq) for pq in parsed_pairs]
        if any(r is None for r in actual_rates):
            continue
        actual_fib_mean = sum(actual_rates) / len(actual_rates)
        declared_fib_mean = entry.get("fib_mean")
        if declared_fib_mean is not None and abs(actual_fib_mean - declared_fib_mean) > 0.01:
            err("PP_ORDER",
                f"order {declared_order}: declared fib_mean={declared_fib_mean}, "
                f"genuine recompute gives {actual_fib_mean:.4f}")

        nonfib_pairs_here = nonfib_by_order.get(declared_order, [])
        if nonfib_pairs_here:
            actual_nonfib_mean = sum(rate_cache[pq] for pq in nonfib_pairs_here) / len(nonfib_pairs_here)
            declared_nonfib_mean = entry.get("nonfib_mean")
            if declared_nonfib_mean is not None and abs(actual_nonfib_mean - declared_nonfib_mean) > 0.01:
                err("PP_ORDER",
                    f"order {declared_order}: declared nonfib_mean={declared_nonfib_mean}, "
                    f"genuine recompute gives {actual_nonfib_mean:.4f}")
            if actual_nonfib_mean > 0:
                actual_ratio = actual_fib_mean / actual_nonfib_mean
                declared_ratio = entry.get("ratio")
                if declared_ratio is not None and abs(actual_ratio - declared_ratio) > 0.05:
                    err("PP_ORDER",
                        f"order {declared_order}: declared ratio={declared_ratio}, "
                        f"genuine recompute gives {actual_ratio:.4f}")

    # PP_MECH — mechanism
    mechanism = cert.get("mechanism", {})
    if not mechanism.get("lcm_principle"):
        err("PP_MECH", "mechanism.lcm_principle required")

    # PP_HONEST — caveats
    caveats = cert.get("caveats", [])
    if not caveats:
        err("PP_HONEST", "caveats section required for honest reporting")

    # PP_W — sources
    sources = cert.get("sources", [])
    if len(sources) < 2:
        err("PP_W", f"need >= 2 sources, got {len(sources)}")

    # PP_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("PP_F", f"declared PASS but {len(errors) - 1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("PP_F: declared FAIL but no fail_ledger and all checks pass")

    return {
        "ok": not has_errors,
        "errors": errors,
        "warnings": warnings,
        "schema": SCHEMA,
    }


def self_test():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    results = {"pass_count": 0, "fail_count": 0, "errors": []}

    for fname in sorted(os.listdir(fixture_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(fixture_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            cert = json.load(f)
        out = validate(cert)
        declared = cert.get("result", "UNKNOWN")
        if declared == "PASS" and out["ok"]:
            results["pass_count"] += 1
        elif declared == "FAIL" and not out["ok"]:
            results["fail_count"] += 1
        else:
            results["errors"].append({
                "fixture": fname,
                "declared": declared,
                "validator_ok": out["ok"],
                "issues": out["errors"],
            })

    results["ok"] = len(results["errors"]) == 0
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("cert_file", nargs="?")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if args.cert_file:
        with open(args.cert_file, "r", encoding="utf-8") as f:
            cert = json.load(f)
        result = validate(cert)
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
