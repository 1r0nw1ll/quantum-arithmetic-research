#!/usr/bin/env python3
"""Executable promotion-test stubs for the top 5 PHILOMATH QA crosswalk items."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FIXTURES = Path("qa_ingestion_sources/qa_philomath_top5_fixtures.json")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_fixtures(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def rho_9(n: int) -> int:
    return n % 9


def prime_spoke_allowed_24(n: int) -> bool:
    return n > 1 and math.gcd(n, 24) == 1


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def semiprime_prime_factors(n: int) -> list[int] | None:
    factors: list[int] = []
    remaining = n
    d = 2
    while d * d <= remaining and len(factors) <= 2:
        while remaining % d == 0:
            factors.append(d)
            remaining //= d
            if len(factors) > 2:
                return None
        d = 3 if d == 2 else d + 2
    if remaining > 1:
        factors.append(remaining)
    if len(factors) == 2 and all(is_prime(x) for x in factors):
        return factors
    return None


def classify_number(n: int) -> list[str]:
    labels: list[str] = []
    labels.append("even" if n % 2 == 0 else "odd")
    if rho_9(n) == 0:
        labels.append("dr0")
    if prime_spoke_allowed_24(n):
        labels.append("spoke24")
    if is_prime(n):
        labels.append("prime")
    else:
        if n > 1:
            labels.append("composite")
    root = math.isqrt(n)
    if root * root == n:
        labels.append("square")
    if semiprime_prime_factors(n) is not None:
        labels.append("semiprime")
    return sorted(labels)


def difference_of_squares_pair(a: int, b: int) -> tuple[int, int]:
    lo = min(a, b)
    hi = max(a, b)
    midpoint = (lo + hi) // 2
    delta = (hi - lo) // 2
    return midpoint, delta


def encode_semiprime_geometry(n: int, p: int, q: int) -> dict[str, object]:
    midpoint, delta = difference_of_squares_pair(p, q)
    return {
        "n": n,
        "nodes": [min(p, q), max(p, q)],
        "midpoint": midpoint,
        "delta": delta,
    }


def decode_semiprime_geometry(graph: dict[str, object]) -> tuple[int, int]:
    midpoint = int(graph["midpoint"])
    delta = int(graph["delta"])
    return midpoint - delta, midpoint + delta


def validate_semiprime_geometry(graph: dict[str, object]) -> bool:
    try:
        n = int(graph["n"])
        nodes = [int(x) for x in graph["nodes"]]
        a, b = decode_semiprime_geometry(graph)
    except Exception:
        return False
    if sorted(nodes) != sorted([a, b]):
        return False
    if a * b != n:
        return False
    return semiprime_prime_factors(n) == sorted([a, b])


def run_tests(fixtures: dict[str, object]) -> dict[str, object]:
    results: dict[str, object] = {}

    digital_cases = fixtures["digital_root_cases"]
    digital_ok = True
    for case in digital_cases:
        n = int(case["n"])
        expected = int(case["expected_mod9"])
        digital_ok = digital_ok and rho_9(n) == expected

    digital_ops = fixtures["digital_root_operation_cases"]
    digital_ops_ok = True
    for case in digital_ops:
        a = int(case["a"])
        b = int(case["b"])
        sum_expected = int(case["expected_sum_mod9"])
        prod_expected = int(case["expected_product_mod9"])
        digital_ops_ok = digital_ops_ok and rho_9(a + b) == sum_expected
        digital_ops_ok = digital_ops_ok and rho_9(a * b) == prod_expected

    results["digital_root"] = {"ok": digital_ok and digital_ops_ok}

    wheel_ok = True
    wheel_summary: list[dict[str, object]] = []
    for case in fixtures["prime_wheel_cases"]:
        n = int(case["n"])
        allowed = prime_spoke_allowed_24(n)
        expected = bool(case["expected_allowed"])
        is_prime_truth = bool(case["is_prime_ground_truth"])
        if is_prime_truth and n > 3:
            wheel_ok = wheel_ok and allowed
        wheel_ok = wheel_ok and allowed == expected
        wheel_summary.append(
            {"n": n, "allowed": allowed, "expected_allowed": expected, "is_prime_truth": is_prime_truth}
        )
    results["prime_wheel"] = {"ok": wheel_ok, "cases": wheel_summary}

    factor_ok = True
    factor_summary: list[dict[str, object]] = []
    for case in fixtures["factor_recovery_cases"]:
        n = int(case["n"])
        p, q = [int(x) for x in case["factors"]]
        midpoint, delta = difference_of_squares_pair(p, q)
        recovered = (midpoint - delta, midpoint + delta)
        recovered_n = midpoint * midpoint - delta * delta
        ok = recovered_n == n and tuple(sorted(recovered)) == tuple(sorted((p, q)))
        factor_ok = factor_ok and ok
        factor_summary.append(
            {"n": n, "factors": [p, q], "midpoint": midpoint, "delta": delta, "ok": ok}
        )
    results["factor_recovery"] = {"ok": factor_ok, "cases": factor_summary}

    classification_ok = True
    classification_summary: list[dict[str, object]] = []
    for case in fixtures["classification_cases"]:
        n = int(case["n"])
        got = classify_number(n)
        expected = sorted([str(x) for x in case["expected_labels"]])
        ok = got == expected
        classification_ok = classification_ok and ok
        classification_summary.append({"n": n, "got": got, "expected": expected, "ok": ok})
    results["classification"] = {"ok": classification_ok, "cases": classification_summary}

    geometry_ok = True
    geometry_summary: list[dict[str, object]] = []
    for case in fixtures["semiprime_geometry_positive_cases"]:
        n = int(case["n"])
        p, q = [int(x) for x in case["factors"]]
        graph = encode_semiprime_geometry(n, p, q)
        ok = validate_semiprime_geometry(graph)
        geometry_ok = geometry_ok and ok
        geometry_summary.append({"graph": graph, "ok": ok})
    for case in fixtures["semiprime_geometry_negative_cases"]:
        graph = dict(case["graph"])
        expected = bool(case["should_validate"])
        ok = validate_semiprime_geometry(graph) == expected
        geometry_ok = geometry_ok and ok
        geometry_summary.append({"graph": graph, "ok": ok, "expected": expected})
    results["semiprime_geometry"] = {"ok": geometry_ok, "cases": geometry_summary}

    all_ok = all(bool(section["ok"]) for section in results.values())
    return {"ok": all_ok, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PHILOMATH top-5 promotion-test stubs.")
    parser.add_argument("--fixtures", default=str(FIXTURES), help="Path to fixtures JSON")
    parser.add_argument("--self-test", action="store_true", help="Run and emit machine-readable status")
    args = parser.parse_args()

    fixtures = load_fixtures(Path(args.fixtures))
    outcome = run_tests(fixtures)
    print(canonical_json(outcome))
    if args.self_test and outcome["ok"]:
        return 0
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
