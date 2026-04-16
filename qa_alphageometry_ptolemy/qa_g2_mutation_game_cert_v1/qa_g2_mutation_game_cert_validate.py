#!/usr/bin/env python3
"""
qa_g2_mutation_game_cert_validate.py  [family 251]

Primary source: (Wildberger, 2020) The Mutation Game, Algebra Colloquium 27.
Secondary: (Humphreys, 1972) Introduction to Lie Algebras, Springer GTM 9 §12.1.
Companion file: ~/Downloads/MutationGameCoxeterGraphs.pdf
Theory: docs/theory/QA_G2_MUTATION_GAME.md (commit b86442f).
Builds on cert [244] qa_mutation_game_root_lattice_cert and cert [250] ADE.

QA_COMPLIANCE: {
    'signal_injection': 'none (static algebraic construction)',
    'dynamics': 'integer Weyl mutations on Z^2 for G_2',
    'float_state': false,
    'observer_projection': 'none — root classification is QA-discrete-layer',
    'time': 'integer path-length in Weyl group (T1 clean)'
}
"""

QA_COMPLIANCE = "cert_validator - G2 Mutation Game; integer Weyl mutations on Z^2; BFS on tuples; no pow operator; no float state"

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_G2_MUTATION_GAME_CERT.v1"
FAMILY_NAME = "qa_g2_mutation_game_cert"

EXPECTED_POSITIVE = [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)]
EXPECTED_NEGATIVE = [(-2, -3), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (0, -1)]
EXPECTED_HUMPHREYS = [(0, 1), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2)]
G_SR = [[2, -3], [-3, 6]]


def s0(p):
    return (p[1] - p[0], p[1])


def s1(p):
    return (p[0], 3 * p[0] - p[1])


def bfs_orbit():
    seen = set()
    queue = [(1, 0), (0, 1)]
    while queue:
        p = queue.pop()
        if p in seen:
            continue
        seen.add(p)
        for s in (s0, s1):
            np = s(p)
            if np not in seen:
                queue.append(np)
    return seen


def swap(p):
    return (p[1], p[0])


def norm(v):
    return (
        v[0] * v[0] * G_SR[0][0]
        + 2 * v[0] * v[1] * G_SR[0][1]
        + v[1] * v[1] * G_SR[1][1]
    )


def coxeter_step(p):
    return s0(s1(p))


def iterate_step(p, k):
    q = p
    for _ in range(k):
        q = coxeter_step(q)
    return q


def _as_tuple_list(value):
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(c, int) for c in item)
        ):
            out.append((item[0], item[1]))
    return out


def _expected_data():
    orbit = bfs_orbit()
    positive = sorted(
        p for p in orbit if p[0] >= 0 and p[1] >= 0 and p != (0, 0)
    )
    negative = sorted(
        p for p in orbit if p[0] <= 0 and p[1] <= 0 and p != (0, 0)
    )
    humphreys = sorted(map(swap, positive))
    norms = {h: norm(h) for h in humphreys}
    return {
        "orbit": orbit,
        "orbit_size": len(orbit),
        "positive": positive,
        "negative": negative,
        "negative_is_negation_of_positive": sorted(
            (-p[0], -p[1]) for p in positive
        ) == negative,
        "humphreys": humphreys,
        "norms": norms,
        "norm_values": sorted(norms.values()),
        "involutions": all(s0(s0(p)) == p and s1(s1(p)) == p for p in orbit),
        "coxeter_order_6": all(iterate_step(p, 6) == p for p in orbit),
        "coxeter_strict": all(
            any(iterate_step(p, k) != p for p in orbit) for k in (1, 2, 3, 4, 5)
        ),
    }


def _run_checks(fixture):
    checks = {}
    expected = _expected_data()
    claims = fixture.get("claims", {})

    checks["G2M_1"] = (
        fixture.get("schema_version") == SCHEMA_VERSION
        and expected["orbit_size"] == 12
        and claims.get("orbit_size") == expected["orbit_size"]
        and claims.get("seeds") == [[1, 0], [0, 1]]
        and claims.get("generators") == ["s0", "s1"]
    )

    fixture_positive = sorted(_as_tuple_list(claims.get("positive_populations")))
    fixture_negative = sorted(_as_tuple_list(claims.get("negative_populations")))
    checks["G2M_2"] = (
        expected["positive"] == EXPECTED_POSITIVE
        and expected["negative"] == EXPECTED_NEGATIVE
        and expected["negative_is_negation_of_positive"]
        and len(expected["positive"]) == 6
        and len(expected["negative"]) == 6
        and fixture_positive == EXPECTED_POSITIVE
        and fixture_negative == EXPECTED_NEGATIVE
    )

    fixture_humphreys = sorted(_as_tuple_list(claims.get("humphreys_positive_roots")))
    fixture_norms = sorted(claims.get("humphreys_norms", []))
    checks["G2M_3"] = (
        expected["humphreys"] == EXPECTED_HUMPHREYS
        and expected["norm_values"] == [2, 2, 2, 6, 6, 6]
        and len([v for v in expected["norm_values"] if v == 2]) == 3
        and len([v for v in expected["norm_values"] if v == 6]) == 3
        and fixture_humphreys == EXPECTED_HUMPHREYS
        and fixture_norms == [2, 2, 2, 6, 6, 6]
        and claims.get("humphreys_bijection") == "swap(p0,p1)=(p1,p0)"
        and claims.get("gram") == G_SR
    )

    checks["G2M_4"] = (
        expected["involutions"] is True
        and claims.get("s0_squared_identity") is True
        and claims.get("s1_squared_identity") is True
    )

    checks["G2M_5"] = (
        expected["coxeter_order_6"] is True
        and expected["coxeter_strict"] is True
        and claims.get("coxeter_order") == 6
        and claims.get("coxeter_strict_before_6") is True
    )

    src = fixture.get("source_attribution", "")
    checks["G2M_SRC"] = (
        "Wildberger" in src
        and "2020" in src
        and "Humphreys" in src
        and "1972" in src
        and "§12.1" in src
        and "[244]" in src
        and "[250]" in src
        and "docs/theory/QA_G2_MUTATION_GAME.md" in src
        and "b86442f" in src
    )

    witnesses = fixture.get("witnesses", [])
    witness_kinds = {w.get("kind") for w in witnesses if isinstance(w, dict)}
    checks["G2M_WITNESS"] = (
        fixture.get("family") == FAMILY_NAME
        and witness_kinds.issuperset({"G2M_1", "G2M_2", "G2M_3", "G2M_4", "G2M_5"})
        and claims.get("cartan") == [[2, -1], [-3, 2]]
        and claims.get("directed_edge_counts") == {"0->1": 3, "1->0": 1}
    )

    checks["G2M_F"] = isinstance(fixture.get("fail_ledger"), list)
    return checks


def validate_fixture(path):
    with open(path, encoding="utf-8") as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    actual = "PASS" if all(checks.values()) else "FAIL"
    return {"ok": actual == expected, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    fdir = Path(__file__).parent / "fixtures"
    results = {fp.name: validate_fixture(fp) for fp in sorted(fdir.glob("*.json"))}
    ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": ok, "checks": results}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixture")
    parser.add_argument("fixture_path", nargs="?")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    path = args.fixture or args.fixture_path
    if path:
        result = validate_fixture(path)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0 if result["ok"] else 1
    parser.print_usage(sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
