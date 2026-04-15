#!/usr/bin/env python3
"""
qa_ade_mutation_game_cert_validate.py  [family 250]

Primary source: (Wildberger, 2020) The Mutation Game, Algebra Colloquium 27.
Secondary: (Humphreys, 1972) Introduction to Lie Algebras, Springer GTM 9 §9.3 Table 1.
Companion file: ~/Downloads/MutationGameCoxeterGraphs.pdf
Builds on cert [244] qa_mutation_game_root_lattice_cert (E_8 only).

QA_COMPLIANCE: {
    'signal_injection': 'none (static algebraic construction)',
    'dynamics': 'integer Weyl mutations on Z^n for ADE Dynkin types',
    'float_state': false,
    'observer_projection': 'none — root classification is QA-discrete-layer',
    'time': 'integer path-length in Weyl group (T1 clean)'
}
"""

QA_COMPLIANCE = "cert_validator - ADE Mutation Game; integer Weyl mutations on Z^n; BFS on tuples; no pow operator; no float state"

import json
import sys
from collections import deque
from pathlib import Path

import sympy as sp

SCHEMA_VERSION = "QA_ADE_MUTATION_GAME_CERT.v1"
FAMILY_NAME = "qa_ade_mutation_game_cert"

ADE_TYPES = (
    ("A5", 5, ((0, 1), (1, 2), (2, 3), (3, 4)), 30, 6),
    ("D5", 5, ((0, 1), (1, 2), (2, 3), (2, 4)), 40, 4),
    ("E6", 6, ((0, 2), (2, 3), (3, 4), (4, 5), (1, 3)), 72, 3),
    ("E7", 7, ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 3)), 126, 2),
    ("E8", 8, ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)), 240, 1),
)


def adjacency(n, edges):
    A = [[0] * n for _ in range(n)]
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1
    return A


def cartan(n, edges):
    A = adjacency(n, edges)
    return [[2 if i == j else -A[i][j] for j in range(n)] for i in range(n)]


def mutate(p, i, A):
    n = len(p)
    q = list(p)
    q[i] = -p[i] + sum(A[j][i] * p[j] for j in range(n) if j != i)
    return tuple(q)


def bfs_orbit(n, A):
    seed = tuple(1 if i == 0 else 0 for i in range(n))
    seen = {seed}
    queue = deque([seed])
    while queue:
        p = queue.popleft()
        for i in range(n):
            q = mutate(p, i, A)
            if q not in seen:
                seen.add(q)
                queue.append(q)
    return seen


def quad(v, G):
    n = len(v)
    total = 0
    for i in range(n):
        vi = v[i]
        if vi == 0:
            continue
        for j in range(n):
            total += vi * G[i][j] * v[j]
    return total


def _expected_data():
    out = {}
    for name, n, edges, expected_size, expected_det in ADE_TYPES:
        A = adjacency(n, edges)
        G = cartan(n, edges)
        orbit = bfs_orbit(n, A)
        det = int(sp.Matrix(G).det())
        positive = {v for v in orbit if all(c >= 0 for c in v) and any(c > 0 for c in v)}
        negative = {v for v in orbit if all(c <= 0 for c in v) and any(c < 0 for c in v)}
        out[name] = {
            "n": n,
            "edges": edges,
            "G": G,
            "det": det,
            "orbit_size": len(orbit),
            "expected_size": expected_size,
            "expected_det": expected_det,
            "norms_set": sorted({quad(v, G) for v in orbit}),
            "pos": len(positive),
            "neg": len(negative),
            "neg_is_negation_of_pos": {tuple(-c for c in v) for v in positive} == negative,
        }
    return out


def _run_checks(fixture):
    checks = {}
    expected = _expected_data()

    checks["ADE_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    cartan_dets_ok = all(
        expected[name]["det"] == expected[name]["expected_det"]
        for name, *_ in ADE_TYPES
    )
    fix_types = fixture.get("types", {})
    fixture_dets_ok = all(
        fix_types.get(name, {}).get("det") == expected[name]["det"]
        for name, *_ in ADE_TYPES
    )
    checks["ADE_CARTAN_DETS"] = cartan_dets_ok and fixture_dets_ok

    bfs_ok = all(
        expected[name]["orbit_size"] == expected[name]["expected_size"]
        for name, *_ in ADE_TYPES
    )
    fixture_sizes_ok = all(
        fix_types.get(name, {}).get("orbit_size") == expected[name]["orbit_size"]
        for name, *_ in ADE_TYPES
    )
    checks["ADE_BFS_SIZES"] = bfs_ok and fixture_sizes_ok

    norms_ok = all(expected[name]["norms_set"] == [2] for name, *_ in ADE_TYPES)
    checks["ADE_ROOT_NORM"] = norms_ok

    sign_ok = all(
        expected[name]["pos"] == expected[name]["neg"]
        and expected[name]["pos"] * 2 == expected[name]["orbit_size"]
        and expected[name]["neg_is_negation_of_pos"]
        for name, *_ in ADE_TYPES
    )
    checks["ADE_SIGN_SPLIT"] = sign_ok

    src = fixture.get("source_attribution", "")
    checks["ADE_SRC"] = (
        "Wildberger" in src
        and "Humphreys" in src
        and "2020" in src
        and "[244]" in src
    )

    witnesses = fixture.get("witnesses", [])
    witness_kinds = {w.get("kind") for w in witnesses if isinstance(w, dict)}
    edges_match = all(
        fix_types.get(name, {}).get("edges") == [list(e) for e in expected[name]["edges"]]
        for name, *_ in ADE_TYPES
    )
    checks["ADE_WITNESS"] = (
        witness_kinds.issuperset(
            {"ADE_CARTAN_DETS", "ADE_BFS_SIZES", "ADE_ROOT_NORM", "ADE_SIGN_SPLIT"}
        )
        and edges_match
        and fixture.get("family") == FAMILY_NAME
        and set(fix_types.keys()) == {name for name, *_ in ADE_TYPES}
    )

    checks["ADE_F"] = isinstance(fixture.get("fail_ledger"), list)
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


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if len(sys.argv) > 1:
        result = validate_fixture(sys.argv[1])
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)
    print("Usage: python qa_ade_mutation_game_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
