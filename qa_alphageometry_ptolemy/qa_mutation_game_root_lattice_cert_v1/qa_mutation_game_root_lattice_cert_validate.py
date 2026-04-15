#!/usr/bin/env python3
"""
qa_mutation_game_root_lattice_cert_validate.py  [family 244]
QA_COMPLIANCE: {
    'signal_injection': 'none (static algebraic construction)',
    'dynamics': 'integer Weyl mutations on ℤ^8',
    'float_state': false,
    'observer_projection': 'none — root classification is QA-discrete-layer',
    'time': 'integer path-length in Weyl group (T1 clean)'
}
"""

QA_COMPLIANCE = "cert_validator - Mutation Game E8 root lattice; integer Weyl mutations on Z^8; BFS on tuples; no pow operator; no float state"

import json
import sys
from collections import deque
from pathlib import Path

import sympy as sp

SCHEMA_VERSION = "QA_MUTATION_GAME_ROOT_LATTICE_CERT.v1"
FAMILY_NAME = "qa_mutation_game_root_lattice_cert"

# 0-indexed internal encoding, matching the implementation note.
EDGE_LIST = ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3))
N_VERTICES = 8


def adjacency_matrix():
    adj = [[0 for _ in range(N_VERTICES)] for _ in range(N_VERTICES)]
    for u, v in EDGE_LIST:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj


def cartan_matrix():
    adj = adjacency_matrix()
    return [
        [2 if i == j else -adj[i][j] for j in range(N_VERTICES)]
        for i in range(N_VERTICES)
    ]


def mutate(p, i, adj):
    q = list(p)
    q[i] = -p[i] + sum(adj[j][i] * p[j] for j in range(len(p)) if j != i)
    return tuple(q)


def orbit_from(seed, adj):
    start = tuple(seed)
    seen = {start}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for i in range(len(p)):
            nxt = mutate(p, i, adj)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def apply_word(p, word, adj):
    q = tuple(p)
    for i in word:
        q = mutate(q, i, adj)
    return q


def quadratic_form(v, g):
    total = 0
    for i, vi in enumerate(v):
        for j, vj in enumerate(v):
            total += vi * g[i][j] * vj
    return total


def _expected_orbit_data():
    adj = adjacency_matrix()
    seed = tuple(1 if i == 0 else 0 for i in range(N_VERTICES))
    orbit = orbit_from(seed, adj)
    positive = sorted(
        v for v in orbit if all(c >= 0 for c in v) and any(c > 0 for c in v)
    )
    seeds = [tuple(1 if j == i else 0 for j in range(N_VERTICES)) for i in range(N_VERTICES)]
    seed_orbits = [orbit_from(s, adj) for s in seeds]
    return {
        "adj": adj,
        "cartan": cartan_matrix(),
        "orbit": orbit,
        "positive": positive,
        "seed_orbits": seed_orbits,
        "sample": max(positive, key=lambda v: (sum(v), v)),
    }


def _check_braid_relations(sample, adj):
    for i in range(N_VERTICES):
        for j in range(i + 1, N_VERTICES):
            m = 3 if adj[i][j] else 2
            q = sample
            for _ in range(m):
                q = mutate(q, i, adj)
                q = mutate(q, j, adj)
            if q != sample:
                return False
    return True


def _run_checks(fixture):
    checks = {}
    expected = _expected_orbit_data()
    adj = expected["adj"]
    cartan = expected["cartan"]
    orbit = expected["orbit"]
    positive = expected["positive"]
    sample = expected["sample"]
    g = sp.Matrix(cartan)

    checks["MGR_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    fixture_cartan = fixture.get("cartan")
    fixture_edges = [tuple(edge) for edge in fixture.get("adjacency_edges", [])]
    checks["MGR_CARTAN"] = (
        fixture_cartan == cartan
        and fixture_edges == list(EDGE_LIST)
        and g.det() == 1
        and all(entry in (2, 0, -1) for row in cartan for entry in row)
    )

    fixture_orbit_size = fixture.get("orbit_size")
    checks["MGR_BFS_240"] = (
        len(orbit) == 240
        and fixture_orbit_size == 240
        and all(seed_orbit == orbit for seed_orbit in expected["seed_orbits"])
    )

    checks["MGR_ROOT_NORM"] = all(quadratic_form(v, cartan) == 2 for v in orbit)

    negative = {tuple(-c for c in v) for v in positive}
    checks["MGR_SIGN_SPLIT"] = (
        len(positive) == 120
        and len(negative) == 120
        and negative == {v for v in orbit if all(c <= 0 for c in v) and any(c < 0 for c in v)}
    )

    involution_ok = all(mutate(mutate(v, i, adj), i, adj) == v for v in orbit for i in range(N_VERTICES))
    braid_ok = _check_braid_relations(sample, adj)
    checks["MGR_INVOLUTION_BRAID"] = involution_ok and braid_ok

    src = fixture.get("source_attribution", "")
    checks["MGR_SRC"] = (
        "Wildberger" in src
        and "Mutation Game" in src
        and "2020" in src
        and "Algebra Colloquium" in src
    )

    witnesses = fixture.get("witnesses", [])
    witness_kinds = {w.get("kind") for w in witnesses if isinstance(w, dict)}
    checks["MGR_WITNESS"] = (
        witness_kinds.issuperset(
            {
                "MGR_CARTAN",
                "MGR_BFS_240",
                "MGR_ROOT_NORM",
                "MGR_SIGN_SPLIT",
                "MGR_INVOLUTION_BRAID",
            }
        )
        and fixture.get("roots_positive") == [list(v) for v in positive]
        and fixture.get("roots_negative_are_negation") is True
        and fixture.get("family") == FAMILY_NAME
        and fixture.get("case") == "e8_240_roots"
    )

    checks["MGR_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_mutation_game_root_lattice_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
