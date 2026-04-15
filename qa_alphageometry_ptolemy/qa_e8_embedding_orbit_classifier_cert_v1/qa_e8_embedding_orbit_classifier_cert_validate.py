#!/usr/bin/env python3
"""
qa_e8_embedding_orbit_classifier_cert_validate.py  [family 249]

Primary source: (Wildberger, 2020) The Mutation Game, Algebra Colloquium 27.
Companion file: ~/Downloads/MutationGameCoxeterGraphs.pdf
Builds on cert [244] qa_mutation_game_root_lattice_cert (E_8 mutation primitive).

QA_COMPLIANCE: {
    'signal_injection': 'none (static algebraic construction)',
    'dynamics': 'integer T-iteration on (b,e) at m=9; integer quadratic form on Z^8',
    'float_state': false,
    'observer_projection': 'none — orbit classification is QA-discrete-layer',
    'time': 'integer T-step path length (T1 clean)'
}
"""

QA_COMPLIANCE = "cert_validator - E_8 embedding orbit classifier; integer Q on Z^8; T-step on (b,e); no float state; no pow operator"

import json
import sys
from pathlib import Path

import sympy as sp

SCHEMA_VERSION = "QA_E8_EMBEDDING_ORBIT_CLASSIFIER_CERT.v1"
FAMILY_NAME = "qa_e8_embedding_orbit_classifier_cert"

E8_EDGES = ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3))
N = 8
M = 9


def cartan_e8():
    A = [[0] * N for _ in range(N)]
    for u, v in E8_EDGES:
        A[u][v] = 1
        A[v][u] = 1
    return [[2 if i == j else -A[i][j] for j in range(N)] for i in range(N)]


def Q(v, G):
    total = 0
    for i in range(N):
        vi = v[i]
        if vi == 0:
            continue
        for j in range(N):
            total += vi * G[i][j] * v[j]
    return total


def t_step(be):
    b, e = be
    return (e, ((b + e - 1) % M) + 1)


def t_orbits():
    seen = {}
    orbits = []
    for b in range(1, M + 1):
        for e in range(1, M + 1):
            if (b, e) in seen:
                continue
            pt = (b, e)
            orb = []
            while pt not in seen:
                seen[pt] = len(orbits)
                orb.append(pt)
                pt = t_step(pt)
            orbits.append(tuple(orb))
    return orbits


def E_diag(b, e):
    d = ((b + e - 1) % M) + 1
    a = ((b + 2 * e - 1) % M) + 1
    return (b, e, d, a, 0, 0, 0, 0)


def E_tri(b, e):
    d = ((b + e - 1) % M) + 1
    a = ((b + 2 * e - 1) % M) + 1
    return (b, e, b, e, d, d, a, a)


def Q_diag_closed_form(b, e):
    d = ((b + e - 1) % M) + 1
    a = ((b + 2 * e - 1) % M) + 1
    return 2 * (b * b + e * e + d * d + a * a) - 2 * (b * d + e * a + d * a)


def _profile(orbits, embedding, G):
    rows = []
    for orb in orbits:
        Qs = [Q(embedding(b, e), G) for (b, e) in orb]
        rows.append(
            {
                "first": list(orb[0]),
                "size": len(orb),
                "min_Q": min(Qs),
                "max_Q": max(Qs),
                "n_distinct_Q": len(set(Qs)),
                "Q_multiset_sorted": sorted(Qs),
            }
        )
    return rows


def _expected_data():
    G = cartan_e8()
    orbits = t_orbits()
    diag_rows = _profile(orbits, E_diag, G)
    tri_rows = _profile(orbits, E_tri, G)
    return {
        "cartan": G,
        "orbits": orbits,
        "diag_rows": diag_rows,
        "tri_rows": tri_rows,
    }


def _check_diag_formula():
    # Symbolic identity Q(E_diag(b,e)) = 2(b^2+e^2+d^2+a^2) - 2(bd+ea+da)
    # under raw d=b+e, a=b+2e (no mod). We verify the polynomial identity
    # symbolically via SymPy expansion, then exhaustively numerically with mod.
    G = cartan_e8()
    b, e = sp.symbols("b e", integer=True)
    d = b + e
    a = b + 2 * e
    p = sp.Matrix([[b], [e], [d], [a], [0], [0], [0], [0]])
    Q_sym = (p.T * sp.Matrix(G) * p)[0, 0]
    expected = 2 * (b * b + e * e + d * d + a * a) - 2 * (b * d + e * a + d * a)
    if sp.expand(Q_sym - expected) != 0:
        return False
    # Exhaustive numeric check at m=9 (uses A1 mod for d,a)
    for bb in range(1, M + 1):
        for ee in range(1, M + 1):
            if Q(E_diag(bb, ee), G) != Q_diag_closed_form(bb, ee):
                return False
    return True


def _run_checks(fixture):
    checks = {}
    expected = _expected_data()
    G = expected["cartan"]
    orbits = expected["orbits"]
    diag_rows = expected["diag_rows"]
    tri_rows = expected["tri_rows"]

    checks["E8E_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    # Reference E_8 Cartan from [244] — must match canonical
    fixture_cartan = fixture.get("cartan")
    checks["E8E_CARTAN_LOAD"] = (
        fixture_cartan == G
        and fixture.get("adjacency_edges") == [list(e) for e in E8_EDGES]
        and sp.Matrix(G).det() == 1
    )

    # T-orbit census: 5 orbits, sizes {1,8,24,24,24}, sum 81
    sizes = sorted(len(o) for o in orbits)
    checks["E8E_T_ORBITS"] = (
        len(orbits) == 5
        and sizes == [1, 8, 24, 24, 24]
        and sum(sizes) == 81
        and fixture.get("orbit_count") == 5
        and sorted(fixture.get("orbit_sizes", [])) == [1, 8, 24, 24, 24]
    )

    checks["E8E_DIAG_FORMULA"] = _check_diag_formula()

    # min Q under E_diag, ordered by orbit-first-point lex sort, must be the 5 distinct values
    diag_min_set = sorted(row["min_Q"] for row in diag_rows)
    checks["E8E_DIAG_MIN_Q"] = (
        diag_min_set == [8, 16, 28, 72, 162]
        and len(set(diag_min_set)) == 5
        and fixture.get("diag_min_Q_sorted") == diag_min_set
    )

    # Per-orbit Q-multiset is a complete classifier under E_diag
    diag_multisets = [tuple(row["Q_multiset_sorted"]) for row in diag_rows]
    tri_multisets = [tuple(row["Q_multiset_sorted"]) for row in tri_rows]
    checks["E8E_DIAG_MULTISET"] = len(set(diag_multisets)) == 5
    checks["E8E_TRI_PROFILE"] = (
        len(set(tri_multisets)) == 5
        and len(tri_rows) == 5
        and fixture.get("tri_recorded") is True
    )

    src = fixture.get("source_attribution", "")
    checks["E8E_SRC"] = (
        "Wildberger" in src
        and "Mutation Game" in src
        and "2020" in src
        and "[244]" in src
    )

    witnesses = fixture.get("witnesses", [])
    witness_kinds = {w.get("kind") for w in witnesses if isinstance(w, dict)}
    diag_rows_fix = fixture.get("diag_rows")
    diag_rows_serializable = [
        {
            "first": row["first"],
            "size": row["size"],
            "min_Q": row["min_Q"],
            "max_Q": row["max_Q"],
            "n_distinct_Q": row["n_distinct_Q"],
        }
        for row in diag_rows
    ]
    checks["E8E_WITNESS"] = (
        witness_kinds.issuperset(
            {"E8E_CARTAN_LOAD", "E8E_T_ORBITS", "E8E_DIAG_FORMULA", "E8E_DIAG_MIN_Q", "E8E_DIAG_MULTISET"}
        )
        and diag_rows_fix == diag_rows_serializable
        and fixture.get("family") == FAMILY_NAME
        and fixture.get("canonical_embedding") == "E_diag"
    )

    checks["E8E_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_e8_embedding_orbit_classifier_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
