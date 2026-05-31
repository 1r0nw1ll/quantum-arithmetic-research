# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Wildberger 2005 ISBN 978-0-9757492-0-8; Cormen 2009 ISBN 978-0-262-03384-8) -->
"""Cert [288]: QA Anchor Geodesic Separation.

PRIMARY CLAIM (edge-level, universally true for any tree):
  For every edge (v,u) in tree T with anchors L, R:
    Δb = d(u,L) - d(v,L),  Δe = d(u,R) - d(v,R)
    Δb * Δe = -1  iff  edge (v,u) is on the unique L-R path P_LR
    Δb * Δe = +1  iff  edge (v,u) is NOT on P_LR

NODE-LEVEL COROLLARY:
  qa_monotone_dir_score(v) = |{u ∈ N(v) : Δb*Δe > 0}|
                            = number of incident edges NOT on P_LR
  Therefore:
    score(v) = 0  iff  ALL incident edges are on P_LR
               iff  v is on P_LR AND v has no off-path branches
    score(v) > 0  iff  v has at least one off-path incident edge
               (includes both off-path nodes AND on-path junctions)

  In the path-with-branch construction this separates branch body (score=2)
  from path interior (score=0), with junctions and leaves at score=1.
  Degree is naturally embedded: no separate degree term is needed.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

Node = str
Adjacency = Dict[Node, List[Node]]


# ---------------------------------------------------------------------------
# Checks
# AGS_1 — Δb, Δe ∈ {-1, +1} for every edge (BFS ±1 in trees)
# AGS_2 — on-path edges: Δb*Δe = -1
# AGS_3 — off-path edges: Δb*Δe = +1
# AGS_4 — score(v) = count of incident off-path edges (definitional corollary)
# AGS_5 — score(v) = 0 iff all incident edges are on P_LR
# SRC    — primary-source exempt marker present
# F      — fixture on_path/off_path node sets match BFS computation
# ---------------------------------------------------------------------------


def _bfs(adjacency: Adjacency, source: Node) -> Dict[Node, int]:
    dist: Dict[Node, int] = {source: 0}
    q: deque[Node] = deque([source])
    while q:
        v = q.popleft()
        for u in adjacency.get(v, []):
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def _lr_path_edges(adjacency: Adjacency, L: Node, R: Node) -> set:
    """Set of frozenset({v,u}) for edges on the unique L-R path."""
    dist_l = _bfs(adjacency, L)
    dist_r = _bfs(adjacency, R)
    total = dist_l[R]
    on_path: set = set()
    for v in adjacency:
        for u in adjacency[v]:
            if dist_l[v] + 1 + dist_r[u] == total or dist_r[v] + 1 + dist_l[u] == total:
                on_path.add(frozenset({v, u}))
    return on_path


def _sign_products(adjacency: Adjacency, L: Node, R: Node) -> List[Tuple[str, str, int, int, int]]:
    dist_l = _bfs(adjacency, L)
    dist_r = _bfs(adjacency, R)
    result = []
    for v in sorted(adjacency):
        for u in sorted(adjacency[v]):
            db = dist_l[u] - dist_l[v]
            de = dist_r[u] - dist_r[v]
            result.append((v, u, db, de, db * de))
    return result


def _monotone_dir_score(adjacency: Adjacency, L: Node, R: Node) -> Dict[Node, int]:
    dist_l = _bfs(adjacency, L)
    dist_r = _bfs(adjacency, R)
    scores: Dict[Node, int] = {}
    for v in adjacency:
        count = 0
        for u in adjacency[v]:
            if (dist_l[u] - dist_l[v]) * (dist_r[u] - dist_r[v]) > 0:
                count += 1
        scores[v] = count
    return scores


def _off_path_edge_count(adjacency: Adjacency, L: Node, R: Node) -> Dict[Node, int]:
    """For each node, count incident edges NOT on P_LR."""
    on_path_edges = _lr_path_edges(adjacency, L, R)
    counts: Dict[Node, int] = {}
    for v in adjacency:
        counts[v] = sum(1 for u in adjacency[v] if frozenset({v, u}) not in on_path_edges)
    return counts


def _on_path_nodes(adjacency: Adjacency, L: Node, R: Node) -> set:
    dist_l = _bfs(adjacency, L)
    dist_r = _bfs(adjacency, R)
    total = dist_l[R]
    return {v for v in adjacency if dist_l[v] + dist_r[v] == total}


def validate_fixture(fixture: dict) -> dict:
    results: dict = {}
    adjacency: Adjacency = {k: list(v) for k, v in fixture["adjacency"].items()}
    L: Node = fixture["anchor_L"]
    R: Node = fixture["anchor_R"]
    expected_on_path: List[Node] = fixture.get("on_path_nodes", [])
    expected_off_path: List[Node] = fixture.get("off_path_nodes", [])

    edges = _sign_products(adjacency, L, R)
    on_path_edges = _lr_path_edges(adjacency, L, R)
    scores = _monotone_dir_score(adjacency, L, R)
    off_path_counts = _off_path_edge_count(adjacency, L, R)
    on_path_nodes = _on_path_nodes(adjacency, L, R)

    # AGS_1: Δb, Δe each in {-1, +1}
    results["AGS_1"] = all(db in (-1, 1) and de in (-1, 1) for _, _, db, de, _ in edges)

    # AGS_2: on-path edges product = -1
    results["AGS_2"] = all(
        prod == -1 for v, u, _, _, prod in edges if frozenset({v, u}) in on_path_edges
    )

    # AGS_3: off-path edges product = +1
    results["AGS_3"] = all(
        prod == 1 for v, u, _, _, prod in edges if frozenset({v, u}) not in on_path_edges
    )

    # AGS_4: score(v) equals the off-path incident edge count
    results["AGS_4"] = all(scores[v] == off_path_counts[v] for v in adjacency)

    # AGS_5: score(v) = 0 iff all incident edges are on P_LR
    results["AGS_5"] = all(
        (scores[v] == 0) == (off_path_counts[v] == 0) for v in adjacency
    )

    # F: on_path/off_path fixture labels match BFS
    if expected_on_path or expected_off_path:
        results["F"] = (
            set(expected_on_path) == on_path_nodes
            and set(expected_off_path) == (set(adjacency) - on_path_nodes)
        )
    else:
        results["F"] = True

    return results


def self_test() -> bool:
    failures = []

    def make_path(n: int) -> Tuple[Adjacency, str, str]:
        adj: Adjacency = {f"P{i}": [] for i in range(n)}
        for i in range(n - 1):
            adj[f"P{i}"].append(f"P{i+1}")
            adj[f"P{i+1}"].append(f"P{i}")
        return adj, "P0", f"P{n-1}"

    def make_star(n: int) -> Tuple[Adjacency, str, str]:
        adj: Adjacency = {"C": [f"L{i}" for i in range(n)]}
        for i in range(n):
            adj[f"L{i}"] = ["C"]
        return adj, "L0", f"L{n-1}"

    def make_binary_tree(depth: int) -> Tuple[Adjacency, str, str]:
        adj: Adjacency = {}
        idx = [0]

        def build(parent: str | None, d: int) -> str:
            name = f"N{idx[0]}"
            idx[0] += 1
            adj[name] = []
            if parent:
                adj[name].append(parent)
                adj[parent].append(name)
            if d > 0:
                build(name, d - 1)
                build(name, d - 1)
            return name

        build(None, depth)
        leaves = [v for v in adj if len(adj[v]) == 1]
        if len(leaves) < 2:
            return adj, list(adj)[0], list(adj)[0]
        return adj, leaves[0], leaves[-1]

    def make_caterpillar(spine: int, legs: int) -> Tuple[Adjacency, str, str]:
        adj: Adjacency = {f"S{i}": [] for i in range(spine)}
        for i in range(spine - 1):
            adj[f"S{i}"].append(f"S{i+1}")
            adj[f"S{i+1}"].append(f"S{i}")
        for i in range(spine):
            for j in range(legs):
                leaf = f"S{i}L{j}"
                adj[leaf] = [f"S{i}"]
                adj[f"S{i}"].append(leaf)
        return adj, "S0", f"S{spine-1}"

    test_cases = (
        [make_path(n) for n in (3, 5, 8, 12, 24)]
        + [make_star(n) for n in (3, 5, 8)]
        + [make_binary_tree(d) for d in (2, 3, 4)]
        + [make_caterpillar(s, l) for s in (4, 6) for l in (1, 2, 3)]
    )

    for adj, L, R in test_cases:
        if L == R:
            continue
        edges = _sign_products(adj, L, R)
        on_path_edges = _lr_path_edges(adj, L, R)
        scores = _monotone_dir_score(adj, L, R)
        off_path_counts = _off_path_edge_count(adj, L, R)

        for v, u, db, de, prod in edges:
            if db not in (-1, 1) or de not in (-1, 1):
                failures.append(f"AGS_1: ({v},{u}) db={db} de={de}")
                continue
            on = frozenset({v, u}) in on_path_edges
            if prod != (-1 if on else 1):
                failures.append(f"AGS_2/3: ({v},{u}) on={on} prod={prod}")

        for v in adj:
            if scores[v] != off_path_counts[v]:
                failures.append(f"AGS_4: {v} score={scores[v]} off_path_edges={off_path_counts[v]}")
            if (scores[v] == 0) != (off_path_counts[v] == 0):
                failures.append(f"AGS_5: {v} score={scores[v]}")

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
        return False
    return True


FAMILY_ID = 288
CERT_SLUG = "qa_anchor_geodesic_separation_cert_v1"


def validate_cert_family(cert_dir) -> Tuple[bool, List[str]]:
    import json
    from pathlib import Path
    errors: List[str] = []
    fixture_dir = Path(cert_dir) / "fixtures"
    if not fixture_dir.is_dir():
        errors.append("missing fixtures/ directory")
        return False, errors
    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        if all_pass == expect_pass:
            pass_count += 1
        else:
            fail_count += 1
            errors.append(f"fixture {path.name}: expected={'PASS' if expect_pass else 'FAIL'} got={'PASS' if all_pass else 'FAIL'} checks={checks}")
    if fail_count:
        errors.append(f"{fail_count} fixture(s) had wrong outcome")
    return fail_count == 0, errors


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="QA Anchor Geodesic Separation Cert validator [288]")
    parser.add_argument("cert_dir", nargs="?", default=str(Path(__file__).parent))
    parser.add_argument("--self-test", action="store_true", dest="selftest")
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir)
    fixture_dir = cert_dir / "fixtures"

    if args.selftest:
        st_ok = self_test()
        fam_ok, fam_errors = validate_cert_family(cert_dir)
        fix_files = list(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
        pass_files = [f for f in fix_files if "pass_" in f.name]
        fail_files = [f for f in fix_files if "fail_" in f.name]
        errors = ([] if st_ok else ["self_test FAIL"]) + fam_errors
        payload = {
            "ok": st_ok and fam_ok,
            "family_id": FAMILY_ID,
            "slug": CERT_SLUG,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": errors,
        }
        print(json.dumps(payload, sort_keys=True))
        sys.exit(0 if payload["ok"] else 1)

    # Interactive mode
    if not self_test():
        print("SELF_TEST FAIL")
        sys.exit(1)
    print("SELF_TEST PASS")

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        ok = all_pass == expect_pass
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        print(f"{'PASS' if ok else 'FAIL'} {path.name}: {checks}")

    print(f"\nFixtures: {pass_count} PASS, {fail_count} FAIL")
    if fail_count:
        sys.exit(1)
