#!/usr/bin/env python3
# noqa: DECL-1 (benchmark verification infrastructure — not empirical QA code)
"""
verify_extension.py — Lane C formal verifier for the candidate claim
`control_extension_oddN_fullSigma`.

Derives EVERY edge and EVERY SCC from the generator definitions in CLAIM.md
alone. Does not read the paper or the blind reproducer's derivation.

The claim under test (see CLAIM.md for full statement):
  Caps(N, N) = {(b, e) : 1 <= b, e <= N}, 1-based.
  sigma(b, e) = (b, e+1)         legal iff e <= N-1
  mu(b, e)    = (e, b)           always legal
  lam2(b, e)  = (2b, 2e)         legal iff b, e <= floor(N/2)
  nu(b, e)    = (b/2, e/2)       legal iff b, e both even
  Sigma = {sigma, mu, lam2, nu}  (the FULL generator set)

Expected outputs:
  N == 1          -> #SCC = 1, max|SCC| = 1
  N even >= 2     -> #SCC = 1, max|SCC| = N*N  (single giant SCC)
  N odd  >= 3     -> #SCC = N+1
                     component sizes (sorted desc) = [(N-1)*(N-1), 2, 2, ..., 2, 1]
                       one inner SCC = Caps(N-1, N-1)
                       (N-1) border 2-cycles = {(N, k), (k, N)} for k in 1..N-1
                       one singleton = {(N, N)}

Usage:
  python3 verify_extension.py
  python3 verify_extension.py --N 30
  python3 verify_extension.py --range "1,2,3,4,5,6,7,8,9,10,15,16,20,30,31,64,65"

Writes per-N fixtures to ./fixtures/<N>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"


# ---------------------------------------------------------------- generators

def sigma(be, N):
    b, e = be
    if e <= N - 1:
        return (b, e + 1)
    return None  # OUT_OF_BOUNDS


def mu(be, N):
    b, e = be
    return (e, b)  # always in-bounds on square Caps


def lam2(be, N):
    b, e = be
    half = N // 2
    if b <= half and e <= half:
        return (2 * b, 2 * e)
    return None  # OUT_OF_BOUNDS


def nu(be, N):
    b, e = be
    if (b % 2 == 0) and (e % 2 == 0):
        return (b // 2, e // 2)
    return None  # PARITY


GENERATORS = {"sigma": sigma, "mu": mu, "lam2": lam2, "nu": nu}


# ---------------------------------------------------------------- graph build

def build_graph(N):
    """Return (nodes, adj, edge_counts, failure_counts)."""
    nodes = [(b, e) for b in range(1, N + 1) for e in range(1, N + 1)]
    adj = {v: set() for v in nodes}
    edge_counts = {name: 0 for name in GENERATORS}
    failure_counts = {"sigma_OOB": 0, "lam2_OOB": 0, "nu_PARITY": 0}

    for v in nodes:
        for name, g in GENERATORS.items():
            w = g(v, N)
            if w is None:
                if name == "sigma":
                    failure_counts["sigma_OOB"] += 1
                elif name == "lam2":
                    failure_counts["lam2_OOB"] += 1
                elif name == "nu":
                    failure_counts["nu_PARITY"] += 1
                continue
            if w not in adj:
                # should never happen for these generators on Caps(N,N),
                # but guard anyway
                continue
            adj[v].add(w)
            edge_counts[name] += 1

    return nodes, adj, edge_counts, failure_counts


# ---------------------------------------------------------------- SCC (Tarjan)

def tarjan_scc(nodes, adj):
    """Iterative Tarjan's SCC. Returns list of sets."""
    idx = {}
    low = {}
    onstack = set()
    stack = []
    result = []
    counter = [0]

    # Iterative DFS; recursion blows the stack for N >= ~128.
    for start in nodes:
        if start in idx:
            continue
        work = [(start, iter(adj[start]))]
        idx[start] = counter[0]
        low[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        onstack.add(start)

        while work:
            v, it = work[-1]
            found_next = False
            for w in it:
                if w not in idx:
                    idx[w] = counter[0]
                    low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    onstack.add(w)
                    work.append((w, iter(adj[w])))
                    found_next = True
                    break
                elif w in onstack:
                    low[v] = min(low[v], idx[w])
            if not found_next:
                if low[v] == idx[v]:
                    comp = set()
                    while True:
                        u = stack.pop()
                        onstack.discard(u)
                        comp.add(u)
                        if u == v:
                            break
                    result.append(comp)
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[v])

    return result


# ---------------------------------------------------------------- claim check

def expected(N):
    """Return (expected_sizes_sorted_desc, structural_description)."""
    if N == 1:
        return [1], {"type": "singleton_only"}
    if N % 2 == 0:
        return [N * N], {"type": "giant", "size": N * N}
    # odd N >= 3
    inner = (N - 1) * (N - 1)
    sizes = [inner] + [2] * (N - 1) + [1]
    structural = {
        "type": "parity_decomposition",
        "inner_size": inner,
        "inner_domain": f"Caps({N - 1}, {N - 1})",
        "border_count": N - 1,
        "border_pattern": "{(N, k), (k, N)} for k in 1..N-1",
        "singleton": (N, N),
    }
    return sizes, structural


def verify(N):
    nodes, adj, edge_counts, failure_counts = build_graph(N)
    sccs = tarjan_scc(nodes, adj)
    sizes = sorted((len(c) for c in sccs), reverse=True)
    exp_sizes, structural = expected(N)

    # sizes check
    sizes_match = sizes == exp_sizes

    # structural check for odd N >= 3
    structural_match = True
    structural_notes = []
    if N >= 3 and N % 2 == 1:
        # find the inner SCC (expected size (N-1)^2)
        inner_expected = {(b, e) for b in range(1, N) for e in range(1, N)}
        singleton_expected = {(N, N)}
        border_expected = [{(N, k), (k, N)} for k in range(1, N)]

        scc_sets = [set(c) for c in sccs]
        inner_found = any(s == inner_expected for s in scc_sets)
        singleton_found = any(s == singleton_expected for s in scc_sets)
        borders_found = all(any(s == b for s in scc_sets) for b in border_expected)

        if not inner_found:
            structural_match = False
            structural_notes.append("inner SCC != Caps(N-1, N-1)")
        if not singleton_found:
            structural_match = False
            structural_notes.append("singleton SCC != {(N, N)}")
        if not borders_found:
            structural_match = False
            structural_notes.append("border 2-cycles != {(N,k),(k,N)}")

    passed = sizes_match and structural_match

    # fixture for reproducibility
    FIXTURES.mkdir(exist_ok=True)
    one_rep_per_comp = sorted(
        (sorted(c)[0] for c in sccs),
        key=lambda t: (-len(next(cc for cc in sccs if t in cc)), t),
    )
    fixture = {
        "N": N,
        "num_nodes": len(nodes),
        "num_sccs": len(sccs),
        "scc_sizes_desc": sizes,
        "expected_sizes_desc": exp_sizes,
        "sizes_match": sizes_match,
        "structural_match": structural_match,
        "structural_notes": structural_notes,
        "edge_counts": edge_counts,
        "failure_counts": failure_counts,
        "expected_structural": structural,
        "representative_nodes_per_component": [list(r) for r in one_rep_per_comp],
        "passed": passed,
    }
    with open(FIXTURES / f"{N}.json", "w") as f:
        json.dump(fixture, f, indent=2, default=str)
        f.write("\n")

    return fixture


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--range",
        default="1,2,3,4,5,6,7,8,9,10,15,16,20,30,31,64,65",
        help="comma-separated list of N values",
    )
    ap.add_argument("--N", type=int, default=None, help="single N")
    args = ap.parse_args()

    if args.N is not None:
        Ns = [args.N]
    else:
        Ns = [int(x) for x in args.range.split(",")]

    results = []
    for N in Ns:
        fx = verify(N)
        status = "PASS" if fx["passed"] else "FAIL"
        print(f"N={N:>3}  SCCs={fx['num_sccs']:>4}  "
              f"sizes={fx['scc_sizes_desc'][:6]}{'...' if len(fx['scc_sizes_desc']) > 6 else ''}  "
              f"{status}")
        if not fx["passed"]:
            print(f"   expected: {fx['expected_sizes_desc'][:6]}")
            print(f"   notes: {fx['structural_notes']}")
        results.append(fx)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed} / {total} N values pass the claim.")

    # verbose dump for selected N values so the proof sketch has numbers
    print("\nEdge counts (spot check):")
    for r in results:
        if r["N"] in {3, 4, 5, 6, 30, 31}:
            print(f"  N={r['N']:>3}  "
                  f"|sigma|={r['edge_counts']['sigma']:>6}  "
                  f"|mu|={r['edge_counts']['mu']:>6}  "
                  f"|lam2|={r['edge_counts']['lam2']:>6}  "
                  f"|nu|={r['edge_counts']['nu']:>6}  "
                  f"sigma_OOB={r['failure_counts']['sigma_OOB']:>4}  "
                  f"lam2_OOB={r['failure_counts']['lam2_OOB']:>4}  "
                  f"nu_PARITY={r['failure_counts']['nu_PARITY']:>4}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
