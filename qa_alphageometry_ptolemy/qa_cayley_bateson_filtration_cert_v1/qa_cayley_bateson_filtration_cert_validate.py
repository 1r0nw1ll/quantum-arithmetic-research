#!/usr/bin/env python3
"""
qa_cayley_bateson_filtration_cert_validate.py

Validator for QA_CAYLEY_BATESON_FILTRATION_CERT.v1  [family 211]

Certifies: The tiered reachability classes of family [191] Bateson Learning
Levels are exactly the connected components of nested undirected Cayley
graphs on S_m = {(b,e) : 1 <= b,e <= m} under stratified generator sets.

Theorem (Bateson-Cayley equivalence on S_9):
    Let Cay_undirected(S_9, Gamma) denote the undirected graph with vertex
    set S_9 and an edge {s, g(s)} for each s in S_9 and each g in Gamma.

    For tiers L1, L2a, L2b define the generator sets:
        Gamma_{L1}  = {T}
        Gamma_{L2a} = Gamma_{L1} U {scalar_k : k in (Z/9Z)*} U {swap}
        Gamma_{L2b} = Gamma_{L2a} U {scalar_k : gcd(k,9)>1, k!=1} U {const_(9,9)}

    Then the connected components of Cay_undirected(S_9, Gamma_{tier}) equal
    the tier-reachability classes of [191]:

        Cay_undirected(S_9, Gamma_{L1})  has components (24, 24, 24, 8, 1)
            = the five T-orbits, total sum_sq = 1793 = 81 + 1712
        Cay_undirected(S_9, Gamma_{L2a}) has components (72, 8, 1)
            = cosmos / satellite / singularity families, sum_sq = 5249
        Cay_undirected(S_9, Gamma_{L2b}) has 1 component (81)
            = the full state space, sum_sq = 6561

    Non-cumulative differences 1712 / 3456 / 1312 match the
    EXPECTED_TIER_COUNTS_S9 constants of [191] exactly.

Undirected convention is essential: scalar_3 and const_(9,9) are non-bijective
on S_9, so reachability is only symmetric under the standard Cayley-graph
convention (generators closed under inverses / edges undirected).

Source: Arthur Cayley, "Desiderata and Suggestions No. 2: The Theory of
Groups: Graphical Representation" (1878) introduces the Cayley graph of a
group with respect to a generating set. Max Dehn's word problem (1911) is
the related decidability question on these graphs. Prerequisite cert:
family [191] qa_bateson_learning_levels_cert.

Checks:
    CBF_1    — schema_version matches
    CBF_GEN  — generator sets declared for L1, L2a, L2b and match expected
    CBF_COMP — component-size multisets match [24,24,24,8,1] / [72,8,1] / [81]
    CBF_CUMU — cumulative sum-of-squares matches 1793 / 5249 / 6561
    CBF_DIFF — non-cumulative diffs match [191] (1712, 3456, 1312)
    CBF_L1   — L1 generators include T
    CBF_L2A  — L2a adds >=1 (Z/9Z)* scalar
    CBF_L2B  — L2b adds >=1 non-unit scalar and a constant
    CBF_191  — cross-reference to family 191 present
    CBF_SRC  — source attribution to Cayley present
    CBF_WIT  — component-merge witnesses declared
    CBF_F    — fail_ledger well-formed

QA axiom compliance: integer state alphabet {1..m}, A1-compliant scalar ops,
no floats, no observer input to QA logic, path time = integer BFS steps.
"""

QA_COMPLIANCE = "cert_validator — validates finite S_9 Cayley-graph connectivity; integer state space; no observer, no floats, no continuous dynamics"

import json
import sys
from collections import deque
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_CAYLEY_BATESON_FILTRATION_CERT.v1"

# [191] constants — theorem reference, not measurements.
EXPECTED_TIER_COUNTS_S9 = {"0": 81, "1": 1712, "2a": 3456, "2b": 1312}
EXPECTED_CUMULATIVE_S9 = {"L1": 1793, "L2a": 5249, "L2b": 6561}
EXPECTED_COMPONENT_SIZES_S9 = {
    "L1":  [24, 24, 24, 8, 1],
    "L2a": [72, 8, 1],
    "L2b": [81],
}


# -----------------------------------------------------------------------------
# QA primitives (integer-only, A1-compliant) — stdlib-only, no imports from qa_lab
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    """A1: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def op_T(m):
    def _T(s):
        b, e = s
        return (e, qa_mod(b + e, m))
    return _T


def op_scalar(k, m):
    def _sc(s):
        b, e = s
        return (qa_mod(k * b, m), qa_mod(k * e, m))
    return _sc


def op_swap():
    return lambda s: (s[1], s[0])


def op_constant(target):
    return lambda _s: target


def enumerate_state_space(m):
    return [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]


def build_generators(tier, m):
    """Return dict of named operators for the given tier on S_m."""
    if tier not in ("L1", "L2a", "L2b"):
        raise ValueError(f"unknown tier: {tier!r}")
    gens = {"T": op_T(m)}
    if tier in ("L2a", "L2b"):
        for k in range(2, m):
            if gcd(k, m) == 1:
                gens[f"scalar_{k}"] = op_scalar(k, m)
        gens["swap"] = op_swap()
    if tier == "L2b":
        for k in range(2, m):
            if gcd(k, m) != 1:
                gens[f"scalar_{k}"] = op_scalar(k, m)
        gens["const_singularity"] = op_constant((m, m))
    return gens


# -----------------------------------------------------------------------------
# Undirected Cayley graph as adjacency dict (stdlib only)
# -----------------------------------------------------------------------------

def build_cayley_adjacency(state_space, generators):
    """Return adjacency dict: state -> set of neighbor states.

    Edges are undirected: for every (s, g(s)) we add both directions.
    Self-loops (g(s) == s) are dropped from the neighbor sets since they
    do not affect connectivity.
    """
    adj = {s: set() for s in state_space}
    for s in state_space:
        for g in generators.values():
            t = g(s)
            if t not in adj:
                continue
            if t == s:
                continue
            adj[s].add(t)
            adj[t].add(s)
    return adj


def connected_components(adj):
    """Return list of components (each a set of states), by BFS."""
    seen = set()
    comps = []
    for start in adj:
        if start in seen:
            continue
        comp = set()
        q = deque([start])
        while q:
            v = q.popleft()
            if v in seen:
                continue
            seen.add(v)
            comp.add(v)
            for nb in adj[v]:
                if nb not in seen:
                    q.append(nb)
        comps.append(comp)
    return comps


def component_sizes(adj):
    return sorted((len(c) for c in connected_components(adj)), reverse=True)


def compute_tier_summary(tier, m=9):
    """For a tier on S_m, return (component_sizes, cumulative_sum_of_squares)."""
    space = enumerate_state_space(m)
    gens = build_generators(tier, m)
    adj = build_cayley_adjacency(space, gens)
    sizes = component_sizes(adj)
    sum_sq = sum(sz * sz for sz in sizes)
    return sizes, sum_sq


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CBF_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"CBF_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # CBF_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("CBF_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("CBF_F: fail_ledger must be a list")

    # If declared FAIL, short-circuit after structural checks
    if cert.get("result") == "FAIL":
        return errors, warnings

    # CBF_SRC: source attribution
    src = cert.get("source_attribution", "")
    if "Cayley" not in str(src):
        warnings.append("CBF_SRC: source_attribution should credit Arthur Cayley (Cayley graph)")

    # CBF_191: cross-reference to family 191
    xrefs = cert.get("cross_references", [])
    has_191 = any(
        isinstance(x, dict) and x.get("family") == 191
        for x in xrefs
    ) if isinstance(xrefs, list) else False
    if not has_191:
        errors.append("CBF_191: missing cross_reference to family 191 qa_bateson_learning_levels_cert (prerequisite)")

    # CBF_GEN: generator sets declared
    gen_decl = cert.get("generator_sets", {})
    for tier in ("L1", "L2a", "L2b"):
        if tier not in gen_decl:
            errors.append(f"CBF_GEN: generator_sets missing tier {tier}")

    # CBF_L1 / L2A / L2B: generator content checks
    l1_gens = gen_decl.get("L1", [])
    l2a_gens = gen_decl.get("L2a", [])
    l2b_gens = gen_decl.get("L2b", [])

    if "T" not in l1_gens:
        errors.append("CBF_L1: L1 generator set must contain 'T'")

    l2a_added = set(l2a_gens) - set(l1_gens)
    has_unit_scalar = any(
        name.startswith("scalar_") and int(name.split("_")[1]) in (2, 4, 5, 7, 8)
        for name in l2a_added
    )
    if not has_unit_scalar:
        errors.append("CBF_L2A: L2a must add at least one (Z/9Z)* unit scalar (scalar_2, _4, _5, _7, or _8)")

    l2b_added = set(l2b_gens) - set(l2a_gens)
    has_nonunit_scalar = any(
        name.startswith("scalar_") and int(name.split("_")[1]) in (3, 6)
        for name in l2b_added
    )
    has_constant = any(name.startswith("const") for name in l2b_added)
    if not has_nonunit_scalar:
        errors.append("CBF_L2B: L2b must add a non-unit scalar (scalar_3 or scalar_6)")
    if not has_constant:
        errors.append("CBF_L2B: L2b must add a constant operator (e.g. const_singularity)")

    # CBF_COMP / CBF_CUMU: independently recompute components on S_9 and check.
    # This is the core theorem check — the cert cannot lie about these.
    actual_summary = {}
    for tier in ("L1", "L2a", "L2b"):
        sizes, sum_sq = compute_tier_summary(tier, m=9)
        actual_summary[tier] = {"sizes": sizes, "sum_sq": sum_sq}

        expected_sizes = EXPECTED_COMPONENT_SIZES_S9[tier]
        if sizes != expected_sizes:
            errors.append(
                f"CBF_COMP: tier {tier} actual component sizes {sizes} != expected {expected_sizes}"
            )

        expected_cumulative = EXPECTED_CUMULATIVE_S9[tier]
        if sum_sq != expected_cumulative:
            errors.append(
                f"CBF_CUMU: tier {tier} cumulative sum-of-squares {sum_sq} != expected {expected_cumulative}"
            )

        decl_sizes = cert.get("component_sizes_s9", {}).get(tier)
        if decl_sizes is not None and list(decl_sizes) != sizes:
            errors.append(
                f"CBF_COMP: declared component_sizes_s9[{tier}] = {decl_sizes} != actual {sizes}"
            )

        decl_cumulative = cert.get("cumulative_pair_count_s9", {}).get(tier)
        if decl_cumulative is not None and int(decl_cumulative) != sum_sq:
            errors.append(
                f"CBF_CUMU: declared cumulative_pair_count_s9[{tier}] = {decl_cumulative} != actual {sum_sq}"
            )

    # CBF_DIFF: non-cumulative diffs match [191]
    l1_cum = actual_summary["L1"]["sum_sq"]
    l2a_cum = actual_summary["L2a"]["sum_sq"]
    l2b_cum = actual_summary["L2b"]["sum_sq"]
    diffs = {
        "1":  l1_cum - 81,
        "2a": l2a_cum - l1_cum,
        "2b": l2b_cum - l2a_cum,
    }
    for t in ("1", "2a", "2b"):
        expected = EXPECTED_TIER_COUNTS_S9[t]
        if diffs[t] != expected:
            errors.append(
                f"CBF_DIFF: non-cumulative tier {t} diff {diffs[t]} != [191] expected {expected}"
            )
        decl_diff = cert.get("tier_differences", {}).get(t)
        if decl_diff is not None and int(decl_diff) != diffs[t]:
            errors.append(
                f"CBF_DIFF: declared tier_differences[{t}] = {decl_diff} != actual {diffs[t]}"
            )

    # CBF_WIT: component-merge witnesses declared
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 3:
        errors.append(
            f"CBF_WIT: need >= 3 component-merge witnesses (one per tier transition), got {len(witnesses) if isinstance(witnesses, list) else 'none'}"
        )

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("cbf_pass_equivalence.json", True),
        ("cbf_fail_missing_field.json", False),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Cayley Bateson Filtration Cert [211] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
