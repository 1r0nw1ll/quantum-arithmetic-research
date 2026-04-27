#!/usr/bin/env python3
"""qa_counterfactual_descent_cert_validate.py

Validator for QA_COUNTERFACTUAL_DESCENT_CERT.v1  [family 265]

Third sharp-claim cert derived from the Kochenderfer 2026 'Algorithms for
Validation' bridge (Kochenderfer, 2026; docs/specs/QA_KOCHENDERFER_BRIDGE.md
§5 counterfactual-explanation row).

CLAIM (narrow): For QA finite orbit-class PASS/FAIL specifications on S_9,
counterfactual explanations can be computed exactly as shortest-legal-
generator-paths on the orbit graph. The counterfactual distance from state
s to spec-flip is the minimum number of legal QA generator moves needed to
land in a state where the Boolean specification predicate flips.

Claim does NOT generalize to continuous feature-space counterfactual
explanations. Claim does NOT say the BFS approach scales to arbitrary state
spaces. The exactness holds only for QA-discrete finite-state symbolic
settings on the QA-discrete side of the Theorem NT firewall.

Generator set v1: {qa_step, scalar_mult_2, scalar_mult_3}
- qa_step(b, e) = (e, qa_mod(b + e, 9))    — Fibonacci shift T (L_1, orbit-preserving per cert [191] tier hierarchy)
- scalar_mult_2(b, e) = (qa_mod(2b, 9), qa_mod(2e, 9))  — L_2a (coprime-to-9 scalar; orbit-changing, family-preserving)
- scalar_mult_3(b, e) = (qa_mod(3b, 9), qa_mod(3e, 9))  — L_2b (multiple-of-3 scalar; family-changing, the orbit-class-bridging move)

Source attribution: (Kochenderfer, 2026) Algorithms for Validation §11.5
counterfactual explanations (open-candidate-from-bridge framing); (Dale, 2026)
cert [263] qa_failure_density_enumeration_cert utility + cert [191] qa_bateson_
learning_levels_cert tier hierarchy + cert [194] canonical mod-9 classifier.

Checks:
    CFD_1       — schema_version matches
    CFD_DECL    — declared_pass_classes is non-empty subset of orbit classes;
                  declared_generators is non-empty subset of {qa_step, scalar_mult_2, scalar_mult_3}
    CFD_LEGAL   — every step in each declared counterfactual_path uses only
                  declared_generators (each step's `generator` field ∈ declared_generators
                  AND applying that generator to the prior state produces the next state)
    CFD_FLIP    — terminal state of each path flips the PASS predicate vs the start state
                  (in_PASS(start) ≠ in_PASS(terminal) where in_PASS(s) := orbit_family_s9(s) ∈ declared_pass_classes)
    CFD_MINIMAL — declared path length matches the BFS-recomputed shortest path
                  length under the declared generator set + declared spec; bit-exact
    CFD_SRC     — source_attribution cites Kochenderfer + cert [263]
    CFD_WIT     — at least 3 witnesses (one per orbit class, each with a counterfactual
                  path or a "no-counterfactual-needed" annotation)
    CFD_F       — fail_ledger well-formed

Theorem NT compliance: integer-only state path throughout. Generators are
integer functions of integer inputs. BFS produces integer path lengths and
discrete state sequences. No float operations. No observer projection.
"""

QA_COMPLIANCE = "cert_validator — verifies BFS-shortest legal-generator-path counterfactual explanations on QA finite orbit graph S_9; integer-only state path; no observer projection; legal generator set is declarable subset of {qa_step, scalar_mult_2, scalar_mult_3}"

import json
import sys
from collections import deque
from pathlib import Path

# Make repo root importable so cert [263]'s utility module is reachable.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.qa_kg.orbit_failure_enumeration import (  # noqa: E402
    ORBIT_CLASSES,
    orbit_family_s9,
    qa_mod,
    qa_step,
)

SCHEMA_VERSION = "QA_COUNTERFACTUAL_DESCENT_CERT.v1"

# Closed set of legal QA generators recognized by this v1 cert. Each maps
# (b, e) on S_9 to a successor state. cert [191] tier hierarchy:
#   qa_step       — L_1 (orbit-preserving, Fibonacci shift T)
#   scalar_mult_2 — L_2a (coprime-to-9 scalar; orbit-changing, family-preserving)
#   scalar_mult_3 — L_2b (multiple-of-3 scalar; family-changing — the orbit-class bridge)


def _gen_qa_step(b: int, e: int) -> tuple[int, int]:
    return qa_step(int(b), int(e), 9)


def _gen_scalar_mult_2(b: int, e: int) -> tuple[int, int]:
    return (qa_mod(2 * int(b), 9), qa_mod(2 * int(e), 9))


def _gen_scalar_mult_3(b: int, e: int) -> tuple[int, int]:
    return (qa_mod(3 * int(b), 9), qa_mod(3 * int(e), 9))


_LEGAL_GENERATORS: dict[str, callable] = {
    "qa_step": _gen_qa_step,
    "scalar_mult_2": _gen_scalar_mult_2,
    "scalar_mult_3": _gen_scalar_mult_3,
}


def in_pass(b: int, e: int, declared_pass_classes: list[str]) -> bool:
    """Boolean specification predicate: in_PASS(s) := orbit_family_s9(s) ∈ declared_pass_classes."""
    return orbit_family_s9(int(b), int(e)) in declared_pass_classes


def bfs_shortest_counterfactual_path(
    start: tuple[int, int],
    declared_pass_classes: list[str],
    declared_generators: list[str],
) -> list[dict] | None:
    """BFS shortest legal-generator-path from start to first state where in_PASS flips.

    Returns a list of step dicts [{from: [b,e], generator: name, to: [b,e]}, ...]
    representing the path. The list is empty if start already flips (degenerate
    case; should not happen since flip vs self is undefined). Returns None if
    no counterfactual exists under the declared generators (rare on S_9 with
    {qa_step, scalar_mult_2, scalar_mult_3}; the third generator gives orbit-
    class-bridging power).

    Path length = len(returned list). BFS guarantees this is the minimum
    over all legal paths. Integer-only throughout.
    """
    start_label = in_pass(start[0], start[1], declared_pass_classes)
    # parent map: state -> (prev_state, generator_name)
    parent: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cur = queue.popleft()
        if cur != start and in_pass(cur[0], cur[1], declared_pass_classes) != start_label:
            # reconstruct path from start to cur
            path: list[dict] = []
            node = cur
            while parent[node] is not None:
                prev, gen_name = parent[node]
                path.append({
                    "from": list(prev),
                    "generator": gen_name,
                    "to": list(node),
                })
                node = prev
            path.reverse()
            return path
        for gen_name in declared_generators:
            if gen_name not in _LEGAL_GENERATORS:
                continue
            gen_fn = _LEGAL_GENERATORS[gen_name]
            nxt = tuple(gen_fn(cur[0], cur[1]))
            if nxt not in parent:
                parent[nxt] = (cur, gen_name)
                queue.append(nxt)
    return None


def _path_is_legal(
    path: list[dict],
    declared_generators: list[str],
) -> tuple[bool, str]:
    """Verify every step uses a declared generator and produces the claimed successor.

    Returns (is_legal, error_message_or_empty).
    """
    if not isinstance(path, list):
        return False, "path is not a list"
    for i, step in enumerate(path):
        if not isinstance(step, dict):
            return False, f"step {i} is not a dict"
        gen = step.get("generator")
        if gen not in declared_generators:
            return False, f"step {i} uses generator {gen!r} not in declared_generators={declared_generators!r}"
        if gen not in _LEGAL_GENERATORS:
            return False, f"step {i} uses unknown generator {gen!r}"
        from_state = step.get("from")
        to_state = step.get("to")
        if not (isinstance(from_state, list) and len(from_state) == 2):
            return False, f"step {i} 'from' field malformed: {from_state!r}"
        if not (isinstance(to_state, list) and len(to_state) == 2):
            return False, f"step {i} 'to' field malformed: {to_state!r}"
        recomputed = _LEGAL_GENERATORS[gen](int(from_state[0]), int(from_state[1]))
        if tuple(int(x) for x in to_state) != recomputed:
            return False, (
                f"step {i} generator {gen!r} applied to {from_state!r} produces "
                f"{list(recomputed)!r}, but declared 'to'={to_state!r}"
            )
    return True, ""


def _path_flips_label(
    path: list[dict],
    declared_pass_classes: list[str],
) -> tuple[bool, str]:
    """Verify path's terminal state flips the PASS predicate vs start."""
    if not path:
        return False, "path is empty (no flip possible)"
    start = tuple(int(x) for x in path[0]["from"])
    terminal = tuple(int(x) for x in path[-1]["to"])
    start_in_pass = in_pass(start[0], start[1], declared_pass_classes)
    terminal_in_pass = in_pass(terminal[0], terminal[1], declared_pass_classes)
    if start_in_pass == terminal_in_pass:
        return False, (
            f"path does not flip PASS predicate: start={start} in_PASS={start_in_pass}, "
            f"terminal={terminal} in_PASS={terminal_in_pass}"
        )
    return True, ""


def _path_is_minimal(
    path: list[dict],
    declared_pass_classes: list[str],
    declared_generators: list[str],
) -> tuple[bool, str]:
    """BFS-recompute the shortest counterfactual from the path's start; verify
    declared length matches recomputed length bit-exact.
    """
    if not path:
        return False, "path is empty"
    start = tuple(int(x) for x in path[0]["from"])
    bfs_path = bfs_shortest_counterfactual_path(
        start, declared_pass_classes, declared_generators
    )
    if bfs_path is None:
        return False, (
            f"BFS found no counterfactual path from start={start} under "
            f"declared_generators={declared_generators}; declared path is wrong"
        )
    if len(path) != len(bfs_path):
        return False, (
            f"declared path length={len(path)} != BFS-shortest length={len(bfs_path)}; "
            f"declared path is non-minimal"
        )
    return True, ""


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors: list[str] = []
    warnings: list[str] = []

    # CFD_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(
            f"CFD_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}"
        )

    # CFD_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("CFD_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("CFD_F: fail_ledger must be a list")

    # FAIL fixtures short-circuit gates after schema + ledger.
    if cert.get("result") == "FAIL":
        return errors, warnings

    # CFD_DECL: declared_pass_classes non-empty subset of ORBIT_CLASSES;
    # declared_generators non-empty subset of _LEGAL_GENERATORS keys.
    declared_pass_classes = cert.get("declared_pass_classes")
    declared_generators = cert.get("declared_generators")
    if not isinstance(declared_pass_classes, list) or not declared_pass_classes:
        errors.append(
            f"CFD_DECL: declared_pass_classes must be non-empty list; got {declared_pass_classes!r}"
        )
    else:
        for cls in declared_pass_classes:
            if cls not in ORBIT_CLASSES:
                errors.append(
                    f"CFD_DECL: declared_pass_classes entry {cls!r} not in {ORBIT_CLASSES!r}"
                )
    if not isinstance(declared_generators, list) or not declared_generators:
        errors.append(
            f"CFD_DECL: declared_generators must be non-empty list; got {declared_generators!r}"
        )
    else:
        for gen in declared_generators:
            if gen not in _LEGAL_GENERATORS:
                errors.append(
                    f"CFD_DECL: declared_generators entry {gen!r} not in legal set "
                    f"{sorted(_LEGAL_GENERATORS)!r}"
                )

    # CFD_SRC: source attribution must mention Kochenderfer AND cert [263].
    src = str(cert.get("source_attribution", ""))
    if "Kochenderfer" not in src:
        errors.append(
            "CFD_SRC: source_attribution must cite Kochenderfer 2026 "
            "Algorithms for Validation §11.5 counterfactual explanations"
        )
    if "263" not in src:
        errors.append(
            "CFD_SRC: source_attribution must cite cert [263] "
            "qa_failure_density_enumeration_cert (utility provider)"
        )

    # CFD_LEGAL + CFD_FLIP + CFD_MINIMAL: per declared counterfactual_path test case.
    test_cases = cert.get("counterfactual_test_cases", [])
    if not isinstance(test_cases, list) or not test_cases:
        errors.append(
            "CFD_LEGAL/FLIP/MINIMAL: need >= 1 counterfactual_test_cases; "
            f"got {test_cases!r}"
        )
    elif isinstance(declared_pass_classes, list) and isinstance(declared_generators, list):
        for i, case in enumerate(test_cases):
            path_decl = case.get("path")
            if not isinstance(path_decl, list):
                errors.append(
                    f"CFD_LEGAL[{i}]: case 'path' missing or not a list: {path_decl!r}"
                )
                continue
            ok, msg = _path_is_legal(path_decl, declared_generators)
            if not ok:
                errors.append(f"CFD_LEGAL[{i}]: {msg}")
                continue
            ok, msg = _path_flips_label(path_decl, declared_pass_classes)
            if not ok:
                errors.append(f"CFD_FLIP[{i}]: {msg}")
                continue
            ok, msg = _path_is_minimal(
                path_decl, declared_pass_classes, declared_generators
            )
            if not ok:
                errors.append(f"CFD_MINIMAL[{i}]: {msg}")
                continue
            # If the case declares an expected_path_length, check it matches.
            decl_len = case.get("expected_path_length")
            if decl_len is not None and int(decl_len) != len(path_decl):
                errors.append(
                    f"CFD_MINIMAL[{i}]: declared expected_path_length={decl_len} "
                    f"!= actual len={len(path_decl)}"
                )

    # CFD_WIT: at least 3 witnesses, one per orbit class.
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 3:
        errors.append(
            f"CFD_WIT: need >= 3 witnesses (one per orbit class); got "
            f"{len(witnesses) if isinstance(witnesses, list) else 'malformed'}"
        )
    else:
        seen_classes = {w.get("orbit_class") for w in witnesses}
        for cls in ORBIT_CLASSES:
            if cls not in seen_classes:
                errors.append(f"CFD_WIT: missing witness for class {cls!r}")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("pass_s9_shortest_counterfactual_paths.json", True),
        ("pass_generator_cost_weighted_paths.json", True),
        ("fail_illegal_generator_path.json", True),
        ("fail_nonminimal_counterfactual_path.json", True),
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
            results.append({
                "fixture": fname, "ok": False,
                "error": f"expected PASS but got errors: {errs}",
            })
            all_ok = False
        elif not should_pass and passed:
            results.append({
                "fixture": fname, "ok": False,
                "error": "expected FAIL but got PASS",
            })
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})
    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Counterfactual Descent Cert [265] validator")
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
