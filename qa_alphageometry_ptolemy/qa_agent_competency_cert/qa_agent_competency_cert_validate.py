#!/usr/bin/env python3
"""QA Agent Competency Cert family [123] validator — QA_AGENT_COMPETENCY_CERT.v1

Formalizes the Levin morphogenetic agent architecture for QA Lab.
Certifies that a competency profile is structurally valid and machine-checkable.

Schema: QA_AGENT_COMPETENCY_CERT.v1
Fields:
  schema_version          "QA_AGENT_COMPETENCY_CERT.v1"
  agent_name              str — unique agent identifier
  category                str — sort | search | traverse | optimize | learn | composite | control | empirical
  goal                    str — what the agent achieves (≥10 chars)
  cognitive_horizon       str — local | bounded | global | adaptive
  convergence             str — guaranteed | probabilistic | conditional | none
  orbit_signature         str — cosmos | satellite | singularity | mixed
  levin_cell_type         str — stem | progenitor | differentiated
  failure_modes           list[str] — ≥1 structural failure condition
  composition_rules       list[str] — ≥1 composition constraint
  dedifferentiation_cond  list[str] — when to reset to stem state (≥1)
  recommitment_cond       list[str] — when committed state should be revisited (≥1)
  parent_cert             str — parent cert this profile maps to
  result                  "PASS" | "FAIL"

Validator checks:
  V1  schema_version matches exactly                    → SCHEMA_VERSION_MISMATCH
  V2  cognitive_horizon is a known value                → UNKNOWN_COGNITIVE_HORIZON
  V3  convergence is a known value                      → UNKNOWN_CONVERGENCE_TYPE
  V4  orbit_signature is a known value                  → UNKNOWN_ORBIT_SIGNATURE
  V5  levin_cell_type is a known value                  → UNKNOWN_LEVIN_CELL_TYPE
  V6  failure_modes is nonempty list                    → EMPTY_FAILURE_MODES
  V7  dedifferentiation_cond is nonempty list           → EMPTY_DEDIFFERENTIATION_COND
  V8  orbit_signature matches levin_cell_type           → CELL_ORBIT_MISMATCH
       (differentiated ↔ cosmos, progenitor ↔ satellite/mixed, stem ↔ singularity)
  V9  goal is at least 10 characters                    → GOAL_TOO_SHORT
  V10 composition_rules is nonempty list                → EMPTY_COMPOSITION_RULES

Usage:
  python qa_agent_competency_cert_validate.py --self-test
  python qa_agent_competency_cert_validate.py --file fixtures/acc_pass_merge_sort.json
"""

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


import json
import sys
import argparse
from pathlib import Path

SCHEMA_VERSION = "QA_AGENT_COMPETENCY_CERT.v1"

KNOWN_COGNITIVE_HORIZONS = frozenset(["local", "bounded", "global", "adaptive"])
KNOWN_CONVERGENCES       = frozenset(["guaranteed", "probabilistic", "conditional", "none"])
KNOWN_ORBIT_SIGNATURES   = frozenset(["cosmos", "satellite", "singularity", "mixed"])
KNOWN_LEVIN_CELL_TYPES   = frozenset(["stem", "progenitor", "differentiated"])

# Cell type → allowed orbit signatures
CELL_ORBIT_RULES = {
    "differentiated": frozenset(["cosmos"]),
    "progenitor":     frozenset(["satellite", "mixed"]),
    "stem":           frozenset(["singularity"]),
}

KNOWN_FAIL_TYPES = frozenset([
    "SCHEMA_VERSION_MISMATCH",
    "UNKNOWN_COGNITIVE_HORIZON",
    "UNKNOWN_CONVERGENCE_TYPE",
    "UNKNOWN_ORBIT_SIGNATURE",
    "UNKNOWN_LEVIN_CELL_TYPE",
    "EMPTY_FAILURE_MODES",
    "EMPTY_DEDIFFERENTIATION_COND",
    "CELL_ORBIT_MISMATCH",
    "GOAL_TOO_SHORT",
    "EMPTY_COMPOSITION_RULES",
])


class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def validate_agent_competency_cert(cert: dict) -> dict:
    out = _Out()
    detected_fails: set = set()

    # ── required top-level fields ──────────────────────────────────────────
    required = [
        "schema_version", "agent_name", "category", "goal",
        "cognitive_horizon", "convergence", "orbit_signature",
        "levin_cell_type", "failure_modes", "composition_rules",
        "dedifferentiation_cond", "recommitment_cond",
        "parent_cert", "result",
    ]
    for f in required:
        if f not in cert:
            out.fail(f"missing required field: {f!r}")

    if out.errors:
        return _reconcile(cert, out, detected_fails)

    # ── V1: schema_version ────────────────────────────────────────────────
    if cert.get("schema_version") != SCHEMA_VERSION:
        detected_fails.add("SCHEMA_VERSION_MISMATCH")

    # ── V2: cognitive_horizon ─────────────────────────────────────────────
    ch = cert.get("cognitive_horizon", "")
    if ch not in KNOWN_COGNITIVE_HORIZONS:
        detected_fails.add("UNKNOWN_COGNITIVE_HORIZON")

    # ── V3: convergence ───────────────────────────────────────────────────
    conv = cert.get("convergence", "")
    if conv not in KNOWN_CONVERGENCES:
        detected_fails.add("UNKNOWN_CONVERGENCE_TYPE")

    # ── V4: orbit_signature ───────────────────────────────────────────────
    orbit = cert.get("orbit_signature", "")
    if orbit not in KNOWN_ORBIT_SIGNATURES:
        detected_fails.add("UNKNOWN_ORBIT_SIGNATURE")

    # ── V5: levin_cell_type ───────────────────────────────────────────────
    cell_type = cert.get("levin_cell_type", "")
    if cell_type not in KNOWN_LEVIN_CELL_TYPES:
        detected_fails.add("UNKNOWN_LEVIN_CELL_TYPE")

    # ── V6: failure_modes nonempty ────────────────────────────────────────
    fm = cert.get("failure_modes", [])
    if not isinstance(fm, list) or len(fm) == 0:
        detected_fails.add("EMPTY_FAILURE_MODES")

    # ── V7: dedifferentiation_cond nonempty ───────────────────────────────
    dc = cert.get("dedifferentiation_cond", [])
    if not isinstance(dc, list) or len(dc) == 0:
        detected_fails.add("EMPTY_DEDIFFERENTIATION_COND")

    # ── V8: cell type ↔ orbit signature consistency ───────────────────────
    if cell_type in CELL_ORBIT_RULES and orbit in KNOWN_ORBIT_SIGNATURES:
        allowed = CELL_ORBIT_RULES[cell_type]
        if orbit not in allowed:
            detected_fails.add("CELL_ORBIT_MISMATCH")

    # ── V9: goal length ───────────────────────────────────────────────────
    goal = cert.get("goal", "")
    if not isinstance(goal, str) or len(goal) < 10:
        detected_fails.add("GOAL_TOO_SHORT")

    # ── V10: composition_rules nonempty ───────────────────────────────────
    cr = cert.get("composition_rules", [])
    if not isinstance(cr, list) or len(cr) == 0:
        detected_fails.add("EMPTY_COMPOSITION_RULES")

    return _reconcile(cert, out, detected_fails)


def _reconcile(cert: dict, out: _Out, detected_fails: set) -> dict:
    all_fails = list(out.errors) + list(detected_fails)
    declared_result = cert.get("result", "")
    ok = len(all_fails) == 0
    actual_result = "PASS" if ok else "FAIL"

    issues = []
    for f in out.errors:
        issues.append({"fail_type": "SCHEMA_ERROR", "detail": f})
    for f in detected_fails:
        issues.append({"fail_type": f})

    return {
        "ok": ok,
        "result": actual_result,
        "result_matches_declared": actual_result == declared_result,
        "issues": issues,
        "warnings": out.warnings,
    }


# ─── Self-test fixtures ────────────────────────────────────────────────────

_PASS_MERGE_SORT = {
    "schema_version": "QA_AGENT_COMPETENCY_CERT.v1",
    "agent_name": "merge_sort_agent",
    "category": "sort",
    "goal": "Sort N elements via divide-and-conquer merge in O(N log N) — guaranteed convergence.",
    "cognitive_horizon": "global",
    "convergence": "guaranteed",
    "orbit_signature": "cosmos",
    "levin_cell_type": "differentiated",
    "failure_modes": [
        "O(N) auxiliary space — fails on memory-constrained streams",
        "Not in-place — poor cache locality on large arrays",
    ],
    "composition_rules": [
        "Requires spine role in any sorting organ (provides O(N log N) guarantee)",
        "Can run in parallel with parallel_merge_sort variant",
    ],
    "dedifferentiation_cond": [
        "success_rate < 0.4 over 10 tasks",
        "orbit_drift_count > 3 consecutive cycles",
    ],
    "recommitment_cond": [
        "input_size exceeds available memory — switch to external merge sort",
        "streaming input requires different sort strategy",
    ],
    "parent_cert": "QA_CORE_SPEC.v1",
    "result": "PASS",
}

_PASS_GRADIENT_DESCENT = {
    "schema_version": "QA_AGENT_COMPETENCY_CERT.v1",
    "agent_name": "gradient_descent_agent",
    "category": "optimize",
    "goal": "Minimize loss L(theta) by iterating theta <- theta - eta * grad(L) until convergence criterion met.",
    "cognitive_horizon": "local",
    "convergence": "conditional",
    "orbit_signature": "mixed",
    "levin_cell_type": "progenitor",
    "failure_modes": [
        "Local minima: non-convex landscape traps agent in satellite orbit",
        "Vanishing gradient: near-zero gradient -> singularity (no movement)",
        "Learning rate too high: divergence -> escapes orbit boundary",
    ],
    "composition_rules": [
        "Requires orbit_damper partner (Adam/RMSprop) to stay in cosmos on non-convex landscapes",
        "Composes with scheduler as progenitor-to-cosmos promoter",
        "MUST be paired with a cosmos spine agent when used in critical organ",
    ],
    "dedifferentiation_cond": [
        "loss_plateau detected for 5 consecutive cycles (satellite loop)",
        "gradient_norm < 1e-6 for 3 cycles (singularity trap)",
    ],
    "recommitment_cond": [
        "switch to second-order methods when Hessian is available",
        "replace with evolutionary optimizer on purely non-differentiable objective",
    ],
    "parent_cert": "QA_CORE_SPEC.v1",
    "result": "PASS",
}

_FAIL_CELL_ORBIT_MISMATCH = {
    "schema_version": "QA_AGENT_COMPETENCY_CERT.v1",
    "agent_name": "bad_agent",
    "category": "sort",
    "goal": "Sort N elements but declared as stem cell with cosmos orbit — contradiction.",
    "cognitive_horizon": "local",
    "convergence": "guaranteed",
    "orbit_signature": "cosmos",     # cosmos is not allowed for stem cells
    "levin_cell_type": "stem",       # stem must be singularity
    "failure_modes": ["some failure"],
    "composition_rules": ["some rule"],
    "dedifferentiation_cond": ["always"],
    "recommitment_cond": ["never"],
    "parent_cert": "QA_CORE_SPEC.v1",
    "result": "FAIL",
}


def run_self_test() -> bool:
    test_cases = [
        ("acc_pass_merge_sort.json",        _PASS_MERGE_SORT,           True,  None),
        ("acc_pass_gradient_descent.json",  _PASS_GRADIENT_DESCENT,     True,  None),
        ("acc_fail_cell_orbit_mismatch.json", _FAIL_CELL_ORBIT_MISMATCH, False, "CELL_ORBIT_MISMATCH"),
    ]
    results = []
    all_ok = True

    for fixture_name, cert, expect_pass, expect_fail_type in test_cases:
        r = validate_agent_competency_cert(cert)
        fail_types = {i["fail_type"] for i in r.get("issues", [])}

        if expect_pass:
            passed = r["ok"] and r["result"] == "PASS"
        else:
            passed = (not r["ok"]) and (expect_fail_type in fail_types)

        if not passed:
            all_ok = False

        results.append({
            "fixture": fixture_name,
            "ok": passed,
            "label": "PASS" if passed else "FAIL",
            "errors": [] if passed else [f"unexpected: ok={r['ok']} issues={list(fail_types)}"],
            "warnings": r.get("warnings", []),
        })

    output = {"ok": all_ok, "results": results}
    print(json.dumps(output, indent=2))
    return all_ok


# ─── Fixture writing ───────────────────────────────────────────────────────

def write_fixtures():
    """Write fixture files alongside this script."""
    here = Path(__file__).resolve().parent / "fixtures"
    here.mkdir(exist_ok=True)

    fixtures = [
        ("acc_pass_merge_sort.json", _PASS_MERGE_SORT),
        ("acc_pass_gradient_descent.json", _PASS_GRADIENT_DESCENT),
        ("acc_fail_cell_orbit_mismatch.json", _FAIL_CELL_ORBIT_MISMATCH),
    ]
    for name, data in fixtures:
        path = here / name
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  wrote: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=f"{SCHEMA_VERSION} validator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="Run built-in self-test")
    group.add_argument("--file", type=Path, help="Validate a cert file")
    group.add_argument("--write-fixtures", action="store_true", help="Write fixture files")
    args = parser.parse_args(argv)

    if args.self_test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    if args.write_fixtures:
        write_fixtures()
        sys.exit(0)

    if args.file:
        try:
            cert = json.loads(args.file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"ERROR: could not read {args.file}: {e}")
            sys.exit(2)
        result = validate_agent_competency_cert(cert)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
