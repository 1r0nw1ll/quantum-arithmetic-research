#!/usr/bin/env python3
"""QA Core Spec Kernel [107] validator — QA_CORE_SPEC.v1

Checks:
  V1  generator names unique                            → DUPLICATE_GENERATOR_NAME
  V2  failure_algebra.types nonempty                    → MISSING_FAILURE_ALGEBRA
  V3  validation.gates == [0,1,2,3,4,5]                 → BAD_GATE_SEQUENCE
  V4  logging.required_fields ⊇ {move,fail_type,invariant_diff} → LOGGING_INCOMPLETE
  V5  all preserves_invariants refs resolve             → INVARIANT_REFERENCE_UNRESOLVED

Usage:
  python qa_core_spec_validate.py --self-test
  python qa_core_spec_validate.py --file fixtures/qa_core_spec_minimal_pass.json
  python qa_core_spec_validate.py --file fixtures/qa_core_spec_fail_*.json
"""

import json
import sys
import argparse
from pathlib import Path


# ── required logging fields ──────────────────────────────────────────────────
REQUIRED_LOG_FIELDS = frozenset(["move", "fail_type", "invariant_diff"])

# ── required gate sequence ────────────────────────────────────────────────────
REQUIRED_GATES = [0, 1, 2, 3, 4, 5]

# ── known spec-level fail types ───────────────────────────────────────────────
CORE_SPEC_FAIL_TYPES = frozenset([
    "DUPLICATE_GENERATOR_NAME",
    "MISSING_FAILURE_ALGEBRA",
    "BAD_GATE_SEQUENCE",
    "LOGGING_INCOMPLETE",
    "INVARIANT_REFERENCE_UNRESOLVED",
])


# ── output accumulator ────────────────────────────────────────────────────────
class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


# ── validator ─────────────────────────────────────────────────────────────────
def validate_core_spec(cert: dict) -> dict:
    out = _Out()
    detected_fails: set[str] = set()

    # ── schema_version / cert_type ──────────────────────────────────────────
    if cert.get("schema_version") != "QA_CORE_SPEC.v1":
        out.fail(f"schema_version must be 'QA_CORE_SPEC.v1', got {cert.get('schema_version')!r}")
    if cert.get("cert_type") != "qa_core_spec":
        out.fail(f"cert_type must be 'qa_core_spec', got {cert.get('cert_type')!r}")

    # ── required top-level fields ───────────────────────────────────────────
    for field in ["certificate_id", "spec_name", "spec_scope",
                  "state_space", "generators", "invariants",
                  "reachability", "failure_algebra", "logging",
                  "validation", "certificate_contract",
                  "validation_checks", "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    if out.errors:
        return _reconcile(cert, out, detected_fails)

    # ── V1: generator names unique ──────────────────────────────────────────
    generators = cert.get("generators", [])
    names = [g.get("name") for g in generators if isinstance(g, dict)]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        detected_fails.add("DUPLICATE_GENERATOR_NAME")

    # ── V2: failure_algebra.types nonempty ──────────────────────────────────
    fa = cert.get("failure_algebra", {})
    fa_types = fa.get("types", [])
    if not fa_types:
        detected_fails.add("MISSING_FAILURE_ALGEBRA")

    # ── V3: validation.gates == [0,1,2,3,4,5] ──────────────────────────────
    gates = cert.get("validation", {}).get("gates", [])
    if gates != REQUIRED_GATES:
        detected_fails.add("BAD_GATE_SEQUENCE")

    # ── V4: logging.required_fields ⊇ {move, fail_type, invariant_diff} ────
    log_fields = set(cert.get("logging", {}).get("required_fields", []))
    if not REQUIRED_LOG_FIELDS.issubset(log_fields):
        detected_fails.add("LOGGING_INCOMPLETE")

    # ── V5: all preserves_invariants references resolve ──────────────────────
    invariant_names = {inv.get("name") for inv in cert.get("invariants", [])
                       if isinstance(inv, dict)}
    for gen in generators:
        if not isinstance(gen, dict):
            continue
        for ref in gen.get("preserves_invariants", []):
            if ref not in invariant_names:
                detected_fails.add("INVARIANT_REFERENCE_UNRESOLVED")
                break

    return _reconcile(cert, out, detected_fails)


def _reconcile(cert: dict, out: _Out, detected_fails: set) -> dict:
    declared_result = cert.get("result", "")
    declared_ledger = cert.get("fail_ledger", [])
    declared_fail_types = {e.get("fail_type") for e in declared_ledger
                           if isinstance(e, dict)}

    # Unknown fail types in ledger
    for ft in declared_fail_types:
        if ft not in CORE_SPEC_FAIL_TYPES:
            out.warn(f"unrecognised fail_type in fail_ledger: {ft!r}")

    if declared_result == "PASS":
        if detected_fails:
            for ft in sorted(detected_fails):
                out.fail(f"cert declares PASS but detected: {ft}")
    elif declared_result == "FAIL":
        # Detected fails not in ledger
        missing_from_ledger = detected_fails - declared_fail_types
        for ft in sorted(missing_from_ledger):
            out.warn(f"detected {ft} but not declared in fail_ledger")
        # Ledger entries not detected
        phantom = declared_fail_types - detected_fails
        for ft in sorted(phantom):
            out.warn(f"fail_ledger declares {ft} but validator did not detect it")
    else:
        out.fail(f"result must be 'PASS' or 'FAIL', got {declared_result!r}")

    ok = len(out.errors) == 0
    label = "PASS" if ok else "FAIL"
    if ok and out.warnings:
        label = "PASS_WITH_WARNINGS"

    return {
        "ok": ok,
        "label": label,
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "errors": out.errors,
        "warnings": out.warnings,
        "detected_fails": sorted(detected_fails),
    }


# ── file entry point ──────────────────────────────────────────────────────────
def validate_file(path: Path) -> dict:
    with open(path) as f:
        cert = json.load(f)
    return validate_core_spec(cert)


# ── self-test ─────────────────────────────────────────────────────────────────
def self_test() -> dict:
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = {
        "qa_core_spec_minimal_pass.json": True,
        "qa_core_spec_fail_missing_failure_algebra.json": True,
        "qa_core_spec_fail_bad_gate_sequence.json": True,
        "qa_core_spec_fail_duplicate_generator_name.json": True,
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        r = validate_file(fpath)
        passed = r["ok"] == expect_ok
        if not passed:
            all_ok = False
        results.append({
            "fixture": fname,
            "ok": passed,
            "label": r["label"],
            "errors": r["errors"],
            "warnings": r["warnings"],
        })

    return {"ok": all_ok, "results": results}


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QA Core Spec [107] validator")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test against fixture suite")
    parser.add_argument("--file", type=Path,
                        help="Validate a single cert file")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if args.file:
        result = validate_file(args.file)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
