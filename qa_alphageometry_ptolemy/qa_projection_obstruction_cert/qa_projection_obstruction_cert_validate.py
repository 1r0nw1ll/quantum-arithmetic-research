#!/usr/bin/env python3
"""QA Projection Obstruction Cert family [129] validator — QA_PROJECTION_OBSTRUCTION_CERT.v1

Family extension of QA_ENGINEERING_CORE_CERT.v1.

This family separates three layers explicitly:
1. native symbolic closure
2. representation-basis mismatch (still discrete)
3. physical device realization

This avoids conflating selector/encoding debt with actual electronic-device failure.

Validation checks:
  IH1  inherits_from == 'QA_ENGINEERING_CORE_CERT.v1'             → INVALID_PARENT_ENGINEERING_REFERENCE
  IH2  spec_scope == 'family_extension'                           → SPEC_SCOPE_MISMATCH
  IH3  gate_policy_respected == [0,1,2,3,4,5]                    → GATE_POLICY_INCOMPATIBLE
  PO1  engineering_context.native_invariants nonempty             → EMPTY_NATIVE_INVARIANTS
  PO2  native_witness references resolve + verdict valid          → NATIVE_WITNESS_INVALID / INVARIANT_REFERENCE_UNRESOLVED
  PO3  each representation layer is structurally valid            → REPRESENTATION_LAYER_INVALID
  PO4  layer obstruction_tags match recomputed debt tags          → REPRESENTATION_LEDGER_MISMATCH
  PO5  layer verdict matches recomputed debt class                → REPRESENTATION_VERDICT_MISMATCH
  PO6  physical_realization is structurally valid                 → PHYSICAL_LAYER_INVALID
  PO7  physical_realization verdict matches assessment status     → PHYSICAL_VERDICT_MISMATCH / PHYSICAL_LEDGER_MISMATCH
  PO8  any recomputed debt tags must appear in fail_ledger        → OBSTRUCTION_LEDGER_REQUIRED
  PO9  overall_verdict matches native + representation + physical → OVERALL_VERDICT_MISMATCH

Usage:
  python qa_projection_obstruction_cert_validate.py --self-test
  python qa_projection_obstruction_cert_validate.py --file fixtures/ppo_pass_arto_ternary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCHEMA_VERSION = "QA_PROJECTION_OBSTRUCTION_CERT.v1"
PARENT_SCHEMA = "QA_ENGINEERING_CORE_CERT.v1"
REQUIRED_GATES = [0, 1, 2, 3, 4, 5]

VALID_VERDICTS = frozenset(["CONSISTENT", "PARTIAL", "CONTRADICTS", "INCONCLUSIVE"])
VALID_LAYER_KINDS = frozenset([
    "display_projection",
    "rail_projection",
    "selector_cover",
    "published_topology",
    "other",
])
VALID_PHYSICAL_STATUS = frozenset(["UNASSESSED", "MEASURED"])
REPRESENTATION_OBSTRUCTION_TYPES = frozenset([
    "STATE_SPACE_RESIDUAL",
    "COST_INFLATION",
    "SELECTOR_AND_MERGE_DEBT",
    "TOPOLOGY_PART_COUNT_DEBT",
])
PHYSICAL_OBSTRUCTION_TYPES = frozenset([
    "INSUFFICIENT_STABLE_STATES",
    "THRESHOLD_MARGIN_WEAK",
    "NOISE_MARGIN_WEAK",
    "FANOUT_LIMITED",
    "TIMING_UNVERIFIED",
])
STRUCTURAL_FAIL_TYPES = frozenset([
    "INVALID_PARENT_ENGINEERING_REFERENCE",
    "SPEC_SCOPE_MISMATCH",
    "GATE_POLICY_INCOMPATIBLE",
    "EMPTY_NATIVE_INVARIANTS",
    "NATIVE_WITNESS_INVALID",
    "INVARIANT_REFERENCE_UNRESOLVED",
    "REPRESENTATION_LAYER_INVALID",
    "REPRESENTATION_LEDGER_MISMATCH",
    "REPRESENTATION_VERDICT_MISMATCH",
    "PHYSICAL_LAYER_INVALID",
    "PHYSICAL_LEDGER_MISMATCH",
    "PHYSICAL_VERDICT_MISMATCH",
    "OBSTRUCTION_LEDGER_REQUIRED",
    "OVERALL_VERDICT_MISMATCH",
])
KNOWN_FAIL_TYPES = (
    REPRESENTATION_OBSTRUCTION_TYPES | PHYSICAL_OBSTRUCTION_TYPES | STRUCTURAL_FAIL_TYPES
)


class _Out:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _to_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _to_positive_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 0.0:
            return number
    return None


def _recompute_representation_tags(layer: dict[str, object]) -> set[str] | None:
    native_state_count = _to_nonnegative_int(layer.get("native_state_count"))
    observed_state_count = _to_nonnegative_int(layer.get("observed_state_count"))
    unused_state_count = _to_nonnegative_int(layer.get("unused_state_count"))
    cost_ratio = _to_positive_number(layer.get("cost_ratio_vs_baseline"))
    selector_count = _to_nonnegative_int(layer.get("selector_count"))
    merge_count = _to_nonnegative_int(layer.get("merge_count"))
    topology_ratio = _to_positive_number(layer.get("topology_ratio_vs_baseline"))

    if (
        native_state_count is None
        or observed_state_count is None
        or unused_state_count is None
        or cost_ratio is None
        or selector_count is None
        or merge_count is None
        or topology_ratio is None
    ):
        return None

    tags: set[str] = set()
    if unused_state_count > 0 or observed_state_count < native_state_count:
        tags.add("STATE_SPACE_RESIDUAL")
    if cost_ratio > 1.0:
        tags.add("COST_INFLATION")
    if selector_count > 0 or merge_count > 0:
        tags.add("SELECTOR_AND_MERGE_DEBT")
    if topology_ratio > 1.0:
        tags.add("TOPOLOGY_PART_COUNT_DEBT")
    return tags


def _expected_representation_verdict(tags: set[str]) -> str:
    if not tags:
        return "CONSISTENT"
    if tags == {"STATE_SPACE_RESIDUAL"}:
        return "PARTIAL"
    return "CONTRADICTS"


def _recompute_physical_tags(physical: dict[str, object]) -> tuple[set[str] | None, str | None]:
    status = physical.get("assessment_status")
    verdict = physical.get("verdict")
    if status not in VALID_PHYSICAL_STATUS or verdict not in VALID_VERDICTS:
        return None, None

    if status == "UNASSESSED":
        return set(), "INCONCLUSIVE"

    required_symbol_states = _to_nonnegative_int(physical.get("required_symbol_states"))
    stable_state_count = _to_nonnegative_int(physical.get("stable_state_count"))
    threshold_margin_ratio = _to_positive_number(physical.get("threshold_margin_ratio"))
    noise_margin_ratio = _to_positive_number(physical.get("noise_margin_ratio"))
    required_fanout = _to_nonnegative_int(physical.get("required_fanout"))
    fanout_supported = _to_nonnegative_int(physical.get("fanout_supported"))
    timing_characterized = physical.get("timing_characterized")

    if (
        required_symbol_states is None
        or stable_state_count is None
        or threshold_margin_ratio is None
        or noise_margin_ratio is None
        or required_fanout is None
        or fanout_supported is None
        or not isinstance(timing_characterized, bool)
    ):
        return None, None

    tags: set[str] = set()
    if stable_state_count < required_symbol_states:
        tags.add("INSUFFICIENT_STABLE_STATES")
    if threshold_margin_ratio < 1.0:
        tags.add("THRESHOLD_MARGIN_WEAK")
    if noise_margin_ratio < 1.0:
        tags.add("NOISE_MARGIN_WEAK")
    if fanout_supported < required_fanout:
        tags.add("FANOUT_LIMITED")
    if not timing_characterized:
        tags.add("TIMING_UNVERIFIED")

    if not tags:
        expected = "CONSISTENT"
    elif tags == {"TIMING_UNVERIFIED"}:
        expected = "PARTIAL"
    else:
        expected = "CONTRADICTS"
    return tags, expected


def _reconcile(cert: dict[str, object], out: _Out, detected: set[str]) -> dict[str, object]:
    fail_ledger = cert.get("fail_ledger", [])
    declared_fail_types = {
        entry.get("fail_type")
        for entry in fail_ledger
        if isinstance(entry, dict) and isinstance(entry.get("fail_type"), str)
    }
    declared_result = cert.get("result", "")

    for fail_type in declared_fail_types:
        if fail_type not in KNOWN_FAIL_TYPES:
            out.fail(f"unknown fail_type in fail_ledger: {fail_type!r}")

    undeclared = detected - declared_fail_types
    for fail_type in sorted(undeclared):
        out.fail(f"detected fail {fail_type!r} not declared in fail_ledger")

    overclaimed = declared_fail_types - detected
    for fail_type in sorted(overclaimed):
        if fail_type in STRUCTURAL_FAIL_TYPES:
            out.fail(f"fail_ledger claims structural fail {fail_type!r} but validator did not detect it")

    if declared_result == "PASS":
        if detected:
            out.fail(f"result='PASS' but structural failures detected: {sorted(detected)}")
    elif declared_result == "FAIL":
        if not detected:
            out.fail("result='FAIL' but validator detected no structural failures")
    else:
        out.fail(f"result must be 'PASS' or 'FAIL', got {declared_result!r}")

    ok = len(out.errors) == 0
    label = "PASS" if ok and not out.warnings else ("PASS_WITH_WARNINGS" if ok else "FAIL")
    return {
        "ok": ok,
        "label": label,
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "errors": out.errors,
        "warnings": out.warnings,
        "detected_fails": sorted(detected),
    }


def validate_projection_obstruction_cert(cert: dict[str, object]) -> dict[str, object]:
    out = _Out()
    detected: set[str] = set()

    if cert.get("schema_version") != SCHEMA_VERSION:
        out.fail(f"schema_version must be {SCHEMA_VERSION!r}, got {cert.get('schema_version')!r}")
        return _reconcile(cert, out, detected)
    if cert.get("cert_type") != "projection_obstruction":
        out.fail(f"cert_type must be 'projection_obstruction', got {cert.get('cert_type')!r}")
        return _reconcile(cert, out, detected)

    for field in [
        "certificate_id",
        "title",
        "created_utc",
        "inherits_from",
        "spec_scope",
        "core_kernel_compatibility",
        "engineering_context",
        "native_witness",
        "representation_layers",
        "physical_realization",
        "overall_verdict",
        "validation_checks",
        "fail_ledger",
        "result",
    ]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")
    if out.errors:
        return _reconcile(cert, out, detected)

    if cert.get("inherits_from") != PARENT_SCHEMA:
        detected.add("INVALID_PARENT_ENGINEERING_REFERENCE")
        out.fail(
            f"IH1 INVALID_PARENT_ENGINEERING_REFERENCE: inherits_from must be {PARENT_SCHEMA!r}, "
            f"got {cert.get('inherits_from')!r}"
        )

    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")
        out.fail(
            f"IH2 SPEC_SCOPE_MISMATCH: spec_scope must be 'family_extension', got {cert.get('spec_scope')!r}"
        )

    compatibility = cert.get("core_kernel_compatibility", {})
    if not isinstance(compatibility, dict) or compatibility.get("gate_policy_respected") != REQUIRED_GATES:
        detected.add("GATE_POLICY_INCOMPATIBLE")
        out.fail("IH3 GATE_POLICY_INCOMPATIBLE: gate_policy_respected must be [0,1,2,3,4,5]")

    engineering_context = cert.get("engineering_context", {})
    if not isinstance(engineering_context, dict):
        detected.add("EMPTY_NATIVE_INVARIANTS")
        out.fail("PO1 EMPTY_NATIVE_INVARIANTS: engineering_context must be an object")
        return _reconcile(cert, out, detected)

    native_invariants = engineering_context.get("native_invariants", [])
    invariant_ids: set[str] = set()
    if not isinstance(native_invariants, list) or not native_invariants:
        detected.add("EMPTY_NATIVE_INVARIANTS")
        out.fail("PO1 EMPTY_NATIVE_INVARIANTS: native_invariants must be a non-empty list")
    else:
        for invariant in native_invariants:
            if (
                isinstance(invariant, dict)
                and isinstance(invariant.get("id"), str)
                and invariant.get("id").strip()
            ):
                invariant_ids.add(invariant["id"])
        if not invariant_ids:
            detected.add("EMPTY_NATIVE_INVARIANTS")
            out.fail("PO1 EMPTY_NATIVE_INVARIANTS: native_invariants must declare non-empty ids")

    native_witness = cert.get("native_witness", {})
    if not isinstance(native_witness, dict):
        detected.add("NATIVE_WITNESS_INVALID")
        out.fail("PO2 NATIVE_WITNESS_INVALID: native_witness must be an object")
        return _reconcile(cert, out, detected)

    native_supported = native_witness.get("lawful_invariants_supported", [])
    native_verdict = native_witness.get("verdict")
    if (
        not isinstance(native_supported, list)
        or not native_supported
        or not all(isinstance(item, str) and item.strip() for item in native_supported)
        or native_verdict not in VALID_VERDICTS
    ):
        detected.add("NATIVE_WITNESS_INVALID")
        out.fail(
            "PO2 NATIVE_WITNESS_INVALID: lawful_invariants_supported must be a non-empty string list "
            "and native_witness.verdict must be one of the valid verdicts"
        )
    for reference in native_supported if isinstance(native_supported, list) else []:
        if reference not in invariant_ids:
            detected.add("INVARIANT_REFERENCE_UNRESOLVED")
            out.fail(
                f"PO2 INVARIANT_REFERENCE_UNRESOLVED: native_witness references unknown invariant {reference!r}"
            )

    representation_layers = cert.get("representation_layers", [])
    if not isinstance(representation_layers, list) or not representation_layers:
        detected.add("REPRESENTATION_LAYER_INVALID")
        out.fail("PO3 REPRESENTATION_LAYER_INVALID: representation_layers must be a non-empty list")
        return _reconcile(cert, out, detected)

    representation_tags_required: set[str] = set()
    any_representation_debt = False

    for index, layer in enumerate(representation_layers, start=1):
        if not isinstance(layer, dict):
            detected.add("REPRESENTATION_LAYER_INVALID")
            out.fail(f"PO3 REPRESENTATION_LAYER_INVALID: layer {index} must be an object")
            continue

        layer_id = layer.get("layer_id", f"layer_{index}")
        kind = layer.get("kind")
        preserved_refs = layer.get("preserved_invariants", [])
        declared_tags = layer.get("obstruction_tags", [])
        declared_verdict = layer.get("verdict")

        if (
            not isinstance(layer_id, str)
            or not layer_id.strip()
            or kind not in VALID_LAYER_KINDS
            or declared_verdict not in VALID_VERDICTS
            or not isinstance(preserved_refs, list)
            or not all(isinstance(item, str) and item.strip() for item in preserved_refs)
            or not isinstance(declared_tags, list)
            or not all(
                isinstance(item, str) and item in REPRESENTATION_OBSTRUCTION_TYPES for item in declared_tags
            )
        ):
            detected.add("REPRESENTATION_LAYER_INVALID")
            out.fail(
                f"PO3 REPRESENTATION_LAYER_INVALID: layer {layer_id!r} has invalid kind, verdict, "
                "preserved_invariants, or obstruction_tags"
            )
            continue

        for reference in preserved_refs:
            if reference not in invariant_ids:
                detected.add("INVARIANT_REFERENCE_UNRESOLVED")
                out.fail(
                    f"PO3 INVARIANT_REFERENCE_UNRESOLVED: layer {layer_id!r} references unknown invariant {reference!r}"
                )

        recomputed_tags = _recompute_representation_tags(layer)
        if recomputed_tags is None:
            detected.add("REPRESENTATION_LAYER_INVALID")
            out.fail(
                f"PO3 REPRESENTATION_LAYER_INVALID: layer {layer_id!r} has missing or invalid quantitative witnesses"
            )
            continue

        declared_tag_set = set(declared_tags)
        if declared_tag_set != recomputed_tags:
            detected.add("REPRESENTATION_LEDGER_MISMATCH")
            out.fail(
                f"PO4 REPRESENTATION_LEDGER_MISMATCH: layer {layer_id!r} declares "
                f"{sorted(declared_tag_set)} but recomputed tags are {sorted(recomputed_tags)}"
            )

        expected_verdict = _expected_representation_verdict(recomputed_tags)
        if declared_verdict != expected_verdict:
            detected.add("REPRESENTATION_VERDICT_MISMATCH")
            out.fail(
                f"PO5 REPRESENTATION_VERDICT_MISMATCH: layer {layer_id!r} declares {declared_verdict!r} "
                f"but expected {expected_verdict!r}"
            )

        if recomputed_tags:
            any_representation_debt = True
            representation_tags_required.update(recomputed_tags)

    physical_realization = cert.get("physical_realization", {})
    if not isinstance(physical_realization, dict):
        detected.add("PHYSICAL_LAYER_INVALID")
        out.fail("PO6 PHYSICAL_LAYER_INVALID: physical_realization must be an object")
        return _reconcile(cert, out, detected)

    if not isinstance(physical_realization.get("notes", ""), str):
        detected.add("PHYSICAL_LAYER_INVALID")
        out.fail("PO6 PHYSICAL_LAYER_INVALID: physical_realization.notes must be a string")

    declared_physical_tags = physical_realization.get("device_tags", [])
    if not isinstance(declared_physical_tags, list) or not all(
        isinstance(item, str) and item in PHYSICAL_OBSTRUCTION_TYPES for item in declared_physical_tags
    ):
        detected.add("PHYSICAL_LAYER_INVALID")
        out.fail("PO6 PHYSICAL_LAYER_INVALID: physical_realization.device_tags must be a valid string list")
        declared_physical_tag_set: set[str] = set()
    else:
        declared_physical_tag_set = set(declared_physical_tags)

    recomputed_physical_tags, expected_physical_verdict = _recompute_physical_tags(physical_realization)
    if recomputed_physical_tags is None or expected_physical_verdict is None:
        detected.add("PHYSICAL_LAYER_INVALID")
        out.fail(
            "PO6 PHYSICAL_LAYER_INVALID: physical_realization has invalid assessment_status, verdict, or metrics"
        )
        recomputed_physical_tags = set()
        expected_physical_verdict = "INCONCLUSIVE"

    if declared_physical_tag_set != recomputed_physical_tags:
        detected.add("PHYSICAL_LEDGER_MISMATCH")
        out.fail(
            "PO7 PHYSICAL_LEDGER_MISMATCH: physical_realization.device_tags="
            f"{sorted(declared_physical_tag_set)} but recomputed tags are {sorted(recomputed_physical_tags)}"
        )

    if physical_realization.get("verdict") != expected_physical_verdict:
        detected.add("PHYSICAL_VERDICT_MISMATCH")
        out.fail(
            "PO7 PHYSICAL_VERDICT_MISMATCH: physical_realization.verdict="
            f"{physical_realization.get('verdict')!r}, expected {expected_physical_verdict!r}"
        )

    ledger_required_tags = representation_tags_required | recomputed_physical_tags
    declared_fail_types = {
        entry.get("fail_type")
        for entry in cert.get("fail_ledger", [])
        if isinstance(entry, dict) and isinstance(entry.get("fail_type"), str)
    }
    if ledger_required_tags and not ledger_required_tags.issubset(declared_fail_types):
        detected.add("OBSTRUCTION_LEDGER_REQUIRED")
        out.fail(
            "PO8 OBSTRUCTION_LEDGER_REQUIRED: fail_ledger missing obstruction tags "
            f"{sorted(ledger_required_tags - declared_fail_types)}"
        )

    overall_expected = "CONTRADICTS"
    if native_verdict == "CONSISTENT":
        if not any_representation_debt and expected_physical_verdict == "CONSISTENT":
            overall_expected = "CONSISTENT"
        else:
            overall_expected = "PARTIAL"

    if cert.get("overall_verdict") != overall_expected:
        detected.add("OVERALL_VERDICT_MISMATCH")
        out.fail(
            f"PO9 OVERALL_VERDICT_MISMATCH: overall_verdict={cert.get('overall_verdict')!r}, "
            f"expected {overall_expected!r}"
        )

    return _reconcile(cert, out, detected)


def validate_file(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        cert = json.load(handle)
    return validate_projection_obstruction_cert(cert)


def self_test() -> dict[str, object]:
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = {
        "ppo_pass_arto_ternary.json": True,
        "ppo_fail_physical_conflation.json": False,
        "ppo_fail_bad_invariant_ref.json": False,
    }
    results: list[dict[str, object]] = []
    all_ok = True

    for filename, expect_ok in expected.items():
        path = fixtures_dir / filename
        if not path.exists():
            results.append({"fixture": filename, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        payload = validate_file(path)
        passed = payload["ok"] == expect_ok
        if not passed:
            all_ok = False
        results.append(
            {
                "fixture": filename,
                "ok": passed,
                "label": payload["label"],
                "errors": payload["errors"],
                "warnings": payload["warnings"],
            }
        )

    return {"ok": all_ok, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="QA Projection Obstruction Cert [129] validator")
    parser.add_argument("--self-test", action="store_true", help="Run the fixture self-test")
    parser.add_argument("--file", type=Path, help="Validate a single certificate file")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if args.file:
        result = validate_file(args.file)
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
