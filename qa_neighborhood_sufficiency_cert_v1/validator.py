#!/usr/bin/env python3
"""
validator.py

QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1 / v1.1 validator (Machine Tract).

Certifies the Neighborhood Sufficiency Principle with branching Gate 3:
- Gate 1: JSON schema validity
- Gate 2: Canonical SHA-256 digest integrity
- Gate 3: Dominance logic (BRANCHING in v1.1):
    - DOMINANT or absent: require patch[r*] > spec (positive delta)
    - FAILS_BOUNDARY_CONTAMINATION: require patch[r*] < spec + boundary_metrics present
    - INCONCLUSIVE: pass without delta check
- Gate 4: Monotonic improvement up to minimal_radius (DOMINANT only)
- Gate 5: Bounded plateau beyond minimal_radius (DOMINANT only)

Failure modes: NOT_DOMINANT, NO_PLATEAU, SPLIT_MISMATCH, BOUNDARY_CONTAMINATION,
               MISSING_BOUNDARY_METRICS, NOT_A_FAILURE_CASE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"gate": self.gate, "status": self.status.value,
                "message": self.message, "details": self.details}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _validate_schema(obj: Dict[str, Any]) -> None:
    schema = _load_json(_schema_path())
    try:
        import jsonschema
    except ModuleNotFoundError:
        _validate_schema_minimal(obj, schema)
        return
    jsonschema.validate(instance=obj, schema=schema)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expect_type(value: Any, expected_type: type, path: str) -> None:
    _expect(isinstance(value, expected_type), f"{path} must be {expected_type.__name__}")


def _expect_required(obj: Dict[str, Any], fields: List[str], path: str) -> None:
    missing = [field for field in fields if field not in obj]
    _expect(not missing, f"{path} missing required fields: {', '.join(missing)}")


def _expect_number(value: Any, path: str) -> None:
    _expect(isinstance(value, (int, float)) and not isinstance(value, bool), f"{path} must be number")


def _expect_oa(value: Any, path: str) -> None:
    _expect_number(value, path)
    _expect(0.0 <= float(value) <= 1.0, f"{path} must be in [0,1]")


def _expect_hash64(value: Any, path: str) -> None:
    _expect(
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value),
        f"{path} must be 64 lowercase hex chars",
    )


def _validate_schema_minimal(obj: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Dependency-free checker for this family's concrete schema."""

    _expect_type(obj, dict, "$")
    _expect_required(obj, schema.get("required", []), "$")
    _expect(
        obj.get("schema_version") in {
            "QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1",
            "QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1.1",
        },
        "schema_version invalid",
    )
    for field in ("dataset", "classifier", "split_protocol"):
        _expect_type(obj.get(field), str, field)
    _expect_type(obj.get("seed"), int, "seed")

    generators = obj.get("generators")
    _expect_type(generators, dict, "generators")
    _expect_required(generators, ["spec", "patch"], "generators")
    spec_gen = generators["spec"]
    _expect_type(spec_gen, dict, "generators.spec")
    _expect_required(spec_gen, ["type", "dim"], "generators.spec")
    _expect_type(spec_gen["type"], str, "generators.spec.type")
    _expect(isinstance(spec_gen["dim"], int) and spec_gen["dim"] >= 1, "generators.spec.dim invalid")
    patch_gen = generators["patch"]
    _expect_type(patch_gen, dict, "generators.patch")
    _expect_required(patch_gen, ["type", "radius_values", "feature_dim"], "generators.patch")
    _expect_type(patch_gen["type"], str, "generators.patch.type")
    radius_values = patch_gen["radius_values"]
    _expect(isinstance(radius_values, list) and len(radius_values) >= 1,
            "generators.patch.radius_values must be nonempty")
    _expect(all(isinstance(r, int) and r >= 1 for r in radius_values),
            "generators.patch.radius_values entries invalid")
    _expect(isinstance(patch_gen["feature_dim"], int) and patch_gen["feature_dim"] >= 1,
            "generators.patch.feature_dim invalid")

    metrics = obj.get("metrics")
    _expect_type(metrics, dict, "metrics")
    _expect_required(metrics, ["spec", "patch", "minimal_radius", "failure_mode"], "metrics")
    spec_metrics = metrics["spec"]
    _expect_type(spec_metrics, dict, "metrics.spec")
    _expect_required(spec_metrics, ["oa"], "metrics.spec")
    _expect_oa(spec_metrics["oa"], "metrics.spec.oa")
    patch_metrics = metrics["patch"]
    _expect(isinstance(patch_metrics, dict) and len(patch_metrics) >= 1,
            "metrics.patch must be nonempty object")
    for key, value in patch_metrics.items():
        _expect_type(key, str, f"metrics.patch key {key!r}")
        _expect_type(value, dict, f"metrics.patch.{key}")
        _expect_required(value, ["oa"], f"metrics.patch.{key}")
        _expect_oa(value["oa"], f"metrics.patch.{key}.oa")
    _expect(isinstance(metrics["minimal_radius"], int) and metrics["minimal_radius"] >= 1,
            "metrics.minimal_radius invalid")
    _expect(
        metrics["failure_mode"] in {
            "BOUNDARY_CONTAMINATION",
            "INSUFFICIENT_CONTEXT",
            "NONE",
            "BOUNDARY_CONTAMINATION_DOMINATES",
        },
        "metrics.failure_mode invalid",
    )

    dominance = obj.get("dominance")
    _expect_type(dominance, dict, "dominance")
    _expect_required(dominance, ["patch_dominates_spec", "plateau_tolerance_pp"], "dominance")
    _expect_type(dominance["patch_dominates_spec"], bool, "dominance.patch_dominates_spec")
    _expect_number(dominance["plateau_tolerance_pp"], "dominance.plateau_tolerance_pp")
    _expect(float(dominance["plateau_tolerance_pp"]) >= 0.0,
            "dominance.plateau_tolerance_pp must be nonnegative")

    if "dominance_result" in obj:
        _expect(
            obj["dominance_result"] in {"DOMINANT", "FAILS_BOUNDARY_CONTAMINATION", "INCONCLUSIVE"},
            "dominance_result invalid",
        )
    if "boundary_metrics" in obj and obj["boundary_metrics"] is not None:
        boundary = obj["boundary_metrics"]
        _expect_type(boundary, dict, "boundary_metrics")
        if "fragmentation_proxy" in boundary:
            _expect_number(boundary["fragmentation_proxy"], "boundary_metrics.fragmentation_proxy")
        if "thin_region_proxy" in boundary:
            _expect_number(boundary["thin_region_proxy"], "boundary_metrics.thin_region_proxy")
        if "expected_fail_type" in boundary:
            _expect(boundary["expected_fail_type"] == "BOUNDARY_CONTAMINATION_DOMINATES",
                    "boundary_metrics.expected_fail_type invalid")
        if "notes" in boundary:
            _expect_type(boundary["notes"], str, "boundary_metrics.notes")

    digests = obj.get("digests")
    _expect_type(digests, dict, "digests")
    _expect_required(digests, ["canonical_sha256"], "digests")
    _expect_hash64(digests["canonical_sha256"], "digests.canonical_sha256")


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    copy = json.loads(_canonical_json_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_json_compact(copy))


def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Schema valid"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL, f"Schema invalid: {e}"))
        return results

    # Gate 2 — Canonical hash integrity
    want = obj.get("digests", {}).get("canonical_sha256", "")
    got = _compute_canonical_sha256(obj)
    if want == "0" * 64:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL,
                                  "canonical_sha256 is placeholder", {"got": got}))
        return results
    if want != got:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL,
                                  "canonical_sha256 mismatch", {"want": want, "got": got}))
        return results
    results.append(GateResult("gate_2_canonical_hash", GateStatus.PASS, "canonical_sha256 matches"))

    # Gate 3 — Dominance logic (branching for v1.1)
    metrics = obj["metrics"]
    spec_oa = metrics["spec"]["oa"]
    minimal_radius = metrics["minimal_radius"]
    patch_metrics = metrics["patch"]
    r_key = str(minimal_radius)

    if r_key not in patch_metrics:
        results.append(GateResult("gate_3_dominance", GateStatus.FAIL,
                                  f"minimal_radius={minimal_radius} not in patch metrics",
                                  {"keys": list(patch_metrics.keys())}))
        return results

    patch_oa_at_rstar = patch_metrics[r_key]["oa"]
    delta = patch_oa_at_rstar - spec_oa

    # Read dominance_result (v1.1); fall back to old patch_dominates_spec boolean (v1)
    dominance_result = obj.get("dominance_result")
    if dominance_result is None:
        # v1 backward-compat: infer from patch_dominates_spec field
        dominance_result = "DOMINANT" if obj.get("dominance", {}).get("patch_dominates_spec", True) else "FAILS_BOUNDARY_CONTAMINATION"

    if dominance_result == "DOMINANT":
        if patch_oa_at_rstar <= spec_oa:
            results.append(GateResult("gate_3_dominance", GateStatus.FAIL,
                                      f"NOT_DOMINANT: patch[{minimal_radius}]={patch_oa_at_rstar:.4f} <= spec={spec_oa:.4f}",
                                      {"patch_oa": patch_oa_at_rstar, "spec_oa": spec_oa,
                                       "dominance_result": dominance_result}))
            return results
        results.append(GateResult("gate_3_dominance", GateStatus.PASS,
                                   f"DOMINANT: patch[{minimal_radius}]={patch_oa_at_rstar:.4f} > spec={spec_oa:.4f}",
                                   {"delta_pp": round(delta * 100, 4)}))

    elif dominance_result == "FAILS_BOUNDARY_CONTAMINATION":
        boundary_metrics = obj.get("boundary_metrics")
        if boundary_metrics is None:
            results.append(GateResult("gate_3_dominance", GateStatus.FAIL,
                                      "MISSING_BOUNDARY_METRICS: dominance_result=FAILS_BOUNDARY_CONTAMINATION requires boundary_metrics",
                                      {"dominance_result": dominance_result}))
            return results
        if patch_oa_at_rstar >= spec_oa:
            results.append(GateResult("gate_3_dominance", GateStatus.FAIL,
                                      f"NOT_A_FAILURE_CASE: patch[{minimal_radius}]={patch_oa_at_rstar:.4f} >= spec={spec_oa:.4f} (expected patch < spec)",
                                      {"patch_oa": patch_oa_at_rstar, "spec_oa": spec_oa,
                                       "dominance_result": dominance_result}))
            return results
        expected = boundary_metrics.get("expected_fail_type", "")
        if expected != "BOUNDARY_CONTAMINATION_DOMINATES":
            results.append(GateResult("gate_3_dominance", GateStatus.FAIL,
                                      f"BOUNDARY_FAILTYPE_MISMATCH: expected_fail_type={expected!r}",
                                      {"expected_fail_type": expected}))
            return results
        results.append(GateResult("gate_3_dominance", GateStatus.PASS,
                                   f"VALIDATED_FAILURE: BOUNDARY_CONTAMINATION_DOMINATES confirmed "
                                   f"patch[{minimal_radius}]={patch_oa_at_rstar:.4f} < spec={spec_oa:.4f}",
                                   {"delta_pp": round(delta * 100, 4)}))

    else:  # INCONCLUSIVE
        results.append(GateResult("gate_3_dominance", GateStatus.PASS,
                                   f"INCONCLUSIVE: delta={delta*100:.2f}pp; no directional claim",
                                   {"delta_pp": round(delta * 100, 4)}))

    # Gates 4 + 5 only apply to DOMINANT certs
    if dominance_result != "DOMINANT":
        results.append(GateResult("gate_4_monotonicity", GateStatus.PASS,
                                   f"Skipped (dominance_result={dominance_result})"))
        results.append(GateResult("gate_5_plateau", GateStatus.PASS,
                                   f"Skipped (dominance_result={dominance_result})"))
        return results

    # Gate 4 — Monotonic improvement up to minimal_radius
    radius_values = sorted(int(k) for k in patch_metrics.keys())
    below_rstar = [r for r in radius_values if r < minimal_radius]
    for r in below_rstar:
        oa = patch_metrics[str(r)]["oa"]
        if oa >= patch_oa_at_rstar:
            results.append(GateResult("gate_4_monotonicity", GateStatus.FAIL,
                                       f"patch[{r}]={oa:.4f} >= patch[{minimal_radius}]={patch_oa_at_rstar:.4f} (expected strictly less)",
                                       {"r": r, "oa": oa, "r_star": minimal_radius, "oa_rstar": patch_oa_at_rstar}))
            return results
    results.append(GateResult("gate_4_monotonicity", GateStatus.PASS,
                               f"Monotonically increasing up to r*={minimal_radius}"))

    # Gate 5 — Bounded plateau beyond minimal_radius
    tol = obj["dominance"]["plateau_tolerance_pp"] / 100.0
    above_rstar = [r for r in radius_values if r > minimal_radius]
    for r in above_rstar:
        oa = patch_metrics[str(r)]["oa"]
        delta_r = abs(oa - patch_oa_at_rstar)
        if delta_r > tol:
            results.append(GateResult("gate_5_plateau", GateStatus.FAIL,
                                       f"NO_PLATEAU: patch[{r}]={oa:.4f} deviates {delta_r*100:.2f}pp from patch[{minimal_radius}]={patch_oa_at_rstar:.4f} (tol={tol*100:.1f}pp)",
                                       {"r": r, "oa": oa, "r_star_oa": patch_oa_at_rstar, "delta_pp": delta_r * 100}))
            return results
    if above_rstar:
        results.append(GateResult("gate_5_plateau", GateStatus.PASS,
                                   f"Plateau confirmed for r > {minimal_radius} within {obj['dominance']['plateau_tolerance_pp']}pp"))
    else:
        results.append(GateResult("gate_5_plateau", GateStatus.PASS,
                                   "No radii above r* to check (single-point plateau trivially holds)"))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(base, "fixtures")
    fixtures = [
        ("valid_houston.json",                              True,  None),
        ("valid_indian_pines.json",                         True,  None),
        ("valid_salinas.json",                              True,  None),
        ("valid_ksc_failure.json",                          True,  None),
        ("invalid_not_dominant.json",                       False, "gate_3_dominance"),
        ("invalid_no_plateau.json",                         False, "gate_5_plateau"),
        ("invalid_digest_mismatch.json",                    False, "gate_2_canonical_hash"),
        ("invalid_claims_dominant_but_negative_delta.json", False, "gate_3_dominance"),
    ]
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        if not os.path.exists(path):
            details.append({"fixture": name, "ok": None, "expected_ok": should_pass,
                             "failed_gates": [], "note": "MISSING"})
            continue
        obj = _load_json(path)
        res = validate_cert(obj)
        passed = _report_ok(res)
        if should_pass != passed:
            ok = False
        fail_gates = [r.gate for r in res if r.status == GateStatus.FAIL]
        if (not should_pass) and expected_fail_gate and expected_fail_gate not in fail_gates:
            ok = False
        details.append({"fixture": name, "ok": passed, "expected_ok": should_pass,
                         "failed_gates": fail_gates})
    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1.1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (skip)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(as_json=args.json)
    if not args.file:
        ap.print_help()
        return 2
    obj = _load_json(args.file)
    results = validate_cert(obj)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
