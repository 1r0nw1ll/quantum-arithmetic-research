#!/usr/bin/env python3
"""
validator.py

QA_LOCALITY_BOUNDARY_CERT.v1 validator (Machine Tract).

Certifies the Boundary Condition for locality-based generators:
- Gate 1: JSON schema validity
- Gate 2: Canonical SHA-256 digest integrity
- Gate 3: Failure curve — all patch[r] deltas <= 0 (patch never beats spec)
- Gate 4: Declared all_deltas_nonpositive flag must match computed reality
- Gate 5: boundary_geometry.fragmentation_scale_lt_r_star must be True
           (required structural explanation for the failure)

Failure modes: NOT_A_BOUNDARY_CASE, DELTA_FLAG_MISMATCH,
               MISSING_FRAGMENTATION_EXPLANATION, SCHEMA_INVALID, HASH_MISMATCH
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
    import jsonschema
    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


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

    # Gate 3 — Failure curve: all patch deltas <= 0
    fc = obj["failure_curve"]
    spec_oa = fc["spec_oa"]
    patch_oa_by_radius = fc["patch_oa_by_radius"]

    positive_deltas = {}
    for r_key, patch_oa in patch_oa_by_radius.items():
        delta = patch_oa - spec_oa
        if delta > 0:
            positive_deltas[r_key] = {"patch_oa": patch_oa, "spec_oa": spec_oa,
                                      "delta_pp": round(delta * 100, 4)}

    if positive_deltas:
        results.append(GateResult("gate_3_failure_curve", GateStatus.FAIL,
                                  f"NOT_A_BOUNDARY_CASE: patch[r] > spec at radii {list(positive_deltas.keys())}",
                                  {"positive_delta_radii": positive_deltas}))
        return results

    worst_delta_pp = round(min(patch_oa - spec_oa for patch_oa in patch_oa_by_radius.values()) * 100, 4)
    results.append(GateResult("gate_3_failure_curve", GateStatus.PASS,
                               f"All {len(patch_oa_by_radius)} radii have delta <= 0 (worst {worst_delta_pp:+.2f}pp)",
                               {"worst_delta_pp": worst_delta_pp}))

    # Gate 4 — Declared flag must match computed reality
    declared = fc["all_deltas_nonpositive"]
    computed = (len(positive_deltas) == 0)
    if declared != computed:
        results.append(GateResult("gate_4_delta_flag", GateStatus.FAIL,
                                  f"DELTA_FLAG_MISMATCH: declared all_deltas_nonpositive={declared} but computed={computed}",
                                  {"declared": declared, "computed": computed}))
        return results
    results.append(GateResult("gate_4_delta_flag", GateStatus.PASS,
                               f"all_deltas_nonpositive flag consistent (={declared})"))

    # Gate 5 — Structural explanation: fragmentation_scale_lt_r_star must be True
    bg = obj["boundary_geometry"]
    if not bg["fragmentation_scale_lt_r_star"]:
        results.append(GateResult("gate_5_fragmentation_explanation", GateStatus.FAIL,
                                  "MISSING_FRAGMENTATION_EXPLANATION: fragmentation_scale_lt_r_star must be True "
                                  "to certify a boundary-contamination failure",
                                  {"fragmentation_scale_lt_r_star": False,
                                   "fragmentation_proxy": bg["fragmentation_proxy"],
                                   "thin_region_proxy": bg["thin_region_proxy"]}))
        return results
    results.append(GateResult("gate_5_fragmentation_explanation", GateStatus.PASS,
                               f"fragmentation_scale_lt_r_star=True confirmed "
                               f"(fragmentation_proxy={bg['fragmentation_proxy']}, "
                               f"thin_region_proxy={bg['thin_region_proxy']})"))

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
        ("valid_ksc_boundary.json",        True,  None),
        ("invalid_not_a_boundary_case.json", False, "gate_3_failure_curve"),
        ("invalid_digest_mismatch.json",    False, "gate_2_canonical_hash"),
    ]
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        if not os.path.exists(path):
            details.append({"fixture": name, "ok": None, "expected_ok": should_pass,
                             "failed_gates": [], "note": "MISSING"})
            ok = False
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
        print("=== QA_LOCALITY_BOUNDARY_CERT.v1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_LOCALITY_BOUNDARY_CERT.v1 validator")
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
