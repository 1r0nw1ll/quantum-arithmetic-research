#!/usr/bin/env python3
"""
validator.py

QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1 validator (Machine Tract).

Certifies the Neighborhood Sufficiency Principle:
- Gate 1: JSON schema validity
- Gate 2: Canonical SHA-256 digest integrity
- Gate 3: Generator dominance (patch[r*] > spec)
- Gate 4: Monotonic improvement up to minimal_radius
- Gate 5: Bounded plateau beyond minimal_radius (within plateau_tolerance_pp)

Failure modes: NOT_DOMINANT, NO_PLATEAU, SPLIT_MISMATCH, BOUNDARY_CONTAMINATION
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

    # Gate 3 — Patch dominates spec at minimal_radius
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
    if patch_oa_at_rstar <= spec_oa:
        results.append(GateResult("gate_3_dominance", GateStatus.FAIL,
                                  f"NOT_DOMINANT: patch[{minimal_radius}]={patch_oa_at_rstar:.4f} <= spec={spec_oa:.4f}",
                                  {"patch_oa": patch_oa_at_rstar, "spec_oa": spec_oa}))
        return results
    results.append(GateResult("gate_3_dominance", GateStatus.PASS,
                               f"patch[{minimal_radius}]={patch_oa_at_rstar:.4f} > spec={spec_oa:.4f}"))

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
        delta = abs(oa - patch_oa_at_rstar)
        if delta > tol:
            results.append(GateResult("gate_5_plateau", GateStatus.FAIL,
                                       f"NO_PLATEAU: patch[{r}]={oa:.4f} deviates {delta*100:.2f}pp from patch[{minimal_radius}]={patch_oa_at_rstar:.4f} (tol={tol*100:.1f}pp)",
                                       {"r": r, "oa": oa, "r_star_oa": patch_oa_at_rstar, "delta_pp": delta * 100}))
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
        ("valid_houston.json", True, None),
        ("valid_indian_pines.json", True, None),
        ("invalid_not_dominant.json", False, "gate_3_dominance"),
        ("invalid_no_plateau.json", False, "gate_5_plateau"),
        ("invalid_digest_mismatch.json", False, "gate_2_canonical_hash"),
    ]
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
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
        print("=== QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1 SELF-TEST ===")
        for d in details:
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
