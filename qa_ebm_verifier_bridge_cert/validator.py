#!/usr/bin/env python3
"""
validator.py

QA_EBM_VERIFIER_BRIDGE_CERT.v1 validator (Machine Tract).

This cert family makes "verifier-gated acceptance" a checkable QA artifact:
- subject binding: state_after -> state_sha256
- verdict coherence: passed <-> fail_type
- invariant_diff binding: duplicates critical derived fields for deterministic auditing

Hash spec:
canonical_sha256 = sha256(canonical_json_compact(cert_with_canonical_sha256_zeroed))
where canonical_json_compact uses sort_keys=True, separators=(',',':'), ensure_ascii=False.
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
        return {
            "gate": self.gate,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


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


def _compute_state_sha256(state_after: str) -> str:
    # Deterministic binding: sha256(UTF-8 state string)
    return _sha256_hex(state_after)


def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Schema valid"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL, f"Schema invalid: {e}"))
        return results

    # Gate 2 — Canonical hash
    want = obj.get("digests", {}).get("canonical_sha256", "")
    got = _compute_canonical_sha256(obj)
    if not isinstance(want, str) or len(want) != 64:
        results.append(GateResult(
            "gate_2_canonical_hash",
            GateStatus.FAIL,
            "canonical_sha256 missing/invalid",
            {"want": want, "got": got},
        ))
        return results
    if want == "0" * 64:
        results.append(GateResult(
            "gate_2_canonical_hash",
            GateStatus.FAIL,
            "canonical_sha256 is placeholder",
            {"got": got},
        ))
        return results
    if want != got:
        results.append(GateResult(
            "gate_2_canonical_hash",
            GateStatus.FAIL,
            "canonical_sha256 mismatch",
            {"want": want, "got": got},
        ))
        return results
    results.append(GateResult("gate_2_canonical_hash", GateStatus.PASS, "canonical_sha256 matches"))

    # Gate 3 — Subject binding + verdict coherence + invariant_diff binding
    subject = obj["subject"]
    state_after = subject["state_after"]
    computed_state_sha = _compute_state_sha256(state_after)
    if subject["state_sha256"] != computed_state_sha:
        results.append(GateResult(
            "gate_3_subject_binding",
            GateStatus.FAIL,
            "subject.state_sha256 mismatch",
            {"computed": computed_state_sha, "got": subject["state_sha256"]},
        ))
        return results

    verdict = obj["verdict"]
    passed = bool(verdict["passed"])
    fail_type = verdict["fail_type"]
    if passed and fail_type != "OK":
        results.append(GateResult(
            "gate_3_subject_binding",
            GateStatus.FAIL,
            "passed=true requires fail_type=OK",
            {"passed": passed, "fail_type": fail_type},
        ))
        return results
    if (not passed) and fail_type == "OK":
        results.append(GateResult(
            "gate_3_subject_binding",
            GateStatus.FAIL,
            "passed=false forbids fail_type=OK",
            {"passed": passed, "fail_type": fail_type},
        ))
        return results

    inv = obj["invariant_diff"]
    if inv["state_sha256"] != subject["state_sha256"]:
        results.append(GateResult(
            "gate_3_subject_binding",
            GateStatus.FAIL,
            "invariant_diff.state_sha256 mismatch",
            {"subject": subject["state_sha256"], "invariant_diff": inv["state_sha256"]},
        ))
        return results
    if inv["verdict_fail_type"] != fail_type:
        results.append(GateResult(
            "gate_3_subject_binding",
            GateStatus.FAIL,
            "invariant_diff.verdict_fail_type mismatch",
            {"verdict_fail_type": fail_type, "invariant_diff": inv["verdict_fail_type"]},
        ))
        return results

    results.append(GateResult(
        "gate_3_subject_binding",
        GateStatus.PASS,
        "Subject bound + verdict coherent + invariant_diff bound",
    ))
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
        ("valid_min.json", True, None),
        ("invalid_missing_invariant_diff.json", False, "gate_1_schema_validity"),
        ("invalid_verdict_inconsistent.json", False, "gate_3_subject_binding"),
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
        details.append({
            "fixture": name,
            "ok": passed,
            "expected_ok": should_pass,
            "failed_gates": fail_gates,
        })

    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_EBM_VERIFIER_BRIDGE_CERT.v1 SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_EBM_VERIFIER_BRIDGE_CERT.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test on fixtures")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
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

