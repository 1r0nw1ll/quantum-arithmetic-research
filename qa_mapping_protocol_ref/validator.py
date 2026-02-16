#!/usr/bin/env python3
"""
validator.py

QA_MAPPING_PROTOCOL_REF.v1 validator (Machine Tract).

Validates a reference object that points to a QA_MAPPING_PROTOCOL.v1 mapping file.

Checks:
  - REF schema validity
  - ref_path resolves within repo root (no escape)
  - referenced file exists
  - optional ref_sha256 matches file bytes sha256
  - referenced file validates against QA_MAPPING_PROTOCOL.v1 schema
  - referenced determinism_contract meets v1 essentials
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    check: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _repo_root() -> str:
    # qa_mapping_protocol_ref/ is expected to live at repo root
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _mapping_schema_path() -> str:
    return os.path.join(_repo_root(), "qa_mapping_protocol", "schema.json")


def _validate_schema(obj: Dict[str, Any], schema_path: str) -> None:
    import jsonschema

    schema = _load_json(schema_path)
    jsonschema.validate(instance=obj, schema=schema)


def _enforce_mapping_determinism(mapping_obj: Dict[str, Any]) -> Tuple[bool, str]:
    dc = mapping_obj.get("determinism_contract", {})
    inv_diff_ok = isinstance(dc, dict) and dc.get("invariant_diff_defined") is True
    proof = dc.get("nondeterminism_proof") if isinstance(dc, dict) else None
    proof_ok = isinstance(proof, str) and bool(proof.strip())
    if inv_diff_ok and proof_ok:
        return True, "Determinism contract satisfied"
    return False, "Determinism contract missing/invalid"


def validate_ref(ref_obj: Dict[str, Any]) -> List[CheckResult]:
    results: List[CheckResult] = []

    # REF schema
    try:
        _validate_schema(ref_obj, _schema_path())
        results.append(CheckResult("ref_schema", CheckStatus.PASS, "Valid QA_MAPPING_PROTOCOL_REF.v1 schema"))
    except Exception as e:
        results.append(CheckResult("ref_schema", CheckStatus.FAIL, f"REF schema validation failed: {e}"))
        return results

    repo_root = _repo_root()
    repo_root_abs = os.path.normpath(os.path.abspath(repo_root))
    rel = ref_obj.get("ref_path", "")
    resolved_abs = os.path.normpath(os.path.abspath(os.path.join(repo_root_abs, rel)))
    if os.path.commonpath([repo_root_abs, resolved_abs]) != repo_root_abs:
        results.append(CheckResult(
            "ref_path_escape",
            CheckStatus.FAIL,
            "ref_path escapes repo root",
            {"ref_path": rel, "resolved": resolved_abs, "repo_root": repo_root_abs},
        ))
        return results
    results.append(CheckResult("ref_path_escape", CheckStatus.PASS, "ref_path stays within repo root",
                               {"resolved": resolved_abs}))

    if not os.path.exists(resolved_abs):
        results.append(CheckResult("ref_exists", CheckStatus.FAIL, "Referenced mapping file not found",
                                   {"resolved": resolved_abs}))
        return results
    results.append(CheckResult("ref_exists", CheckStatus.PASS, "Referenced mapping file exists",
                               {"resolved": resolved_abs}))

    if "ref_sha256" in ref_obj:
        want = str(ref_obj.get("ref_sha256", "")).lower().strip()
        got = _sha256_file(resolved_abs)
        if want != got:
            results.append(CheckResult(
                "ref_sha256",
                CheckStatus.FAIL,
                "ref_sha256 mismatch",
                {"expected": want, "got": got},
            ))
            return results
        results.append(CheckResult("ref_sha256", CheckStatus.PASS, "ref_sha256 matches", {"sha256": got}))

    # Validate referenced mapping
    try:
        mapping_obj = _load_json(resolved_abs)
        _validate_schema(mapping_obj, _mapping_schema_path())
        results.append(CheckResult("mapping_schema", CheckStatus.PASS, "Referenced mapping validates schema"))
    except Exception as e:
        results.append(CheckResult("mapping_schema", CheckStatus.FAIL, f"Referenced mapping schema invalid: {e}"))
        return results

    ok, msg = _enforce_mapping_determinism(mapping_obj)
    results.append(CheckResult(
        "mapping_determinism_contract",
        CheckStatus.PASS if ok else CheckStatus.FAIL,
        msg,
    ))

    return results


def _all_passed(results: List[CheckResult]) -> bool:
    return all(r.status == CheckStatus.PASS for r in results)


def _print_human(results: List[CheckResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.check}: {r.message}")


def _print_json(results: List[CheckResult]) -> None:
    payload = {"ok": _all_passed(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    valid = _load_json(os.path.join(base, "fixtures", "valid_ref.json"))
    invalid = _load_json(os.path.join(base, "fixtures", "invalid_ref_missing_path.json"))

    vr = validate_ref(valid)
    ir = validate_ref(invalid)

    ok = _all_passed(vr) and (not _all_passed(ir))

    if as_json:
        print(json.dumps({
            "ok": ok,
            "valid_ok": _all_passed(vr),
            "invalid_ok": _all_passed(ir),
        }, indent=2, sort_keys=True))
    else:
        print("=== QA_MAPPING_PROTOCOL_REF.v1 SELF-TEST ===")
        print(f"valid_ref.json:  {'PASS' if _all_passed(vr) else 'FAIL'}")
        print(f"invalid_ref_missing_path.json:  {'PASS' if _all_passed(ir) else 'FAIL'} (expected FAIL)")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_MAPPING_PROTOCOL_REF.v1 validator")
    ap.add_argument("file", nargs="?", help="mapping_protocol_ref JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test on fixtures")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = _load_json(args.file)
    results = validate_ref(obj)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _all_passed(results) else 'FAIL'}")
    return 0 if _all_passed(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
