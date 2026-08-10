#!/usr/bin/env python3
"""Validator for QA_SELF_IMPROVING_NEURAL_QA_ACTIVATION_CERT.v1."""

from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator — validates self-improving neural QA activation plans; "
    "manual capacity/config patches remain non-mutating unless post-activation replay and rollback proof exist"
)

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_ACTIVATION_PLAN.v0"
PLAN_HASH_DOMAIN = "qa_self_improving_neural_qa_activation_plan_v0"
ROLLBACK_HASH_DOMAIN = b"qa_general_ml_config_proposal_file_v0\x00"
MAX_PARAMETERS = 1_000_000_000
MAX_MEMORY_MB = 131_072
MAX_RUNTIME_SEC = 86_400


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(ROLLBACK_HASH_DOMAIN)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return ROOT / path


def require_obj(parent: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    val = parent.get(key)
    if not isinstance(val, dict):
        errors.append(f"{key}: expected object")
        return {}
    return val


def require_int(parent: dict[str, Any], key: str, errors: list[str], minimum: int = 0) -> int:
    val = parent.get(key)
    if not isinstance(val, int) or isinstance(val, bool):
        errors.append(f"{key}: expected integer")
        return 0
    if val < minimum:
        errors.append(f"{key}: expected >= {minimum}")
    return val


def require_str(parent: dict[str, Any], key: str, errors: list[str]) -> str:
    val = parent.get(key)
    if not isinstance(val, str) or not val:
        errors.append(f"{key}: expected non-empty string")
        return ""
    return val


def require_hex64(parent: dict[str, Any], key: str, errors: list[str]) -> str:
    val = require_str(parent, key, errors)
    if len(val) != 64 or any(ch not in "0123456789abcdef" for ch in val):
        errors.append(f"{key}: expected 64-char lowercase hex")
    return val


def validate_checks(plan: dict[str, Any], errors: list[str]) -> None:
    checks = plan.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("checks: expected non-empty list")
        return
    passed_names = set()
    for idx, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"checks[{idx}]: expected object")
            continue
        name = require_str(check, "name", errors)
        if check.get("passed") is not True:
            errors.append(f"checks[{idx}]: every activation check must pass")
        passed_names.add(name)
    required = {
        "packet_was_accepted",
        "manual_activation_required",
        "rollback_hash_verified",
        "resource_bounds_within_hard_caps",
        "runtime_not_mutated_by_gate",
        "zero_protected_harm",
    }
    missing = sorted(required - passed_names)
    if missing:
        errors.append(f"checks: missing required checks {missing}")


def validate_runtime_mutation_proof(plan: dict[str, Any], errors: list[str]) -> None:
    if plan.get("runtime_mutated") is not True:
        return
    proof = require_obj(plan, "post_activation_proof", errors)
    replay = require_obj(proof, "post_activation_replay", errors)
    if replay.get("deterministic_replay") is not True:
        errors.append("post_activation_replay.deterministic_replay: expected true")
    require_int(replay, "protected_cases_replayed", errors, 1)
    harmed = require_int(replay, "protected_cases_harmed", errors, 0)
    if harmed != 0:
        errors.append("post_activation_replay.protected_cases_harmed: expected 0")
    rollback_proof = require_obj(proof, "rollback_proof", errors)
    if rollback_proof.get("verified") is not True:
        errors.append("rollback_proof.verified: expected true")
    require_hex64(rollback_proof, "artifact_hash", errors)
    require_hex64(rollback_proof, "computed_hash", errors)
    if rollback_proof.get("artifact_hash") != rollback_proof.get("computed_hash"):
        errors.append("rollback_proof: artifact_hash must equal computed_hash")


def validate_plan_obj(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    require_str(plan, "update_id", errors)
    require_hex64(plan, "packet_hash", errors)
    if plan.get("candidate_kind") not in {"capacity_patch", "configuration_patch"}:
        errors.append("candidate_kind must be capacity_patch or configuration_patch")
    if plan.get("activation_policy") != "manual_after_cert":
        errors.append("activation_policy must be manual_after_cert")
    if plan.get("manual_approval_required") is not True:
        errors.append("manual_approval_required must be true")
    if plan.get("activation_status") not in {"planned_not_applied", "applied_with_post_replay"}:
        errors.append("activation_status must be planned_not_applied or applied_with_post_replay")
    runtime_mutated = plan.get("runtime_mutated")
    if not isinstance(runtime_mutated, bool):
        errors.append("runtime_mutated must be boolean")
    if runtime_mutated is False and plan.get("activation_status") != "planned_not_applied":
        errors.append("non-mutating plan must use activation_status=planned_not_applied")
    if runtime_mutated is True and plan.get("activation_status") != "applied_with_post_replay":
        errors.append("mutating plan must use activation_status=applied_with_post_replay")

    diff = plan.get("config_diff")
    if not isinstance(diff, list) or not diff:
        errors.append("config_diff: expected non-empty list")

    bounds = require_obj(plan, "resource_bounds", errors)
    max_parameters = require_int(bounds, "max_parameters", errors, 1)
    max_memory_mb = require_int(bounds, "max_memory_mb", errors, 1)
    max_runtime_sec = require_int(bounds, "max_runtime_sec", errors, 1)
    if max_parameters > MAX_PARAMETERS:
        errors.append("resource_bounds.max_parameters exceeds hard cap")
    if max_memory_mb > MAX_MEMORY_MB:
        errors.append("resource_bounds.max_memory_mb exceeds hard cap")
    if max_runtime_sec > MAX_RUNTIME_SEC:
        errors.append("resource_bounds.max_runtime_sec exceeds hard cap")

    rollback = require_obj(plan, "rollback", errors)
    rollback_ref = require_str(rollback, "artifact_ref", errors)
    rollback_hash = require_hex64(rollback, "artifact_hash", errors)
    computed_hash = require_hex64(rollback, "computed_hash", errors)
    if rollback.get("verified") is not True:
        errors.append("rollback.verified must be true")
    if rollback_hash != computed_hash:
        errors.append("rollback artifact_hash must equal computed_hash")
    if rollback_ref:
        rollback_path = resolve_path(rollback_ref)
        if not rollback_path.exists():
            errors.append(f"rollback artifact missing: {rollback_ref}")
        else:
            actual = file_hash(rollback_path)
            if actual != rollback_hash:
                errors.append(f"rollback hash mismatch: expected {rollback_hash}, got {actual}")

    replay = require_obj(plan, "replay_gate", errors)
    if replay.get("deterministic_replay") is not True:
        errors.append("replay_gate.deterministic_replay must be true")
    if require_int(replay, "new_failures_fixed", errors, 1) <= 0:
        errors.append("replay_gate.new_failures_fixed must be positive")
    require_int(replay, "protected_cases_replayed", errors, 1)
    if require_int(replay, "protected_cases_harmed", errors, 0) != 0:
        errors.append("replay_gate.protected_cases_harmed must be 0")

    validate_checks(plan, errors)
    validate_runtime_mutation_proof(plan, errors)

    plan_hash = require_hex64(plan, "plan_hash", errors)
    core = {key: val for key, val in plan.items() if key != "plan_hash"}
    expected = domain_sha256(PLAN_HASH_DOMAIN, core)
    if plan_hash != expected:
        errors.append(f"plan_hash mismatch: expected {expected}, got {plan_hash}")
    return errors


def validate_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "errors": [f"{path}: file not found"], "path": str(path)}
    raw = path.read_text(encoding="utf-8").strip()
    errors: list[str] = []
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "errors": [f"invalid JSON: {exc}"], "path": str(path)}
    if not isinstance(plan, dict):
        return {"ok": False, "errors": ["plan root must be object"], "path": str(path)}
    if canonical_json(plan) != raw:
        errors.append("plan is not canonical JSON")
    errors.extend(validate_plan_obj(plan))
    return {
        "ok": not errors,
        "errors": errors,
        "path": str(path),
        "update_id": plan.get("update_id"),
        "runtime_mutated": plan.get("runtime_mutated"),
        "activation_status": plan.get("activation_status"),
    }


def self_test() -> dict[str, Any]:
    fixtures = Path(__file__).resolve().parent / "fixtures"
    cases = [
        ("pass_activation_plan.json", True),
        ("fail_runtime_mutated_without_proof.json", False),
        ("fail_bad_rollback_hash.json", False),
    ]
    results = []
    ok = True
    for name, should_pass in cases:
        result = validate_plan(fixtures / name)
        passed = bool(result["ok"])
        case_ok = passed is should_pass
        ok = ok and case_ok
        results.append({"fixture": name, "ok": case_ok, "passed": passed, "result": result})
    return {"ok": ok, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    paths = args.paths or sorted((Path(__file__).resolve().parent / "fixtures").glob("*.json"))
    all_ok = True
    for path in paths:
        result = validate_plan(path)
        all_ok = all_ok and bool(result["ok"])
        print(json.dumps(result, sort_keys=True))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
