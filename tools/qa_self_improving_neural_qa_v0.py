#!/usr/bin/env python3
"""Validate Self-Improving Neural QA v0 update packets.

This is a scaffold validator, not a registered cert family. It checks the
promotion boundary for neural-model updates: a neural model may propose an
update, but QA replay decides whether it is promoted.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_v0_validator — neural proposals are observer "
    "artifacts; accepted promotion requires deterministic QA replay, zero "
    "protected harm, and invariant checks; rejected promotion preserves failed "
    "gate evidence"
)

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0"
ALLOWED_CANDIDATE_KINDS = {
    "adapter",
    "calibration_patch",
    "capacity_patch",
    "correction_rule",
    "configuration_patch",
    "generator_pattern",
    "routing_patch",
}
CONFIG_PATCH_KINDS = {"capacity_patch", "configuration_patch"}
ALLOWED_CONFIG_KEYS = {
    "adapter.rank",
    "model.depth",
    "model.hidden_dim",
    "model.num_heads",
    "model.width",
    "runtime.max_memory_mb",
    "runtime.max_parameters",
    "runtime.max_runtime_sec",
    "training.batch_size",
    "training.learning_rate",
    "training.max_steps",
}
CONFIG_PATCH_OPS = {"set", "increase", "decrease"}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(Exception):
    """Raised when a v0 update packet fails validation."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValidationError("packet root must be a JSON object")
    return obj


def require_obj(parent: dict[str, Any], key: str) -> dict[str, Any]:
    obj = parent.get(key)
    if not isinstance(obj, dict):
        raise ValidationError(f"{key}: expected object")
    return obj


def require_str(parent: dict[str, Any], key: str) -> str:
    val = parent.get(key)
    if not isinstance(val, str) or not val:
        raise ValidationError(f"{key}: expected non-empty string")
    return val


def require_int(parent: dict[str, Any], key: str, minimum: int | None = None) -> int:
    val = parent.get(key)
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValidationError(f"{key}: expected integer")
    if minimum is not None and val < minimum:
        raise ValidationError(f"{key}: expected >= {minimum}, got {val}")
    return val


def require_hex64(parent: dict[str, Any], key: str) -> str:
    val = require_str(parent, key)
    if HEX64_RE.match(val) is None:
        raise ValidationError(f"{key}: expected 64-char lowercase hex hash")
    return val


def require_number(parent: dict[str, Any], key: str, minimum: float | None = None) -> int | float:
    val = parent.get(key)
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise ValidationError(f"{key}: expected number")
    if minimum is not None and val < minimum:
        raise ValidationError(f"{key}: expected >= {minimum}, got {val}")
    return val


def validate_evidence(packet: dict[str, Any]) -> dict[str, Any] | None:
    evidence = packet.get("evidence")
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValidationError("evidence: expected object")
    require_str(evidence, "source_replay_ref")
    require_hex64(evidence, "source_replay_hash")
    return evidence


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and value is not None


def validate_config_patch_candidate(candidate: dict[str, Any]) -> None:
    patch = candidate.get("config_patch")
    if not isinstance(patch, dict):
        raise ValidationError("candidate.config_patch: expected object for configuration/capacity patch")
    require_str(patch, "objective")
    if patch.get("activation_policy") != "manual_after_cert":
        raise ValidationError("candidate.config_patch.activation_policy: expected 'manual_after_cert'")

    diff = patch.get("diff")
    if not isinstance(diff, list) or not diff:
        raise ValidationError("candidate.config_patch.diff: expected non-empty list")
    if len(diff) > 16:
        raise ValidationError("candidate.config_patch.diff: expected at most 16 entries")
    for idx, item in enumerate(diff):
        if not isinstance(item, dict):
            raise ValidationError(f"candidate.config_patch.diff[{idx}]: expected object")
        op = require_str(item, "op")
        if op not in CONFIG_PATCH_OPS:
            raise ValidationError(
                f"candidate.config_patch.diff[{idx}].op: expected one of {sorted(CONFIG_PATCH_OPS)}"
            )
        key = require_str(item, "key")
        if key not in ALLOWED_CONFIG_KEYS:
            raise ValidationError(f"candidate.config_patch.diff[{idx}].key: unsupported key {key!r}")
        if not _scalar(item.get("before")) or not _scalar(item.get("after")):
            raise ValidationError(
                f"candidate.config_patch.diff[{idx}]: before/after must be scalar values"
            )
        if item["before"] == item["after"]:
            raise ValidationError(f"candidate.config_patch.diff[{idx}]: before and after must differ")

    bounds = patch.get("resource_bounds")
    if not isinstance(bounds, dict):
        raise ValidationError("candidate.config_patch.resource_bounds: expected object")
    max_parameters = require_int(bounds, "max_parameters", 1)
    max_memory_mb = require_int(bounds, "max_memory_mb", 1)
    max_runtime_sec = require_int(bounds, "max_runtime_sec", 1)
    if max_parameters > 1_000_000_000:
        raise ValidationError("candidate.config_patch.resource_bounds.max_parameters: exceeds hard cap")
    if max_memory_mb > 131_072:
        raise ValidationError("candidate.config_patch.resource_bounds.max_memory_mb: exceeds hard cap")
    if max_runtime_sec > 86_400:
        raise ValidationError("candidate.config_patch.resource_bounds.max_runtime_sec: exceeds hard cap")

    rollback = patch.get("rollback")
    if not isinstance(rollback, dict):
        raise ValidationError("candidate.config_patch.rollback: expected object")
    require_str(rollback, "artifact_ref")
    require_hex64(rollback, "artifact_hash")


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"schema_version: expected {SCHEMA_VERSION!r}, got {packet.get('schema_version')!r}"
        )
    update_id = require_str(packet, "update_id")

    base_model = require_obj(packet, "base_model")
    require_str(base_model, "name")
    require_str(base_model, "kind")
    if base_model.get("neural") is not True:
        raise ValidationError("base_model.neural: expected true")

    candidate = require_obj(packet, "candidate")
    kind = require_str(candidate, "kind")
    if kind not in ALLOWED_CANDIDATE_KINDS:
        raise ValidationError(
            f"candidate.kind: expected one of {sorted(ALLOWED_CANDIDATE_KINDS)}, got {kind!r}"
        )
    require_str(candidate, "description")
    require_str(candidate, "artifact_ref")
    if kind in CONFIG_PATCH_KINDS:
        validate_config_patch_candidate(candidate)
    evidence = validate_evidence(packet)

    replay = require_obj(packet, "replay_gate")
    fixed = require_int(replay, "new_failures_fixed", 0)
    protected = require_int(replay, "protected_cases_replayed", 1)
    harmed = require_int(replay, "protected_cases_harmed", 0)
    if replay.get("deterministic_replay") is not True:
        raise ValidationError("replay_gate.deterministic_replay: expected true")
    require_hex64(replay, "trace_hash_before")
    require_hex64(replay, "trace_hash_after")

    checks = packet.get("invariant_checks")
    if not isinstance(checks, list) or not checks:
        raise ValidationError("invariant_checks: expected non-empty list")
    all_checks_pass = True
    for idx, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValidationError(f"invariant_checks[{idx}]: expected object")
        require_str(check, "name")
        if not isinstance(check.get("passed"), bool):
            raise ValidationError(f"invariant_checks[{idx}].passed: expected boolean")
        all_checks_pass = all_checks_pass and bool(check["passed"])

    promotion = require_obj(packet, "promotion")
    decision = promotion.get("decision")
    if decision not in {"accepted", "rejected"}:
        raise ValidationError("promotion.decision: expected 'accepted' or 'rejected'")
    ledger_hash = require_hex64(promotion, "ledger_hash")

    if decision == "accepted":
        if fixed <= 0:
            raise ValidationError("accepted packet requires replay_gate.new_failures_fixed > 0")
        if harmed != 0:
            raise ValidationError("accepted packet requires replay_gate.protected_cases_harmed == 0")
        if not all_checks_pass:
            raise ValidationError("accepted packet requires every invariant check to pass")
        if kind in CONFIG_PATCH_KINDS:
            required_checks = {
                "bounded_resource_delta",
                "rollback_available",
                "manual_activation_required",
            }
            passed_checks = {
                str(check.get("name"))
                for check in checks
                if isinstance(check, dict) and check.get("passed") is True
            }
            missing_checks = sorted(required_checks - passed_checks)
            if missing_checks:
                raise ValidationError(
                    "accepted configuration/capacity patch missing passed checks "
                    f"{missing_checks}"
                )
    else:
        require_str(promotion, "rejection_reason")
        if fixed > 0 and harmed == 0 and all_checks_pass:
            raise ValidationError("rejected packet must preserve at least one failed gate")

    expected_hash = domain_sha256(
        "qa_self_improving_neural_qa_update_v0",
        {
            "update_id": update_id,
            "base_model": base_model,
            "candidate": candidate,
            "evidence": evidence,
            "replay_gate": replay,
            "invariant_checks": checks,
        },
    )

    return {
        "ok": True,
        "update_id": update_id,
        "candidate_kind": kind,
        "decision": decision,
        "has_evidence": evidence is not None,
        "new_failures_fixed": fixed,
        "protected_cases_harmed": harmed,
        "protected_cases_replayed": protected,
        "ledger_hash": ledger_hash,
        "computed_packet_hash": expected_hash,
    }


def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _pass_packet() -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "update_id": "sinqa.v0.selftest.pass",
        "base_model": {
            "name": "hsi_ensemble_selftest",
            "kind": "classifier",
            "neural": True,
        },
        "candidate": {
            "kind": "correction_rule",
            "description": "self-test residual correction",
            "artifact_ref": "results/selftest_rules.json",
        },
        "replay_gate": {
            "new_failures_fixed": 2,
            "protected_cases_replayed": 20,
            "protected_cases_harmed": 0,
            "deterministic_replay": True,
            "trace_hash_before": _hex("before"),
            "trace_hash_after": _hex("after"),
        },
        "invariant_checks": [
            {"name": "zero_harm", "passed": True},
            {"name": "integer_state", "passed": True},
        ],
    }
    promoted = dict(core)
    promoted["promotion"] = {
        "decision": "accepted",
        "ledger_hash": domain_sha256("qa_self_improving_neural_qa_ledger_v0", core),
    }
    return promoted


def _rejected_packet() -> dict[str, Any]:
    pkt = _pass_packet()
    pkt["update_id"] = "sinqa.v0.selftest.rejected"
    pkt["replay_gate"]["protected_cases_harmed"] = 1
    pkt["invariant_checks"][0]["passed"] = False
    pkt["promotion"] = {
        "decision": "rejected",
        "rejection_reason": "protected_harm",
        "ledger_hash": domain_sha256(
            "qa_self_improving_neural_qa_ledger_v0",
            {
                k: v
                for k, v in pkt.items()
                if k != "promotion"
            },
        ),
    }
    return pkt


def _capacity_packet() -> dict[str, Any]:
    pkt = _pass_packet()
    pkt["update_id"] = "sinqa.v0.selftest.capacity"
    pkt["candidate"] = {
        "kind": "capacity_patch",
        "description": "self-test bounded adapter capacity increase",
        "artifact_ref": "results/selftest_capacity_patch.json",
        "config_patch": {
            "objective": "increase adapter rank for replay-verified residuals",
            "activation_policy": "manual_after_cert",
            "diff": [
                {"op": "increase", "key": "adapter.rank", "before": 8, "after": 16},
                {"op": "set", "key": "training.max_steps", "before": 1000, "after": 1500},
            ],
            "resource_bounds": {
                "max_parameters": 25000000,
                "max_memory_mb": 4096,
                "max_runtime_sec": 3600,
            },
            "rollback": {
                "artifact_ref": "results/selftest_capacity_patch.rollback.json",
                "artifact_hash": _hex("capacity-rollback"),
            },
        },
    }
    pkt["invariant_checks"].extend([
        {"name": "bounded_resource_delta", "passed": True},
        {"name": "rollback_available", "passed": True},
        {"name": "manual_activation_required", "passed": True},
    ])
    pkt["promotion"] = {
        "decision": "accepted",
        "ledger_hash": domain_sha256(
            "qa_self_improving_neural_qa_ledger_v0",
            {
                k: v
                for k, v in pkt.items()
                if k != "promotion"
            },
        ),
    }
    return pkt


def self_test() -> dict[str, Any]:
    malformed = _rejected_packet()
    malformed["promotion"].pop("rejection_reason")
    cases = [
        ("accepted", _pass_packet(), True),
        ("rejected", _rejected_packet(), True),
        ("capacity", _capacity_packet(), True),
        ("malformed", malformed, False),
    ]
    results: list[dict[str, Any]] = []
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, packet, should_pass in cases:
            path = root / f"{name}.json"
            path.write_text(canonical_json(packet), encoding="utf-8")
            try:
                summary = validate_packet(load_json(path))
                passed = True
                error = None
            except Exception as exc:  # noqa: BLE001 - self-test reports validator failures.
                summary = None
                passed = False
                error = str(exc)
            case_ok = passed is should_pass
            ok = ok and case_ok
            results.append(
                {
                    "case": name,
                    "ok": case_ok,
                    "passed_validator": passed,
                    "summary": summary,
                    "error": error,
                }
            )
    return {"ok": ok, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    if not args.paths:
        print("error: provide packet paths or --self-test", file=sys.stderr)
        return 2

    all_ok = True
    for path in args.paths:
        try:
            summary = validate_packet(load_json(path))
            print(json.dumps({"path": str(path), **summary}, sort_keys=True))
        except Exception as exc:  # noqa: BLE001 - CLI reports validation failure.
            all_ok = False
            print(json.dumps({"path": str(path), "ok": False, "error": str(exc)}, sort_keys=True))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
