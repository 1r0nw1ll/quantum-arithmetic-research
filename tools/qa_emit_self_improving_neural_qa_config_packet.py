#!/usr/bin/env python3
"""Emit self-improving neural QA configuration/capacity proposal packets."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_config_packet_emitter — emits bounded "
    "configuration/capacity proposals with rollback and manual activation"
)

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0"
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(b"qa_self_improving_neural_qa_file_v0\x00")
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def build_packet(
    proposal: dict[str, Any],
    proposal_path: Path,
    base_model_name: str,
    base_model_kind: str,
    update_id: str | None,
) -> dict[str, Any]:
    kind = str(proposal.get("kind") or "capacity_patch")
    if kind not in {"capacity_patch", "configuration_patch"}:
        raise ValueError("proposal.kind must be capacity_patch or configuration_patch")

    config_patch = proposal.get("config_patch")
    if not isinstance(config_patch, dict):
        raise ValueError("proposal.config_patch must be an object")

    fixed = int(proposal.get("new_failures_fixed", 0))
    harmed = int(proposal.get("protected_cases_harmed", 0))
    protected = int(proposal.get("protected_cases_replayed", 0))
    deterministic = bool(proposal.get("deterministic_replay", True))
    proposal_hash = file_hash(proposal_path)

    if update_id is None:
        seed = {
            "kind": kind,
            "proposal_path": str(proposal_path),
            "proposal_hash": proposal_hash,
            "config_patch": config_patch,
            "fixed": fixed,
            "harmed": harmed,
            "protected": protected,
        }
        update_id = "sinqa.v0." + domain_sha256("qa_sinqa_config_update_id_v0", seed)[:16]

    before_trace = {
        "base_model_name": base_model_name,
        "base_model_kind": base_model_kind,
        "proposal_hash": proposal_hash,
        "config_before": proposal.get("config_before", {}),
    }
    after_trace = {
        "proposal_hash": proposal_hash,
        "config_after": proposal.get("config_after", {}),
        "new_failures_fixed": fixed,
        "protected_cases_harmed": harmed,
        "protected_cases_replayed": protected,
    }

    checks = [
        {"name": "zero_harm", "passed": harmed == 0},
        {"name": "positive_incremental_fix", "passed": fixed > 0},
        {"name": "deterministic_replay", "passed": deterministic},
        {"name": "protected_replay_nonempty", "passed": protected > 0},
        {"name": "bounded_resource_delta", "passed": bool(proposal.get("bounded_resource_delta", True))},
        {"name": "rollback_available", "passed": bool(proposal.get("rollback_available", True))},
        {
            "name": "manual_activation_required",
            "passed": config_patch.get("activation_policy") == "manual_after_cert",
        },
    ]
    checks.extend(proposal.get("invariant_checks", []))

    core = {
        "schema_version": SCHEMA_VERSION,
        "update_id": update_id,
        "base_model": {
            "name": base_model_name,
            "kind": base_model_kind,
            "neural": True,
        },
        "candidate": {
            "kind": kind,
            "description": str(proposal.get("description") or config_patch.get("objective") or kind),
            "artifact_ref": str(proposal_path),
            "config_patch": config_patch,
        },
        "evidence": {
            "source_replay_ref": str(proposal_path),
            "source_replay_hash": proposal_hash,
        },
        "replay_gate": {
            "new_failures_fixed": fixed,
            "protected_cases_replayed": protected,
            "protected_cases_harmed": harmed,
            "deterministic_replay": deterministic,
            "trace_hash_before": domain_sha256("qa_sinqa_config_before_v0", before_trace),
            "trace_hash_after": domain_sha256("qa_sinqa_config_after_v0", after_trace),
        },
        "invariant_checks": checks,
    }

    rejection_reasons: list[str] = []
    if fixed <= 0:
        rejection_reasons.append("no_positive_incremental_fix")
    if harmed > 0:
        rejection_reasons.append("protected_harm")
    if protected <= 0:
        rejection_reasons.append("empty_protected_replay")
    if not deterministic:
        rejection_reasons.append("non_deterministic_replay")
    if not all(bool(check.get("passed")) for check in checks):
        rejection_reasons.append("failed_invariant_check")

    packet = dict(core)
    promotion = {
        "decision": "accepted" if not rejection_reasons else "rejected",
        "ledger_hash": domain_sha256("qa_self_improving_neural_qa_ledger_v0", core),
    }
    if rejection_reasons:
        promotion["rejection_reason"] = ",".join(rejection_reasons)
    packet["promotion"] = promotion
    return packet


def append_ledger(packet: dict[str, Any], ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "packet_hash": domain_sha256("qa_self_improving_neural_qa_packet_v0", packet),
        "packet": packet,
    }
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(row) + "\n")


def self_test() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proposal_path = root / "proposal.json"
        proposal = {
            "kind": "capacity_patch",
            "description": "increase adapter rank under bounded replay",
            "new_failures_fixed": 3,
            "protected_cases_replayed": 100,
            "protected_cases_harmed": 0,
            "config_patch": {
                "objective": "increase adapter rank",
                "activation_policy": "manual_after_cert",
                "diff": [{"op": "increase", "key": "adapter.rank", "before": 8, "after": 16}],
                "resource_bounds": {
                    "max_parameters": 25000000,
                    "max_memory_mb": 4096,
                    "max_runtime_sec": 3600,
                },
                "rollback": {
                    "artifact_ref": "results/selftest.rollback.json",
                    "artifact_hash": _hex("rollback"),
                },
            },
        }
        proposal_path.write_text(canonical_json(proposal), encoding="utf-8")
        packet = build_packet(proposal, proposal_path, "selftest", "classifier", None)
        ok = packet["promotion"]["decision"] == "accepted"
        return {"ok": ok, "decision": packet["promotion"]["decision"], "update_id": packet["update_id"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=False)
    parser.add_argument("--base-model-name", default="qa_neural_model")
    parser.add_argument("--base-model-kind", default="classifier")
    parser.add_argument("--update-id", default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--append-ledger", action="store_true")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.proposal is None or args.out is None:
        parser.error("--proposal and --out are required unless --self-test is used")

    proposal = load_json(args.proposal)
    packet = build_packet(
        proposal=proposal,
        proposal_path=args.proposal,
        base_model_name=args.base_model_name,
        base_model_kind=args.base_model_kind,
        update_id=args.update_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(canonical_json(packet), encoding="utf-8")
    if args.append_ledger:
        append_ledger(packet, args.ledger)
    print(json.dumps({
        "ok": True,
        "packet": str(args.out),
        "decision": packet["promotion"]["decision"],
        "ledger": str(args.ledger) if args.append_ledger else None,
        "update_id": packet["update_id"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
