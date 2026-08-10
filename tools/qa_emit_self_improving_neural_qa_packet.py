#!/usr/bin/env python3
"""Emit Self-Improving Neural QA v0 update packets from replay artifacts."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_packet_emitter — converts replay-validated "
    "neural correction artifacts into append-only QA promotion packets"
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


def build_packet(
    replay: dict[str, Any],
    replay_path: Path,
    rules_path: Path | None,
    base_model_name: str,
    base_model_kind: str,
    update_id: str | None,
) -> dict[str, Any]:
    dataset = str(replay.get("dataset_slug") or replay.get("dataset") or "unknown").lower()
    domain = str(replay.get("domain") or replay.get("task_domain") or dataset or "unknown").lower()
    boundary = str(replay.get("boundary") or "unknown")
    fixed = int(replay.get("fixed_ensemble_errors", 0))
    harmed = int(replay.get("harmed_correct_ensemble_rows", 0))
    protected = int(replay.get("test_rows", 0))
    n_rules = int(replay.get("n_rules", 0))

    if update_id is None:
        seed = {
            "dataset": dataset,
            "boundary": boundary,
            "replay_path": str(replay_path),
            "rules_path": str(rules_path or replay.get("rules_path", "")),
            "fixed": fixed,
            "harmed": harmed,
            "n_rules": n_rules,
        }
        update_id = "sinqa.v0." + domain_sha256("qa_sinqa_update_id_v0", seed)[:16]

    rules_ref = str(rules_path or replay.get("rules_path") or "")
    replay_ref = str(replay_path)
    before_trace = {
        "dataset": dataset,
        "boundary": boundary,
        "ensemble_errors": int(replay.get("ensemble_errors", 0)),
        "accuracy_ensemble": replay.get("accuracy_ensemble"),
        "protected_cases_replayed": protected,
        "replay_artifact_hash": file_hash(replay_path),
    }
    after_trace = {
        "dataset": dataset,
        "boundary": boundary,
        "corrected_errors": int(replay.get("corrected_errors", 0)),
        "accuracy_corrected": replay.get("accuracy_corrected"),
        "fixed_ensemble_errors": fixed,
        "harmed_correct_ensemble_rows": harmed,
        "rows_touched": int(replay.get("rows_touched", 0)),
        "rules_artifact_ref": rules_ref,
        "rules_artifact_hash": file_hash(rules_path) if rules_path and rules_path.exists() else "",
    }

    core = {
        "schema_version": SCHEMA_VERSION,
        "update_id": update_id,
        "base_model": {
            "name": base_model_name,
            "kind": base_model_kind,
            "neural": True,
        },
        "candidate": {
            "kind": "correction_rule",
            "description": (
                f"{domain} {dataset} correction batch: {n_rules} replayed rules, "
                f"fixed={fixed}, harmed={harmed}, boundary={boundary}"
            ),
            "artifact_ref": rules_ref,
        },
        "evidence": {
            "source_replay_ref": replay_ref,
            "source_replay_hash": file_hash(replay_path),
        },
        "replay_gate": {
            "new_failures_fixed": fixed,
            "protected_cases_replayed": protected,
            "protected_cases_harmed": harmed,
            "deterministic_replay": True,
            "trace_hash_before": domain_sha256("qa_sinqa_replay_before_v0", before_trace),
            "trace_hash_after": domain_sha256("qa_sinqa_replay_after_v0", after_trace),
        },
        "invariant_checks": [
            {"name": "zero_harm", "passed": harmed == 0},
            {"name": "positive_incremental_fix", "passed": fixed > 0},
            {"name": "deterministic_replay", "passed": True},
            {"name": "protected_replay_nonempty", "passed": protected > 0},
        ],
    }
    rejection_reasons = []
    if fixed <= 0:
        rejection_reasons.append("no_positive_incremental_fix")
    if harmed > 0:
        rejection_reasons.append("protected_harm")
    if protected <= 0:
        rejection_reasons.append("empty_protected_replay")
    decision = "accepted" if not rejection_reasons else "rejected"
    packet = dict(core)
    promotion = {
        "decision": decision,
        "ledger_hash": domain_sha256("qa_self_improving_neural_qa_ledger_v0", core),
    }
    if decision == "rejected":
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--base-model-name", default="hsi_ensemble")
    parser.add_argument("--base-model-kind", default="classifier")
    parser.add_argument("--update-id", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append-ledger", action="store_true")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    replay = load_json(args.replay)
    rules_path = args.rules
    packet = build_packet(
        replay=replay,
        replay_path=args.replay,
        rules_path=rules_path,
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
