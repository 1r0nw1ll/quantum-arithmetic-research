#!/usr/bin/env python3
"""Validate SINQA accepted-update evidence against forgetting/regression drift."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_antiforgetting — read-only replay memory check "
    "for accepted SINQA updates, source evidence hashes, and zero-harm gates"
)

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
PACKET_HASH_DOMAIN = "qa_self_improving_neural_qa_packet_v0"
SOURCE_FILE_HASH_PREFIX = b"qa_self_improving_neural_qa_file_v0\x00"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(SOURCE_FILE_HASH_PREFIX)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - validator reports parse failures.
        return None, f"{path}: invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, f"{path}: expected JSON object"
    return obj, None


def read_ledger(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{path}: file not found"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            errors.append(f"line {line_no}: empty line")
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: row must be object")
            continue
        rows.append(row)
    return rows, errors


def decision(packet: dict[str, Any]) -> str:
    promotion = packet.get("promotion")
    value = promotion.get("decision") if isinstance(promotion, dict) else None
    return value if value in {"accepted", "rejected"} else "unknown"


def evidence(packet: dict[str, Any]) -> dict[str, Any]:
    obj = packet.get("evidence")
    return obj if isinstance(obj, dict) else {}


def replay_gate(packet: dict[str, Any]) -> dict[str, Any]:
    obj = packet.get("replay_gate")
    return obj if isinstance(obj, dict) else {}


def candidate(packet: dict[str, Any]) -> dict[str, Any]:
    obj = packet.get("candidate")
    return obj if isinstance(obj, dict) else {}


def int_value(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def classify_domain(packet: dict[str, Any], source_ref: str) -> str:
    cand = candidate(packet)
    text = " ".join(
        str(part).lower()
        for part in [source_ref, cand.get("artifact_ref"), cand.get("description")]
        if part is not None
    )
    if "self_improving_neural_qa/general_ml" in text or "general_ml" in text:
        return "general_ml"
    if "hsi" in text or "salinas" in text or "paviau" in text or "pavia" in text or "indian_pines" in text:
        return "hsi"
    if cand.get("kind") in {"capacity_patch", "configuration_patch"}:
        return "config"
    return "other"


def source_metrics(source: dict[str, Any]) -> dict[str, Any]:
    schema = source.get("schema_version")
    if schema == "QA_GENERAL_ML_CORRECTION_REPLAY.v0" or "fixed_ensemble_errors" in source:
        return {
            "kind": "correction_replay",
            "fixed": int_value(source, "fixed_ensemble_errors"),
            "protected": int_value(source, "test_rows"),
            "harmed": int_value(source, "harmed_correct_ensemble_rows"),
            "deterministic": True,
        }
    if schema == "QA_GENERAL_ML_SINQA_CONFIG_PROPOSAL.v0" or "new_failures_fixed" in source:
        return {
            "kind": "config_proposal",
            "fixed": int_value(source, "new_failures_fixed"),
            "protected": int_value(source, "protected_cases_replayed"),
            "harmed": int_value(source, "protected_cases_harmed"),
            "deterministic": source.get("deterministic_replay", True) is True,
        }
    return {
        "kind": str(schema or "unknown"),
        "fixed": 0,
        "protected": 0,
        "harmed": 0,
        "deterministic": False,
    }


def validate_item(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    errors: list[str] = []
    packet = row.get("packet")
    packet_hash = row.get("packet_hash")
    if not isinstance(packet, dict):
        return {"ok": False, "line": line_no, "errors": [f"line {line_no}: packet must be object"]}
    if packet_hash != domain_sha256(PACKET_HASH_DOMAIN, packet):
        errors.append(f"line {line_no}: packet_hash mismatch")

    ev = evidence(packet)
    source_ref = ev.get("source_replay_ref")
    source_hash = ev.get("source_replay_hash")
    if not isinstance(source_ref, str) or not source_ref:
        errors.append(f"line {line_no}: accepted packet missing evidence.source_replay_ref")
        source_ref = ""
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        errors.append(f"line {line_no}: accepted packet missing 64-hex evidence.source_replay_hash")
        source_hash = ""

    gate = replay_gate(packet)
    if int_value(gate, "new_failures_fixed") <= 0:
        errors.append(f"line {line_no}: accepted packet no longer records positive fixes")
    if int_value(gate, "protected_cases_replayed") <= 0:
        errors.append(f"line {line_no}: accepted packet no longer records protected replay")
    if int_value(gate, "protected_cases_harmed") != 0:
        errors.append(f"line {line_no}: accepted packet records protected harm")
    if gate.get("deterministic_replay") is not True:
        errors.append(f"line {line_no}: accepted packet replay is not deterministic")

    source_path = resolve_path(source_ref) if source_ref else ROOT
    source_obj: dict[str, Any] | None = None
    metrics = {"kind": "missing", "fixed": 0, "protected": 0, "harmed": 0, "deterministic": False}
    computed_hash = None
    if source_ref:
        if not source_path.exists():
            errors.append(f"line {line_no}: source evidence file not found: {source_ref}")
        else:
            computed_hash = file_hash(source_path)
            if source_hash and computed_hash != source_hash:
                errors.append(f"line {line_no}: source evidence hash mismatch for {source_ref}")
            source_obj, parse_error = read_json_object(source_path)
            if parse_error is not None:
                errors.append(f"line {line_no}: {parse_error}")
            elif source_obj is not None:
                metrics = source_metrics(source_obj)
                if metrics["fixed"] <= 0:
                    errors.append(f"line {line_no}: source evidence has no positive fix")
                if metrics["protected"] <= 0:
                    errors.append(f"line {line_no}: source evidence has no protected replay cases")
                if metrics["harmed"] != 0:
                    errors.append(f"line {line_no}: source evidence has protected harm")
                if metrics["deterministic"] is not True:
                    errors.append(f"line {line_no}: source evidence is not deterministic")
                if metrics["fixed"] != int_value(gate, "new_failures_fixed"):
                    errors.append(f"line {line_no}: source fixed count disagrees with packet replay_gate")
                if metrics["protected"] != int_value(gate, "protected_cases_replayed"):
                    errors.append(f"line {line_no}: source protected count disagrees with packet replay_gate")
                if metrics["harmed"] != int_value(gate, "protected_cases_harmed"):
                    errors.append(f"line {line_no}: source harmed count disagrees with packet replay_gate")

    return {
        "ok": not errors,
        "line": line_no,
        "update_id": packet.get("update_id"),
        "candidate_kind": candidate(packet).get("kind"),
        "domain": classify_domain(packet, source_ref),
        "source_ref": source_ref,
        "source_hash": source_hash,
        "computed_source_hash": computed_hash,
        "source_kind": metrics["kind"],
        "fixed": metrics["fixed"],
        "protected": metrics["protected"],
        "harmed": metrics["harmed"],
        "errors": errors,
    }


def validate_antiforgetting(path: Path, max_items: int, min_accepted: int) -> dict[str, Any]:
    rows, errors = read_ledger(path)
    accepted: list[tuple[int, dict[str, Any]]] = []
    for line_no, row in enumerate(rows, start=1):
        packet = row.get("packet")
        if isinstance(packet, dict) and decision(packet) == "accepted":
            accepted.append((line_no, row))
    selected = accepted[-max_items:] if max_items > 0 else accepted
    items = [validate_item(row, line_no) for line_no, row in selected]
    item_errors = [error for item in items for error in item["errors"]]
    domains: dict[str, int] = {}
    source_kinds: dict[str, int] = {}
    for item in items:
        domains[item["domain"]] = domains.get(item["domain"], 0) + 1
        source_kinds[item["source_kind"]] = source_kinds.get(item["source_kind"], 0) + 1
    if len(selected) < min_accepted:
        errors.append(f"accepted evidence suite too small: checked {len(selected)}, required {min_accepted}")
    return {
        "ok": not errors and not item_errors,
        "ledger": str(path),
        "accepted_total": len(accepted),
        "checked": len(selected),
        "max_items": max_items,
        "min_accepted": min_accepted,
        "domains": dict(sorted(domains.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "missing": sum(1 for item in items if any("file not found" in error for error in item["errors"])),
        "hash_mismatches": sum(1 for item in items if any("hash mismatch" in error for error in item["errors"])),
        "harm_regressions": sum(1 for item in items if any("protected harm" in error for error in item["errors"])),
        "fix_regressions": sum(1 for item in items if any("no positive fix" in error for error in item["errors"])),
        "errors": errors + item_errors,
        "items": items,
    }


def print_text(result: dict[str, Any]) -> None:
    print(f"SINQA anti-forgetting: {'OK' if result['ok'] else 'ATTENTION'}")
    print(
        f"checked={result['checked']} accepted_total={result['accepted_total']} "
        f"missing={result['missing']} hash_mismatches={result['hash_mismatches']} "
        f"harm_regressions={result['harm_regressions']} domains={result['domains']}"
    )
    for error in result["errors"][:5]:
        print(f"error: {error}")


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "replay.json"
        ledger = root / "ledger.jsonl"
        source_obj = {
            "schema_version": "QA_GENERAL_ML_CORRECTION_REPLAY.v0",
            "dataset_slug": "selftest",
            "fixed_ensemble_errors": 2,
            "harmed_correct_ensemble_rows": 0,
            "test_rows": 5,
        }
        source.write_text(canonical_json(source_obj), encoding="utf-8")
        packet = {
            "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0",
            "update_id": "sinqa.v0.antiforgetting.selftest",
            "base_model": {"name": "selftest", "kind": "classifier", "neural": True},
            "candidate": {"kind": "correction_rule", "description": "general_ml selftest", "artifact_ref": "rules.json"},
            "evidence": {"source_replay_ref": str(source), "source_replay_hash": file_hash(source)},
            "replay_gate": {
                "new_failures_fixed": 2,
                "protected_cases_replayed": 5,
                "protected_cases_harmed": 0,
                "deterministic_replay": True,
                "trace_hash_before": hashlib.sha256(b"before").hexdigest(),
                "trace_hash_after": hashlib.sha256(b"after").hexdigest(),
            },
            "invariant_checks": [{"name": "zero_harm", "passed": True}],
            "promotion": {"decision": "accepted", "ledger_hash": hashlib.sha256(b"ledger").hexdigest()},
        }
        row = {"packet_hash": domain_sha256(PACKET_HASH_DOMAIN, packet), "packet": packet}
        ledger.write_text(canonical_json(row) + "\n", encoding="utf-8")
        good = validate_antiforgetting(ledger, max_items=8, min_accepted=1)
        source_obj["harmed_correct_ensemble_rows"] = 1
        source.write_text(canonical_json(source_obj), encoding="utf-8")
        bad = validate_antiforgetting(ledger, max_items=8, min_accepted=1)
        return {
            "ok": good["ok"] is True and bad["ok"] is False and bad["hash_mismatches"] == 1,
            "good": good,
            "bad": bad,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--min-accepted", type=int, default=1)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.max_items < 1:
        print(json.dumps({"ok": False, "errors": ["--max-items must be >= 1"]}, sort_keys=True))
        return 2
    if args.min_accepted < 0:
        print(json.dumps({"ok": False, "errors": ["--min-accepted must be >= 0"]}, sort_keys=True))
        return 2
    result = validate_antiforgetting(resolve_path(args.ledger), args.max_items, args.min_accepted)
    if args.text:
        print_text(result)
    else:
        print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
