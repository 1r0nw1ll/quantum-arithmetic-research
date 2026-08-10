#!/usr/bin/env python3
"""Run bounded self-improving neural QA proposal/replay rounds.

The loop is intentionally not a daemon. Each round selects one unseen replay
artifact, converts it into a candidate update packet, validates the resulting
ledger transaction, and appends only if the transaction is clean.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_loop — bounded continual learner runner; neural "
    "artifacts propose updates, deterministic QA ledger validation decides commit"
)

import argparse
import glob as globlib
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa")
DEFAULT_TRANSCRIPT = Path("results/self_improving_neural_qa/loop_transcript.jsonl")
DEFAULT_REPLAY_GLOB = "results/**/*correction_replay*.json"
DEFAULT_CONFIG_GLOB = "results/**/*sinqa_config_proposal*.json"
DEFAULT_RUNTIME_CONFIG = Path("results/self_improving_neural_qa/runtime/qa_general_ml_adapter_config.json")
ZERO_HASH = "0" * 64


def load_tool(path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": f"_loaded_{path.stem}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ZERO_HASH
    h = hashlib.sha256()
    h.update(b"qa_self_improving_neural_qa_loop_file_v0\x00")
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def load_runtime_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"enabled": False, "path": None, "exists": False, "hash": ZERO_HASH, "config": {}}
    resolved = resolve_path(path)
    if not resolved.exists():
        return {"enabled": True, "path": str(path), "exists": False, "hash": ZERO_HASH, "config": {}}
    obj = load_json(resolved)
    if obj is None:
        return {"enabled": True, "path": str(path), "exists": True, "hash": file_sha256(resolved), "config": {}}
    allowed = {
        "activation_plan_hash",
        "activation_update_id",
        "adapter.rank",
        "runtime.max_memory_mb",
        "runtime.max_parameters",
        "runtime.max_runtime_sec",
        "training.max_steps",
    }
    return {
        "enabled": True,
        "path": str(path),
        "exists": True,
        "hash": file_sha256(resolved),
        "config": {key: obj[key] for key in sorted(allowed) if key in obj},
    }


def iter_glob_paths(glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    if pattern_path.is_absolute():
        matches = globlib.glob(str(pattern_path), recursive=True)
    else:
        matches = globlib.glob(str(ROOT / glob_pattern), recursive=True)
    return sorted(Path(match) for match in matches)


def repo_relative(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def classify_replay(replay: dict[str, Any]) -> str | None:
    try:
        fixed = int(replay.get("fixed_ensemble_errors", 0))
        harmed = int(replay.get("harmed_correct_ensemble_rows", 0))
        protected = int(replay.get("test_rows", 0))
    except Exception:
        return None
    if protected <= 0:
        return None
    if fixed > 0 and harmed == 0:
        return "accepted"
    if fixed <= 0 or harmed > 0:
        return "rejected"
    return None


def packet_filename(packet: dict[str, Any], source_path: Path) -> str:
    update_id = str(packet["update_id"]).replace(".", "_")
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", source_path.stem)
    return f"{update_id}_{slug}.json"


def ledger_seen(ledger: Path) -> dict[str, set[str]]:
    seen = {"update_ids": set(), "packet_hashes": set(), "source_replay_hashes": set()}
    if not ledger.exists():
        return seen
    for raw in ledger.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            packet = row["packet"]
        except Exception:
            continue
        update_id = packet.get("update_id")
        packet_hash = row.get("packet_hash")
        evidence = packet.get("evidence") if isinstance(packet, dict) else None
        source_hash = evidence.get("source_replay_hash") if isinstance(evidence, dict) else None
        if isinstance(update_id, str):
            seen["update_ids"].add(update_id)
        if isinstance(packet_hash, str):
            seen["packet_hashes"].add(packet_hash)
        if isinstance(source_hash, str):
            seen["source_replay_hashes"].add(source_hash)
    return seen


def candidate_domain_rank(item: dict[str, Any]) -> int:
    lane = str(item.get("candidate_lane", "replay")).lower()
    if lane == "config":
        return 0
    domain = str(item.get("candidate_domain", "")).lower()
    replay = str(item.get("replay", "")).lower()
    if domain == "general_ml" or "/self_improving_neural_qa/general_ml/" in replay:
        return 1
    if "hsi" in domain or "hsi" in replay or domain in {"salinas", "paviau", "pavia", "indian_pines"}:
        return 3
    return 2


def candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    domain_rank = candidate_domain_rank(item)
    decision_rank = 0 if item["decision"] == "accepted" else 1
    fixed_rank = -int(item["fixed"])
    return (domain_rank, decision_rank, fixed_rank, item["replay"])


def discover_candidates(
    glob_pattern: str,
    ledger: Path,
    base_model_name: str,
    base_model_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    emitter = load_tool(TOOLS / "qa_emit_self_improving_neural_qa_packet.py")
    build_packet = emitter["build_packet"]
    domain_sha256 = emitter["domain_sha256"]
    seen = ledger_seen(ledger)
    candidates: list[dict[str, Any]] = []
    counts = {
        "scanned": 0,
        "skipped_invalid": 0,
        "skipped_existing_update_id": 0,
        "skipped_existing_source": 0,
    }

    for replay_path in iter_glob_paths(glob_pattern):
        counts["scanned"] += 1
        replay = load_json(replay_path)
        if replay is None:
            counts["skipped_invalid"] += 1
            continue
        decision = classify_replay(replay)
        if decision is None:
            counts["skipped_invalid"] += 1
            continue
        raw_rules_path = str(replay.get("rules_path") or "")
        rules_path = Path(raw_rules_path) if raw_rules_path else None
        if rules_path is not None and not rules_path.is_absolute():
            rules_path = ROOT / rules_path
        packet = build_packet(
            replay=replay,
            replay_path=repo_relative(replay_path),
            rules_path=repo_relative(rules_path) if rules_path is not None else None,
            base_model_name=base_model_name,
            base_model_kind=base_model_kind,
            update_id=None,
        )
        update_id = packet["update_id"]
        source_hash = packet["evidence"]["source_replay_hash"]
        if update_id in seen["update_ids"]:
            counts["skipped_existing_update_id"] += 1
            continue
        if source_hash in seen["source_replay_hashes"]:
            counts["skipped_existing_source"] += 1
            continue
        row = {
            "packet_hash": domain_sha256("qa_self_improving_neural_qa_packet_v0", packet),
            "packet": packet,
        }
        candidates.append(
            {
                "decision": decision,
                "fixed": int(replay.get("fixed_ensemble_errors", 0)),
                "harmed": int(replay.get("harmed_correct_ensemble_rows", 0)),
                "protected": int(replay.get("test_rows", 0)),
                "replay": str(replay_path),
                "packet": packet,
                "row": row,
                "source_replay_hash": source_hash,
                "candidate_lane": "replay",
                "candidate_domain": str(
                    replay.get("domain")
                    or replay.get("task_domain")
                    or replay.get("dataset_slug")
                    or replay.get("dataset")
                    or ""
                ).lower(),
            }
        )
    candidates.sort(key=candidate_sort_key)
    return candidates, counts


def discover_config_candidates(
    glob_pattern: str,
    ledger: Path,
    base_model_name: str,
    base_model_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    emitter = load_tool(TOOLS / "qa_emit_self_improving_neural_qa_config_packet.py")
    build_packet = emitter["build_packet"]
    domain_sha256 = emitter["domain_sha256"]
    seen = ledger_seen(ledger)
    candidates: list[dict[str, Any]] = []
    counts = {
        "scanned": 0,
        "skipped_invalid": 0,
        "skipped_existing_update_id": 0,
        "skipped_existing_source": 0,
    }

    for proposal_path in iter_glob_paths(glob_pattern):
        counts["scanned"] += 1
        proposal = load_json(proposal_path)
        if proposal is None:
            counts["skipped_invalid"] += 1
            continue
        try:
            packet = build_packet(
                proposal=proposal,
                proposal_path=repo_relative(proposal_path),
                base_model_name=base_model_name,
                base_model_kind=base_model_kind,
                update_id=None,
            )
            replay_gate = packet["replay_gate"]
            promotion = packet["promotion"]
            evidence = packet["evidence"]
        except Exception:
            counts["skipped_invalid"] += 1
            continue

        update_id = packet["update_id"]
        source_hash = evidence["source_replay_hash"]
        if update_id in seen["update_ids"]:
            counts["skipped_existing_update_id"] += 1
            continue
        if source_hash in seen["source_replay_hashes"]:
            counts["skipped_existing_source"] += 1
            continue
        row = {
            "packet_hash": domain_sha256("qa_self_improving_neural_qa_packet_v0", packet),
            "packet": packet,
        }
        candidates.append(
            {
                "decision": promotion["decision"],
                "fixed": int(replay_gate.get("new_failures_fixed", 0)),
                "harmed": int(replay_gate.get("protected_cases_harmed", 0)),
                "protected": int(replay_gate.get("protected_cases_replayed", 0)),
                "replay": str(proposal_path),
                "packet": packet,
                "row": row,
                "source_replay_hash": source_hash,
                "candidate_lane": "config",
                "candidate_domain": "config",
            }
        )
    candidates.sort(key=candidate_sort_key)
    return candidates, counts


def merge_discovery_counts(target: dict[str, int], prefix: str, counts: dict[str, int]) -> None:
    for key, val in counts.items():
        target[f"{prefix}_{key}"] += val


def validate_transaction(ledger: Path, row: dict[str, Any]) -> dict[str, Any]:
    validator = load_tool(TOOLS / "qa_self_improving_neural_qa_ledger_validate.py")
    validate_ledger = validator["validate_ledger"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_ledger = Path(tmp) / "ledger.jsonl"
        if ledger.exists():
            shutil.copyfile(ledger, tmp_ledger)
        with tmp_ledger.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(row) + "\n")
        return validate_ledger(tmp_ledger)


def ledger_summary(ledger: Path) -> dict[str, Any]:
    if not ledger.exists():
        return {
            "ok": True,
            "rows": 0,
            "accepted": 0,
            "rejected": 0,
            "unique_update_ids": 0,
            "unique_packet_hashes": 0,
            "unique_source_replay_hashes": 0,
            "errors": [],
        }
    validator = load_tool(TOOLS / "qa_self_improving_neural_qa_ledger_validate.py")
    return validator["validate_ledger"](ledger)


def previous_transcript_hash(transcript: Path) -> str:
    if not transcript.exists():
        return ZERO_HASH
    prev = ZERO_HASH
    for raw in transcript.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            record_hash = row.get("record_hash")
        except Exception:
            continue
        if isinstance(record_hash, str):
            prev = record_hash
    return prev


def transcript_row_count(transcript: Path) -> int:
    if not transcript.exists():
        return 0
    return sum(1 for raw in transcript.read_text(encoding="utf-8").splitlines() if raw.strip())


def append_transcript(record: dict[str, Any], transcript: Path) -> dict[str, Any]:
    transcript.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "record_hash": domain_sha256("qa_self_improving_neural_qa_loop_record_v0", record),
        "record": record,
    }
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(row) + "\n")
    return row


def commit_candidate(candidate: dict[str, Any], ledger: Path, out_dir: Path) -> dict[str, Any]:
    validation = validate_transaction(ledger, candidate["row"])
    if validation.get("ok") is not True:
        return {"ok": False, "reason": "transaction_validation_failed", "validation": validation}

    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = Path(candidate["replay"])
    packet_path = out_dir / packet_filename(candidate["packet"], replay_path)
    packet_path.write_text(canonical_json(candidate["packet"]), encoding="utf-8")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(candidate["row"]) + "\n")
    return {
        "ok": True,
        "packet": str(packet_path),
        "update_id": candidate["packet"]["update_id"],
        "decision": candidate["decision"],
        "fixed": candidate["fixed"],
        "harmed": candidate["harmed"],
        "protected": candidate["protected"],
        "replay": candidate["replay"],
        "candidate_lane": candidate.get("candidate_lane", "replay"),
    }


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    runtime_config = load_runtime_config(getattr(args, "runtime_config", DEFAULT_RUNTIME_CONFIG))
    accepted_commits = 0
    rejected_commits = 0
    rounds: list[dict[str, Any]] = []
    aggregate_discovery = {
        "replay_scanned": 0,
        "replay_skipped_invalid": 0,
        "replay_skipped_existing_update_id": 0,
        "replay_skipped_existing_source": 0,
        "config_scanned": 0,
        "config_skipped_invalid": 0,
        "config_skipped_existing_update_id": 0,
        "config_skipped_existing_source": 0,
    }

    for round_index in range(1, args.rounds + 1):
        candidates, discovery = discover_candidates(
            glob_pattern=args.glob,
            ledger=args.ledger,
            base_model_name=args.base_model_name,
            base_model_kind=args.base_model_kind,
        )
        config_candidates, config_discovery = discover_config_candidates(
            glob_pattern=args.config_glob,
            ledger=args.ledger,
            base_model_name=args.base_model_name,
            base_model_kind=args.base_model_kind,
        )
        candidates.extend(config_candidates)
        candidates.sort(key=candidate_sort_key)
        merge_discovery_counts(aggregate_discovery, "replay", discovery)
        merge_discovery_counts(aggregate_discovery, "config", config_discovery)
        if not candidates:
            rounds.append({"round": round_index, "status": "no_candidate"})
            break

        selected = None
        for candidate in candidates:
            if candidate["decision"] == "accepted" and accepted_commits >= args.max_accepted:
                continue
            if candidate["decision"] == "rejected" and rejected_commits >= args.max_rejected:
                continue
            selected = candidate
            break
        if selected is None:
            rounds.append({"round": round_index, "status": "limits_reached"})
            break

        if args.dry_run:
            before_summary = ledger_summary(args.ledger)
            result = {
                "ok": True,
                "dry_run": True,
                "update_id": selected["packet"]["update_id"],
                "decision": selected["decision"],
                "fixed": selected["fixed"],
                "harmed": selected["harmed"],
                "protected": selected["protected"],
                "replay": selected["replay"],
                "candidate_lane": selected.get("candidate_lane", "replay"),
                "runtime_config": runtime_config,
            }
        else:
            before_summary = ledger_summary(args.ledger)
            ledger_hash_before = file_sha256(args.ledger)
            result = commit_candidate(selected, args.ledger, args.out_dir)
            if result.get("ok") is not True:
                rounds.append({"round": round_index, "status": "failed", "result": result})
                break
            after_summary = ledger_summary(args.ledger)
            ledger_hash_after = file_sha256(args.ledger)
            transcript_record = {
                "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_LOOP_ROUND.v0",
                "round": transcript_row_count(args.transcript) + 1,
                "previous_record_hash": previous_transcript_hash(args.transcript),
                "ledger": str(args.ledger),
                "ledger_hash_before": ledger_hash_before,
                "ledger_hash_after": ledger_hash_after,
                "ledger_summary_before": before_summary,
                "ledger_summary_after": after_summary,
                "packet_hash": selected["row"]["packet_hash"],
                "update_id": selected["packet"]["update_id"],
                "decision": selected["decision"],
                "fixed": selected["fixed"],
                "harmed": selected["harmed"],
                "protected": selected["protected"],
                "source_replay_hash": selected["source_replay_hash"],
                "replay": selected["replay"],
                "candidate_lane": selected.get("candidate_lane", "replay"),
                "promoted_state_mutated": selected["decision"] == "accepted",
                "runtime_config": runtime_config,
                "stop_rule": {
                    "max_accepted": args.max_accepted,
                    "max_rejected": args.max_rejected,
                    "rounds_requested": args.rounds,
                },
            }
            transcript_row = append_transcript(transcript_record, args.transcript)
            result["transcript_hash"] = transcript_row["record_hash"]
            if selected["decision"] == "accepted":
                accepted_commits += 1
            else:
                rejected_commits += 1
        rounds.append({"round": round_index, "status": "proposed", "result": result})

        if args.dry_run:
            break
        if accepted_commits >= args.max_accepted and rejected_commits >= args.max_rejected:
            break

    final_validation = None
    if args.ledger.exists():
        validator = load_tool(TOOLS / "qa_self_improving_neural_qa_ledger_validate.py")
        final_validation = validator["validate_ledger"](args.ledger)

    return {
        "ok": all(item["status"] != "failed" for item in rounds),
        "dry_run": args.dry_run,
        "rounds": rounds,
        "committed": {"accepted": accepted_commits, "rejected": rejected_commits},
        "discovery": aggregate_discovery,
        "ledger": str(args.ledger),
        "transcript": str(args.transcript),
        "runtime_config": runtime_config,
        "final_validation": final_validation,
    }


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        ledger = tmp_root / "ledger.jsonl"
        out_dir = tmp_root / "packets"
        proposal = {
            "kind": "capacity_patch",
            "description": "loop self-test bounded adapter capacity proposal",
            "new_failures_fixed": 3,
            "protected_cases_replayed": 20,
            "protected_cases_harmed": 0,
            "config_patch": {
                "objective": "increase adapter rank in loop self-test",
                "activation_policy": "manual_after_cert",
                "diff": [{"op": "increase", "key": "adapter.rank", "before": 8, "after": 16}],
                "resource_bounds": {
                    "max_parameters": 25000000,
                    "max_memory_mb": 4096,
                    "max_runtime_sec": 3600,
                },
                "rollback": {
                    "artifact_ref": "results/selftest_capacity.rollback.json",
                    "artifact_hash": hashlib.sha256(b"loop-config-rollback").hexdigest(),
                },
            },
        }
        proposal_path = tmp_root / "selftest_sinqa_config_proposal.json"
        proposal_path.write_text(canonical_json(proposal), encoding="utf-8")
        args = argparse.Namespace(
            rounds=1,
            glob="results/qa_hsi_salinas*correction_replay*.json",
            config_glob=str(tmp_root / "*sinqa_config_proposal*.json"),
            ledger=ledger,
            out_dir=out_dir,
            base_model_name="selftest_hsi_ensemble",
            base_model_kind="classifier",
            max_accepted=1,
            max_rejected=0,
            dry_run=False,
            transcript=tmp_root / "loop_transcript.jsonl",
            runtime_config=tmp_root / "missing_runtime_config.json",
        )
        result = run_loop(args)
        config_ledger = tmp_root / "config_ledger.jsonl"
        config_out_dir = tmp_root / "config_packets"
        config_args = argparse.Namespace(
            rounds=1,
            glob=str(tmp_root / "no_replay_match_*.json"),
            config_glob=str(tmp_root / "*sinqa_config_proposal.json"),
            ledger=config_ledger,
            out_dir=config_out_dir,
            base_model_name="selftest_config_model",
            base_model_kind="classifier",
            max_accepted=1,
            max_rejected=0,
            dry_run=False,
            transcript=tmp_root / "config_loop_transcript.jsonl",
            runtime_config=tmp_root / "missing_runtime_config.json",
        )
        config_result = run_loop(config_args)
        priority_root = tmp_root / "priority"
        priority_root.mkdir()
        hsi_replay = {
            "dataset": "Salinas",
            "dataset_slug": "salinas",
            "domain": "hsi",
            "boundary": "selftest_hsi",
            "ensemble_errors": 100,
            "corrected_errors": 1,
            "fixed_ensemble_errors": 99,
            "harmed_correct_ensemble_rows": 0,
            "test_rows": 100,
            "n_rules": 1,
            "rows_touched": 99,
            "rules_path": "",
        }
        general_replay = {
            "dataset_slug": "selftest_general_ml",
            "domain": "general_ml",
            "boundary": "selftest_general",
            "ensemble_errors": 2,
            "corrected_errors": 1,
            "fixed_ensemble_errors": 1,
            "harmed_correct_ensemble_rows": 0,
            "test_rows": 2,
            "n_rules": 1,
            "rows_touched": 1,
            "rules_path": "",
        }
        (priority_root / "a_hsi_correction_replay.json").write_text(
            canonical_json(hsi_replay), encoding="utf-8"
        )
        (priority_root / "b_general_correction_replay.json").write_text(
            canonical_json(general_replay), encoding="utf-8"
        )
        priority_args = argparse.Namespace(
            rounds=1,
            glob=str(priority_root / "*correction_replay.json"),
            config_glob=str(priority_root / "no_config_*.json"),
            ledger=tmp_root / "priority_ledger.jsonl",
            out_dir=tmp_root / "priority_packets",
            base_model_name="selftest_priority_model",
            base_model_kind="classifier",
            max_accepted=1,
            max_rejected=0,
            dry_run=True,
            transcript=tmp_root / "priority_transcript.jsonl",
            runtime_config=tmp_root / "missing_runtime_config.json",
        )
        priority_result = run_loop(priority_args)
        ok = (
            result["ok"] is True
            and result["committed"]["accepted"] == 1
            and result["final_validation"]["ok"] is True
            and ledger.exists()
            and len(list(out_dir.glob("*.json"))) == 1
            and args.transcript.exists()
            and config_result["ok"] is True
            and config_result["committed"]["accepted"] == 1
            and config_result["rounds"][0]["result"]["candidate_lane"] == "config"
            and config_result["final_validation"]["ok"] is True
            and priority_result["ok"] is True
            and priority_result["rounds"][0]["result"]["replay"].endswith(
                "b_general_correction_replay.json"
            )
        )
        return {
            "ok": ok,
            "result": result,
            "config_result": config_result,
            "priority_result": priority_result,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--glob", default=DEFAULT_REPLAY_GLOB)
    parser.add_argument("--config-glob", default=DEFAULT_CONFIG_GLOB)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--base-model-name", default="hsi_ensemble")
    parser.add_argument("--base-model-kind", default="classifier")
    parser.add_argument("--max-accepted", type=int, default=1)
    parser.add_argument("--max-rejected", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.rounds < 1:
        print("error: --rounds must be >= 1", file=sys.stderr)
        return 2
    result = run_loop(args)
    print(json.dumps(result, sort_keys=True))
    final_validation = result.get("final_validation")
    return 0 if result["ok"] and (final_validation is None or final_validation.get("ok") is True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
