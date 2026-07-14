#!/usr/bin/env python3
"""Emit bounded capacity proposals from accepted SINQA neural replays.

This worker only creates proposal artifacts. It does not edit runtime config,
does not apply activation plans, and does not append to the SINQA ledger.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "general_ml_capacity_proposal_worker — scans accepted neural general-ML "
    "replay evidence and emits manual-activation capacity proposals"
)

import argparse
import glob as globlib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa/general_ml")
DEFAULT_RUNTIME_CONFIG = Path("results/self_improving_neural_qa/runtime/qa_general_ml_adapter_config.json")
DEFAULT_ARCHIVE_DIR = Path("results/self_improving_neural_qa/artifact_archive")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + canonical_json(obj).encode("utf-8")).hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def load_tool(path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": f"_loaded_{path.stem}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def current_capacity(path: Path) -> dict[str, int]:
    cfg = load_json(resolve_path(path)) or {}
    return {
        "adapter_rank": int(cfg.get("adapter.rank", 12)),
        "max_steps": int(cfg.get("training.max_steps", 1500)),
    }


def proposal_path_for(replay_path: Path, out_dir: Path) -> Path:
    slug = replay_path.stem.replace("qa_general_ml_correction_replay_", "")
    return out_dir / f"qa_general_ml_sinqa_config_proposal_{slug}.json"


def stable_profile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): stable_profile(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [stable_profile(v) for v in obj]
    return obj


def config_signature(obj: dict[str, Any]) -> dict[str, Any] | None:
    if obj.get("schema_version") != "QA_GENERAL_ML_SINQA_CONFIG_PROPOSAL.v0":
        return None
    patch = obj.get("config_patch") if isinstance(obj.get("config_patch"), dict) else {}
    return {
        "artifact_kind": "config_proposal",
        "schema_version": obj.get("schema_version"),
        "kind": obj.get("kind"),
        "diff": stable_profile(patch.get("diff")),
        "resource_bounds": stable_profile(patch.get("resource_bounds")),
        "fixed": obj.get("new_failures_fixed"),
        "harmed": obj.get("protected_cases_harmed"),
        "protected": obj.get("protected_cases_replayed"),
    }


def config_signature_hash(obj: dict[str, Any]) -> str | None:
    signature = config_signature(obj)
    if signature is None:
        return None
    return domain_sha256("qa_sinqa_artifact_signature_v0", signature)


def existing_config_signature_hashes(out_dir: Path, archive_dir: Path) -> set[str]:
    hashes: set[str] = set()
    patterns = [
        str(out_dir / "qa_general_ml_sinqa_config_proposal_*.json"),
        str(archive_dir / "**" / "qa_general_ml_sinqa_config_proposal_*.json"),
    ]
    for pattern in patterns:
        for path in globlib.glob(pattern, recursive=True):
            obj = load_json(Path(path))
            if obj is None:
                continue
            sig_hash = config_signature_hash(obj)
            if sig_hash is not None:
                hashes.add(sig_hash)
    return hashes


def iter_accepted_replay_refs(ledger: Path) -> list[str]:
    resolved = resolve_path(ledger)
    if not resolved.exists():
        return []
    refs: list[str] = []
    for raw in resolved.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        packet = row.get("packet")
        if not isinstance(packet, dict):
            continue
        if packet.get("promotion", {}).get("decision") != "accepted":
            continue
        evidence = packet.get("evidence")
        if not isinstance(evidence, dict):
            continue
        ref = evidence.get("source_replay_ref")
        if isinstance(ref, str):
            refs.append(ref)
    return refs


def replay_is_neural_general_ml(path: Path) -> bool:
    replay = load_json(path)
    if not replay:
        return False
    if replay.get("domain") != "general_ml":
        return False
    baseline = str(replay.get("baseline_control") or "")
    candidate = str(replay.get("candidate_control") or "")
    boundary = str(replay.get("boundary") or "")
    return baseline.startswith("baseline_") and (
        candidate.startswith("neural_adapter_") or "_to_neural_adapter_" in boundary
    )


def discover_candidates(ledger: Path, out_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for ref in iter_accepted_replay_refs(ledger):
        replay_path = resolve_path(Path(ref))
        if not replay_path.exists() or not replay_is_neural_general_ml(replay_path):
            continue
        if proposal_path_for(replay_path, out_dir).exists():
            continue
        candidates.append(replay_path)
    return sorted(set(candidates), key=repo_relative)


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_path(args.out_dir)
    candidates = discover_candidates(resolve_path(args.ledger), out_dir)
    emitted: list[dict[str, Any]] = []
    emitter = load_tool(ROOT / "tools" / "qa_emit_general_ml_config_proposal.py")
    cap = current_capacity(args.runtime_config)
    rank_before = cap["adapter_rank"]
    steps_before = cap["max_steps"]
    rank_after = min(args.max_adapter_rank, rank_before + args.adapter_rank_step)
    steps_after = min(args.max_steps_cap, steps_before + args.max_steps_step)
    if rank_after <= rank_before and steps_after <= steps_before:
        return {
            "ok": True,
            "emitted": [],
            "candidate_count": len(candidates),
            "reason": "capacity already at proposal cap",
        }
    existing_signatures = existing_config_signature_hashes(out_dir, resolve_path(args.archive_dir))
    skipped_existing_signature = 0
    for replay_path in candidates[: max(0, args.max_emits)]:
        replay = load_json(replay_path)
        if replay is None:
            continue
        probe = emitter["build_proposal"](
            replay_path=replay_path,
            replay=replay,
            rollback_ref="results/self_improving_neural_qa/general_ml/novelty_probe.rollback.json",
            rollback_hash="0" * 64,
            adapter_rank_before=rank_before,
            adapter_rank_after=rank_after,
            max_steps_before=steps_before,
            max_steps_after=steps_after,
        )
        sig_hash = config_signature_hash(probe)
        if sig_hash is not None and sig_hash in existing_signatures:
            skipped_existing_signature += 1
            continue
        if args.dry_run:
            emitted.append(
                {
                    "dry_run": True,
                    "replay": repo_relative(replay_path),
                    "proposal": repo_relative(proposal_path_for(replay_path, out_dir)),
                    "adapter_rank_before": rank_before,
                    "adapter_rank_after": rank_after,
                    "max_steps_before": steps_before,
                    "max_steps_after": steps_after,
                }
            )
            if sig_hash is not None:
                existing_signatures.add(sig_hash)
            continue
        summary = emitter["write_outputs"](
            replay_path=replay_path,
            out_dir=out_dir,
            adapter_rank_before=rank_before,
            adapter_rank_after=rank_after,
            max_steps_before=steps_before,
            max_steps_after=steps_after,
        )
        emitted.append(summary)
        if sig_hash is not None:
            existing_signatures.add(sig_hash)
    return {
        "ok": True,
        "dry_run": args.dry_run,
        "candidate_count": len(candidates),
        "skipped_existing_signature": skipped_existing_signature,
        "emitted": emitted,
    }


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        replay = root / "qa_general_ml_correction_replay_demo_baseline_majority_to_neural_adapter_rank12.json"
        replay.write_text(
            canonical_json(
                {
                    "domain": "general_ml",
                    "boundary": "baseline_majority_to_neural_adapter_rank12_paired_general_ml",
                    "baseline_control": "baseline_majority",
                    "candidate_control": "neural_adapter_rank12",
                    "fixed_ensemble_errors": 4,
                    "harmed_correct_ensemble_rows": 0,
                    "test_rows": 4,
                }
            ),
            encoding="utf-8",
        )
        ledger = root / "ledger.jsonl"
        ledger.write_text(
            canonical_json(
                {
                    "packet_hash": "0" * 64,
                    "packet": {
                        "promotion": {"decision": "accepted"},
                        "evidence": {"source_replay_ref": str(replay)},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runtime = root / "runtime.json"
        runtime.write_text(canonical_json({"adapter.rank": 12, "training.max_steps": 1500}), encoding="utf-8")
        args = argparse.Namespace(
            ledger=ledger,
            out_dir=root / "out",
            runtime_config=runtime,
            archive_dir=root / "archive",
            max_emits=1,
            adapter_rank_step=4,
            max_adapter_rank=24,
            max_steps_step=500,
            max_steps_cap=3000,
            dry_run=False,
        )
        first = run_worker(args)
        second = run_worker(args)
        ok = (
            first["ok"] is True
            and len(first["emitted"]) == 1
            and second["ok"] is True
            and len(second["emitted"]) == 0
        )
        return {"ok": ok, "first": first, "second": second}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--max-emits", type=int, default=1)
    parser.add_argument("--adapter-rank-step", type=int, default=4)
    parser.add_argument("--max-adapter-rank", type=int, default=24)
    parser.add_argument("--max-steps-step", type=int, default=500)
    parser.add_argument("--max-steps-cap", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.max_emits < 0:
        print("error: --max-emits must be non-negative")
        return 2
    result = run_worker(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
