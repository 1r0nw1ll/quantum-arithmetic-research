#!/usr/bin/env python3
"""Produce fresh general-ML replay artifacts for SINQA ingestion.

The worker scans existing QA-ML result summaries for paired-control benchmark
tables, selects unseen improvement boundaries, and emits replay/rules artifacts
through the existing general-ML correction replay emitter. It does not mutate
the SINQA ledger; the scheduled learner remains the promotion authority.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "general_ml_replay_worker — bounded deterministic producer for fresh "
    "non-HSI replay artifacts consumed by self-improving neural QA"
)

import argparse
import glob as globlib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_GLOB = "experiments/qa_ml/results_*.json"
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa/general_ml")
DEFAULT_RUNTIME_CONFIG = Path("results/self_improving_neural_qa/runtime/qa_general_ml_adapter_config.json")
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
DEFAULT_ARCHIVE_DIR = Path("results/self_improving_neural_qa/artifact_archive")
ZERO_HASH = "0" * 64
NEURAL_ADAPTER_SCHEMA = "QA_GENERAL_ML_NEURAL_ADAPTER_QA_RESIDUE.v0"
NEURAL_BENCHMARK_SELECTOR_SCHEMA = "QA_GENERAL_ML_NEURAL_BENCHMARK_SELECTOR.v0"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(b"qa_general_ml_replay_worker_file_v0\x00")
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_sha256(domain: str, obj: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + canonical_json(obj).encode("utf-8")).hexdigest()


def load_tool(path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": f"_loaded_{path.stem}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def iter_glob_paths(glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
    return sorted(Path(match) for match in globlib.glob(pattern, recursive=True))


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_runtime_config(path: Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {"exists": False, "hash": ZERO_HASH, "config": {}}
    obj = load_json(resolved) or {}
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
        "exists": True,
        "hash": file_hash(resolved),
        "config": {key: obj[key] for key in sorted(allowed) if key in obj},
    }


def existing_source_hashes(ledger: Path) -> set[str]:
    seen: set[str] = set()
    resolved = resolve_path(ledger)
    if not resolved.exists():
        return seen
    for raw in resolved.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            packet = row.get("packet")
            evidence = packet.get("evidence") if isinstance(packet, dict) else None
            source_hash = evidence.get("source_replay_hash") if isinstance(evidence, dict) else None
        except Exception:
            continue
        if isinstance(source_hash, str):
            seen.add(source_hash)
    return seen


def output_paths(source_path: Path, out_dir: Path, baseline: str, candidate: str) -> tuple[Path, Path]:
    slug = source_path.stem.replace("results_", "")
    replay = out_dir / f"qa_general_ml_correction_replay_{slug}_{baseline}_to_{candidate}.json"
    rules = out_dir / f"qa_general_ml_rules_{slug}_{baseline}_to_{candidate}.json"
    return replay, rules


def rounded_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    return value


def replay_signature(replay: dict[str, Any]) -> dict[str, Any] | None:
    if replay.get("schema_version") != "QA_GENERAL_ML_CORRECTION_REPLAY.v0":
        return None
    paired_cases = replay.get("paired_cases")
    case_profile = []
    if isinstance(paired_cases, list):
        for case in paired_cases:
            if isinstance(case, dict):
                case_profile.append(
                    {
                        "case_id": case.get("case_id"),
                        "fixed": case.get("fixed"),
                        "harmed": case.get("harmed"),
                        "delta": rounded_float(case.get("delta")),
                    }
                )
    return {
        "artifact_kind": "general_ml_replay",
        "schema_version": replay.get("schema_version"),
        "dataset_slug": replay.get("dataset_slug"),
        "boundary": replay.get("boundary"),
        "baseline_control": replay.get("baseline_control"),
        "candidate_control": replay.get("candidate_control"),
        "fixed": replay.get("fixed_ensemble_errors"),
        "harmed": replay.get("harmed_correct_ensemble_rows"),
        "protected": replay.get("test_rows"),
        "accuracy_corrected": rounded_float(replay.get("accuracy_corrected")),
        "accuracy_ensemble": rounded_float(replay.get("accuracy_ensemble")),
        "case_profile": case_profile,
    }


def replay_signature_hash(replay: dict[str, Any]) -> str | None:
    signature = replay_signature(replay)
    if signature is None:
        return None
    return domain_sha256("qa_sinqa_artifact_signature_v0", signature)


def existing_replay_signature_hashes(out_dir: Path, archive_dir: Path) -> set[str]:
    hashes: set[str] = set()
    patterns = [
        str(out_dir / "qa_general_ml_correction_replay_*.json"),
        str(archive_dir / "**" / "qa_general_ml_correction_replay_*.json"),
    ]
    for pattern in patterns:
        for path in globlib.glob(pattern, recursive=True):
            replay = load_json(Path(path))
            if replay is None:
                continue
            sig_hash = replay_signature_hash(replay)
            if sig_hash is not None:
                hashes.add(sig_hash)
    return hashes


def allowed_control_pairs(result: dict[str, Any], control_names: set[str]) -> list[tuple[str, str]]:
    if result.get("schema") in {NEURAL_ADAPTER_SCHEMA, NEURAL_BENCHMARK_SELECTOR_SCHEMA}:
        baselines = sorted(name for name in control_names if name.startswith("baseline_"))
        candidates = sorted(name for name in control_names if name.startswith(("neural_adapter_", "neural_benchmark_")))
        return [(baseline, candidate) for baseline in baselines for candidate in candidates]
    return [
        (baseline, candidate)
        for baseline in sorted(control_names)
        for candidate in sorted(control_names)
        if baseline != candidate
    ]


def score_control(emitter: dict[str, Any], control: dict[str, Any]) -> float:
    return float(emitter["score"](control))


def discover_boundaries(
    source_glob: str,
    out_dir: Path,
    ledger: Path,
    archive_dir: Path,
    min_delta: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    emitter = load_tool(ROOT / "tools" / "qa_emit_general_ml_correction_replay.py")
    seen_hashes = existing_source_hashes(ledger)
    seen_signatures = existing_replay_signature_hashes(out_dir, archive_dir)
    candidates: list[dict[str, Any]] = []
    counts = {
        "sources_scanned": 0,
        "sources_skipped_invalid": 0,
        "boundaries_scanned": 0,
        "boundaries_skipped_existing_output": 0,
        "boundaries_skipped_existing_ledger": 0,
        "boundaries_skipped_existing_signature": 0,
        "boundaries_skipped_no_signal": 0,
    }

    for source_path in iter_glob_paths(source_glob):
        counts["sources_scanned"] += 1
        result = load_json(source_path)
        if result is None or not isinstance(result.get("moduli_results"), dict):
            counts["sources_skipped_invalid"] += 1
            continue
        controls_by_case: list[dict[str, Any]] = []
        control_names: set[str] = set()
        for case in result["moduli_results"].values():
            if not isinstance(case, dict) or not isinstance(case.get("controls"), dict):
                continue
            controls = case["controls"]
            controls_by_case.append(controls)
            control_names.update(name for name, val in controls.items() if isinstance(val, dict))
        for baseline, candidate in allowed_control_pairs(result, control_names):
            counts["boundaries_scanned"] += 1
            replay_path, rules_path = output_paths(source_path, out_dir, baseline, candidate)
            if replay_path.exists() or rules_path.exists():
                counts["boundaries_skipped_existing_output"] += 1
                continue

            paired = []
            for case_id, case in sorted(result["moduli_results"].items()):
                if not isinstance(case, dict) or not isinstance(case.get("controls"), dict):
                    continue
                controls = case["controls"]
                base_obj = controls.get(baseline)
                cand_obj = controls.get(candidate)
                if not isinstance(base_obj, dict) or not isinstance(cand_obj, dict):
                    continue
                base_score = score_control(emitter, base_obj)
                cand_score = score_control(emitter, cand_obj)
                delta = cand_score - base_score
                paired.append((str(case_id), delta))
            if not paired:
                counts["boundaries_skipped_no_signal"] += 1
                continue
            fixed = sum(1 for _, delta in paired if delta > min_delta)
            harmed = sum(1 for _, delta in paired if delta < -min_delta)
            if fixed <= 0:
                counts["boundaries_skipped_no_signal"] += 1
                continue

            with tempfile.TemporaryDirectory() as tmp:
                scratch = Path(tmp)
                probe_rules = scratch / "rules.json"
                probe_replay = emitter["build_replay"](
                    source_path=source_path,
                    result=result,
                    rules_path=probe_rules,
                    baseline_name=baseline,
                    candidate_name=candidate,
                    min_delta=min_delta,
                )
                probe_replay_path = scratch / "replay.json"
                probe_replay_path.write_text(canonical_json(probe_replay), encoding="utf-8")
                source_hash = emitter["file_hash"](probe_replay_path)
                signature_hash = replay_signature_hash(probe_replay)
            if source_hash in seen_hashes:
                counts["boundaries_skipped_existing_ledger"] += 1
                continue
            if signature_hash is not None and signature_hash in seen_signatures:
                counts["boundaries_skipped_existing_signature"] += 1
                continue

            candidates.append(
                {
                    "source": source_path,
                    "baseline": baseline,
                    "candidate": candidate,
                    "fixed": fixed,
                    "harmed": harmed,
                    "protected": len(paired),
                    "mean_delta": sum(delta for _, delta in paired) / len(paired),
                }
            )

    candidates.sort(
        key=lambda item: (
            0 if item["harmed"] == 0 else 1,
            -int(item["fixed"]),
            int(item["harmed"]),
            -float(item["mean_delta"]),
            repo_relative(item["source"]),
            item["baseline"],
            item["candidate"],
        )
    )
    return candidates, counts


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_path(args.out_dir)
    ledger = resolve_path(args.ledger)
    runtime_config = load_runtime_config(args.runtime_config)
    candidates, discovery = discover_boundaries(
        source_glob=args.source_glob,
        out_dir=out_dir,
        ledger=ledger,
        archive_dir=resolve_path(args.archive_dir),
        min_delta=args.min_delta,
    )
    emitter = load_tool(ROOT / "tools" / "qa_emit_general_ml_correction_replay.py")
    emitted: list[dict[str, Any]] = []
    for candidate in candidates[: max(0, args.max_emits)]:
        if args.dry_run:
            emitted.append(
                {
                    "dry_run": True,
                    "source": repo_relative(candidate["source"]),
                    "baseline": candidate["baseline"],
                    "candidate": candidate["candidate"],
                    "fixed": candidate["fixed"],
                    "harmed": candidate["harmed"],
                    "protected": candidate["protected"],
                }
            )
            continue
        summary = emitter["write_outputs"](
            source_path=candidate["source"],
            out_dir=out_dir,
            baseline_name=candidate["baseline"],
            candidate_name=candidate["candidate"],
            min_delta=args.min_delta,
        )
        emitted.append(summary)
    return {
        "ok": True,
        "dry_run": args.dry_run,
        "runtime_config": runtime_config,
        "discovery": discovery,
        "candidate_count": len(candidates),
        "emitted": emitted,
    }


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "results_general_demo.json"
        source.write_text(
            canonical_json(
                {
                    "schema": "QA_GENERAL_WORKER_SELFTEST.v1",
                    "moduli_results": {
                        "case_a": {
                            "controls": {
                                "baseline": {"valid_rate_mean": 0.1},
                                "candidate": {"valid_rate_mean": 0.9},
                            }
                        },
                        "case_b": {
                            "controls": {
                                "baseline": {"valid_rate_mean": 0.4},
                                "candidate": {"valid_rate_mean": 0.4},
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        neural_source = root / "results_sinqa_neural_demo.json"
        neural_source.write_text(
            canonical_json(
                {
                    "schema": NEURAL_ADAPTER_SCHEMA,
                    "moduli_results": {
                        "case_a": {
                            "controls": {
                                "baseline_majority": {"valid_rate_mean": 0.9},
                                "neural_adapter_rank12": {"valid_rate_mean": 0.1},
                            }
                        },
                        "case_b": {
                            "controls": {
                                "baseline_majority": {"valid_rate_mean": 0.8},
                                "neural_adapter_rank12": {"valid_rate_mean": 0.2},
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            source_glob=str(root / "results_*.json"),
            out_dir=root / "out",
            ledger=root / "ledger.jsonl",
            runtime_config=root / "missing_runtime_config.json",
            min_delta=0.0,
            max_emits=10,
            archive_dir=root / "archive",
            dry_run=False,
        )
        first = run_worker(args)
        second = run_worker(args)
        replays = sorted((root / "out").glob("*correction_replay*.json"))
        ok = (
            first["ok"] is True
            and len(first["emitted"]) == 1
            and first["emitted"][0]["fixed"] == 1
            and first["emitted"][0]["harmed"] == 0
            and len(second["emitted"]) == 0
            and len(replays) == 1
            and not any(
                item.get("baseline") == "neural_adapter_rank12" and item.get("candidate") == "baseline_majority"
                for item in first["emitted"]
            )
        )
        return {"ok": ok, "first": first, "second": second}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glob", default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-emits", type=int, default=1)
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
