#!/usr/bin/env python3
"""Replay-backed lemma-mining and proof-compression evaluation for Family [31]."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from compile_kernel_traces import (
    canonical_bytes,
    domain_hash,
    lean_executable,
    load_json,
    parse_tactic_steps,
    sha256_bytes,
    write_json,
)
from qa_math_compiler_validator import validate_lemma_mining


ROOT = Path(__file__).resolve().parent
MINING = ROOT / "lemma_mining"
LIBRARY = MINING / "QAMinedLemmas.lean"
EVALUATION_PATH = MINING / "evaluation.v1.json"
LEGACY_PACK_PATH = MINING / "lemma_pack.v1.json"
EVALUATION_DOMAIN = "qa.math_compiler.lemma_mining_evaluation.v1"
COMPRESSED_TRACE_DOMAIN = "qa.math_compiler.compressed_elaboration.v1"

CASES = [
    ("ex19_imp_trans", "qaCompose", "discovery"),
    ("ex33_contraposition", "qaCompose", "held_out"),
    ("ex36_list_append_nil", "qaListInduction", "discovery"),
    ("ex38_list_append_assoc", "qaListInduction", "held_out"),
    ("ex40_list_map_id", "qaListInduction", "held_out"),
    ("ex41_list_map_comp", "qaListInduction", "held_out"),
]

CANDIDATES = [
    {
        "lemma_id": "qaCompose",
        "statement": "(β → γ) → (α → β) → α → γ",
        "usage_count": 2,
        "compression_gain_steps": 6,
        "proof_status": "FOUND",
        "dependency_imports": ["Init.Prelude"],
        "evidence_pattern": "three introductions followed by nested function application",
    },
    {
        "lemma_id": "qaListInduction",
        "statement": "P [] → (∀ x xs, P xs → P (x :: xs)) → ∀ xs, P xs",
        "usage_count": 4,
        "compression_gain_steps": 4,
        "proof_status": "FOUND",
        "dependency_imports": ["Init.Data.List.Basic"],
        "evidence_pattern": "list induction with a definitional base and congruent cons step",
    },
    {
        "lemma_id": "repeated_intro_token",
        "statement": "Repeated local tactic token `intro h`",
        "usage_count": 22,
        "compression_gain_steps": 0,
        "proof_status": "NEEDS_PROOF",
        "dependency_imports": ["Init.Prelude"],
        "evidence_pattern": "frequent syntax token without a stable typed theorem boundary",
    },
]


def run_lean(executable: str, source: Path, env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.run(
        [executable, str(source)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout_sha256": sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8")),
    }


def compressed_elaboration(
    executable: str,
    source: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    source_text = source.read_text(encoding="utf-8")
    source_lines = source_text.splitlines()
    insert_at = 0
    while insert_at < len(source_lines) and (
        source_lines[insert_at].startswith("import ") or not source_lines[insert_at].strip()
    ):
        insert_at += 1
    instrumented_lines = list(source_lines)
    instrumented_lines.insert(insert_at, "set_option trace.Elab.info true")
    instrumented = "\n".join(instrumented_lines) + "\n"
    proc = subprocess.run(
        [executable, "--json", "--stdin"],
        cwd=str(ROOT),
        env=env,
        input=instrumented,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    messages = []
    chunks = []
    for raw_line in proc.stdout.splitlines():
        if not raw_line.strip():
            continue
        message = json.loads(raw_line)
        message["fileName"] = "<stdin>"
        messages.append(message)
        if message.get("kind") == "trace" and "[Elab.info]" in message.get("data", ""):
            chunks.append(message["data"])
    info_tree = "\n".join(chunks)
    steps = parse_tactic_steps(info_tree, source_text)
    body = {
        "returncode": proc.returncode,
        "source_sha256": sha256_bytes(source.read_bytes()),
        "message_count": len(messages),
        "proof_step_count": len(steps),
        "proof_steps_sha256": sha256_bytes(canonical_bytes(steps)),
        "info_tree_sha256": sha256_bytes(info_tree.encode("utf-8")),
        "messages_sha256": sha256_bytes(canonical_bytes(messages)),
    }
    body["compressed_trace_sha256"] = domain_hash(COMPRESSED_TRACE_DOMAIN, body)
    return body


def build() -> tuple[Dict[str, Any], Dict[str, Any]]:
    executable = lean_executable()
    with tempfile.TemporaryDirectory(prefix="qa-lemma-mining-") as temp_text:
        temp = Path(temp_text)
        library_olean = temp / "QAMinedLemmas.olean"
        library_proc = subprocess.run(
            [executable, str(LIBRARY), "-o", str(library_olean)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if library_proc.returncode != 0:
            raise RuntimeError(f"mined library failed: {library_proc.stderr}")
        env = dict(os.environ)
        env["LEAN_PATH"] = str(temp)

        rows: List[Dict[str, Any]] = []
        for example_id, lemma_id, split in CASES:
            baseline_path = (
                ROOT / "demo_pack_v1" / "examples" / example_id / "elaboration.json"
            )
            compressed_path = MINING / "compressed" / f"{example_id}.lean"
            baseline = load_json(baseline_path)
            first_exec = run_lean(executable, compressed_path, env)
            second_exec = run_lean(executable, compressed_path, env)
            first_trace = compressed_elaboration(executable, compressed_path, env)
            second_trace = compressed_elaboration(executable, compressed_path, env)
            if first_exec != second_exec or first_trace != second_trace:
                raise RuntimeError(f"{example_id}: compressed replay is nondeterministic")
            if first_exec["returncode"] != 0 or first_trace["returncode"] != 0:
                raise RuntimeError(f"{example_id}: compressed proof failed")
            baseline_steps = max(1, int(baseline["proof_step_count"]))
            compressed_steps = max(1, int(first_trace["proof_step_count"]))
            if compressed_steps > baseline_steps:
                raise RuntimeError(f"{example_id}: compressed proof is longer")
            reduction = (baseline_steps - compressed_steps) / baseline_steps
            rows.append(
                {
                    "example_id": example_id,
                    "evaluation_split": split,
                    "lemma_id": lemma_id,
                    "baseline_elaboration_sha256": baseline[
                        "elaboration_trace_sha256"
                    ],
                    "compressed_source": str(compressed_path.relative_to(ROOT)),
                    "compressed_source_sha256": sha256_bytes(
                        compressed_path.read_bytes()
                    ),
                    "baseline_steps": baseline_steps,
                    "compressed_steps": compressed_steps,
                    "reduction": reduction,
                    "compressed_trace_sha256": first_trace[
                        "compressed_trace_sha256"
                    ],
                    "deterministic_replays": 2,
                    "kernel_status": "SUCCESS",
                }
            )

    reductions = [row["reduction"] for row in rows]
    held_out = [row for row in rows if row["evaluation_split"] == "held_out"]
    evaluation: Dict[str, Any] = {
        "schema_id": "QA_LEMMA_MINING_EVALUATION.v1",
        "evaluation_id": "LEAN50_LEMMA_MINING_EVAL_V1",
        "created_utc": "2026-06-20T00:00:00Z",
        "toolchain_id": "lean4.31.0",
        "source_corpus_sha256": load_json(
            ROOT / "demo_pack_v1" / "corpus.json"
        )["corpus_sha256"],
        "library_source": str(LIBRARY.relative_to(ROOT)),
        "library_source_sha256": sha256_bytes(LIBRARY.read_bytes()),
        "candidate_count": len(CANDIDATES),
        "certified_candidate_count": sum(
            candidate["proof_status"] == "FOUND" for candidate in CANDIDATES
        ),
        "candidates": CANDIDATES,
        "benchmark_rows": rows,
        "failure_records": [
            {
                "lemma_id": "repeated_intro_token",
                "fail_type": "NON_GENERALIZABLE_PATTERN",
                "invariant_diff": {
                    "reason": "token frequency does not define a typed reusable theorem",
                    "observed_usage_count": 22,
                },
            }
        ],
        "metrics": {
            "benchmark_count": len(rows),
            "held_out_count": len(held_out),
            "replay_success_rate": 1.0,
            "held_out_replay_success_rate": 1.0,
            "median_reduction": float(statistics.median(reductions)),
            "held_out_median_reduction": float(
                statistics.median(row["reduction"] for row in held_out)
            ),
            "total_baseline_steps": sum(row["baseline_steps"] for row in rows),
            "total_compressed_steps": sum(row["compressed_steps"] for row in rows),
            "total_saved_steps": sum(
                row["baseline_steps"] - row["compressed_steps"] for row in rows
            ),
            "useful_candidate_rate": 2.0 / 3.0,
        },
        "invariant_diff": {
            "length_metric": "max(1, atomic tactic nodes from trace.Elab.info)",
            "candidate_discovery": "repeated typed proof-step skeletons",
            "held_out_policy": "first occurrence discovers; later occurrences evaluate",
            "no_external_dependencies": True,
        },
    }
    evaluation["evaluation_sha256"] = domain_hash(EVALUATION_DOMAIN, evaluation)

    legacy_candidates = [
        {
            key: candidate[key]
            for key in [
                "lemma_id",
                "statement",
                "usage_count",
                "compression_gain_steps",
                "proof_status",
                "dependency_imports",
            ]
        }
        for candidate in CANDIDATES
    ]
    legacy = {
        "schema_id": "QA_LEMMA_MINING_SCHEMA.v1",
        "mining_id": "LEAN50_REPLAY_BACKED_LEMMA_PACK_V1",
        "created_utc": "2026-06-20T00:00:00Z",
        "source_traces": [
            row["baseline_elaboration_sha256"] for row in rows
        ],
        "baseline_trace_lengths": [row["baseline_steps"] for row in rows],
        "compressed_trace_lengths": [row["compressed_steps"] for row in rows],
        "target_median_reduction": 0.5,
        "lemma_candidates": legacy_candidates,
        "failure_records": evaluation["failure_records"],
        "metrics": {
            "median_reduction": evaluation["metrics"]["median_reduction"],
            "total_candidates": len(legacy_candidates),
        },
        "invariant_diff": {
            "evaluation_ref": evaluation["evaluation_sha256"],
            "replay_backed": True,
            "held_out_count": len(held_out),
        },
    }
    result = validate_lemma_mining(legacy)
    if not result.ok:
        raise RuntimeError(
            f"legacy lemma pack invalid: {result.fail_type} {result.invariant_diff}"
        )
    return evaluation, legacy


def certificate_check() -> Dict[str, Any]:
    evaluation = load_json(EVALUATION_PATH)
    legacy = load_json(LEGACY_PACK_PATH)
    errors = []
    body = dict(evaluation)
    supplied = body.pop("evaluation_sha256", None)
    if supplied != domain_hash(EVALUATION_DOMAIN, body):
        errors.append("EVALUATION_HASH_MISMATCH")
    if evaluation.get("library_source_sha256") != sha256_bytes(LIBRARY.read_bytes()):
        errors.append("LIBRARY_SOURCE_HASH_MISMATCH")
    corpus = load_json(ROOT / "demo_pack_v1" / "corpus.json")
    if evaluation.get("source_corpus_sha256") != corpus.get("corpus_sha256"):
        errors.append("SOURCE_CORPUS_HASH_MISMATCH")
    rows = evaluation.get("benchmark_rows", [])
    reductions = []
    for row in rows:
        source = ROOT / row["compressed_source"]
        if not source.is_file():
            errors.append(f"COMPRESSED_SOURCE_MISSING:{row['example_id']}")
            continue
        if row.get("compressed_source_sha256") != sha256_bytes(source.read_bytes()):
            errors.append(f"COMPRESSED_SOURCE_HASH_MISMATCH:{row['example_id']}")
        baseline = row.get("baseline_steps")
        compressed = row.get("compressed_steps")
        if not isinstance(baseline, int) or not isinstance(compressed, int):
            errors.append(f"STEP_COUNT_INVALID:{row['example_id']}")
            continue
        reduction = (baseline - compressed) / baseline
        reductions.append(reduction)
        if abs(float(row.get("reduction", -1)) - reduction) > 1e-12:
            errors.append(f"REDUCTION_MISMATCH:{row['example_id']}")
    metrics = evaluation.get("metrics", {})
    if reductions and abs(
        float(metrics.get("median_reduction", -1))
        - float(statistics.median(reductions))
    ) > 1e-12:
        errors.append("MEDIAN_REDUCTION_MISMATCH")
    if metrics.get("total_saved_steps") != sum(
        row["baseline_steps"] - row["compressed_steps"] for row in rows
    ):
        errors.append("TOTAL_SAVED_STEPS_MISMATCH")
    legacy_result = validate_lemma_mining(legacy)
    if not legacy_result.ok:
        errors.append(f"LEGACY_PACK_INVALID:{legacy_result.fail_type}")
    if legacy.get("invariant_diff", {}).get("evaluation_ref") != supplied:
        errors.append("LEGACY_EVALUATION_REF_MISMATCH")
    return {
        "ok": not errors,
        "errors": errors,
        "evaluation_sha256": supplied,
        "benchmark_count": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "check", "certificate-check"])
    args = parser.parse_args()
    if args.mode == "certificate-check":
        result = certificate_check()
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0 if result["ok"] else 1
    evaluation, legacy = build()
    if args.mode == "write":
        write_json(EVALUATION_PATH, evaluation)
        write_json(LEGACY_PACK_PATH, legacy)
    else:
        if load_json(EVALUATION_PATH) != evaluation:
            raise RuntimeError("lemma mining evaluation rebuild mismatch")
        if load_json(LEGACY_PACK_PATH) != legacy:
            raise RuntimeError("lemma mining legacy pack rebuild mismatch")
    print(
        json.dumps(
            {
                "ok": True,
                "evaluation_sha256": evaluation["evaluation_sha256"],
                **evaluation["metrics"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
