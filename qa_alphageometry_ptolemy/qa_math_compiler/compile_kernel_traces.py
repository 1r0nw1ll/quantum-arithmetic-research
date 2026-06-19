#!/usr/bin/env python3
"""Compile Lean kernel executions into deterministic Family [31] trace artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "kernel_trace_manifest.json"
TOOLCHAINS_PATH = ROOT / "toolchains.json"
INDEX_PATH = ROOT / "kernel_trace_index.json"
TRACE_DOMAIN = "qa.math_compiler.kernel_trace.v1"
EXECUTION_DOMAIN = "qa.math_compiler.kernel_execution.v1"
INDEX_DOMAIN = "qa.math_compiler.kernel_trace_index.v1"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def domain_hash(domain: str, value: object) -> str:
    return sha256_bytes(domain.encode("utf-8") + b"\x00" + canonical_bytes(value))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def normalize_output(text: str) -> str:
    normalized = text.replace(str(ROOT.parent.parent.resolve()) + os.sep, "")
    normalized = normalized.replace(str(ROOT.resolve()) + os.sep, "qa_math_compiler/")
    return "\n".join(line.rstrip() for line in normalized.splitlines()).strip()


def lean_executable() -> str:
    executable = shutil.which("lean")
    if executable is None:
        candidate = Path.home() / ".elan" / "bin" / "lean"
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError("Lean executable not found")
    return executable


def lean_version(executable: str) -> str:
    proc = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return normalize_output(proc.stdout or proc.stderr).splitlines()[0]


def run_lean(
    executable: str,
    source_path: Path,
    source_hash: str,
    toolchain_id: str,
) -> Dict[str, Any]:
    proc = subprocess.run(
        [executable, str(source_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = {
        "source_sha256": source_hash,
        "toolchain_id": toolchain_id,
        "returncode": proc.returncode,
        "stdout": normalize_output(proc.stdout),
        "stderr": normalize_output(proc.stderr),
    }
    receipt["execution_sha256"] = domain_hash(EXECUTION_DOMAIN, receipt)
    return receipt


def toolchain_hashes(
    toolchains: Dict[str, Any],
    toolchain_id: str,
) -> Tuple[str, str]:
    manifest_hash = sha256_bytes(canonical_bytes(toolchains))
    execution_toolchain_hash = domain_hash(
        "qa.math_compiler.lean_toolchain.v1",
        {
            "manifest_sha256": manifest_hash,
            "toolchain_id": toolchain_id,
        },
    )
    return manifest_hash, execution_toolchain_hash


def trace_artifact(
    case: Dict[str, Any],
    source_hash: str,
    task_hash: str,
    first: Dict[str, Any],
    toolchain_id: str,
    manifest_hash: str,
) -> Dict[str, Any]:
    expected_status = case["expected_status"]
    success = first["returncode"] == 0
    actual_status = "SUCCESS" if success else "FAIL"
    result: Dict[str, Any] = {"status": actual_status}
    if success:
        result["witness_hash"] = first["execution_sha256"]
    else:
        result["fail_type"] = case["fail_type"]
        result["invariant_diff"] = {
            "diagnostic_sha256": sha256_bytes(
                (first["stdout"] + "\n" + first["stderr"]).encode("utf-8")
            ),
            "returncode": first["returncode"],
        }
    trace_body = {
        "case_id": case["case_id"],
        "expected_status": expected_status,
        "source_sha256": source_hash,
        "task_sha256": task_hash,
        "execution_sha256": first["execution_sha256"],
        "toolchain_id": toolchain_id,
    }
    trace_id = domain_hash(TRACE_DOMAIN, trace_body)
    return {
        "schema_id": "QA_MATH_COMPILER_TRACE_SCHEMA.v1",
        "trace_id": trace_id,
        "created_utc": "2026-06-19T00:00:00Z",
        "agent_id": "qa-kernel-trace-compiler",
        "source_layer": "human",
        "target_layer": "formal",
        "generator": "lean_kernel_execution",
        "input_hash": task_hash,
        "output_hash": source_hash,
        "toolchain_id": toolchain_id,
        "result": result,
        "merkle_parent": manifest_hash,
        "invariant_diff": {
            "case_id": case["case_id"],
            "expected_status": expected_status,
            "kernel_derived": True,
            "proof_method": case.get("proof_method"),
            "source_sha256": source_hash,
            "execution_sha256": first["execution_sha256"],
        },
    }


def replay_artifact(
    case: Dict[str, Any],
    trace: Dict[str, Any],
    first: Dict[str, Any],
    second: Dict[str, Any],
    version: str,
    manifest_hash: str,
    execution_toolchain_hash: str,
) -> Dict[str, Any]:
    success = first["returncode"] == 0
    replay_status = "SUCCESS" if success else "FAIL"
    replay_successes = 1 if success else 0
    return {
        "schema_id": "QA_MATH_COMPILER_REPLAY_BUNDLE_SCHEMA.v1",
        "bundle_id": f"KERNEL_{case['case_id'].upper()}_REPLAY",
        "created_utc": "2026-06-19T00:00:01Z",
        "toolchain": {
            "lean_version": version,
            "lake_lock_hash": manifest_hash,
            "toolchain_hash": execution_toolchain_hash,
        },
        "benchmark": {
            "trace_count": 1,
            "min_replay_rate": 1.0 if success else 0.0,
        },
        "traces": [
            {
                "trace_id": trace["trace_id"],
                "seed": 0,
                "trace_hash": first["execution_sha256"],
                "replay_hash": second["execution_sha256"],
                "result_status": trace["result"]["status"],
                "replay_status": replay_status,
            }
        ],
        "metrics": {
            "deterministic_replays": 1,
            "total_replays": 1,
            "replay_successes": replay_successes,
            "replay_rate": float(replay_successes),
            "infra_flake_count": 0,
        },
        "invariant_diff": {
            "case_id": case["case_id"],
            "expected_status": case["expected_status"],
            "kernel_derived": True,
        },
    }


def compile_case(
    case: Dict[str, Any],
    executable: str,
    pinned_version: str,
    toolchain_id: str,
    manifest_hash: str,
    execution_toolchain_hash: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    source_path = ROOT / case["source"]
    source_hash = sha256_bytes(source_path.read_bytes())
    artifact_dir = ROOT / case["artifact_dir"]
    task_path = artifact_dir / "task.json"
    task_hash = (
        sha256_bytes(canonical_bytes(load_json(task_path)))
        if task_path.exists()
        else source_hash
    )
    first = run_lean(executable, source_path, source_hash, toolchain_id)
    second = run_lean(executable, source_path, source_hash, toolchain_id)
    actual_status = "SUCCESS" if first["returncode"] == 0 else "FAIL"
    if actual_status != case["expected_status"]:
        raise RuntimeError(
            f"{case['case_id']}: expected {case['expected_status']}, got {actual_status}"
        )
    diagnostic = first["stdout"] + "\n" + first["stderr"]
    if actual_status == "FAIL" and re.search(case["diagnostic_pattern"], diagnostic) is None:
        raise RuntimeError(
            f"{case['case_id']}: diagnostic did not match {case['diagnostic_pattern']!r}"
        )
    if first["execution_sha256"] != second["execution_sha256"]:
        raise RuntimeError(f"{case['case_id']}: nondeterministic Lean execution receipt")
    trace = trace_artifact(
        case,
        source_hash,
        task_hash,
        first,
        toolchain_id,
        manifest_hash,
    )
    replay = replay_artifact(
        case,
        trace,
        first,
        second,
        pinned_version,
        manifest_hash,
        execution_toolchain_hash,
    )
    index_row = {
        "case_id": case["case_id"],
        "expected_status": case["expected_status"],
        "source": case["source"],
        "source_sha256": source_hash,
        "artifact_dir": case["artifact_dir"],
        "trace_id": trace["trace_id"],
        "execution_sha256": first["execution_sha256"],
        "diagnostic_sha256": (
            sha256_bytes(diagnostic.encode("utf-8"))
            if actual_status == "FAIL"
            else None
        ),
        "fail_type": case.get("fail_type"),
    }
    return trace, replay, index_row


def update_linked_artifacts(artifact_dir: Path, trace: Dict[str, Any]) -> None:
    pair_path = artifact_dir / "pair.json"
    if pair_path.exists():
        pair = load_json(pair_path)
        pair["trace_ref"] = {
            "trace_id": trace["trace_id"],
            "result_status": trace["result"]["status"],
            "replay_status": "SUCCESS",
        }
        write_json(pair_path, pair)
    status_path = artifact_dir / "status.json"
    if status_path.exists():
        status = load_json(status_path)
        status["kernel_derived"] = True
        status["toolchain_id"] = trace["toolchain_id"]
        status["trace_id"] = trace["trace_id"]
        write_json(status_path, status)


def compile_all(write: bool) -> Dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    toolchains = load_json(TOOLCHAINS_PATH)
    executable = lean_executable()
    version = lean_version(executable)
    lean_cfg = toolchains["assistants"]["lean4"]
    if re.search(lean_cfg["version_pattern"], version) is None:
        raise RuntimeError(f"Lean version mismatch: {version}")
    manifest_hash, execution_toolchain_hash = toolchain_hashes(
        toolchains,
        manifest["toolchain_id"],
    )
    rows: List[Dict[str, Any]] = []
    generated: Dict[str, Dict[str, Any]] = {}
    for case in manifest["cases"]:
        trace, replay, row = compile_case(
            case,
            executable,
            str(lean_cfg["version"]),
            manifest["toolchain_id"],
            manifest_hash,
            execution_toolchain_hash,
        )
        artifact_dir = ROOT / case["artifact_dir"]
        generated[str(artifact_dir / "trace.json")] = trace
        generated[str(artifact_dir / "replay.json")] = replay
        rows.append(row)
        if write:
            write_json(artifact_dir / "trace.json", trace)
            write_json(artifact_dir / "replay.json", replay)
            update_linked_artifacts(artifact_dir, trace)
    index_body: Dict[str, Any] = {
        "schema_id": "QA_MATH_COMPILER_KERNEL_TRACE_INDEX.v1",
        "toolchain_id": manifest["toolchain_id"],
        "toolchain_manifest_sha256": manifest_hash,
        "case_count": len(rows),
        "success_count": sum(row["expected_status"] == "SUCCESS" for row in rows),
        "failure_count": sum(row["expected_status"] == "FAIL" for row in rows),
        "cases": rows,
    }
    index_body["index_sha256"] = domain_hash(INDEX_DOMAIN, index_body)
    if write:
        write_json(INDEX_PATH, index_body)
    return {"generated": generated, "index": index_body}


def compare_live() -> List[str]:
    compiled = compile_all(write=False)
    errors: List[str] = []
    for path_text, expected in compiled["generated"].items():
        path = Path(path_text)
        if not path.exists() or load_json(path) != expected:
            errors.append(f"ARTIFACT_MISMATCH:{path.relative_to(ROOT)}")
    if not INDEX_PATH.exists() or load_json(INDEX_PATH) != compiled["index"]:
        errors.append("INDEX_MISMATCH")
    return errors


def check_artifacts() -> List[str]:
    from qa_math_compiler_validator import validate_replay_bundle, validate_trace

    errors: List[str] = []
    manifest = load_json(MANIFEST_PATH)
    toolchains = load_json(TOOLCHAINS_PATH)
    index = load_json(INDEX_PATH)
    body = dict(index)
    supplied_hash = body.pop("index_sha256", None)
    if supplied_hash != domain_hash(INDEX_DOMAIN, body):
        errors.append("INDEX_HASH_MISMATCH")
    if index.get("case_count") != len(manifest["cases"]):
        errors.append("INDEX_CASE_COUNT_MISMATCH")
    if index.get("toolchain_id") != manifest.get("toolchain_id"):
        errors.append("INDEX_TOOLCHAIN_MISMATCH")
    if index.get("toolchain_manifest_sha256") != sha256_bytes(canonical_bytes(toolchains)):
        errors.append("INDEX_TOOLCHAIN_HASH_MISMATCH")
    success_count = sum(
        case.get("expected_status") == "SUCCESS" for case in manifest["cases"]
    )
    failure_count = sum(
        case.get("expected_status") == "FAIL" for case in manifest["cases"]
    )
    if index.get("success_count") != success_count:
        errors.append("INDEX_SUCCESS_COUNT_MISMATCH")
    if index.get("failure_count") != failure_count:
        errors.append("INDEX_FAILURE_COUNT_MISMATCH")
    index_rows = {
        row.get("case_id"): row
        for row in index.get("cases", [])
        if isinstance(row, dict)
    }
    if len(index_rows) != len(manifest["cases"]):
        errors.append("INDEX_CASE_SET_MISMATCH")
    for case in manifest["cases"]:
        artifact_dir = ROOT / case["artifact_dir"]
        trace_path = artifact_dir / "trace.json"
        replay_path = artifact_dir / "replay.json"
        if not trace_path.exists() or not replay_path.exists():
            errors.append(f"MISSING_ARTIFACT:{case['case_id']}")
            continue
        trace = load_json(trace_path)
        replay = load_json(replay_path)
        row = index_rows.get(case["case_id"])
        if row is None:
            errors.append(f"INDEX_ROW_MISSING:{case['case_id']}")
            continue
        trace_result = validate_trace(trace)
        replay_result = validate_replay_bundle(replay)
        if not trace_result.ok:
            errors.append(f"TRACE_INVALID:{case['case_id']}:{trace_result.fail_type}")
        if not replay_result.ok:
            errors.append(f"REPLAY_INVALID:{case['case_id']}:{replay_result.fail_type}")
        if trace["result"]["status"] != case["expected_status"]:
            errors.append(f"STATUS_MISMATCH:{case['case_id']}")
        source_hash = sha256_bytes((ROOT / case["source"]).read_bytes())
        if row.get("source_sha256") != source_hash:
            errors.append(f"SOURCE_HASH_MISMATCH:{case['case_id']}")
        if row.get("trace_id") != trace.get("trace_id"):
            errors.append(f"INDEX_TRACE_ID_MISMATCH:{case['case_id']}")
        replay_row = replay.get("traces", [{}])[0]
        if row.get("execution_sha256") != replay_row.get("trace_hash"):
            errors.append(f"INDEX_EXECUTION_HASH_MISMATCH:{case['case_id']}")
        if case["expected_status"] == "FAIL":
            if trace["result"].get("fail_type") != case["fail_type"]:
                errors.append(f"FAIL_TYPE_MISMATCH:{case['case_id']}")
            diagnostic_hash = trace["result"].get("invariant_diff", {}).get(
                "diagnostic_sha256"
            )
            if row.get("diagnostic_sha256") != diagnostic_hash:
                errors.append(f"DIAGNOSTIC_HASH_MISMATCH:{case['case_id']}")
        elif row.get("diagnostic_sha256") is not None:
            errors.append(f"UNEXPECTED_DIAGNOSTIC_HASH:{case['case_id']}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_artifacts:
        errors = check_artifacts()
    elif args.write:
        compile_all(write=True)
        errors = check_artifacts()
    else:
        errors = compare_live()
        errors.extend(check_artifacts())
    print(
        canonical_bytes(
            {
                "ok": not errors,
                "errors": errors,
                "index": str(INDEX_PATH),
            }
        ).decode("utf-8")
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
