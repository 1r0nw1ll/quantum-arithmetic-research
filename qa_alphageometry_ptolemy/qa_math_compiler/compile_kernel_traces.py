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
SEMANTIC_CERTIFICATE_PATH = ROOT / "semantic_replay_certificate.v1.json"
TRACE_DOMAIN = "qa.math_compiler.kernel_trace.v1"
EXECUTION_DOMAIN = "qa.math_compiler.kernel_execution.v1"
INDEX_DOMAIN = "qa.math_compiler.kernel_trace_index.v1"
SEMANTIC_CASE_DOMAIN = "qa.math_compiler.semantic_replay_case.v1"
SEMANTIC_CERTIFICATE_DOMAIN = "qa.math_compiler.semantic_replay_certificate.v1"
ELABORATION_TRACE_DOMAIN = "qa.math_compiler.lean_elaboration_trace.v1"
PROOF_STEP_DOMAIN = "qa.math_compiler.lean_proof_step.v1"


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
    normalized = normalized.replace(str(Path.home().resolve()) + os.sep, "$HOME/")
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


def execution_command(
    case: Dict[str, Any],
    executable: str,
    *lean_args: str,
) -> Tuple[List[str], Path, str | None]:
    lake_project = case.get("lake_project")
    if lake_project is None:
        return [executable, *lean_args], ROOT, None
    project_dir = ROOT / lake_project
    lock_path = project_dir / "lake-manifest.json"
    if not project_dir.is_dir():
        raise RuntimeError(f"{case['case_id']}: Lake project missing: {lake_project}")
    if not lock_path.is_file():
        raise RuntimeError(
            f"{case['case_id']}: Lake lockfile missing: "
            f"{lock_path.relative_to(ROOT)}"
        )
    lake = shutil.which("lake")
    if lake is None:
        candidate = Path.home() / ".elan" / "bin" / "lake"
        if not candidate.is_file():
            raise RuntimeError("Lake executable not found")
        lake = str(candidate)
    lock_hash = sha256_bytes(lock_path.read_bytes())
    return [lake, "env", "lean", *lean_args], project_dir, lock_hash


def run_lean(
    case: Dict[str, Any],
    executable: str,
    source_path: Path,
    source_hash: str,
    toolchain_id: str,
) -> Dict[str, Any]:
    command, cwd, dependency_lock_sha256 = execution_command(
        case,
        executable,
        str(source_path),
    )
    proc = subprocess.run(
        command,
        cwd=str(cwd),
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
    if dependency_lock_sha256 is not None:
        receipt["dependency_lock_sha256"] = dependency_lock_sha256
    receipt["execution_sha256"] = domain_hash(EXECUTION_DOMAIN, receipt)
    return receipt


def source_fragment(
    source: str,
    start_line: int,
    start_column: int,
    end_line: int,
    end_column: int,
) -> str:
    lines = source.splitlines()
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        return ""
    selected = lines[start_line - 1 : end_line]
    if not selected:
        return ""
    first = selected[0][start_column:]
    if len(selected) == 1:
        return first[: max(0, end_column - start_column)]
    selected[0] = first
    selected[-1] = selected[-1][:end_column]
    return "\n".join(selected).strip()


def parse_tactic_steps(info_text: str, source: str) -> List[Dict[str, Any]]:
    lines = info_text.splitlines()
    header = re.compile(
        r"^(?P<indent>\s*)• \[Tactic\] @ "
        r"⟨(?P<sl>\d+), (?P<sc>\d+)⟩-⟨(?P<el>\d+), (?P<ec>\d+)⟩"
        r"(?: @ (?P<elaborator>\S+))?"
    )
    nodes: List[Dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        match = header.match(line)
        if match is None:
            continue
        nodes.append(
            {
                "line_index": line_index,
                "indent": len(match.group("indent")),
                "start_line": int(match.group("sl")) - 1,
                "start_column": int(match.group("sc")),
                "end_line": int(match.group("el")) - 1,
                "end_column": int(match.group("ec")),
                "elaborator": match.group("elaborator") or "anonymous",
            }
        )
    for index, node in enumerate(nodes):
        boundary = len(lines)
        has_child_tactic = False
        for later in nodes[index + 1 :]:
            if later["indent"] <= node["indent"]:
                boundary = later["line_index"]
                break
            has_child_tactic = True
        node["has_child_tactic"] = has_child_tactic
        block = lines[node["line_index"] + 1 : boundary]
        before_lines: List[str] = []
        after_lines: List[str] = []
        target = None
        for line in block:
            stripped = line.strip()
            if stripped.startswith("• ["):
                break
            if stripped == "before" or stripped.startswith("before "):
                target = before_lines
                target.append(stripped[len("before") :].strip())
            elif stripped == "after" or stripped.startswith("after "):
                target = after_lines
                target.append(stripped[len("after") :].strip())
            elif target is not None:
                target.append(stripped)
        node["goals_before"] = "\n".join(before_lines).strip()
        node["goals_after"] = "\n".join(after_lines).strip()

    steps: List[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        if node["has_child_tactic"]:
            continue
        key = (
            node["start_line"],
            node["start_column"],
            node["end_line"],
            node["end_column"],
            node["elaborator"],
            node["goals_before"],
            node["goals_after"],
        )
        if key in seen:
            continue
        seen.add(key)
        step_body = {
            "source_range": {
                "start_line": node["start_line"],
                "start_column": node["start_column"],
                "end_line": node["end_line"],
                "end_column": node["end_column"],
            },
            "source_text": source_fragment(
                source,
                node["start_line"],
                node["start_column"],
                node["end_line"],
                node["end_column"],
            ),
            "elaborator": node["elaborator"],
            "goals_before": node["goals_before"],
            "goals_after": node["goals_after"],
        }
        steps.append(
            {
                "step_index": len(steps),
                **step_body,
                "step_sha256": domain_hash(PROOF_STEP_DOMAIN, step_body),
            }
        )
    return steps


def run_elaboration_trace(
    case: Dict[str, Any],
    executable: str,
    source_path: Path,
    source_hash: str,
    toolchain_id: str,
) -> Dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)
    insert_at = 0
    while insert_at < len(source_lines):
        stripped = source_lines[insert_at].strip()
        if stripped.startswith("import ") or (insert_at > 0 and not stripped):
            insert_at += 1
            continue
        break
    source_lines.insert(insert_at, "set_option trace.Elab.info true\n")
    instrumented = "".join(source_lines)
    command, cwd, dependency_lock_sha256 = execution_command(
        case,
        executable,
        "--json",
        "--stdin",
    )
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        input=instrumented,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    messages = []
    info_chunks = []
    for raw_line in proc.stdout.splitlines():
        if not raw_line.strip():
            continue
        message = json.loads(raw_line)
        message["fileName"] = "<stdin>"
        messages.append(message)
        if message.get("kind") == "trace" and "[Elab.info]" in message.get("data", ""):
            info_chunks.append(message["data"])
    info_text = "\n".join(info_chunks)
    node_counts = {
        kind: len(re.findall(rf"\[{re.escape(kind)}(?:\]|\()", info_text))
        for kind in [
            "Command",
            "Term",
            "Tactic",
            "MacroExpansion",
            "Completion",
            "Completion-Id",
            "CustomInfo",
        ]
    }
    steps = parse_tactic_steps(info_text, source)
    body: Dict[str, Any] = {
        "schema_id": "QA_MATH_COMPILER_LEAN_ELABORATION_TRACE.v1",
        "source_sha256": source_hash,
        "toolchain_id": toolchain_id,
        "trace_channel": "trace.Elab.info",
        "wrapper_line_offset": 1,
        "returncode": proc.returncode,
        "message_count": len(messages),
        "node_counts": node_counts,
        "proof_step_count": len(steps),
        "proof_steps": steps,
        "info_tree": info_text,
        "info_tree_sha256": sha256_bytes(info_text.encode("utf-8")),
        "messages_sha256": sha256_bytes(canonical_bytes(messages)),
    }
    if dependency_lock_sha256 is not None:
        body["dependency_lock_sha256"] = dependency_lock_sha256
    body["elaboration_trace_sha256"] = domain_hash(ELABORATION_TRACE_DOMAIN, body)
    return body


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
            **(
                {"upstream_declaration": case["upstream_declaration"]}
                if "upstream_declaration" in case
                else {}
            ),
            **(
                {"dependency_lock_sha256": first["dependency_lock_sha256"]}
                if "dependency_lock_sha256" in first
                else {}
            ),
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
            "lake_lock_hash": first.get("dependency_lock_sha256", manifest_hash),
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
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    source_path = ROOT / case["source"]
    source_hash = sha256_bytes(source_path.read_bytes())
    artifact_dir = ROOT / case["artifact_dir"]
    task_path = artifact_dir / "task.json"
    task_hash = (
        sha256_bytes(canonical_bytes(load_json(task_path)))
        if task_path.exists()
        else source_hash
    )
    first = run_lean(case, executable, source_path, source_hash, toolchain_id)
    second = run_lean(case, executable, source_path, source_hash, toolchain_id)
    elaboration_first = run_elaboration_trace(
        case, executable, source_path, source_hash, toolchain_id
    )
    elaboration_second = run_elaboration_trace(
        case, executable, source_path, source_hash, toolchain_id
    )
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
    if elaboration_first != elaboration_second:
        raise RuntimeError(f"{case['case_id']}: nondeterministic Lean elaboration trace")
    if elaboration_first["returncode"] != first["returncode"]:
        raise RuntimeError(f"{case['case_id']}: elaboration/execution status mismatch")
    trace = trace_artifact(
        case,
        source_hash,
        task_hash,
        first,
        toolchain_id,
        manifest_hash,
    )
    case_toolchain_hash = execution_toolchain_hash
    if "dependency_lock_sha256" in first:
        case_toolchain_hash = domain_hash(
            "qa.math_compiler.lean_toolchain_with_dependencies.v1",
            {
                "base_toolchain_hash": execution_toolchain_hash,
                "dependency_lock_sha256": first["dependency_lock_sha256"],
            },
        )
    replay = replay_artifact(
        case,
        trace,
        first,
        second,
        pinned_version,
        manifest_hash,
        case_toolchain_hash,
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
        "elaboration_trace_sha256": elaboration_first["elaboration_trace_sha256"],
        "proof_step_count": elaboration_first["proof_step_count"],
    }
    trace["invariant_diff"]["elaboration_trace_sha256"] = elaboration_first[
        "elaboration_trace_sha256"
    ]
    trace["invariant_diff"]["proof_step_count"] = elaboration_first["proof_step_count"]
    return trace, replay, elaboration_first, index_row


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
        trace, replay, elaboration, row = compile_case(
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
        generated[str(artifact_dir / "elaboration.json")] = elaboration
        rows.append(row)
        if write:
            write_json(artifact_dir / "trace.json", trace)
            write_json(artifact_dir / "replay.json", replay)
            write_json(artifact_dir / "elaboration.json", elaboration)
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


def semantic_certificate(index: Dict[str, Any]) -> Dict[str, Any]:
    cases = []
    for row in index["cases"]:
        semantic_body = {
            "case_id": row["case_id"],
            "source_sha256": row["source_sha256"],
            "toolchain_id": index["toolchain_id"],
            "observed_status": row["expected_status"],
            "fail_type": row["fail_type"],
        }
        cases.append(
            {
                **semantic_body,
                "semantic_sha256": domain_hash(
                    SEMANTIC_CASE_DOMAIN,
                    semantic_body,
                ),
            }
        )
    body: Dict[str, Any] = {
        "schema_id": "QA_MATH_COMPILER_SEMANTIC_REPLAY_CERTIFICATE.v1",
        "toolchain_id": index["toolchain_id"],
        "case_count": len(cases),
        "success_count": sum(
            case["observed_status"] == "SUCCESS" for case in cases
        ),
        "failure_count": sum(
            case["observed_status"] == "FAIL" for case in cases
        ),
        "cases": cases,
    }
    body["certificate_sha256"] = domain_hash(
        SEMANTIC_CERTIFICATE_DOMAIN,
        body,
    )
    return body


def validate_semantic_certificate(certificate: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if (
        certificate.get("schema_id")
        != "QA_MATH_COMPILER_SEMANTIC_REPLAY_CERTIFICATE.v1"
    ):
        errors.append("SEMANTIC_SCHEMA_ID_MISMATCH")
    cases = certificate.get("cases")
    if not isinstance(cases, list):
        return errors + ["SEMANTIC_CASES_INVALID"]
    if certificate.get("case_count") != len(cases):
        errors.append("SEMANTIC_CASE_COUNT_MISMATCH")
    success_count = sum(
        isinstance(case, dict) and case.get("observed_status") == "SUCCESS"
        for case in cases
    )
    failure_count = sum(
        isinstance(case, dict) and case.get("observed_status") == "FAIL"
        for case in cases
    )
    if certificate.get("success_count") != success_count:
        errors.append("SEMANTIC_SUCCESS_COUNT_MISMATCH")
    if certificate.get("failure_count") != failure_count:
        errors.append("SEMANTIC_FAILURE_COUNT_MISMATCH")
    for case in cases:
        if not isinstance(case, dict):
            errors.append("SEMANTIC_CASE_INVALID")
            continue
        body = {
            "case_id": case.get("case_id"),
            "source_sha256": case.get("source_sha256"),
            "toolchain_id": case.get("toolchain_id"),
            "observed_status": case.get("observed_status"),
            "fail_type": case.get("fail_type"),
        }
        if case.get("semantic_sha256") != domain_hash(
            SEMANTIC_CASE_DOMAIN,
            body,
        ):
            errors.append(f"SEMANTIC_CASE_HASH_MISMATCH:{case.get('case_id')}")
    body = dict(certificate)
    supplied_hash = body.pop("certificate_sha256", None)
    if supplied_hash != domain_hash(SEMANTIC_CERTIFICATE_DOMAIN, body):
        errors.append("SEMANTIC_CERTIFICATE_HASH_MISMATCH")
    return errors


def compare_live() -> List[str]:
    compiled = compile_all(write=False)
    errors: List[str] = []
    for path_text, expected in compiled["generated"].items():
        path = Path(path_text)
        if not path.exists() or load_json(path) != expected:
            errors.append(f"ARTIFACT_MISMATCH:{path.relative_to(ROOT)}")
    if not INDEX_PATH.exists() or load_json(INDEX_PATH) != compiled["index"]:
        errors.append("INDEX_MISMATCH")
    live_semantic = semantic_certificate(compiled["index"])
    if (
        not SEMANTIC_CERTIFICATE_PATH.exists()
        or load_json(SEMANTIC_CERTIFICATE_PATH) != live_semantic
    ):
        errors.append("SEMANTIC_CERTIFICATE_MISMATCH")
    return errors


def check_artifacts() -> List[str]:
    from qa_math_compiler_validator import validate_replay_bundle, validate_trace

    errors: List[str] = []
    manifest = load_json(MANIFEST_PATH)
    toolchains = load_json(TOOLCHAINS_PATH)
    index = load_json(INDEX_PATH)
    if not SEMANTIC_CERTIFICATE_PATH.exists():
        errors.append("SEMANTIC_CERTIFICATE_MISSING")
    else:
        stored_semantic = load_json(SEMANTIC_CERTIFICATE_PATH)
        errors.extend(validate_semantic_certificate(stored_semantic))
        expected_semantic = semantic_certificate(index)
        if stored_semantic != expected_semantic:
            errors.append("SEMANTIC_CERTIFICATE_INDEX_MISMATCH")
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
        elaboration_path = artifact_dir / "elaboration.json"
        if (
            not trace_path.exists()
            or not replay_path.exists()
            or not elaboration_path.exists()
        ):
            errors.append(f"MISSING_ARTIFACT:{case['case_id']}")
            continue
        trace = load_json(trace_path)
        replay = load_json(replay_path)
        elaboration = load_json(elaboration_path)
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
        if (
            row.get("elaboration_trace_sha256")
            != elaboration.get("elaboration_trace_sha256")
        ):
            errors.append(f"INDEX_ELABORATION_HASH_MISMATCH:{case['case_id']}")
        if row.get("proof_step_count") != elaboration.get("proof_step_count"):
            errors.append(f"INDEX_PROOF_STEP_COUNT_MISMATCH:{case['case_id']}")
        if elaboration.get("source_sha256") != source_hash:
            errors.append(f"ELABORATION_SOURCE_HASH_MISMATCH:{case['case_id']}")
        if elaboration.get("info_tree_sha256") != sha256_bytes(
            elaboration.get("info_tree", "").encode("utf-8")
        ):
            errors.append(f"INFO_TREE_HASH_MISMATCH:{case['case_id']}")
        elaboration_body = dict(elaboration)
        elaboration_hash = elaboration_body.pop("elaboration_trace_sha256", None)
        if elaboration_hash != domain_hash(
            ELABORATION_TRACE_DOMAIN, elaboration_body
        ):
            errors.append(f"ELABORATION_HASH_MISMATCH:{case['case_id']}")
        for step_index, step in enumerate(elaboration.get("proof_steps", [])):
            step_body = dict(step)
            step_hash = step_body.pop("step_sha256", None)
            step_body.pop("step_index", None)
            if step.get("step_index") != step_index:
                errors.append(f"PROOF_STEP_ORDER_MISMATCH:{case['case_id']}")
            if step_hash != domain_hash(PROOF_STEP_DOMAIN, step_body):
                errors.append(f"PROOF_STEP_HASH_MISMATCH:{case['case_id']}")
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
    parser.add_argument("--write-semantic-certificate", action="store_true")
    parser.add_argument("--check-semantic-certificate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_semantic_certificate:
        compiled = compile_all(write=False)
        write_json(
            SEMANTIC_CERTIFICATE_PATH,
            semantic_certificate(compiled["index"]),
        )
        errors = validate_semantic_certificate(
            load_json(SEMANTIC_CERTIFICATE_PATH)
        )
    elif args.check_semantic_certificate:
        compiled = compile_all(write=False)
        live = semantic_certificate(compiled["index"])
        stored = load_json(SEMANTIC_CERTIFICATE_PATH)
        errors = validate_semantic_certificate(stored)
        if stored != live:
            errors.append("SEMANTIC_REPLAY_PARITY_MISMATCH")
    elif args.check_artifacts:
        errors = check_artifacts()
    elif args.write:
        compile_all(write=True)
        write_json(
            SEMANTIC_CERTIFICATE_PATH,
            semantic_certificate(load_json(INDEX_PATH)),
        )
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
