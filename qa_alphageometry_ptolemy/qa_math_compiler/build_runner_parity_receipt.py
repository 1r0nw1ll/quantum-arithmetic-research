#!/usr/bin/env python3
"""Build and compare deterministic receipts from independent clean runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
RECEIPT_DOMAIN = "qa.math_compiler.independent_runner_receipt.v1"
ELABORATION_SET_DOMAIN = "qa.math_compiler.elaboration_set.v1"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def domain_hash(domain: str, value: object) -> str:
    return sha256_bytes(domain.encode("utf-8") + b"\x00" + canonical_bytes(value))


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return value


def run_text(executable: str, *args: str) -> str:
    proc = subprocess.run(
        [executable, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{executable} {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


def lean_executable() -> str:
    executable = shutil.which("lean")
    if executable is not None:
        return executable
    candidate = Path.home() / ".elan" / "bin" / "lean"
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError("Lean executable not found")


def source_revision() -> str:
    revision = os.environ.get("GITHUB_SHA")
    if revision:
        return revision
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("unable to determine source revision")
    return proc.stdout.strip()


def elaboration_set(
    generated: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    rows = []
    if generated is None:
        sources = [
            (path, load_json(path))
            for path in sorted(ROOT.rglob("elaboration.json"))
        ]
    else:
        sources = [
            (Path(path_text), data)
            for path_text, data in generated.items()
            if path_text.endswith("/elaboration.json")
        ]
        sources.sort(key=lambda row: row[0].relative_to(ROOT).as_posix())
    for path, data in sources:
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "source_sha256": data["source_sha256"],
                "elaboration_trace_sha256": data["elaboration_trace_sha256"],
                "proof_step_count": data["proof_step_count"],
                "returncode": data["returncode"],
            }
        )
    body = {
        "artifact_count": len(rows),
        "proof_step_count": sum(row["proof_step_count"] for row in rows),
        "failure_count": sum(row["returncode"] != 0 for row in rows),
        "artifacts": rows,
    }
    body["elaboration_set_sha256"] = domain_hash(ELABORATION_SET_DOMAIN, body)
    return body


def build_receipt(live_replay: bool = False) -> Dict[str, Any]:
    lean = lean_executable()
    if live_replay:
        from compile_kernel_traces import compile_all, semantic_certificate

        compiled = compile_all(write=False)
        kernel_index = compiled["index"]
        semantic = semantic_certificate(kernel_index)
        elaborations = elaboration_set(compiled["generated"])
    else:
        semantic = load_json(ROOT / "semantic_replay_certificate.v1.json")
        kernel_index = load_json(ROOT / "kernel_trace_index.json")
        elaborations = elaboration_set()
    corpus = load_json(ROOT / "demo_pack_v1" / "corpus.json")
    lemma = load_json(ROOT / "lemma_mining" / "evaluation.v1.json")
    supply = load_json(ROOT / "supply_chain_manifest.v1.json")
    body: Dict[str, Any] = {
        "schema_id": "QA_MATH_COMPILER_INDEPENDENT_RUNNER_RECEIPT.v1",
        "receipt_id": "FAMILY31_TWO_CLEAN_RUNNER_PARITY_V1",
        "source_revision": source_revision(),
        "toolchain": {
            "lean_version": run_text(lean, "--version").splitlines()[0],
            "lean_git_commit": run_text(lean, "-g"),
        },
        "semantic_replay": {
            "certificate_sha256": semantic["certificate_sha256"],
            "case_count": semantic["case_count"],
            "success_count": semantic["success_count"],
            "failure_count": semantic["failure_count"],
        },
        "kernel_index": {
            "index_sha256": kernel_index["index_sha256"],
            "case_count": kernel_index["case_count"],
        },
        "certified_corpus_sha256": corpus["corpus_sha256"],
        "lemma_evaluation_sha256": lemma["evaluation_sha256"],
        "portable_supply_root_sha256": supply["portable_root_sha256"],
        "elaboration_set": elaborations,
        "receipt_generation": "live_replay" if live_replay else "stored_artifacts",
        "verified_gates": [
            "kernel_trace_live_replay",
            "semantic_replay_certificate",
            "lemma_mining_replay",
            "portable_supply_chain",
        ],
        "invariant_diff": {
            "runner_metadata_excluded": True,
            "wall_clock_excluded": True,
            "comparison_policy": "canonical JSON byte equality after independent clean checkout and replay",
        },
    }
    body["receipt_sha256"] = domain_hash(RECEIPT_DOMAIN, body)
    return body


def validate_receipt(receipt: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if (
        receipt.get("schema_id")
        != "QA_MATH_COMPILER_INDEPENDENT_RUNNER_RECEIPT.v1"
    ):
        errors.append("SCHEMA_ID_MISMATCH")
    body = dict(receipt)
    supplied = body.pop("receipt_sha256", None)
    if supplied != domain_hash(RECEIPT_DOMAIN, body):
        errors.append("RECEIPT_HASH_MISMATCH")
    elaborations = receipt.get("elaboration_set")
    if not isinstance(elaborations, dict):
        errors.append("ELABORATION_SET_MISSING")
    else:
        elab_body = dict(elaborations)
        elab_hash = elab_body.pop("elaboration_set_sha256", None)
        if elab_hash != domain_hash(ELABORATION_SET_DOMAIN, elab_body):
            errors.append("ELABORATION_SET_HASH_MISMATCH")
    return errors


def compare(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_errors = validate_receipt(left)
    right_errors = validate_receipt(right)
    equal = canonical_bytes(left) == canonical_bytes(right)
    errors = [
        *(f"LEFT:{error}" for error in left_errors),
        *(f"RIGHT:{error}" for error in right_errors),
    ]
    if not equal:
        errors.append("INDEPENDENT_RUNNER_PARITY_MISMATCH")
    return {
        "ok": not errors,
        "errors": errors,
        "receipt_sha256": left.get("receipt_sha256") if equal else None,
    }


def synthetic_receipt() -> Dict[str, Any]:
    elaboration_body = {
        "artifact_count": 1,
        "proof_step_count": 1,
        "failure_count": 0,
        "artifacts": [
            {
                "path": "synthetic/elaboration.json",
                "source_sha256": "1" * 64,
                "elaboration_trace_sha256": "2" * 64,
                "proof_step_count": 1,
                "returncode": 0,
            }
        ],
    }
    elaboration_body["elaboration_set_sha256"] = domain_hash(
        ELABORATION_SET_DOMAIN, elaboration_body
    )
    body: Dict[str, Any] = {
        "schema_id": "QA_MATH_COMPILER_INDEPENDENT_RUNNER_RECEIPT.v1",
        "receipt_id": "SYNTHETIC_SELF_TEST",
        "source_revision": "3" * 40,
        "toolchain": {
            "lean_version": "Lean (version 4.31.0, synthetic)",
            "lean_git_commit": "4" * 40,
        },
        "semantic_replay": {
            "certificate_sha256": "5" * 64,
            "case_count": 1,
            "success_count": 1,
            "failure_count": 0,
        },
        "kernel_index": {"index_sha256": "6" * 64, "case_count": 1},
        "certified_corpus_sha256": "7" * 64,
        "lemma_evaluation_sha256": "8" * 64,
        "portable_supply_root_sha256": "9" * 64,
        "elaboration_set": elaboration_body,
        "receipt_generation": "stored_artifacts",
        "verified_gates": ["synthetic"],
        "invariant_diff": {
            "runner_metadata_excluded": True,
            "wall_clock_excluded": True,
            "comparison_policy": "synthetic self-test",
        },
    }
    body["receipt_sha256"] = domain_hash(RECEIPT_DOMAIN, body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--live-replay", action="store_true")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    subparsers.add_parser("self-test")
    args = parser.parse_args()

    if args.command == "build":
        receipt = build_receipt(live_replay=args.live_replay)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_bytes(receipt) + b"\n")
        result = {
            "ok": True,
            "output": str(output),
            "receipt_sha256": receipt["receipt_sha256"],
        }
    elif args.command == "compare":
        result = compare(load_json(Path(args.left)), load_json(Path(args.right)))
    else:
        first = synthetic_receipt()
        second = synthetic_receipt()
        result = compare(first, second)
        if result["ok"]:
            tampered = json.loads(json.dumps(second))
            tampered["semantic_replay"]["case_count"] += 1
            if compare(first, tampered)["ok"]:
                result = {"ok": False, "errors": ["TAMPER_TEST_NOT_DETECTED"]}

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
