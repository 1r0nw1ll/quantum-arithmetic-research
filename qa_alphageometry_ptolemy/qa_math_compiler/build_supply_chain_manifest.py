#!/usr/bin/env python3
"""Build and verify the Family [31] supply-chain manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "supply_chain_manifest.v1.json"
MANIFEST_DOMAIN = "qa.math_compiler.supply_chain_manifest.v1"
PORTABLE_ROOT_DOMAIN = "qa.math_compiler.portable_artifact_root.v1"
PLATFORM_ROOT_DOMAIN = "qa.math_compiler.platform_toolchain_root.v1"

GENERATED_NAMES = {
    "corpus.json",
    "elaboration.json",
    "evaluation.v1.json",
    "kernel_trace_index.json",
    "lemma_pack.v1.json",
    "live_kernel_certificate.v1.json",
    "pair.json",
    "replay.json",
    "semantic_replay_certificate.v1.json",
    "status.json",
    "task.json",
    "trace.json",
}

LIBRARY_RELATIVE_PATHS = [
    "Init.olean",
    "Init/Omega.olean",
    "Init/Prelude.olean",
    "Init/Tactics.olean",
    "Lean.olean",
    "Lean/Elab.olean",
    "Lean/Elab/InfoTree/Main.olean",
    "Lean/Elab/Tactic.olean",
]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def domain_hash(domain: str, value: object) -> str:
    return sha256_bytes(domain.encode("utf-8") + b"\x00" + canonical_bytes(value))


def file_record(path: Path, relative_to: Path | None = None) -> Dict[str, Any]:
    stat = path.stat()
    display = (
        path.relative_to(relative_to).as_posix()
        if relative_to is not None
        else path.name
    )
    return {
        "path": display,
        "size_bytes": stat.st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def portable_paths() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if path == MANIFEST_PATH:
            continue
        if (
            path.suffix in {".lean", ".py"}
            or "schemas" in rel.parts
            or path.name in GENERATED_NAMES
            or path.name
            in {
                "kernel_trace_manifest.json",
                "toolchains.json",
                "index.json",
            }
        ):
            yield path


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
    if executable is None:
        candidate = Path.home() / ".elan" / "bin" / "lean"
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError("Lean executable not found")
    return executable


def resolved_binary(name: str, lean_path: Path) -> Path:
    sibling = lean_path.parent / name
    if sibling.is_file():
        return sibling.resolve()
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required Lean binary not found: {name}")
    return Path(executable).resolve()


def platform_receipt() -> Dict[str, Any]:
    lean_command = lean_executable()
    prefix = Path(run_text(lean_command, "--print-prefix")).resolve()
    libdir = Path(run_text(lean_command, "--print-libdir")).resolve()
    lean = prefix / "bin" / "lean"
    version = run_text(lean_command, "--version").splitlines()[0]
    git_commit = run_text(lean_command, "-g")
    binaries = [
        {
            "name": name,
            **file_record(resolved_binary(name, lean)),
        }
        for name in ["lean", "lake", "leanc"]
    ]
    libraries = []
    for rel in LIBRARY_RELATIVE_PATHS:
        path = libdir / rel
        if not path.is_file():
            raise RuntimeError(f"required Lean library missing: {rel}")
        libraries.append(file_record(path, libdir))
    body: Dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "lean_version": version,
        "lean_git_commit": git_commit,
        "toolchain_prefix_name": prefix.name,
        "binaries": binaries,
        "libraries": libraries,
    }
    body["platform_root_sha256"] = domain_hash(PLATFORM_ROOT_DOMAIN, body)
    return body


def build_portable_body() -> Dict[str, Any]:
    artifacts = [
        file_record(path, ROOT)
        for path in sorted(portable_paths(), key=lambda item: item.relative_to(ROOT).as_posix())
    ]
    return {
        "file_count": len(artifacts),
        "total_size_bytes": sum(row["size_bytes"] for row in artifacts),
        "files": artifacts,
    }


def build(existing: Dict[str, Any] | None = None) -> Dict[str, Any]:
    portable_body = build_portable_body()
    receipt = platform_receipt()
    receipts = {}
    if existing is not None:
        for row in existing.get("platform_receipts", []):
            key = f"{row.get('platform')}/{row.get('machine')}"
            receipts[key] = row
    receipts[f"{receipt['platform']}/{receipt['machine']}"] = receipt
    body: Dict[str, Any] = {
        "schema_id": "QA_MATH_COMPILER_SUPPLY_CHAIN_MANIFEST.v1",
        "manifest_id": "FAMILY31_SUPPLY_CHAIN_V1",
        "created_utc": "2026-06-20T00:00:00Z",
        "hash_algorithm": "sha256",
        "portable_artifacts": portable_body,
        "portable_root_sha256": domain_hash(PORTABLE_ROOT_DOMAIN, portable_body),
        "platform_receipts": [receipts[key] for key in sorted(receipts)],
        "invariant_diff": {
            "portable_scope": "Family [31] sources, schemas, proof inputs, and generated certificates",
            "platform_scope": "Lean executable set and selected compiled core libraries",
            "path_policy": "repository-relative or toolchain-lib-relative; no absolute paths",
            "cross_platform_policy": "portable root must match; platform roots are compared only within identical platform/machine keys",
        },
    }
    body["manifest_sha256"] = domain_hash(MANIFEST_DOMAIN, body)
    return body


def load_manifest() -> Dict[str, Any]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("supply-chain manifest must be an object")
    return value


def validate_hashes(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    portable = manifest.get("portable_artifacts", {})
    if manifest.get("portable_root_sha256") != domain_hash(
        PORTABLE_ROOT_DOMAIN, portable
    ):
        errors.append("PORTABLE_ROOT_HASH_MISMATCH")
    for receipt in manifest.get("platform_receipts", []):
        body = dict(receipt)
        supplied = body.pop("platform_root_sha256", None)
        if supplied != domain_hash(PLATFORM_ROOT_DOMAIN, body):
            errors.append(
                f"PLATFORM_ROOT_HASH_MISMATCH:{receipt.get('platform')}/{receipt.get('machine')}"
            )
    body = dict(manifest)
    supplied = body.pop("manifest_sha256", None)
    if supplied != domain_hash(MANIFEST_DOMAIN, body):
        errors.append("MANIFEST_HASH_MISMATCH")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "check", "check-portable"])
    args = parser.parse_args()
    existing = load_manifest() if MANIFEST_PATH.exists() else None
    if args.mode == "write":
        manifest = build(existing)
        MANIFEST_PATH.write_bytes(canonical_bytes(manifest) + b"\n")
        result = {
            "ok": True,
            "manifest_sha256": manifest["manifest_sha256"],
            "portable_root_sha256": manifest["portable_root_sha256"],
            "portable_file_count": manifest["portable_artifacts"]["file_count"],
            "platform_receipt_count": len(manifest["platform_receipts"]),
        }
    else:
        if existing is None:
            raise RuntimeError("supply-chain manifest missing")
        errors = validate_hashes(existing)
        current_portable = build_portable_body()
        if existing["portable_artifacts"] != current_portable:
            errors.append("PORTABLE_ARTIFACT_DRIFT")
        if args.mode == "check":
            current = build(existing)
            key = f"{platform.system()}/{platform.machine()}"
            stored = {
                f"{row['platform']}/{row['machine']}": row
                for row in existing["platform_receipts"]
            }
            current_receipt = {
                f"{row['platform']}/{row['machine']}": row
                for row in current["platform_receipts"]
            }[key]
            if key not in stored:
                errors.append(f"PLATFORM_RECEIPT_MISSING:{key}")
            elif stored[key] != current_receipt:
                errors.append(f"PLATFORM_TOOLCHAIN_DRIFT:{key}")
        result = {
            "ok": not errors,
            "errors": errors,
            "manifest_sha256": existing.get("manifest_sha256"),
            "portable_root_sha256": existing.get("portable_root_sha256"),
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
