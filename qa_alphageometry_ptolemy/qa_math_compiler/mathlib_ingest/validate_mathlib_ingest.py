#!/usr/bin/env python3
"""Validate the pinned Mathlib upstream-proof registry and optional source checkout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "upstream_registry.v1.json"
NEGATIVE_FIXTURE = ROOT / "fixtures" / "registry_invalid_digest.json"
REGISTRY_DOMAIN = "qa.math_compiler.upstream_proof_registry.v1"
CERTIFICATE_DOMAIN = "qa.math_compiler.upstream_proof_certificate.v1"
SCHEMA_ID = "QA_MATH_COMPILER_UPSTREAM_PROOF_REGISTRY.v1"
PINNED_COMMIT = "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
PINNED_RELEASE = "v4.31.0"
PINNED_LICENSE = "Apache-2.0"
REQUIRED_CATEGORIES = {
    "arithmetic",
    "finite-counting",
    "finite-sets",
    "induction",
    "lists",
}
CERTIFICATE_FILES = {
    "task_sha256": "task.json",
    "trace_sha256": "trace.json",
    "elaboration_sha256": "elaboration.json",
    "replay_sha256": "replay.json",
    "pair_sha256": "pair.json",
}


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def registry_body(registry: dict[str, Any]) -> dict[str, Any]:
    body = dict(registry)
    body.pop("registry_sha256", None)
    return body


def verify_source_checkout(
    registry: dict[str, Any],
    source_root: Path,
    errors: list[str],
) -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        errors.append("SOURCE_GIT_HEAD_UNAVAILABLE")
        return
    if commit != PINNED_COMMIT:
        errors.append("SOURCE_COMMIT_MISMATCH")

    source = registry["source"]
    fixed_files = {
        "LICENSE": source["license_sha256"],
        "lean-toolchain": source["lean_toolchain_sha256"],
        "lake-manifest.json": source["lake_manifest_sha256"],
    }
    for relative, expected in fixed_files.items():
        path = source_root / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            errors.append(f"SOURCE_FILE_HASH_MISMATCH:{relative}")

    for entry in registry["entries"]:
        source_path = source_root / entry["source_path"]
        if not source_path.is_file():
            errors.append(f"SOURCE_FILE_MISSING:{entry['source_path']}")
            continue
        source_bytes = source_path.read_bytes()
        if sha256_bytes(source_bytes) != entry["source_file_sha256"]:
            errors.append(f"SOURCE_FILE_HASH_MISMATCH:{entry['source_path']}")
            continue
        lines = source_bytes.splitlines(keepends=True)
        start = entry["start_line"]
        end = entry["end_line"]
        if start == 0 and end == 0:
            continue  # synthetic sentinel — no source range to verify
        if start < 1 or end < start or end > len(lines):
            errors.append(f"SOURCE_RANGE_INVALID:{entry['declaration']}")
            continue
        fragment = b"".join(lines[start - 1 : end])
        if sha256_bytes(fragment) != entry["declaration_source_sha256"]:
            errors.append(f"DECLARATION_HASH_MISMATCH:{entry['declaration']}")


def verify_certificate(entry: dict[str, Any], errors: list[str]) -> None:
    declaration = entry["declaration"]
    certificate = entry.get("certificate")
    if not isinstance(certificate, dict):
        errors.append(f"CERTIFICATE_MISSING:{declaration}")
        return
    certificate_body = dict(certificate)
    supplied_hash = certificate_body.pop("certificate_sha256", None)
    if supplied_hash != domain_hash(CERTIFICATE_DOMAIN, certificate_body):
        errors.append(f"CERTIFICATE_HASH_MISMATCH:{declaration}")
    certificate_dir = certificate.get("certificate_dir")
    if (
        not isinstance(certificate_dir, str)
        or certificate_dir.startswith("/")
        or ".." in Path(certificate_dir).parts
    ):
        errors.append(f"CERTIFICATE_DIR_INVALID:{declaration}")
        return
    artifact_dir = ROOT.parent / certificate_dir
    proof_path = artifact_dir / "proof.lean"
    if (
        not proof_path.is_file()
        or sha256_bytes(proof_path.read_bytes()) != certificate.get("proof_sha256")
    ):
        errors.append(f"CERTIFICATE_ARTIFACT_MISMATCH:{declaration}:proof.lean")
    loaded: dict[str, dict[str, Any]] = {}
    for hash_field, filename in CERTIFICATE_FILES.items():
        path = artifact_dir / filename
        if not path.is_file():
            errors.append(f"CERTIFICATE_ARTIFACT_MISSING:{declaration}:{filename}")
            continue
        artifact = load_json(path)
        loaded[filename] = artifact
        if sha256_bytes(canonical_bytes(artifact)) != certificate.get(hash_field):
            errors.append(f"CERTIFICATE_ARTIFACT_MISMATCH:{declaration}:{filename}")
    trace = loaded.get("trace.json")
    replay = loaded.get("replay.json")
    elaboration = loaded.get("elaboration.json")
    if trace is not None and trace.get("trace_id") != certificate.get("trace_id"):
        errors.append(f"CERTIFICATE_TRACE_ID_MISMATCH:{declaration}")
    if elaboration is not None and (
        elaboration.get("dependency_lock_sha256")
        != certificate.get("dependency_lock_sha256")
    ):
        errors.append(f"CERTIFICATE_LOCK_MISMATCH:{declaration}")
    if trace is not None and replay is not None:
        rows = [
            row
            for row in replay.get("traces", [])
            if isinstance(row, dict) and row.get("trace_id") == trace.get("trace_id")
        ]
        if (
            len(rows) != 1
            or rows[0].get("replay_status") != "SUCCESS"
            or certificate.get("replay_status") != "SUCCESS"
        ):
            errors.append(f"CERTIFICATE_REPLAY_NOT_SUCCESS:{declaration}")


def validate_registry(
    registry: dict[str, Any],
    source_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if registry.get("schema_id") != SCHEMA_ID:
        errors.append("SCHEMA_ID_MISMATCH")
    expected_hash = domain_hash(REGISTRY_DOMAIN, registry_body(registry))
    if registry.get("registry_sha256") != expected_hash:
        errors.append("REGISTRY_HASH_MISMATCH")

    source = registry.get("source")
    if not isinstance(source, dict):
        errors.append("SOURCE_METADATA_MISSING")
        source = {}
    if source.get("commit") != PINNED_COMMIT:
        errors.append("SOURCE_COMMIT_MISMATCH")
    if source.get("release") != PINNED_RELEASE:
        errors.append("SOURCE_RELEASE_MISMATCH")
    if source.get("license") != PINNED_LICENSE:
        errors.append("SOURCE_LICENSE_MISMATCH")
    if source.get("repository") != "https://github.com/leanprover-community/mathlib4":
        errors.append("SOURCE_REPOSITORY_MISMATCH")

    entries = registry.get("entries")
    if not isinstance(entries, list) or not 10 <= len(entries) <= 50:
        errors.append("ENTRY_COUNT_OUT_OF_RANGE")
        entries = []
    declarations: set[str] = set()
    categories: set[str] = set()
    file_hashes: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"ENTRY_INVALID:{index}")
            continue
        declaration = entry.get("declaration")
        source_path = entry.get("source_path")
        module = entry.get("module")
        if not isinstance(declaration, str) or not declaration:
            errors.append(f"ENTRY_DECLARATION_INVALID:{index}")
            continue
        if declaration in declarations:
            errors.append(f"DECLARATION_DUPLICATE:{declaration}")
        declarations.add(declaration)
        if (
            not isinstance(source_path, str)
            or source_path.startswith("/")
            or ".." in Path(source_path).parts
            or not source_path.endswith(".lean")
        ):
            errors.append(f"SOURCE_PATH_INVALID:{declaration}")
        if not isinstance(module, str) or module.replace(".", "/") + ".lean" != source_path:
            errors.append(f"MODULE_PATH_MISMATCH:{declaration}")
        if entry.get("split") != "upstream_eval":
            errors.append(f"SPLIT_POLICY_VIOLATION:{declaration}")
        status = entry.get("status")
        if status not in {"PROVENANCE_VERIFIED", "CERTIFIED"}:
            errors.append(f"STATUS_INVALID:{declaration}")
        elif status == "CERTIFIED":
            verify_certificate(entry, errors)
        category = entry.get("category")
        if isinstance(category, str):
            categories.add(category)
        for field in ("source_file_sha256", "declaration_source_sha256"):
            value = entry.get(field)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"HASH_FIELD_INVALID:{declaration}:{field}")
        if isinstance(source_path, str):
            previous = file_hashes.setdefault(
                source_path,
                str(entry.get("source_file_sha256", "")),
            )
            if previous != entry.get("source_file_sha256"):
                errors.append(f"SOURCE_FILE_HASH_INCONSISTENT:{source_path}")
        start = entry.get("start_line")
        end = entry.get("end_line")
        # (0, 0) is the sentinel for synthetic entries without a upstream source range.
        synthetic_range = (start == 0 and end == 0)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or (not synthetic_range and (start < 1 or end < start))
        ):
            errors.append(f"SOURCE_RANGE_INVALID:{declaration}")

    if not REQUIRED_CATEGORIES.issubset(categories):
        errors.append("CATEGORY_COVERAGE_INCOMPLETE")
    if source_root is not None and not errors:
        verify_source_checkout(registry, source_root.resolve(), errors)

    return {
        "ok": not errors,
        "errors": errors,
        "entry_count": len(entries),
        "categories": sorted(categories),
        "registry_sha256": registry.get("registry_sha256"),
        "recomputed_sha256": expected_hash,
        "source_verified": source_root is not None and not errors,
    }


def apply_mutation(value: dict[str, Any], path: str, replacement: object) -> None:
    parts = path.split(".")
    current: Any = value
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    final = parts[-1]
    if isinstance(current, list):
        current[int(final)] = replacement
    else:
        current[final] = replacement


def self_test() -> dict[str, Any]:
    valid = validate_registry(load_json(DEFAULT_REGISTRY))
    fixture = load_json(NEGATIVE_FIXTURE)
    invalid_registry = copy.deepcopy(load_json(DEFAULT_REGISTRY))
    mutation = fixture["mutation"]
    apply_mutation(invalid_registry, mutation["path"], mutation["value"])
    invalid = validate_registry(invalid_registry)
    expected = fixture["expected_fail_type"]
    checks = [
        {"name": "valid registry accepted", "ok": valid["ok"]},
        {
            "name": "tampered registry rejected",
            "ok": not invalid["ok"] and expected in invalid["errors"],
        },
    ]
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", nargs="?", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = (
        self_test()
        if args.self_test
        else validate_registry(load_json(args.registry), args.source_root)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
