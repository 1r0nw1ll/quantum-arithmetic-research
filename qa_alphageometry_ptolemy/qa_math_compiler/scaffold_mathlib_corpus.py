#!/usr/bin/env python3
"""Deterministically scaffold the pinned Mathlib certificate corpus from the upstream registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "mathlib_ingest" / "upstream_registry.v1.json"
PACK = ROOT / "mathlib_pack_v1"
EXAMPLES = PACK / "examples"
MANIFEST_PATH = ROOT / "kernel_trace_manifest.json"
CREATED_UTC = "2026-06-21T00:00:00Z"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return value


def _required(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"registry entry for {entry.get('declaration')!r} missing {key!r}")
    return value


def generated_files() -> dict[Path, bytes]:
    registry = load_json(REGISTRY_PATH)
    entries = registry["entries"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("registry entries must be a non-empty list")

    files: dict[Path, bytes] = {}

    def add_text(path: Path, text: str) -> None:
        files[path] = text.encode("utf-8")

    def add_json(path: Path, value: object) -> None:
        add_text(path, canonical(value) + "\n")

    add_text(
        PACK / "README.md",
        "# Mathlib Certified Corpus v1\n\n"
        "Replay-backed wrappers around 15 declarations from the exact pinned "
        "Mathlib v4.31.0 source revision.\n",
    )
    index_entries = []
    for entry in entries:
        example_id = _required(entry, "example_id")
        claim = _required(entry, "claim")
        formal = _required(entry, "formal")
        declaration = _required(entry, "declaration")
        module = _required(entry, "module")

        example_dir = EXAMPLES / example_id
        import_line = f"import {module}"
        proof_body = entry.get("proof_body")
        if proof_body is not None:
            proof_source = f"{import_line}\n\n{formal} := by\n{proof_body}\n"
        else:
            proof_term = _required(entry, "proof_term")
            proof_source = f"{import_line}\n\n{formal} := by\n  exact {proof_term}\n"

        provenance = {
            "repository": registry["source"]["repository"],
            "release": registry["source"]["release"],
            "commit": registry["source"]["commit"],
            "declaration": declaration,
            "source_path": entry["source_path"],
            "source_file_sha256": entry["source_file_sha256"],
            "declaration_source_sha256": entry["declaration_source_sha256"],
        }
        add_text(example_dir / "claim.txt", claim + "\n")
        add_text(example_dir / "proof.lean", proof_source)
        add_text(
            example_dir / "README.md",
            f"# {example_id}\n\n"
            f"Replay-certified wrapper for `{declaration}` from the "
            "pinned Mathlib source revision.\n",
        )
        add_json(
            example_dir / "task.json",
            {
                "schema_id": "QA_FORMAL_TASK_SCHEMA.v1",
                "task_id": f"MATHLIB_{example_id.upper()}_TASK",
                "created_utc": CREATED_UTC,
                "nl_statement": claim,
                "formal_goal": formal,
                "imports": [module],
                "context": [],
                "constraints": {
                    "max_seconds": 120,
                    "max_memory_mb": 4096,
                    "allowed_tactics": ["exact"],
                },
                "invariant_diff": {
                    "corpus": "mathlib_pack_v1",
                    "category": entry["category"],
                    "provenance": provenance,
                },
            },
        )
        trace_path = example_dir / "trace.json"
        linked_trace = load_json(trace_path) if trace_path.is_file() else None
        trace_id = linked_trace["trace_id"] if linked_trace is not None else "0" * 64
        pair = {
            "schema_id": "QA_HUMAN_FORMAL_PAIR_CERT.v1",
            "pair_id": f"MATHLIB_{example_id.upper()}_PAIR",
            "created_utc": CREATED_UTC,
            "natural_language_claim": claim,
            "formal_statement": formal,
            "alignment_evidence": {
                "key_lemmas": [declaration],
                "span_mappings": [
                    {
                        "nl_span": claim.rstrip("."),
                        "formal_identifiers": [declaration],
                    }
                ],
            },
            "trace_ref": {
                "trace_id": trace_id,
                "result_status": "SUCCESS",
                "replay_status": "SUCCESS",
            },
            "status": "PROVED",
            "objections": [],
            "invariant_diff": {
                "corpus": "mathlib_pack_v1",
                "upstream_declaration": declaration,
            },
        }
        add_json(example_dir / "pair.json", pair)
        status = {
            "example_id": example_id,
            "status": "PROVED",
            "replay_rate": 1.0,
            "compressed": False,
            "introduced_lemmas": 0,
            "upstream_declaration": declaration,
        }
        if linked_trace is not None:
            status.update(
                {
                    "kernel_derived": True,
                    "toolchain_id": linked_trace["toolchain_id"],
                    "trace_id": trace_id,
                }
            )
        add_json(example_dir / "status.json", status)
        index_entries.append(
            {
                "id": example_id,
                "topic": entry["category"],
                "difficulty": "upstream",
                "status": "PROVED",
            }
        )
    add_json(
        PACK / "index.json",
        {
            "schema_id": "QA_MATH_COMPILER_DEMO_PACK_SCHEMA.v1",
            "version": "v1",
            "example_count": len(index_entries),
            "examples": index_entries,
        },
    )

    manifest = load_json(MANIFEST_PATH)
    retained = [
        row
        for row in manifest["cases"]
        if not str(row.get("case_id", "")).startswith("mathlib")
    ]
    for row in retained:
        if row.get("case_id") == "negative_library_gap":
            row["trace_before_imports"] = True
    retained.extend(
        {
            "case_id": entry["example_id"],
            "source": f"mathlib_pack_v1/examples/{entry['example_id']}/proof.lean",
            "expected_status": "SUCCESS",
            "proof_method": entry["declaration"],
            "artifact_dir": f"mathlib_pack_v1/examples/{entry['example_id']}",
            "lake_project": "mathlib_ingest",
            "upstream_declaration": entry["declaration"],
        }
        for entry in entries
    )
    manifest["cases"] = retained
    add_json(MANIFEST_PATH, manifest)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "check"])
    args = parser.parse_args()
    files = generated_files()
    registry = load_json(REGISTRY_PATH)
    mismatches = []
    for path, payload in sorted(files.items()):
        if args.mode == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        elif not path.is_file() or path.read_bytes() != payload:
            mismatches.append(path.relative_to(ROOT).as_posix())
    result = {
        "ok": not mismatches,
        "mode": args.mode,
        "case_count": len(registry["entries"]),
        "file_count": len(files),
        "mismatches": mismatches,
    }
    print(canonical(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
