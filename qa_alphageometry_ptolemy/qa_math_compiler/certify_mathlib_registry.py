#!/usr/bin/env python3
"""Bind pinned Mathlib provenance entries to replay-backed certificate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "mathlib_ingest" / "upstream_registry.v1.json"
PACK = ROOT / "mathlib_pack_v1"
REGISTRY_DOMAIN = "qa.math_compiler.upstream_proof_registry.v1"
CERTIFICATE_DOMAIN = "qa.math_compiler.upstream_proof_certificate.v1"


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return value


def artifact_hash(path: Path) -> str:
    return sha256_bytes(canonical_bytes(load_json(path)))


def build_registry() -> dict[str, Any]:
    registry = load_json(REGISTRY_PATH)
    by_declaration = {
        entry["declaration"]: entry
        for entry in registry["entries"]
        if isinstance(entry, dict)
    }
    seen = set()
    for example_dir in sorted((PACK / "examples").iterdir()):
        if not example_dir.is_dir():
            continue
        status = load_json(example_dir / "status.json")
        declaration = status["upstream_declaration"]
        if declaration not in by_declaration:
            raise RuntimeError(f"unregistered declaration: {declaration}")
        trace = load_json(example_dir / "trace.json")
        replay = load_json(example_dir / "replay.json")
        elaboration = load_json(example_dir / "elaboration.json")
        replay_rows = [
            row
            for row in replay["traces"]
            if row.get("trace_id") == trace["trace_id"]
        ]
        if len(replay_rows) != 1 or replay_rows[0].get("replay_status") != "SUCCESS":
            raise RuntimeError(f"{example_dir.name}: replay is not certified")
        certificate = {
            "pack_id": "mathlib_pack_v1",
            "example_id": example_dir.name,
            "certificate_dir": (
                Path("mathlib_pack_v1") / "examples" / example_dir.name
            ).as_posix(),
            "proof_sha256": sha256_bytes((example_dir / "proof.lean").read_bytes()),
            "task_sha256": artifact_hash(example_dir / "task.json"),
            "trace_sha256": artifact_hash(example_dir / "trace.json"),
            "elaboration_sha256": artifact_hash(example_dir / "elaboration.json"),
            "replay_sha256": artifact_hash(example_dir / "replay.json"),
            "pair_sha256": artifact_hash(example_dir / "pair.json"),
            "trace_id": trace["trace_id"],
            "replay_status": replay_rows[0]["replay_status"],
            "dependency_lock_sha256": elaboration["dependency_lock_sha256"],
        }
        certificate["certificate_sha256"] = domain_hash(
            CERTIFICATE_DOMAIN,
            certificate,
        )
        entry = by_declaration[declaration]
        entry["status"] = "CERTIFIED"
        entry["certificate"] = certificate
        seen.add(declaration)
    if seen != set(by_declaration):
        raise RuntimeError("not every registry declaration has a certificate directory")
    registry_body = dict(registry)
    registry_body.pop("registry_sha256", None)
    registry["registry_sha256"] = domain_hash(REGISTRY_DOMAIN, registry_body)
    return registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "check"])
    args = parser.parse_args()
    built = build_registry()
    stored = load_json(REGISTRY_PATH)
    if args.mode == "write":
        REGISTRY_PATH.write_bytes(canonical_bytes(built) + b"\n")
        ok = True
    else:
        ok = stored == built
    print(
        json.dumps(
            {
                "ok": ok,
                "mode": args.mode,
                "entry_count": len(built["entries"]),
                "certified_count": sum(
                    entry.get("status") == "CERTIFIED"
                    for entry in built["entries"]
                ),
                "registry_sha256": built["registry_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
