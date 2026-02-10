#!/usr/bin/env python3
"""
qa_elliptic_correspondence_bundle_v1.py

Emitter and validator for QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from qa_cert_core import sha256_canonical, sha256_file


HEX64_ZERO = "0" * 64

BUNDLE_SCHEMA_ID = "QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1"
BUNDLE_VERSION = 1

# (artifact_id, relative_path, kind, is_json)
ARTIFACT_SPEC: List[Tuple[str, str, str, bool]] = [
    ("elliptic_cert", "certs/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.json", "json_cert", True),
    ("elliptic_cert_sha256", "certs/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.sha256", "hash_sidecar", False),
    ("elliptic_schema", "schemas/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.schema.json", "json_schema", True),
    ("elliptic_validator", "qa_elliptic_correspondence_validator_v3.py", "source_py", False),
    ("elliptic_certificate_module", "qa_elliptic_correspondence_certificate.py", "source_py", False),
    ("elliptic_mapping", "QA_MAP__ELLIPTIC_CORRESPONDENCE.yaml", "source_yaml", False),
    (
        "elliptic_success_example",
        "examples/elliptic_correspondence/elliptic_correspondence_success.json",
        "json_example",
        True,
    ),
    (
        "elliptic_failure_example",
        "examples/elliptic_correspondence/elliptic_correspondence_ramification_failure.json",
        "json_example",
        True,
    ),
    ("bundle_schema", "schemas/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.schema.json", "json_schema", True),
    ("bundle_emitter", "qa_elliptic_correspondence_bundle_v1.py", "source_py", False),
]


def _manifest_hashable_copy(bundle: Dict[str, Any]) -> Dict[str, Any]:
    cpy = copy.deepcopy(bundle)
    cpy.setdefault("manifest", {})
    cpy["manifest"]["manifest_sha256"] = HEX64_ZERO
    return cpy


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _artifact_entry(base_dir: str, artifact_id: str, rel_path: str, kind: str, is_json: bool) -> Dict[str, Any]:
    abs_path = os.path.join(base_dir, rel_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Missing artifact for bundle emission: {rel_path}")

    entry: Dict[str, Any] = {
        "id": artifact_id,
        "file": rel_path,
        "kind": kind,
        "sha256": sha256_file(abs_path),
    }
    if is_json:
        entry["canonical_sha256"] = sha256_canonical(_load_json(abs_path))
    return entry


def build_bundle(base_dir: str) -> Dict[str, Any]:
    artifacts: Dict[str, Dict[str, Any]] = {}
    for artifact_id, rel_path, kind, is_json in ARTIFACT_SPEC:
        artifacts[artifact_id] = _artifact_entry(base_dir, artifact_id, rel_path, kind, is_json)

    cert_entry = artifacts["elliptic_cert"]
    cert_sha_entry = artifacts["elliptic_cert_sha256"]

    bundle: Dict[str, Any] = {
        "schema_id": BUNDLE_SCHEMA_ID,
        "version": BUNDLE_VERSION,
        "bundle_id": "qa.bundle.elliptic_correspondence.v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hash_spec": {
            "id": "qa.hash_spec.v1",
            "version": "1.0",
            "sha256": "file_bytes (exact file content integrity)",
            "canonical_sha256": "canonical_json (semantic identity, sorted keys, no whitespace)",
            "canonical_spec": "json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False)",
            "source": "qa_cert_core.canonical_json_compact",
        },
        "emitted_cert": {
            "cert_id": "elliptic_correspondence_demo_001",
            "cert_path": cert_entry["file"],
            "cert_sha256": cert_entry["sha256"],
            "sha256_sidecar_path": cert_sha_entry["file"],
        },
        "artifacts": artifacts,
        "manifest": {
            "hash_alg": "sha256",
            "manifest_sha256": HEX64_ZERO,
        },
    }

    bundle["manifest"]["manifest_sha256"] = sha256_canonical(_manifest_hashable_copy(bundle))
    return bundle


def emit_bundle(bundle_path: str, base_dir: str) -> Dict[str, Any]:
    bundle = build_bundle(base_dir)
    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)
        f.write("\n")
    return bundle


def _validate_schema(bundle: Dict[str, Any], schema_path: str) -> Optional[str]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return None

    try:
        schema = _load_json(schema_path)
        jsonschema.validate(instance=bundle, schema=schema)
    except Exception as e:
        return f"schema validation failed: {e}"
    return None


def _validate_sidecar(base_dir: str, bundle: Dict[str, Any]) -> Optional[str]:
    emitted = bundle.get("emitted_cert", {})
    cert_sha = emitted.get("cert_sha256", "")
    sidecar_path = emitted.get("sha256_sidecar_path", "")
    cert_path = emitted.get("cert_path", "")

    abs_sidecar = os.path.join(base_dir, sidecar_path)
    abs_cert = os.path.join(base_dir, cert_path)
    if not os.path.exists(abs_sidecar):
        return f"sidecar missing: {sidecar_path}"
    if not os.path.exists(abs_cert):
        return f"cert missing: {cert_path}"

    with open(abs_sidecar, "r", encoding="utf-8") as f:
        line = f.read().strip()
    parts = line.split()
    if len(parts) < 2:
        return "sidecar format invalid"
    sidecar_sha = parts[0]
    sidecar_file = parts[-1]
    if sidecar_sha != cert_sha:
        return f"sidecar sha mismatch: sidecar={sidecar_sha} bundle={cert_sha}"
    if sidecar_file != cert_path:
        return f"sidecar cert path mismatch: sidecar={sidecar_file} expected={cert_path}"
    return None


def validate_bundle_manifest(bundle_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    errors: List[str] = []
    checks: List[str] = []

    if not os.path.exists(bundle_path):
        return {"ok": False, "errors": [f"bundle not found: {bundle_path}"], "checks": []}

    try:
        bundle = _load_json(bundle_path)
    except Exception as e:
        return {"ok": False, "errors": [f"failed to load bundle: {e}"], "checks": []}

    if bundle.get("schema_id") != BUNDLE_SCHEMA_ID:
        errors.append(f"schema_id mismatch: {bundle.get('schema_id')}")
    else:
        checks.append("schema_id: OK")

    if bundle.get("version") != BUNDLE_VERSION:
        errors.append(f"version mismatch: {bundle.get('version')}")
    else:
        checks.append("version: OK")

    schema_path = os.path.join(base_dir, "schemas", "QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.schema.json")
    schema_error = _validate_schema(bundle, schema_path)
    if schema_error is None:
        checks.append("schema_validation: OK")
    else:
        errors.append(schema_error)

    manifest = bundle.get("manifest", {})
    claimed_manifest_sha = manifest.get("manifest_sha256", "")
    if not (isinstance(claimed_manifest_sha, str) and len(claimed_manifest_sha) == 64):
        errors.append("manifest.manifest_sha256 must be 64-hex")
    else:
        recomputed_manifest_sha = sha256_canonical(_manifest_hashable_copy(bundle))
        if recomputed_manifest_sha != claimed_manifest_sha:
            errors.append(
                f"manifest hash mismatch: claimed={claimed_manifest_sha} recomputed={recomputed_manifest_sha}"
            )
        else:
            checks.append("manifest_sha256: OK")

    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) == 0:
        errors.append("artifacts must be a non-empty object")
        artifacts = {}

    for name, entry in artifacts.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}: artifact entry must be object")
            continue

        rel_path = entry.get("file", "")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(f"{name}: missing file")
            continue

        abs_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(abs_path):
            errors.append(f"{name}: file missing: {rel_path}")
            continue

        claimed_sha = entry.get("sha256", "")
        actual_sha = sha256_file(abs_path)
        if claimed_sha != actual_sha:
            errors.append(f"{name}: sha256 mismatch")
        else:
            checks.append(f"{name}: file sha256 OK")

        if "canonical_sha256" in entry:
            claimed_canonical = entry.get("canonical_sha256", "")
            try:
                obj = _load_json(abs_path)
            except Exception as e:
                errors.append(f"{name}: canonical hash requested but json load failed: {e}")
                continue
            actual_canonical = sha256_canonical(obj)
            if claimed_canonical != actual_canonical:
                errors.append(f"{name}: canonical_sha256 mismatch")
            else:
                checks.append(f"{name}: canonical_sha256 OK")

    sidecar_error = _validate_sidecar(base_dir, bundle)
    if sidecar_error is None:
        checks.append("sidecar_consistency: OK")
    else:
        errors.append(sidecar_error)

    emitted = bundle.get("emitted_cert", {})
    if isinstance(emitted, dict):
        cert_path = emitted.get("cert_path", "")
        cert_sha = emitted.get("cert_sha256", "")
        cert_entry = artifacts.get("elliptic_cert", {}) if isinstance(artifacts, dict) else {}
        if cert_entry:
            if cert_entry.get("file") != cert_path or cert_entry.get("sha256") != cert_sha:
                errors.append("emitted_cert does not match elliptic_cert artifact entry")
            else:
                checks.append("emitted_cert linkage: OK")

    return {
        "ok": len(errors) == 0,
        "bundle_path": bundle_path,
        "artifact_count": len(artifacts),
        "checks": checks,
        "errors": errors,
    }


def _default_bundle_path(base_dir: str) -> str:
    return os.path.join(base_dir, "certs", "QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit/validate QA elliptic correspondence bundle manifest v1.")
    parser.add_argument("--emit", action="store_true", help="Emit bundle manifest")
    parser.add_argument("--check", action="store_true", help="Validate bundle manifest")
    parser.add_argument("--bundle", default="", help="Bundle path (default: certs/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.json)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    bundle_path = args.bundle or _default_bundle_path(base_dir)

    run_emit = args.emit
    run_check = args.check
    if not run_emit and not run_check:
        run_emit = True
        run_check = True

    output: Dict[str, Any] = {}

    if run_emit:
        bundle = emit_bundle(bundle_path, base_dir)
        output["emit"] = {
            "ok": True,
            "bundle_path": bundle_path,
            "artifact_count": len(bundle.get("artifacts", {})),
            "manifest_sha256": bundle.get("manifest", {}).get("manifest_sha256", ""),
        }

    if run_check:
        output["check"] = validate_bundle_manifest(bundle_path, base_dir)

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        if "emit" in output:
            emit = output["emit"]
            print(f"EMIT: OK ({emit['artifact_count']} artifacts) -> {emit['bundle_path']}")
            print(f"manifest_sha256: {emit['manifest_sha256']}")
        if "check" in output:
            chk = output["check"]
            print(f"CHECK: {'PASS' if chk['ok'] else 'FAIL'} ({chk['artifact_count']} artifacts)")
            for line in chk.get("checks", []):
                print(f"  + {line}")
            for err in chk.get("errors", []):
                print(f"  - {err}")

    chk = output.get("check", {"ok": True})
    return 0 if chk.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
