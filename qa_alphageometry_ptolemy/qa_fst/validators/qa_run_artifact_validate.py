# noqa: DECL-1 (infrastructure — schema validator for QA_RUN_ARTIFACT_BUNDLE.v1; not empirical)
#!/usr/bin/env python3
"""
Validate QA_RUN_ARTIFACT_BUNDLE.v1 certificates.

Usage:
  python qa_run_artifact_validate.py <cert_dir> <schema_dir>

Returns:
  0 on success, 1 on validation failure, 2 on usage error.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _manual_shape_checks(spine: Dict[str, Any]) -> Tuple[bool, str]:
    if spine.get("schema_id") != "QA_RUN_ARTIFACT_BUNDLE.v1":
        return False, "bad_schema_id"
    if spine.get("version") != 1:
        return False, "bad_version"

    for key in ("run", "git", "artifacts", "result", "witness_manifest"):
        if not isinstance(spine.get(key), dict):
            return False, f"missing_or_bad_{key}"

    run = spine["run"]
    for key in ("run_id", "tool_id", "tool_version", "timestamp_utc"):
        if not isinstance(run.get(key), str) or not run[key]:
            return False, f"bad_run_{key}"
    inputs_hash = run.get("inputs_hash")
    if inputs_hash is not None and not _is_sha256_hex(inputs_hash):
        return False, "bad_run_inputs_hash"

    git = spine["git"]
    if not isinstance(git.get("head_before"), str) or len(git["head_before"]) < 7:
        return False, "bad_git_head_before"
    if not isinstance(git.get("head_after"), str) or len(git["head_after"]) < 7:
        return False, "bad_git_head_after"
    if not isinstance(git.get("dirty_before"), bool):
        return False, "bad_git_dirty_before"

    artifacts = spine["artifacts"]
    for key in ("run_report", "snapshot", "delta"):
        ref = artifacts.get(key)
        if not isinstance(ref, dict):
            return False, f"bad_artifact_ref_{key}"
        if not isinstance(ref.get("path"), str) or not ref["path"]:
            return False, f"bad_artifact_ref_path_{key}"
        if not _is_sha256_hex(ref.get("sha256")):
            return False, f"bad_artifact_ref_sha_{key}"

    exhibits = artifacts.get("optional_exhibits", [])
    if exhibits is None:
        exhibits = []
    if not isinstance(exhibits, list):
        return False, "bad_optional_exhibits"
    for idx, ref in enumerate(exhibits):
        if not isinstance(ref, dict):
            return False, f"bad_optional_exhibit_{idx}"
        if not isinstance(ref.get("path"), str) or not ref["path"]:
            return False, f"bad_optional_exhibit_path_{idx}"
        if not _is_sha256_hex(ref.get("sha256")):
            return False, f"bad_optional_exhibit_sha_{idx}"

    result = spine["result"]
    status = result.get("status")
    fail_records = result.get("fail_records")
    if status not in {"ok", "partial", "fail"}:
        return False, "bad_result_status"
    if not isinstance(fail_records, list):
        return False, "bad_fail_records"
    for idx, rec in enumerate(fail_records):
        if not isinstance(rec, dict):
            return False, f"bad_fail_record_{idx}"
        for key in ("move", "fail_type", "invariant_diff"):
            if key not in rec:
                return False, f"bad_fail_record_{idx}_missing_{key}"
        if not isinstance(rec["move"], str) or not rec["move"]:
            return False, f"bad_fail_record_{idx}_move"
        if not isinstance(rec["fail_type"], str) or not rec["fail_type"]:
            return False, f"bad_fail_record_{idx}_fail_type"
        if not isinstance(rec["invariant_diff"], dict):
            return False, f"bad_fail_record_{idx}_invariant_diff"

    manifest = spine["witness_manifest"]
    if manifest.get("schema_version") != "QA_SHA256_MANIFEST.v1":
        return False, "bad_manifest_schema_version"
    if not isinstance(manifest.get("generated_utc"), str) or not manifest["generated_utc"]:
        return False, "bad_manifest_generated_utc"
    if not isinstance(manifest.get("entries"), list) or not manifest["entries"]:
        return False, "bad_manifest_entries"
    if not _is_sha256_hex(manifest.get("manifest_sha256")):
        return False, "bad_manifest_sha256"
    for idx, entry in enumerate(manifest["entries"]):
        if not isinstance(entry, dict):
            return False, f"bad_manifest_entry_{idx}"
        if not isinstance(entry.get("id"), str) or not entry["id"]:
            return False, f"bad_manifest_entry_{idx}_id"
        if not _is_sha256_hex(entry.get("sha256")):
            return False, f"bad_manifest_entry_{idx}_sha256"
        if entry.get("canonicalization") not in {"json_sorted_no_ws", "raw_bytes"}:
            return False, f"bad_manifest_entry_{idx}_canonicalization"

    return True, "ok"


def _schema_checks_if_available(spine: Dict[str, Any], schema_dir: Path) -> Tuple[bool, str]:
    schema_path = schema_dir / "QA_RUN_ARTIFACT_BUNDLE.v1.schema.json"
    if not schema_path.exists():
        return False, "missing_schema_file"
    # Ensure schema files exist and are parseable regardless of jsonschema availability.
    try:
        main = _read_json(schema_path)
        ref_manifest = _read_json(schema_dir / "QA_SHA256_MANIFEST.v1.schema.json")
        ref_fail = _read_json(schema_dir / "FAIL_RECORD.v1.schema.json")
    except Exception as exc:
        return False, f"schema_load_failed:{type(exc).__name__}:{exc}"

    try:
        import jsonschema  # type: ignore
    except Exception:
        # Manual checks remain authoritative when jsonschema is not installed.
        return True, "jsonschema_not_installed_manual_checks_used"

    try:
        jsonschema.Draft7Validator.check_schema(main)
        jsonschema.Draft7Validator.check_schema(ref_manifest)
        jsonschema.Draft7Validator.check_schema(ref_fail)
    except Exception as exc:
        return False, f"schema_validation_failed:{type(exc).__name__}:{exc}"
    return True, "ok"


def _manifest_entries(spine: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    manifest = spine["witness_manifest"]
    return manifest.get("entries", [])


def _check_manifest_hash(spine: Dict[str, Any]) -> Tuple[bool, str]:
    manifest = spine["witness_manifest"]
    hash_input = {
        "schema_version": manifest.get("schema_version"),
        "generated_utc": manifest.get("generated_utc"),
        "entries": manifest.get("entries", []),
    }
    canonical = json.dumps(hash_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()
    if manifest.get("manifest_sha256") != expected:
        return False, "manifest_sha256_mismatch"
    return True, "ok"


def _check_manifest_files(cert_dir: Path, spine: Dict[str, Any]) -> Tuple[bool, str]:
    for entry in _manifest_entries(spine):
        rel = entry["id"]
        path = cert_dir / rel
        if not path.exists():
            return False, f"manifest_missing_file:{rel}"
        computed = _sha256_file(path)
        if computed != entry["sha256"]:
            return False, f"manifest_hash_mismatch:{rel}"
    return True, "ok"


def _check_artifact_refs(cert_dir: Path, spine: Dict[str, Any]) -> Tuple[bool, str]:
    refs = [
        spine["artifacts"]["run_report"],
        spine["artifacts"]["snapshot"],
        spine["artifacts"]["delta"],
    ]
    refs.extend(spine["artifacts"].get("optional_exhibits", []))
    for ref in refs:
        rel = ref["path"]
        path = cert_dir / rel
        if not path.exists():
            return False, f"artifact_missing:{rel}"
        computed = _sha256_file(path)
        if computed != ref["sha256"]:
            return False, f"artifact_hash_mismatch:{rel}"
    return True, "ok"


def _check_result_semantics(spine: Dict[str, Any]) -> Tuple[bool, str]:
    status = spine["result"]["status"]
    fail_records = spine["result"]["fail_records"]
    if status == "ok" and fail_records:
        return False, "status_ok_but_fail_records_nonempty"
    if status in {"partial", "fail"} and not fail_records:
        return False, "status_not_ok_but_fail_records_empty"
    return True, "ok"


def validate_cert(cert_dir: Path, schema_dir: Path) -> Tuple[bool, str]:
    spine_path = cert_dir / "spine.json"
    if not spine_path.exists():
        return False, "missing_spine"
    try:
        spine = _read_json(spine_path)
    except Exception as exc:
        return False, f"bad_spine_json:{type(exc).__name__}:{exc}"
    if not isinstance(spine, dict):
        return False, "spine_not_object"

    ok, msg = _manual_shape_checks(spine)
    if not ok:
        return ok, msg

    ok, msg = _schema_checks_if_available(spine, schema_dir)
    if not ok:
        return ok, msg

    ok, msg = _check_manifest_hash(spine)
    if not ok:
        return ok, msg

    ok, msg = _check_manifest_files(cert_dir, spine)
    if not ok:
        return ok, msg

    ok, msg = _check_artifact_refs(cert_dir, spine)
    if not ok:
        return ok, msg

    ok, msg = _check_result_semantics(spine)
    if not ok:
        return ok, msg

    return True, "ok"


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print("usage: qa_run_artifact_validate.py <cert_dir> <schema_dir>", file=sys.stderr)
        return 2
    cert_dir = Path(argv[1]).resolve()
    schema_dir = Path(argv[2]).resolve()
    ok, msg = validate_cert(cert_dir, schema_dir)
    if ok:
        print("OK")
        return 0
    print(f"FAIL:{msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
