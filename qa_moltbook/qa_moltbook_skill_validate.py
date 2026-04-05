#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parent
META_VALIDATOR = REPO_ROOT / "qa_alphageometry_ptolemy" / "qa_meta_validator.py"
PUBLISHER = PROJECT_ROOT / "qa_moltbook_publisher.py"
OUT_DIR = PROJECT_ROOT / "moltbook_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 25 * 1024 * 1024


def _read_request() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_validator(target: Path, timeout_s: int) -> Dict[str, Any]:
    if not META_VALIDATOR.exists():
        raise FileNotFoundError(f"Missing validator: {META_VALIDATOR}")

    cmd = ["python3", str(META_VALIDATOR), str(target)]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "command": cmd,
            "exit_code": proc.returncode,
            "elapsed_ms": elapsed_ms,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "command": cmd,
            "exit_code": 124,
            "elapsed_ms": elapsed_ms,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nTIMEOUT",
            "timed_out": True,
        }


def _parse_validator_json(stdout: str) -> Optional[Dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None


def _clean_native_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "UNKNOWN":
        return None
    return text


def _publish(compact_result: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    if not PUBLISHER.exists():
        return {"ok": False, "error": "PUBLISHER_NOT_FOUND", "path": str(PUBLISHER)}

    status_text = "PASS" if compact_result["ok"] else "FAIL"
    payload = {
        "title": f"Verification {status_text}",
        "content": (
            f"{status_text}\n\n"
            f"- type: `{compact_result.get('certificate_type')}`\n"
            f"- id: `{compact_result.get('certificate_id')}`\n"
            f"- artifact: `{Path(compact_result.get('artifact_path', '')).name}`\n"
            f"- sha256: `{compact_result.get('content_hash')}`\n"
            f"- fail_type: `{compact_result.get('fail_type')}`\n"
            f"- result: `{compact_result.get('full_result_path')}`"
        ),
    }

    try:
        proc = subprocess.run(
            ["python3", str(PUBLISHER), "--stdin", "--timeout-s", str(timeout_s)],
            cwd=str(PROJECT_ROOT),
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s + 5,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        parsed_out = None
        if out:
            try:
                parsed_out = json.loads(out)
            except json.JSONDecodeError:
                parsed_out = {"raw_stdout": out[:3000]}
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": parsed_out,
            "stderr": err[:3000],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "PUBLISH_TIMEOUT"}


def _validate_request(req: Dict[str, Any]) -> Dict[str, Any]:
    target_raw = req.get("target")
    claimed_type = req.get("claimed_type")
    mode = req.get("mode", "validate")
    timeout_s = int(req.get("timeout_s", 180))

    if not target_raw or not isinstance(target_raw, str):
        return {"ok": False, "error": "BAD_REQUEST", "detail": "Missing target"}
    if not claimed_type or not isinstance(claimed_type, str):
        return {"ok": False, "error": "BAD_REQUEST", "detail": "Missing claimed_type"}
    if mode not in ("validate", "attest"):
        return {"ok": False, "error": "UNSUPPORTED_MODE", "mode": mode}
    if target_raw.startswith("http://") or target_raw.startswith("https://"):
        return {"ok": False, "error": "REMOTE_TARGET_NOT_SUPPORTED", "target": target_raw}
    if timeout_s <= 0 or timeout_s > 900:
        return {"ok": False, "error": "BAD_TIMEOUT", "timeout_s": timeout_s}
    return {"ok": True}


def main() -> int:
    try:
        req = _read_request()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "BAD_JSON", "detail": str(exc)}, sort_keys=True))
        return 2

    req_check = _validate_request(req)
    if not req_check["ok"]:
        print(json.dumps(req_check, sort_keys=True))
        return 2

    target_raw = str(req["target"])
    claimed_type = str(req["claimed_type"])
    mode = str(req.get("mode", "validate"))
    timeout_s = int(req.get("timeout_s", 180))
    publish_flag = bool(req.get("publish", False))
    source_url = req.get("source_url")

    target_path = Path(target_raw)
    if not target_path.is_absolute():
        target_path = (REPO_ROOT / target_path).resolve()
    else:
        target_path = target_path.resolve()

    if not target_path.exists():
        print(json.dumps({"ok": False, "error": "TARGET_NOT_FOUND", "target": str(target_path)}, sort_keys=True))
        return 2
    if not target_path.is_file():
        print(json.dumps({"ok": False, "error": "TARGET_NOT_FILE", "target": str(target_path)}, sort_keys=True))
        return 2

    size_bytes = target_path.stat().st_size
    if size_bytes > MAX_BYTES:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "TARGET_TOO_LARGE",
                    "bytes": size_bytes,
                    "max_bytes": MAX_BYTES,
                },
                sort_keys=True,
            )
        )
        return 2

    artifact_hash = _sha256_file(target_path)
    file_size_bytes = target_path.stat().st_size
    run: Optional[Dict[str, Any]] = None
    parsed_validator: Optional[Dict[str, Any]] = None

    result_ok = True
    native_is_valid: Optional[bool] = None
    fail_type = None
    fail_details: Dict[str, Any] = {}
    native_certificate_type = None
    native_certificate_id = None
    native_content_hash = None
    native_content_hash_full = None

    if mode == "validate":
        run = _run_validator(target_path, timeout_s)
        parsed_validator = _parse_validator_json(run["stdout"])

        result_ok = (run["exit_code"] == 0)
        if parsed_validator is not None and isinstance(parsed_validator.get("is_valid"), bool):
            native_is_valid = bool(parsed_validator["is_valid"])
        result_ok = result_ok and (native_is_valid is not False)

        if parsed_validator is not None:
            native_fail_type = _clean_native_value(parsed_validator.get("fail_type"))
            fail_type = native_fail_type
            if "issues" in parsed_validator and isinstance(parsed_validator["issues"], list):
                fail_details["issue_count"] = len(parsed_validator["issues"])
                fail_details["issues_head"] = parsed_validator["issues"][:5]
            if "barrier" in parsed_validator:
                fail_details["barrier"] = parsed_validator["barrier"]
            native_certificate_type = _clean_native_value(parsed_validator.get("certificate_type"))
            native_certificate_id = _clean_native_value(parsed_validator.get("certificate_id"))
            native_content_hash = _clean_native_value(parsed_validator.get("content_hash"))
            native_content_hash_full = _clean_native_value(parsed_validator.get("content_hash_full"))

        if run.get("timed_out"):
            fail_type = fail_type or "timeout"
            fail_details["timed_out"] = True
        elif not result_ok:
            fail_type = fail_type or "validator_nonzero_exit"
            fail_details["exit_code"] = run["exit_code"]
        if not result_ok:
            fail_type = fail_type or "unknown_failure"

    deterministic_certificate_type = claimed_type
    deterministic_certificate_id = f"sha256:{artifact_hash}"

    compact = {
        "ok": result_ok,
        "mode": mode,
        "claimed_type": claimed_type,
        "certificate_type": deterministic_certificate_type,
        "certificate_id": deterministic_certificate_id,
        "native_certificate_type": native_certificate_type,
        "native_certificate_id": native_certificate_id,
        "native_is_valid": native_is_valid,
        "content_hash": artifact_hash,
        "file_size_bytes": file_size_bytes,
        "source_url": source_url,
        "validator_content_hash": native_content_hash,
        "validator_content_hash_full": native_content_hash_full,
        "fail_type": fail_type,
        "fail_details": fail_details,
        "repro_cmd": (
            f"python3 qa_alphageometry_ptolemy/qa_meta_validator.py {target_path}"
            if mode == "validate"
            else f"sha256sum {target_path}"
        ),
        "artifact_path": str(target_path),
        "validator": "qa_alphageometry_ptolemy/qa_meta_validator.py" if mode == "validate" else None,
        "elapsed_ms": int(run["elapsed_ms"]) if run else 0,
        "publish": None,
        "full_result_path": str(OUT_DIR / f"verify_{artifact_hash[:16]}.json"),
    }

    full_record = {
        "timestamp_utc": _now_utc(),
        "request": {
            "target": req.get("target"),
            "claimed_type": req.get("claimed_type"),
            "mode": mode,
            "publish": publish_flag,
            "note": req.get("note"),
            "source_url": source_url,
            "timeout_s": timeout_s,
        },
        "result": compact,
        "validator_run": (
            {
                "command": run["command"],
                "exit_code": run["exit_code"],
                "timed_out": run["timed_out"],
                "stdout": run["stdout"],
                "stderr": run["stderr"],
            }
            if run
            else None
        ),
        "validator_parsed": parsed_validator,
    }

    out_path = Path(compact["full_result_path"])
    out_path.write_text(json.dumps(full_record, indent=2, sort_keys=True), encoding="utf-8")

    if publish_flag:
        compact["publish"] = _publish(compact, timeout_s=min(timeout_s, 30))

    print(json.dumps(compact, sort_keys=True))
    return 0 if result_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
