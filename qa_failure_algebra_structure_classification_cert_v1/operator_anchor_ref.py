"""Reference utilities for Family [88] classification cert.

Loads canonical Family [87] compose table payload used as a default reference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_FAMILY87_CERT_REL = "qa_failure_compose_operator_cert_v1/fixtures/pass_feedback_escalation.json"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def load_family87_reference(repo_root: Path, rel_path: str = DEFAULT_FAMILY87_CERT_REL) -> Dict[str, Any]:
    cert_path = (repo_root / rel_path).resolve()
    with cert_path.open("r", encoding="utf-8") as handle:
        cert = json.load(handle)
    return {
        "path": rel_path,
        "cert_sha256": canonical_sha256(cert),
        "carrier": cert.get("carrier", []),
        "forms": cert.get("forms", []),
        "compose_table": cert.get("compose_table", []),
    }
