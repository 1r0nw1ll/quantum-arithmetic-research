from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None


GENESIS_CHAIN_HASH = "0" * 64
DEFAULT_LEDGER_PATH = "/tmp/qa_dashboard_runs/audit_ledger.jsonl"


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()


def _read_last_nonempty_line(path: Path, *, tail_bytes: int = 65536) -> Optional[str]:
    if not path.exists():
        return None
    size = path.stat().st_size
    if size == 0:
        return None
    with path.open("rb") as f:
        start = max(0, size - tail_bytes)
        f.seek(start)
        data = f.read()
    lines = data.splitlines()
    for raw in reversed(lines):
        if raw.strip():
            try:
                return raw.decode("utf-8")
            except Exception:
                return None
    return None


@dataclass(frozen=True)
class LedgerAppendResult:
    record: Dict[str, Any]
    chain_hash: str
    prev_chain_hash: str


class AuditLedger:
    """Append-only, hash-chained JSONL transparency log for dashboard runs.

    Each line is a JSON object with:
      - prev_chain_hash
      - chain_hash
      - entry (the event payload)
      - entry_json (canonical JSON string)
    """

    def __init__(self, path: Path):
        self.path = path

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_prev_chain_hash(self) -> str:
        last = _read_last_nonempty_line(self.path)
        if not last:
            return GENESIS_CHAIN_HASH
        try:
            obj = json.loads(last)
            if isinstance(obj, dict) and isinstance(obj.get("chain_hash"), str) and len(obj["chain_hash"]) == 64:
                return obj["chain_hash"]
        except Exception:
            pass
        return GENESIS_CHAIN_HASH

    def append(self, entry: Dict[str, Any]) -> LedgerAppendResult:
        self._ensure_parent()

        entry_with_time = dict(entry)
        entry_with_time.setdefault("created_utc", _utc_now_compact())

        entry_json = _canonical_json_compact(entry_with_time)

        # We need prev hash and append to be a single critical section.
        with self.path.open("a+", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)

            prev_hash = self._read_prev_chain_hash()
            chain_hash = _sha256_hex(prev_hash + "\n" + entry_json)

            record = {
                "prev_chain_hash": prev_hash,
                "chain_hash": chain_hash,
                "entry": entry_with_time,
                "entry_json": entry_json,
            }
            f.write(_canonical_json_compact(record) + "\n")
            f.flush()

            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)

        return LedgerAppendResult(record=record, chain_hash=chain_hash, prev_chain_hash=prev_hash)

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj

    def verify(self, *, max_records: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Recompute chain hashes from the beginning."""
        expected_prev = GENESIS_CHAIN_HASH
        count = 0
        for i, rec in enumerate(self.iter_records()):
            if max_records is not None and count >= max_records:
                break
            count += 1

            prev = rec.get("prev_chain_hash")
            ch = rec.get("chain_hash")
            entry_json = rec.get("entry_json")
            if not (isinstance(prev, str) and isinstance(ch, str) and isinstance(entry_json, str)):
                return False, {"ok": False, "error": "record_missing_fields", "index": i}
            if prev != expected_prev:
                return False, {"ok": False, "error": "prev_chain_hash_mismatch", "index": i}
            recomputed = _sha256_hex(expected_prev + "\n" + entry_json)
            if recomputed != ch:
                return False, {"ok": False, "error": "chain_hash_mismatch", "index": i}
            expected_prev = ch

        return True, {"ok": True, "records": count, "last_chain_hash": expected_prev}


_ledger_path = Path(os.environ.get("QA_AUDIT_LEDGER_PATH", DEFAULT_LEDGER_PATH))
ledger = AuditLedger(_ledger_path)

