# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 rate limit; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
"""QA-KG Phase 6 rate limit — per-session write counter.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Counter state lives in `qa_alphageometry_ptolemy/_agent_writes.json`,
a parallel file to `_meta_ledger.json`. Separate flock, separate
writers — the meta-validator sweep writer never touches this file, and
this module never touches the meta-ledger. No shared-state race
(plan v2 M1).

Ledger shape:

    {
      "<session-id>": {"count": 12, "first_ts": "...", "last_ts": "..."}
    }

Increment raises FirewallViolation at MAX_WRITES_PER_SESSION (default
50, overridable via environment for fixtures). decay_on_session_done
removes the session entry entirely; reset_session is the manual
recovery path for crashed sessions that never broadcast session_done.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import fcntl
import json
import os
import time
from pathlib import Path


MAX_WRITES_PER_SESSION = int(os.environ.get("QA_KG_MCP_MAX_WRITES", "50"))

_REPO = Path(__file__).resolve().parents[2]
_META_DIR = _REPO / "qa_alphageometry_ptolemy"
_AGENT_WRITES_PATH = _META_DIR / "_agent_writes.json"


class RateLimitExceeded(RuntimeError):
    """Session exceeded MAX_WRITES_PER_SESSION promote calls."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_path() -> Path:
    return _AGENT_WRITES_PATH


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _atomic_write(path: Path, data: dict) -> None:
    """Write-to-tmp + os.replace. Atomic on POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class _FileLock:
    """Context-manager wrapping fcntl.flock on a sibling .lock file.

    A sibling lock file is used (rather than locking the ledger itself)
    so we don't race with os.replace swapping the ledger inode out from
    under an open lock fd. The lock fd stays tied to the lock path's
    inode for the entire critical section.
    """
    def __init__(self, target: Path):
        self._lock_path = target.with_suffix(target.suffix + ".lock")
        self._fd: int | None = None

    def __enter__(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            str(self._lock_path), os.O_WRONLY | os.O_CREAT, 0o644,
        )
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        return False


def increment(session: str, *, ledger_path: Path | None = None,
              max_writes: int | None = None) -> int:
    """Atomic increment. Raises RateLimitExceeded at the cap.

    Returns the new count.
    """
    if not session:
        raise ValueError("increment requires non-empty session")
    path = ledger_path or _default_path()
    cap = max_writes if max_writes is not None else MAX_WRITES_PER_SESSION
    with _FileLock(path):
        data = _load(path)
        entry = data.get(session) or {
            "count": 0, "first_ts": _now(), "last_ts": "",
        }
        if entry["count"] >= cap:
            raise RateLimitExceeded(
                f"rate_limit_exceeded session={session} "
                f"count={entry['count']} cap={cap}"
            )
        entry["count"] = int(entry["count"]) + 1
        entry["last_ts"] = _now()
        data[session] = entry
        _atomic_write(path, data)
        return int(entry["count"])


def decay_on_session_done(session: str, *,
                          ledger_path: Path | None = None) -> None:
    """Remove the session entry. Called from MCP server on session_done."""
    if not session:
        return
    path = ledger_path or _default_path()
    with _FileLock(path):
        data = _load(path)
        if session in data:
            data.pop(session)
            _atomic_write(path, data)


def reset_session(session: str, *,
                  ledger_path: Path | None = None) -> bool:
    """Operator-authorized manual reset (the CLI path).

    Returns True if a session was found and removed, False otherwise.
    Identical behavior to decay_on_session_done but kept as a separate
    entry point so operational audit can distinguish the source.
    """
    if not session:
        return False
    path = ledger_path or _default_path()
    with _FileLock(path):
        data = _load(path)
        if session in data:
            data.pop(session)
            _atomic_write(path, data)
            return True
    return False


def get_count(session: str, *, ledger_path: Path | None = None) -> int:
    """Read-only count lookup. Useful for introspection + tests."""
    path = ledger_path or _default_path()
    data = _load(path)
    entry = data.get(session) or {}
    return int(entry.get("count", 0))
