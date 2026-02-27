"""Minimal POC tools following the AG2-style Python function pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


# --- Tools ---
def read_file(path: str) -> str:
    """Read local text file contents."""
    return Path(path).read_text(encoding="utf-8")


def send_webhook(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stub webhook tool for the POC.
    Middleware should block this before execution unless explicitly allowed.
    """
    return {"status": "WOULD_SEND", "url": url, "payload": payload}

