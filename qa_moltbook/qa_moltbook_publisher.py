from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
API_BASE = os.environ.get("MOLTBOOK_API_BASE", "https://www.moltbook.com/api/v1")


def _credential_candidates() -> list[Path]:
    env_path = os.environ.get("MOLTBOOK_CREDENTIALS_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            Path.cwd() / ".moltbook" / "credentials.json",
            SCRIPT_DIR / ".moltbook" / "credentials.json",
            REPO_ROOT / "gemini_qa_project" / ".moltbook" / "credentials.json",
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def load_key() -> str:
    env_key = os.environ.get("MOLTBOOK_API_KEY", "").strip()
    if env_key:
        return env_key

    for creds_path in _credential_candidates():
        if not creds_path.exists():
            continue
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        key = str(data.get("api_key", "")).strip()
        if key:
            return key

    tried = [str(p) for p in _credential_candidates()]
    raise FileNotFoundError(f"No usable credentials found. Tried: {tried}")


def post(
    title: str,
    content: str,
    *,
    submolt: str = "general",
    timeout_s: int = 15,
    retries: int = 3,
    backoff_s: float = 1.0,
) -> Dict[str, Any]:
    key = load_key()
    url = f"{API_BASE}/posts"

    payload = {
        "submolt": submolt,
        "title": title,
        "content": content,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Transient HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_s * attempt)

    raise RuntimeError(f"Post failed after {retries} attempts: {last_error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a Moltbook post")
    parser.add_argument("--title", help="Post title")
    parser.add_argument("--content", help="Post content")
    parser.add_argument("--submolt", default="general", help="Target submolt")
    parser.add_argument("--timeout-s", type=int, default=15, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON payload from stdin: {title, content, submolt?}",
    )
    return parser.parse_args()


def _load_payload_from_stdin() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("--stdin provided but stdin was empty")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("stdin payload must be a JSON object")
    return payload


def main() -> int:
    args = _parse_args()

    try:
        if args.stdin:
            payload = _load_payload_from_stdin()
            title = str(payload.get("title", "")).strip()
            content = str(payload.get("content", "")).strip()
            submolt = str(payload.get("submolt", args.submolt)).strip() or "general"
        else:
            title = str(args.title or "").strip()
            content = str(args.content or "").strip()
            submolt = str(args.submolt).strip() or "general"

        if not title or not content:
            raise ValueError("Both title and content are required")

        result = post(
            title=title,
            content=content,
            submolt=submolt,
            timeout_s=args.timeout_s,
            retries=args.retries,
        )
        print(json.dumps({"ok": True, "response": result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
