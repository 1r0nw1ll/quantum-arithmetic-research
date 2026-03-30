#!/usr/bin/env python3
"""
Ingest an Open Brain JSONL migration queue via the deployed MCP HTTP endpoint.

This script is intentionally dependency-free (stdlib only) and uses `curl` for HTTP.

Example:
  python3 tools/open_brain_ingest_queue_mcp.py \
    --queue Documents/open_brain_migration_qalabvault/queue.jsonl \
    --start-line 11 --end-line 20 --parallel 10
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_URL = "https://bepguekxksbgiqleadvq.supabase.co/functions/v1/open-brain-mcp"


@dataclasses.dataclass(frozen=True)
class QueueItem:
    line_no: int
    chunk_id: str
    path: str
    heading: str
    text: str
    migration_run_id: str
    source: str


@dataclasses.dataclass(frozen=True)
class IngestResult:
    line_no: int
    chunk_id: str
    heading: str
    status: str  # captured | duplicate | error
    capture_id: Optional[str]
    created_at: Optional[str]
    message: str


CAPTURE_RE = re.compile(r"Captured\s+\((?P<id>[^)]+)\)\s+at\s+(?P<at>.+)$")
DUP_RE = re.compile(r"Duplicate\s+\(already captured\):\s+(?P<id>\S+)\s+at\s+(?P<at>.+)$")


def _read_token() -> str:
    token = os.environ.get("OPEN_BRAIN_TOKEN", "").strip()
    if token:
        return token
    p = Path.home() / ".open_brain_mcp_key"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("Missing OPEN_BRAIN_TOKEN and ~/.open_brain_mcp_key not found.")


def _json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _curl_json(
    url: str,
    token: str,
    payload: dict[str, Any],
    *,
    connect_timeout_s: float,
    max_time_s: float,
    retries: int,
    retry_delay_s: float,
    retry_max_time_s: float,
) -> str:
    # We send BOTH auth forms because different client setups exist in the wild:
    # - docs recommend `x-brain-key`
    # - older configs use `Authorization: Bearer ...`
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "--fail-with-body",
            url,
            "-H",
            f"Authorization: Bearer {token}",
            "-H",
            f"x-brain-key: {token}",
            "-H",
            "Content-Type: application/json",
            "-H",
            "Accept: application/json, text/event-stream",
            "--connect-timeout",
            str(connect_timeout_s),
            "--max-time",
            str(max_time_s),
            "--retry",
            str(int(max(0, retries))),
            "--retry-all-errors",
            "--retry-delay",
            str(retry_delay_s),
            "--retry-max-time",
            str(retry_max_time_s),
            "--data-binary",
            "@-",
        ],
        input=_json_dumps_canonical(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        if stdout:
            raise RuntimeError(f"curl failed ({proc.returncode}): {stderr} | body={stdout[:2000]}")
        raise RuntimeError(f"curl failed ({proc.returncode}): {stderr}")
    return proc.stdout.decode("utf-8", errors="replace")


def _sse_last_data_json(sse_text: str) -> dict[str, Any]:
    last: Optional[dict[str, Any]] = None
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].lstrip()
        if not raw or raw.strip() == "[DONE]":
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            last = d
    if last is None:
        raise ValueError("No SSE data: JSON line found")
    return last


def _mcp_initialize(
    url: str,
    token: str,
    *,
    connect_timeout_s: float,
    max_time_s: float,
    retries: int,
    retry_delay_s: float,
    retry_max_time_s: float,
) -> None:
    _curl_json(
        url,
        token,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "open-brain-migrate", "version": "0.1"},
            },
        },
        connect_timeout_s=connect_timeout_s,
        max_time_s=max_time_s,
        retries=retries,
        retry_delay_s=retry_delay_s,
        retry_max_time_s=retry_max_time_s,
    )
    # The server doesn't return anything useful for this notification; best-effort.
    try:
        _curl_json(
            url,
            token,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            connect_timeout_s=connect_timeout_s,
            max_time_s=max_time_s,
            retries=retries,
            retry_delay_s=retry_delay_s,
            retry_max_time_s=retry_max_time_s,
        )
    except Exception:
        pass


def _parse_queue_items(queue_path: Path, start_line: int, end_line: int) -> list[QueueItem]:
    items: list[QueueItem] = []
    with queue_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx < start_line:
                continue
            if idx > end_line:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            source = str(obj.get("source") or "")
            path = obj.get("path") or obj.get("source_path") or ""
            heading = obj.get("heading") or obj.get("source_heading") or path
            items.append(
                QueueItem(
                    line_no=idx,
                    chunk_id=str(obj["chunk_id"]),
                    path=str(path),
                    heading=str(heading),
                    text=str(obj["text"]),
                    migration_run_id=str(obj.get("migration_run_id") or ""),
                    source=source,
                )
            )
    return items


def _extract_text_from_mcp_result(d: dict[str, Any]) -> str:
    # Typical MCP tool call result:
    # {"jsonrpc":"2.0","id":...,"result":{"content":[{"type":"text","text":"..."}]}}
    result = d.get("result")
    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return ""
    first = content[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("text", "")).strip()


def _call_capture_thought(
    url: str,
    token: str,
    item: QueueItem,
    source_override: str,
    *,
    connect_timeout_s: float,
    max_time_s: float,
    retries: int,
    retry_delay_s: float,
    retry_max_time_s: float,
    parse_retries: int,
) -> IngestResult:
    source = source_override.strip() if source_override.strip() else (item.source.strip() or "migration")
    payload = {
        "jsonrpc": "2.0",
        "id": item.line_no,
        "method": "tools/call",
        "params": {
            "name": "capture_thought",
            "arguments": {
                "content": item.text,
                "source": source,
                "source_path": item.path,
                "source_heading": item.heading,
                "migration_chunk_id": item.chunk_id,
                "migration_run_id": item.migration_run_id,
            },
        },
    }

    try:
        last_err: Optional[Exception] = None
        for attempt in range(max(0, int(parse_retries)) + 1):
            try:
                sse = _curl_json(
                    url,
                    token,
                    payload,
                    connect_timeout_s=connect_timeout_s,
                    max_time_s=max_time_s,
                    retries=retries,
                    retry_delay_s=retry_delay_s,
                    retry_max_time_s=retry_max_time_s,
                )
                d = _sse_last_data_json(sse)
                break
            except Exception as e:
                last_err = e
                if attempt >= max(0, int(parse_retries)):
                    raise
                time.sleep(0.2 * (attempt + 1))
        else:
            raise RuntimeError(f"unreachable: last_err={last_err!r}")

        if "error" in d:
            err_obj = d.get("error")
            return IngestResult(
                line_no=item.line_no,
                chunk_id=item.chunk_id,
                heading=item.heading,
                status="error",
                capture_id=None,
                created_at=None,
                message=_json_dumps_canonical(err_obj) if isinstance(err_obj, (dict, list)) else str(err_obj),
            )

        text = _extract_text_from_mcp_result(d)

        m = CAPTURE_RE.match(text)
        if m:
            return IngestResult(
                line_no=item.line_no,
                chunk_id=item.chunk_id,
                heading=item.heading,
                status="captured",
                capture_id=m.group("id"),
                created_at=m.group("at"),
                message=text,
            )

        m = DUP_RE.match(text)
        if m:
            return IngestResult(
                line_no=item.line_no,
                chunk_id=item.chunk_id,
                heading=item.heading,
                status="duplicate",
                capture_id=m.group("id"),
                created_at=m.group("at"),
                message=text,
            )

        # Common server-side failure mode during high-volume ingests.
        if text.startswith("Supabase insert failed:") or text.startswith("OpenRouter"):
            return IngestResult(
                line_no=item.line_no,
                chunk_id=item.chunk_id,
                heading=item.heading,
                status="error",
                capture_id=None,
                created_at=None,
                message=text[:4000],
            )

        # If we have a non-empty MCP response but can't interpret it, surface structured context.
        if not text:
            return IngestResult(
                line_no=item.line_no,
                chunk_id=item.chunk_id,
                heading=item.heading,
                status="error",
                capture_id=None,
                created_at=None,
                message=f"Empty response text; last_data={_json_dumps_canonical(d)[:2000]}",
            )

        return IngestResult(
            line_no=item.line_no,
            chunk_id=item.chunk_id,
            heading=item.heading,
            status="error",
            capture_id=None,
            created_at=None,
            message=f"Unrecognized response: {text[:2000]}",
        )
    except Exception as e:
        return IngestResult(
            line_no=item.line_no,
            chunk_id=item.chunk_id,
            heading=item.heading,
            status="error",
            capture_id=None,
            created_at=None,
            message=str(e),
        )


def _print_table(results: list[IngestResult]) -> None:
    rows = []
    for r in results:
        cid_short = r.chunk_id[:10]
        cap = (r.capture_id or "")[:12]
        head = r.heading.replace("\n", " ")
        if len(head) > 60:
            head = head[:57] + "..."
        rows.append((str(r.line_no), cid_short, r.status, cap, head))

    headers = ("line", "chunk_id", "status", "capture_id", "heading")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        # Local (no-network) smoke tests.
        sse = "\n".join(
            [
                "event: message",
                'data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"hello"}]}}',
                "",
                "event: message",
                'data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Captured (abc) at 2026-01-01"}]}}',
            ]
        )
        d = _sse_last_data_json(sse)
        assert _extract_text_from_mcp_result(d).startswith("Captured ("), "SSE last data parse failed"
        print(_json_dumps_canonical({"ok": True}))
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="", help="Path to queue.jsonl (required unless --tool is set)")
    ap.add_argument("--start-line", type=int, default=0)
    ap.add_argument("--end-line", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument(
        "--source",
        default="obsidian_migration",
        help='Value for metadata.source (default: "obsidian_migration"). If empty, uses item.source.',
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only print batch summary (recommended for large ingests).",
    )
    ap.add_argument("--url", default=os.environ.get("OPEN_BRAIN_URL", DEFAULT_URL))
    ap.add_argument("--write-results", default="", help="Optional path to write results JSONL")
    ap.add_argument("--curl-connect-timeout", type=float, default=10.0)
    ap.add_argument("--curl-max-time", type=float, default=120.0)
    ap.add_argument("--curl-retries", type=int, default=6)
    ap.add_argument("--curl-retry-delay", type=float, default=1.0)
    ap.add_argument("--curl-retry-max-time", type=float, default=120.0)
    ap.add_argument(
        "--parse-retries",
        type=int,
        default=2,
        help="Extra retries when SSE parsing yields no usable JSON (in addition to curl retries).",
    )
    ap.add_argument("--tool", default="", help="Call a tool by name and exit (bypasses queue ingest)")
    ap.add_argument(
        "--tool-args",
        default="{}",
        help='JSON object string passed as tool arguments (default: "{}")',
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run local (no-network) smoke tests and exit 0 on success.",
    )
    args = ap.parse_args(argv)

    def call_tool(name: str, tool_args: dict[str, Any]) -> str:
        token = _read_token()
        _mcp_initialize(
            args.url,
            token,
            connect_timeout_s=float(args.curl_connect_timeout),
            max_time_s=float(args.curl_max_time),
            retries=int(args.curl_retries),
            retry_delay_s=float(args.curl_retry_delay),
            retry_max_time_s=float(args.curl_retry_max_time),
        )
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": tool_args},
        }
        sse = _curl_json(
            args.url,
            token,
            payload,
            connect_timeout_s=float(args.curl_connect_timeout),
            max_time_s=float(args.curl_max_time),
            retries=int(args.curl_retries),
            retry_delay_s=float(args.curl_retry_delay),
            retry_max_time_s=float(args.curl_retry_max_time),
        )
        d = _sse_last_data_json(sse)
        if "error" in d:
            err_obj = d.get("error")
            msg = _json_dumps_canonical(err_obj) if isinstance(err_obj, (dict, list)) else str(err_obj)
            raise RuntimeError(msg)
        text = _extract_text_from_mcp_result(d)
        if not text:
            raise RuntimeError(f"Empty response text; last_data={_json_dumps_canonical(d)[:2000]}")
        return text

    if str(args.tool).strip():
        name = str(args.tool).strip()
        try:
            tool_args = json.loads(str(args.tool_args))
        except Exception as e:
            print(f"Invalid --tool-args JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(tool_args, dict):
            print("--tool-args must be a JSON object.", file=sys.stderr)
            return 2
        try:
            print(call_tool(name, tool_args))
            return 0
        except Exception as e:
            print(f"Tool call failed: {e}", file=sys.stderr)
            return 1

    if int(args.start_line) < 1 or int(args.end_line) < int(args.start_line):
        print("Invalid line range.", file=sys.stderr)
        return 2

    if not str(args.queue).strip():
        print("Missing --queue (or set --tool to call a tool).", file=sys.stderr)
        return 2

    queue_path = Path(str(args.queue))
    items = _parse_queue_items(queue_path, args.start_line, args.end_line)
    if not items:
        print("No items found in that range.", file=sys.stderr)
        return 2

    token = _read_token()
    _mcp_initialize(
        args.url,
        token,
        connect_timeout_s=float(args.curl_connect_timeout),
        max_time_s=float(args.curl_max_time),
        retries=int(args.curl_retries),
        retry_delay_s=float(args.curl_retry_delay),
        retry_max_time_s=float(args.curl_retry_max_time),
    )

    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as ex:
        futs = [
            ex.submit(
                _call_capture_thought,
                args.url,
                token,
                item,
                str(args.source),
                connect_timeout_s=float(args.curl_connect_timeout),
                max_time_s=float(args.curl_max_time),
                retries=int(args.curl_retries),
                retry_delay_s=float(args.curl_retry_delay),
                retry_max_time_s=float(args.curl_retry_max_time),
                parse_retries=int(args.parse_retries),
            )
            for item in items
        ]
        results = [f.result() for f in futs]

    results.sort(key=lambda r: r.line_no)
    if not args.quiet:
        _print_table(results)

    if args.write_results:
        out_path = Path(args.write_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(_json_dumps_canonical(dataclasses.asdict(r)) + "\n")

    captured = sum(1 for r in results if r.status == "captured")
    duped = sum(1 for r in results if r.status == "duplicate")
    errored = sum(1 for r in results if r.status == "error")
    print(f"\nBatch summary: captured={captured}, duplicate={duped}, error={errored}")

    if errored:
        print("Errors (first 3):", file=sys.stderr)
        for r in [x for x in results if x.status == "error"][:3]:
            print(f"- line {r.line_no} chunk {r.chunk_id[:10]}: {r.message[:200]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
