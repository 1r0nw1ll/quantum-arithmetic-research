#!/usr/bin/env python3
"""
Prepare a ChatGPT data export (conversations.json inside a zip) for Open Brain ingestion.

This script:
  - Streams the export using `jq --stream` (so it can handle very large exports).
  - Builds a JSONL queue of chunks you can ingest using any MCP client by calling
    `capture_thought` per item.

It does NOT write to Supabase directly.

Example:
  python3 tools/open_brain_prepare_chatgpt_export.py \\
    --zip chat_data/<export>.zip \\
    --out Documents/open_brain_migration_chatgpt \\
    --max-conversations 20 \\
    --max-chunks 200
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class QueueItem:
    chunk_id: str
    migration_run_id: str
    source: str
    source_path: str
    source_heading: str
    text: str


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iso_from_epoch(epoch: Optional[float]) -> str:
    if not epoch:
        return ""
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def run_stream_pairs(zip_path: Path) -> Iterator[Tuple[List[object], object]]:
    """
    Yields (path, value) pairs from `jq --stream` reading conversations.json from the zip.
    """
    unzip_cmd = ["unzip", "-p", str(zip_path), "conversations.json"]
    jq_cmd = ["jq", "-c", "--stream", "."]
    with subprocess.Popen(unzip_cmd, stdout=subprocess.PIPE) as unzip_proc:
        if unzip_proc.stdout is None:
            raise RuntimeError("unzip stdout missing")
        with subprocess.Popen(jq_cmd, stdin=unzip_proc.stdout, stdout=subprocess.PIPE) as jq_proc:
            unzip_proc.stdout.close()
            if jq_proc.stdout is None:
                raise RuntimeError("jq stdout missing")
            for line in jq_proc.stdout:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # jq --stream emits:
                # - [path, value] for scalar leaves
                # - [path] as structural end markers
                if not isinstance(item, list) or len(item) != 2:
                    continue
                path, value = item
                yield path, value

            rc = jq_proc.wait()
            if rc != 0:
                raise RuntimeError(f"jq --stream failed with code {rc}")
        rc = unzip_proc.wait()
        if rc != 0:
            raise RuntimeError(f"unzip failed with code {rc}")


def collect_conversation_index(zip_path: Path) -> Dict[int, Dict[str, object]]:
    """
    First pass: collect lightweight metadata per conversation index.
    Returns conv_idx -> {title, create_time, update_time}.
    """
    conv: Dict[int, Dict[str, object]] = defaultdict(dict)
    for path, value in run_stream_pairs(zip_path):
        if not path or not isinstance(path[0], int):
            continue
        idx = int(path[0])
        if len(path) == 2 and path[1] in ("title", "create_time", "update_time"):
            conv[idx][str(path[1])] = value
    return conv


def pick_conversations_by_update_time(
    conv_index: Dict[int, Dict[str, object]],
    max_conversations: int,
) -> List[int]:
    """
    Picks the most recently updated conversations by numeric update_time.
    """
    scored: List[Tuple[float, int]] = []
    for idx, meta in conv_index.items():
        upd = meta.get("update_time")
        try:
            score = float(upd) if upd is not None else 0.0
        except Exception:
            score = 0.0
        scored.append((score, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:max_conversations]]


@dataclasses.dataclass
class PartialMessage:
    role: Optional[str] = None
    create_time: Optional[float] = None
    content_type: Optional[str] = None
    parts: Dict[int, str] = dataclasses.field(default_factory=dict)

    def materialize_text(self) -> str:
        if self.content_type != "text":
            return ""
        if not self.parts:
            return ""
        return "\n".join(v for _, v in sorted(self.parts.items())).strip()


def should_keep_message(
    *,
    role: str,
    text: str,
    roles: Set[str],
    assistant_regex: Optional[re.Pattern[str]],
) -> bool:
    if role not in roles:
        return False
    if role != "assistant":
        return True
    if assistant_regex is None:
        return True
    return bool(assistant_regex.search(text))


def iter_queue_items_from_zip(
    *,
    zip_path: Path,
    conv_meta: Dict[int, Dict[str, object]],
    selected_conversations: Set[int],
    migration_run_id: str,
    roles: Set[str],
    assistant_filter: str,
    max_chars: int,
    max_chunks: int,
) -> Iterator[QueueItem]:
    assistant_regex = re.compile(assistant_filter, re.IGNORECASE) if assistant_filter else None

    current_idx: Optional[int] = None
    current_messages: Dict[str, PartialMessage] = {}

    def flush(idx: int) -> Iterator[QueueItem]:
        meta = conv_meta.get(idx, {})
        title = str(meta.get("title") or f"conversation_{idx}")
        conv_path = f"chatgpt_export/{title}"
        # materialize messages we kept
        materialized: List[Tuple[float, str, str]] = []
        for msg_id, pm in current_messages.items():
            role = pm.role or ""
            text = pm.materialize_text()
            if not text.strip():
                continue
            if not should_keep_message(role=role, text=text, roles=roles, assistant_regex=assistant_regex):
                continue
            ct = float(pm.create_time) if pm.create_time else 0.0
            materialized.append((ct, msg_id, role + "\n" + text))

        materialized.sort(key=lambda t: (t[0], t[1]))

        emitted = 0
        for ct, msg_id, payload in materialized:
            msg_iso = iso_from_epoch(ct)
            heading = f"{msg_iso} {msg_id}"
            context = f'From ChatGPT export "{title}"'
            if msg_iso:
                context += f" at {msg_iso}"
            full_text = f"{context}:\n\n{payload}"
            # chunk if oversized
            if len(full_text) <= max_chars:
                chunk_id = sha256_hex(normalize_for_hash(full_text))
                yield QueueItem(
                    chunk_id=chunk_id,
                    migration_run_id=migration_run_id,
                    source="chatgpt_export",
                    source_path=conv_path,
                    source_heading=heading,
                    text=full_text,
                )
                emitted += 1
            else:
                # naive chunking by paragraphs
                paras = [p.strip() for p in re.split(r"\n{2,}", full_text) if p.strip()]
                expanded: List[str] = []
                for p in paras:
                    if len(p) <= max_chars:
                        expanded.append(p)
                        continue
                    # Split oversized paragraphs (e.g., tool output blobs) by lines,
                    # falling back to hard slices for extremely long lines.
                    buf: List[str] = []
                    size = 0

                    def flush() -> None:
                        nonlocal buf, size
                        if not buf:
                            return
                        expanded.append("\n".join(buf).strip())
                        buf = []
                        size = 0

                    for ln in p.splitlines():
                        ln = ln.rstrip("\n")
                        if not ln:
                            flush()
                            continue

                        while ln and len(ln) > max_chars:
                            flush()
                            expanded.append(ln[:max_chars].strip())
                            ln = ln[max_chars:]

                        add_len = len(ln) + (1 if buf else 0)
                        if size + add_len > max_chars and buf:
                            flush()
                        buf.append(ln)
                        size += add_len

                    flush()

                paras = [p for p in expanded if p.strip()]
                buf: List[str] = []
                size = 0
                part = 1
                for p in paras:
                    if size + len(p) + 2 > max_chars and buf:
                        chunk_text = "\n\n".join(buf).strip()
                        chunk_id = sha256_hex(normalize_for_hash(chunk_text))
                        yield QueueItem(
                            chunk_id=chunk_id,
                            migration_run_id=migration_run_id,
                            source="chatgpt_export",
                            source_path=conv_path,
                            source_heading=f"{heading} part {part}",
                            text=chunk_text,
                        )
                        emitted += 1
                        part += 1
                        buf = []
                        size = 0
                    buf.append(p)
                    size += len(p) + 2
                if buf:
                    chunk_text = "\n\n".join(buf).strip()
                    chunk_id = sha256_hex(normalize_for_hash(chunk_text))
                    yield QueueItem(
                        chunk_id=chunk_id,
                        migration_run_id=migration_run_id,
                        source="chatgpt_export",
                        source_path=conv_path,
                        source_heading=f"{heading} part {part}",
                        text=chunk_text,
                    )
                    emitted += 1

            if emitted >= max_chunks:
                return

    for path, value in run_stream_pairs(zip_path):
        if not path or not isinstance(path[0], int):
            continue
        idx = int(path[0])
        if idx not in selected_conversations:
            continue

        if current_idx is None:
            current_idx = idx
        if idx != current_idx:
            for item in flush(current_idx):
                yield item
                max_chunks -= 1
                if max_chunks <= 0:
                    return
            current_messages = {}
            current_idx = idx

        # Interested in mapping messages only
        # Path: [idx, "mapping", <node_id>, "message", ...]
        if len(path) < 4 or path[1] != "mapping":
            continue
        node_id = path[2]
        if not isinstance(node_id, str):
            continue

        if len(path) >= 6 and path[3] == "message" and path[4] == "author" and path[5] == "role":
            pm = current_messages.setdefault(node_id, PartialMessage())
            pm.role = str(value) if value is not None else None
            continue
        if len(path) >= 5 and path[3] == "message" and path[4] == "create_time":
            pm = current_messages.setdefault(node_id, PartialMessage())
            try:
                pm.create_time = float(value) if value is not None else None
            except Exception:
                pm.create_time = None
            continue
        if len(path) >= 6 and path[3] == "message" and path[4] == "content" and path[5] == "content_type":
            pm = current_messages.setdefault(node_id, PartialMessage())
            pm.content_type = str(value) if value is not None else None
            continue
        if len(path) >= 7 and path[3] == "message" and path[4] == "content" and path[5] == "parts":
            part_idx = path[6]
            if isinstance(part_idx, int) and isinstance(value, str):
                pm = current_messages.setdefault(node_id, PartialMessage())
                pm.parts[int(part_idx)] = value

    if current_idx is not None:
        yield from flush(current_idx)


def write_jsonl(path: Path, items: Iterable[QueueItem]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(
                json.dumps(
                    {
                        "chunk_id": it.chunk_id,
                        "migration_run_id": it.migration_run_id,
                        "source": it.source,
                        "source_path": it.source_path,
                        "source_heading": it.source_heading,
                        "text": it.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True, help="Path to ChatGPT export zip")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("Documents/open_brain_migration_chatgpt_export"),
        help="Output directory (default: Documents/open_brain_migration_chatgpt_export)",
    )
    ap.add_argument("--max-conversations", type=int, default=20, help="Most-recent conversations to include (default: 20)")
    ap.add_argument("--max-chunks", type=int, default=250, help="Max chunks to emit (default: 250)")
    ap.add_argument("--max-chars", type=int, default=1800, help="Max characters per chunk (default: 1800)")
    ap.add_argument(
        "--roles",
        type=str,
        default="user",
        help='Comma-separated roles to include (default: "user"). Options: user,assistant',
    )
    ap.add_argument(
        "--assistant-filter",
        type=str,
        default=r"\\b(decision|action items|todo|next steps|plan|summary)\\b",
        help="Regex: if including assistant, keep only messages matching this (default: decision/action items/todo/etc.)",
    )
    ap.add_argument("--migration-run-id", type=str, default="", help="Optional run id (default: auto)")
    args = ap.parse_args()

    zip_path: Path = args.zip
    if not zip_path.exists():
        raise SystemExit(f"--zip not found: {zip_path}")

    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    unknown = roles.difference({"user", "assistant"})
    if unknown:
        raise SystemExit(f"Unknown roles: {sorted(unknown)}")

    migration_run_id = args.migration_run_id.strip()
    if not migration_run_id:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        migration_run_id = f"chatgpt_export_{ts}"

    out_dir: Path = args.out
    queue_path = out_dir / "queue.jsonl"
    summary_path = out_dir / "summary.json"

    print("Indexing conversations (pass 1)...")
    conv_index = collect_conversation_index(zip_path)
    selected = pick_conversations_by_update_time(conv_index, args.max_conversations)
    selected_set = set(selected)

    print(f"Selected {len(selected)} conversations by update_time.")
    print("Extracting messages (pass 2)...")
    items = list(
        iter_queue_items_from_zip(
            zip_path=zip_path,
            conv_meta=conv_index,
            selected_conversations=selected_set,
            migration_run_id=migration_run_id,
            roles=roles,
            assistant_filter=args.assistant_filter,
            max_chars=args.max_chars,
            max_chunks=args.max_chunks,
        )
    )

    written = write_jsonl(queue_path, items)
    summary = {
        "zip": str(zip_path),
        "out": str(out_dir),
        "migration_run_id": migration_run_id,
        "selected_conversations": len(selected),
        "max_conversations": args.max_conversations,
        "max_chunks": args.max_chunks,
        "max_chars": args.max_chars,
        "roles": sorted(roles),
        "assistant_filter": args.assistant_filter if "assistant" in roles else "",
        "emitted_chunks": written,
        "note": "Skim queue.jsonl before ingesting; it may contain sensitive content.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {written} chunks to {queue_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
