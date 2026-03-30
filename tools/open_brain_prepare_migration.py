#!/usr/bin/env python3
"""
Prepare an Obsidian-style Markdown corpus for Open Brain ingestion.

This script does NOT write to Supabase directly. It creates a JSONL queue file
that you can ingest using any MCP-connected client (Codex/Claude/ChatGPT) by
calling the `capture_thought` tool per item.

Typical usage:
  python3 tools/open_brain_prepare_migration.py --vault private/QAnotes --max-files 50 --out Documents/open_brain_migration
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional


FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$", re.MULTILINE)


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    path: str
    heading: str
    text: str
    migration_run_id: str


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_frontmatter(md: str) -> str:
    return re.sub(FRONTMATTER_RE, "", md)


def split_by_headings(md: str) -> list[tuple[str, str]]:
    """
    Returns list of (heading, body) where heading may be "" for pre-heading content.
    """
    matches = list(HEADING_RE.finditer(md))
    if not matches:
        return [("", md.strip())]

    sections: list[tuple[str, str]] = []
    first = matches[0]
    pre = md[: first.start()].strip()
    if pre:
        sections.append(("", pre))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        if body:
            sections.append((heading, body))
        else:
            # Keep empty section headings only if they may carry meaning.
            sections.append((heading, ""))
    return sections


def iter_paragraphs(text: str) -> Iterator[str]:
    for p in re.split(r"\n{2,}", text.strip()):
        p = p.strip()
        if not p:
            continue
        yield p


def split_long_block(text: str, max_chars: int) -> Iterator[str]:
    if len(text) <= max_chars:
        yield text
        return

    lines = text.splitlines()
    buf: list[str] = []
    size = 0

    def flush() -> Optional[str]:
        nonlocal buf, size
        if not buf:
            return None
        out = "\n".join(buf).strip()
        buf = []
        size = 0
        return out

    for ln in lines:
        ln = ln.rstrip("\n")
        if not ln:
            candidate = flush()
            if candidate:
                yield candidate
            continue

        while ln and len(ln) > max_chars:
            candidate = flush()
            if candidate:
                yield candidate
            yield ln[:max_chars].strip()
            ln = ln[max_chars:]

        add_len = len(ln) + (1 if buf else 0)
        if size + add_len > max_chars and buf:
            candidate = flush()
            if candidate:
                yield candidate
        buf.append(ln)
        size += add_len

    candidate = flush()
    if candidate:
        yield candidate


def chunk_section(
    *,
    rel_path: str,
    heading: str,
    body: str,
    max_chars: int,
    migration_run_id: str,
) -> Iterator[Chunk]:
    if not body.strip():
        return

    buffer: list[str] = []
    size = 0

    def flush() -> Optional[str]:
        nonlocal buffer, size
        if not buffer:
            return None
        out = "\n\n".join(buffer).strip()
        buffer = []
        size = 0
        return out

    for para in iter_paragraphs(body):
        for block in split_long_block(para, max_chars):
            if size + len(block) + 2 > max_chars and buffer:
                flushed = flush()
                if flushed:
                    yield make_chunk(
                        rel_path=rel_path,
                        heading=heading,
                        content=flushed,
                        migration_run_id=migration_run_id,
                    )
            buffer.append(block)
            size += len(block) + 2

    flushed = flush()
    if flushed:
        yield make_chunk(
            rel_path=rel_path,
            heading=heading,
            content=flushed,
            migration_run_id=migration_run_id,
        )


def make_chunk(*, rel_path: str, heading: str, content: str, migration_run_id: str) -> Chunk:
    context = f'From Obsidian note "{rel_path}"'
    if heading:
        context += f' (section: "{heading}")'
    full_text = f"{context}:\n\n{content.strip()}"
    chunk_id = sha256_hex(normalize_for_hash(full_text))
    return Chunk(
        chunk_id=chunk_id,
        source="obsidian",
        path=rel_path,
        heading=heading,
        text=full_text,
        migration_run_id=migration_run_id,
    )


def should_skip_path(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if ".obsidian" in parts:
        return True
    if ".trash" in parts or ".git" in parts:
        return True
    return False


def iter_markdown_files(vault_dir: Path) -> Iterator[Path]:
    for p in vault_dir.rglob("*.md"):
        if should_skip_path(p):
            continue
        if not p.is_file():
            continue
        yield p


def read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_chunks_from_vault(
    *,
    vault_dir: Path,
    max_files: int,
    file_offset: int,
    order: str,
    max_chunks: int,
    max_chars: int,
    migration_run_id: str,
) -> Iterator[Chunk]:
    files = list(iter_markdown_files(vault_dir))
    if order not in {"newest", "oldest"}:
        raise ValueError(f"Unknown order: {order}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=(order == "newest"))
    if file_offset:
        files = files[file_offset:]
    files = files[:max_files]

    seen: set[str] = set()
    emitted = 0

    for path in files:
        rel_path = str(path.relative_to(vault_dir))
        md = strip_frontmatter(read_text_lossy(path))
        md = md.strip()
        if not md:
            continue

        for heading, body in split_by_headings(md):
            for chunk in chunk_section(
                rel_path=rel_path,
                heading=heading,
                body=body,
                max_chars=max_chars,
                migration_run_id=migration_run_id,
            ):
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                yield chunk
                emitted += 1
                if emitted >= max_chunks:
                    return


def write_jsonl(path: Path, items: Iterable[Chunk]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in items:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "path": c.path,
                        "heading": c.heading,
                        "text": c.text,
                        "migration_run_id": c.migration_run_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", type=Path, required=True, help="Path to Obsidian vault directory")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("Documents/open_brain_migration"),
        help="Output directory (default: Documents/open_brain_migration)",
    )
    ap.add_argument("--max-files", type=int, default=50, help="Max markdown files to read (default: 50)")
    ap.add_argument("--file-offset", type=int, default=0, help="Skip this many files after sorting (default: 0)")
    ap.add_argument(
        "--order",
        type=str,
        default="newest",
        help='File sort order by mtime: "newest" or "oldest" (default: "newest")',
    )
    ap.add_argument("--max-chunks", type=int, default=250, help="Max chunks to emit (default: 250)")
    ap.add_argument("--max-chars", type=int, default=1500, help="Max characters per chunk (default: 1500)")
    ap.add_argument(
        "--migration-run-id",
        type=str,
        default="",
        help="Optional run id to stamp into each queue item (default: auto)",
    )
    args = ap.parse_args()

    vault_dir: Path = args.vault
    if not vault_dir.exists() or not vault_dir.is_dir():
        raise SystemExit(f"--vault must be an existing directory: {vault_dir}")

    out_dir: Path = args.out
    queue_path = out_dir / "queue.jsonl"
    summary_path = out_dir / "summary.json"

    migration_run_id = args.migration_run_id.strip()
    if not migration_run_id:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        migration_run_id = f"{vault_dir.name}_{ts}"
    os.environ["OPEN_BRAIN_MIGRATION_RUN_ID"] = migration_run_id

    chunks = list(
        iter_chunks_from_vault(
            vault_dir=vault_dir,
            max_files=args.max_files,
            file_offset=args.file_offset,
            order=args.order,
            max_chunks=args.max_chunks,
            max_chars=args.max_chars,
            migration_run_id=migration_run_id,
        )
    )
    written = write_jsonl(queue_path, chunks)

    summary = {
        "vault": str(vault_dir),
        "order": args.order,
        "file_offset": args.file_offset,
        "max_files": args.max_files,
        "max_chunks": args.max_chunks,
        "max_chars": args.max_chars,
        "emitted_chunks": written,
        "migration_run_id": migration_run_id,
        "note": "Ingest by reading queue.jsonl and calling capture_thought per item.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {written} chunks to {queue_path}")
    print(f"Wrote summary to {summary_path}")
    print("")
    print("Next (in Codex chat):")
    print('  1) Open queue.jsonl, skim the first 10 items for relevance.')
    print('  2) Ingest in small batches: 10–25 chunks at a time via open-brain capture_thought.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
