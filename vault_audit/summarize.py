"""
Summarisation helpers for vault audit records.

Responsibilities:
    * Batch raw file payloads into prompt-friendly chunks.
    * Track which artefacts have already been summarised.
    * Persist intermediary caches for reproducibility.

This module will be a focal point for Gemini's analytical routines once the
core batching utilities are implemented by Codex.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

CHUNK_DIR = "chunks"
SUMMARY_DIR = "summaries"
MANIFEST_NAME = "manifest.jsonl"


@dataclass
class SummaryRequest:
    """Descriptor mapping a vault artefact to a summarisation task."""

    source_path: Path
    text: str
    chunk_index: int
    total_chunks: int


class SummaryPlanner:
    """Builds `SummaryRequest` entries from raw vault records."""

    def __init__(self, max_chunk_tokens: int = 1500) -> None:
        self.max_chunk_tokens = max_chunk_tokens

    def plan(self, payloads: Iterable[Tuple[Path, str]]) -> Iterator[SummaryRequest]:
        """
        Break payloads into manageable segments.

        A token approximation based on character count is used to minimise
        dependencies. Chunks preserve paragraph boundaries where possible to
        keep downstream prompts coherent for Gemini.
        """
        for source_path, text in payloads:
            sections = self._chunk_text(text)
            total = len(sections)

            if total == 0:
                continue

            for idx, section in enumerate(sections):
                yield SummaryRequest(
                    source_path=source_path,
                    text=section,
                    chunk_index=idx,
                    total_chunks=total,
                )

    # --- Internal helpers -------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """Split `text` into <= max_chunk_tokens segments."""
        if not text.strip():
            return []

        limit = max(self.max_chunk_tokens, 200)  # guard against tiny limits
        if len(text) <= limit:
            return [text]

        sections: List[str] = []
        current: List[str] = []
        current_len = 0

        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            # If paragraph alone exceeds limit, hard-wrap it.
            if para_len > limit:
                sections.extend(self._split_long_paragraph(para, limit))
                continue

            if current_len + para_len + 2 <= limit:
                current.append(para)
                current_len += para_len + 2
            else:
                if current:
                    sections.append("\n\n".join(current))
                current = [para]
                current_len = para_len

        if current:
            sections.append("\n\n".join(current))

        return sections

    def _split_long_paragraph(self, paragraph: str, limit: int) -> List[str]:
        """Hard wrap a single paragraph that exceeds the chunk limit."""
        words = paragraph.split()
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for word in words:
            word_len = len(word)
            if current_len + word_len + 1 <= limit:
                current.append(word)
                current_len += word_len + 1
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [word]
                current_len = word_len

        if current:
            chunks.append(" ".join(current))

        return chunks


class SummaryCache:
    """Tracks completed summaries and prevents duplicate Gemini work."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.chunks_dir = self.cache_dir / CHUNK_DIR
        self.summaries_dir = self.cache_dir / SUMMARY_DIR
        self.manifest_path = self.cache_dir / MANIFEST_NAME

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(exist_ok=True)
        self.summaries_dir.mkdir(exist_ok=True)

    # --- Public API -------------------------------------------------------

    def load_index(self) -> List[Path]:
        """
        Retrieve the list of already-summarised artefacts.

        Returns paths to summary files stored in the cache.
        """
        return sorted(self.summaries_dir.glob("*.md"))

    def save_summary(self, request: SummaryRequest, summary: str) -> None:
        """
        Persist Gemini output for later aggregation.

        Summaries are keyed by the hashed combination of source path,
        chunk index, and contents of the prompt chunk to avoid collisions.
        """
        key = self._chunk_key(request)
        summary_path = self.summaries_dir / f"{key}.md"
        summary_path.write_text(summary, encoding="utf-8")

    def store_chunk(self, request: SummaryRequest) -> Tuple[Path, bool]:
        """
        Persist the chunk text to disk and append metadata to the manifest.

        Returns:
            (chunk_path, created_flag) where `created_flag` indicates whether
            a new chunk file was written during this invocation.
        """
        key = self._chunk_key(request)
        chunk_path = self.chunks_dir / f"{key}.txt"

        created = False
        if not chunk_path.exists():
            chunk_path.write_text(request.text, encoding="utf-8")
            self._append_manifest_record(request, key, chunk_path)
            created = True

        return chunk_path, created

    def chunk_exists(self, request: SummaryRequest) -> bool:
        """Check if a chunk for the given request already exists."""
        key = self._chunk_key(request)
        return (self.chunks_dir / f"{key}.txt").exists()

    def summary_path_for_key(self, key: str) -> Path:
        """Return the expected path for a summary with the given cache key."""
        return self.summaries_dir / f"{key}.md"

    def iter_manifest(self) -> Iterator[dict]:
        """Yield manifest records describing stored chunks."""
        if not self.manifest_path.exists():
            return iter(())

        def _generator() -> Iterator[dict]:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

        return _generator()

    def request_from_record(self, record: dict) -> SummaryRequest:
        """Reconstruct a `SummaryRequest` from a manifest record."""
        chunk_path = Path(record["chunk_path"])
        text = chunk_path.read_text(encoding="utf-8")
        return SummaryRequest(
            source_path=Path(record["source_path"]),
            text=text,
            chunk_index=record["chunk_index"],
            total_chunks=record["total_chunks"],
        )

    # --- Internal helpers -------------------------------------------------

    def _chunk_key(self, request: SummaryRequest) -> str:
        """Stable hash combining source path, chunk index, and text."""
        hasher = hashlib.sha256()
        hasher.update(str(request.source_path).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(request.chunk_index).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(request.total_chunks).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(request.text.encode("utf-8"))
        return hasher.hexdigest()

    def _append_manifest_record(self, request: SummaryRequest, key: str, chunk_path: Path) -> None:
        """Append chunk metadata to the manifest JSONL file."""
        record = {
            "chunk_key": key,
            "source_path": str(request.source_path),
            "chunk_index": request.chunk_index,
            "total_chunks": request.total_chunks,
            "chunk_path": str(chunk_path),
        }

        with self.manifest_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.write("\n")
