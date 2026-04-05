"""
Headless Gemini agent for summarising vault audit chunks.

Usage:
    python -m vault_audit.gemini_agent --cache-dir vault_audit_cache --limit 5

The agent scans the chunk manifest produced by `vault_audit.cli summarize`,
invokes the Gemini CLI for chunks lacking summaries, and persists the results
through `SummaryCache`.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

from .summarize import SummaryCache, SummaryRequest

logger = logging.getLogger("GeminiAgent")

DEFAULT_PROMPT = """You are Gemini, assisting with a research audit.
Summarise the provided text chunk concisely (3-5 bullet points) focusing on key facts, definitions,
and chronology markers. Note if the text appears to continue mid-sentence or references other sections.

Source file: {source_path}
Chunk: {chunk_index}/{total_chunks}

--- BEGIN TEXT ---
{chunk_text}
--- END TEXT ---

Return output in Markdown with a heading `### Summary` followed by bullet points.
"""


class GeminiAgent:
    """Orchestrates Gemini CLI invocations for cached chunk files."""

    def __init__(
        self,
        cache_dir: Path,
        gemini_command: Sequence[str] | str = "gemini",
        prompt_template: str = DEFAULT_PROMPT,
        timeout: int = 120,
    ) -> None:
        self.cache = SummaryCache(cache_dir)
        if isinstance(gemini_command, str):
            self.gemini_command = shlex.split(gemini_command)
        else:
            self.gemini_command = list(gemini_command)
        self.prompt_template = prompt_template
        self.timeout = timeout

    def run(self, limit: Optional[int] = None, dry_run: bool = False) -> int:
        """Process up to `limit` pending chunks (all if None)."""
        processed = 0

        for record in self.cache.iter_manifest():
            key = record["chunk_key"]
            summary_path = self.cache.summary_path_for_key(key)
            if summary_path.exists():
                continue

            request = self.cache.request_from_record(record)
            prompt = self.prompt_template.format(
                source_path=request.source_path,
                chunk_index=request.chunk_index + 1,  # human friendly
                total_chunks=request.total_chunks,
                chunk_text=request.text,
            )

            if dry_run:
                logger.info("DRY RUN: would summarise %s chunk %s", request.source_path, key)
            else:
                summary = self._invoke_gemini(prompt)
                self.cache.save_summary(request, summary)
                logger.info("Saved summary for %s", summary_path)

            processed += 1
            if limit is not None and processed >= limit:
                break

        return processed

    # --- Internal helpers -------------------------------------------------

    def _invoke_gemini(self, prompt: str) -> str:
        """Call the Gemini CLI with the provided prompt."""
        try:
            completed = subprocess.run(
                self.gemini_command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Gemini CLI not found: {self.gemini_command}") from exc

        if completed.returncode != 0:
            logger.error("Gemini CLI failed: %s", completed.stderr.strip())
            raise RuntimeError("Gemini CLI invocation failed.")

        return completed.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the Gemini agent."""
    parser = argparse.ArgumentParser(description="Headless Gemini summarisation agent")
    parser.add_argument("--cache-dir", type=Path, default=Path("vault_audit_cache"), help="Summary cache directory")
    parser.add_argument("--gemini-cmd", default="gemini", help="Gemini CLI command")
    parser.add_argument("--prompt-file", type=Path, help="Optional file containing prompt template")
    parser.add_argument("--limit", type=int, help="Maximum number of chunks to process (default: all)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Inspect queue without invoking Gemini")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    prompt_template = DEFAULT_PROMPT
    if args.prompt_file:
        prompt_template = args.prompt_file.read_text(encoding="utf-8")

    agent = GeminiAgent(
        cache_dir=args.cache_dir,
        gemini_command=args.gemini_cmd,
        prompt_template=prompt_template,
        timeout=args.timeout,
    )

    processed = agent.run(limit=args.limit, dry_run=args.dry_run)
    logger.info("Processed %d chunk(s).", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
