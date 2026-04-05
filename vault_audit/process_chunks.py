"""
Utility runner to process Gemini summaries in controllable batches.

This avoids long single-shot CLI invocations that may hit external timeouts.
Example:
    python -m vault_audit.process_chunks --cache-dir vault_audit_cache --total 200 --batch 10 --timeout 300
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from .gemini_agent import DEFAULT_PROMPT, GeminiAgent

logger = logging.getLogger("GeminiBatchRunner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch Gemini summarisation runner")
    parser.add_argument("--cache-dir", type=Path, default=Path("vault_audit_cache"), help="Summary cache directory")
    parser.add_argument("--gemini-cmd", default="gemini", help="Gemini CLI command")
    parser.add_argument("--prompt-file", type=Path, help="Optional prompt override")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout passed to GeminiAgent")
    parser.add_argument("--total", type=int, help="Total number of chunks to process (default: unlimited)")
    parser.add_argument("--batch", type=int, default=10, help="Chunks per iteration (default: 10)")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between batches")
    parser.add_argument("--dry-run", action="store_true", help="Inspect queue without invoking Gemini")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)

    if args.prompt_file:
        prompt_template = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt_template = DEFAULT_PROMPT

    agent = GeminiAgent(
        cache_dir=args.cache_dir,
        gemini_command=args.gemini_cmd,
        prompt_template=prompt_template,
        timeout=args.timeout,
    )

    processed_total = 0
    target = args.total if args.total is not None else float("inf")

    while processed_total < target:
        to_run = min(args.batch, int(target - processed_total)) if args.total is not None else args.batch

        logger.info("Starting batch (size=%s, processed_total=%s)", to_run, processed_total)
        processed = agent.run(limit=to_run, dry_run=args.dry_run)

        if processed == 0:
            logger.info("No more pending chunks; exiting.")
            break

        processed_total += processed
        logger.info("Batch complete: processed=%s, cumulative=%s", processed, processed_total)

        if processed_total >= target:
            break

        if args.sleep:
            time.sleep(args.sleep)

    logger.info("Finished processing. Total chunks summarised: %s", processed_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
