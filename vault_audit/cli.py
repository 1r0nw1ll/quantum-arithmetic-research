"""
Command-line interface for the vault audit toolkit.

Entry points:
    * `scan`      – crawl the vault and emit structured metadata (delegates to `walker`).
    * `summarize` – batch payloads for Gemini and persist summaries (`summarize` module).
    * `report`    – compile final chronological assessments (`report` module).

This CLI will coordinate Codex-authored implementations and provide hooks
for Gemini to inject analytical insights.
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from . import report, summarize, walker


def _decode_payload(raw: bytes) -> str:
    """Decode raw bytes to UTF-8, replacing undecodable sequences."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments and subcommands."""
    parser = argparse.ArgumentParser(description="Vault audit orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Traverse vault and collect file metadata")
    scan_parser.add_argument("--root", action="append", required=True, help="Vault root directory")
    scan_parser.add_argument("--out", type=Path, required=True, help="Output path for metadata index")
    scan_parser.add_argument("--include-blobs", action="store_true", help="Store raw payloads alongside metadata")
    scan_parser.add_argument(
        "--include", action="append", help="Glob pattern(s) to include (default: all files)"
    )
    scan_parser.add_argument(
        "--exclude", action="append", help="Glob pattern(s) to exclude (evaluated against root-relative paths)"
    )

    summarize_parser = subparsers.add_parser("summarize", help="Generate Gemini-ready payload batches")
    summarize_parser.add_argument("--index", type=Path, required=True, help="Input metadata index path")
    summarize_parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for cached summaries")
    summarize_parser.add_argument("--max-chars", type=int, default=4000, help="Approximate max characters per chunk")

    report_parser = subparsers.add_parser("report", help="Build final chronological audit report")
    report_parser.add_argument("--records", type=Path, required=True, help="Metadata index path")
    report_parser.add_argument("--summaries", type=Path, required=True, help="Path to summarised outputs")
    report_parser.add_argument("--out-dir", type=Path, required=True, help="Directory for report artefacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point wired for collaboration with Codex and Gemini."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        walker_instance = walker.VaultWalker(
            roots=[Path(root) for root in args.root],
            include_blobs=args.include_blobs,
            include_globs=args.include,
            exclude_globs=args.exclude,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in walker_instance.walk():
                try:
                    relative_path = record.path.relative_to(record.root).as_posix()
                except ValueError:
                    relative_path = record.path.as_posix()

                payload_b64: str | None = None
                if record.payload is not None:
                    payload_b64 = base64.b64encode(record.payload).decode("ascii")

                entry: Dict[str, Any] = {
                    "root": record.root.as_posix(),
                    "path": record.path.as_posix(),
                    "relative_path": relative_path,
                    "size_bytes": record.size_bytes,
                    "mtime_ns": record.mtime_ns,
                    "mtime_iso": datetime.fromtimestamp(
                        record.mtime_ns / 1_000_000_000, tz=timezone.utc
                    ).isoformat(),
                    "sha256": record.sha256,
                    "payload_b64": payload_b64,
                }

                json.dump(entry, handle)
                handle.write("\n")

        return 0

    if args.command == "summarize":
        planner = summarize.SummaryPlanner(max_chunk_tokens=args.max_chars)
        cache = summarize.SummaryCache(args.cache_dir)

        def payload_iter() -> Iterator[Tuple[Path, str]]:
            with args.index.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    path = Path(entry["path"])
                    payload_b64 = entry.get("payload_b64")

                    if payload_b64:
                        raw_bytes = base64.b64decode(payload_b64)
                    else:
                        raw_bytes = path.read_bytes()

                    text = _decode_payload(raw_bytes)
                    yield path, text

        total_requests = 0
        created_chunks = 0

        for request in planner.plan(payload_iter()):
            total_requests += 1
            _, created = cache.store_chunk(request)
            if created:
                created_chunks += 1

        print(
            "Prepared "
            f"{total_requests} chunk tasks; "
            f"{created_chunks} new chunks saved to cache (total {len(list(cache.chunks_dir.glob('*.txt')))})."
        )
        return 0

    if args.command == "report":
        builder = report.ReportBuilder(args.out_dir)
        audit_report = builder.build(args.records, args.summaries)
        print("Report generated:")
        print(f"- Markdown: {audit_report.markdown_path}")
        print(f"- Metadata: {audit_report.metadata_path}")
        if audit_report.plots:
            for plot in audit_report.plots:
                print(f"- Plot: {plot}")
        return 0

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
