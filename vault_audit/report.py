"""
Reporting utilities for the vault audit workflow.

Responsibilities:
    * Aggregate metadata and summaries produced by `walker` and `summarize`.
    * Produce human-readable artefacts (Markdown, CSV, charts) documenting
      chronological coverage of the vault.
    * Emit machine-readable reports for downstream automation.

Gemini will assist with narrative synthesis after Codex implements the
aggregation routines scaffolded below.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


@dataclass
class AuditReport:
    """Container for the final audit artefacts."""

    markdown_path: Path
    metadata_path: Path
    plots: List[Path]


class ReportBuilder:
    """Transforms collected data into publishable audit reports."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, records_path: Path, summaries_path: Path) -> AuditReport:
        """
        Construct the audit report from indexed records and summaries.

        Generates:
            * `audit_summary.json`  – machine-readable metrics.
            * `audit_report.md`     – human-readable summary.
            * `activity_hist.png`   – histogram of modification times.
        """
        records = self._load_records(records_path)
        summaries_dir = summaries_path / "summaries"

        totals = self._compute_totals(records, summaries_dir)
        histogram_path = self._write_histogram(records)

        metadata_path = self.output_dir / "audit_summary.json"
        metadata_path.write_text(json.dumps(totals, indent=2), encoding="utf-8")

        markdown_path = self.output_dir / "audit_report.md"
        markdown_path.write_text(self._render_markdown(totals, histogram_path), encoding="utf-8")

        plots = [p for p in [histogram_path] if p is not None]

        return AuditReport(
            markdown_path=markdown_path,
            metadata_path=metadata_path,
            plots=plots,
        )

    # --- Internal helpers -------------------------------------------------

    def _load_records(self, records_path: Path) -> List[dict]:
        """Load newline-delimited JSON metadata records."""
        records: List[dict] = []
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _compute_totals(self, records: List[dict], summaries_dir: Path) -> dict:
        """Compute aggregate coverage statistics."""
        if not records:
            raise ValueError("No records available to build a report.")

        total_files = len(records)
        total_bytes = sum(rec.get("size_bytes", 0) for rec in records)
        roots = sorted({rec["root"] for rec in records})

        mtimes = sorted(rec["mtime_ns"] for rec in records)
        earliest = datetime.fromtimestamp(mtimes[0] / 1_000_000_000, tz=timezone.utc)
        latest = datetime.fromtimestamp(mtimes[-1] / 1_000_000_000, tz=timezone.utc)

        completed_summaries = 0
        if summaries_dir.exists():
            completed_summaries = len(list(summaries_dir.glob("*.md")))

        return {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "roots": roots,
            "time_range": {
                "earliest_iso": earliest.isoformat(),
                "latest_iso": latest.isoformat(),
            },
            "completed_summaries": completed_summaries,
        }

    def _write_histogram(self, records: List[dict]) -> Path:
        """Generate a histogram plot of file modification times."""
        timestamps = [
            datetime.fromtimestamp(rec["mtime_ns"] / 1_000_000_000, tz=timezone.utc)
            for rec in records
        ]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(timestamps, bins=50, color="#4C72B0", edgecolor="white")
        ax.set_title("Vault File Modification Timeline")
        ax.set_xlabel("Modification Time (UTC)")
        ax.set_ylabel("File Count")
        fig.autofmt_xdate()

        plot_path = self.output_dir / "activity_hist.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def _render_markdown(self, totals: dict, histogram_path: Path | None) -> str:
        """Render a simple Markdown report summarising the audit."""
        lines = [
            "# Vault Audit Report",
            "",
            f"- Total files scanned: **{totals['total_files']}**",
            f"- Total size: **{totals['total_bytes']:,} bytes**",
            f"- Roots: {', '.join(totals['roots'])}",
            f"- Time span: {totals['time_range']['earliest_iso']} → {totals['time_range']['latest_iso']}",
            f"- Summaries completed: **{totals['completed_summaries']}** chunks",
            "",
        ]

        if histogram_path:
            rel_plot = histogram_path.name
            lines.append(f"![Activity histogram]({rel_plot})")
            lines.append("")

        lines.append("Generated by `vault_audit.report.ReportBuilder`.")
        return "\n".join(lines)
