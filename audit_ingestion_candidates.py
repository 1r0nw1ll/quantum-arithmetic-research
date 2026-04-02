#!/usr/bin/env python3
"""Audit ingestion candidate documents and generate status artifacts.

Entry point:
    python audit_ingestion_candidates.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


# --- Configuration ---
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".tex",
    ".csv",
    ".rst",
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "results",
    "results_argmax",
    "results_baseline",
    "results_chromo",
    "results_chromo_default",
    "results_kpeaks5",
    "results_subsample4",
}
DEFAULT_SCAN_ROOTS = ("qa_alphageometry_ptolemy", "qa_lab", "docs")
EXCLUDED_CANDIDATE_FILENAMES = {"INGESTION_AUDIT.md", "INGESTION_AUDIT.json"}


# --- Helpers ---
def resolve_candidate_dir(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        candidate = repo_root / explicit
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Candidate directory not found: {candidate}")

    for name in ("ingestion candidates", "ingestion_candidates"):
        candidate = repo_root / name
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find ingestion folder. Tried: 'ingestion candidates', 'ingestion_candidates'."
    )


def iter_text_files(
    repo_root: Path, excluded_roots: set[Path], max_size_bytes: int, scan_roots: List[str]
) -> Iterable[Path]:
    excluded_roots_resolved = {p.resolve() for p in excluded_roots}
    seen: set[Path] = set()

    # Always scan top-level files at repo root.
    for path in repo_root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > max_size_bytes:
                continue
        except OSError:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path

    for relative_root in scan_roots:
        root_path = (repo_root / relative_root).resolve()
        if not root_path.exists() or not root_path.is_dir():
            continue

        if root_path in excluded_roots_resolved:
            continue

        for root, dirs, files in os.walk(root_path):
            root_dir = Path(root)
            root_resolved = root_dir.resolve()

            if root_resolved in excluded_roots_resolved:
                dirs[:] = []
                continue

            dirs[:] = [
                d
                for d in dirs
                if d not in SKIP_DIR_NAMES
                and (root_dir / d).resolve() not in excluded_roots_resolved
            ]

            for filename in files:
                path = root_dir / filename
                if path.suffix.lower() not in TEXT_EXTENSIONS:
                    continue

                try:
                    if path.stat().st_size > max_size_bytes:
                        continue
                except OSError:
                    continue

                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def size_human(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def to_rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# --- Core Audit ---
def build_audit(repo_root: Path, candidate_dir: Path, max_text_size: int, scan_roots: List[str]) -> Dict:
    candidate_files = sorted(
        [
            p
            for p in candidate_dir.iterdir()
            if p.is_file() and p.name not in EXCLUDED_CANDIDATE_FILENAMES
        ],
        key=lambda p: p.name.lower(),
    )

    if not candidate_files:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(repo_root),
            "candidate_dir": str(candidate_dir),
            "total_files": 0,
            "processed_files": 0,
            "pending_files": 0,
            "by_extension": {},
            "largest_files": [],
            "duplicates": [],
            "files": [],
        }

    candidate_relative_names = [f"{candidate_dir.name}/{file.name}" for file in candidate_files]
    marker_alias = (
        "ingestion_candidates/"
        if candidate_dir.name == "ingestion candidates"
        else "ingestion candidates/"
    )

    patterns_by_name: Dict[str, tuple[str, str]] = {}
    for idx, candidate in enumerate(candidate_files):
        primary = candidate_relative_names[idx]
        alias = f"{marker_alias}{candidate.name}"
        patterns_by_name[candidate.name] = (primary, alias)

    references_by_name: Dict[str, set[str]] = defaultdict(set)
    excluded_roots = {candidate_dir}

    for text_file in iter_text_files(
        repo_root, excluded_roots=excluded_roots, max_size_bytes=max_text_size, scan_roots=scan_roots
    ):
        try:
            content = text_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel_text_path = to_rel_posix(text_file, repo_root)
        if "ingestion candidates/" not in content and "ingestion_candidates/" not in content:
            continue

        for name, patterns in patterns_by_name.items():
            if patterns[0] in content or patterns[1] in content:
                references_by_name[name].add(rel_text_path)

    by_extension = Counter()
    hash_groups: Dict[str, List[Path]] = defaultdict(list)
    file_records = []

    for candidate in candidate_files:
        try:
            size_bytes = candidate.stat().st_size
        except OSError:
            size_bytes = 0

        extension = candidate.suffix.lower().lstrip(".") or "(none)"
        by_extension[extension] += 1

        try:
            sha256 = sha256_file(candidate)
            hash_groups[sha256].append(candidate)
        except OSError:
            sha256 = ""

        refs = sorted(references_by_name.get(candidate.name, set()))
        file_records.append(
            {
                "name": candidate.name,
                "relative_path": f"{candidate_dir.name}/{candidate.name}",
                "extension": extension,
                "size_bytes": size_bytes,
                "size_human": size_human(size_bytes),
                "sha256": sha256,
                "reference_count": len(refs),
                "referenced_by": refs,
                "status": "processed" if refs else "pending",
            }
        )

    file_records.sort(key=lambda r: (r["status"] != "pending", r["name"].lower()))

    duplicates = []
    for digest, paths in hash_groups.items():
        if len(paths) < 2:
            continue
        duplicates.append(
            {
                "sha256": digest,
                "files": [f"{candidate_dir.name}/{p.name}" for p in sorted(paths, key=lambda p: p.name.lower())],
            }
        )

    largest_files = sorted(file_records, key=lambda r: r["size_bytes"], reverse=True)[:10]

    processed = sum(1 for record in file_records if record["status"] == "processed")
    pending = len(file_records) - processed

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "candidate_dir": str(candidate_dir),
        "total_files": len(file_records),
        "processed_files": processed,
        "pending_files": pending,
        "by_extension": dict(sorted(by_extension.items(), key=lambda kv: kv[0])),
        "largest_files": [
            {
                "name": item["name"],
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "size_human": item["size_human"],
                "reference_count": item["reference_count"],
            }
            for item in largest_files
        ],
        "duplicates": sorted(duplicates, key=lambda d: d["files"][0].lower()),
        "files": file_records,
    }


# --- Rendering ---
def render_markdown(audit: Dict) -> str:
    lines: List[str] = []

    lines.append("# Ingestion Candidates Audit")
    lines.append("")
    lines.append(f"- Generated (UTC): `{audit['generated_at']}`")
    lines.append(f"- Candidate directory: `{audit['candidate_dir']}`")
    lines.append(f"- Total files: `{audit['total_files']}`")
    lines.append(f"- Processed (referenced): `{audit['processed_files']}`")
    lines.append(f"- Pending (unreferenced): `{audit['pending_files']}`")
    lines.append("")

    lines.append("## Extension Breakdown")
    lines.append("")
    lines.append("| Extension | Count |")
    lines.append("|---|---:|")
    for extension, count in audit["by_extension"].items():
        lines.append(f"| `{extension}` | {count} |")
    lines.append("")

    lines.append("## Largest Files (Top 10)")
    lines.append("")
    lines.append("| File | Size | References |")
    lines.append("|---|---:|---:|")
    for item in audit["largest_files"]:
        lines.append(
            f"| `{item['relative_path']}` | {item['size_human']} | {item['reference_count']} |"
        )
    lines.append("")

    lines.append("## Duplicate Content (SHA-256)")
    lines.append("")
    if not audit["duplicates"]:
        lines.append("No duplicate files detected.")
    else:
        for group in audit["duplicates"]:
            lines.append(f"- `{group['sha256']}`")
            for path in group["files"]:
                lines.append(f"  - `{path}`")
    lines.append("")

    lines.append("## File Status")
    lines.append("")
    lines.append("| Status | File | Size | Refs |")
    lines.append("|---|---|---:|---:|")
    for item in audit["files"]:
        status = "✅ processed" if item["status"] == "processed" else "⏳ pending"
        lines.append(
            f"| {status} | `{item['relative_path']}` | {item['size_human']} | {item['reference_count']} |"
        )

    lines.append("")
    lines.append(
        "Reference status is inferred from explicit path mentions in repo text files; it is a practical signal, not a formal ingestion proof."
    )

    return "\n".join(lines) + "\n"


# --- CLI ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ingestion candidate files and generate reports.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to scan (default: current directory).",
    )
    parser.add_argument(
        "--candidate-dir",
        default=None,
        help="Explicit candidate directory relative to repo root.",
    )
    parser.add_argument(
        "--max-text-size-bytes",
        type=int,
        default=2_000_000,
        help="Maximum text file size to scan for references (default: 2,000,000).",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output markdown path relative to repo root (default: <candidate_dir>/INGESTION_AUDIT.md).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path relative to repo root (default: <candidate_dir>/INGESTION_AUDIT.json).",
    )
    parser.add_argument(
        "--scan-roots",
        nargs="*",
        default=list(DEFAULT_SCAN_ROOTS),
        help=(
            "Directories (relative to repo root) to scan recursively for references. "
            "Top-level repo files are always scanned."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    candidate_dir = resolve_candidate_dir(repo_root, args.candidate_dir)

    audit = build_audit(
        repo_root=repo_root,
        candidate_dir=candidate_dir,
        max_text_size=args.max_text_size_bytes,
        scan_roots=args.scan_roots,
    )

    output_md = (
        repo_root / args.output_md
        if args.output_md
        else candidate_dir / "INGESTION_AUDIT.md"
    )
    output_json = (
        repo_root / args.output_json
        if args.output_json
        else candidate_dir / "INGESTION_AUDIT.json"
    )

    output_md.write_text(render_markdown(audit), encoding="utf-8")
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Audit complete: {audit['total_files']} files, {audit['processed_files']} processed, {audit['pending_files']} pending")
    print(f"Wrote markdown: {to_rel_posix(output_md, repo_root)}")
    print(f"Wrote json: {to_rel_posix(output_json, repo_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
