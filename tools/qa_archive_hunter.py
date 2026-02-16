"""
QA Archive Hunter (chat-missing targets -> zip lookup)

Purpose
  - When chat references scripts/paths that are missing from the current tree,
    scan repo zip snapshots and report where those missing files likely live.

Entry points
  - python tools/qa_archive_hunter.py

Default behavior
  - Reads latest `_forensics/forensics_*/chat_python_targets.tsv`
  - Filters to `exists=0`
  - Scans top-level `*.zip` in repo root
  - Writes `_forensics/archive_hunt_<timestamp>/...`

Privacy
  - Only handles filenames/paths + counts; does not emit raw chat text.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UTC = dt.timezone.utc

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "venv",
    "qa_lab",
    "chat_data_extracted",
    "vault_audit_cache",
}


def _utc_stamp() -> str:
    return dt.datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_latest_forensics_dir(root: Path) -> Path:
    base = root / "_forensics"
    if not base.exists():
        raise FileNotFoundError("Missing _forensics/. Run `python tools/project_forensics.py` first.")
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("forensics_")]
    if not runs:
        raise FileNotFoundError("No forensics runs found. Run `python tools/project_forensics.py` first.")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _normalize_path(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    if s.startswith("./"):
        s = s[2:]
    return s.strip("/")


def _basename(posix_path: str) -> str:
    posix_path = _normalize_path(posix_path)
    if "/" in posix_path:
        return posix_path.rsplit("/", 1)[-1]
    return posix_path


def _walk_zip_roots(
    root: Path,
    zip_roots: list[str],
    *,
    max_depth: int,
) -> list[Path]:
    out: list[Path] = []

    def add_zip(p: Path) -> None:
        if p.is_file() and p.suffix.lower() == ".zip":
            out.append(p)

    if not zip_roots:
        # Default: only top-level *.zip at repo root.
        for p in sorted(root.glob("*.zip")):
            add_zip(p)
        return out

    for zr in zip_roots:
        p = (root / zr).resolve() if not Path(zr).is_absolute() else Path(zr).resolve()
        if not p.exists():
            continue
        if p.is_file():
            add_zip(p)
            continue
        if not p.is_dir():
            continue

        base_depth = len(p.parts)
        for dirpath, dirnames, filenames in os.walk(p, topdown=True):
            d = Path(dirpath)
            rel_depth = len(d.parts) - base_depth
            if rel_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS]
            for name in filenames:
                if not name.lower().endswith(".zip"):
                    continue
                add_zip(d / name)

    # De-dupe, stable sort.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    uniq.sort(key=lambda x: x.as_posix())
    return uniq


@dataclass(frozen=True)
class Target:
    raw_path: str
    norm_path: str
    basename: str
    chat_count: int


def _read_targets_from_tsv(path: Path) -> list[Target]:
    targets: list[Target] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            raw = (row.get("path") or "").strip()
            if not raw:
                continue
            norm = _normalize_path(raw)
            base = _basename(norm)
            try:
                count = int((row.get("count") or "0").strip())
            except ValueError:
                count = 0
            targets.append(Target(raw_path=raw, norm_path=norm, basename=base, chat_count=count))
    return targets


def _zip_entry_ts(info: zipfile.ZipInfo) -> str:
    try:
        dt_obj = dt.datetime(*info.date_time, tzinfo=UTC)
        return dt_obj.isoformat()
    except Exception:
        return ""


def _write_tsv(path: Path, header: list[str], rows: Iterable[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for r in rows:
            handle.write("\t".join(r) + "\n")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Scan repo zip snapshots for chat-missing python targets.")
    ap.add_argument("--forensics-dir", default="", help="Use specific _forensics/forensics_* dir (default: latest).")
    ap.add_argument(
        "--targets-tsv",
        default="",
        help="Override TSV input (default: <forensics>/chat_python_targets.tsv).",
    )
    ap.add_argument("--min-count", type=int, default=10, help="Minimum chat count to consider a target.")
    ap.add_argument("--top", type=int, default=200, help="Max targets to scan (after filtering).")
    ap.add_argument(
        "--zip-root",
        action="append",
        default=[],
        help="Zip file or directory to search (repeatable). Default: repo root top-level *.zip only.",
    )
    ap.add_argument(
        "--zip-max-depth",
        type=int,
        default=2,
        help="Max directory depth when scanning --zip-root directories.",
    )
    ap.add_argument(
        "--max-matches-per-target",
        type=int,
        default=60,
        help="Cap detailed match rows per target (basename matches can explode).",
    )
    ap.add_argument("--out", default="", help="Output directory (default: _forensics/archive_hunt_<ts>/).")
    args = ap.parse_args(argv)

    root = _repo_root()
    forensics_dir = Path(args.forensics_dir) if args.forensics_dir else _find_latest_forensics_dir(root)
    if not forensics_dir.is_absolute():
        forensics_dir = (root / forensics_dir).resolve()

    targets_tsv = Path(args.targets_tsv) if args.targets_tsv else (forensics_dir / "chat_python_targets.tsv")
    if not targets_tsv.is_absolute():
        targets_tsv = (root / targets_tsv).resolve()
    if not targets_tsv.exists():
        raise FileNotFoundError(f"Missing targets TSV: {targets_tsv}")

    out_dir = Path(args.out) if args.out else (root / "_forensics" / f"archive_hunt_{_utc_stamp()}")
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_targets = _read_targets_from_tsv(targets_tsv)
    # Default: keep only missing targets if the TSV has an 'exists' column.
    exists_col = False
    with targets_tsv.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        exists_col = "exists" in (header or "")

    if exists_col:
        missing: list[Target] = []
        with targets_tsv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                raw = (row.get("path") or "").strip()
                if not raw:
                    continue
                exists = (row.get("exists") or "").strip()
                if exists not in ("0", "false", "False", ""):
                    continue
                norm = _normalize_path(raw)
                base = _basename(norm)
                try:
                    count = int((row.get("count") or "0").strip())
                except ValueError:
                    count = 0
                missing.append(Target(raw_path=raw, norm_path=norm, basename=base, chat_count=count))
        all_targets = missing

    # Filter + rank.
    filt = [t for t in all_targets if t.chat_count >= int(args.min_count)]
    filt.sort(key=lambda t: (t.chat_count, t.norm_path), reverse=True)
    filt = filt[: max(0, int(args.top))]

    zip_paths = _walk_zip_roots(root, list(args.zip_root), max_depth=max(1, int(args.zip_max_depth)))

    scan_rows: list[list[str]] = []
    match_rows: list[list[str]] = []

    found_targets: set[str] = set()
    found_by_target: dict[str, set[str]] = defaultdict(set)  # target -> {zip}

    for zp in zip_paths:
        try:
            zf = zipfile.ZipFile(zp, "r")
        except Exception as e:
            scan_rows.append([str(zp.relative_to(root)), "ERROR", str(e)])
            continue

        with zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            scan_rows.append([str(zp.relative_to(root)), "OK", str(len(infos))])

            full_set: set[str] = set()
            by_base: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
            for info in infos:
                name = _normalize_path(info.filename)
                full_set.add(name)
                by_base[_basename(name)].append(info)

            for target in filt:
                tkey = target.norm_path
                per_target_budget = int(args.max_matches_per_target)
                emitted = 0

                # Exact path hits
                if tkey in full_set:
                    for info in by_base.get(target.basename, []):
                        if _normalize_path(info.filename) != tkey:
                            continue
                        if emitted >= per_target_budget:
                            break
                        match_rows.append(
                            [
                                target.raw_path,
                                str(target.chat_count),
                                str(zp.relative_to(root)),
                                "path",
                                _normalize_path(info.filename),
                                str(info.file_size),
                                _zip_entry_ts(info),
                            ]
                        )
                        emitted += 1
                    found_targets.add(tkey)
                    found_by_target[tkey].add(str(zp.relative_to(root)))

                # Basename hits (including cases where exact path differs)
                for info in by_base.get(target.basename, []):
                    if emitted >= per_target_budget:
                        break
                    entry = _normalize_path(info.filename)
                    if entry == tkey:
                        continue
                    match_rows.append(
                        [
                            target.raw_path,
                            str(target.chat_count),
                            str(zp.relative_to(root)),
                            "basename",
                            entry,
                            str(info.file_size),
                            _zip_entry_ts(info),
                        ]
                    )
                    emitted += 1
                    found_targets.add(tkey)
                    found_by_target[tkey].add(str(zp.relative_to(root)))

    # Outputs
    _write_json(
        out_dir / "CONFIG.json",
        {
            "generated_utc": dt.datetime.now(tz=UTC).isoformat(),
            "repo_root": str(root),
            "forensics_dir": str(forensics_dir),
            "targets_tsv": str(targets_tsv),
            "min_count": int(args.min_count),
            "top": int(args.top),
            "zip_roots": list(args.zip_root),
            "zip_max_depth": int(args.zip_max_depth),
            "zip_count": len(zip_paths),
            "targets_considered": len(filt),
        },
    )

    _write_tsv(out_dir / "zip_scan.tsv", ["zip_path", "status", "entries_or_error"], scan_rows)
    _write_tsv(
        out_dir / "matches.tsv",
        ["target", "chat_count", "zip_path", "match_kind", "zip_entry", "zip_entry_size", "zip_entry_mtime_utc"],
        match_rows,
    )

    missing_rows: list[list[str]] = []
    for t in filt:
        if t.norm_path not in found_targets:
            missing_rows.append([t.raw_path, str(t.chat_count)])
    _write_tsv(out_dir / "missing.tsv", ["target", "chat_count"], missing_rows)

    # Summary report (markdown)
    found_n = len(filt) - len(missing_rows)
    report_lines: list[str] = []
    report_lines.append("# Archive Hunt Report")
    report_lines.append("")
    report_lines.append(f"- Generated: `{dt.datetime.now(tz=UTC).isoformat()}`")
    report_lines.append(f"- Forensics input: `{forensics_dir}`")
    report_lines.append(f"- Targets TSV: `{targets_tsv}`")
    report_lines.append(f"- Targets considered: `{len(filt)}` (min_count={int(args.min_count)}, top={int(args.top)})")
    report_lines.append(f"- Zip files scanned: `{len(zip_paths)}`")
    report_lines.append(f"- Targets found in at least one zip: `{found_n}`")
    report_lines.append(f"- Targets not found: `{len(missing_rows)}`")
    report_lines.append("")
    report_lines.append("## Top found targets (by chat count)")
    report_lines.append("")
    top_found = [t for t in filt if t.norm_path in found_targets][:20]
    for t in top_found:
        zips = sorted(found_by_target.get(t.norm_path, set()))
        zips_short = ", ".join(zips[:6]) + (f", (+{len(zips)-6} more)" if len(zips) > 6 else "")
        report_lines.append(f"- `{t.raw_path}` (count={t.chat_count}) -> {zips_short}")
    report_lines.append("")
    report_lines.append("## Outputs")
    report_lines.append("")
    report_lines.append(f"- `zip_scan.tsv` — scanned zips + entry counts/errors")
    report_lines.append(f"- `matches.tsv` — detailed match rows (capped per target)")
    report_lines.append(f"- `missing.tsv` — targets not found in any scanned zip")

    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[qa_archive_hunter] wrote: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

