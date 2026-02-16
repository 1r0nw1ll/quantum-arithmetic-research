"""
QA Cartography Gate (forensics regression detection)

Purpose
  - Compare the latest two `_forensics/forensics_*` runs and flag suspicious drift:
      - scanned file/byte counts
      - top-level byte distribution
      - extension counts
      - keyword hotspot top-K changes
      - chat-missing python target changes (if present)

Entry point
  - python tools/qa_cartography_gate.py

Default outputs
  - `_forensics/cartography_gate_<timestamp>/REPORT.md`
  - `_forensics/cartography_gate_<timestamp>/summary.json`

Privacy
  - Uses aggregated counts only (no raw chat text).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UTC = dt.timezone.utc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return dt.datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _find_forensics_runs(root: Path) -> list[Path]:
    base = root / "_forensics"
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("forensics_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _pct_change(prev: float, curr: float) -> float:
    if prev == 0:
        return math.inf if curr != 0 else 0.0
    return (curr - prev) / prev


def _human_bytes(num_bytes: int) -> str:
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


@dataclass(frozen=True)
class GateFinding:
    level: str  # PASS|WARN|FAIL
    code: str
    message: str


def _max_level(levels: Iterable[str]) -> str:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    best = "PASS"
    for lv in levels:
        if order.get(lv, 0) > order[best]:
            best = lv
    return best


def _read_keyword_hotspots(path: Path, *, limit: int) -> list[tuple[int, str]]:
    # returns [(score, relpath), ...]
    rows: list[tuple[int, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                score = int((row.get("score") or "0").strip())
            except ValueError:
                score = 0
            rel = (row.get("path") or "").strip()
            if not rel:
                continue
            rows.append((score, rel))
            if len(rows) >= limit:
                break
    return rows


@dataclass(frozen=True)
class ChatTarget:
    path: str
    count: int
    exists: int


def _read_chat_python_targets(path: Path) -> dict[str, ChatTarget]:
    out: dict[str, ChatTarget] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rel = (row.get("path") or "").strip()
            if not rel:
                continue
            try:
                count = int((row.get("count") or "0").strip())
            except ValueError:
                count = 0
            try:
                exists = int((row.get("exists") or "0").strip())
            except ValueError:
                exists = 0
            out[rel] = ChatTarget(path=rel, count=count, exists=exists)
    return out


def _top_deltas(
    prev_map: dict[str, int],
    curr_map: dict[str, int],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    keys = set(prev_map) | set(curr_map)
    rows: list[dict[str, Any]] = []
    for k in keys:
        pv = int(prev_map.get(k, 0))
        cv = int(curr_map.get(k, 0))
        d = cv - pv
        pct = _pct_change(float(pv), float(cv))
        rows.append({"key": k, "prev": pv, "curr": cv, "delta": d, "pct": pct})
    rows.sort(key=lambda r: (abs(int(r["delta"])), r["key"]), reverse=True)
    return rows[:top_k]


def _format_pct(p: float) -> str:
    if p == math.inf:
        return "inf"
    return f"{p * 100:.1f}%"


def _format_delta(prev: int, curr: int) -> str:
    d = curr - prev
    sign = "+" if d >= 0 else ""
    return f"{sign}{d}"


def _gate(
    *,
    prev: dict[str, Any],
    curr: dict[str, Any],
    prev_hotspots: list[tuple[int, str]],
    curr_hotspots: list[tuple[int, str]],
    prev_chat_targets: dict[str, ChatTarget],
    curr_chat_targets: dict[str, ChatTarget],
    warn_drop_pct: float,
    fail_drop_pct: float,
    min_chat_count: int,
) -> tuple[str, list[GateFinding], dict[str, Any]]:
    findings: list[GateFinding] = []

    prev_files = int(prev.get("scanned_files") or 0)
    curr_files = int(curr.get("scanned_files") or 0)
    prev_bytes = int(prev.get("scanned_bytes") or 0)
    curr_bytes = int(curr.get("scanned_bytes") or 0)

    files_pct = _pct_change(float(prev_files), float(curr_files))
    bytes_pct = _pct_change(float(prev_bytes), float(curr_bytes))

    if files_pct < -fail_drop_pct:
        findings.append(
            GateFinding(
                level="FAIL",
                code="FILES_DROP",
                message=f"scanned_files dropped {files_pct*100:.1f}% ({prev_files} -> {curr_files})",
            )
        )
    elif files_pct < -warn_drop_pct:
        findings.append(
            GateFinding(
                level="WARN",
                code="FILES_DROP",
                message=f"scanned_files dropped {files_pct*100:.1f}% ({prev_files} -> {curr_files})",
            )
        )

    if bytes_pct < -fail_drop_pct:
        findings.append(
            GateFinding(
                level="FAIL",
                code="BYTES_DROP",
                message=f"scanned_bytes dropped {bytes_pct*100:.1f}% ({_human_bytes(prev_bytes)} -> {_human_bytes(curr_bytes)})",
            )
        )
    elif bytes_pct < -warn_drop_pct:
        findings.append(
            GateFinding(
                level="WARN",
                code="BYTES_DROP",
                message=f"scanned_bytes dropped {bytes_pct*100:.1f}% ({_human_bytes(prev_bytes)} -> {_human_bytes(curr_bytes)})",
            )
        )

    # Chat target drift (newly missing high-frequency targets = suspicious).
    newly_missing: list[dict[str, Any]] = []
    newly_found: list[dict[str, Any]] = []
    if prev_chat_targets and curr_chat_targets:
        keys = set(prev_chat_targets) | set(curr_chat_targets)
        for k in keys:
            pv = prev_chat_targets.get(k)
            cv = curr_chat_targets.get(k)
            prev_exists = int(pv.exists) if pv else 0
            curr_exists = int(cv.exists) if cv else 0
            curr_count = int(cv.count) if cv else 0
            if curr_count < min_chat_count:
                continue
            if prev_exists == 1 and curr_exists == 0:
                newly_missing.append({"path": k, "count": curr_count})
            if prev_exists == 0 and curr_exists == 1:
                newly_found.append({"path": k, "count": curr_count})
        newly_missing.sort(key=lambda r: (int(r["count"]), r["path"]), reverse=True)
        newly_found.sort(key=lambda r: (int(r["count"]), r["path"]), reverse=True)
        if newly_missing:
            findings.append(
                GateFinding(
                    level="FAIL",
                    code="CHAT_TARGETS_NEWLY_MISSING",
                    message=f"{len(newly_missing)} high-frequency chat python targets became missing (exists 1 -> 0).",
                )
            )

    # Hotspot drift summary (informational unless severe).
    prev_top = {p for _, p in prev_hotspots}
    curr_top = {p for _, p in curr_hotspots}
    entered = sorted(curr_top - prev_top)
    exited = sorted(prev_top - curr_top)

    status = _max_level([f.level for f in findings] or ["PASS"])
    metrics = {
        "scanned_files": {
            "prev": prev_files,
            "curr": curr_files,
            "delta": curr_files - prev_files,
            "pct": files_pct,
        },
        "scanned_bytes": {
            "prev": prev_bytes,
            "curr": curr_bytes,
            "delta": curr_bytes - prev_bytes,
            "pct": bytes_pct,
        },
        "git_head": {
            "prev": (prev.get("git_head") or ""),
            "curr": (curr.get("git_head") or ""),
            "changed": (prev.get("git_head") or "") != (curr.get("git_head") or ""),
        },
        "hotspots_top_k": {
            "k": len(curr_hotspots),
            "entered": entered[:50],
            "exited": exited[:50],
        },
        "chat_targets": {
            "min_chat_count": min_chat_count,
            "newly_missing_high_freq": newly_missing[:50],
            "newly_found_high_freq": newly_found[:50],
        },
    }
    return status, findings, metrics


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Compare two forensics runs and flag suspicious drift.")
    ap.add_argument("--forensics-dir", default="", help="Current forensics dir (default: latest).")
    ap.add_argument("--prev-forensics-dir", default="", help="Previous forensics dir (default: previous by mtime).")
    ap.add_argument("--hotspot-top", type=int, default=100, help="Top-K keyword hotspots to compare.")
    ap.add_argument("--min-chat-count", type=int, default=25, help="Min chat count for 'high-frequency' targets.")
    ap.add_argument("--warn-drop-pct", type=float, default=0.05, help="Warn if scanned_* drops more than this pct.")
    ap.add_argument("--fail-drop-pct", type=float, default=0.20, help="Fail if scanned_* drops more than this pct.")
    ap.add_argument("--fail-on-warn", action="store_true", help="Exit nonzero on WARN as well.")
    ap.add_argument("--out", default="", help="Output dir (default: _forensics/cartography_gate_<ts>/).")
    ap.add_argument("--json", action="store_true", help="Print summary JSON to stdout.")
    args = ap.parse_args(argv)

    root = _repo_root()
    runs = _find_forensics_runs(root)
    if not runs:
        raise SystemExit("No _forensics/forensics_* runs found. Run `python tools/project_forensics.py` first.")

    curr_dir = Path(args.forensics_dir) if args.forensics_dir else runs[0]
    if not curr_dir.is_absolute():
        curr_dir = (root / curr_dir).resolve()

    prev_dir = Path(args.prev_forensics_dir) if args.prev_forensics_dir else (runs[1] if len(runs) > 1 else Path())
    if not prev_dir:
        raise SystemExit("Need at least 2 forensics runs (or pass --prev-forensics-dir).")
    if not prev_dir.is_absolute():
        prev_dir = (root / prev_dir).resolve()

    prev_summary = _read_json(prev_dir / "repo_summary.json")
    curr_summary = _read_json(curr_dir / "repo_summary.json")

    prev_hotspots = _read_keyword_hotspots(prev_dir / "keyword_hotspots.tsv", limit=int(args.hotspot_top))
    curr_hotspots = _read_keyword_hotspots(curr_dir / "keyword_hotspots.tsv", limit=int(args.hotspot_top))

    prev_chat_targets = _read_chat_python_targets(prev_dir / "chat_python_targets.tsv")
    curr_chat_targets = _read_chat_python_targets(curr_dir / "chat_python_targets.tsv")

    status, findings, metrics = _gate(
        prev=prev_summary,
        curr=curr_summary,
        prev_hotspots=prev_hotspots,
        curr_hotspots=curr_hotspots,
        prev_chat_targets=prev_chat_targets,
        curr_chat_targets=curr_chat_targets,
        warn_drop_pct=float(args.warn_drop_pct),
        fail_drop_pct=float(args.fail_drop_pct),
        min_chat_count=int(args.min_chat_count),
    )

    out_dir = Path(args.out) if args.out else (root / "_forensics" / f"cartography_gate_{_utc_stamp()}")
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rich deltas for report/summary.
    ext_deltas = _top_deltas(
        prev_map={k: int(v) for k, v in (prev_summary.get("ext_counts") or {}).items()},
        curr_map={k: int(v) for k, v in (curr_summary.get("ext_counts") or {}).items()},
        top_k=40,
    )
    top_level_deltas = _top_deltas(
        prev_map={k: int(v) for k, v in (prev_summary.get("top_level_bytes") or {}).items()},
        curr_map={k: int(v) for k, v in (curr_summary.get("top_level_bytes") or {}).items()},
        top_k=30,
    )

    summary_obj = {
        "status": status,
        "prev": {"dir": str(prev_dir), "run_id": prev_summary.get("run_id"), "generated_utc": prev_summary.get("generated_utc")},
        "curr": {"dir": str(curr_dir), "run_id": curr_summary.get("run_id"), "generated_utc": curr_summary.get("generated_utc")},
        "findings": [f.__dict__ for f in findings],
        "metrics": metrics,
        "deltas": {"ext_counts_top": ext_deltas, "top_level_bytes_top": top_level_deltas},
    }
    _write_json(out_dir / "summary.json", summary_obj)

    # Human report.
    lines: list[str] = []
    lines.append("# Cartography Gate Report")
    lines.append("")
    lines.append(f"- Status: `{status}`")
    lines.append(f"- Prev: `{prev_summary.get('run_id','')}` ({prev_dir})")
    lines.append(f"- Curr: `{curr_summary.get('run_id','')}` ({curr_dir})")
    lines.append(f"- Git head changed: `{metrics['git_head']['changed']}`")
    lines.append("")
    if findings:
        lines.append("## Findings")
        for f in findings:
            lines.append(f"- `{f.level}` `{f.code}`: {f.message}")
        lines.append("")

    sf = metrics["scanned_files"]
    sb = metrics["scanned_bytes"]
    lines.append("## Core deltas")
    lines.append(
        f"- scanned_files: {sf['prev']} -> {sf['curr']} ({_format_delta(sf['prev'], sf['curr'])}, {_format_pct(sf['pct'])})"
    )
    lines.append(
        f"- scanned_bytes: {_human_bytes(sb['prev'])} -> {_human_bytes(sb['curr'])} "
        f"({_human_bytes(sb['delta']) if sb['delta'] >= 0 else '-' + _human_bytes(abs(sb['delta']))}, {_format_pct(sb['pct'])})"
    )
    lines.append("")

    lines.append("## Top extension count deltas")
    for r in ext_deltas:
        lines.append(
            f"- `{r['key']}`: {r['prev']} -> {r['curr']} ({_format_delta(r['prev'], r['curr'])}, {_format_pct(float(r['pct']))})"
        )
    lines.append("")

    lines.append("## Top top-level byte deltas")
    for r in top_level_deltas:
        lines.append(
            f"- `{r['key']}`: {_human_bytes(r['prev'])} -> {_human_bytes(r['curr'])} "
            f"({_human_bytes(r['delta']) if r['delta'] >= 0 else '-' + _human_bytes(abs(r['delta']))}, {_format_pct(float(r['pct']))})"
        )
    lines.append("")

    ct = metrics["chat_targets"]
    if prev_chat_targets and curr_chat_targets:
        lines.append("## Chat python target drift (high-frequency)")
        lines.append(f"- min_chat_count: `{ct['min_chat_count']}`")
        lines.append(f"- newly missing (exists 1 -> 0): `{len(ct['newly_missing_high_freq'])}`")
        for r in ct["newly_missing_high_freq"][:20]:
            lines.append(f"  - `{r['path']}` (count={r['count']})")
        lines.append(f"- newly found (exists 0 -> 1): `{len(ct['newly_found_high_freq'])}`")
        for r in ct["newly_found_high_freq"][:20]:
            lines.append(f"  - `{r['path']}` (count={r['count']})")
        lines.append("")

    hs = metrics["hotspots_top_k"]
    lines.append("## Hotspot drift (top-K by score)")
    lines.append(f"- K: `{hs['k']}`")
    lines.append(f"- entered: `{len(hs['entered'])}`")
    for p in hs["entered"][:25]:
        lines.append(f"  - `{p}`")
    lines.append(f"- exited: `{len(hs['exited'])}`")
    for p in hs["exited"][:25]:
        lines.append(f"  - `{p}`")
    lines.append("")

    report_path = out_dir / "REPORT.md"
    _write_text(report_path, "\n".join(lines) + "\n")

    exit_code = 0
    if status == "FAIL":
        exit_code = 1
    elif status == "WARN" and args.fail_on_warn:
        exit_code = 2

    if args.json:
        try:
            import sys as _sys

            _sys.stdout.write(json.dumps(summary_obj, indent=2, sort_keys=True) + "\n")
            _sys.stdout.flush()
        except BrokenPipeError:
            # Allow piping to `head` without noisy tracebacks, but preserve exit code.
            try:
                import os as _os
                import sys as _sys

                _sys.stdout = open(_os.devnull, "w")
            except Exception:
                pass
            return exit_code
    else:
        print(f"[qa_cartography_gate] wrote: {report_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
