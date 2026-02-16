"""
QA Trace Ledger (run wrapper + run_id manifests)

Purpose
  - Standardize new execution trace into:
      trace/runs/<run_id>/
        RUN_MANIFEST.json
        stdout.log
        stderr.log
        artifacts/
        checks/
  - Append a small ledger entry into a markdown file (default: trace/TRACE_RUNS_LOCAL.md).

Entry points
  - python tools/qa_trace_ledger.py init --tool-id TOOL.<name>.v1
  - python tools/qa_trace_ledger.py run --tool-id TOOL.<name>.v1 -- <command...>

Safety / scope
  - Writes only under trace/ by default (plus the chosen ledger markdown file).
  - Does not modify semantic core files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


UTC = dt.timezone.utc

RUN_LEDGER_BEGIN = "<!-- RUN_LEDGER:BEGIN -->"
RUN_LEDGER_END = "<!-- RUN_LEDGER:END -->"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return dt.datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def _iso_utc(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    s = ts.isoformat()
    return s.replace("+00:00", "Z")


def _sanitize_run_token(raw: str, *, max_len: int = 80) -> str:
    raw = (raw or "").strip()
    raw = raw.replace(" ", "_")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("._-")
    if not raw:
        return "run"
    return raw[:max_len]


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _ensure_marked_ledger(path: Path) -> None:
    if path.exists():
        content = path.read_text(encoding="utf-8", errors="replace")
        if RUN_LEDGER_BEGIN in content and RUN_LEDGER_END in content:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Trace Runs (local)",
                "",
                "Auto-appended by `python tools/qa_trace_ledger.py ...`.",
                "This file is ignored by git by default (trace layer).",
                "",
                RUN_LEDGER_BEGIN,
                RUN_LEDGER_END,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _ledger_contains_run_id(content: str, run_id: str) -> bool:
    needle = f"`run_id`: `{run_id}`"
    return needle in content


def _append_ledger_entry(ledger_path: Path, entry_md: str, *, run_id: str) -> bool:
    _ensure_marked_ledger(ledger_path)
    content = ledger_path.read_text(encoding="utf-8", errors="replace")
    if _ledger_contains_run_id(content, run_id):
        return False
    if RUN_LEDGER_BEGIN not in content or RUN_LEDGER_END not in content:
        raise ValueError(f"Ledger file missing markers: {ledger_path}")
    before, rest = content.split(RUN_LEDGER_BEGIN, 1)
    middle, after = rest.split(RUN_LEDGER_END, 1)
    middle = middle.rstrip() + "\n" + entry_md.rstrip() + "\n"
    new_content = before + RUN_LEDGER_BEGIN + "\n" + middle + RUN_LEDGER_END + after
    ledger_path.write_text(new_content, encoding="utf-8")
    return True


def _default_ledger_path(root: Path) -> Path:
    return root / "trace" / "TRACE_RUNS_LOCAL.md"


def _run_dir(root: Path, run_id: str) -> Path:
    return root / "trace" / "runs" / run_id


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_head(root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _git_status_porcelain(root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        )
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _cmd_display(cmd: list[str]) -> str:
    try:
        return shlex.join(cmd)
    except Exception:
        return " ".join(cmd)


def _make_entry_md(
    *,
    run_id: str,
    when_utc: str,
    tool_id: str,
    cmd_display: str,
    cwd_display: str,
    run_dir_rel: str,
    status: str,
    exit_code: int | None,
    note: str,
) -> str:
    lines: list[str] = []
    lines.append(f"- `run_id`: `{run_id}`")
    lines.append(f"  - `when`: `{when_utc}`")
    lines.append(f"  - `tool`: `{tool_id}`")
    if cmd_display:
        lines.append(f"  - `cmd`: `{cmd_display}`")
    if cwd_display:
        lines.append(f"  - `cwd`: `{cwd_display}`")
    lines.append(f"  - `dir`: `{run_dir_rel}`")
    lines.append(f"  - `status`: `{status}`")
    if exit_code is not None:
        lines.append(f"  - `exit_code`: `{exit_code}`")
    if note:
        safe_note = note.strip().replace("\n", " ").strip()
        if safe_note:
            lines.append(f"  - `notes`: {safe_note}")
    return "\n".join(lines) + "\n"


def _relpath_or_abs(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p.resolve())


def cmd_init(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="qa_trace_ledger.py init", description="Create a new trace run folder + manifest.")
    ap.add_argument("--tool-id", required=True, help="Tool identifier (e.g., TOOL.project_forensics.v1).")
    ap.add_argument("--run-id", default="", help="Override run_id (default: <utc>_<tool-id>).")
    ap.add_argument("--note", default="", help="Short note for the ledger entry.")
    ap.add_argument(
        "--ledger",
        default="",
        help="Ledger markdown path to append into (default: trace/TRACE_RUNS_LOCAL.md). Use 'none' to disable.",
    )
    ap.add_argument("--json", action="store_true", help="Print a machine-readable summary to stdout.")
    args = ap.parse_args(argv)

    root = _repo_root()
    tool_id = args.tool_id.strip()
    run_id = _sanitize_run_token(args.run_id) if args.run_id else f"{_utc_stamp()}_{_sanitize_run_token(tool_id)}"

    run_dir = _run_dir(root, run_id)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "checks").mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now(tz=UTC)
    git_head = _git_head(root)
    git_status = _git_status_porcelain(root)
    git_status_path = run_dir / "GIT_STATUS_PORCELAIN.txt"
    if git_status:
        _write_text(git_status_path, git_status + "\n")
    git_status_sha = _sha256_file(git_status_path) if git_status_path.exists() else ""
    manifest = {
        "run_id": run_id,
        "tool_id": tool_id,
        "created_at_utc": _iso_utc(now),
        "status": "INIT",
        "cwd": ".",
        "cmd": [],
        "git": {
            "head": git_head,
            "is_dirty": bool(git_status),
            "status_porcelain_v1_path": "GIT_STATUS_PORCELAIN.txt" if git_status else "",
            "status_porcelain_v1_sha256": git_status_sha,
            "status_porcelain_v1_lines": len(git_status.splitlines()) if git_status else 0,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "outputs": {
            "run_dir": _relpath_or_abs(root, run_dir),
            "manifest": "RUN_MANIFEST.json",
            "stdout_log": "stdout.log",
            "stderr_log": "stderr.log",
            "artifacts_dir": "artifacts/",
            "checks_dir": "checks/",
        },
        "note": args.note.strip(),
    }
    _write_json(run_dir / "RUN_MANIFEST.json", manifest)

    ledger_path: Path | None = None
    updated = False
    ledger_arg = (args.ledger or "").strip()
    if ledger_arg.lower() != "none":
        ledger_path = Path(ledger_arg) if ledger_arg else _default_ledger_path(root)
        if not ledger_path.is_absolute():
            ledger_path = (root / ledger_path).resolve()
        entry = _make_entry_md(
            run_id=run_id,
            when_utc=_iso_utc(now),
            tool_id=tool_id,
            cmd_display="",
            cwd_display=".",
            run_dir_rel=_relpath_or_abs(root, run_dir) + "/",
            status="INIT",
            exit_code=None,
            note=args.note,
        )
        updated = _append_ledger_entry(ledger_path, entry, run_id=run_id)

    out = {
        "run_id": run_id,
        "run_dir": _relpath_or_abs(root, run_dir),
        "manifest": _relpath_or_abs(root, run_dir / "RUN_MANIFEST.json"),
        "ledger": _relpath_or_abs(root, ledger_path) if ledger_path else "",
        "ledger_updated": updated,
    }
    if args.json:
        sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"[qa_trace_ledger] init: {out['run_dir']}\n")
        if ledger_path:
            sys.stdout.write(f"[qa_trace_ledger] ledger: {out['ledger']} (updated={updated})\n")
    return 0


def cmd_run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="qa_trace_ledger.py run",
        description="Run a command and capture stdout/stderr + manifest under trace/runs/<run_id>/.",
    )
    ap.add_argument("--tool-id", required=True, help="Tool identifier (e.g., TOOL.project_forensics.v1).")
    ap.add_argument("--run-id", default="", help="Override run_id (default: <utc>_<tool-id>).")
    ap.add_argument("--cwd", default=".", help="Working directory for the command (default: repo root).")
    ap.add_argument("--note", default="", help="Short note for the ledger entry.")
    ap.add_argument(
        "--ledger",
        default="",
        help="Ledger markdown path to append into (default: trace/TRACE_RUNS_LOCAL.md). Use 'none' to disable.",
    )
    ap.add_argument("--json", action="store_true", help="Print a machine-readable summary to stdout.")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (prefix with --).")
    args = ap.parse_args(argv)

    cmd = list(args.cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("Missing command. Example: python tools/qa_trace_ledger.py run --tool-id TOOL.x.v1 -- python ...")

    root = _repo_root()
    tool_id = args.tool_id.strip()
    run_id = _sanitize_run_token(args.run_id) if args.run_id else f"{_utc_stamp()}_{_sanitize_run_token(tool_id)}"

    run_dir = _run_dir(root, run_id)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "checks").mkdir(parents=True, exist_ok=True)

    cwd = Path(args.cwd)
    if not cwd.is_absolute():
        cwd = (root / cwd).resolve()

    started = dt.datetime.now(tz=UTC)
    git_head = _git_head(root)
    git_status = _git_status_porcelain(root)
    git_status_path = run_dir / "GIT_STATUS_PORCELAIN.txt"
    if git_status:
        _write_text(git_status_path, git_status + "\n")
    git_status_sha = _sha256_file(git_status_path) if git_status_path.exists() else ""
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    exit_code = 1
    try:
        with stdout_path.open("wb") as out_handle, stderr_path.open("wb") as err_handle:
            r = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=out_handle,
                stderr=err_handle,
                env=os.environ.copy(),
            )
            exit_code = int(r.returncode)
    finally:
        ended = dt.datetime.now(tz=UTC)

    status = "PASS" if exit_code == 0 else "FAIL"

    manifest = {
        "run_id": run_id,
        "tool_id": tool_id,
        "started_at_utc": _iso_utc(started),
        "ended_at_utc": _iso_utc(ended),
        "duration_seconds": max(0.0, (ended - started).total_seconds()),
        "status": status,
        "exit_code": exit_code,
        "cwd": _relpath_or_abs(root, cwd),
        "cmd": cmd,
        "cmd_display": _cmd_display(cmd),
        "git": {
            "head": git_head,
            "is_dirty": bool(git_status),
            "status_porcelain_v1_path": "GIT_STATUS_PORCELAIN.txt" if git_status else "",
            "status_porcelain_v1_sha256": git_status_sha,
            "status_porcelain_v1_lines": len(git_status.splitlines()) if git_status else 0,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "outputs": {
            "run_dir": _relpath_or_abs(root, run_dir),
            "manifest": "RUN_MANIFEST.json",
            "stdout_log": "stdout.log",
            "stderr_log": "stderr.log",
            "artifacts_dir": "artifacts/",
            "checks_dir": "checks/",
        },
        "hashes": {
            "stdout_sha256": _sha256_file(stdout_path) if stdout_path.exists() else "",
            "stderr_sha256": _sha256_file(stderr_path) if stderr_path.exists() else "",
        },
        "note": args.note.strip(),
    }
    _write_json(run_dir / "RUN_MANIFEST.json", manifest)

    ledger_path: Path | None = None
    updated = False
    ledger_arg = (args.ledger or "").strip()
    if ledger_arg.lower() != "none":
        ledger_path = Path(ledger_arg) if ledger_arg else _default_ledger_path(root)
        if not ledger_path.is_absolute():
            ledger_path = (root / ledger_path).resolve()
        entry = _make_entry_md(
            run_id=run_id,
            when_utc=_iso_utc(started),
            tool_id=tool_id,
            cmd_display=_cmd_display(cmd),
            cwd_display=_relpath_or_abs(root, cwd),
            run_dir_rel=_relpath_or_abs(root, run_dir) + "/",
            status=status,
            exit_code=exit_code,
            note=args.note,
        )
        updated = _append_ledger_entry(ledger_path, entry, run_id=run_id)

    out = {
        "run_id": run_id,
        "run_dir": _relpath_or_abs(root, run_dir),
        "manifest": _relpath_or_abs(root, run_dir / "RUN_MANIFEST.json"),
        "ledger": _relpath_or_abs(root, ledger_path) if ledger_path else "",
        "ledger_updated": updated,
        "exit_code": exit_code,
        "status": status,
    }
    if args.json:
        sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"[qa_trace_ledger] run: {out['run_dir']} (status={status}, exit_code={exit_code})\n")
        if ledger_path:
            sys.stdout.write(f"[qa_trace_ledger] ledger: {out['ledger']} (updated={updated})\n")
    return exit_code


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="qa_trace_ledger.py", description="Trace run ledger helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="Create a run folder + manifest (no command execution).")
    sub.add_parser("run", help="Run a command and capture logs/manifest.")
    args, rest = ap.parse_known_args(argv)

    if args.cmd == "init":
        return cmd_init(rest)
    if args.cmd == "run":
        return cmd_run(rest)
    raise SystemExit(f"Unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
