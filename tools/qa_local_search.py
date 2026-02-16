"""
Local QA Search Index (SQLite FTS5)

Why:
  - Fast, offline search over the repo + forensics metadata
  - Designed to be callable by local agents (optionally JSON output)

Entry points:
  - Build:   python tools/qa_local_search.py build
  - Search:  python tools/qa_local_search.py search "meta validator" --top 10
  - Inspect: python tools/qa_local_search.py show qa_alphageometry_ptolemy/qa_meta_validator.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


UTC = dt.timezone.utc


# --- Defaults ---

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "target",
    "qa_venv",
    "venv",
}

CORE_SCAN_DIRS = (
    "qa_alphageometry_ptolemy",
    "qa_alphageometry",
    "qa_competency",
    "qa_agent_security",
    "docs",
    "Documents",
    "tools",
    "demos",
    "tests",
    "vault_audit",
    "results",
    "artifacts",
)

INDEXABLE_EXTS = {
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".sh",
    ".csv",
    ".tla",
    ".tex",
}


# --- Forensics inputs (from tools/project_forensics.py) ---


def find_latest_forensics_dir(root: Path) -> Path:
    base = root / "_forensics"
    if not base.exists():
        raise FileNotFoundError("Missing _forensics/. Run `python tools/project_forensics.py` first.")
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("forensics_")]
    if not runs:
        raise FileNotFoundError("No forensics runs found. Run `python tools/project_forensics.py` first.")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def parse_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return rows
        cols = header.rstrip("\n").split("\t")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < len(cols):
                parts += [""] * (len(cols) - len(parts))
            rows.append({cols[i]: parts[i] for i in range(len(cols))})
    return rows


@dataclass
class ForensicsMeta:
    hotspots: dict[str, tuple[int, int, int]]  # path -> (score, evidence, claims)
    chat_mentions: dict[str, int]  # path -> count
    script_artifacts: dict[str, list[str]]  # script -> artifact paths


def load_forensics_meta(forensics_dir: Path) -> ForensicsMeta:
    hotspots_path = forensics_dir / "keyword_hotspots.tsv"
    chat_mentions_path = forensics_dir / "chat_path_mentions.tsv"
    artifacts_path = forensics_dir / "script_artifacts.tsv"

    hotspots: dict[str, tuple[int, int, int]] = {}
    if hotspots_path.exists():
        for row in parse_tsv(hotspots_path):
            path = (row.get("path") or "").strip()
            if not path:
                continue
            try:
                score = int(row.get("score") or "0")
                evidence = int(row.get("evidence") or "0")
                claims = int(row.get("claims") or "0")
            except ValueError:
                continue
            hotspots[path] = (score, evidence, claims)

    chat_mentions: dict[str, int] = {}
    if chat_mentions_path.exists():
        for row in parse_tsv(chat_mentions_path):
            path = (row.get("path") or "").strip()
            if not path:
                continue
            try:
                chat_mentions[path] = int(row.get("count") or "0")
            except ValueError:
                continue

    script_artifacts: dict[str, list[str]] = {}
    if artifacts_path.exists():
        for row in parse_tsv(artifacts_path):
            script = (row.get("script") or "").strip()
            resolved = (row.get("resolved_path") or "").strip()
            exists = (row.get("exists") or "").strip()
            if not script or not resolved:
                continue
            if exists != "1":
                continue
            script_artifacts.setdefault(script, [])
            if resolved not in script_artifacts[script]:
                script_artifacts[script].append(resolved)

    for k in list(script_artifacts.keys()):
        script_artifacts[k].sort()

    return ForensicsMeta(hotspots=hotspots, chat_mentions=chat_mentions, script_artifacts=script_artifacts)


# --- Repo scan ---


def _git_ls_files(root: Path) -> set[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def _git_head(root: Path) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        return None


def _is_text_ext(path: Path) -> bool:
    return path.suffix.lower() in INDEXABLE_EXTS


def iter_repo_files(
    root: Path,
    *,
    scope: str,
    include: list[str],
    max_bytes: int,
) -> Iterator[Path]:
    scan_roots: list[Path] = []
    if scope == "core":
        # Root files only (no huge walk)
        for p in root.iterdir():
            if p.is_file():
                scan_roots.append(p)
        for name in CORE_SCAN_DIRS:
            p = root / name
            if p.exists() and p.is_dir():
                scan_roots.append(p)
        for inc in include:
            p = root / inc
            if p.exists():
                scan_roots.append(p)
    else:
        scan_roots.append(root)

    seen: set[Path] = set()
    for base in scan_roots:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)
        if base.is_file():
            if _is_text_ext(base) and base.stat().st_size <= max_bytes:
                yield base
            continue

        for dirpath, dirnames, filenames in os.walk(base, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for name in filenames:
                p = Path(dirpath) / name
                try:
                    if not _is_text_ext(p):
                        continue
                    if p.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
                yield p


# --- Title/category inference (best-effort) ---


def _read_text_limited(path: Path, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _first_heading(md: str) -> str | None:
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            t = s.lstrip("#").strip()
            if t:
                return t
    return None


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip().lstrip("\ufeff").strip()
        if s:
            return s
    return None


def _first_docstring_line(py: str) -> str | None:
    lines = py.splitlines()
    i = 0
    if lines and lines[0].startswith("#!"):
        i += 1
    if i < len(lines) and re.search(r"coding[:=]\s*[-\w.]+", lines[i]):
        i += 1

    candidate = "\n".join(lines[i:]).lstrip()
    if candidate.startswith('"""') or candidate.startswith("'''"):
        quote = candidate[:3]
        rest = candidate[3:]
        end = rest.find(quote)
        if end == -1:
            return None
        body = rest[:end]
        return _first_nonempty_line(body)
    return None


def _tex_title(tex: str) -> str | None:
    m = re.search(r"\\title\\{([^}]{1,200})\\}", tex)
    if m:
        return m.group(1).strip()
    return None


def infer_title(root: Path, rel_path: str) -> str:
    path = root / rel_path
    ext = path.suffix.lower()
    text = _read_text_limited(path, max_bytes=200_000)
    if text:
        if ext == ".md":
            h = _first_heading(text)
            if h:
                return h
        if ext == ".py":
            d = _first_docstring_line(text)
            if d:
                return d
        if ext == ".tex":
            t = _tex_title(text)
            if t:
                return t
        line = _first_nonempty_line(text)
        if line:
            return line[:120]
    return Path(rel_path).name


def infer_category(rel_path: str) -> str:
    p = rel_path.replace("\\", "/")
    if p.startswith("qa_alphageometry_ptolemy/"):
        if "meta_validator" in p:
            return "Meta-validator"
        if "validator" in p or p.endswith("_validate.py"):
            return "Validator"
        if "/certs/" in p or p.endswith("_certificate.py"):
            return "Certificate schema"
        return "QA core (ptolemy)"
    if p.startswith("qa_alphageometry/"):
        return "QA core (alphageometry)"
    if p.startswith("qa_competency/"):
        return "QA competency"
    if p.startswith("qa_agent_security/"):
        return "Security/guardrails"
    if p.startswith("docs/Google AI Studio/"):
        return "Imported notes"
    if p.startswith("docs/ai_chats/"):
        return "Imported notes"
    if p.startswith("docs/"):
        return "Docs"
    if p.startswith("Documents/"):
        return "Paper/docs"
    if p.startswith("demos/"):
        return "Demo"
    if p.startswith("tools/"):
        return "Tooling"
    return "Misc"


# --- SQLite index ---


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS build_info (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_meta (
  path TEXT PRIMARY KEY,
  ext TEXT,
  size_bytes INTEGER,
  mtime_utc TEXT,
  tracked INTEGER,
  title TEXT,
  category TEXT,
  hotspot_score INTEGER,
  hotspot_evidence INTEGER,
  hotspot_claims INTEGER,
  chat_mentions INTEGER,
  artifact_count INTEGER
);

CREATE VIRTUAL TABLE IF NOT EXISTS file_fts USING fts5(
  path,
  title,
  body,
  tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS script_artifacts (
  script TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  PRIMARY KEY (script, artifact_path)
);

CREATE INDEX IF NOT EXISTS idx_script_artifacts_script ON script_artifacts(script);
"""


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA_SQL)


def reset_db(con: sqlite3.Connection) -> None:
    # NOTE: Some SQLite builds have trouble dropping FTS5 virtual tables while in WAL mode
    # (we've observed `OperationalError: unable to open database file`). Switch to DELETE
    # journaling before dropping, then re-enable WAL via SCHEMA_SQL.
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    except sqlite3.OperationalError:
        pass
    try:
        con.execute("PRAGMA journal_mode=DELETE;")
    except sqlite3.OperationalError:
        pass
    con.executescript(
        """
        DROP TABLE IF EXISTS build_info;
        DROP TABLE IF EXISTS file_meta;
        DROP TABLE IF EXISTS script_artifacts;
        DROP TABLE IF EXISTS file_fts;
        """
    )
    con.executescript(SCHEMA_SQL)


def build_index(
    *,
    root: Path,
    db_path: Path,
    forensics_dir: Path,
    scope: str,
    include: list[str],
    max_bytes: int,
    overwrite: bool,
) -> None:
    if db_path.exists() and not overwrite:
        raise FileExistsError(f"DB already exists: {db_path} (use --overwrite)")

    con = connect_db(db_path)
    try:
        if overwrite:
            reset_db(con)
        else:
            init_db(con)

        meta = load_forensics_meta(forensics_dir)
        tracked = _git_ls_files(root)

        # Script artifacts table
        con.execute("DELETE FROM script_artifacts")
        for script, artifacts in meta.script_artifacts.items():
            for a in artifacts:
                con.execute(
                    "INSERT OR REPLACE INTO script_artifacts(script, artifact_path) VALUES(?,?)",
                    (script, a),
                )

        con.execute("DELETE FROM file_meta")
        con.execute("DELETE FROM file_fts")

        count = 0
        with con:
            for path in iter_repo_files(root, scope=scope, include=include, max_bytes=max_bytes):
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue

                ext = path.suffix.lower()
                text = _read_text_limited(path, max_bytes=max_bytes)
                if text is None:
                    continue

                title = infer_title(root, rel)
                category = infer_category(rel)

                hs = meta.hotspots.get(rel, (0, 0, 0))
                chat_count = meta.chat_mentions.get(rel, 0)
                artifact_count = len(meta.script_artifacts.get(rel, []))

                con.execute(
                    """
                    INSERT OR REPLACE INTO file_meta(
                      path, ext, size_bytes, mtime_utc, tracked, title, category,
                      hotspot_score, hotspot_evidence, hotspot_claims,
                      chat_mentions, artifact_count
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        rel,
                        ext,
                        int(st.st_size),
                        dt.datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
                        int(rel in tracked),
                        title,
                        category,
                        int(hs[0]),
                        int(hs[1]),
                        int(hs[2]),
                        int(chat_count),
                        int(artifact_count),
                    ),
                )
                con.execute(
                    "INSERT INTO file_fts(path, title, body) VALUES(?,?,?)",
                    (rel, title, text),
                )
                count += 1

        # Build metadata
        con.execute("DELETE FROM build_info")
        info = {
            "generated_utc": dt.datetime.now(tz=UTC).isoformat(),
            "repo_root": str(root),
            "git_head": _git_head(root) or "",
            "forensics_dir": str(forensics_dir),
            "scope": scope,
            "max_bytes": str(max_bytes),
            "indexed_files": str(count),
        }
        for k, v in info.items():
            con.execute("INSERT OR REPLACE INTO build_info(k, v) VALUES(?,?)", (k, v))
        con.commit()

        print(f"[qa_local_search] indexed {count} files into {db_path}", file=sys.stderr)
    finally:
        con.close()


def _rows_to_jsonable(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    return out


def search_index(
    *,
    db_path: Path,
    query: str,
    top: int,
    json_out: bool,
) -> int:
    con = connect_db(db_path)
    try:
        sql = """
        SELECT
          m.path,
          m.title,
          m.category,
          m.hotspot_score,
          m.chat_mentions,
          m.artifact_count,
          bm25(file_fts) AS rank,
          snippet(file_fts, 2, '[', ']', '…', 12) AS snippet
        FROM file_fts
        JOIN file_meta m ON m.path = file_fts.path
        WHERE file_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """
        rows = list(con.execute(sql, (query, int(top))))
        if json_out:
            print(json.dumps({"query": query, "results": _rows_to_jsonable(rows)}, indent=2))
        else:
            for r in rows:
                print(f"{r['path']} :: {r['title']} (rank={r['rank']:.3f})")
                print(f"  {r['snippet']}")
        return 0
    finally:
        con.close()


def show_path(
    *,
    db_path: Path,
    path: str,
    json_out: bool,
) -> int:
    con = connect_db(db_path)
    try:
        row = con.execute("SELECT * FROM file_meta WHERE path = ?", (path,)).fetchone()
        if row is None:
            print(f"[qa_local_search] not found in index: {path}", file=sys.stderr)
            return 2

        artifacts = [r["artifact_path"] for r in con.execute("SELECT artifact_path FROM script_artifacts WHERE script = ? ORDER BY artifact_path", (path,))]

        payload = {k: row[k] for k in row.keys()}
        payload["artifacts"] = artifacts

        if json_out:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Path: {payload['path']}")
            print(f"Title: {payload.get('title')}")
            print(f"Category: {payload.get('category')}")
            print(f"Tracked: {payload.get('tracked')}  Ext: {payload.get('ext')}")
            print(f"Size: {payload.get('size_bytes')}  Mtime: {payload.get('mtime_utc')}")
            print(
                "Hotspot: "
                f"score={payload.get('hotspot_score')} "
                f"evidence={payload.get('hotspot_evidence')} "
                f"claims={payload.get('hotspot_claims')}"
            )
            print(f"Chat mentions: {payload.get('chat_mentions')}  Artifact count: {payload.get('artifact_count')}")
            if artifacts:
                print("Artifacts:")
                for a in artifacts[:25]:
                    print(f"  - {a}")
                if len(artifacts) > 25:
                    print(f"  - (+{len(artifacts) - 25} more)")
        return 0
    finally:
        con.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local repo search index (SQLite FTS5).")
    parser.add_argument("--db", default="_forensics/qa_local_search.sqlite", help="SQLite db path.")
    parser.add_argument("--forensics-dir", default=None, help="Use specific _forensics/forensics_* run.")
    parser.add_argument("--scope", choices=("core", "full"), default="core")
    parser.add_argument("--include", action="append", default=[], help="Extra relative path(s) to include (core scope).")
    parser.add_argument("--max-bytes", type=int, default=1_000_000, help="Max file size to index as text.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing db.")
    parser.add_argument("--json", dest="json_out", action="store_true", help="JSON output (for agents).")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Build the SQLite index")

    s = sub.add_parser("search", help="FTS search")
    s.add_argument("query", help="FTS query string")
    s.add_argument("--top", type=int, default=10, help="Max results")

    sh = sub.add_parser("show", help="Show metadata for one path")
    sh.add_argument("path", help="Repo-relative path")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = (root / db_path).resolve()

    forensics_dir = Path(args.forensics_dir) if args.forensics_dir else find_latest_forensics_dir(root)
    if not forensics_dir.is_absolute():
        forensics_dir = (root / forensics_dir).resolve()

    if args.cmd == "build":
        build_index(
            root=root,
            db_path=db_path,
            forensics_dir=forensics_dir,
            scope=args.scope,
            include=args.include,
            max_bytes=int(args.max_bytes),
            overwrite=bool(args.overwrite),
        )
        return 0

    if args.cmd == "search":
        return search_index(
            db_path=db_path,
            query=str(args.query),
            top=int(args.top),
            json_out=bool(args.json_out),
        )

    if args.cmd == "show":
        return show_path(
            db_path=db_path,
            path=str(args.path),
            json_out=bool(args.json_out),
        )

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
