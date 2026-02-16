"""
Project Forensics (Repo + Chat Export)

Entry point:
  python tools/project_forensics.py

Default outputs:
  - Raw artifacts: `_forensics/<run_id>/...`
  - Human report:  `_forensics/<run_id>/REPORT.md`

Privacy:
  This tool avoids emitting raw ChatGPT message text. It only aggregates counts
  of paths/commands/keywords.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


# --- Config ---

DEFAULT_MAX_TEXT_BYTES = 2_000_000
DEFAULT_MAX_HASH_BYTES = 5_000_000

# User-provided constraint: QA control-theorem formalization date (UTC).
# Any file with mtime strictly before this cutoff should be treated as legacy and re-vetted.
DEFAULT_CONTROL_CUTOFF_UTC_DATE = "2026-01-10"

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
    "vault_audit",
    "tools",
    "demos",
    "tests",
    "results",
    "artifacts",
)

TEXT_EXTS = {
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
    ".cff",
}

HASH_EXTS = {
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".sh",
    ".tla",
    ".tex",
}

CLAIM_RE = re.compile(r"(?i)\b(theorem|lemma|corollary|proposition|axiom|conjecture|proof|qed|∎|□)\b")
EVIDENCE_RE = re.compile(r"(?i)\b(pass|failed|failure|verified|verification|meta-validator|self-test|witness|certpack)\b")

PATH_WITH_SLASH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,10})"
)
BARE_FILENAME_RE = re.compile(r"\b([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,10})\b")

CMD_LINE_RE = re.compile(r"(?m)^\s*(python3?|pip|pytest|cargo|make|just|git)(?=\s|$)[^\n\r]*$")
PY_TARGET_RE = re.compile(r"(?P<path>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py)\b")

KNOWN_PATH_SUFFIXES = {
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
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tex",
    ".tla",
    ".odt",
    ".zip",
}

ARTIFACT_EXTS = (
    "png",
    "csv",
    "json",
    "jsonl",
    "log",
    "txt",
    "pdf",
    "npy",
    "npz",
    "pth",
    "pt",
)

ARTIFACT_TOKEN_RE = re.compile(
    r"(?P<quote>['\"])(?P<token>[^'\"\n\r]{1,240}\.(?:"
    + "|".join(re.escape(ext) for ext in ARTIFACT_EXTS)
    + r"))(?P=quote)"
)


# --- Utility ---


def _utc_now_stamp() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _human_bytes(num_bytes: int) -> str:
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_path(raw: str) -> str:
    path = raw.strip()
    path = path.strip("`'\"()[]{}<>,.:;")
    path = path.replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_text_limited(path: Path, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _score_keywords(claims: int, evidence: int) -> int:
    return (3 * evidence) + (1 * claims)


def _normalize_family_key(stem: str) -> str:
    key = stem.lower()
    key = re.sub(r"\bcopy\b", "", key)
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")

    # Iteratively strip common suffixes / version tokens.
    suffix_re = re.compile(
        r"(?:_|-)(final|corrected|fixed|enhanced|quick|fast|baseline|debug|old|new|v\d+(?:[._]\d+)*)$",
        re.IGNORECASE,
    )
    while True:
        new_key = re.sub(suffix_re, "", key).strip("_-")
        new_key = re.sub(r"\d+$", "", new_key).strip("_-")
        if new_key == key:
            break
        key = new_key
    return key


def _top_n(items: Iterator[tuple[str, int]], n: int) -> list[tuple[str, int]]:
    return sorted(items, key=lambda t: t[1], reverse=True)[:n]


# --- Repo scan ---


@dataclass
class RepoForensicsConfig:
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES
    control_cutoff_utc_date: str = DEFAULT_CONTROL_CUTOFF_UTC_DATE
    control_cutoff_ts: float = 0.0
    skip_dir_names: set[str] = None  # type: ignore[assignment]
    scan_dirs: list[Path] = None  # type: ignore[assignment]
    content_top_allow: set[str] = None  # type: ignore[assignment]
    progress_every: int = 0

    def __post_init__(self) -> None:
        if not self.control_cutoff_ts:
            # Interpret YYYY-MM-DD in UTC at 00:00:00.
            try:
                d = dt.date.fromisoformat(self.control_cutoff_utc_date)
                self.control_cutoff_ts = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()
            except Exception:
                self.control_cutoff_ts = 0.0
        if self.skip_dir_names is None:
            self.skip_dir_names = set(SKIP_DIR_NAMES)
        if self.scan_dirs is None:
            self.scan_dirs = []
        if self.content_top_allow is None:
            # Default: only read/hash content under these top-level dirs + root files.
            self.content_top_allow = {
                "qa_alphageometry_ptolemy",
                "qa_alphageometry",
                "qa_competency",
                "qa_agent_security",
                "docs",
                "Documents",
                "vault_audit",
                "tools",
                "demos",
                "tests",
            }


@dataclass
class RepoForensicsResult:
    scanned_files: int
    scanned_bytes: int
    control_cutoff_utc_date: str
    pre_control_files: int
    pre_control_bytes: int
    post_control_files: int
    post_control_bytes: int
    ext_counts: collections.Counter[str]
    top_level_bytes: collections.Counter[str]
    largest_files: list[tuple[int, str]]
    staleness_buckets: collections.Counter[str]

    dupes_by_hash: dict[str, list[str]]
    basenames: dict[str, list[str]]
    families: dict[tuple[str, str], list[str]]

    keyword_hits: list[tuple[int, int, int, str]]  # (score, evidence, claims, path)

    git_head: str | None
    git_dirty_summary: dict[str, int]
    script_artifacts: dict[str, list[tuple[str, str, bool]]]  # script -> [(token, resolved_rel, exists)]


def iter_repo_files(root: Path, skip_dir_names: set[str]) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in skip_dir_names]
        for name in filenames:
            yield Path(dirpath) / name


def iter_repo_files_scoped(root: Path, scan_dirs: list[Path], skip_dir_names: set[str]) -> Iterator[Path]:
    if not scan_dirs:
        yield from iter_repo_files(root, skip_dir_names)
        return

    for base in scan_dirs:
        base_path = (root / base).resolve() if not base.is_absolute() else base
        if not base_path.exists():
            continue
        if base_path.is_file():
            yield base_path
            continue
        if not base_path.is_dir():
            continue

        for dirpath, dirnames, filenames in os.walk(base_path, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in skip_dir_names]
            for name in filenames:
                yield Path(dirpath) / name


def classify_staleness(now_ts: float, mtime_ts: float) -> str:
    age_days = max(0.0, (now_ts - mtime_ts) / 86400.0)
    if age_days <= 7:
        return "0-7d"
    if age_days <= 30:
        return "8-30d"
    if age_days <= 90:
        return "31-90d"
    if age_days <= 365:
        return "91-365d"
    return "365d+"


def git_head_and_dirty(root: Path) -> tuple[str | None, dict[str, int]]:
    head: str | None = None
    try:
        head = (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        head = None

    summary = {"untracked": 0, "modified": 0, "added": 0, "deleted": 0, "renamed": 0, "other": 0}
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if not line:
                continue
            if line.startswith("?? "):
                summary["untracked"] += 1
                continue
            code = line[:2]
            if "R" in code:
                summary["renamed"] += 1
            elif "A" in code:
                summary["added"] += 1
            elif "D" in code:
                summary["deleted"] += 1
            elif "M" in code:
                summary["modified"] += 1
            else:
                summary["other"] += 1
    except Exception:
        pass

    return head, summary


def scan_repo(root: Path, cfg: RepoForensicsConfig) -> RepoForensicsResult:
    ext_counts: collections.Counter[str] = collections.Counter()
    top_level_bytes: collections.Counter[str] = collections.Counter()
    staleness_buckets: collections.Counter[str] = collections.Counter()
    basenames: dict[str, list[str]] = collections.defaultdict(list)
    families: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    keyword_hits: list[tuple[int, int, int, str]] = []

    dupes_by_hash_all: dict[str, list[str]] = collections.defaultdict(list)

    largest_files: list[tuple[int, str]] = []
    scanned_files = 0
    scanned_bytes = 0
    pre_control_files = 0
    pre_control_bytes = 0
    post_control_files = 0
    post_control_bytes = 0
    now_ts = dt.datetime.now(tz=dt.timezone.utc).timestamp()

    for path in iter_repo_files_scoped(root, cfg.scan_dirs, cfg.skip_dir_names):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue

        try:
            st = path.stat()
        except OSError:
            continue

        size = st.st_size
        scanned_files += 1
        scanned_bytes += size

        if cfg.control_cutoff_ts and st.st_mtime < cfg.control_cutoff_ts:
            pre_control_files += 1
            pre_control_bytes += size
        else:
            post_control_files += 1
            post_control_bytes += size

        ext = path.suffix.lower() or "<none>"
        ext_counts[ext] += 1

        parts = Path(rel).parts
        if not parts:
            top = "<root>"
        elif len(parts) == 1:
            top = "<root_files>"
        else:
            top = parts[0]
        top_level_bytes[top] += size

        staleness_buckets[classify_staleness(now_ts, st.st_mtime)] += 1

        # Largest files list (keep top 50)
        largest_files.append((size, rel))
        largest_files.sort(key=lambda t: t[0], reverse=True)
        if len(largest_files) > 50:
            largest_files = largest_files[:50]

        base = path.name
        if len(basenames[base]) < 30:
            basenames[base].append(rel)

        stem = path.stem
        family_key = _normalize_family_key(stem)
        if family_key and family_key != stem.lower():
            families[(family_key, ext)].append(rel)

        if cfg.progress_every and (scanned_files % cfg.progress_every == 0):
            print(f"[project_forensics] scanned {scanned_files} files...", file=sys.stderr)

        allow_content = (top == "<root>") or (top in cfg.content_top_allow)

        # Keyword scan (text only)
        if allow_content and ext in TEXT_EXTS:
            text = _read_text_limited(path, cfg.max_text_bytes)
            if text is not None:
                claims = len(CLAIM_RE.findall(text))
                evidence = len(EVIDENCE_RE.findall(text))
                if claims or evidence:
                    score = _score_keywords(claims=claims, evidence=evidence)
                    keyword_hits.append((score, evidence, claims, rel))

        # Exact duplicate scan (hash small text/code files only)
        if allow_content and ext in HASH_EXTS and size <= cfg.max_hash_bytes:
            try:
                digest = _sha256_file(path)
            except OSError:
                digest = ""
            if digest:
                dupes_by_hash_all[digest].append(rel)

    dupes_by_hash = {h: paths for h, paths in dupes_by_hash_all.items() if len(paths) > 1}

    keyword_hits.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    keyword_hits = keyword_hits[:300]

    # Filter families to only multi-variant groups
    families_multi: dict[tuple[str, str], list[str]] = {
        k: sorted(v) for k, v in families.items() if len(v) > 1
    }

    head, dirty = git_head_and_dirty(root)

    return RepoForensicsResult(
        scanned_files=scanned_files,
        scanned_bytes=scanned_bytes,
        control_cutoff_utc_date=cfg.control_cutoff_utc_date,
        pre_control_files=pre_control_files,
        pre_control_bytes=pre_control_bytes,
        post_control_files=post_control_files,
        post_control_bytes=post_control_bytes,
        ext_counts=ext_counts,
        top_level_bytes=top_level_bytes,
        largest_files=sorted(largest_files, key=lambda t: t[0], reverse=True),
        staleness_buckets=staleness_buckets,
        dupes_by_hash=dupes_by_hash,
        basenames=dict(basenames),
        families=families_multi,
        keyword_hits=keyword_hits,
        git_head=head,
        git_dirty_summary=dirty,
        script_artifacts={},
    )


# --- Chat export scan ---


def iter_json_array_items(text_stream: io.TextIOBase) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0

    def refill() -> bool:
        nonlocal buf
        chunk = text_stream.read(1_048_576)
        if chunk == "":
            return False
        buf += chunk
        return True

    # Seek to the opening '['
    while True:
        if pos >= len(buf) and not refill():
            raise ValueError("Empty JSON stream (no '[' found).")
        while pos < len(buf) and buf[pos].isspace():
            pos += 1
        if pos < len(buf):
            if buf[pos] == "\ufeff":
                pos += 1
                continue
            if buf[pos] != "[":
                raise ValueError(f"Expected '[' at position {pos}, got {buf[pos]!r}.")
            pos += 1
            break

    while True:
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf):
                break
            if not refill():
                return

        if buf[pos] == "]":
            return

        try:
            item, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if not refill():
                snippet = buf[pos : pos + 200]
                raise ValueError(f"Invalid or truncated JSON near: {snippet!r}") from None
            continue

        yield item
        pos = end

        if pos > 2_000_000:
            buf = buf[pos:]
            pos = 0


def extract_text_from_message(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if not isinstance(content, Mapping):
        return None
    if content.get("content_type") != "text":
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    text_parts = [p for p in parts if isinstance(p, str)]
    if not text_parts:
        return None
    return "\n".join(text_parts)


@dataclass
class ChatForensicsResult:
    conversations: int
    messages_text: int
    path_mentions: collections.Counter[str]
    filename_mentions: collections.Counter[str]
    command_programs: collections.Counter[str]
    python_targets: collections.Counter[str]


def find_latest_chat_export_zip(root: Path) -> Path | None:
    chat_dir = root / "chat_data"
    if not chat_dir.exists():
        return None
    zips = sorted(chat_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def scan_chat_export(zip_path: Path) -> ChatForensicsResult:
    path_mentions: collections.Counter[str] = collections.Counter()
    filename_mentions: collections.Counter[str] = collections.Counter()
    command_programs: collections.Counter[str] = collections.Counter()
    python_targets: collections.Counter[str] = collections.Counter()

    conversations = 0
    messages_text = 0

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("conversations.json", "r") as raw:
            text_stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            for conv in iter_json_array_items(text_stream):
                if not isinstance(conv, Mapping):
                    continue
                conversations += 1
                mapping = conv.get("mapping")
                if not isinstance(mapping, Mapping):
                    continue

                for node in mapping.values():
                    if not isinstance(node, Mapping):
                        continue
                    message = node.get("message")
                    if not isinstance(message, Mapping):
                        continue
                    text = extract_text_from_message(message)
                    if not text:
                        continue
                    messages_text += 1

                    # Paths with slashes (prefer these)
                    if "/" in text or "\\" in text:
                        for m in PATH_WITH_SLASH_RE.finditer(text):
                            raw_path = m.group("path")
                            path = _normalize_path(raw_path)
                            if len(path) > 240:
                                continue
                            suffix = Path(path).suffix.lower()
                            if suffix in KNOWN_PATH_SUFFIXES:
                                path_mentions[path] += 1

                    # Bare filenames (often ambiguous)
                    if "." in text:
                        for m in BARE_FILENAME_RE.finditer(text):
                            name = m.group(1)
                            if "/" in name or "\\" in name:
                                continue
                            suffix = Path(name).suffix.lower()
                            if suffix in KNOWN_PATH_SUFFIXES:
                                filename_mentions[name] += 1

                    # Commands (line-based)
                    for m in CMD_LINE_RE.finditer(text):
                        line = m.group(0).strip()
                        program = line.split(maxsplit=1)[0].lower()
                        command_programs[program] += 1
                        if program.startswith("python"):
                            mt = PY_TARGET_RE.search(line)
                            if mt:
                                tgt = _normalize_path(mt.group("path"))
                                python_targets[tgt] += 1

    return ChatForensicsResult(
        conversations=conversations,
        messages_text=messages_text,
        path_mentions=path_mentions,
        filename_mentions=filename_mentions,
        command_programs=command_programs,
        python_targets=python_targets,
    )


def resolve_python_targets(
    *,
    root: Path,
    python_targets: collections.Counter[str],
    basenames: Mapping[str, list[str]],
) -> collections.Counter[str]:
    resolved: collections.Counter[str] = collections.Counter()
    for target, count in python_targets.items():
        if "/" in target or "\\" in target:
            resolved[target] += count
            continue
        if (root / target).exists():
            resolved[target] += count
            continue
        candidates = basenames.get(target)
        if candidates and len(candidates) == 1:
            resolved[candidates[0]] += count
            continue
        resolved[target] += count
    return resolved


# --- Reporting ---


def write_tsv(path: Path, header: str, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header.rstrip("\n") + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def render_report(
    *,
    root: Path,
    run_id: str,
    repo: RepoForensicsResult,
    du_top: list[tuple[str, int]] | None,
    chat_zip: Path | None,
    chat: ChatForensicsResult | None,
) -> str:
    lines: list[str] = []
    lines.append("# Project Forensics Report")
    lines.append("")
    lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Repo root: `{root}`")
    lines.append(f"- Generated: `{dt.datetime.now(tz=dt.timezone.utc).isoformat()}`")
    if repo.git_head:
        lines.append(f"- Git HEAD: `{repo.git_head}`")
    lines.append(
        "- Git working tree (counts): "
        + ", ".join(f"`{k}={v}`" for k, v in repo.git_dirty_summary.items())
    )
    lines.append("")

    if du_top:
        lines.append("## Census (Whole Repo, top-level by `du`)")
        for path, b in du_top[:25]:
            lines.append(f"- `{path}`: `{_human_bytes(b)}`")
        lines.append("")

    lines.append("## Census (Scanned Subtree)")
    lines.append(f"- Files scanned (excludes {sorted(SKIP_DIR_NAMES)} dirs): `{repo.scanned_files}`")
    lines.append(f"- Bytes scanned: `{_human_bytes(repo.scanned_bytes)}`")
    lines.append(f"- Control-theorem cutoff (UTC date): `{repo.control_cutoff_utc_date}`")
    lines.append(
        "- Legacy (pre-cutoff; revet required): "
        + f"`files={repo.pre_control_files}` `bytes={_human_bytes(repo.pre_control_bytes)}`"
    )
    lines.append(
        "- Post-cutoff: "
        + f"`files={repo.post_control_files}` `bytes={_human_bytes(repo.post_control_bytes)}`"
    )
    lines.append("")

    lines.append("### Top-Level Size (Approx, from scanned files)")
    for top, b in repo.top_level_bytes.most_common(25):
        lines.append(f"- `{top}`: `{_human_bytes(b)}`")
    lines.append("")

    lines.append("### Staleness (by mtime, scanned files)")
    for bucket, count in repo.staleness_buckets.most_common():
        lines.append(f"- `{bucket}`: `{count}`")
    lines.append("")

    lines.append("### Extensions (top)")
    for ext, count in repo.ext_counts.most_common(25):
        lines.append(f"- `{ext}`: `{count}`")
    lines.append("")

    lines.append("### Largest Files (top)")
    for size, rel in repo.largest_files[:25]:
        lines.append(f"- `{rel}`: `{_human_bytes(size)}`")
    lines.append("")

    lines.append("## Redundancy Signals")
    lines.append(f"- Exact duplicate groups (hashed <= {DEFAULT_MAX_HASH_BYTES} bytes, text/code): `{len(repo.dupes_by_hash)}`")
    if repo.dupes_by_hash:
        lines.append("")
        lines.append("### Exact Duplicates (sample)")
        shown = 0
        for digest, paths in sorted(repo.dupes_by_hash.items(), key=lambda t: len(t[1]), reverse=True):
            lines.append(f"- `{digest[:12]}`: `{len(paths)}` files (e.g. `{paths[0]}`)")
            shown += 1
            if shown >= 15:
                break
        lines.append("")

    # Version families: show a few with many variants
    lines.append("### Version Families (normalized stem)")
    family_items = sorted(repo.families.items(), key=lambda t: len(t[1]), reverse=True)
    if not family_items:
        lines.append("- (none detected)")
    else:
        for (key, ext), paths in family_items[:20]:
            lines.append(f"- `{key}{ext}`: `{len(paths)}` variants (e.g. `{paths[0]}`)")
    lines.append("")

    lines.append("## Claim/Evidence Hotspots (keyword counts)")
    if not repo.keyword_hits:
        lines.append("- (no keyword hits detected under scan limits)")
    else:
        for score, evidence, claims, rel in repo.keyword_hits[:30]:
            lines.append(f"- `{rel}`: score `{score}` (evidence={evidence}, claims={claims})")
    lines.append("")

    if repo.control_cutoff_utc_date:
        # Show top legacy hotspots for re-vetting.
        cutoff_ts = RepoForensicsConfig(control_cutoff_utc_date=repo.control_cutoff_utc_date).control_cutoff_ts
        legacy_hits: list[tuple[int, int, int, str, str]] = []  # score,e,c,path,mtime_utc
        for score, evidence, claims, rel in repo.keyword_hits:
            full = root / rel
            try:
                st = full.stat()
            except OSError:
                continue
            if cutoff_ts and st.st_mtime < cutoff_ts:
                mtime_utc = dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).strftime("%Y-%m-%d")
                legacy_hits.append((score, evidence, claims, rel, mtime_utc))
        if legacy_hits:
            legacy_hits.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            lines.append("## Legacy Hotspots (pre-cutoff; revet required)")
            for score, evidence, claims, rel, mtime_utc in legacy_hits[:25]:
                lines.append(f"- `{rel}`: score `{score}` (evidence={evidence}, claims={claims}) mtime_utc=`{mtime_utc}`")
            lines.append("")

    if repo.script_artifacts:
        lines.append("## Script → Artifact Links (existing files only)")
        shown = 0
        for script, artifacts in sorted(repo.script_artifacts.items(), key=lambda t: len(t[1]), reverse=True):
            # Keep report compact: show up to 3 artifacts per script
            sample = ", ".join(f"`{tok}`→`{res}`" for tok, res, _ in artifacts[:3])
            lines.append(f"- `{script}`: `{len(artifacts)}` artifacts ({sample})")
            shown += 1
            if shown >= 25:
                break
        lines.append("")

    lines.append("## Chat ↔ Repo Links")
    if chat_zip is None:
        lines.append("- Chat export: (not found)")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- Chat export zip: `{chat_zip}`")
    if chat is None:
        lines.append("- Chat parsing: (failed)")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- Conversations parsed: `{chat.conversations}`")
    lines.append(f"- Text messages parsed: `{chat.messages_text}`")
    lines.append("")

    lines.append("### Top Mentioned Repo Paths (existence check)")
    existing: list[tuple[str, int]] = []
    missing: list[tuple[str, int]] = []
    for p, c in chat.path_mentions.most_common(200):
        if (root / p).exists():
            existing.append((p, c))
        else:
            missing.append((p, c))
    for p, c in existing[:25]:
        lines.append(f"- `{p}`: `{c}`")
    lines.append("")

    if missing:
        lines.append("### Top Mentioned Missing/External Paths")
        for p, c in missing[:25]:
            lines.append(f"- `{p}`: `{c}`")
        lines.append("")

    lines.append("### Command Programs (chat)")
    for prog, c in chat.command_programs.most_common(20):
        lines.append(f"- `{prog}`: `{c}`")
    lines.append("")

    lines.append("### Python Targets (chat, existence check)")
    py_exist: list[tuple[str, int]] = []
    py_missing: list[tuple[str, int]] = []
    for p, c in chat.python_targets.most_common(200):
        if (root / p).exists():
            py_exist.append((p, c))
        else:
            py_missing.append((p, c))
    for p, c in py_exist[:25]:
        lines.append(f"- `{p}`: `{c}`")
    lines.append("")
    if py_missing:
        lines.append("### Python Targets (chat, missing/external)")
        for p, c in py_missing[:25]:
            lines.append(f"- `{p}`: `{c}`")
        lines.append("")

    return "\n".join(lines)


def resolve_artifact_token(root: Path, script_path: Path, token: str) -> tuple[str, bool] | None:
    tok = token.strip()
    tok = tok.replace("\\", "/")
    if tok.startswith(("http://", "https://")):
        return None
    if tok.startswith(("/", "~")):
        return None
    if any(ch in tok for ch in ("\n", "\r", "\t")):
        return None
    if len(tok) > 240:
        return None

    candidates = [root / tok, script_path.parent / tok]
    for cand in candidates:
        try:
            if cand.exists():
                try:
                    rel = cand.relative_to(root).as_posix()
                except ValueError:
                    rel = cand.as_posix()
                return (rel, True)
        except OSError:
            continue

    # Keep a normalized relative string for traceability, but mark missing.
    return (tok, False)


def scan_script_artifacts(
    *,
    root: Path,
    cfg: RepoForensicsConfig,
) -> dict[str, list[tuple[str, str, bool]]]:
    results: dict[str, list[tuple[str, str, bool]]] = collections.defaultdict(list)

    for path in iter_repo_files_scoped(root, cfg.scan_dirs, cfg.skip_dir_names):
        if path.suffix.lower() != ".py":
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        text = _read_text_limited(path, max(cfg.max_text_bytes, 200_000))
        if text is None:
            continue

        seen_tokens: set[str] = set()
        for m in ARTIFACT_TOKEN_RE.finditer(text):
            token = m.group("token")
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            resolved = resolve_artifact_token(root, path, token)
            if resolved is None:
                continue
            resolved_path, exists = resolved
            if not exists:
                continue
            results[rel].append((token, resolved_path, exists))

        # Stabilize output ordering
        results[rel].sort(key=lambda t: t[1])

    # Remove scripts with no existing artifacts
    return {k: v for k, v in results.items() if v}


# --- CLI ---


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forensic scan of project folder + chat export.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: _forensics/<run_id>).")
    parser.add_argument(
        "--scope",
        choices=("core", "full"),
        default="core",
        help="Scan scope. 'core' avoids massive data/build trees; 'full' walks everything (can take a long time).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional relative dir(s) to include in core scope (repeatable).",
    )
    parser.add_argument("--max-text-bytes", type=int, default=DEFAULT_MAX_TEXT_BYTES)
    parser.add_argument("--max-hash-bytes", type=int, default=DEFAULT_MAX_HASH_BYTES)
    parser.add_argument(
        "--control-cutoff-utc-date",
        default=DEFAULT_CONTROL_CUTOFF_UTC_DATE,
        help="UTC date (YYYY-MM-DD). Files with mtime strictly before this are flagged as legacy/revet-required.",
    )
    parser.add_argument("--chat-zip", default=None, help="ChatGPT export zip path (default: latest under chat_data/).")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N scanned files (0 disables).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    run_id = f"forensics_{_utc_now_stamp()}"
    out_dir = Path(args.out_dir) if args.out_dir else (root / "_forensics" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_dirs: list[Path] = []
    if args.scope == "core":
        # Root-level files only (avoid walking huge trees via ".")
        for p in root.iterdir():
            if p.name in SKIP_DIR_NAMES:
                continue
            if p.is_file():
                scan_dirs.append(Path(p.name))

        # Core directories (if present)
        for name in CORE_SCAN_DIRS:
            d = root / name
            if d.exists() and d.is_dir():
                scan_dirs.append(Path(name))

        # Add any results* dirs automatically
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("results") and Path(p.name) not in scan_dirs:
                scan_dirs.append(Path(p.name))

        for inc in args.include:
            scan_dirs.append(Path(inc))

    cfg = RepoForensicsConfig(
        max_text_bytes=args.max_text_bytes,
        max_hash_bytes=args.max_hash_bytes,
        control_cutoff_utc_date=args.control_cutoff_utc_date,
        scan_dirs=scan_dirs,
        progress_every=args.progress_every,
    )
    repo = scan_repo(root, cfg)
    repo.script_artifacts = scan_script_artifacts(root=root, cfg=cfg)

    # Whole-repo size census (top-level) via du.
    du_top: list[tuple[str, int]] | None = None
    try:
        proc = subprocess.run(
            ["du", "-B1", "-d", "2", "."],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        du_rows: list[tuple[str, int]] = []
        for line in proc.stdout.splitlines():
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            b = _safe_int(parts[0], default=0)
            p = parts[1].lstrip("./") or "."
            du_rows.append((p, b))
        du_rows.sort(key=lambda t: t[1], reverse=True)
        du_top = du_rows
        write_tsv(
            out_dir / "du_d2_bytes.tsv",
            "path\tbytes",
            [[p, str(b)] for p, b in du_rows],
        )
    except Exception:
        du_top = None

    # Persist core outputs
    (out_dir / "repo_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
                "git_head": repo.git_head,
                "git_dirty_summary": repo.git_dirty_summary,
                "scanned_files": repo.scanned_files,
                "scanned_bytes": repo.scanned_bytes,
                "control_cutoff_utc_date": repo.control_cutoff_utc_date,
                "pre_control_files": repo.pre_control_files,
                "pre_control_bytes": repo.pre_control_bytes,
                "post_control_files": repo.post_control_files,
                "post_control_bytes": repo.post_control_bytes,
                "top_level_bytes": repo.top_level_bytes,
                "ext_counts": repo.ext_counts,
                "staleness_buckets": repo.staleness_buckets,
            },
            indent=2,
            default=lambda o: dict(o),
        ),
        encoding="utf-8",
    )

    # Duplicates
    (out_dir / "dupes_by_hash.json").write_text(
        json.dumps(repo.dupes_by_hash, indent=2),
        encoding="utf-8",
    )

    # Families
    family_rows: list[list[str]] = []
    for (key, ext), paths in sorted(repo.families.items(), key=lambda t: len(t[1]), reverse=True):
        family_rows.append([f"{key}{ext}", str(len(paths)), paths[0], paths[-1]])
    write_tsv(out_dir / "version_families.tsv", "family\tcount\texample_first\texample_last", family_rows[:500])

    # Hotspots
    hotspot_rows = [
        [str(score), str(evidence), str(claims), path] for score, evidence, claims, path in repo.keyword_hits
    ]
    write_tsv(out_dir / "keyword_hotspots.tsv", "score\tevidence\tclaims\tpath", hotspot_rows)

    # Script artifacts
    artifact_rows: list[list[str]] = []
    for script, artifacts in sorted(repo.script_artifacts.items()):
        for token, resolved, exists in artifacts:
            artifact_rows.append([script, token, resolved, str(int(exists))])
    write_tsv(out_dir / "script_artifacts.tsv", "script\ttoken\tresolved_path\texists", artifact_rows)

    # Chat scan
    chat_zip: Path | None = None
    chat_result: ChatForensicsResult | None = None
    if args.chat_zip:
        chat_zip = (root / args.chat_zip).resolve() if not os.path.isabs(args.chat_zip) else Path(args.chat_zip)
    else:
        chat_zip = find_latest_chat_export_zip(root)
    if chat_zip and chat_zip.exists():
        try:
            chat_result = scan_chat_export(chat_zip)
            chat_result.python_targets = resolve_python_targets(
                root=root, python_targets=chat_result.python_targets, basenames=repo.basenames
            )

            chat_paths_rows = [[p, str(c), str(int((root / p).exists()))] for p, c in chat_result.path_mentions.most_common(2000)]
            write_tsv(out_dir / "chat_path_mentions.tsv", "path\tcount\texists", chat_paths_rows)

            chat_cmd_rows = [[p, str(c)] for p, c in chat_result.command_programs.most_common(200)]
            write_tsv(out_dir / "chat_command_programs.tsv", "program\tcount", chat_cmd_rows)

            py_rows = [[p, str(c), str(int((root / p).exists()))] for p, c in chat_result.python_targets.most_common(2000)]
            write_tsv(out_dir / "chat_python_targets.tsv", "path\tcount\texists", py_rows)
        except Exception as exc:
            print(f"[project_forensics] chat parse failed: {exc}", file=sys.stderr)

    report_md = render_report(
        root=root,
        run_id=run_id,
        repo=repo,
        du_top=du_top,
        chat_zip=chat_zip,
        chat=chat_result,
    )
    (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"[project_forensics] wrote: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
