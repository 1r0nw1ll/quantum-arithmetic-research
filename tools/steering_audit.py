"""
Steering / Chat Export / Repo Audit

Entry point:
  python tools/steering_audit.py --out chat_data/steering_audit.md

Notes:
- Reads `conversations.json` from a ChatGPT export ZIP (or extracted directory) without
  writing a full extraction to disk.
- Produces *aggregated* statistics only; no raw message text is emitted by default.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping


# --- Config ---

CHUNK_SIZE_CHARS = 1_048_576  # ~1 MiB text chunks for streaming JSON decode

STEERING_TERMS = (
    "axiom",
    "theorem",
    "invariant",
    "certificate",
    "validator",
    "proof",
    "drift",
    "reachability",
    "mapping",
    "schema",
    "witness",
    "failure",
    "recompute",
    "hook",
    "bundle",
    "generalization",
    "tla",
)

NORMATIVE_TERMS = (
    "must",
    "should",
    "never",
    "always",
    "do not",
    "don't",
)


# Prefer slash-paths (repo relative). Also capture root-level filenames via basename matching.
PATH_WITH_SLASH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,10})"
)
INLINE_CODE_RE = re.compile(r"`([^`]{1,400})`")
BARE_FILENAME_RE = re.compile(r"\b([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,10})\b")

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
}


LENGTH_BUCKETS = (
    (0, 20),
    (21, 50),
    (51, 100),
    (101, 200),
    (201, 500),
    (501, 1_000),
    (1_001, 2_000),
    (2_001, 5_000),
    (5_001, 10_000),
    (10_001, None),
)


# --- Helpers ---


def _human_bytes(num_bytes: int) -> str:
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


def _ts_to_dt_utc(ts: Any) -> dt.datetime | None:
    if ts is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc)
    except (OSError, OverflowError, TypeError, ValueError):
        return None


def _normalize_mention_path(raw: str) -> str:
    path = raw.strip()
    path = path.strip("`'\"()[]{}<>,.:;")
    path = path.replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def _bucket_for_length(n: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if hi is None and n >= lo:
            return f"{lo}+"
        if hi is not None and lo <= n <= hi:
            return f"{lo}-{hi}"
    return "unknown"


_LINE_ANCHOR_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<col>\d+))?$")


def _strip_line_anchors(raw: str) -> str:
    candidate = raw.split("#", 1)[0]
    match = _LINE_ANCHOR_RE.match(candidate)
    if match:
        return match.group("path")
    return candidate


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


# --- Streaming JSON array decoding ---


def iter_json_array_items(text_stream: io.TextIOBase) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0

    def refill() -> bool:
        nonlocal buf
        chunk = text_stream.read(CHUNK_SIZE_CHARS)
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
        # Skip separators/whitespace
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


# --- Chat export ingestion ---


@dataclass
class RepoIndex:
    tracked_paths: set[str]
    basename_to_paths: dict[str, list[str]]


def build_repo_index(repo_root: Path) -> RepoIndex:
    tracked: list[str] = []
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        tracked = []

    tracked_set = set(tracked)
    basename_to_paths: dict[str, list[str]] = collections.defaultdict(list)
    for p in tracked:
        basename_to_paths[Path(p).name].append(p)

    return RepoIndex(tracked_paths=tracked_set, basename_to_paths=dict(basename_to_paths))


def _open_conversations_stream_from_zip(zip_path: Path) -> io.TextIOBase:
    zf = zipfile.ZipFile(zip_path)
    # Keep ZipFile alive by returning a TextIOWrapper over a ZipExtFile plus attaching zf.
    raw = zf.open("conversations.json", "r")
    text = io.TextIOWrapper(raw, encoding="utf-8")
    setattr(text, "_zipfile_holder", zf)
    return text


def _open_conversations_stream_from_dir(export_dir: Path) -> io.TextIOBase:
    return export_dir.joinpath("conversations.json").open("r", encoding="utf-8")


def find_latest_chat_export_zip(chat_data_dir: Path) -> Path | None:
    if not chat_data_dir.exists():
        return None
    zips = sorted(chat_data_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def resolve_chat_source(repo_root: Path, zip_path: str | None, export_dir: str | None) -> tuple[str, Path]:
    if zip_path:
        p = (repo_root / zip_path).resolve() if not os.path.isabs(zip_path) else Path(zip_path)
        return ("zip", p)
    if export_dir:
        p = (repo_root / export_dir).resolve() if not os.path.isabs(export_dir) else Path(export_dir)
        return ("dir", p)

    chat_data_dir = repo_root / "chat_data"
    latest = find_latest_chat_export_zip(chat_data_dir)
    if latest is not None:
        return ("zip", latest)

    fallback = repo_root / "chat_data_extracted"
    if fallback.exists():
        return ("dir", fallback)

    raise FileNotFoundError(
        "No chat export found. Provide --zip, or put an export zip in chat_data/, "
        "or ensure chat_data_extracted/ exists."
    )


def extract_text_from_message(message: Mapping[str, Any]) -> tuple[str | None, str]:
    content = message.get("content")
    if not isinstance(content, Mapping):
        return None, "unknown"

    content_type = _safe_str(content.get("content_type", "unknown"))
    if content_type != "text":
        return None, content_type

    parts = content.get("parts")
    if not isinstance(parts, list):
        return None, content_type

    text_parts = [p for p in parts if isinstance(p, str)]
    if not text_parts:
        return None, content_type
    return "\n".join(text_parts), content_type


# --- Metrics ---


@dataclass
class ChatMetrics:
    conversations: int = 0
    mapping_nodes_total: int = 0
    messages_total: int = 0
    current_path_messages_total: int = 0

    roles: collections.Counter[str] = field(default_factory=collections.Counter)
    content_types: collections.Counter[str] = field(default_factory=collections.Counter)
    months: collections.Counter[str] = field(default_factory=collections.Counter)
    term_messages: collections.Counter[str] = field(default_factory=collections.Counter)
    normative_messages: collections.Counter[str] = field(default_factory=collections.Counter)
    file_mentions: collections.Counter[str] = field(default_factory=collections.Counter)

    length_buckets_by_role: dict[str, collections.Counter[str]] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    chars_by_role: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))

    first_message_time: dt.datetime | None = None
    last_message_time: dt.datetime | None = None


def _iter_current_path_node_ids(mapping: Mapping[str, Any], current_node: str) -> list[str]:
    path: list[str] = []
    node_id: str | None = current_node
    visited: set[str] = set()
    while node_id and node_id not in visited:
        visited.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, Mapping):
            break
        msg = node.get("message")
        if isinstance(msg, Mapping):
            path.append(node_id)
        parent = node.get("parent")
        node_id = parent if isinstance(parent, str) else None
    path.reverse()
    return path


def _update_time_bounds(metrics: ChatMetrics, when: dt.datetime | None) -> None:
    if when is None:
        return
    if metrics.first_message_time is None or when < metrics.first_message_time:
        metrics.first_message_time = when
    if metrics.last_message_time is None or when > metrics.last_message_time:
        metrics.last_message_time = when


def _count_terms_in_message(
    metrics: ChatMetrics,
    text: str,
) -> None:
    lowered = text.lower()
    for term in STEERING_TERMS:
        if term in lowered:
            metrics.term_messages[term] += 1
    for term in NORMATIVE_TERMS:
        if term in lowered:
            metrics.normative_messages[term] += 1


def _count_file_mentions_in_message(
    metrics: ChatMetrics,
    text: str,
    repo_root: Path,
    repo_index: RepoIndex,
) -> None:
    if "." not in text:
        return

    # Paths with slashes (most reliable)
    if "/" in text:
        for match in PATH_WITH_SLASH_RE.finditer(text):
            raw = match.group("path")
            path = _normalize_mention_path(raw)
            metrics.file_mentions[path] += 1

    # Inline code often contains paths with spaces or punctuation
    if "`" in text:
        for match in INLINE_CODE_RE.finditer(text):
            raw = match.group(1)
            candidate = _strip_line_anchors(_normalize_mention_path(raw))
            if len(candidate) > 240:
                continue
            if any(ch in candidate for ch in ("\n", "\r", "\t")):
                continue
            if "." not in candidate:
                continue

            suffix = Path(candidate).suffix.lower()
            if suffix not in KNOWN_PATH_SUFFIXES:
                continue

            try:
                if candidate in repo_index.tracked_paths or repo_root.joinpath(candidate).exists():
                    metrics.file_mentions[candidate] += 1
            except OSError:
                continue

    # Bare filenames (resolve only if unique in repo)
    for match in BARE_FILENAME_RE.finditer(text):
        name = match.group(1)
        if "/" in name or "\\" in name:
            continue
        candidates = repo_index.basename_to_paths.get(name)
        if not candidates:
            continue
        if len(candidates) == 1:
            metrics.file_mentions[candidates[0]] += 1
        else:
            # Track ambiguity explicitly but keep it out of cross-link existence checks.
            metrics.file_mentions[f"<ambiguous>/{name}"] += 1


def compute_chat_metrics(
    *,
    repo_root: Path,
    repo_index: RepoIndex,
    source_kind: str,
    source_path: Path,
    max_conversations: int | None,
    progress_every: int,
) -> ChatMetrics:
    if source_kind == "zip":
        stream = _open_conversations_stream_from_zip(source_path)
    elif source_kind == "dir":
        stream = _open_conversations_stream_from_dir(source_path)
    else:
        raise ValueError(f"Unknown chat source kind: {source_kind!r}")

    metrics = ChatMetrics()

    try:
        for conv in iter_json_array_items(stream):
            if not isinstance(conv, Mapping):
                continue
            metrics.conversations += 1
            if max_conversations is not None and metrics.conversations > max_conversations:
                break

            if progress_every > 0 and metrics.conversations % progress_every == 0:
                print(f"[steering_audit] parsed {metrics.conversations} conversations...", file=sys.stderr)

            mapping = conv.get("mapping")
            if not isinstance(mapping, Mapping):
                continue
            metrics.mapping_nodes_total += len(mapping)

            current_node = conv.get("current_node")
            if isinstance(current_node, str):
                metrics.current_path_messages_total += len(_iter_current_path_node_ids(mapping, current_node))

            for node in mapping.values():
                if not isinstance(node, Mapping):
                    continue
                message = node.get("message")
                if not isinstance(message, Mapping):
                    continue

                author = message.get("author")
                role = "unknown"
                if isinstance(author, Mapping):
                    role = _safe_str(author.get("role", "unknown"))
                metrics.roles[role] += 1
                metrics.messages_total += 1

                create_time = _ts_to_dt_utc(message.get("create_time"))
                _update_time_bounds(metrics, create_time)
                if create_time is not None:
                    metrics.months[create_time.strftime("%Y-%m")] += 1

                text, content_type = extract_text_from_message(message)
                metrics.content_types[content_type] += 1

                if text is None:
                    continue

                text_len = len(text)
                metrics.chars_by_role[role] += text_len
                metrics.length_buckets_by_role[role][_bucket_for_length(text_len)] += 1

                _count_terms_in_message(metrics, text)
                _count_file_mentions_in_message(metrics, text, repo_root, repo_index)
    finally:
        try:
            stream.close()
        except Exception:
            pass

    return metrics


@dataclass
class RepoMetrics:
    tracked_files: int
    tracked_bytes: int
    ext_counts: collections.Counter[str]
    top_level_counts: collections.Counter[str]
    largest_files: list[tuple[str, int]]


def compute_repo_metrics(repo_root: Path, repo_index: RepoIndex, top_n_largest: int = 20) -> RepoMetrics:
    ext_counts: collections.Counter[str] = collections.Counter()
    top_level_counts: collections.Counter[str] = collections.Counter()
    sizes: list[tuple[str, int]] = []
    total_bytes = 0

    for rel in repo_index.tracked_paths:
        p = repo_root / rel
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            continue
        total_bytes += size
        sizes.append((rel, size))

        suffix = Path(rel).suffix.lower() or "<none>"
        ext_counts[suffix] += 1

        parts = Path(rel).parts
        top = parts[0] if parts else "<root>"
        top_level_counts[top] += 1

    sizes.sort(key=lambda t: t[1], reverse=True)
    return RepoMetrics(
        tracked_files=len(repo_index.tracked_paths),
        tracked_bytes=total_bytes,
        ext_counts=ext_counts,
        top_level_counts=top_level_counts,
        largest_files=sizes[:top_n_largest],
    )


# --- Rendering ---


def _fmt_dt(when: dt.datetime | None) -> str:
    if when is None:
        return "unknown"
    return when.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_report_markdown(
    *,
    repo_root: Path,
    chat_source_kind: str,
    chat_source_path: Path,
    chat_metrics: ChatMetrics,
    repo_metrics: RepoMetrics,
    repo_index: RepoIndex,
    top_file_mentions: int,
    top_terms: int,
    top_months: int,
) -> str:
    lines: list[str] = []
    lines.append("# Steering / Chat / Repo Audit")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Repo root: `{repo_root}`")
    lines.append(f"- Chat export source: `{chat_source_kind}:{chat_source_path}`")
    if chat_source_path.exists():
        try:
            lines.append(f"- Chat export size: `{_human_bytes(chat_source_path.stat().st_size)}`")
        except Exception:
            pass
    lines.append("")

    lines.append("## Chat Summary")
    lines.append(f"- Conversations: `{chat_metrics.conversations}`")
    lines.append(f"- Mapping nodes (total): `{chat_metrics.mapping_nodes_total}`")
    lines.append(f"- Messages (total): `{chat_metrics.messages_total}`")
    lines.append(f"- Messages on current paths (total): `{chat_metrics.current_path_messages_total}`")
    lines.append(f"- First message time: `{_fmt_dt(chat_metrics.first_message_time)}`")
    lines.append(f"- Last message time: `{_fmt_dt(chat_metrics.last_message_time)}`")
    lines.append("")

    lines.append("### Roles")
    for role, count in chat_metrics.roles.most_common():
        chars = chat_metrics.chars_by_role.get(role, 0)
        lines.append(f"- `{role}`: `{count}` messages, `{_human_bytes(chars)}` text")
    lines.append("")

    lines.append("### Content Types")
    for ctype, count in chat_metrics.content_types.most_common(15):
        lines.append(f"- `{ctype}`: `{count}`")
    lines.append("")

    lines.append("### Activity (Top Months)")
    for month, count in chat_metrics.months.most_common(top_months):
        lines.append(f"- `{month}`: `{count}` messages")
    lines.append("")

    lines.append("### Steering Terms (Messages Containing Term)")
    for term, count in chat_metrics.term_messages.most_common(top_terms):
        lines.append(f"- `{term}`: `{count}`")
    lines.append("")

    lines.append("### Normative Terms (Messages Containing Term)")
    for term, count in chat_metrics.normative_messages.most_common():
        lines.append(f"- `{term}`: `{count}`")
    lines.append("")

    lines.append("### Message Length Buckets (Text Messages Only)")
    for role, buckets in sorted(chat_metrics.length_buckets_by_role.items(), key=lambda t: t[0]):
        if not buckets:
            continue
        bucket_str = ", ".join(f"{k}:{v}" for k, v in buckets.most_common())
        lines.append(f"- `{role}`: {bucket_str}")
    lines.append("")

    repo_file_mentions: list[tuple[str, int, bool, bool]] = []
    external_file_mentions: list[tuple[str, int]] = []
    ambiguous_mentions: list[tuple[str, int]] = []

    for path, count in chat_metrics.file_mentions.most_common():
        if path.startswith("<ambiguous>/"):
            ambiguous_mentions.append((path.removeprefix("<ambiguous>/"), count))
            continue

        tracked = path in repo_index.tracked_paths
        exists = repo_root.joinpath(path).exists()
        if tracked or exists:
            repo_file_mentions.append((path, count, exists, tracked))
        else:
            external_file_mentions.append((path, count))

    lines.append("### File Mentions (Repo-Traced)")
    if repo_file_mentions:
        for path, count, exists, tracked in repo_file_mentions[:top_file_mentions]:
            lines.append(f"- `{path}`: `{count}` (exists={exists}, tracked={tracked})")
    else:
        lines.append("- (none detected)")
    lines.append("")

    if ambiguous_mentions:
        lines.append("### File Mentions (Ambiguous Basenames)")
        for name, count in ambiguous_mentions[: min(20, len(ambiguous_mentions))]:
            candidates = repo_index.basename_to_paths.get(name, [])
            shown = ", ".join(f"`{c}`" for c in candidates[:5])
            more = "" if len(candidates) <= 5 else f" (+{len(candidates) - 5} more)"
            lines.append(f"- `{name}`: `{count}` -> {shown}{more}")
        lines.append("")

    if external_file_mentions:
        lines.append("### File Mentions (External/Missing)")
        for path, count in external_file_mentions[: min(30, len(external_file_mentions))]:
            lines.append(f"- `{path}`: `{count}`")
        lines.append("")

    lines.append("## Repo Summary (Tracked Files)")
    lines.append(f"- Tracked files: `{repo_metrics.tracked_files}`")
    lines.append(f"- Tracked bytes: `{_human_bytes(repo_metrics.tracked_bytes)}`")
    lines.append("")

    lines.append("### Top-Level Paths")
    for top, count in repo_metrics.top_level_counts.most_common(20):
        lines.append(f"- `{top}`: `{count}` files")
    lines.append("")

    lines.append("### Extensions")
    for ext, count in repo_metrics.ext_counts.most_common(25):
        lines.append(f"- `{ext}`: `{count}` files")
    lines.append("")

    lines.append("### Largest Tracked Files")
    for path, size in repo_metrics.largest_files:
        lines.append(f"- `{path}`: `{_human_bytes(size)}`")
    lines.append("")

    return "\n".join(lines)


# --- CLI ---


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit chat export + repo for steering signals.")
    parser.add_argument("--zip", dest="zip_path", default=None, help="Path to ChatGPT export zip.")
    parser.add_argument("--dir", dest="export_dir", default=None, help="Path to extracted export directory.")
    parser.add_argument(
        "--out",
        dest="out_path",
        default="chat_data/steering_audit.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Limit number of conversations processed (debug/smoke).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N conversations (0 disables).",
    )
    parser.add_argument("--top-file-mentions", type=int, default=40)
    parser.add_argument("--top-terms", type=int, default=15)
    parser.add_argument("--top-months", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd()

    repo_index = build_repo_index(repo_root)
    repo_metrics = compute_repo_metrics(repo_root, repo_index)

    source_kind, source_path = resolve_chat_source(repo_root, args.zip_path, args.export_dir)
    chat_metrics = compute_chat_metrics(
        repo_root=repo_root,
        repo_index=repo_index,
        source_kind=source_kind,
        source_path=source_path,
        max_conversations=args.max_conversations,
        progress_every=args.progress_every,
    )

    report = render_report_markdown(
        repo_root=repo_root,
        chat_source_kind=source_kind,
        chat_source_path=source_path,
        chat_metrics=chat_metrics,
        repo_metrics=repo_metrics,
        repo_index=repo_index,
        top_file_mentions=args.top_file_mentions,
        top_terms=args.top_terms,
        top_months=args.top_months,
    )

    out_path = (repo_root / args.out_path).resolve() if not os.path.isabs(args.out_path) else Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[steering_audit] wrote report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
