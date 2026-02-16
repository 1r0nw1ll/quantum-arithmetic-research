"""
Generate a first-pass RESULTS_REGISTRY from forensics outputs.

Entry point:
  python tools/generate_results_registry.py

Defaults:
  - Reads latest `_forensics/forensics_*/` run
  - Writes `Documents/RESULTS_REGISTRY.md`

Inputs (produced by `tools/project_forensics.py`):
  - keyword_hotspots.tsv
  - script_artifacts.tsv
  - chat_path_mentions.tsv (optional)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UTC = dt.timezone.utc
DEFAULT_CONTROL_CUTOFF_UTC_DATE = "2026-01-10"


def _parse_utc_date(d: str) -> dt.datetime | None:
    try:
        date = dt.date.fromisoformat(d)
        return dt.datetime(date.year, date.month, date.day, tzinfo=UTC)
    except Exception:
        return None


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=UTC).isoformat()


def _human_bytes(num_bytes: int) -> str:
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


def _read_text_limited(path: Path, max_bytes: int = 200_000) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


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


def find_latest_forensics_dir(root: Path) -> Path:
    base = root / "_forensics"
    if not base.exists():
        raise FileNotFoundError("No _forensics directory found. Run `python tools/project_forensics.py` first.")
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("forensics_")]
    if not runs:
        raise FileNotFoundError("No forensics runs found under _forensics/. Run `python tools/project_forensics.py` first.")
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
            row = {cols[i]: parts[i] for i in range(len(cols))}
            rows.append(row)
    return rows


def _first_heading(md_text: str) -> str | None:
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            title = s.lstrip("#").strip()
            if title:
                return title
    return None


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip().lstrip("\ufeff").strip()
        if s:
            return s
    return None


def _first_docstring_line(py_text: str) -> str | None:
    # Very small docstring extractor (handles leading module docstring only).
    lines = py_text.splitlines()
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
        line = _first_nonempty_line(body)
        if line:
            return line
    return None


def _tex_title(tex_text: str) -> str | None:
    m = re.search(r"\\title\\{([^}]{1,200})\\}", tex_text)
    if m:
        return m.group(1).strip()
    return None


def infer_title(root: Path, rel_path: str) -> str:
    path = root / rel_path
    ext = path.suffix.lower()
    text = _read_text_limited(path)
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
        if "validator" in p or p.endswith("_validate.py") or "/validate" in p:
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
        return "Imported notes (Google AI Studio)"
    if p.startswith("docs/ai_chats/"):
        return "Imported notes (ai_chats)"
    if p.startswith("docs/"):
        return "Docs"
    if p.startswith("Documents/"):
        return "Paper/docs"
    if p.startswith("demos/"):
        return "Demo"
    if p.startswith("tools/"):
        return "Tooling"
    return "Misc"


def suggest_commands(root: Path, rel_path: str) -> list[str]:
    path = root / rel_path
    if path.suffix.lower() != ".py":
        return []
    text = _read_text_limited(path, max_bytes=400_000) or ""
    cmds: list[str] = []
    # Prefer explicit demo flags if present.
    if "--demo" in text:
        cmds.append(f"python {rel_path} --demo")
    # If it has argparse, `--help` usually works.
    if "argparse.ArgumentParser" in text or "import argparse" in text:
        cmds.append(f"python {rel_path} --help")
    # Fallback
    cmds.append(f"python {rel_path}")
    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in cmds:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


@dataclass
class Hotspot:
    score: int
    evidence: int
    claims: int
    path: str


def load_hotspots(keyword_hotspots_path: Path, top_n: int) -> list[Hotspot]:
    rows = parse_tsv(keyword_hotspots_path)
    items: list[Hotspot] = []
    for row in rows:
        try:
            score = int(row.get("score", "0") or "0")
            evidence = int(row.get("evidence", "0") or "0")
            claims = int(row.get("claims", "0") or "0")
            path = row.get("path", "").strip()
        except ValueError:
            continue
        if not path:
            continue
        items.append(Hotspot(score=score, evidence=evidence, claims=claims, path=path))
    items.sort(key=lambda h: (h.score, h.evidence, h.claims), reverse=True)
    return items[:top_n]


def load_script_artifacts(script_artifacts_path: Path) -> dict[str, list[str]]:
    rows = parse_tsv(script_artifacts_path)
    mapping: dict[str, list[str]] = {}
    for row in rows:
        script = row.get("script", "").strip()
        resolved = row.get("resolved_path", "").strip()
        exists = row.get("exists", "").strip()
        if not script or not resolved:
            continue
        if exists not in ("1", "true", "True", "YES", "yes"):
            continue
        mapping.setdefault(script, [])
        if resolved not in mapping[script]:
            mapping[script].append(resolved)
    # Stable ordering
    for k in list(mapping.keys()):
        mapping[k] = sorted(mapping[k])
    return mapping


def load_chat_path_mentions(chat_path_mentions_path: Path) -> dict[str, int]:
    if not chat_path_mentions_path.exists():
        return {}
    rows = parse_tsv(chat_path_mentions_path)
    out: dict[str, int] = {}
    for row in rows:
        path = row.get("path", "").strip()
        count = row.get("count", "").strip()
        if not path:
            continue
        try:
            out[path] = int(count or "0")
        except ValueError:
            continue
    return out


def render_registry_markdown(
    *,
    root: Path,
    forensics_dir: Path,
    hotspots: list[Hotspot],
    script_artifacts: dict[str, list[str]],
    chat_mentions: dict[str, int],
    tracked_paths: set[str],
    extra_scripts: list[str],
    control_cutoff_utc_date: str,
) -> str:
    cutoff_dt = _parse_utc_date(control_cutoff_utc_date)
    lines: list[str] = []
    lines.append("# Results Registry (Auto-Generated)")
    lines.append("")
    lines.append(f"- Generated: `{_utc_now_iso()}`")
    lines.append(f"- Forensics input: `{forensics_dir}`")
    lines.append(f"- Control-theorem cutoff (UTC date): `{control_cutoff_utc_date}` (pre-cutoff = revet required)")
    lines.append("- This is a *triage index*: it ranks likely “result nodes” using keyword hotspots and script→artifact links.")
    lines.append("")
    lines.append("## How to update")
    lines.append("- Re-run forensics: `python tools/project_forensics.py`")
    lines.append("- Re-generate this file: `python tools/generate_results_registry.py`")
    lines.append("")

    lines.append("## Hotspots (claim/evidence keyword density)")
    for i, hs in enumerate(hotspots, start=1):
        path = hs.path
        full = root / path
        exists = full.exists()
        tracked = path in tracked_paths
        size = full.stat().st_size if exists else 0
        if exists:
            mtime_dt = dt.datetime.fromtimestamp(full.stat().st_mtime, tz=UTC)
            mtime = mtime_dt.strftime("%Y-%m-%d")
            revet_required = int(bool(cutoff_dt and mtime_dt < cutoff_dt))
        else:
            mtime = "missing"
            revet_required = 0
        title = infer_title(root, path) if exists else Path(path).name
        category = infer_category(path)
        chat_count = chat_mentions.get(path, 0)

        lines.append(f"### R-{i:03d}: {title}")
        lines.append(f"- Path: `{path}`")
        lines.append(f"- Category: `{category}`")
        lines.append(f"- Hotspot score: `{hs.score}` (evidence={hs.evidence}, claims={hs.claims})")
        lines.append(
            "- File: "
            + f"exists=`{int(exists)}` tracked=`{int(tracked)}` revet_required=`{revet_required}` "
            + f"size=`{_human_bytes(size)}` mtime=`{mtime}` chat_mentions=`{chat_count}`"
        )

        artifacts = script_artifacts.get(path, [])
        if artifacts:
            sample = artifacts[:8]
            lines.append("- Linked artifacts (from script strings):")
            for a in sample:
                lines.append(f"  - `{a}`")
            if len(artifacts) > len(sample):
                lines.append(f"  - (+{len(artifacts) - len(sample)} more)")

        cmds = suggest_commands(root, path)
        if cmds:
            lines.append("- Suggested reproduction commands (best-effort):")
            for c in cmds[:3]:
                lines.append(f"  - `{c}`")

        # For scripts with JSON cert artifacts, suggest meta-validator on one sample.
        if artifacts:
            json_certs = [a for a in artifacts if a.endswith(".json") and ("/certs/" in a or "/examples/" in a)]
            if json_certs:
                lines.append("- Suggested evidence check:")
                sample_cert = json_certs[0]
                lines.append(f"  - `python qa_alphageometry_ptolemy/qa_meta_validator.py {sample_cert}`")

        lines.append("")

    lines.append("## Artifact-Producing Scripts (top)")
    if not extra_scripts:
        lines.append("- (none)")
        lines.append("")
        return "\n".join(lines)

    for j, script in enumerate(extra_scripts, start=1):
        artifacts = script_artifacts.get(script, [])
        if not artifacts:
            continue
        exists = (root / script).exists()
        tracked = script in tracked_paths
        title = infer_title(root, script) if exists else Path(script).name
        chat_count = chat_mentions.get(script, 0)
        if exists:
            mtime_dt = dt.datetime.fromtimestamp((root / script).stat().st_mtime, tz=UTC)
            revet_required = int(bool(cutoff_dt and mtime_dt < cutoff_dt))
        else:
            revet_required = 0
        lines.append(f"### E-{j:03d}: {title}")
        lines.append(
            f"- Script: `{script}` (exists={int(exists)}, tracked={int(tracked)}, revet_required={revet_required}, chat_mentions={chat_count})"
        )
        lines.append(f"- Artifacts: `{len(artifacts)}`")
        for a in artifacts[:10]:
            lines.append(f"  - `{a}`")
        if len(artifacts) > 10:
            lines.append(f"  - (+{len(artifacts) - 10} more)")
        cmds = suggest_commands(root, script)
        if cmds:
            lines.append("- Suggested run:")
            lines.append(f"  - `{cmds[0]}`")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate Documents/RESULTS_REGISTRY.md from forensics outputs.")
    parser.add_argument("--forensics-dir", default=None, help="Path to a _forensics/forensics_* directory.")
    parser.add_argument("--out", default="Documents/RESULTS_REGISTRY.md", help="Output markdown path.")
    parser.add_argument("--top-hotspots", type=int, default=20, help="Number of keyword hotspots to include.")
    parser.add_argument("--top-scripts", type=int, default=20, help="Number of artifact-producing scripts to include.")
    parser.add_argument(
        "--control-cutoff-utc-date",
        default=DEFAULT_CONTROL_CUTOFF_UTC_DATE,
        help="UTC date (YYYY-MM-DD). Files with mtime strictly before this are flagged as legacy/revet-required.",
    )
    args = parser.parse_args(argv)

    root = Path.cwd()
    forensics_dir = Path(args.forensics_dir) if args.forensics_dir else find_latest_forensics_dir(root)
    forensics_dir = forensics_dir.resolve() if not forensics_dir.is_absolute() else forensics_dir

    keyword_hotspots_path = forensics_dir / "keyword_hotspots.tsv"
    script_artifacts_path = forensics_dir / "script_artifacts.tsv"
    chat_path_mentions_path = forensics_dir / "chat_path_mentions.tsv"

    if not keyword_hotspots_path.exists():
        raise FileNotFoundError(f"Missing {keyword_hotspots_path}. Re-run `python tools/project_forensics.py`.")
    if not script_artifacts_path.exists():
        raise FileNotFoundError(f"Missing {script_artifacts_path}. Re-run `python tools/project_forensics.py`.")

    hotspots = load_hotspots(keyword_hotspots_path, top_n=args.top_hotspots)
    script_artifacts = load_script_artifacts(script_artifacts_path)
    chat_mentions = load_chat_path_mentions(chat_path_mentions_path)
    tracked_paths = _git_ls_files(root)

    # Extra scripts: top by artifact count, excluding those already in hotspots.
    hotspot_paths = {hs.path for hs in hotspots}
    sorted_scripts = sorted(script_artifacts.items(), key=lambda t: len(t[1]), reverse=True)
    extra_scripts: list[str] = []
    for script, artifacts in sorted_scripts:
        if script in hotspot_paths:
            continue
        if not script.endswith(".py"):
            continue
        extra_scripts.append(script)
        if len(extra_scripts) >= args.top_scripts:
            break

    md = render_registry_markdown(
        root=root,
        forensics_dir=forensics_dir,
        hotspots=hotspots,
        script_artifacts=script_artifacts,
        chat_mentions=chat_mentions,
        tracked_paths=tracked_paths,
        extra_scripts=extra_scripts,
        control_cutoff_utc_date=args.control_cutoff_utc_date,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[results_registry] wrote: {out_path}", file=os.sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
