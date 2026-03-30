#!/usr/bin/env python3
"""
Build comprehensive activity log for the last N days.
Combines: Open Brain (all entries), git log, results files, qa-collab events.
Output: reviews/activity_log_YYYY-MM-DD.jsonl + reviews/activity_log_YYYY-MM-DD.md
"""

import json, os, subprocess, sys, urllib.request, urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent
REVIEWS_DIR = REPO_ROOT / "reviews"
COLLAB_LOG = REPO_ROOT / "qa_lab" / "logs" / "collab_events.jsonl"
OPEN_BRAIN_URL = "https://bepguekxksbgiqleadvq.supabase.co/functions/v1/open-brain-mcp"
OPEN_BRAIN_KEY = "a51c0a16163ad5af47b3f3a7f1b3f33605c19f502ad5b7af47101a72c88e3376"

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def ob_call(method, args):
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": method, "arguments": args}
    }).encode()
    req = urllib.request.Request(
        OPEN_BRAIN_URL, data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-brain-key": OPEN_BRAIN_KEY,
            "Authorization": f"Bearer {OPEN_BRAIN_KEY}",
        }, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode()
    data_lines = [l[5:].strip() for l in raw.splitlines()
                  if l.startswith("data:") and l.strip() != "data:"]
    raw = data_lines[-1] if data_lines else "{}"
    data = json.loads(raw)
    content = data.get("result", {}).get("content", [])
    return content[0].get("text", "") if content else ""


def parse_ob_text(text):
    """Parse the bullet-list format from Open Brain."""
    entries = []
    current = None
    for line in text.splitlines():
        if line.startswith("- 20"):
            if current:
                entries.append(current)
            rest = line[2:]
            try:
                ts = rest[:rest.index(" ")]
                rest = rest[len(ts)+1:]
                etype = "note"
                if rest.startswith("("):
                    etype = rest[1:rest.index(")")]
                    rest = rest[len(etype)+3:]
                tags = []
                if rest.startswith("["):
                    tag_end = rest.index("]")
                    tags = [t.strip() for t in rest[1:tag_end].split(",")]
                    rest = rest[tag_end+2:]
                current = {"source": "open_brain", "timestamp": ts,
                           "type": etype, "tags": tags, "content": rest}
            except (ValueError, IndexError):
                current = {"source": "open_brain", "timestamp": "",
                           "type": "note", "tags": [], "content": line}
        elif current:
            current["content"] += "\n" + line
    if current:
        entries.append(current)
    return entries


def fetch_all_ob_entries(days):
    """Fetch all Open Brain entries using list_thoughts by type."""
    all_entries = []
    seen_timestamps = set()
    types = ["observation", "idea", "task", "reference", "note"]
    print(f"  Fetching Open Brain ({days} days)...")
    for t in types:
        try:
            text = ob_call("list_thoughts", {"since_days": days, "limit": 200, "type": t})
            entries = parse_ob_text(text)
            new = 0
            for e in entries:
                key = e["timestamp"] + e["content"][:50]
                if key not in seen_timestamps:
                    seen_timestamps.add(key)
                    all_entries.append(e)
                    new += 1
            print(f"    type={t}: {new} new entries")
        except Exception as ex:
            print(f"    type={t}: failed ({ex})")
    # Also fetch without type filter to catch unlabeled
    try:
        text = ob_call("recent_thoughts", {"since_days": days, "limit": 50})
        entries = parse_ob_text(text)
        new = 0
        for e in entries:
            key = e["timestamp"] + e["content"][:50]
            if key not in seen_timestamps:
                seen_timestamps.add(key)
                all_entries.append(e)
                new += 1
        print(f"    recent_thoughts top-50: {new} new entries")
    except Exception as ex:
        print(f"    recent_thoughts: failed ({ex})")
    all_entries.sort(key=lambda x: x["timestamp"], reverse=True)
    return all_entries


def fetch_git_log(days):
    print(f"  Fetching git log ({days} days)...")
    result = subprocess.run(
        ["git", "log", f"--since={days} days ago",
         "--pretty=format:%H|%ad|%an|%s|%b", "--date=iso", "-z"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    commits = []
    for block in result.stdout.split("\x00"):
        block = block.strip()
        if not block:
            continue
        parts = block.split("|", 4)
        if len(parts) >= 4:
            commits.append({
                "source": "git",
                "timestamp": parts[1].strip(),
                "type": "commit",
                "tags": ["git"],
                "hash": parts[0][:8],
                "author": parts[2],
                "content": parts[3] + ("\n" + parts[4].strip() if len(parts) > 4 and parts[4].strip() else ""),
            })
    print(f"    {len(commits)} commits")
    return commits


def fetch_results_files(days):
    print(f"  Scanning results files ({days} days)...")
    cutoff = datetime.now() - timedelta(days=days)
    entries = []
    results_dir = REPO_ROOT / "results"
    if not results_dir.exists():
        return entries
    for f in sorted(results_dir.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime < cutoff:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            summary = data.get("summary", data.get("verdict", data.get("result", "")))
            if isinstance(summary, dict):
                summary = json.dumps(summary)[:200]
            entries.append({
                "source": "results_file",
                "timestamp": mtime.strftime("%Y-%m-%dT%H:%M:%S"),
                "type": "result",
                "tags": ["experiment", "result"],
                "filename": f.name,
                "content": f"{f.name}: {str(summary)[:300]}",
            })
        except Exception:
            entries.append({
                "source": "results_file",
                "timestamp": mtime.strftime("%Y-%m-%dT%H:%M:%S"),
                "type": "result",
                "tags": ["experiment"],
                "filename": f.name,
                "content": f.name,
            })
    print(f"    {len(entries)} result files")
    return entries


def fetch_collab_events(days):
    print(f"  Reading qa-collab events ({days} days)...")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    events = []
    if not COLLAB_LOG.exists():
        return events
    with open(COLLAB_LOG) as f:
        for line in f:
            try:
                ev = json.loads(line.strip())
                ts = ev.get("timestamp", "")
                if ts:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt >= cutoff:
                        events.append({
                            "source": "qa_collab",
                            "timestamp": ts,
                            "type": "collab_event",
                            "tags": [ev.get("topic", "")],
                            "content": f"[{ev.get('topic','')}] {json.dumps(ev.get('payload',''))[:200]}",
                        })
            except (json.JSONDecodeError, ValueError):
                pass
    print(f"    {len(events)} events")
    return events


NOISE_DIRS = {".venv", "node_modules", "__pycache__", ".git", "archive",
              "Documents/open_brain_migration", "Documents/open_brain_migration_chatgpt",
              "Documents/open_brain_migration_qanotes", "Documents/open_brain_migration_qalabvault"}


def is_noise_path(rel: str) -> bool:
    parts = Path(rel).parts
    return any(p.startswith(".venv") or p in NOISE_DIRS or "open_brain_migration" in p
               for p in parts)


def fetch_modified_files(days):
    """Find recently modified meaningful source files."""
    print(f"  Scanning modified source files ({days} days)...")
    result = subprocess.run(
        ["git", "diff", "--name-only", f"HEAD@{{{days} days ago}}", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    files = [f.strip() for f in result.stdout.splitlines()
             if f.strip() and not is_noise_path(f.strip())]
    # Untracked new files modified recently
    cutoff = datetime.now() - timedelta(days=days)
    for ext in ["*.py", "*.md", "*.tex", "*.json", "*.yaml", "*.rs", "*.sh"]:
        for f in REPO_ROOT.glob(f"**/{ext}"):
            rel = str(f.relative_to(REPO_ROOT))
            if is_noise_path(rel):
                continue
            try:
                if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff:
                    if rel not in files:
                        files.append(rel)
            except OSError:
                pass
    files.sort()
    print(f"    {len(files)} modified/new files")
    return files


def infer_agent(content, tags, etype=""):
    c = content.lower()
    tag_str = " ".join(tags).lower() if tags else ""

    # Explicit attribution
    if any(x in c for x in ["from codex", "codex as of", "permanent reference future-work priorities from codex"]):
        return "codex"
    if any(x in c for x in ["from claude", "claude here", "orientation gap"]):
        return "claude"
    if "from gemini" in c or "gemini" in tag_str:
        return "gemini"
    if "from chatgpt" in c or "chatgpt" in tag_str:
        return "chatgpt"

    # Git = Will or Codex (commit messages)
    if "git" in tag_str or etype == "commit":
        return "will/codex"

    # Strong Codex signals: preregistered experiment pattern
    if any(x in c for x in [
        "preregister", "preregistered", "success criteria:", "success criteria :",
        "pass if ", "fail if ", "partial if ",
        "honest verdict:", "honest interpretation:",
        "artifacts under results/",
        "completed on 2026-", "completed and passed", "completed and failed",
        "eeg_hi2_0_", "run eeg", "run signal",
        "sweep ", "threshold sweep", "ablation", "scale-run",
    ]):
        return "codex"

    # Strong Claude signals: synthesis, certs, meta
    if any(x in c for x in [
        "permanent reference", "vision-aligned", "cert gap", "mission:",
        "the actual mission", "orientation gap", "two-tract", "meta-validator",
        "cert famil", "family [", "shipped", "pond science institute",
    ]):
        return "claude"

    # Arto Heino ternary work — Codex ran the experiments
    if "arto" in tag_str or "ternary" in tag_str or "ternary-logic" in tag_str:
        return "codex"

    # EEG / signal / compression experiments = Codex
    if any(x in tag_str for x in ["eeg", "signal", "compression", "arto-heino"]):
        return "codex"
    if any(x in tag_str for x in ["eeg", "chb-mit", "hi-2.0", "hi2", "signal-domain"]):
        return "codex"

    # Reference entries = Claude
    if etype == "reference":
        return "claude"

    return "unknown"


def build_log(days):
    REVIEWS_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    print(f"\nBuilding activity log for last {days} days...\n")

    ob_entries = fetch_all_ob_entries(days)
    git_commits = fetch_git_log(days)
    result_files = fetch_results_files(days)
    collab_events = fetch_collab_events(days)
    modified_files = fetch_modified_files(days)

    all_entries = ob_entries + git_commits + result_files + collab_events
    all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    # Write JSONL log
    jsonl_path = REVIEWS_DIR / f"activity_log_{date_str}.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"\nJSONL log: {jsonl_path} ({len(all_entries)} entries)")

    # Write human-readable markdown log
    md_path = REVIEWS_DIR / f"activity_log_{date_str}.md"
    lines = []
    lines.append(f"# QA Project Activity Log — Last {days} Days")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}  |  **Period:** {(now - timedelta(days=days)).strftime('%Y-%m-%d')} → {date_str}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"| Source | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Open Brain entries | {len(ob_entries)} |")
    lines.append(f"| Git commits | {len(git_commits)} |")
    lines.append(f"| Experiment result files | {len(result_files)} |")
    lines.append(f"| qa-collab events | {len(collab_events)} |")
    lines.append(f"| Modified/new source files | {len(modified_files)} |")
    lines.append(f"| **Total log entries** | **{len(all_entries)}** |")
    lines.append("")

    # By-agent breakdown
    by_agent = defaultdict(list)
    for e in all_entries:
        agent = infer_agent(e.get("content", ""), e.get("tags", []))
        by_agent[agent].append(e)

    lines.append("## Activity by Agent")
    for agent, items in sorted(by_agent.items(), key=lambda x: -len(x[1])):
        lines.append(f"- **{agent}**: {len(items)} entries")
    lines.append("")

    # Git commits section
    lines.append("## Git Commits")
    lines.append("")
    for c in git_commits:
        lines.append(f"- `{c['hash']}` `{c['timestamp'][:10]}` — {c['content'].splitlines()[0]}")
    lines.append("")

    # Modified files
    lines.append("## Modified / New Files")
    lines.append("")
    for f in sorted(modified_files)[:80]:
        lines.append(f"- `{f}`")
    if len(modified_files) > 80:
        lines.append(f"- *(+{len(modified_files)-80} more)*")
    lines.append("")

    # Experiment results
    lines.append("## Experiment Results (by date)")
    lines.append("")
    for e in result_files:
        lines.append(f"- `{e['timestamp'][:16]}` {e['content'][:200]}")
    lines.append("")

    # Open Brain timeline
    lines.append("## Open Brain Timeline")
    lines.append("")
    current_date = None
    by_type = defaultdict(int)
    for e in ob_entries:
        ts = e.get("timestamp", "")[:10]
        etype = e.get("type", "note")
        by_type[etype] += 1
        if ts != current_date:
            current_date = ts
            lines.append(f"\n### {ts}")
        agent = infer_agent(e.get("content", ""), e.get("tags", []))
        tags = ", ".join(e.get("tags", []))[:60]
        snippet = e.get("content", "").replace("\n", " ")[:180]
        lines.append(f"- `[{etype}]` [{agent}] **{tags}**")
        lines.append(f"  {snippet}")
    lines.append("")
    lines.append("## Open Brain Entry Type Breakdown")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- {t}: {n}")
    lines.append("")

    # qa-collab events
    if collab_events:
        lines.append("## qa-collab Events")
        lines.append("")
        for e in collab_events:
            lines.append(f"- `{e['timestamp'][:16]}` {e['content'][:160]}")
        lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"Markdown log: {md_path}")

    return jsonl_path, md_path, {
        "open_brain": len(ob_entries),
        "git_commits": len(git_commits),
        "result_files": len(result_files),
        "collab_events": len(collab_events),
        "modified_files": len(modified_files),
        "total": len(all_entries),
    }


if __name__ == "__main__":
    jsonl_path, md_path, stats = build_log(DAYS)
    print(f"\nDone. Total entries: {stats['total']}")
    print(f"  Open Brain:    {stats['open_brain']}")
    print(f"  Git commits:   {stats['git_commits']}")
    print(f"  Result files:  {stats['result_files']}")
    print(f"  Collab events: {stats['collab_events']}")
    print(f"  Source files:  {stats['modified_files']}")
