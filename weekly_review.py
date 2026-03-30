#!/usr/bin/env python3
"""
QA Project Weekly Performance Review
=====================================
Pulls from Open Brain (Supabase MCP), qa-collab events (local JSONL),
and git log — then generates a mission-aligned markdown report.

Mission: Validate and propagate QA as the mathematical language of nature.
         Ben Iverson's discrete arithmetic + Dale Pond's SVP/Keely framework.
         Papers are artifacts of rigor, not the goal.

Usage:
    python weekly_review.py             # last 7 days
    python weekly_review.py --days 14   # last 14 days
    python weekly_review.py --out path  # custom output path
"""

QA_COMPLIANCE = "tooling — no QA arithmetic state; reads metadata and generates reports"


import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent
COLLAB_LOG = REPO_ROOT / "qa_lab" / "logs" / "collab_events.jsonl"
REVIEWS_DIR = REPO_ROOT / "reviews"

# Open Brain Supabase MCP endpoint (from .mcp.json)
OPEN_BRAIN_URL = "https://bepguekxksbgiqleadvq.supabase.co/functions/v1/open-brain-mcp"
OPEN_BRAIN_KEY = "a51c0a16163ad5af47b3f3a7f1b3f33605c19f502ad5b7af47101a72c88e3376"

# ── Agent Attribution ────────────────────────────────────────────────────────

# Keywords that suggest which agent produced an entry
AGENT_SIGNALS = {
    "codex": [
        "preregister", "preregistered", "success criteria", "pass if", "fail if",
        "partial if", "honest verdict", "honest interpretation", "artifacts under",
        "run script", "sweep", "batch", "scale-run", "eeg_hi2", "signal_experiment",
        "from codex", "codex as of",
    ],
    "claude": [
        "from claude", "claude here", "synthesis role", "orientation gap",
        "cert gap", "permanent reference", "mission:", "the actual mission",
        "wildberger bridge", "open brain capture", "memory", "orchestrator",
        "family [", "meta-validator", "two-tract", "shipped",
    ],
    "chatgpt": [
        "from chatgpt", "gpt", "openai", "chatgpt",
        "ideation", "brainstorm",
    ],
    "gemini": [
        "from gemini", "gemini", "docs/", "documentation review",
    ],
    "will": [
        "will said", "will's", "directed by", "natebjones",
    ],
}

# ── Mission Scoring ──────────────────────────────────────────────────────────

# Each dimension 0-3, total out of 12
SPINE_KEYWORDS = [
    "ben iverson", "dale pond", "svp", "keely", "wildberger", "chromogeometry",
    "rational trig", "orbit", "cosmos", "satellite", "singularity", "mod 24", "mod-24",
    "pisano", "fibonacci", "null point", "uhg", "spread poly", "e8", "harmonic index",
    "qa spine", "iota", "myriad", "music of spheres", "sympathetic",
]
HONESTY_KEYWORDS = [
    "preregistered", "honest verdict", "honest interpretation", "fail", "partial",
    "contradicts", "negative result", "does not prove", "does not support",
    "limitation", "caveat",
]
DEPTH_KEYWORDS = [
    "theorem", "proof", "cert", "certificate", "explains", "reveals", "insight",
    "mechanism", "why", "because", "therefore", "implies", "derived", "discovered",
    "new understanding", "deeper",
]
PROPAGATION_KEYWORDS = [
    "accessible", "humanity", "propagat", "publish", "paper", "arxiv",
    "demo", "quickstart", "documentation", "teach", "example", "practical",
    "application", "real-world", "toolchain",
]


def score_entry(content: str) -> dict:
    c = content.lower()
    spine = min(3, sum(1 for k in SPINE_KEYWORDS if k in c))
    honesty = min(3, sum(1 for k in HONESTY_KEYWORDS if k in c))
    depth = min(3, sum(1 for k in DEPTH_KEYWORDS if k in c))
    propagation = min(3, sum(1 for k in PROPAGATION_KEYWORDS if k in c))
    return {
        "spine": spine,
        "honesty": honesty,
        "depth": depth,
        "propagation": propagation,
        "total": spine + honesty + depth + propagation,
    }


def attribute_agent(content: str, tags: list) -> str:
    c = content.lower()
    tag_str = " ".join(tags).lower() if tags else ""
    scores = {agent: 0 for agent in AGENT_SIGNALS}
    for agent, signals in AGENT_SIGNALS.items():
        for sig in signals:
            if sig in c or sig in tag_str:
                scores[agent] += 1
    best = max(scores, key=lambda a: scores[a])
    return best if scores[best] > 0 else "unknown"


# ── Data Fetchers ────────────────────────────────────────────────────────────

def fetch_open_brain(since_days: int) -> list:
    """Call Open Brain MCP via HTTP JSON-RPC."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "recent_thoughts",
            "arguments": {"since_days": since_days, "limit": 50}
        }
    }).encode()

    req = urllib.request.Request(
        OPEN_BRAIN_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-brain-key": OPEN_BRAIN_KEY,
            "Authorization": f"Bearer {OPEN_BRAIN_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            # MCP returns SSE: "event: message\ndata: {...}\n\n"
            data_lines = [l[5:].strip() for l in raw.splitlines()
                          if l.startswith("data:") and l.strip() != "data:"]
            if data_lines:
                raw = data_lines[-1]
            data = json.loads(raw)
            content = data.get("result", {}).get("content", [])
            if content and isinstance(content, list):
                text = content[0].get("text", "")
                return parse_open_brain_text(text)
            return []
    except Exception as e:
        print(f"[open-brain] fetch failed: {e}", file=sys.stderr)
        return []


def parse_open_brain_text(text: str) -> list:
    """Parse the bullet-list format returned by recent_thoughts."""
    entries = []
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- 20"):  # timestamp line
            if current:
                entries.append(current)
            # Parse: - TIMESTAMP (TYPE) [TAGS] CONTENT
            # Example: - 2026-03-30T00:38:04.805Z (observation) [tag1, tag2] body...
            try:
                rest = line[2:]  # strip "- "
                ts_end = rest.index(" ")
                ts = rest[:ts_end]
                rest = rest[ts_end+1:]
                entry_type = "note"
                if rest.startswith("("):
                    t_end = rest.index(")")
                    entry_type = rest[1:t_end]
                    rest = rest[t_end+2:]
                tags = []
                if rest.startswith("["):
                    tag_end = rest.index("]")
                    tags = [t.strip() for t in rest[1:tag_end].split(",")]
                    rest = rest[tag_end+2:]
                current = {
                    "timestamp": ts,
                    "type": entry_type,
                    "tags": tags,
                    "content": rest,
                }
            except (ValueError, IndexError):
                current = {"timestamp": "", "type": "note", "tags": [], "content": line}
        elif current:
            current["content"] += "\n" + line

    if current:
        entries.append(current)
    return entries


def read_collab_events(since: datetime) -> list:
    """Read qa-collab events from local JSONL."""
    if not COLLAB_LOG.exists():
        return []
    events = []
    try:
        with open(COLLAB_LOG) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    ts_str = ev.get("timestamp", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts >= since:
                            events.append(ev)
                except (json.JSONDecodeError, ValueError):
                    pass
    except OSError:
        pass
    return events


def read_git_log(since_days: int) -> list:
    """Read git commits from the last N days."""
    try:
        result = subprocess.run(
            ["git", "log", f"--since={since_days} days ago",
             "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso"],
            capture_output=True, text=True, cwd=REPO_ROOT
        )
        commits = []
        for line in result.stdout.splitlines():
            parts = line.split("|", 4)
            if len(parts) == 5:
                commits.append({
                    "hash": parts[0][:8],
                    "author": parts[1],
                    "email": parts[2],
                    "date": parts[3],
                    "subject": parts[4],
                })
        return commits
    except Exception:
        return []


# ── Report Generation ────────────────────────────────────────────────────────

def agent_display_name(agent: str) -> str:
    return {
        "codex": "Codex (Code Execution)",
        "claude": "Claude (Orchestrator/Analyst)",
        "chatgpt": "ChatGPT (Research/Ideation)",
        "gemini": "Gemini (Docs/Synthesis)",
        "will": "Will (Director)",
        "unknown": "Unattributed",
    }.get(agent, agent.title())


def run_linter_stats() -> dict:
    """Run qa_axiom_linter.py --all and return error/warning counts and file breakdown."""
    import subprocess, re
    linter = Path(__file__).parent / "tools" / "qa_axiom_linter.py"
    if not linter.exists():
        return {"available": False, "error": "linter not found"}
    try:
        proc = subprocess.run(
            [sys.executable, str(linter), "--all"],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + stderr

        errors, warnings = 0, 0
        error_files: dict = {}   # file -> list of (rule, snippet)
        warning_files: dict = {}

        for line in combined.splitlines():
            m_e = re.match(r"^(.+?)\s+ERROR\s+(\S+)\s*(.*)", line)
            m_w = re.match(r"^(.+?)\s+WARN\s+(\S+)\s*(.*)", line)
            if m_e:
                errors += 1
                fname = m_e.group(1).strip()
                error_files.setdefault(fname, []).append((m_e.group(2), m_e.group(3)[:60]))
            elif m_w:
                warnings += 1
                fname = m_w.group(1).strip()
                warning_files.setdefault(fname, []).append((m_w.group(2), m_w.group(3)[:60]))

        return {
            "available": True,
            "returncode": proc.returncode,
            "errors": errors,
            "warnings": warnings,
            "error_files": error_files,
            "warning_files": warning_files,
            "raw_tail": combined[-800:] if len(combined) > 800 else combined,
        }
    except Exception as ex:
        return {"available": True, "error": str(ex), "errors": -1, "warnings": -1}


def generate_report(thoughts: list, collab_events: list, commits: list,
                    since_days: int, week_of: str, lint_stats: dict = None) -> str:

    # Group thoughts by agent
    by_agent = defaultdict(list)
    for t in thoughts:
        agent = attribute_agent(t["content"], t.get("tags", []))
        score = score_entry(t["content"])
        t["_agent"] = agent
        t["_score"] = score
        by_agent[agent].append(t)

    # Aggregate scores per agent
    agent_scores = {}
    for agent, items in by_agent.items():
        if not items:
            continue
        totals = [i["_score"]["total"] for i in items]
        dims = {d: sum(i["_score"][d] for i in items) / len(items)
                for d in ("spine", "honesty", "depth", "propagation")}
        agent_scores[agent] = {
            "count": len(items),
            "mean_total": sum(totals) / len(totals),
            "dims": dims,
            "items": items,
        }

    # Find stuck points: entries tagged with unsolved/stuck/fail/gap
    stuck_tags = {"gap", "stuck", "blocked", "fail", "partial", "todo", "cert-gap"}
    stuck = [t for t in thoughts
             if t["type"] in ("idea", "task") or
             any(tag in stuck_tags for tag in t.get("tags", [])) or
             any(w in t["content"].lower() for w in ["cert gap", "not yet cert", "missing", "todo", "blocked"])]

    lines = []
    lines.append(f"# QA Weekly Performance Review")
    lines.append(f"**Week of:** {week_of}  |  **Period:** last {since_days} days  |  **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("> **Mission:** Validate and propagate QA as the mathematical language of nature.")
    lines.append("> Ben Iverson's discrete arithmetic + Dale Pond's SVP/Keely. Papers are artifacts, not the goal.")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append(f"| Source | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Open Brain entries | {len(thoughts)} |")
    lines.append(f"| Git commits | {len(commits)} |")
    lines.append(f"| Collab events | {len(collab_events)} |")
    lines.append(f"| Stuck / open items | {len(stuck)} |")
    lines.append("")

    # Overall mission alignment
    all_scores = [t["_score"]["total"] for t in thoughts]
    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        bar = "█" * int(overall) + "░" * (12 - int(overall))
        lines.append(f"**Overall mission alignment:** {overall:.1f}/12  `{bar}`")
        lines.append("")

    # Agent scorecards
    lines.append("## Agent Scorecards")
    lines.append("")

    agent_order = ["codex", "claude", "chatgpt", "gemini", "will", "unknown"]
    for agent in agent_order:
        if agent not in agent_scores:
            continue
        stats = agent_scores[agent]
        name = agent_display_name(agent)
        score = stats["mean_total"]
        bar = "█" * int(score) + "░" * (12 - int(score))
        lines.append(f"### {name}")
        lines.append(f"**Entries this week:** {stats['count']}  |  **Mission score:** {score:.1f}/12  `{bar}`")
        lines.append("")
        lines.append(f"| Dimension | Score (avg) |")
        lines.append(f"|-----------|-------------|")
        dims = stats["dims"]
        lines.append(f"| QA Spine connection | {dims['spine']:.1f}/3 |")
        lines.append(f"| Honest methodology  | {dims['honesty']:.1f}/3 |")
        lines.append(f"| Understanding depth | {dims['depth']:.1f}/3 |")
        lines.append(f"| Propagation value   | {dims['propagation']:.1f}/3 |")
        lines.append("")

        # Top items
        top = sorted(stats["items"], key=lambda x: x["_score"]["total"], reverse=True)[:3]
        if top:
            lines.append("**Top contributions:**")
            for item in top:
                ts = item["timestamp"][:10] if item["timestamp"] else "?"
                snippet = item["content"][:120].replace("\n", " ").strip()
                lines.append(f"- `{ts}` [{item['type']}] {snippet}...")
        lines.append("")

        # Low-scoring items (potential drift)
        low = [i for i in stats["items"] if i["_score"]["total"] <= 2]
        if low:
            lines.append(f"**Possible mission drift ({len(low)} entries scoring ≤2/12):**")
            for item in low[:3]:
                ts = item["timestamp"][:10] if item["timestamp"] else "?"
                snippet = item["content"][:100].replace("\n", " ").strip()
                lines.append(f"- `{ts}` {snippet}...")
        lines.append("")

    # Git commits
    if commits:
        lines.append("## Git Commits This Week")
        lines.append("")
        for c in commits[:20]:
            lines.append(f"- `{c['hash']}` {c['date'][:10]} — {c['subject']}")
        if len(commits) > 20:
            lines.append(f"- *(+{len(commits)-20} more)*")
        lines.append("")

    # Stuck points
    lines.append("## Where Is QA Understanding Stuck?")
    lines.append("")
    if stuck:
        for item in stuck[:10]:
            ts = item["timestamp"][:10] if item["timestamp"] else "?"
            snippet = item["content"][:150].replace("\n", " ").strip()
            lines.append(f"- `{ts}` [{item['type']}] {snippet}")
    else:
        lines.append("*No obvious blockers detected this week.*")
    lines.append("")

    # Top 3 next actions (highest-value open ideas/tasks)
    open_items = [t for t in thoughts if t["type"] in ("idea", "task")]
    open_items.sort(key=lambda x: x["_score"]["total"], reverse=True)
    lines.append("## Top Open Items (by mission score)")
    lines.append("")
    if open_items:
        for item in open_items[:5]:
            ts = item["timestamp"][:10] if item["timestamp"] else "?"
            snippet = item["content"][:160].replace("\n", " ").strip()
            score = item["_score"]["total"]
            lines.append(f"- `{ts}` score={score}/12 — {snippet}...")
    else:
        lines.append("*No open ideas/tasks found.*")
    lines.append("")

    # Linting health
    lines.append("## Linting Health (qa_axiom_linter)")
    lines.append("")
    if lint_stats is None or not lint_stats.get("available"):
        lines.append("*Linter not available — run `python tools/qa_axiom_linter.py --all` manually.*")
    elif "error" in lint_stats and lint_stats.get("errors", 0) < 0:
        lines.append(f"*Linter error: {lint_stats['error']}*")
    else:
        e = lint_stats["errors"]
        w = lint_stats["warnings"]
        status = "🔴 ERRORS PRESENT — must fix before commit" if e > 0 else "🟢 Clean (0 errors)"
        lines.append(f"**Status:** {status}")
        lines.append(f"**Errors:** {e}  |  **Warnings:** {w}")
        lines.append("")
        if lint_stats.get("error_files"):
            lines.append("### Error files")
            for fname, issues in sorted(lint_stats["error_files"].items())[:10]:
                lines.append(f"- `{fname}`: {', '.join(r for r,_ in issues[:3])}")
        if lint_stats.get("warning_files"):
            lines.append("### Warning files (top 10)")
            for fname, issues in sorted(lint_stats["warning_files"].items())[:10]:
                lines.append(f"- `{fname}`: {', '.join(r for r,_ in issues[:3])}")
        lines.append("")
        lines.append("**Improvement target:** reduce warning count week-over-week; zero errors is mandatory.")
    lines.append("")

    # Recursive: note for next review
    lines.append("---")
    lines.append("*This report is auto-generated. Review it, correct agent attributions or scores, and feed corrections back as Open Brain observations tagged `weekly-review`.*")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QA Weekly Performance Review")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (default: 7)")
    parser.add_argument("--out", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    since_days = args.days
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=since_days)
    week_of = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")

    print(f"Fetching last {since_days} days of data...", file=sys.stderr)

    thoughts = fetch_open_brain(since_days)
    print(f"  Open Brain: {len(thoughts)} entries", file=sys.stderr)

    print(f"  Running linter...", file=sys.stderr)
    lint_stats = run_linter_stats()
    print(f"  Linter: {lint_stats.get('errors',0)} errors, {lint_stats.get('warnings',0)} warnings", file=sys.stderr)

    collab = read_collab_events(since)
    print(f"  qa-collab:  {len(collab)} events", file=sys.stderr)

    commits = read_git_log(since_days)
    print(f"  git:        {len(commits)} commits", file=sys.stderr)

    report = generate_report(thoughts, collab, commits, since_days, week_of, lint_stats)

    # Output
    if args.out:
        out_path = Path(args.out)
    else:
        REVIEWS_DIR.mkdir(exist_ok=True)
        out_path = REVIEWS_DIR / f"review_{now.strftime('%Y-%m-%d')}.md"

    out_path.write_text(report)
    print(f"\nReport written to: {out_path}", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
