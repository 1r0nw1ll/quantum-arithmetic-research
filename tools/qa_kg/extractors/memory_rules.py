"""Parse MEMORY.md Hard Rules + CLAUDE.md Hard Rules as Rule-typed nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Coord via Candidate F: b = dr(char_ord_sum(title+body)), e = NODE_TYPE_RANK['Rule'] = 6.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import re
from pathlib import Path

from tools.qa_kg.kg import KG, Node


_REPO = Path(__file__).resolve().parents[3]


def _default_memory_md_path() -> Path:
    """Mirror Claude Code's per-project memory layout.

    Claude Code stores per-project memory under
    `~/.claude/projects/<slug>/memory/MEMORY.md` where `<slug>` is the
    repo's absolute path with `/` replaced by `-` and a leading dash.
    Computing this from `_REPO` rather than hardcoding the player2 slug
    means Mac (`-Users-player3-signal-experiments`) and Linux
    (`-home-player2-signal-experiments`) both find their actual memory
    file in production builds without one machine silently extracting
    zero MEMORY rules.
    """
    # Claude Code replaces both `/` and `_` in the absolute path with `-`
    # to form the project slug (so `signal_experiments` → `signal-experiments`).
    slug = "-" + str(_REPO).lstrip("/").replace("/", "-").replace("_", "-")
    return Path.home() / ".claude" / "projects" / slug / "memory" / "MEMORY.md"


_MEMORY_MD = _default_memory_md_path()
_CLAUDE_MD = _REPO / "CLAUDE.md"
_H3_HARD = re.compile(r"^###\s+(.+?)\s*\((?:HARD)(?:[^)]*)\)", re.M)


def _extract_rules(text: str, source_file: str) -> list[tuple[str, str, str, str]]:
    """Returns list of (nid, title, body, source_locator)."""
    results: list[tuple[str, str, str, str]] = []
    for m in _H3_HARD.finditer(text):
        title = m.group(1).strip()
        start = m.end()
        nxt = re.search(r"^#{2,3}\s+", text[start:], re.M)
        body = text[start:start + nxt.start()].strip() if nxt else text[start:].strip()
        body = body[:1200]
        nid = "rule:" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        results.append((nid, title, body, f"file:{source_file}"))
    return results


def populate(
    kg: KG,
    *,
    memory_md_path: Path | None = None,
) -> list[str]:
    """Populate Rule-typed nodes from MEMORY.md + CLAUDE.md Hard Rules.

    Phase 5 C#1: `memory_md_path` overrides the hardcoded user-home path
    so fixture-driven builds can freeze MEMORY.md content. None == default.
    CLAUDE.md is always read from repo root (covered by pinned-repo
    materialization in the determinism cert's D2/D3 gates).
    """
    ids: list[str] = []

    mem_path = memory_md_path if memory_md_path is not None else _MEMORY_MD
    if mem_path.exists():
        text = mem_path.read_text(encoding="utf-8")
        # Stable cross-platform identifier: store "MEMORY.md" rather than
        # str(mem_path), since the latter encodes Path.home() ("/Users/..."
        # on Mac vs "/home/..." on Linux) and the fixture worktree's
        # absolute path. Those leaks were the entire source of [228] D2/D3
        # cross-platform graph_hash divergence prior to this commit.
        # The actual file location remains discoverable from the cert's
        # _meta_ledger.json git_head + the manifest's repo_head, which is
        # where it semantically belongs (build-time provenance, not graph
        # row content).
        for nid, title, body, src_loc in _extract_rules(text, "MEMORY.md"):
            kg.upsert_node(Node(
                id=nid, node_type="Rule", title=title, body=body,
                source="MEMORY.md", vetted_by="MEMORY.md",
                authority="internal",
                epistemic_status="interpretation",
                method="memory_rule",
                source_locator=src_loc,
            ))
            ids.append(nid)

    if _CLAUDE_MD.exists():
        text = _CLAUDE_MD.read_text(encoding="utf-8")
        for nid, title, body, src_loc in _extract_rules(text, "CLAUDE.md"):
            nid = f"rule:claude:{nid.removeprefix('rule:')}"
            kg.upsert_node(Node(
                id=nid, node_type="Rule", title=f"[CLAUDE.md] {title}", body=body,
                source="CLAUDE.md", vetted_by="CLAUDE.md",
                authority="internal",
                epistemic_status="interpretation",
                method="memory_rule",
                source_locator=src_loc,
            ))
            ids.append(nid)

    return ids
