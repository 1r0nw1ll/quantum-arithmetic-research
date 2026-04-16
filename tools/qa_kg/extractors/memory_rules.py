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
_MEMORY_MD = Path.home() / ".claude/projects/-home-player2-signal-experiments/memory/MEMORY.md"
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


def populate(kg: KG) -> list[str]:
    ids: list[str] = []

    if _MEMORY_MD.exists():
        text = _MEMORY_MD.read_text(encoding="utf-8")
        for nid, title, body, src_loc in _extract_rules(text, str(_MEMORY_MD)):
            kg.upsert_node(Node(
                id=nid, node_type="Rule", title=title, body=body,
                source=str(_MEMORY_MD), vetted_by="MEMORY.md",
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
