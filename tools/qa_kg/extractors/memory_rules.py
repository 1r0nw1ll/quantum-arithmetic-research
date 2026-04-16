"""Parse MEMORY.md Hard Rules as Rule-typed nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Coord via Candidate F: b = dr(char_ord_sum(title+body)), e = NODE_TYPE_RANK['Rule'] = 6.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import re
from pathlib import Path

from tools.qa_kg.kg import KG, Node


_MEMORY_MD = Path.home() / ".claude/projects/-home-player2-signal-experiments/memory/MEMORY.md"
_H3_HARD = re.compile(r"^###\s+(.+?)\s*\((?:HARD)(?:[^)]*)\)", re.M)


def populate(kg: KG) -> list[str]:
    if not _MEMORY_MD.exists():
        return []
    text = _MEMORY_MD.read_text(encoding="utf-8")
    ids: list[str] = []
    for m in _H3_HARD.finditer(text):
        title = m.group(1).strip()
        start = m.end()
        nxt = re.search(r"^#{2,3}\s+", text[start:], re.M)
        body = text[start:start + nxt.start()].strip() if nxt else text[start:].strip()
        body = body[:1200]
        nid = "rule:" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        kg.upsert_node(Node(
            id=nid, node_type="Rule", title=title, body=body,
            source=str(_MEMORY_MD), vetted_by="MEMORY.md",
        ))
        ids.append(nid)
    return ids
