"""Parse MEMORY.md and ingest Hard Rule entries as Cosmos nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Hard Rules are the project's non-negotiable constraints — they qualify for
Cosmos (canonical) tier. Detected by section headers 'Hard Rules' / '### ... (HARD)'.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import re
from pathlib import Path

from tools.qa_kg.kg import KG, Node
from tools.qa_kg.orbit import Tier


MEMORY_MD = Path.home() / ".claude/projects/-home-far-player2-signal-experiments/memory/MEMORY.md"
# Canonical path fallback (ALT since Path normalization dashes vary)
_ALT = Path.home() / ".claude/projects/-home-player2-signal-experiments/memory/MEMORY.md"


def _find_memory_md() -> Path | None:
    for p in (_ALT, MEMORY_MD):
        if p.exists():
            return p
    return None


_H3_HARD = re.compile(r"^###\s+(.+?)\s*\((?:HARD)(?:[^)]*)\)", re.M)
_H2_HARD_BLOCK = re.compile(r"^##\s+Hard Rules.*$", re.M | re.I)


def populate(kg: KG) -> list[str]:
    path = _find_memory_md()
    if path is None:
        return []
    text = path.read_text(encoding="utf-8")

    ids: list[str] = []
    for m in _H3_HARD.finditer(text):
        title = m.group(1).strip()
        start = m.end()
        # body = text until next H2/H3 header or EOF
        nxt = re.search(r"^#{2,3}\s+", text[start:], re.M)
        body = text[start:start + nxt.start()].strip() if nxt else text[start:].strip()
        # trim body to first paragraph + bullets, cap at 1200 chars
        body = body[:1200]
        nid = "rule:" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        kg.upsert_node(Node(
            id=nid,
            node_type="Rule",
            title=title,
            body=body,
            tier=Tier.COSMOS,
            source=str(path),
            vetted_by="MEMORY.md",
        ))
        ids.append(nid)
    return ids
