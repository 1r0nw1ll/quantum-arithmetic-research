# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG infrastructure; ingests agent-attributed notes -->
"""Ingest agent-attributed notes from collab-bus, activity log, and OB.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Phase 2: all three sources produce Thought nodes with authority=agent.
  - collab-bus events (session_handoff_note, kg_promotion)
  - collab_log_activity rows
  - OB thoughts with explicit originSessionId

Edges: keyword-co-occurs only, method='keyword', confidence=0.3.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import hashlib
import logging
import re
from typing import Iterable

from tools.qa_kg.kg import KG, Edge, FirewallViolation, Node
from tools.qa_kg.extractors.axioms import CANONICAL_AXIOM_CODES
from tools.qa_kg.extractors.ob import OBThought


_log = logging.getLogger("qa_kg.extractors.agent_notes")
_CERT_REF_RE = re.compile(r"\[(\d+)\]")
_AXIOM_RE = re.compile(r"\b(" + "|".join(CANONICAL_AXIOM_CODES) + r")\b")


def _try_edge(kg: KG, edge: Edge) -> bool:
    try:
        kg.upsert_edge(edge)
        return True
    except FirewallViolation as exc:
        _log.warning("FirewallViolation in agent_notes edge: %s", exc)
        return False
    except ValueError as exc:
        _log.debug("Edge skipped (unknown node): %s", exc)
        return False


def _emit_keyword_edges(kg: KG, node_id: str, body: str, provenance_prefix: str) -> int:
    """Emit keyword-co-occurs edges for cert refs + axiom codes. Returns count."""
    cert_ids = {r["id"] for r in kg.conn.execute(
        "SELECT id FROM nodes WHERE node_type='Cert'"
    ).fetchall()}
    edges = 0
    for n in _CERT_REF_RE.findall(body):
        tgt = f"cert:{n}"
        if tgt in cert_ids:
            if _try_edge(kg, Edge(
                src_id=node_id, dst_id=tgt,
                edge_type="keyword-co-occurs",
                confidence=0.3, method="keyword",
                provenance=f"{provenance_prefix}.cert_ref",
            )):
                edges += 1
    for code in set(_AXIOM_RE.findall(body)):
        if _try_edge(kg, Edge(
            src_id=node_id, dst_id=f"axiom:{code}",
            edge_type="keyword-co-occurs",
            confidence=0.3, method="keyword",
            provenance=f"{provenance_prefix}.axiom_code:{code}",
        )):
            edges += 1
    return edges


def _event_id(event: dict) -> str:
    """Deterministic ID from event type + timestamp + session."""
    raw = f"{event.get('event_type', '')}/{event.get('ts', '')}/{event.get('session', '')}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"agent_event:{h}"


def ingest_collab_events(kg: KG, events: Iterable[dict]) -> dict[str, int]:
    """Ingest collab-bus events (session_handoff_note, kg_promotion).

    Each event dict should have: event_type, ts, session, data (dict with body/summary).
    All become authority=agent, epistemic_status=observation.
    """
    nodes = 0
    edges = 0
    for ev in events:
        etype = ev.get("event_type", "")
        if etype not in ("session_handoff_note", "kg_promotion"):
            continue
        nid = _event_id(ev)
        data = ev.get("data", {})
        body_text = data.get("body") or data.get("summary") or str(data)
        title = f"[{etype}] {body_text[:100]}"
        kg.upsert_node(Node(
            id=nid, node_type="Thought",
            title=title,
            body=f"event_type: {etype}\nts: {ev.get('ts', '')}\n"
                 f"session: {ev.get('session', '')}\n---\n{body_text[:2000]}",
            source=f"collab_bus:{etype}:{ev.get('ts', '')}",
            authority="agent",
            epistemic_status="observation",
            method="collab_bus",
            source_locator=f"collab:{nid.removeprefix('agent_event:')}",
        ))
        nodes += 1
        edges += _emit_keyword_edges(kg, nid, body_text, "extractors.agent_notes.collab")
    return {"nodes": nodes, "edges": edges}


def ingest_activity_log(kg: KG, rows: Iterable[dict]) -> dict[str, int]:
    """Ingest collab_log_activity rows.

    Each row dict should have: ts, session, activity, detail (optional).
    All become authority=agent, epistemic_status=observation.
    """
    nodes = 0
    edges = 0
    for row in rows:
        raw = f"activity/{row.get('ts', '')}/{row.get('session', '')}/{row.get('activity', '')}"
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        nid = f"agent_activity:{h}"
        activity = row.get("activity", "")
        detail = row.get("detail", "")
        body_text = f"{activity}\n{detail}" if detail else activity
        kg.upsert_node(Node(
            id=nid, node_type="Thought",
            title=f"[activity] {activity[:100]}",
            body=f"ts: {row.get('ts', '')}\nsession: {row.get('session', '')}\n"
                 f"---\n{body_text[:2000]}",
            source=f"collab_activity:{row.get('ts', '')}",
            authority="agent",
            epistemic_status="observation",
            method="collab_activity_log",
            source_locator=f"collab:{nid.removeprefix('agent_activity:')}",
        ))
        nodes += 1
        edges += _emit_keyword_edges(kg, nid, body_text, "extractors.agent_notes.activity")
    return {"nodes": nodes, "edges": edges}


def ingest_ob_with_session(kg: KG, thoughts: Iterable[OBThought]) -> dict[str, int]:
    """Ingest OB thoughts that have explicit originSessionId.

    Filters to thoughts whose body contains session attribution markers.
    All become authority=agent, epistemic_status=observation,
    method=ob_capture_session (distinct from ob.py's ob_capture).
    """
    _SESSION_RE = re.compile(r"(?:originSessionId|session[-_]id)\s*[:=]\s*\S+", re.I)
    nodes = 0
    edges = 0
    for t in thoughts:
        if not _SESSION_RE.search(t.body):
            continue
        title = (t.body.split("\n", 1)[0] or "")[:120]
        tags_str = ",".join(t.tags)
        body_short = t.body[:2000]
        kg.upsert_node(Node(
            id=t.id, node_type="Thought",
            title=title or f"OB-agent {t.ts}",
            body=f"ts: {t.ts}\ntype: {t.thought_type}\ntags: {tags_str}\n---\n{body_short}",
            source=f"open_brain:{t.ts}",
            authority="agent",
            epistemic_status="observation",
            method="ob_capture_session",
            source_locator=f"ob:{t.id.removeprefix('ob:')}",
        ))
        nodes += 1
        edges += _emit_keyword_edges(kg, t.id, t.body, "extractors.agent_notes.ob_session")
    return {"nodes": nodes, "edges": edges}
