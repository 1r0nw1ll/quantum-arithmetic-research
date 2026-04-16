"""Discover cert families as Cert-typed nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Candidate F handles coord assignment: b = dr(char_ord_sum(title+body)), e = rank 2.
Source: qa_meta_validator.FAMILY_SWEEPS (registered) + filesystem discovery.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import json
import os
import re
import sys
from pathlib import Path

from tools.qa_kg.kg import KG, Node


REPO = Path(__file__).resolve().parents[3]
META_DIR = REPO / "qa_alphageometry_ptolemy"


def _import_family_sweeps():
    if str(META_DIR) not in sys.path:
        sys.path.insert(0, str(META_DIR))
    try:
        import qa_meta_validator as mv
        return getattr(mv, "FAMILY_SWEEPS", None)
    except Exception:
        return None


def _discover_filesystem_families() -> list[tuple[int | None, str, str]]:
    if not META_DIR.exists():
        return []
    seen: set[str] = set()
    out: list[tuple[int | None, str, str]] = []
    for mp in META_DIR.rglob("mapping_protocol*.json"):
        rel_dir = mp.parent.relative_to(META_DIR)
        rel_str = str(rel_dir).replace(os.sep, "/")
        if rel_str in seen:
            continue
        seen.add(rel_str)
        label = rel_dir.name or "qa_alphageometry_ptolemy_root"
        out.append((None, label, rel_str))
    return out


_SIBLING_VERSION_RE = re.compile(r"^(.+)_v(\d+)$")


def _is_frozen(cert_dir: Path) -> bool:
    """True if the cert dir's mapping protocol has a `_status` value that
    begins with 'frozen' (case-insensitive). Accepts both the canonical
    short form `"frozen"` and longer banner strings like
    `"FROZEN — AUDIT-ONLY. ..."` (historical v1 pattern)."""
    for fname in ("mapping_protocol_ref.json", "mapping_protocol.json"):
        mp = cert_dir / fname
        if mp.exists():
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
                status = data.get("_status")
                if isinstance(status, str) and status.strip().lower().startswith("frozen"):
                    return True
            except (json.JSONDecodeError, OSError):
                pass
    return False


def _has_successor_sibling(cert_dir: Path) -> bool:
    """True if a sibling directory matching <base>_v<N+1> exists next to
    `cert_dir` where `cert_dir.name` matches `<base>_v<N>`. Used by the
    Phase 3 lifecycle bridge (§6a) to distinguish frozen-with-successor
    (→ 'superseded') from frozen-without-successor (→ 'deprecated')."""
    m = _SIBLING_VERSION_RE.match(cert_dir.name)
    if not m:
        return False
    base, n_str = m.group(1), m.group(2)
    try:
        n = int(n_str)
    except ValueError:
        return False
    parent = cert_dir.parent
    for sibling in parent.iterdir():
        if not sibling.is_dir():
            continue
        sm = _SIBLING_VERSION_RE.match(sibling.name)
        if sm and sm.group(1) == base:
            try:
                if int(sm.group(2)) > n:
                    return True
            except ValueError:
                continue
    return False


def _lifecycle_for_status(cert_dir: Path | None) -> str:
    """Translate `_status` file marker to `lifecycle_state` column value.

    Phase 3 §6a lifecycle bridge. Keeps KG8 (file-level frozen) aligned
    with KG13 (node-level lifecycle_state) so the two mechanisms don't
    drift.

      frozen + successor sibling exists  → 'superseded'
      frozen + no successor sibling      → 'deprecated'
      not frozen (missing _status or current) → 'current'
    """
    if cert_dir is None:
        return "current"
    if not _is_frozen(cert_dir):
        return "current"
    if _has_successor_sibling(cert_dir):
        return "superseded"
    return "deprecated"


def populate(kg: KG, *, run_validator: bool = False) -> list[str]:
    sweeps = _import_family_sweeps()
    ids: list[str] = []
    seen: set[str] = set()

    if sweeps:
        for entry in sweeps:
            fam_id = entry[0]
            label = entry[1]
            pass_desc = entry[3] if len(entry) > 3 else ""
            doc_slug = entry[4] if len(entry) > 4 else ""
            root_rel = entry[5] if len(entry) > 5 else ""
            nid = f"cert:{fam_id}"
            if nid in seen:
                continue
            seen.add(nid)
            body = "\n".join([
                f"fam_id: {fam_id}",
                f"doc_slug: {doc_slug}",
                f"family_root_rel: {root_rel}",
                f"pass_desc: {str(pass_desc)[:400]}",
            ])
            cert_dir = META_DIR / root_rel if root_rel and root_rel != "." else None
            lifecycle = _lifecycle_for_status(cert_dir)
            source_loc = (
                f"file:qa_alphageometry_ptolemy/{root_rel}"
                if root_rel and root_rel != "."
                else "file:qa_alphageometry_ptolemy/qa_meta_validator.py"
            )
            kg.upsert_node(Node(
                id=nid, node_type="Cert", title=str(label), body=body,
                source=(f"qa_alphageometry_ptolemy/{root_rel}" if root_rel and root_rel != "."
                        else "qa_alphageometry_ptolemy/qa_meta_validator.py:FAMILY_SWEEPS"),
                vetted_by="",
                authority="derived",
                epistemic_status="certified",
                method="cert_validator",
                source_locator=source_loc,
                lifecycle_state=lifecycle,
            ))
            ids.append(nid)

    for _fam_id, label, rel in _discover_filesystem_families():
        nid = f"cert:fs:{label}"
        if nid in seen:
            continue
        seen.add(nid)
        cert_dir = META_DIR / rel
        lifecycle = _lifecycle_for_status(cert_dir)
        kg.upsert_node(Node(
            id=nid, node_type="Cert", title=label,
            body=f"filesystem-discovered; family_root_rel: {rel}",
            source=f"qa_alphageometry_ptolemy/{rel}",
            vetted_by="",
            authority="derived",
            epistemic_status="certified",
            method="cert_validator",
            source_locator=f"file:qa_alphageometry_ptolemy/{rel}",
            lifecycle_state=lifecycle,
        ))
        ids.append(nid)

    return ids
