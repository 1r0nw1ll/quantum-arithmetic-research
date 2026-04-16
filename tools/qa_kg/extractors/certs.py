"""Discover cert families as Cert-typed nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Candidate F handles coord assignment: b = dr(char_ord_sum(title+body)), e = rank 2.
Source: qa_meta_validator.FAMILY_SWEEPS (registered) + filesystem discovery.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import json
import os
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


def _is_frozen(cert_dir: Path) -> bool:
    """Check if a cert directory has _status: frozen in its mapping_protocol_ref.json."""
    for fname in ("mapping_protocol_ref.json", "mapping_protocol.json"):
        mp = cert_dir / fname
        if mp.exists():
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
                if data.get("_status") == "frozen":
                    return True
            except (json.JSONDecodeError, OSError):
                pass
    return False


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
            frozen = _is_frozen(cert_dir) if cert_dir else False
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
                lifecycle_state="deprecated" if frozen else "current",
            ))
            ids.append(nid)

    for _fam_id, label, rel in _discover_filesystem_families():
        nid = f"cert:fs:{label}"
        if nid in seen:
            continue
        seen.add(nid)
        cert_dir = META_DIR / rel
        frozen = _is_frozen(cert_dir)
        kg.upsert_node(Node(
            id=nid, node_type="Cert", title=label,
            body=f"filesystem-discovered; family_root_rel: {rel}",
            source=f"qa_alphageometry_ptolemy/{rel}",
            vetted_by="",
            authority="derived",
            epistemic_status="certified",
            method="cert_validator",
            source_locator=f"file:qa_alphageometry_ptolemy/{rel}",
            lifecycle_state="deprecated" if frozen else "current",
        ))
        ids.append(nid)

    return ids
