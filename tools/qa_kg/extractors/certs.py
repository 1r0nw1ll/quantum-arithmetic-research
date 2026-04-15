"""Discover cert families and register them as Cosmos nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Canonical source is qa_meta_validator.FAMILY_SWEEPS — the one list the validator
itself iterates over. We import it directly rather than regex-parse.

Fallback: for families that exist on disk but aren't yet in FAMILY_SWEEPS (or when
the import fails), we scan for directories containing mapping_protocol*.json.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import os
import sys
from pathlib import Path

from tools.qa_kg.kg import KG, Node
from tools.qa_kg.orbit import Tier


REPO = Path(__file__).resolve().parents[3]
META_DIR = REPO / "qa_alphageometry_ptolemy"


def _import_family_sweeps():
    """Import FAMILY_SWEEPS without running the validator's __main__ path."""
    if str(META_DIR) not in sys.path:
        sys.path.insert(0, str(META_DIR))
    try:
        import qa_meta_validator as mv
        return getattr(mv, "FAMILY_SWEEPS", None)
    except Exception:
        return None


def _discover_filesystem_families() -> list[tuple[int | None, str, str]]:
    """Fallback: families = subdirs with mapping_protocol*.json. Returns (id, label, rel)."""
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


def populate(kg: KG, *, run_validator: bool = False) -> list[str]:
    """Register cert family nodes from FAMILY_SWEEPS + filesystem fallback.

    run_validator is accepted for API compatibility with earlier drafts but
    not used here — trust the registry's presence as the Cosmos gate, same as
    the meta-validator's own trust model. A later pass (task #3, [225] cert)
    will re-validate and demote to Satellite on failure.
    """
    sweeps = _import_family_sweeps()
    ids: list[str] = []
    seen_ids: set[str] = set()

    if sweeps:
        for entry in sweeps:
            # (fam_id, label, validator_fn, pass_desc, doc_slug, family_root_rel, must_have_dedicated_root)
            fam_id = entry[0]
            label = entry[1]
            pass_desc = entry[3] if len(entry) > 3 else ""
            doc_slug = entry[4] if len(entry) > 4 else ""
            root_rel = entry[5] if len(entry) > 5 else ""
            nid = f"cert:{fam_id}"
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            body = "\n".join([
                f"fam_id: {fam_id}",
                f"doc_slug: {doc_slug}",
                f"family_root_rel: {root_rel}",
                f"pass_desc: {str(pass_desc)[:400]}",
            ])
            kg.upsert_node(Node(
                id=nid, node_type="Cert", title=str(label), body=body,
                tier=Tier.COSMOS,
                source=f"qa_alphageometry_ptolemy/{root_rel}" if root_rel and root_rel != "." else "qa_alphageometry_ptolemy/qa_meta_validator.py:FAMILY_SWEEPS",
                vetted_by=nid,
            ))
            ids.append(nid)

    # Filesystem fallback: families with mapping_protocol*.json but not in registry.
    for fam_id, label, rel in _discover_filesystem_families():
        nid = f"cert:fs:{label}"
        if nid in seen_ids:
            continue
        # Skip if already covered via root_rel match in FAMILY_SWEEPS.
        seen_ids.add(nid)
        kg.upsert_node(Node(
            id=nid, node_type="Cert", title=label,
            body=f"filesystem-discovered; family_root_rel: {rel}",
            tier=Tier.SATELLITE,  # unregistered = in-flight, not canonical
            source=f"qa_alphageometry_ptolemy/{rel}",
        ))
        ids.append(nid)

    return ids
