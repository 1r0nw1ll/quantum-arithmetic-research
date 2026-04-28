# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG diagnostic v0; reads graph + fixtures, emits curation queue. No empirical QA state, no observer projections. -->
"""QA-KG diagnostic v0 — single-invocation report.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Run (in order):
  python -m tools.qa_kg.cli build      # rebuild qa_kg.db from current fixtures
  python tools/qa_kg/diagnostics.py    # this script

Skipping the build step makes the report reflect the *last* DB state, not the
current fixture state — see docs/specs/QA_MEM_CURATION_METHOD.md for the
worked Wildberger example where ~92 claim nodes were already in fixtures but
not yet in the DB at diagnostic time.

Emits four sections, fixed order, plain text:
  1. snapshot header (one copy-pasteable line of counts)
  2. contradiction report (every contradicts edge with anchor + cert context)
  3. citation density report (claim->work anchor distribution + top works)
  4. fixture coverage / curation queue (per-corpus derived_from gap, ranked
     by curation need so the operator knows which fixture to author next)

No flags, no JSON, no candidate cert suggestions, no graph rendering. Add
those only when curation in practice proves they're needed.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

_QA_KG_DIR = Path(__file__).resolve().parent
_DB_PATH = _QA_KG_DIR / "qa_kg.db"
_FIXTURES_DIR = _QA_KG_DIR / "fixtures"
_FIXTURE_GLOB = "source_claims_*.json"

_RULE = "=" * 72
_THIN_RULE = "-" * 72


def _snapshot_header(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    nodes = cur.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges = cur.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    derived = cur.execute(
        "SELECT COUNT(*) FROM edges WHERE edge_type='derived-from'"
    ).fetchone()[0]
    claims = cur.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_type='Claim'"
    ).fetchone()[0]
    works = cur.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_type='Work'"
    ).fetchone()[0]
    certs = cur.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_type='Cert'"
    ).fetchone()[0]

    fixtures = sorted(_FIXTURES_DIR.glob(_FIXTURE_GLOB))
    with_links = sum(
        1
        for f in fixtures
        if json.loads(f.read_text(encoding="utf-8")).get("derived_from")
    )

    print(
        f"nodes={nodes} edges={edges} derived_from={derived} "
        f"fixtures_with_links={with_links}/{len(fixtures)} "
        f"claims={claims} works={works} certs={certs}"
    )


def _build_anchor_index(con: sqlite3.Connection) -> dict[str, list[str]]:
    anchors: dict[str, list[str]] = defaultdict(list)
    q = """
        SELECT e.src_id, n.title
        FROM edges e
        JOIN nodes n ON e.dst_id = n.id
        WHERE e.edge_type='quoted-from'
    """
    for src, title in con.execute(q):
        anchors[src].append(title)
    return anchors


def _build_cert_index(con: sqlite3.Connection) -> dict[str, list[str]]:
    cert_links: dict[str, list[str]] = defaultdict(list)
    q = """
        SELECT e.dst_id, n.title
        FROM edges e
        JOIN nodes n ON e.src_id = n.id
        WHERE e.edge_type='derived-from' AND n.node_type='Cert'
    """
    for dst, title in con.execute(q):
        cert_links[dst].append(title)
    return cert_links


def _contradiction_report(con: sqlite3.Connection) -> None:
    print()
    print(_RULE)
    print("CONTRADICTION REPORT")
    print(_RULE)

    q = """
        SELECT e.src_id, n1.title, e.dst_id, n2.title, e.provenance
        FROM edges e
        JOIN nodes n1 ON e.src_id = n1.id
        JOIN nodes n2 ON e.dst_id = n2.id
        WHERE e.edge_type='contradicts'
        ORDER BY e.src_id, e.dst_id
    """
    rows = list(con.execute(q))
    print(f"{len(rows)} contradicts edges\n")

    anchors = _build_anchor_index(con)
    cert_links = _build_cert_index(con)

    for src, src_title, dst, dst_title, prov in rows:
        reason = ""
        if prov:
            try:
                p = json.loads(prov)
                reason = p.get("reason", "")
            except json.JSONDecodeError:
                pass
        print(f"  {src_title[:64]}")
        print(f"    contradicts: {dst_title[:64]}")
        print(
            f"    src: {len(anchors.get(src, []))} works / {len(cert_links.get(src, []))} certs"
            f"   |   dst: {len(anchors.get(dst, []))} works / {len(cert_links.get(dst, []))} certs"
        )
        if reason:
            print(f"    reason: {reason}")
        print()


def _citation_density_report(con: sqlite3.Connection) -> None:
    print(_RULE)
    print("CITATION DENSITY REPORT")
    print(_RULE)

    total_claims = con.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_type='Claim'"
    ).fetchone()[0]

    claim_work_count: dict[str, int] = defaultdict(int)
    q = """
        SELECT e.src_id
        FROM edges e
        JOIN nodes n ON e.src_id = n.id
        WHERE e.edge_type='quoted-from' AND n.node_type='Claim'
    """
    for (src,) in con.execute(q):
        claim_work_count[src] += 1

    with_anchors = len(claim_work_count)
    no_anchors = total_claims - with_anchors
    pct = (100 * with_anchors / total_claims) if total_claims else 0

    print(f"total claims:          {total_claims}")
    print(f"claims with >=1 work:  {with_anchors} ({pct:.0f}%)")
    print(f"claims with 0 works:   {no_anchors}")
    print()

    dist: dict[int, int] = defaultdict(int)
    for c in claim_work_count.values():
        dist[c] += 1
    dist[0] = no_anchors
    print("anchors-per-claim distribution:")
    for k in sorted(dist):
        print(f"  {k:>2} anchors:  {dist[k]:>4} claims")
    print()

    work_claim_count: dict[str, int] = defaultdict(int)
    q2 = """
        SELECT e.dst_id
        FROM edges e
        JOIN nodes n ON e.dst_id = n.id
        WHERE e.edge_type='quoted-from' AND n.node_type='Work'
    """
    for (dst,) in con.execute(q2):
        work_claim_count[dst] += 1

    work_titles = {
        wid: title
        for wid, title in con.execute(
            "SELECT id, title FROM nodes WHERE node_type='Work'"
        )
    }

    top_works = sorted(work_claim_count.items(), key=lambda kv: -kv[1])[:10]
    print("top 10 works by claim anchor count:")
    for wid, count in top_works:
        title = work_titles.get(wid, wid)[:60]
        print(f"  {count:>3}  {title}")
    print()


def _fixture_coverage_report() -> None:
    print(_RULE)
    print("FIXTURE COVERAGE / CURATION QUEUE")
    print(_RULE)

    fixtures = sorted(_FIXTURES_DIR.glob(_FIXTURE_GLOB))
    rows = []
    cross_fixture_total = 0
    cross_fixture_corpora = 0
    axiom_total = 0

    for f in fixtures:
        d = json.loads(f.read_text(encoding="utf-8"))
        name = f.stem.replace("source_claims_", "")
        works = len(d.get("works", []))
        own_claim_ids = {
            f"sc:{c['id']}" for c in d.get("claims", []) if isinstance(c, dict) and "id" in c
        }
        claims = len(own_claim_ids)
        derived = d.get("derived_from", []) or []
        derived_dsts = [
            e["dst"] for e in derived if isinstance(e, dict) and "dst" in e
        ]
        own_linked = own_claim_ids & set(derived_dsts)
        axiom_targets = [d for d in derived_dsts if d.startswith("axiom:")]
        cross_fixture = [
            d for d in derived_dsts
            if not d.startswith("axiom:") and d not in own_claim_ids
        ]
        own_unlinked = claims - len(own_linked)
        pct = (100 * len(own_linked) / claims) if claims else 0.0

        if cross_fixture:
            cross_fixture_corpora += 1
            cross_fixture_total += len(cross_fixture)
        axiom_total += len(axiom_targets)

        rows.append((name, works, claims, len(derived), own_unlinked, pct))

    rows.sort(key=lambda r: (r[3] > 0, -r[2]))

    print(
        f"{'corpus':<22}{'works':>6}{'claims':>7}{'der_from':>10}"
        f"{'own_unlinked':>14}{'%own_linked':>13}"
    )
    print(_THIN_RULE)
    for name, w, c, d, u, p in rows:
        flag = "  <- needs curation" if d == 0 and c > 0 else ""
        print(f"{name:<22}{w:>6}{c:>7}{d:>10}{u:>14}{p:>12.0f}%{flag}")
    print()
    print(
        f"cross-fixture derived_from: {cross_fixture_total} entries across "
        f"{cross_fixture_corpora} fixtures | axiom-target derived_from: {axiom_total}"
    )
    print()


def main() -> None:
    if not _DB_PATH.exists():
        raise FileNotFoundError(f"qa_kg.db not found at {_DB_PATH}")
    if not _FIXTURES_DIR.is_dir():
        raise FileNotFoundError(f"fixtures dir not found at {_FIXTURES_DIR}")

    con = sqlite3.connect(_DB_PATH)
    try:
        _snapshot_header(con)
        _contradiction_report(con)
        _citation_density_report(con)
        _fixture_coverage_report()
    finally:
        con.close()


if __name__ == "__main__":
    main()
