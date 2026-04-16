"""QA-KG command-line entry.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Usage:
    python -m tools.qa_kg.cli build            # populate axioms + memory rules + certs
    python -m tools.qa_kg.cli stats
    python -m tools.qa_kg.cli search "<query>" [--tier cosmos] [-k 10]
    python -m tools.qa_kg.cli search-ranked "<query>" \\
        [--min-authority internal] [--domain ""] [--valid-at 2026-04-16T00:00:00Z] [-k 10]
    python -m tools.qa_kg.cli digest [--tier cosmos] [-k 40]
    python -m tools.qa_kg.cli check [--tier singularity]
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import argparse
import json
import sys
from pathlib import Path

from tools.qa_kg.kg import connect
from tools.qa_kg.predicate import check_all
from tools.qa_kg.extractors import axioms as x_axioms
from tools.qa_kg.extractors import memory_rules as x_rules
from tools.qa_kg.extractors import certs as x_certs
from tools.qa_kg.extractors import edges as x_edges
from tools.qa_kg.extractors import ob as x_ob
from tools.qa_kg.extractors import arag as x_arag
from tools.qa_kg.extractors import source_claims as x_source_claims


def _fmt(row) -> dict:
    return {k: row[k] for k in row.keys()}


def cmd_build(args) -> int:
    kg = connect(args.db)
    a = x_axioms.populate(kg)
    r = x_rules.populate(kg)
    c = x_certs.populate(kg, run_validator=args.validate)
    # Phase 3 source-claim seed. Runs AFTER certs.populate so cert nodes
    # (cert:fs:qa_kg_consistency_cert_v<N>) exist for the supersedes chain
    # in the seed to attach to, and BEFORE edges.populate so keyword
    # co-occurrence edges can link against SourceClaim bodies too.
    sc = x_source_claims.populate(kg)
    e = x_edges.populate(kg)
    print(
        f"axioms: {len(a)}  rules: {len(r)}  certs: {len(c)}  "
        f"source_claims: works={sc['works']} claims={sc['claims']} "
        f"observations={sc['observations']} contradicts={sc['contradicts']} "
        f"supersedes={sc['supersedes']}  edges: {e}"
    )
    return 0


def cmd_stats(args) -> int:
    kg = connect(args.db)
    print(json.dumps(kg.stats(), indent=2))
    return 0


def cmd_search(args) -> int:
    kg = connect(args.db)
    rows = kg.search(args.query, tier=args.tier, k=args.k)
    print(json.dumps([_fmt(r) for r in rows], indent=2, default=str))
    return 0


def cmd_search_ranked(args) -> int:
    """Phase 4 authority-tiered ranker entry point. Prints score_breakdown."""
    import datetime as _dt
    kg = connect(args.db)
    valid_at = None
    if args.valid_at:
        s = args.valid_at.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        valid_at = _dt.datetime.fromisoformat(s)
    domain = args.domain if args.domain else None
    hits = kg.search_authority_ranked(
        args.query, min_authority=args.min_authority,
        domain=domain, valid_at=valid_at, k=args.k,
    )
    out = []
    for h in hits:
        out.append({
            "id": h.node["id"],
            "title": h.node["title"],
            "authority": h.authority,
            "lifecycle_state": h.node["lifecycle_state"],
            "epistemic_status": h.node["epistemic_status"],
            "score": h.score,
            "contradiction_state": h.contradiction_state,
            "provenance_depth": h.provenance_depth,
            "score_breakdown": h.score_breakdown,
        })
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_digest(args) -> int:
    kg = connect(args.db)
    rows = kg.digest(tier=args.tier, limit=args.k)
    for r in rows:
        print(f"[{r['tier']}] ({r['idx_b']},{r['idx_e']}) {r['id']} — {r['title']}")
    return 0


def cmd_check(args) -> int:
    kg = connect(args.db)
    results = check_all(kg.conn, tier=args.tier)
    for nid, res in results.items():
        flag = "OK" if res.ok else "FAIL"
        print(f"{flag}  {nid}  {res.msg}")
    fails = [nid for nid, r in results.items() if not r.ok]
    return 0 if not fails else 2


def cmd_arag_search(args) -> int:
    rows = x_arag.search(args.query, k=args.k, source=args.source)
    for r in rows:
        print(f"[archive/{r['source']}] ({r['arag_coord'][0]},{r['arag_coord'][1]}) {r['msg_id']}  score={r['score']:.2f}")
        print(f"    {r['conv_title']} ({r['ts']})")
        print(f"    {r['preview'][:200]}")
    return 0


def cmd_arag_promote(args) -> int:
    kg = connect(args.db)
    nid = x_arag.promote(kg, args.msg_id, reason=args.reason)
    print(f"promoted to node: {nid} (tier=unassigned)")
    return 0


def cmd_ingest_ob(args) -> int:
    kg = connect(args.db)
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    stats = x_ob.ingest_markdown(kg, text)
    print(json.dumps(stats, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qa_kg")
    p.add_argument("--db", default=None, help="override QA_KG_DB path")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build"); s.add_argument("--validate", action="store_true")
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--tier")
    s.add_argument("-k", type=int, default=10); s.set_defaults(fn=cmd_search)
    s = sub.add_parser("search-ranked", help="Phase 4 authority-tiered ranker")
    s.add_argument("query")
    s.add_argument("--min-authority", default="internal",
                   choices=("primary", "derived", "internal", "agent"))
    s.add_argument("--domain", default="", help="exact domain filter (empty = no filter)")
    s.add_argument("--valid-at", default="", help="ISO-8601 timestamp; default = now")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(fn=cmd_search_ranked)
    s = sub.add_parser("digest"); s.add_argument("--tier", default="cosmos")
    s.add_argument("-k", type=int, default=40); s.set_defaults(fn=cmd_digest)
    s = sub.add_parser("check"); s.add_argument("--tier"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("ingest-ob"); s.add_argument("--file", type=Path, default=None,
        help="Path to markdown file from mcp__open-brain__recent_thoughts; defaults to stdin")
    s.set_defaults(fn=cmd_ingest_ob)
    s = sub.add_parser("arag-search"); s.add_argument("query")
    s.add_argument("-k", type=int, default=10); s.add_argument("--source")
    s.set_defaults(fn=cmd_arag_search)
    s = sub.add_parser("arag-promote"); s.add_argument("msg_id")
    s.add_argument("--reason", default=""); s.set_defaults(fn=cmd_arag_promote)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
