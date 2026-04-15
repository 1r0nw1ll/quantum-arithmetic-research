"""QA-KG command-line entry.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Usage:
    python -m tools.qa_kg.cli build            # populate axioms + memory rules + certs
    python -m tools.qa_kg.cli stats
    python -m tools.qa_kg.cli search "<query>" [--tier cosmos] [-k 10]
    python -m tools.qa_kg.cli digest [--tier cosmos] [-k 40]
    python -m tools.qa_kg.cli check [--tier singularity]
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import argparse
import json
import sys

from tools.qa_kg.kg import connect
from tools.qa_kg.predicate import check_all
from tools.qa_kg.extractors import axioms as x_axioms
from tools.qa_kg.extractors import memory_rules as x_rules
from tools.qa_kg.extractors import certs as x_certs
from tools.qa_kg.extractors import edges as x_edges


def _fmt(row) -> dict:
    return {k: row[k] for k in row.keys()}


def cmd_build(args) -> int:
    kg = connect(args.db)
    a = x_axioms.populate(kg)
    r = x_rules.populate(kg)
    c = x_certs.populate(kg, run_validator=args.validate)
    e = x_edges.populate(kg)
    print(f"axioms: {len(a)}  rules: {len(r)}  certs: {len(c)}  edges: {e}")
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


def cmd_digest(args) -> int:
    kg = connect(args.db)
    rows = kg.digest(tier=args.tier, limit=args.k)
    for r in rows:
        print(f"[{r['tier']}] ({r['coord_b']},{r['coord_e']}) {r['id']} — {r['title']}")
    return 0


def cmd_check(args) -> int:
    kg = connect(args.db)
    results = check_all(kg.conn, tier=args.tier)
    for nid, res in results.items():
        flag = "OK" if res.ok else "FAIL"
        print(f"{flag}  {nid}  {res.msg}")
    fails = [nid for nid, r in results.items() if not r.ok]
    return 0 if not fails else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qa_kg")
    p.add_argument("--db", default=None, help="override QA_KG_DB path")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build"); s.add_argument("--validate", action="store_true")
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--tier")
    s.add_argument("-k", type=int, default=10); s.set_defaults(fn=cmd_search)
    s = sub.add_parser("digest"); s.add_argument("--tier", default="cosmos")
    s.add_argument("-k", type=int, default=40); s.set_defaults(fn=cmd_digest)
    s = sub.add_parser("check"); s.add_argument("--tier"); s.set_defaults(fn=cmd_check)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
