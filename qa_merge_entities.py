#!/usr/bin/env python3
"""
qa_merge_entities.py

Entry: python qa_merge_entities.py --inputs qa_entities.json qa_entities_repo.json --output qa_entities_merged.json

Merges multiple entity catalogs (entities + relationships) by name, deduping entities
and unioning relationships.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge(inputs: List[Path]) -> dict:
    ent_by_name: Dict[str, dict] = {}
    rel_set = set()
    for p in inputs:
        data = load(p)
        for e in data.get("entities", []):
            name = e.get("name").strip()
            if name in ent_by_name:
                # Merge symbols and prefer existing definition unless new has longer one
                existing = ent_by_name[name]
                syms = list(dict.fromkeys((existing.get("symbols", []) or []) + (e.get("symbols", []) or [])))
                existing["symbols"] = syms
                if len(e.get("definition", "")) > len(existing.get("definition", "")):
                    existing["definition"] = e.get("definition", "")
            else:
                ent_by_name[name] = e
        for r in data.get("relationships", []):
            key = (r.get("source"), r.get("target"), r.get("relationship", "RELATED_TO"))
            rel_set.add(key)
    rels = [
        {"source": s, "target": t, "relationship": rel}
        for (s, t, rel) in sorted(rel_set)
    ]
    ents = list(ent_by_name.values())
    return {
        "source": ", ".join(str(p) for p in inputs),
        "counts": {"entities": len(ents), "relationships": len(rels)},
        "entities": ents,
        "relationships": rels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple QA entity catalogs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--output", default="qa_entities_merged.json", help="Output JSON path")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    merged = merge(inputs)
    out = Path(args.output)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Merged {len(inputs)} catalogs → {out} | entities={merged['counts']['entities']}, relationships={merged['counts']['relationships']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

