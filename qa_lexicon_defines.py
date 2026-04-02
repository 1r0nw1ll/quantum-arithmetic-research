#!/usr/bin/env python3
"""
qa_lexicon_defines.py

Create DEFINES edges from lexicon entries by linking each lexicon entity to
any known entity names mentioned in its definition.

Inputs:
  --lexicon qa_entities.json (from qa_entity_extractor.py)
  --universe qa_entities_merged.json (names to match against)

Output:
  qa_lexicon_defines_edges.json with edges [{source, target, relationship: "DEFINES", count}]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_entities(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("entities", [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build DEFINES edges from lexicon definitions")
    ap.add_argument("--lexicon", default="qa_entities.json")
    ap.add_argument("--universe", default="qa_entities_merged.json")
    ap.add_argument("--output", default="qa_lexicon_defines_edges.json")
    args = ap.parse_args()

    lex_ents = load_entities(Path(args.lexicon))
    uni_ents = load_entities(Path(args.universe))
    uni_names = [e.get("name") for e in uni_ents if e.get("name")]

    edges = []
    for e in lex_ents:
        src = e.get("name")
        definition = (e.get("definition") or "").strip()
        if not src or not definition:
            continue
        dlow = definition.lower()
        for tgt in uni_names:
            if not tgt or tgt == src:
                continue
            if tgt.lower() in dlow:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "relationship": "DEFINES",
                    "count": 1,
                })

    payload = {"edges": edges, "source": str(args.lexicon)}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Lexicon DEFINES edges: {len(edges)} → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

