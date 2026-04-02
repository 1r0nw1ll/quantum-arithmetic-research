#!/usr/bin/env python3
"""
qa_generate_seed_candidates.py

Generate an expanded seed-candidate list from discovered entities with
stronger proper-noun heuristics, including discovery frequency.

Usage:
  python qa_generate_seed_candidates.py \
    --discovered qa_entities_discovered.json \
    --merged qa_entities_merged.json \
    --output qa_seed_candidates.json \
    --top-k 500 --include-default-seeds
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


DEFAULT_SEEDS = [
    {
        "name": "Norman Wildberger",
        "aliases": [],
        "type": "person",
        "notes": "Creator of Rational Trigonometry, mentioned as a contemporary analogue to QA.",
    },
    {
        "name": "E.T. Whittaker",
        "aliases": ["Whittaker"],
        "type": "person",
        "notes": "Historical mathematician relevant to the project's context.",
    },
    {
        "name": "Ben Iverson",
        "aliases": [],
        "type": "person",
        "notes": "Primary developer of Quantum Arithmetic.",
    },
    {
        "name": "Dale Pond",
        "aliases": [],
        "type": "person",
        "notes": "Expanded on Ben Iverson's work on Quantum Arithmetic.",
    },
    {
        "name": "A. Garrett Lisi",
        "aliases": ["Lisi"],
        "type": "person",
        "notes": "Work on E8 embedding referenced in theoretical review.",
    },
    {
        "name": "Rational Trigonometry",
        "aliases": ["RT"],
        "type": "concept",
        "notes": "Contemporary analogue to QA's principles.",
    },
    {
        "name": "Modular Forms",
        "aliases": [],
        "type": "concept",
        "notes": "Analogue to QA's modular symmetry.",
    },
    {
        "name": "Louvain Method",
        "aliases": ["Louvain"],
        "type": "concept",
        "notes": "Community detection algorithm for QA graphs.",
    },
]


STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "to",
        "on",
        "for",
        "by",
        "with",
        "is",
        "are",
        "this",
        "that",
        "these",
        "those",
        "you",
        "your",
        "all",
        "not",
        "but",
        "can",
        "data",
        "results",
        "return",
        "returns",
        "total",
        "info",
        "why",
        "path",
        "phase",
        "download",
        "expand",
        "while",
        "acc",
        "mod",
        "api",
        "adam",
        "x_train",
        "step",
        "steps",
        "analysis",
        "how",
        "assistant",
        "abstract",
        "implement",
        "implementing",
        "applications",
        "application",
    ]
)


def type_guess(name: str) -> str:
    # Dataset/benchmark detection
    if re.search(r"\b(dataset|benchmark|network)\b", name, re.I):
        return "dataset"
    # Concepts by head nouns
    if re.search(
        r"\b(method|algorithm|correlator|operator|system|bound|kernel|theorem|principle|model|loss)\b",
        name,
        re.I,
    ):
        return "concept"
    # Likely person: two or more tokens, capitalized or initials
    tokens = name.split()
    if len(tokens) >= 2 and all(
        re.match(r"^[A-Z][\w\.'-]*$", t) or re.match(r"^[A-Z]\.$", t) for t in tokens
    ):
        return "person"
    # Single capitalized token that isn't a known stopword
    if re.match(r"^[A-Z][\w'’.-]*$", name) and name.lower() not in STOP:
        return "concept"
    return "concept"


def good_candidate(name: str) -> bool:
    s = name.strip()
    if len(s) < 3:
        return False
    if s.lower() in STOP:
        return False
    # Prefer names with uppercase letters or multiple tokens (proper nouns)
    if any(c.isupper() for c in s) or len(s.split()) >= 2:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate expanded seed candidates with frequencies")
    ap.add_argument("--discovered", default="qa_entities_discovered.json")
    ap.add_argument("--merged", default="qa_entities_merged.json")
    ap.add_argument("--output", default="qa_seed_candidates.json")
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--include-default-seeds", action="store_true")
    args = ap.parse_args()

    disc = json.loads(Path(args.discovered).read_text(encoding="utf-8"))
    merged = json.loads(Path(args.merged).read_text(encoding="utf-8"))
    merged_names = {e["name"] for e in merged.get("entities", [])}

    items = disc.get("entities", [])
    # Rank and filter
    candidates = []
    for e in items:
        name = (e.get("name") or "").strip()
        freq = int(e.get("frequency", 0))
        if not name or name in merged_names:
            continue
        if not good_candidate(name):
            continue
        candidates.append((name, freq))

    # Deduplicate case-insensitively and sort by frequency desc
    seen_lower = set()
    uniq = []
    for name, freq in sorted(candidates, key=lambda x: x[1], reverse=True):
        key = name.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        uniq.append((name, freq))

    if len(uniq) > args.top_k:
        uniq = uniq[: args.top_k]

    payload: List[Dict] = []

    if args.include_default_seeds:
        for s in DEFAULT_SEEDS:
            entry = dict(s)
            entry["frequency"] = None
            payload.append(entry)

    for name, freq in uniq:
        payload.append(
            {
                "name": name,
                "aliases": [],
                "type": type_guess(name),
                "notes": f"Discovered (frequency={freq})",
                "frequency": freq,
            }
        )

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(payload)} seed candidates → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

