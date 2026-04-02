#!/usr/bin/env python3
"""
qa_entity_extractor.py — Extract canonical QA entities from the research lexicon

Entry point:
  python qa_entity_extractor.py --in private/QAnotes/research_log_lexicon.md \
                                --out artifacts/knowledge/qa_entities.json

Notes
- Parses the canonical lexicon Markdown and extracts entities with metadata.
- Focuses on sections:
  1) Official Terms (bolded term headings)
  2) Mathematical Primitives (term: definition form)
- Outputs a deterministic JSON list consumable by the encoder/graph scripts.

Usage
  - Minimal: python qa_entity_extractor.py
  - Custom paths: python qa_entity_extractor.py --in path/to/lexicon.md --out artifacts/knowledge/qa_entities.json
  - Prints a short summary upon completion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def slugify(name: str) -> str:
    s = name.strip()
    s = re.sub(r"[\s/]+", "-", s)
    s = re.sub(r"[^A-Za-z0-9\-\._]", "", s)
    return s.lower()


def maybe_symbol_from_name(name: str) -> str | None:
    # Capture short uppercase symbols in parentheses, e.g., "Harmonic Index (HI)"
    m = re.search(r"\(([^)]+)\)", name)
    if m:
        cand = m.group(1).strip()
        if 1 <= len(cand) <= 6 and cand.upper() == cand and re.fullmatch(r"[A-Z0-9]+", cand or ""):  # pragma: no cover
            return cand
    return None


def parse_lexicon(md_text: str) -> List[Dict]:
    """Parse the lexicon markdown into a list of entity dicts.

    Returns list of dicts with keys: name, slug, definition, section, symbol (optional).
    """
    lines = md_text.splitlines()
    entities: List[Dict] = []
    section = None

    def add_entity(name: str, definition: str, section_name: str) -> None:
        name = name.strip().strip("- ")
        definition = definition.strip()
        if not name:
            return
        symbol = maybe_symbol_from_name(name)
        entities.append({
            "name": name,
            "slug": slugify(name),
            "definition": definition,
            "section": section_name,
            **({"symbol": symbol} if symbol else {}),
        })

    # State machine over sections 1 and 2
    section_re = re.compile(r"^##\s+(\d+)\)")
    bullet_bold_re = re.compile(r"^-\s+\*\*(.+?)\*\*:?\s*(.*)")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        msec = section_re.match(line)
        if msec:
            section = int(msec.group(1))
            continue

        # Section 1: Official Terms — pattern: - **Term:** definition
        if section == 1:
            mb = bullet_bold_re.match(line)
            if mb:
                name = mb.group(1)
                definition = mb.group(2)
                add_entity(name, definition, "Official Terms")
                continue

        # Section 2: Mathematical Primitives — pattern: - Term: definition (not bold)
        if section == 2 and line.startswith("-"):
            # Remove leading dash and whitespace
            content = line[1:].strip()
            # Split on first colon
            if ":" in content:
                name_part, definition = content.split(":", 1)
                name = name_part.strip(" `")
                add_entity(name, definition, "Mathematical Primitives")

    return entities


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract QA entities from the canonical lexicon")
    parser.add_argument("--in", dest="in_path", default="private/QAnotes/research_log_lexicon.md",
                        help="Input lexicon Markdown path")
    parser.add_argument("--out", dest="out_path", default="artifacts/knowledge/qa_entities.json",
                        help="Output JSON path for extracted entities")
    args = parser.parse_args(argv)

    if not os.path.exists(args.in_path):
        print(f"[ERROR] Input lexicon not found: {args.in_path}", file=sys.stderr)
        return 2

    with open(args.in_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    entities = parse_lexicon(md_text)

    ensure_dir(args.out_path)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": os.path.abspath(args.in_path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(entities),
            "entities": entities,
        }, f, indent=2, ensure_ascii=False)

    print(f"[qa_entity_extractor] Extracted {len(entities)} entities → {args.out_path}")
    return 0


if __name__ == "__main__":  # --- Main ---
    raise SystemExit(main())

