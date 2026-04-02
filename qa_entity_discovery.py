#!/usr/bin/env python3
"""
qa_entity_discovery.py

Rule-based/heuristic entity discovery from vault chunks.

Discovers candidate entities via:
- Title-Case multiword sequences
- Acronyms (ALLCAPS >1 letters)
- Lowercase noun-phrase endings like "... network|experiment|dataset|benchmark"

Outputs: qa_entities_discovered.json (entities only; relationships left empty).

Usage:
  python qa_entity_discovery.py --chunks vault_audit_cache/chunks \
    --output qa_entities_discovered.json --limit 0 --min-count 5 --max-entities 1000
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CHUNKS_DIR_DEFAULT = Path("vault_audit_cache/chunks")
OUTPUT_DEFAULT = Path("qa_entities_discovered.json")


def _load_manifest_map(manifest_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not manifest_path.exists():
        return mapping
    for line in manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        cp = rec.get("chunk_path")
        sp = rec.get("source_path")
        if cp and sp:
            mapping[cp] = sp
    return mapping


def list_chunk_files(chunks_dir: Path, limit: Optional[int]) -> List[Path]:
    files = [p for p in chunks_dir.iterdir() if p.is_file() and p.suffix == ".txt"]
    # Filter: prefer Markdown notes
    manifest_map = _load_manifest_map(chunks_dir.parent / "manifest.jsonl")
    filtered = []
    for p in files:
        sp = manifest_map.get(str(p))
        if sp is None:
            filtered.append(p)
            continue
        sp_low = sp.lower()
        if sp_low.endswith(".md") and "/.obsidian/" not in sp_low:
            filtered.append(p)
    filtered.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    if limit is None:
        return filtered
    return filtered[:limit]


def discover_entities_in_text(text: str) -> List[str]:
    # Normalize whitespace
    t = re.sub(r"\s+", " ", text)
    candidates: List[str] = []

    # 1) Title-Case multiword up to 5 tokens (allow apostrophes/hyphens)
    pat_title = re.compile(r"\b([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,4})\b")
    candidates += [m.group(1).strip() for m in pat_title.finditer(t)]

    # 2) Acronyms of length >= 2
    pat_acronym = re.compile(r"\b([A-Z]{2,}(?:\-[A-Z]{2,})?)\b")
    candidates += [m.group(1).strip() for m in pat_acronym.finditer(t)]

    # 3) Lowercase noun phrases ending with key heads (up to 5 tokens)
    heads = r"experiment|network|dataset|benchmark|graph|study|analysis|protocol|pipeline"
    pat_lower = re.compile(rf"\b([a-z][a-z]+(?:\s+[a-z][a-z]+){{0,4}}\s+(?:{heads}))\b")
    candidates += [m.group(1).strip() for m in pat_lower.finditer(t)]

    return candidates


def canonicalize(name: str) -> str:
    # Collapse internal spaces, trim
    name = re.sub(r"\s+", " ", name).strip()
    return name


STOP = {"the", "a", "an", "and", "or", "of", "in", "to", "on", "for", "by", "with", "is", "are"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover entities from vault chunks")
    ap.add_argument("--chunks", default=str(CHUNKS_DIR_DEFAULT))
    ap.add_argument("--output", default=str(OUTPUT_DEFAULT))
    ap.add_argument("--limit", type=int, default=0, help="Number of recent chunks to scan (0=all)")
    ap.add_argument("--min-count", type=int, default=5, help="Minimum occurrences to keep an entity")
    ap.add_argument("--max-entities", type=int, default=1000, help="Cap on discovered entities")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks)
    limit = None if args.limit == 0 else args.limit
    files = list_chunk_files(chunks_dir, limit)

    counts: Counter[str] = Counter()
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for cand in discover_entities_in_text(text):
            c = canonicalize(cand)
            # Basic filters
            if len(c) < 3:
                continue
            if c.lower() in STOP:
                continue
            counts[c] += 1

    # Rank by frequency
    ranked = counts.most_common()
    # Filter by threshold and cap
    filtered = [(n, c) for (n, c) in ranked if c >= args.min_count]
    if len(filtered) > args.max_entities:
        filtered = filtered[: args.max_entities]

    entities = []
    for name, c in filtered:
        entities.append({
            "name": name,
            "definition": "",
            "symbols": [],
            "source_section": "Discovered",
            "type": "discovered",
            "frequency": int(c),
        })

    payload = {"source": "discovered", "counts": {"entities": len(entities)}, "entities": entities, "relationships": []}
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Discovered {len(entities)} entities (min_count={args.min_count}, max={args.max_entities}) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

