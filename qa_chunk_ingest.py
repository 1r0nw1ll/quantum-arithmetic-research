#!/usr/bin/env python3
"""
qa_chunk_ingest.py

Entry point: python qa_chunk_ingest.py

Scans the 100 most recent files under vault_audit_cache/chunks/ by mtime,
detects co-occurrences of known entity names from qa_entities.json, and
emits additional edges (CO_OCCURS) between entities seen in the same chunk.

Output: qa_chunk_edges.json with a list of edges and counts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional


CHUNKS_DIR_DEFAULT = Path("vault_audit_cache/chunks")
ENTITIES_JSON_DEFAULT = Path("qa_entities.json")
OUTPUT_DEFAULT = Path("qa_chunk_edges.json")


def load_entity_names(entities_json: Path) -> List[str]:
    payload = json.loads(entities_json.read_text(encoding="utf-8"))
    return [e["name"] for e in payload.get("entities", [])]


def _load_manifest_map(manifest_path: Path) -> Dict[str, str]:
    """Map chunk_path -> source_path from JSONL manifest, if available."""
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


def select_recent_chunks(chunks_dir: Path, limit: Optional[int] = 100) -> List[Path]:
    files = [p for p in chunks_dir.iterdir() if p.is_file() and p.suffix == ".txt"]
    # Filter using manifest to prefer Markdown notes and ignore .obsidian configs
    manifest_map = _load_manifest_map(chunks_dir.parent / "manifest.jsonl")
    filtered = []
    for p in files:
        cp = str(p)
        sp = manifest_map.get(cp)
        if sp is None:
            # If unknown, keep it for now
            filtered.append(p)
            continue
        sp_low = sp.lower()
        if sp_low.endswith(".md") and "/.obsidian/" not in sp_low:
            filtered.append(p)
    # Sort by mtime desc
    filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if limit is None:
        return filtered
    return filtered[:limit]


def find_mentions(text: str, names: List[str]) -> List[str]:
    t = text.lower()
    found = []
    # Alias support similar to extractor
    ALIASES = {
        "Harmonic Index": ["HI", "Harmonic Index (HI)"],
        "State Deviation": ["Deviation"],
        "E₈ alignment statistic": ["E8 alignment", "E₈ alignment", "alignment"],
        "Baseline Fingerprint": ["Baseline"],
        "QA‑CPLearn engine": ["QA‑CPLearn"],
    }
    alias_to_name = {al.lower(): canon for canon, als in ALIASES.items() for al in als}
    for n in names:
        if n.lower() in t:
            found.append(n)
        else:
            # alias match
            for al, canon in alias_to_name.items():
                if canon == n and al in t:
                    found.append(n)
                    break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest recent vault chunks for co-occurrence edges")
    parser.add_argument("--chunks", default=str(CHUNKS_DIR_DEFAULT), help="Path to chunks directory")
    parser.add_argument("--entities", default=str(ENTITIES_JSON_DEFAULT), help="Path to qa_entities.json")
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT), help="Path to write qa_chunk_edges.json")
    parser.add_argument("--limit", type=int, default=500, help="Number of recent chunks to scan (0=all)")
    args = parser.parse_args()

    names = load_entity_names(Path(args.entities))
    limit = None if int(args.limit) == 0 else int(args.limit)
    chunk_files = select_recent_chunks(Path(args.chunks), limit=limit)

    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    chunk_hit = 0
    for fp in chunk_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        mentions = find_mentions(text, names)
        mentions = sorted(set(mentions))
        if len(mentions) < 2:
            continue
        chunk_hit += 1
        for a, b in combinations(mentions, 2):
            pair_counts[(a, b)] += 1
            pair_counts[(b, a)] += 1  # store both directions for DiGraph

    edges = []
    for (src, tgt), count in pair_counts.items():
        edges.append({
            "source": src,
            "target": tgt,
            "relationship": "CO_OCCURS",
            "count": int(count),
        })

    payload = {
        "scanned_files": len(chunk_files),
        "cooccurrence_chunks": chunk_hit,
        "edges": edges,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Scanned {len(chunk_files)} chunks; {chunk_hit} with ≥2 mentions; {len(edges)} edges → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
