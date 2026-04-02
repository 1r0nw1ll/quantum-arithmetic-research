#!/usr/bin/env python3
"""
qa_relation_miner.py

Mine typed relationships from vault chunks using simple patterns.

Patterns supported (directional):
  X uses Y              -> USES
  X computes Y          -> COMPUTES
  X measures Y          -> MEASURES
  X produces Y          -> PRODUCES
  X predicts Y          -> PREDICTS
  X validates Y         -> VALIDATES
  X refutes Y           -> REFUTES
  X defines Y           -> DEFINES
  X is a/an/the Y       -> IS_A
  X are Y               -> IS_A

Heuristic: in a sentence containing a keyword and ≥2 entity mentions, take the
nearest entity to the left of the keyword as source, and nearest to the right
as target. Aggregate counts across the corpus.

Output: JSON with list of typed edges with counts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_entity_names(entities_json: Path) -> List[str]:
    payload = json.loads(entities_json.read_text(encoding="utf-8"))
    entities = payload.get("entities", []) if isinstance(payload, dict) else []
    return [e.get("name") for e in entities if e.get("name")]


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
    manifest = _load_manifest_map(chunks_dir.parent / "manifest.jsonl")
    filtered: List[Path] = []
    for p in files:
        sp = manifest.get(str(p))
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


KEYWORDS = [
    ("uses", "USES"),
    ("computes", "COMPUTES"),
    ("computed as", "COMPUTES"),
    ("measures", "MEASURES"),
    ("produces", "PRODUCES"),
    ("predicts", "PREDICTS"),
    ("validates", "VALIDATES"),
    ("refutes", "REFUTES"),
    ("defines", "DEFINES"),
]

IS_PATTERNS = [
    re.compile(r"\bis\s+(?:a|an|the)\s+", re.I),
    re.compile(r"\bare\s+", re.I),
]


def mine_sentence(sentence: str, entities: List[str]) -> List[Tuple[str, str, str]]:
    s = sentence.strip()
    if not s:
        return []
    s_low = s.lower()
    # Find mentions with positions
    mentions: List[Tuple[int, int, str]] = []  # (start, end, name)
    for name in entities:
        name_low = name.lower()
        idx = 0
        while True:
            idx = s_low.find(name_low, idx)
            if idx == -1:
                break
            mentions.append((idx, idx + len(name_low), name))
            idx += len(name_low)
    if len(mentions) < 2:
        return []
    mentions.sort(key=lambda x: x[0])
    results: List[Tuple[str, str, str]] = []
    # 1) Keyword-driven relations
    for kw, rel in KEYWORDS:
        pos = s_low.find(kw)
        if pos == -1:
            continue
        left = [m for m in mentions if m[1] <= pos]
        right = [m for m in mentions if m[0] >= pos]
        if not left or not right:
            continue
        src = left[-1][2]
        tgt = right[0][2]
        if src != tgt:
            results.append((src, rel, tgt))
    # 2) IS_A patterns
    for pat in IS_PATTERNS:
        m = pat.search(s_low)
        if not m:
            continue
        pos = m.start()
        left = [m for m in mentions if m[1] <= pos]
        right = [m for m in mentions if m[0] >= pos]
        if not left or not right:
            continue
        src = left[-1][2]
        tgt = right[0][2]
        if src != tgt:
            results.append((src, "IS_A", tgt))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine typed relations from chunks")
    ap.add_argument("--chunks", default="vault_audit_cache/chunks")
    ap.add_argument("--entities", default="qa_entities_merged.json")
    ap.add_argument("--output", default="qa_typed_edges.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    entities = load_entity_names(Path(args.entities))
    files = list_chunk_files(Path(args.chunks), None if args.limit == 0 else args.limit)

    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # Naive sentence split
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        for sent in sentences:
            for (src, rel, tgt) in mine_sentence(sent, entities):
                counts[(src, rel, tgt)] += 1

    edges = []
    kept = 0
    for (src, rel, tgt), c in counts.items():
        if c >= args.min_count:
            edges.append({"source": src, "target": tgt, "relationship": rel, "count": int(c)})
            kept += 1

    payload = {"edges": edges, "min_count": args.min_count, "scanned_files": len(files)}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Typed edges: kept {kept} (min_count={args.min_count}) from {len(files)} files → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

