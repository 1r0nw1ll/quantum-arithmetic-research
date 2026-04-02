#!/usr/bin/env python3
"""
qa_build_pipeline.py

End-to-end builder for QA-GraphRAG.

Runs:
  1) Extract lexicon entities
  2) Ingest repo Bell-test docs
  3) Merge catalogs
  4) Ingest vault chunks (co-occurrence edges)
  5) Encode entities
  6) Build graph (GraphML), optionally with full E8 roots

Usage:
  python qa_build_pipeline.py --chunk-limit 500 --full-e8
  python qa_build_pipeline.py --chunk-limit 0 --full-e8  # 0 = all chunks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent.resolve()


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
    return res.returncode


def main() -> int:
    p = argparse.ArgumentParser(description="Build the QA-GraphRAG knowledge graph")
    p.add_argument("--lexicon", default="private/QAnotes/research_log_lexicon.md")
    p.add_argument("--chunk-dir", default="vault_audit_cache/chunks")
    p.add_argument("--chunk-limit", type=int, default=500, help="Number of recent chunks to scan (0=all)")
    p.add_argument("--full-e8", action="store_true", help="Use full E8 roots for alignments")
    p.add_argument("--discover-limit", type=int, default=0, help="Discovery scan chunk limit (0=all)")
    p.add_argument("--discover-min-count", type=int, default=5, help="Discovery min frequency threshold")
    p.add_argument("--discover-max", type=int, default=1000, help="Discovery max entities cap")
    p.add_argument("--typed-min-count", type=int, default=2, help="Typed relation min count")
    p.add_argument("--cooccur-min-count", type=int, default=5, help="Min count for CO_OCCURS edges")
    p.add_argument("--out-entities", default="qa_entities_merged.json")
    p.add_argument("--seeds", default="qa_entities_seed.json", help="Optional seed entities JSON to include if exists")
    p.add_argument("--out-graph", default="qa_knowledge_graph.graphml")
    args = p.parse_args()

    # 1) Lexicon
    if run([sys.executable, "qa_entity_extractor.py", "--input", args.lexicon, "--output", "qa_entities.json"]) != 0:
        return 1

    # 2) Repo ingest
    if run([sys.executable, "qa_repo_ingest.py", "--output", "qa_entities_repo.json"]) != 0:
        return 1

    # 2.5) Seeds (optional)
    seeds_path = ROOT / args.seeds
    if not seeds_path.exists():
        # Create default seeds if missing
        run([sys.executable, "qa_seed_entities.py", "--output", str(seeds_path)])

    # 3) Discovery from chunks (promote new entities)
    discover_out = ROOT / "qa_entities_discovered.json"
    if run([sys.executable, "qa_entity_discovery.py", "--chunks", args.chunk_dir, "--output", str(discover_out),
            "--limit", str(args.discover_limit), "--min-count", str(args.discover_min_count), "--max-entities", str(args.discover_max)]) != 0:
        return 1

    # 4) Merge (lexicon + repo + seeds + discovered)
    if run([sys.executable, "qa_merge_entities.py", "--inputs", "qa_entities.json", "qa_entities_repo.json", str(seeds_path), str(discover_out), "--output", args.out_entities]) != 0:
        return 1

    # 5) Chunk ingest (co-occurrence)
    limit_str = str(args.chunk_limit)
    if run([sys.executable, "qa_chunk_ingest.py", "--chunks", args.chunk_dir, "--entities", args.out_entities, "--output", "qa_chunk_edges.json", "--limit", limit_str]) != 0:
        return 1

    # 6) Typed relations mining
    typed_out = ROOT / "qa_typed_edges.json"
    if run([sys.executable, "qa_relation_miner.py", "--chunks", args.chunk_dir, "--entities", args.out_entities,
            "--output", str(typed_out), "--limit", str(args.discover_limit), "--min-count", str(args.typed_min_count)]) != 0:
        return 1

    # 6b) Lexicon DEFINES edges
    defines_out = ROOT / "qa_lexicon_defines_edges.json"
    if run([sys.executable, "qa_lexicon_defines.py", "--lexicon", "qa_entities.json", "--universe", args.out_entities, "--output", str(defines_out)]) != 0:
        return 1

    # 6c) Combine typed edges
    combine_out = ROOT / "qa_typed_edges_combined.json"
    try:
        import json as _json
        t = _json.loads(typed_out.read_text(encoding='utf-8'))
        d = _json.loads(defines_out.read_text(encoding='utf-8'))
        edges = (t.get('edges', []) or []) + (d.get('edges', []) or [])
        # Deduplicate by (src, rel, tgt)
        seen = set()
        merged = []
        for e in edges:
            key = (e.get('source'), e.get('relationship'), e.get('target'))
            if key in seen:
                continue
            seen.add(key)
            merged.append(e)
        combine_out.write_text(_json.dumps({"edges": merged}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"✓ Combined typed edges: {len(merged)} → {combine_out}")
    except Exception as _e:
        print(f"Failed to combine typed edges: {_e}")
        return 1

    # 7) Encode
    if run([sys.executable, "qa_entity_encoder.py", "--entities", args.out_entities, "--output", "qa_entity_encodings.json"]) != 0:
        return 1

    # 8) Build graph
    cmd = [sys.executable, "qa_knowledge_graph.py", "--entities", args.out_entities, "--encodings", "qa_entity_encodings.json", "--chunk-edges", "qa_chunk_edges.json", "--typed-edges", str(combine_out), "--cooccur-min-count", str(args.cooccur_min_count), "--output", args.out_graph]
    if args.full_e8:
        cmd.append("--full-e8")
    if run(cmd) != 0:
        return 1

    print(f"\n✓ Build complete:\n  Entities: {args.out_entities}\n  Graph:    {args.out_graph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
