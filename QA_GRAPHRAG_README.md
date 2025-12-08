# QA-GraphRAG Phase 1 Prototype

This prototype builds a small knowledge graph from the canonical QA lexicon using Quantum Arithmetic (QA) tuples instead of neural embeddings.

## Files
- `qa_entity_extractor.py` — parse `private/QAnotes/research_log_lexicon.md` into `qa_entities.json`.
- `qa_entity_encoder.py` — encode entities to QA tuples; writes `qa_entity_encodings.json`.
- `qa_knowledge_graph.py` — build a NetworkX DiGraph and export `qa_knowledge_graph.graphml`.
- `qa_graph_query.py` — CLI to query the graph via QA tuple similarity.
- `qa_graph_viz.py` — render the graph to `qa_knowledge_graph.png`.
- `qa_graphrag_utils.py` — shared E₈ and QA helpers.
 - `qa_chunk_ingest.py` — ingest 100 most recent vault chunks and add co-occurrence edges.

## Quickstart
```bash
# 1) Extract entities from the lexicon
python qa_entity_extractor.py --input private/QAnotes/research_log_lexicon.md --output qa_entities.json

# 2) Encode entities (manual overrides + deterministic hash)
python qa_entity_encoder.py --entities qa_entities.json --output qa_entity_encodings.json

# 3) (Optional) Enrich edges from recent vault chunks
python qa_chunk_ingest.py --chunks vault_audit_cache/chunks --entities qa_entities.json --output qa_chunk_edges.json --limit 100

# 4) Build knowledge graph (GraphML)
python qa_knowledge_graph.py --entities qa_entities.json --encodings qa_entity_encodings.json --chunk-edges qa_chunk_edges.json --output qa_knowledge_graph.graphml

# 5) Run sample queries (try hybrid or personalized PageRank)
python qa_graph_query.py "What is Harmonic Index?" --top-k 5 --method hybrid
python qa_graph_query.py "What is Harmonic Index?" --top-k 5 --method ppr

# 6) Visualize
python qa_graph_viz.py --graph qa_knowledge_graph.graphml --output qa_knowledge_graph.png
```

## One‑Shot Build Pipeline
```bash
# Build full pipeline with recent 500 chunks and full E8 alignment
python qa_build_pipeline.py --chunk-limit 500 --full-e8

# For full vault ingestion (all chunks), set limit to 0
python qa_build_pipeline.py --chunk-limit 0 --full-e8
```

## Encoding Strategy
- Default is deterministic hash-based mapping: name → `(b,e)` mod 24, with `d=(b+e) mod 24` and `a=(b+2e) mod 24`.
- Manual overrides exist for key concepts (e.g., “Harmonic Index”, “E₈ alignment statistic”, “QA tuple”, “Coherence collapse”).
- E₈ alignment is computed against an ideal root (`[1,1,2,3,0,0,0,0]`) for speed; a full 240-root mode is available in utils.

## Notes
- Relationships are inferred via rule-based patterns from definitions (e.g., `USES`, `COMPUTES`, `MEASURES`, `DETECTS`) with a fallback `MENTIONS`.
- Optional co-occurrence edges (`CO_OCCURS`) are derived from the 100 most recent vault chunks.
 - Optional co-occurrence edges (`CO_OCCURS`) are derived from the 500 most recent vault chunks by default; set limit to 0 to scan all.
- GraphML stores tuples as strings (`b,e,d,a`) plus scalar `b,e,d,a` node attributes for compatibility.
- This prototype starts from the lexicon only; scaling to `vault_audit_cache/` is Phase 2.

## Dependencies
- Python 3.10+
- `numpy`, `networkx`, `matplotlib` (for visualization)

## Limitations / Next Steps
- Extend entity list beyond the lexicon (e.g., CHSH, I3322) when broader ingestion is enabled.
- Consider learned weights for tuple vs traversal components; add RWR/PageRank-style personalization.
- Add per-edge relation-type weights (e.g., USES > MENTIONS > CO_OCCURS) and tune.
