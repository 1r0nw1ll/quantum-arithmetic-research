# Handoff to Codex - QA-GraphRAG Phase 1 Implementation
**Date**: 2025-11-14
**From**: Claude Code
**Priority**: HIGH
**Estimated Time**: 8-10 hours

---

## Mission

Implement a working prototype of QA-based GraphRAG (Graph Retrieval-Augmented Generation) that uses Quantum Arithmetic tuples instead of traditional vector embeddings for knowledge graph construction and retrieval.

---

## Context

User has 1,152 Obsidian vault files already processed into:
- 150,061 text chunks (vault_audit_cache/chunks/)
- AI-generated summaries (vault_audit_cache/summaries/)
- Canonical lexicon with 48 official terms (research_log_lexicon.md)

**Your job**: Turn this into a queryable graph database using QA tuples.

---

## Phase 1 Prototype Specification

### Input Data
**Primary Source**: `/home/player2/signal_experiments/private/QAnotes/research_log_lexicon.md`
- Contains 48 canonical QA research terms
- Definitions and relationships
- Official symbols and notation

**Secondary Source**: `/home/player2/signal_experiments/vault_audit_cache/`
- Start with 100 most recent chunks
- Expand to full 150k in Phase 2

### Required Outputs

1. **Entity Extraction Script** (`qa_entity_extractor.py`)
   - Parse research_log_lexicon.md
   - Extract all 48 canonical terms
   - Extract definitions, symbols, relationships
   - Output: JSON file with entity catalog

2. **QA Encoding Module** (`qa_entity_encoder.py`)
   - Map each entity to QA tuple (b,e,d,a) mod 24
   - Three encoding strategies to implement:
     - **Deterministic**: Hash entity name → tuple
     - **Rule-based**: Use entity properties (frequency, importance)
     - **Manual**: Predefined mapping for key concepts
   - Ensure tuples satisfy QA constraints: d=(b+e)%24, a=(b+2e)%24

3. **Graph Construction** (`qa_knowledge_graph.py`)
   - Use NetworkX (already installed in venv)
   - Nodes: Entities with QA tuples
   - Edges: Relationships with transition tuples
   - Compute E8 alignment for edge weights
   - Export to GraphML format

4. **Query Interface** (`qa_graph_query.py`)
   - CLI tool for querying the graph
   - Convert query string → QA tuple
   - Implement simple graph traversal (BFS/DFS initially)
   - Rank results by Harmonic Index
   - Return top-k relevant entities with context

5. **Visualization** (`qa_graph_viz.py`)
   - NetworkX + matplotlib visualization
   - Color nodes by QA tuple properties
   - Edge thickness = E8 alignment strength
   - Save as PNG for documentation

### Test Queries to Support

```bash
python qa_graph_query.py "What is Harmonic Index?"
# Expected: Return definition, related concepts (E8 alignment, loss function)

python qa_graph_query.py "Find all Bell test experiments"
# Expected: Return CHSH, I₃₃₂₂, platonic solid tests

python qa_graph_query.py "How does QA relate to seizure detection?"
# Expected: Return EEG validation, brain→QA mapper, Harmonic Index applications
```

---

## Technical Specifications

### QA Tuple Encoding Strategies

**Option 1: Hash-based (Deterministic)**
```python
def hash_to_qa_tuple(entity_name: str, modulus: int = 24) -> tuple:
    """Map entity name to QA tuple via hashing"""
    h = hashlib.sha256(entity_name.encode()).digest()
    b = int.from_bytes(h[0:4], 'big') % modulus
    e = int.from_bytes(h[4:8], 'big') % modulus
    d = (b + e) % modulus
    a = (b + 2*e) % modulus
    return (b, e, d, a)
```

**Option 2: Rule-based (Data-driven)**
```python
def rule_based_encoding(entity: dict, modulus: int = 24) -> tuple:
    """Encode based on entity properties"""
    # b = term frequency in corpus (normalized to mod 24)
    # e = semantic importance score (manually assigned or computed)
    # d, a computed from constraints
    freq = entity.get('frequency', 0)
    importance = entity.get('importance', 0)

    b = int((freq / max_freq) * modulus) % modulus
    e = int((importance / 10.0) * modulus) % modulus
    d = (b + e) % modulus
    a = (b + 2*e) % modulus
    return (b, e, d, a)
```

**Option 3: Manual (Interpretable)**
```python
CANONICAL_MAPPINGS = {
    "Harmonic Index": (12, 8, 20, 4),    # High importance, central concept
    "E8 alignment": (15, 3, 18, 21),     # Core mathematical concept
    "QA tuple": (6, 6, 12, 18),          # Fundamental primitive
    "Coherence collapse": (3, 9, 12, 21), # Event detection
    # ... 44 more
}
```

**Recommendation**: Start with Option 1 (hash-based) for automation, provide Option 3 as override for key concepts.

### E8 Alignment Computation

```python
import numpy as np

# E8 roots (240 vectors in R^8)
E8_ROOTS = load_e8_roots()  # You can find this in existing code

def compute_e8_alignment(qa_tuple: tuple) -> float:
    """Compute E8 alignment for a QA tuple"""
    # Embed (b,e,d,a) into R^8
    # Use existing embedding scheme from qa_graph_builder_v2.py
    embedded = embed_qa_tuple_to_r8(qa_tuple)

    # Compute max cosine similarity to E8 roots
    similarities = [
        np.dot(embedded, root) / (np.linalg.norm(embedded) * np.linalg.norm(root))
        for root in E8_ROOTS
    ]
    return max(np.abs(similarities))

def compute_harmonic_index(qa_tuple: tuple, loss: float = 0.0) -> float:
    """Harmonic Index: HI = E8_alignment × exp(-k × loss)"""
    alignment = compute_e8_alignment(qa_tuple)
    k = 0.1  # Decay constant from research
    return alignment * np.exp(-k * loss)
```

### Graph Structure

```python
import networkx as nx

class QAKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, name: str, qa_tuple: tuple, metadata: dict):
        """Add entity node with QA encoding"""
        self.graph.add_node(
            name,
            qa_tuple=qa_tuple,
            e8_alignment=compute_e8_alignment(qa_tuple),
            **metadata
        )

    def add_relationship(self, source: str, target: str, rel_type: str):
        """Add edge with QA transition"""
        src_tuple = self.graph.nodes[source]['qa_tuple']
        tgt_tuple = self.graph.nodes[target]['qa_tuple']

        # Compute transition tuple (difference)
        transition = tuple(
            (tgt_tuple[i] - src_tuple[i]) % 24
            for i in range(4)
        )

        self.graph.add_edge(
            source, target,
            relationship=rel_type,
            transition=transition,
            strength=compute_e8_alignment(transition)
        )

    def query(self, query_str: str, top_k: int = 5) -> list:
        """Query graph and return top-k results"""
        # Convert query to QA tuple
        query_tuple = hash_to_qa_tuple(query_str)

        # Find closest nodes by tuple similarity
        similarities = {}
        for node in self.graph.nodes():
            node_tuple = self.graph.nodes[node]['qa_tuple']
            sim = compute_tuple_similarity(query_tuple, node_tuple)
            similarities[node] = sim

        # Return top-k
        top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [node for node, sim in top_nodes]
```

---

## Implementation Checklist

### Step 1: Environment Setup
- [ ] Activate venv: `source /home/player2/signal_experiments/.venv/bin/activate`
- [ ] Verify NetworkX installed: `python -c "import networkx; print(networkx.__version__)"`
- [ ] Install additional deps if needed: `pip install matplotlib scipy`

### Step 2: Entity Extraction
- [ ] Parse research_log_lexicon.md
- [ ] Extract 48 canonical terms with definitions
- [ ] Save to `qa_entities.json`
- [ ] Verify: Should have entries like `{"name": "Harmonic Index", "definition": "...", "symbol": "HI"}`

### Step 3: QA Encoding
- [ ] Implement hash_to_qa_tuple()
- [ ] Encode all 48 entities
- [ ] Verify constraints: d=(b+e)%24, a=(b+2e)%24
- [ ] Save to `qa_entity_encodings.json`

### Step 4: Graph Construction
- [ ] Create NetworkX directed graph
- [ ] Add 48 entity nodes with QA tuples
- [ ] Extract relationships from lexicon (USES, COMPUTES, MEASURES, etc.)
- [ ] Add edges with transition tuples
- [ ] Compute E8 alignment for all edges
- [ ] Export to GraphML: `graph.write_graphml("qa_knowledge_graph.graphml")`

### Step 5: Query Interface
- [ ] Implement CLI: `python qa_graph_query.py "query text"`
- [ ] Test query: "What is Harmonic Index?"
- [ ] Verify returns: definition, related concepts
- [ ] Test query: "Find Bell test experiments"
- [ ] Test query: "How does QA relate to seizure detection?"

### Step 6: Visualization
- [ ] Create graph visualization with NetworkX + matplotlib
- [ ] Color nodes by b value (base state)
- [ ] Edge thickness proportional to E8 alignment
- [ ] Save as `qa_knowledge_graph.png`

### Step 7: Documentation
- [ ] Write README.md explaining usage
- [ ] Document encoding strategy chosen
- [ ] Include example queries and outputs
- [ ] List limitations and future improvements

---

## Existing Code to Reference

### E8 Roots and Alignment
Look at these files for E8 implementation:
- `t003_e8_analysis.py` - E8 root system
- `qa_graph_builder_v2.py` - Graph construction with PyTorch Geometric
- `run_signal_experiments_final.py` - E8 alignment computation (lines ~80-100)

### QA Tuple Constraints
```python
# From CLAUDE.md and multiple experiment files
def is_valid_qa_tuple(b, e, d, a, modulus=24):
    """Verify QA tuple constraints"""
    return (d == (b + e) % modulus) and (a == (b + 2*e) % modulus)
```

---

## Expected Deliverables

1. **Code Files** (5 Python scripts)
   - `qa_entity_extractor.py`
   - `qa_entity_encoder.py`
   - `qa_knowledge_graph.py`
   - `qa_graph_query.py`
   - `qa_graph_viz.py`

2. **Data Files**
   - `qa_entities.json` (48 entities extracted)
   - `qa_entity_encodings.json` (QA tuples for each)
   - `qa_knowledge_graph.graphml` (NetworkX export)

3. **Outputs**
   - `qa_knowledge_graph.png` (visualization)
   - `example_queries.txt` (test results)

4. **Documentation**
   - `QA_GRAPHRAG_README.md` (usage guide)

---

## Testing & Validation

### Unit Tests
```python
def test_qa_encoding():
    """Verify QA tuple constraints"""
    entity = "Harmonic Index"
    b, e, d, a = hash_to_qa_tuple(entity)
    assert d == (b + e) % 24
    assert a == (b + 2*e) % 24

def test_e8_alignment():
    """Verify E8 alignment in [0,1]"""
    tuple = (12, 8, 20, 4)
    alignment = compute_e8_alignment(tuple)
    assert 0.0 <= alignment <= 1.0

def test_graph_construction():
    """Verify graph has 48 nodes"""
    graph = build_qa_knowledge_graph()
    assert len(graph.nodes()) == 48
    assert all('qa_tuple' in graph.nodes[n] for n in graph.nodes())
```

### Integration Tests
```bash
# Test full pipeline
python qa_entity_extractor.py
python qa_entity_encoder.py
python qa_knowledge_graph.py
python qa_graph_query.py "What is Harmonic Index?"
python qa_graph_viz.py
```

Expected output:
```
Query: "What is Harmonic Index?"

Top 5 Results:
1. Harmonic Index (HI) - QA tuple: (12,8,20,4), E8: 0.87
   Definition: Scalar order parameter from QA-Markovian dynamics

2. E8 alignment statistic - QA tuple: (15,3,18,21), E8: 0.92
   Relationship: USES → Harmonic Index

3. Harmonic loss - QA tuple: (8,6,14,20), E8: 0.78
   Relationship: COMPUTES → Harmonic Index

4. QA-Markovian system - QA tuple: (6,6,12,18), E8: 0.85
   Relationship: PRODUCES → Harmonic Index

5. Coherence collapse - QA tuple: (3,9,12,21), E8: 0.73
   Relationship: DETECTED_BY → Harmonic Index
```

---

## Success Criteria

- ✅ All 48 entities from lexicon extracted
- ✅ All QA tuples satisfy constraints (d=(b+e)%24, a=(b+2e)%24)
- ✅ Graph has 48 nodes with QA encodings
- ✅ E8 alignment computed for all edges
- ✅ Query "What is Harmonic Index?" returns definition + related concepts
- ✅ Query "Find Bell test experiments" returns CHSH, I₃₃₂₂
- ✅ Visualization shows graph structure clearly
- ✅ Code runs without errors on clean venv
- ✅ Documentation explains usage and design choices

---

## Phase 2 Preview (Future Work)

After Phase 1 prototype works:
1. Process all 150,061 vault chunks (not just lexicon)
2. Implement QA-Markovian random walk for traversal
3. Add temporal graph evolution (track concept changes over time)
4. Integrate with multi-agent research lab
5. Automatic graph updates from experiment results

---

## Questions / Clarifications

If stuck or need decisions:
1. **Encoding strategy**: Start with hash-based (Option 1), it's deterministic
2. **E8 roots**: Copy from existing code (t003_e8_analysis.py)
3. **Relationship extraction**: Look for keywords in definitions (uses, computes, measures, detects)
4. **Graph type**: Use NetworkX DiGraph (directed edges for asymmetric relationships)

---

## Timeline

- **Hour 1-2**: Environment setup, entity extraction
- **Hour 3-4**: QA encoding implementation
- **Hour 5-6**: Graph construction
- **Hour 7-8**: Query interface
- **Hour 9-10**: Visualization + documentation

**Target completion**: Weekend (Nov 16-17, 2025)

---

## Contact / Handoff

**Previous work**: Claude Code session 2025-11-14
**Full transcript**: `/home/player2/signal_experiments/private/QAnotes/Nexus AI Chat Imports/2025/11/Claude_GraphRAG_Discussion_2025-11-14.md`
**Session closeout**: `/home/player2/signal_experiments/SESSION_CLOSEOUT_2025-11-14.md`

**Next agent**: Gemini (theoretical analysis), OpenCode (integration planning)

---

**Priority**: HIGH
**Blocking**: Multi-agent research lab (needs GraphRAG for context retrieval)
**Impact**: Novel contribution - QA-based knowledge graphs are unexplored

**Good luck! This is genuinely innovative work.** 🚀
