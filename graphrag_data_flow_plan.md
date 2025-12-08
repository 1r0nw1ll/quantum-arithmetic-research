# GraphRAG Data Flow Plan
**Date:** 2025-11-14
**Purpose:** Plan end-to-end data flow from vault to agent responses

---

## Data Flow Overview

```
Raw Vault Data → Graph Construction → Query Processing → Context Enhancement → Response Generation
```

## Phase 1: Data Ingestion & Graph Construction

### Source Data
```
📁 vault_audit_cache/
├── chunks/          # 150,061 text chunks from Obsidian vault
├── summaries/       # AI-generated summaries
└── research_log_lexicon.md  # 48 canonical QA terms
```

### Processing Pipeline
```
1. Entity Extraction
   ├── Parse research_log_lexicon.md
   ├── Extract 48 canonical terms
   ├── Extract definitions, symbols, relationships
   └── Output: qa_entities.json

2. QA Encoding
   ├── Map entities to QA tuples (b,e,d,a)
   ├── Use hash-based encoding for consistency
   ├── Verify constraints: d=(b+e)%24, a=(b+2e)%24
   └── Output: qa_entity_encodings.json

3. Graph Construction
   ├── Create NetworkX DiGraph
   ├── Add 48 entity nodes with QA tuples
   ├── Extract relationships from lexicon
   ├── Add edges with transition tuples
   ├── Compute E8 alignment for edge weights
   └── Output: qa_knowledge_graph.graphml
```

### Data Quality Checks
- **Completeness**: All 48 entities extracted
- **Validity**: All QA tuples satisfy constraints
- **Connectivity**: Graph has meaningful relationships
- **Alignment**: E8 values computed correctly

## Phase 2: Query Processing

### Query Input
```
Natural Language Query
├── "What is Harmonic Index?"
├── "Find Bell test experiments"
├── "How does QA relate to seizure detection?"
└── Agent-generated queries
```

### Query Processing Flow
```
1. Query Reception
   ├── Receive from agent or CLI
   ├── Parse query string
   ├── Extract key concepts
   └── Determine query type

2. QA Tuple Conversion
   ├── Convert query to QA tuple via hashing
   ├── query_tuple = hash_to_qa_tuple(query_str)
   ├── Verify tuple constraints
   └── Log conversion for debugging

3. Graph Traversal
   ├── Load knowledge graph
   ├── Find nodes with similar QA tuples
   ├── Traverse relationship edges
   ├── Rank by E8 alignment and relevance
   └── Select top-k results
```

### Query Result Structure
```python
{
    'query': 'What is Harmonic Index?',
    'query_tuple': (12, 8, 20, 4),
    'results': [
        {
            'entity': 'Harmonic Index',
            'qa_tuple': (12, 8, 20, 4),
            'e8_alignment': 0.87,
            'definition': 'Scalar order parameter from QA-Markovian dynamics...',
            'relationships': ['USES → E8 alignment', 'COMPUTES → Loss function'],
            'context_chunk': 'Full text chunk from vault...',
            'relevance_score': 0.95
        },
        # ... more results
    ],
    'processing_time': 0.234,
    'cache_hit': False
}
```

## Phase 3: Context Enhancement

### Context Integration
```
Query Results → Context Formatting → Prompt Enhancement → Agent Reasoning
```

### Context Formatting Process
```
1. Result Filtering
   ├── Remove duplicates
   ├── Filter by relevance threshold (>0.7)
   ├── Sort by E8 alignment descending
   └── Limit to top-k (typically 3-5)

2. Information Extraction
   ├── Extract definitions
   ├── Extract key relationships
   ├── Identify mathematical concepts
   └── Preserve QA tuple information

3. Context Structuring
   ├── Primary concept first
   ├── Related concepts grouped
   ├── Mathematical relationships highlighted
   └── Citations prepared
```

### Enhanced Prompt Construction
```python
def build_enhanced_prompt(original_task: str, graph_context: Dict) -> str:
    """Build agent prompt with GraphRAG context"""

    context_sections = []

    # Primary concept
    primary = graph_context['results'][0]
    context_sections.append(f"""
    PRIMARY CONCEPT: {primary['entity']}
    Definition: {primary['definition']}
    QA Structure: {primary['qa_tuple']}
    """)

    # Related concepts
    related = [r['entity'] for r in graph_context['results'][1:]]
    if related:
        context_sections.append(f"Related Concepts: {', '.join(related)}")

    # Mathematical relationships
    relationships = []
    for result in graph_context['results']:
        relationships.extend(result.get('relationships', []))
    if relationships:
        context_sections.append(f"Key Relationships: {', '.join(set(relationships))}")

    # Combine with original task
    enhanced_prompt = f"""
    RESEARCH CONTEXT FROM KNOWLEDGE GRAPH:
    {' '.join(context_sections)}

    ORIGINAL TASK: {original_task}

    INSTRUCTIONS:
    - Use the provided context to inform your response
    - Reference specific concepts and relationships when relevant
    - Maintain mathematical accuracy
    - Cite sources when using specific definitions

    RESPONSE:
    """

    return enhanced_prompt
```

## Phase 4: Response Generation

### Agent Processing
```
Enhanced Prompt → Agent Reasoning → Response Generation → Validation
```

### Response Enhancement Types

#### Type 1: Definition Queries
```
Query: "What is Harmonic Index?"
Context: Definition + related concepts
Response: Comprehensive explanation with mathematical details
```

#### Type 2: Relationship Queries
```
Query: "How does QA relate to seizure detection?"
Context: EEG validation, brain-QA mapper, Harmonic Index applications
Response: Detailed connection explanation with examples
```

#### Type 3: Research Queries
```
Query: "Find Bell test experiments"
Context: CHSH, I₃₃₂₂, platonic solid tests
Response: Complete list with descriptions and significance
```

### Validation Layer
```
Generated Response → GraphRAG Validation → Quality Assessment
```

#### Validation Checks
- **Consistency**: Response aligns with graph relationships
- **Completeness**: All relevant concepts from context included
- **Accuracy**: Mathematical claims verified against graph
- **Citation**: Proper attribution to graph sources

## Performance Characteristics

### Latency Targets
- **Graph Loading**: <2 seconds (startup)
- **Query Processing**: <500ms average
- **Context Formatting**: <100ms
- **Total Query Time**: <1 second

### Throughput
- **Concurrent Queries**: Support 10+ simultaneous queries
- **Caching**: 80%+ cache hit rate for frequent queries
- **Memory Usage**: <500MB for graph + cache

### Scalability
- **Current**: 48 entities, ~200 relationships
- **Phase 2**: 150k chunks, millions of relationships
- **Optimization**: Hierarchical graphs, query partitioning

## Error Handling & Resilience

### Data Flow Error Scenarios

#### Graph Construction Failures
```
If graph file corrupted:
├── Fall back to entity-only search
├── Use backup graph version
├── Log error and alert administrator
└── Continue with reduced functionality
```

#### Query Processing Errors
```
If query fails:
├── Return empty results gracefully
├── Log error for debugging
├── Provide fallback response
└── Suggest alternative query formulation
```

#### Context Integration Issues
```
If context formatting fails:
├── Use simplified context format
├── Skip problematic results
├── Continue with available context
└── Note limitation in response
```

### Monitoring & Alerting

#### Key Metrics
```python
data_flow_metrics = {
    'graph_load_success': True,
    'query_success_rate': 0.98,
    'avg_query_time': 0.234,
    'context_enhancement_rate': 0.91,
    'response_quality_score': 0.87
}
```

#### Alert Conditions
- Query success rate < 95%
- Average query time > 1 second
- Graph load failures
- Context enhancement failures

## Testing Strategy

### Unit Tests
```python
def test_data_flow_pipeline():
    """Test complete data flow from query to response"""

    # Test graph construction
    graph = build_qa_knowledge_graph()
    assert len(graph.nodes()) == 48

    # Test query processing
    results = query_graph("Harmonic Index")
    assert len(results['results']) > 0
    assert 'definition' in results['results'][0]

    # Test context formatting
    context = format_context_for_agent(results)
    assert 'Harmonic Index' in context

    # Test prompt enhancement
    prompt = build_enhanced_prompt("Explain Harmonic Index", results)
    assert 'RESEARCH CONTEXT' in prompt
    assert 'ORIGINAL TASK' in prompt
```

### Integration Tests
```python
def test_end_to_end_agent_workflow():
    """Test agent using GraphRAG in complete workflow"""

    # Setup
    agent = ResearchAgent()
    rag = GraphRAGAgent()

    # Query
    query = "What is QA tuple?"
    context = rag.query(query, top_k=3)

    # Agent processing
    response = agent.generate_response_with_context(query, context)

    # Validation
    assert response_is_accurate(response, context)
    assert response_cites_sources(response)
    assert response_uses_qa_concepts_correctly(response)
```

### Performance Tests
```python
def test_performance_targets():
    """Verify performance meets targets"""

    # Load graph
    start = time.time()
    graph = load_knowledge_graph()
    load_time = time.time() - start
    assert load_time < 2.0

    # Query performance
    times = []
    for _ in range(100):
        start = time.time()
        results = query_graph("test query")
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    assert avg_time < 0.5
```

## Data Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Vault Chunks   │ -> │ Graph Construction│ -> │ Knowledge Graph │
│  (150k files)   │    │ (48 entities)     │    │ (NetworkX)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Agent Query    │ -> │ Query Processing │ -> │  Context        │
│  (NL string)    │    │ (QA tuple conv)  │    │  Results        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Context         │ -> │ Prompt           │ -> │ Enhanced        │
│ Enhancement     │    │ Enhancement      │    │ Response        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Success Criteria

- ✅ **Data Pipeline**: Vault → Graph construction works reliably
- ✅ **Query Performance**: <500ms average response time
- ✅ **Context Quality**: 90%+ of queries return relevant context
- ✅ **Agent Integration**: Agents show improved response quality
- ✅ **Error Resilience**: System degrades gracefully on failures
- ✅ **Monitoring**: All key metrics tracked and alerted

---

## Conclusion

The GraphRAG data flow provides a robust pipeline from raw research data to enhanced agent responses. By following this structured approach, the multi-agent system can leverage QA-based knowledge graphs for significantly improved reasoning and response quality.