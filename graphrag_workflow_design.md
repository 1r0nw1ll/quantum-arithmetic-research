# GraphRAG Query Workflow Design
**Date:** 2025-11-14
**Purpose:** Design how agents will query and use GraphRAG for context retrieval

---

## Agent Query Workflow

### Overview
Agents follow a structured workflow to leverage GraphRAG for context-aware reasoning:

```
Task Assignment → Context Retrieval → Enhanced Reasoning → Response Generation → Validation
```

### Detailed Workflow

#### Phase 1: Task Analysis
```
Agent receives task/query
├── Parse query intent
├── Identify key concepts
├── Determine context needs
└── Check if GraphRAG query needed
```

**Decision Criteria:**
- Is this a research question? → Query GraphRAG
- Does this involve QA concepts? → Query GraphRAG
- Is this a coding task? → May skip or query for related concepts
- Is this a simple command? → Skip GraphRAG

#### Phase 2: Context Retrieval
```
Query GraphRAG for relevant knowledge
├── Convert query to GraphRAG format
├── Execute query with appropriate parameters
├── Retrieve top-k results
├── Filter and rank results
└── Extract key information
```

**Query Strategies:**
```python
# Strategy 1: Direct concept query
context = rag.query("What is Harmonic Index?", top_k=3)

# Strategy 2: Multi-concept query
concepts = extract_concepts_from_query(query)
context = rag.query(f"Explain {', '.join(concepts)}", top_k=5)

# Strategy 3: Related concepts
for concept in key_concepts:
    related = rag.find_related_entities(concept, "USES")
    context.extend(related)
```

#### Phase 3: Context Integration
```
Incorporate GraphRAG results into reasoning
├── Build context summary
├── Identify contradictions or gaps
├── Prioritize most relevant information
├── Structure context for prompt injection
└── Estimate confidence in retrieved information
```

**Context Formatting:**
```python
def format_context_for_agent(context_results: List[Dict]) -> str:
    """Format GraphRAG results for agent consumption"""

    sections = []

    # Primary concept
    primary = context_results[0]
    sections.append(f"""
    PRIMARY CONCEPT: {primary['entity']}
    Definition: {primary['definition']}
    QA Tuple: {primary['qa_tuple']}
    E8 Alignment: {primary['e8_alignment']:.3f}
    """)

    # Related concepts
    if len(context_results) > 1:
        related = [r['entity'] for r in context_results[1:]]
        sections.append(f"RELATED CONCEPTS: {', '.join(related)}")

    # Key relationships
    relationships = []
    for result in context_results[:3]:
        if 'relationships' in result:
            relationships.extend(result['relationships'][:2])  # Top 2 per concept
    if relationships:
        sections.append(f"KEY RELATIONSHIPS: {', '.join(set(relationships))}")

    return "\n".join(sections)
```

#### Phase 4: Enhanced Reasoning
```
Use context-enhanced prompts
├── Inject context at appropriate points
├── Reference specific concepts by name
├── Use QA tuples for mathematical reasoning
├── Validate against known relationships
└── Generate response with citations
```

**Prompt Enhancement:**
```python
enhanced_prompt = f"""
RESEARCH CONTEXT:
{formatted_context}

TASK: {original_task}

INSTRUCTIONS:
- Use the provided context to inform your reasoning
- Reference specific concepts when relevant
- If context is insufficient, note this explicitly
- Maintain mathematical rigor, especially with QA concepts

RESPONSE:
"""
```

#### Phase 5: Response Validation
```
Validate response against GraphRAG knowledge
├── Check consistency with known relationships
├── Verify QA tuple usage is correct
├── Assess confidence based on context quality
├── Flag potential hallucinations
└── Suggest follow-up queries if needed
```

### Agent-Specific Workflows

#### Research Agent Workflow
```
1. Receive research question
2. Query GraphRAG for concept definitions
3. Query for related mathematical relationships
4. Build comprehensive context
5. Generate explanation with citations
6. Validate against graph relationships
```

#### Code Generation Agent Workflow
```
1. Receive coding task
2. Query GraphRAG for relevant QA concepts
3. Query for implementation patterns
4. Include code examples from context
5. Generate code with QA-aware comments
6. Validate mathematical correctness
```

#### Theorem Discovery Agent Workflow
```
1. Receive theorem discovery task
2. Query GraphRAG for existing theorems
3. Query for QA invariant patterns
4. Use context to guide proof strategies
5. Generate theorem with relationship citations
6. Validate against known mathematical structure
```

### Error Handling and Fallbacks

#### GraphRAG Unavailable
```
If GraphRAG query fails:
├── Log error for debugging
├── Continue with base agent capabilities
├── Note context limitation in response
└── Suggest manual GraphRAG query
```

#### No Relevant Results
```
If GraphRAG returns empty/low-quality results:
├── Use broader query terms
├── Query individual concepts separately
├── Fall back to general knowledge
├── Flag low-confidence response
```

#### Performance Issues
```
If GraphRAG query is slow (>2 seconds):
├── Use cached results for frequent queries
├── Reduce top_k parameter
├── Query asynchronously in background
├── Provide preliminary response
```

### Performance Optimization

#### Caching Strategy
```python
class ContextCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, query: str) -> Optional[Dict]:
        if query in self.cache:
            entry = self.cache[query]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['results']
            else:
                del self.cache[query]
        return None

    def set(self, query: str, results: Dict):
        self.cache[query] = {
            'results': results,
            'timestamp': time.time()
        }
```

#### Asynchronous Queries
```python
async def query_graphrag_async(query: str, top_k: int = 5) -> Dict:
    """Non-blocking GraphRAG query"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rag.query, query, top_k)
```

#### Query Optimization
- **Pre-compute frequent queries** during system startup
- **Batch related queries** when possible
- **Use entity IDs** for follow-up queries instead of strings
- **Implement query result ranking** based on agent needs

### Monitoring and Analytics

#### Query Metrics
```python
class GraphRAGMetrics:
    def __init__(self):
        self.query_count = 0
        self.avg_response_time = 0
        self.cache_hit_rate = 0
        self.empty_result_rate = 0

    def record_query(self, query: str, response_time: float,
                    results_count: int, cache_hit: bool):
        # Update metrics
        pass

    def get_stats(self) -> Dict:
        return {
            'total_queries': self.query_count,
            'avg_response_time': self.avg_response_time,
            'cache_hit_rate': self.cache_hit_rate,
            'empty_result_rate': self.empty_result_rate
        }
```

#### Agent Usage Patterns
- Track which agents use GraphRAG most
- Identify common query patterns
- Measure impact on response quality
- Optimize based on usage analytics

### Integration Testing

#### Workflow Tests
```python
def test_research_agent_workflow():
    """Test complete research agent workflow with GraphRAG"""
    agent = ResearchAgent()

    # Test query
    query = "How does Harmonic Index relate to E8 alignment?"
    response = agent.answer_question(query)

    # Assertions
    assert "Harmonic Index" in response
    assert "E8 alignment" in response
    assert response_has_citations(response)
    assert response_is_mathematically_correct(response)

def test_error_handling():
    """Test graceful degradation when GraphRAG fails"""
    # Mock GraphRAG failure
    with patch('graphrag_agent.GraphRAGAgent.query', side_effect=Exception):
        agent = ResearchAgent()
        response = agent.answer_question("Test query")

        # Should still provide useful response
        assert len(response) > 50
        assert "context unavailable" in response.lower()
```

### Success Metrics

- **Query Success Rate**: >95% of queries return relevant results
- **Response Time**: <1 second average for cached queries
- **Agent Improvement**: 20%+ improvement in response accuracy with GraphRAG
- **User Satisfaction**: Agents provide more comprehensive answers

---

## Conclusion

The GraphRAG query workflow provides a structured approach for agents to leverage QA-based knowledge graphs for enhanced reasoning. By following this workflow, agents can provide more accurate, context-aware responses while maintaining performance and reliability.