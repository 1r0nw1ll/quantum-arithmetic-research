# GraphRAG Integration Specification
**Date:** 2025-11-14
**Purpose:** Define API for QA-GraphRAG integration into multi-agent research lab

---

## Overview

The QA-GraphRAG system will provide context retrieval capabilities to agents in the multi-agent research lab. It uses QA tuples instead of vector embeddings for knowledge graph construction and retrieval.

## Architecture Integration Points

### Current System Components
- **QA Lab**: Task management, agent prompts, context storage
- **OpenCode Agent**: Communication with OpenCode CLI
- **Codex Agent**: Code generation via Codex CLI
- **QALM**: Local QA-specialized language model (in development)

### GraphRAG Role
- **Context Provider**: Supplies relevant knowledge chunks from vault to agents
- **Knowledge Graph**: QA-tuple based graph of research concepts and relationships
- **Query Interface**: Natural language queries → QA tuples → graph traversal → context results

## API Specification

### GraphRAGAgent Class

```python
class GraphRAGAgent:
    """Agent for querying QA-GraphRAG knowledge graph"""

    def __init__(self, graph_path: str = "qa_knowledge_graph.graphml"):
        """Initialize with pre-built knowledge graph"""
        self.graph_path = graph_path
        self.graph = None  # NetworkX graph
        self.load_graph()

    def load_graph(self) -> bool:
        """Load the QA knowledge graph"""
        # Implementation in qa_graph_query.py

    def query(self, query_str: str, top_k: int = 5,
              include_context: bool = True) -> Dict:
        """
        Query the knowledge graph

        Args:
            query_str: Natural language query
            top_k: Number of top results to return
            include_context: Include full context chunks

        Returns:
            {
                'query': query_str,
                'results': [
                    {
                        'entity': 'Harmonic Index',
                        'qa_tuple': (12, 8, 20, 4),
                        'e8_alignment': 0.87,
                        'definition': '...',
                        'relationships': ['USES → E8 alignment', ...],
                        'context_chunk': '...'  # if include_context=True
                    },
                    ...
                ],
                'traversal_path': [...]  # Optional: show graph traversal
            }
        """
        # Implementation in qa_graph_query.py

    def get_entity_details(self, entity_name: str) -> Dict:
        """Get detailed information about a specific entity"""
        # Return full entity metadata

    def find_related_entities(self, entity_name: str, relationship_type: str = None) -> List:
        """Find entities related to given entity"""
        # Traverse graph edges
```

### Integration Workflow

#### Agent Query Flow
```
Agent Request → GraphRAG Query → Context Retrieval → Enhanced Response
```

1. **Agent receives task/query**
2. **Agent calls GraphRAG for context**:
   ```python
   from graphrag_agent import GraphRAGAgent
   rag = GraphRAGAgent()
   context = rag.query("What is Harmonic Index?", top_k=3)
   ```
3. **Agent incorporates context into prompt**:
   ```python
   enhanced_prompt = f"""
   Context from knowledge graph:
   {context['results'][0]['definition']}
   Related concepts: {', '.join([r['entity'] for r in context['results'][1:]])}

   Original task: {original_task}
   """
   ```
4. **Agent generates response using context**

#### Data Flow
```
Vault Chunks (150k) → GraphRAG Construction → Knowledge Graph → Query Interface → Agent Context → Response Generation
```

## CLI Integration

### Extend opencode_cli.sh

Add GraphRAG commands:
```bash
# Query GraphRAG
opencode rag "What is QA tuple?"
# Returns: Top 5 relevant entities with definitions

# Get entity details
opencode rag-entity "Harmonic Index"
# Returns: Full entity information

# Find related concepts
opencode rag-related "E8 alignment" --type "USES"
# Returns: Entities that use E8 alignment
```

### Python API Usage

```python
# In agent workflows
from graphrag_agent import GraphRAGAgent

class ResearchAgent:
    def __init__(self):
        self.rag = GraphRAGAgent()

    def answer_question(self, question: str) -> str:
        # Get context from GraphRAG
        context = self.rag.query(question, top_k=3)

        # Build enhanced prompt
        context_str = "\n".join([
            f"- {r['entity']}: {r['definition'][:200]}..."
            for r in context['results']
        ])

        prompt = f"""
        Use this context from the knowledge graph:

        {context_str}

        Question: {question}

        Answer based on the context:
        """

        # Generate response (using QALM, Claude, etc.)
        return self.generate_response(prompt)
```

## Implementation Plan

### Phase 1: Core GraphRAG (Codex)
- Build qa_entity_extractor.py
- Build qa_entity_encoder.py
- Build qa_knowledge_graph.py
- Build qa_graph_query.py
- Build qa_graph_viz.py

### Phase 2: Integration (OpenCode)
- Create GraphRAGAgent class
- Add to opencode_agent.py
- Extend opencode_cli.sh
- Create integration tests
- Update documentation

### Phase 3: Agent Integration
- Modify existing agents to use GraphRAG
- Update qa_multi_ai_orchestrator.py
- Test end-to-end workflows
- Performance optimization

## Dependencies

### Required Files (from Codex)
- `qa_knowledge_graph.graphml` - NetworkX graph export
- `qa_entity_encodings.json` - QA tuple mappings
- `qa_graph_query.py` - Query implementation

### Integration Files (to create)
- `graphrag_agent.py` - Agent wrapper class
- `graphrag_integration.py` - Integration utilities
- `test_graphrag_integration.py` - Tests

## Success Criteria

- ✅ GraphRAG loads knowledge graph successfully
- ✅ Query "What is Harmonic Index?" returns definition + related concepts
- ✅ Agents can retrieve context before generating responses
- ✅ CLI commands work: `opencode rag "query"`
- ✅ Integration doesn't break existing workflows
- ✅ Performance: Query response < 2 seconds

## Testing

### Unit Tests
```python
def test_graphrag_agent():
    agent = GraphRAGAgent()
    results = agent.query("Harmonic Index", top_k=3)
    assert len(results['results']) == 3
    assert 'definition' in results['results'][0]

def test_cli_integration():
    # Test CLI commands work
    result = subprocess.run(['opencode', 'rag', 'QA tuple'],
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert 'QA tuple' in result.stdout
```

### Integration Tests
```python
def test_agent_with_graphrag():
    # Test that agents use GraphRAG context
    agent = ResearchAgent()
    response = agent.answer_question("What is Harmonic Index?")

    # Response should contain accurate information from graph
    assert "scalar order parameter" in response.lower()
    assert "QA-Markovian dynamics" in response.lower()
```

## Timeline

- **Week 1**: GraphRAG core implementation (Codex)
- **Week 2**: Integration API and CLI (OpenCode)
- **Week 3**: Agent integration and testing
- **Week 4**: Performance optimization and documentation

## Risk Mitigation

- **Graph Loading**: Ensure graph file exists and is valid
- **Query Performance**: Implement caching for frequent queries
- **Error Handling**: Graceful degradation if GraphRAG unavailable
- **Version Compatibility**: Handle graph schema changes

---

## Conclusion

GraphRAG integration will provide agents with rich, structured context from the research vault, enabling more accurate and informed responses. The API design follows existing patterns in the multi-agent system for seamless integration.