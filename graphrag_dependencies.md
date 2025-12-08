# GraphRAG Deployment Dependencies & Prerequisites
**Date:** 2025-11-14
**Purpose:** Identify all requirements for GraphRAG integration

---

## Software Dependencies

### Core Python Packages
```bash
# Required for GraphRAG
pip install networkx matplotlib scipy numpy

# Already installed in project
# torch torchvision tqdm seaborn scikit-learn
```

### Version Requirements
- **Python**: 3.8+ (matches existing codebase)
- **NetworkX**: 3.0+ (for graph operations)
- **NumPy**: 1.21+ (for numerical computations)
- **Matplotlib**: 3.5+ (for visualization)
- **SciPy**: 1.7+ (for scientific computing)

### Optional Dependencies
```bash
# For advanced visualization
pip install plotly graphviz

# For performance optimization
pip install numba

# For web interface (future)
pip install fastapi uvicorn
```

## Data Prerequisites

### Required Data Files
```
📁 /home/player2/signal_experiments/
├── vault_audit_cache/
│   ├── chunks/                    # 150,061 text chunks
│   └── summaries/                 # AI-generated summaries
├── private/QAnotes/
│   └── research_log_lexicon.md    # 48 canonical QA terms
└── [GraphRAG outputs - to be created]
    ├── qa_entities.json           # Extracted entities
    ├── qa_entity_encodings.json   # QA tuple mappings
    └── qa_knowledge_graph.graphml # NetworkX graph
```

### Data Validation Checks
- **Chunk files**: Verify 150,061 files exist
- **Lexicon file**: Verify research_log_lexicon.md exists and is parseable
- **Summaries**: Verify AI-generated summaries are available

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+ (for graph operations)
- **Storage**: 5GB free space (for graph data and cache)
- **CPU**: Multi-core recommended for parallel processing

### Operating System
- **Linux**: Primary development platform
- **macOS**: Supported (with Homebrew for dependencies)
- **Windows**: Supported (with WSL recommended)

## Integration Prerequisites

### Phase 1: GraphRAG Core (Codex Deliverables)
```bash
# Must be completed before integration
✅ qa_entity_extractor.py      # Entity extraction from lexicon
✅ qa_entity_encoder.py        # QA tuple encoding
✅ qa_knowledge_graph.py       # Graph construction
✅ qa_graph_query.py           # Query interface
✅ qa_graph_viz.py             # Visualization
```

### Phase 2: Integration Components
```bash
# To be created during integration
🔄 graphrag_agent.py           # Agent wrapper class
🔄 graphrag_integration.py     # Integration utilities
🔄 test_graphrag_integration.py # Integration tests
🔄 Updated opencode_agent.py   # Add GraphRAG support
🔄 Updated opencode_cli.sh     # Add GraphRAG commands
```

### Phase 3: Agent Updates
```bash
# Modify existing agents
🔄 qa_multi_ai_orchestrator.py  # Add GraphRAG queries
🔄 Research agents              # Update to use context
🔄 Code generation agents       # Add QA concept queries
```

## Deployment Checklist

### Pre-Deployment
- [ ] GraphRAG core components implemented and tested
- [ ] Knowledge graph built successfully (48 nodes)
- [ ] Query interface working for test queries
- [ ] All dependencies installed
- [ ] System meets hardware requirements

### Deployment Steps
1. **Install Dependencies**
   ```bash
   pip install networkx matplotlib scipy
   ```

2. **Build Knowledge Graph**
   ```bash
   python qa_entity_extractor.py
   python qa_entity_encoder.py
   python qa_knowledge_graph.py
   ```

3. **Test GraphRAG Core**
   ```bash
   python qa_graph_query.py "What is Harmonic Index?"
   python qa_graph_viz.py
   ```

4. **Deploy Integration**
   ```bash
   # Add GraphRAG to opencode_agent.py
   # Update opencode_cli.sh
   # Test CLI integration
   ```

5. **Update Agents**
   ```bash
   # Modify agent classes to use GraphRAG
   # Test agent workflows
   ```

6. **Performance Testing**
   ```bash
   # Run integration tests
   # Verify performance targets
   # Monitor for errors
   ```

## Testing Prerequisites

### Unit Test Dependencies
```python
# In test files
import pytest
import networkx as nx
from graphrag_agent import GraphRAGAgent
```

### Integration Test Setup
```python
# Test fixtures
@pytest.fixture
def sample_graph():
    """Load test knowledge graph"""
    return load_test_graph()

@pytest.fixture
def graphrag_agent(sample_graph):
    """Create GraphRAG agent for testing"""
    return GraphRAGAgent(graph=sample_graph)
```

### Test Data Requirements
- **Mock graph**: Small test graph with known entities
- **Test queries**: Pre-defined queries with expected results
- **Performance baselines**: Target response times and accuracy

## Monitoring & Maintenance

### Logging Dependencies
```python
# Structured logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('graphrag')
```

### Metrics Collection
```python
# Prometheus or similar
# Query latency, success rate, cache hit rate
# Graph size, node/edge counts
```

### Health Checks
```python
def health_check():
    """Verify GraphRAG is operational"""
    try:
        agent = GraphRAGAgent()
        results = agent.query("test", top_k=1)
        return len(results['results']) > 0
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

## Security Considerations

### Data Access
- **Vault data**: Ensure proper access controls
- **Graph files**: Store in secure locations
- **API access**: Implement authentication if needed

### Input Validation
- **Query sanitization**: Prevent injection attacks
- **Result limits**: Cap response sizes
- **Rate limiting**: Prevent abuse

## Rollback Plan

### Version Control
- **Graph versions**: Keep multiple versions of knowledge graph
- **Code versions**: Tag releases for easy rollback
- **Configuration**: Version control all settings

### Rollback Procedure
1. **Stop agents** using GraphRAG
2. **Revert code** to previous version
3. **Restore previous** graph file if needed
4. **Restart agents** with fallback mode
5. **Verify functionality** without GraphRAG

## Future Scaling Considerations

### Phase 2 Expansion
- **150k chunks**: Will require optimized graph storage
- **Performance**: May need graph partitioning or indexing
- **Memory**: Larger graphs may need disk-based storage

### Advanced Features
- **Real-time updates**: Streaming graph updates
- **Multi-modal**: Support for images, code, etc.
- **Federated graphs**: Multiple specialized graphs

## Risk Assessment

### High Risk Items
- **Graph corruption**: Single point of failure
- **Performance degradation**: Large graphs slow queries
- **Integration complexity**: Many agents to update

### Mitigation Strategies
- **Backups**: Regular graph backups
- **Monitoring**: Comprehensive performance monitoring
- **Gradual rollout**: Deploy to one agent at a time
- **Fallbacks**: Agents work without GraphRAG

## Success Verification

### Deployment Success Criteria
- [ ] All dependencies installed successfully
- [ ] Knowledge graph builds without errors
- [ ] Query interface responds correctly
- [ ] CLI commands work
- [ ] Agents integrate without breaking existing functionality
- [ ] Performance meets targets (<1s query time)
- [ ] Error handling works gracefully

### Operational Success Criteria
- [ ] 95%+ query success rate
- [ ] <500ms average response time
- [ ] Agents show improved response quality
- [ ] No critical errors in logs
- [ ] Monitoring alerts configured

---

## Quick Start Checklist

### For New Deployments
```bash
# 1. Check system requirements
python --version  # 3.8+
free -h          # 8GB+ RAM

# 2. Install dependencies
pip install networkx matplotlib scipy

# 3. Verify data exists
ls vault_audit_cache/chunks/ | wc -l  # Should be 150061
ls private/QAnotes/research_log_lexicon.md  # Should exist

# 4. Build and test
python qa_entity_extractor.py
python qa_entity_encoder.py
python qa_knowledge_graph.py
python qa_graph_query.py "test query"

# 5. Deploy integration
# [Integration steps from deployment checklist]
```

This comprehensive dependency and prerequisite list ensures smooth GraphRAG deployment in the multi-agent research lab.