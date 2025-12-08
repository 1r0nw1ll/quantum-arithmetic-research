# Handoff to OpenCode - QA-GraphRAG Integration Architecture
**Date**: 2025-11-14
**From**: Claude Code
**Priority**: MEDIUM
**Estimated Time**: 6-8 hours

---

## Mission

Design the integration architecture for QA-GraphRAG with the existing signal_experiments codebase and future multi-agent research lab. Focus on API design, data flow, and autonomous graph updates.

---

## Context

A QA-based GraphRAG system is being built to provide semantic retrieval over 150k research notes. Your task is to design how this integrates with:

1. **Existing codebase** - 88 Python experiment scripts
2. **Multi-agent research lab** - Autonomous agents conducting experiments
3. **Obsidian vault** - Living research knowledge base
4. **Future systems** - Real-time graph updates, experiment tracking

---

## Integration Points

### 1. Existing Graph Infrastructure

**File**: `/home/player2/signal_experiments/qa_graph_builder_v2.py`

**Current capabilities**:
- Builds PyTorch Geometric graphs from QA tuples
- Creates nodes from (b,e,d,a) tuples
- Edges based on harmonic/modular/geometric properties
- Uses NetworkX underneath

**Your task**:
- Review qa_graph_builder_v2.py architecture
- Identify reusable components for GraphRAG
- Design adapter layer to convert between:
  - Mathematical QA graphs (current)
  - Knowledge QA graphs (new)
- Ensure both can coexist in same codebase

**Deliverable**: `qa_graph_adapter.py` specification

---

### 2. Multi-Agent Research Lab

**Planned architecture** (from Claude session):
```python
Agents:
- Experimenter: Modifies parameters, runs scripts
- Analyzer: Interprets results, generates hypotheses
- Coder: Writes new experiment variations
- Validator: Runs statistical tests
- Documenter: Updates research notes
```

**GraphRAG integration needs**:

**A. Context Retrieval API**
```python
class GraphRAGContextProvider:
    """Provides context to agents from knowledge graph"""

    def get_context(self, agent_query: str, top_k: int = 5) -> dict:
        """
        Agent asks: "What experiments have tested E8 alignment?"
        GraphRAG returns: Relevant scripts, results, notes
        """
        pass

    def get_related_experiments(self, concept: str) -> list:
        """Find all experiments mentioning a concept"""
        pass

    def get_historical_context(self, experiment_name: str) -> dict:
        """Get previous runs, results, failures for an experiment"""
        pass
```

**B. Autonomous Graph Updates**
```python
class GraphRAGUpdater:
    """Automatically updates graph when agents make discoveries"""

    def on_experiment_complete(self, experiment: dict):
        """
        When experiment finishes:
        1. Extract new entities (metrics, parameters, results)
        2. Encode to QA tuples
        3. Add to graph with relationships
        4. Update Obsidian vault
        """
        pass

    def on_hypothesis_generated(self, hypothesis: dict):
        """Add hypotheses as provisional nodes"""
        pass

    def on_bug_discovered(self, bug_report: dict):
        """Link bugs to affected experiments"""
        pass
```

**Your task**:
- Design full API specification
- Define data schemas (JSON/YAML)
- Specify event hooks for agent actions
- Design conflict resolution (multiple agents updating graph)

**Deliverable**: `AGENT_GRAPHRAG_API_SPEC.md`

---

### 3. Obsidian Vault Integration

**Current state**:
- 1,152 markdown files in `/home/player2/signal_experiments/private/QAnotes/`
- Obsidian plugins: nexus-ai-chat-importer, graph-analysis
- Wikilinks [[syntax]] used sparsely
- Manual updates only

**Desired state**:
- Bidirectional sync: Graph ↔ Vault
- Automatic backlinks when graph edges added
- Graph visualization in Obsidian
- Daily graph summaries exported to vault

**Your task**:

**A. Graph → Vault Sync**
```python
class ObsidianGraphSync:
    """Sync graph updates to Obsidian vault"""

    def export_entity_to_note(self, entity: str, qa_tuple: tuple):
        """
        Create/update markdown file for entity:

        # Harmonic Index
        **QA Tuple**: (12, 8, 20, 4)
        **E8 Alignment**: 0.87

        ## Definition
        Scalar order parameter from QA-Markovian dynamics...

        ## Related Concepts
        - [[E8 alignment]]
        - [[QA tuple]]
        - [[Harmonic loss]]

        ## Experiments Using This
        - [[run_signal_experiments_final.py]]
        - [[geometric_autopsy.py]]
        """
        pass

    def update_backlinks(self, source: str, target: str):
        """Add [[wikilinks]] between related notes"""
        pass
```

**B. Vault → Graph Sync**
```python
class VaultWatcher:
    """Monitor vault for changes, update graph"""

    def watch_for_changes(self):
        """Use inotify/watchdog to detect file changes"""
        pass

    def parse_new_entities(self, markdown_file: str) -> list:
        """Extract new concepts from research notes"""
        pass

    def update_graph_from_vault(self):
        """Incremental graph updates from vault changes"""
        pass
```

**Your task**:
- Design sync protocol (polling vs event-driven)
- Handle conflicts (graph vs vault updates)
- Specify markdown template for auto-generated notes
- Plan Obsidian plugin integration (if needed)

**Deliverable**: `OBSIDIAN_SYNC_DESIGN.md`

---

### 4. Experiment Tracking System

**Problem**: 88 Python scripts with no unified tracking
- No record of which experiments have run
- No parameter history
- No automatic linking of results to concepts

**Proposed solution**: GraphRAG as experiment registry

```python
class ExperimentRegistry:
    """Track all experiments in knowledge graph"""

    def register_experiment(self, script_path: str, metadata: dict):
        """
        Add experiment as graph node:
        - QA tuple encoding (based on domain + complexity)
        - Links to concepts it tests
        - Links to datasets it uses
        - Parameter space
        """
        pass

    def log_experiment_run(self, experiment: str, params: dict, results: dict):
        """
        Create run node:
        - Timestamp
        - Parameters used
        - Results achieved
        - Link to experiment parent node
        - Link to affected concepts (if validates/refutes)
        """
        pass

    def query_experiment_history(self, experiment: str) -> list:
        """Get all historical runs with results"""
        pass

    def find_similar_experiments(self, params: dict) -> list:
        """Find experiments with similar parameter settings"""
        pass
```

**Integration with existing scripts**:
```python
# Add to each experiment script:
import qa_graphrag

# At script start
registry = qa_graphrag.ExperimentRegistry()
run_id = registry.start_run(
    experiment="run_signal_experiments_final.py",
    params={"MODULUS": 24, "NUM_NODES": 24, "NUM_STEPS": 1000}
)

# Run experiment...
results = {"HI_mean": 0.73, "E8_alignment": 0.85, "accuracy": 0.89}

# At script end
registry.end_run(run_id, results=results, artifacts=["classification_grid.png"])
```

**Your task**:
- Design experiment metadata schema
- Specify how experiments get QA encoded
- Design parameter space representation
- Plan automatic artifact linking (PNGs, CSVs to graph nodes)

**Deliverable**: `EXPERIMENT_REGISTRY_SPEC.md`

---

### 5. Real-Time Graph Evolution

**Vision**: Graph grows and evolves as research progresses

**Timeline view**:
```
Nov 2023: Initial QA concept nodes
Jan 2024: E8 alignment discovery → High-weight edges added
Mar 2024: Bell test validation → CHSH node, Tsirelson bound node
Oct 2024: Hyperspectral experiments → Domain mismatch discovered
Nov 2024: QA-GraphRAG meta-node added (self-referential!)
```

**Your task**:

**A. Temporal Graph Schema**
```python
class TemporalGraphRAG:
    """Track graph evolution over time"""

    def add_timestamped_entity(self, entity: str, timestamp: datetime):
        """Entities have creation dates"""
        pass

    def add_timestamped_edge(self, source: str, target: str, timestamp: datetime):
        """Relationships evolve over time"""
        pass

    def query_graph_at_time(self, timestamp: datetime) -> Graph:
        """Get graph snapshot at specific time"""
        pass

    def track_concept_evolution(self, concept: str) -> Timeline:
        """Show how understanding of concept changed over time"""
        pass
```

**B. Discovery Detection**
```python
class DiscoveryDetector:
    """Automatically detect research breakthroughs"""

    def detect_new_connection(self, graph_before: Graph, graph_after: Graph):
        """
        If new edge added with high E8 alignment between
        previously unconnected concepts → potential discovery
        """
        pass

    def detect_contradiction(self, new_result: dict, graph: Graph):
        """
        If new result contradicts existing node properties
        → flag for investigation
        """
        pass

    def detect_convergence(self, experiments: list) -> bool:
        """
        If multiple experiments converge on same conclusion
        → high-confidence finding
        """
        pass
```

**Your task**:
- Design temporal graph data model
- Specify discovery detection heuristics
- Plan visualization of graph evolution (animation?)
- Define "breakthrough" metrics

**Deliverable**: `TEMPORAL_GRAPH_DESIGN.md`

---

## System Architecture Diagram

Your deliverables should include a comprehensive architecture diagram showing:

```
┌─────────────────────────────────────────────────────────────┐
│                    QA Research Ecosystem                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  Obsidian    │◄────────┤  QA-GraphRAG │                  │
│  │   Vault      │         │  Knowledge   │                  │
│  │ (1152 notes) │────────►│    Graph     │                  │
│  └──────────────┘         └───────┬──────┘                  │
│         ▲                         │                          │
│         │                         ▼                          │
│         │                 ┌──────────────┐                  │
│         │                 │ Experiment   │                  │
│         │                 │  Registry    │                  │
│         │                 └───────┬──────┘                  │
│         │                         │                          │
│         │                         ▼                          │
│  ┌──────┴────────┐       ┌──────────────┐                  │
│  │ Multi-Agent   │◄──────┤    Context   │                  │
│  │ Research Lab  │       │   Provider   │                  │
│  │               │───────►              │                  │
│  └───────┬───────┘       └──────────────┘                  │
│          │                                                   │
│          ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Experiments  │                                          │
│  │ (88 scripts) │                                          │
│  └──────────────┘                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Data Flow:
1. Experiment runs → Results → Registry → Graph update
2. Agent query → Context Provider → GraphRAG → Retrieved context
3. Graph update → Obsidian sync → New/updated notes
4. Vault edit → Watcher → Graph update → Registry update
5. Discovery detection → Agent notification → New experiments
```

---

## API Specifications

### Core GraphRAG API

```python
# qa_graphrag_api.py

class QAGraphRAG:
    """Main API for QA-based knowledge graph"""

    def __init__(self, graph_path: str = "qa_knowledge_graph.graphml"):
        self.graph = self.load_graph(graph_path)
        self.encoder = QAEntityEncoder()
        self.retriever = QARetriever()

    # Query Interface
    def query(self, query_str: str, top_k: int = 5) -> list:
        """Main query endpoint"""
        pass

    def find_related(self, entity: str, relationship: str = None) -> list:
        """Find entities related to given entity"""
        pass

    def find_path(self, source: str, target: str) -> list:
        """Find semantic path between two concepts"""
        pass

    # Graph Modification
    def add_entity(self, name: str, metadata: dict) -> str:
        """Add new entity, returns QA tuple"""
        pass

    def add_relationship(self, source: str, target: str, rel_type: str):
        """Add edge between entities"""
        pass

    def update_entity(self, name: str, metadata: dict):
        """Update entity properties"""
        pass

    # Bulk Operations
    def ingest_markdown(self, filepath: str):
        """Extract entities from markdown file"""
        pass

    def export_to_obsidian(self, vault_path: str):
        """Export graph as Obsidian notes"""
        pass

    # Analytics
    def get_entity_importance(self, entity: str) -> float:
        """PageRank-style importance score"""
        pass

    def get_graph_summary(self) -> dict:
        """Statistics: node count, edge count, density, etc."""
        pass

    def find_clusters(self) -> dict:
        """Community detection on graph"""
        pass
```

### Agent Integration API

```python
# qa_graphrag_agent_api.py

class AgentGraphRAGInterface:
    """Simplified API for multi-agent lab"""

    def ask(self, question: str) -> str:
        """
        Natural language query, returns formatted answer

        Agent: "What experiments have tested E8 alignment?"
        GraphRAG: "5 experiments found:
                   1. run_signal_experiments_final.py (HI=0.73)
                   2. geometric_autopsy.py (TDA analysis)
                   ..."
        """
        pass

    def get_context_for_task(self, task_description: str) -> dict:
        """
        Agent: "I'm about to run seizure detection experiment"
        GraphRAG: Returns relevant past results, known issues, best params
        """
        pass

    def log_discovery(self, discovery: dict):
        """
        Agent reports: "Found that HI>0.8 predicts seizure in 85% of cases"
        GraphRAG: Adds to graph, links to experiments, updates vault
        """
        pass

    def request_experiment_suggestion(self, constraints: dict) -> list:
        """
        Agent: "Suggest experiments related to Bell tests with runtime <10min"
        GraphRAG: Queries graph, filters by constraints, returns ranked list
        """
        pass
```

---

## Data Schemas

### Entity Schema
```json
{
  "name": "Harmonic Index",
  "type": "concept",
  "qa_tuple": [12, 8, 20, 4],
  "e8_alignment": 0.87,
  "definition": "Scalar order parameter from QA-Markovian dynamics",
  "symbols": ["HI", "H"],
  "first_mentioned": "2024-03-15T10:30:00Z",
  "related_experiments": ["run_signal_experiments_final.py"],
  "metadata": {
    "importance": 10,
    "frequency": 247,
    "validated": true
  }
}
```

### Relationship Schema
```json
{
  "source": "Harmonic Index",
  "target": "E8 alignment",
  "relationship": "USES",
  "transition_tuple": [3, 7, 10, 17],
  "strength": 0.92,
  "evidence": [
    "run_signal_experiments_final.py:85",
    "research_log_lexicon.md:13"
  ],
  "timestamp": "2024-03-20T14:22:00Z"
}
```

### Experiment Run Schema
```json
{
  "run_id": "run_20251114_143022",
  "experiment": "run_signal_experiments_final.py",
  "timestamp": "2025-11-14T14:30:22Z",
  "parameters": {
    "MODULUS": 24,
    "NUM_NODES": 24,
    "NUM_STEPS": 1000
  },
  "results": {
    "HI_mean": 0.73,
    "E8_alignment": 0.85,
    "accuracy": 0.89
  },
  "artifacts": [
    "outputs/images/classification_grid_20251114.png"
  ],
  "concepts_validated": ["Harmonic Index", "E8 alignment"],
  "concepts_refuted": [],
  "agent": "experimenter_agent_01"
}
```

---

## Integration Checklist

### Phase 1: Core Integration (Week 1)
- [ ] Review qa_graph_builder_v2.py architecture
- [ ] Design adapter between mathematical and knowledge graphs
- [ ] Specify GraphRAG core API (query, add, update methods)
- [ ] Design entity and relationship schemas
- [ ] Document data flow diagrams

### Phase 2: Agent Integration (Week 2)
- [ ] Specify agent API (ask, get_context, log_discovery)
- [ ] Design event hooks for agent actions
- [ ] Plan conflict resolution for concurrent updates
- [ ] Create mock agent integration test

### Phase 3: Obsidian Sync (Week 3)
- [ ] Design bidirectional sync protocol
- [ ] Specify markdown templates for auto-generated notes
- [ ] Plan wikilink management
- [ ] Design vault watcher implementation

### Phase 4: Experiment Tracking (Week 4)
- [ ] Design experiment registry schema
- [ ] Specify parameter space representation
- [ ] Plan automatic artifact linking
- [ ] Design experiment similarity search

### Phase 5: Temporal Evolution (Week 5)
- [ ] Design temporal graph data model
- [ ] Specify discovery detection heuristics
- [ ] Plan graph evolution visualization
- [ ] Design "breakthrough" detection metrics

---

## Success Criteria

- ✅ Complete API specification with all methods documented
- ✅ Data schemas for entities, relationships, experiments
- ✅ Architecture diagrams showing all integration points
- ✅ Event flow diagrams for key scenarios
- ✅ Conflict resolution strategies specified
- ✅ Performance considerations addressed (caching, indexing)
- ✅ Security considerations (who can modify graph?)
- ✅ Versioning strategy (graph snapshots, rollback)

---

## Timeline

- **Hour 1-2**: Review existing code (qa_graph_builder_v2.py, experiments)
- **Hour 3-4**: Design core API and data schemas
- **Hour 5-6**: Design agent and Obsidian integrations
- **Hour 7-8**: Document architecture, create diagrams

**Target completion**: Week of Nov 18-22, 2025

---

## Key Files to Review

### Existing Infrastructure
- `qa_graph_builder_v2.py` - Current graph implementation
- `qa_gnn_trainer_v2.py` - GNN training (may have relevant patterns)
- `qa_multi_ai_orchestrator.py` - Multi-AI coordination

### Vault Structure
- `private/QAnotes/.obsidian/` - Obsidian config
- `private/QAnotes/research_log_lexicon.md` - Canonical terms
- `vault_audit_cache/` - Existing chunk processing

### Documentation
- `CLAUDE.md` - Project overview
- `SESSION_CLOSEOUT_NOV12_2025.md` - Recent work
- `MULTI_AI_COLLABORATION_GUIDE.md` - Agent coordination

---

## Contact / Handoff

**Previous work**: Claude Code session 2025-11-14
**Full transcript**: `/home/player2/signal_experiments/private/QAnotes/Nexus AI Chat Imports/2025/11/Claude_GraphRAG_Discussion_2025-11-14.md`
**Session closeout**: `/home/player2/signal_experiments/SESSION_CLOSEOUT_2025-11-14.md`
**Parallel work**: Codex (implementation), Gemini (theory)

---

**Priority**: MEDIUM (enables future autonomous research)
**Complexity**: MEDIUM-HIGH (system integration)
**Impact**: HIGH (foundation for self-improving research system)

**Focus on clean APIs and clear data flows. This will be the glue.** 🔧
