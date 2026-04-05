# QA Terminal Agent - Network Chuck Inspired Multi-AI Orchestration

**Phase 3 Implementation Complete** ✅

Terminal-first multi-AI collaboration system for QA (Quantum Arithmetic) research with persistent context management.

---

## Features

### 🤖 Multi-AI Orchestration
- **Claude Code** - Current session integration
- **Codex** - Code generation (if CLI installed)
- **Gemini** - Mathematical analysis (if CLI installed)
- **QALM** - QA-specialized local model

### 📁 Persistent Context
- YAML-based session state
- Chat history preservation
- Experiment tracking
- Active tuple management
- Research notes and tags

### 🔌 MCP Integration
- Direct MCP tool calling
- Transparent QA computation
- Results integrated into context

### 🎯 QA-Aware System Prompts
- Automatic QA invariant injection
- Modular arithmetic constraints
- Active experiment context
- MCP tool availability

---

## Quick Start

### Install Dependencies
```bash
# Core dependencies (already in qa_lab)
cd /home/player2/signal_experiments/qa_lab
pip install pyyaml  # If not already installed

# Optional AI provider CLIs
pip install openai-codex-cli  # For Codex
npm install -g @google/generative-ai-cli  # For Gemini
```

### List Available AI Providers
```bash
python3 qa-in-terminal/qa_terminal_agent.py --list-providers
```

Output:
```
🤖 Available AI Providers:
  ✅ claude: Claude Code (current session)
  ✅ codex: OpenAI Codex (code generation)
  ✅ gemini: Google Gemini (analysis)
  ✅ qalm: QA Local Model (specialized)
```

### Show Current Context
```bash
python3 qa-in-terminal/qa_terminal_agent.py --show-context
```

### Call MCP Tool Directly
```bash
python3 qa-in-terminal/qa_terminal_agent.py \
  --mcp qa_compute_triangle \
  --mcp-args '{"b": 3.0, "e": 5.0}'
```

---

## Usage Modes

### 1. Interactive Mode (Recommended)
```bash
# Start with default context
python3 qa-in-terminal/qa_terminal_agent.py

# Or use specific context
python3 qa-in-terminal/qa_terminal_agent.py -c qa_lab/qa_contexts/proton_radius.yaml
```

Interactive commands:
- `/switch <provider>` - Switch AI provider
- `/context` - Show context summary
- `/providers` - List available providers
- `/mcp <tool> <args>` - Call MCP tool
- `/quit` or `/exit` - Exit agent

### 2. Single Query Mode
```bash
# Query Claude
python3 qa-in-terminal/qa_terminal_agent.py "Compute QA tuple for (5, 8)"

# Query Gemini with specific context
python3 qa-in-terminal/qa_terminal_agent.py \
  -p gemini \
  -c qa_lab/qa_contexts/theorem_discovery.yaml \
  "Prove that C = d² for all QA tuples"
```

### 3. MCP Tool Mode
```bash
# Compute QA triangle
python3 qa-in-terminal/qa_terminal_agent.py \
  --mcp qa_compute_triangle \
  --mcp-args '{"b": 1.0, "e": 2.0}'
```

---

## Context Files

### Base Context
**File**: `qa_lab/qa_contexts/base_context.yaml`
**Purpose**: General QA research starting point
```bash
python3 qa-in-terminal/qa_terminal_agent.py -c qa_lab/qa_contexts/base_context.yaml
```

### Proton Radius Context
**File**: `qa_lab/qa_contexts/proton_radius.yaml`
**Purpose**: Proton radius ellipse quantization experiments
```bash
python3 qa-in-terminal/qa_terminal_agent.py -c qa_lab/qa_contexts/proton_radius.yaml
```

**Preloaded data**:
- Physical constants (CODATA proton radius, Planck length)
- Best candidate QA tuples for proton radius
- Ongoing experiments with results
- Scaling factors and error analysis

### Theorem Discovery Context
**File**: `qa_lab/qa_contexts/theorem_discovery.yaml`
**Purpose**: Automated theorem generation from QA patterns
```bash
python3 qa-in-terminal/qa_terminal_agent.py -c qa_lab/qa_contexts/theorem_discovery.yaml
```

**Preloaded data**:
- 4 discovered theorems (2 proven, 2 conjectures)
- Algebraic proofs
- Computational evidence
- Next steps for validation

---

## Example Workflows

### Workflow 1: Proton Radius Discovery
```bash
cd /home/player2/signal_experiments
bash qa-in-terminal/workflows/01_proton_radius_discovery.sh
```

**Steps**:
1. Verify environment and context
2. Compute Fibonacci seed tuple (1,2,3,5) via MCP
3. Compute high-resonance tuple (3,5,8,13) via MCP
4. Analyze ellipse properties with Claude
5. Validate with Gemini, QALM, and Codex

### Workflow 2: Theorem Discovery
```bash
cd /home/player2/signal_experiments
bash qa-in-terminal/workflows/02_theorem_discovery.sh
```

**Steps**:
1. Show current theorem knowledge base
2. Generate sample QA tuples for pattern analysis
3. Interactive theorem discovery with multi-AI
4. Automated validation pipeline

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│         QA Terminal Agent (Orchestrator)         │
│  - Persistent YAML context                       │
│  - QA-aware system prompts                       │
│  - Chat history management                       │
└────────┬─────────────────────────────────────────┘
         │
    ┌────┴─────┬──────────┬──────────┬─────────┐
    │          │          │          │         │
    ▼          ▼          ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐  ┌─────┐
│ Claude │ │ Codex  │ │ Gemini │ │ QALM │  │ MCP │
│  Code  │ │  CLI   │ │  CLI   │ │Local │  │Tools│
└────────┘ └────────┘ └────────┘ └──────┘  └─────┘
```

---

## Development

### Adding New Contexts
Create a new YAML file in `qa_lab/qa_contexts/`:
```yaml
---
project_name: "Your Research Project"
modulus_outer: 24
modulus_inner: 9
qa_invariants:
  tuple_structure: "(b, e, d, a) where d = b+e, a = b+2e"
  core_invariants:
    - "J = b·d"
    - "K = d·a"
    - "X = e·d"
mcp_servers:
  - qa-right-triangle
  - qa-resonance
  - qa-hgd-optimizer
active_tuples: []
experiments: []
chat_history: []
tags:
  - your-tag
```

### Adding New AI Providers
Edit `qa_terminal_agent.py` and add to the `providers` dictionary:
```python
self.providers = {
    'your_provider': {
        'call': self.call_your_provider,
        'description': 'Your Provider Description',
        'available': self._check_command_exists('your_cli_command')
    }
}
```

---

## Testing

### Basic Functionality Tests
```bash
cd /home/player2/signal_experiments

# Test 1: List providers
python3 qa-in-terminal/qa_terminal_agent.py --list-providers
# Expected: Shows 4 AI providers with availability status

# Test 2: Show context
python3 qa-in-terminal/qa_terminal_agent.py --show-context
# Expected: Displays base context summary

# Test 3: MCP tool call
python3 qa-in-terminal/qa_terminal_agent.py \
  --mcp qa_compute_triangle \
  --mcp-args '{"b": 1.0, "e": 1.0}'
# Expected: Returns Fibonacci seed tuple (1,1,2,3)
```

### Integration Test
```bash
# Run full Phase 3 test suite
cd qa_lab
python3 test_terminal_agent_phase3.py
```

---

## FAQ

### Q: How do I switch between AI providers in interactive mode?
A: Use the `/switch <provider>` command:
```
claude> /switch gemini
✅ Switched to gemini
gemini> Your query here
```

### Q: Where is my chat history saved?
A: Chat history is saved in the context YAML file you specified. View it with:
```bash
python3 qa-in-terminal/qa_terminal_agent.py --show-context -c your_context.yaml
```

### Q: Can I use this without Codex/Gemini CLIs?
A: Yes! Claude Code and QALM work without external CLIs. Codex and Gemini are optional enhancements.

### Q: How do I call MCP tools from within an AI conversation?
A: In interactive mode, use the `/mcp` command:
```
claude> /mcp qa_compute_triangle {"b": 3, "e": 5}
```

---

## References

- **NetworkChuck Tutorial**: [AI in the Terminal](https://github.com/theNetworkChuck/ai-in-the-terminal)
- **NetworkChuck Docker MCP**: [Docker MCP Tutorial](https://github.com/theNetworkChuck/docker-mcp-tutorial)
- **MCP Specification**: https://modelcontextprotocol.io/
- **QA Lab Documentation**: `../qa_lab/QA_LAB_MCP_INTEGRATION_PLAN.md`

---

## Status

**Phase 3: Terminal AI Workflows** ✅ COMPLETE

**Next**: Phase 4 (Multi-AI Orchestration Enhancement)

**Created**: 2025-11-19
**Last Updated**: 2025-11-19
