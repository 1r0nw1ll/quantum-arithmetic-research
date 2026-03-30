# Quick Reference Card
**OpenCode + Codex + Claude Integration**

---

## 🚀 Quick Start

```bash
# Check what OpenCode is working on
./opencode_cli.sh status

# View active tasks
./opencode_cli.sh tasks

# See dataset stats
./opencode_cli.sh dataset
```

---

## 📋 Common Commands

### OpenCode Integration

```bash
# Status check
./opencode_cli.sh status

# List all work
./opencode_cli.sh list

# View specific task
./opencode_cli.sh task T-006

# Ask questions
./opencode_cli.sh ask "Explain QAAttention module"

# View QA rules
./opencode_cli.sh rules

# List agents
./opencode_cli.sh agents

# View architecture
./opencode_cli.sh arch

# Check evaluation
./opencode_cli.sh eval
```

### Codex Integration

```bash
# Generate code
./opencode_cli.sh codex "Create QA tuple validator"

# Python API
python -c "from opencode_agent import CodexAgent; \
  agent = CodexAgent(); \
  print(agent.generate_code('QA invariant checker'))"
```

### Dataset Operations

```bash
# View stats
./opencode_cli.sh dataset

# Preview examples
head -5 qa_training_dataset.jsonl | python -m json.tool

# Count by type
python -c "import json; \
  data = [json.loads(line) for line in open('qa_training_dataset.jsonl')]; \
  from collections import Counter; \
  print(Counter(d['type'] for d in data))"
```

---

## 📂 Key Files

### Tools
- `opencode_agent.py` - Python API
- `opencode_cli.sh` - CLI wrapper
- `collect_qa_training_data.py` - Dataset curation

### Data
- `qa_training_dataset.jsonl` - 31,606 examples, 11 MB

### Documentation
- `OPENCODE_INTEGRATION_SUMMARY.md` - Complete integration guide
- `T-006_COMPLETION_REPORT.md` - Dataset report
- `INTEGRATION_SESSION_SUMMARY.md` - Session summary
- `QUICK_REFERENCE.md` - This file

### OpenCode QA Lab
- `qa_lab/qa_model_architecture.py` - QALM architecture
- `qa_lab/qa_training_pipeline.py` - Training script
- `qa_lab/context/QA_RULES.yaml` - Core invariants
- `qa_lab/projects/prj-002-qa-language-model.yaml` - Project spec
- `qa_lab/tasks/active/*.yaml` - Active tasks

---

## 🎯 Active Tasks

| ID | Priority | Description | Status |
|----|----------|-------------|--------|
| T-006 | 3.6 | Dataset curation | ✅ COMPLETE |
| T-009 | 4.25 | Bob-iverse integration | 🔄 Pending |
| T-010 | 4.33 | Evaluation vs LLMs | 🔄 Pending |
| T-007 | - | Model architecture | 🔄 Pending |
| T-008 | - | Training pipeline | 🔄 Pending |

---

## 📊 Dataset Quick Stats

```
Total: 31,606 examples
Size: 11 MB

By Type:
  Theorems .......... 9,033 (28.6%)
  Synthetic QA ..... 10,000 (31.6%)
  Real Examples ..... 6,572 (20.8%)
  Q&A Pairs ......... 5,000 (15.8%)
  E8 Mappings ....... 1,000 (3.2%)

By Domain:
  qa_synthetic ..... 10,000
  qa_mathematics .... 9,033
  qa_tuples ......... 6,572
  qa_qa_pairs ....... 5,000
  e8_geometry ....... 1,000
```

---

## 🔧 Python API Examples

### OpenCode Agent

```python
from opencode_agent import OpenCodeAgent

agent = OpenCodeAgent('/home/player2/signal_experiments')

# Get status
status = agent.get_status()
print(agent.parse_response(status))

# List work
work = agent.list_recent_work()
print(agent.parse_response(work))

# Query
response = agent.query("What is the QAAttention module?")
print(agent.parse_response(response))
```

### Codex Agent

```python
from opencode_agent import CodexAgent

codex = CodexAgent('/home/player2/signal_experiments')

# Generate code
code = codex.generate_code("QA tuple validator function")
print(code)

# Execute prompt
response = codex.exec_prompt("Create function to compute J, K, X invariants")
if response['success']:
    print(response['output'])
```

### Multi-AI Workflow

```python
from opencode_agent import OpenCodeAgent, CodexAgent

# Initialize agents
opencode = OpenCodeAgent()
codex = CodexAgent()

# 1. Ask OpenCode about architecture
response = opencode.query("How does QAAttention preserve invariants?")
architecture_info = opencode.parse_response(response)

# 2. Use Codex to generate implementation
code = codex.generate_code(f"Implement: {architecture_info}")

# 3. Claude synthesizes and integrates
# (You handle this part in conversation)
```

---

## 🎓 QA Core Invariants

```yaml
# From qa_lab/context/QA_RULES.yaml

invariants:
  J: "b·d"  # First fundamental
  K: "d·a"  # Second fundamental
  X: "e·d"  # Third fundamental

closure:
  b + e = d
  e + d = a

ellipse_laws:
  inner_ellipse: "a² = d² + 2*d*e + e²"

modular_arithmetic: "mod 9, 24, 72, 288"
```

---

## 🚀 Next Steps

### This Week (T-008: Training)

```bash
# 1. Review architecture
vim qa_lab/qa_model_architecture.py

# 2. Setup training
python qa_training_pipeline.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 100 \
    --batch-size 32

# 3. Monitor
tail -f qa_lab/logs/training.log
```

### Next Week (T-009: Integration)

```python
# Load trained model
from qa_lab.qalm_inference import QALM
qalm = QALM.load('checkpoints/qalm_v1.pt')

# Use in pipeline
python qa_theorem_discovery_orchestrator.py \
    --use-qalm \
    --model checkpoints/qalm_v1.pt
```

### Week 3 (T-010: Evaluation)

```bash
# Benchmark
python qa_model_evaluation.py \
    --model qalm_v1 \
    --compare claude,gemini

# Generate report
python generate_research_report.py
```

---

## 💡 Key Innovations

### QAAttention Module
```python
# From qa_model_architecture.py
self.qa_bias_net = nn.Linear(qa_tuple_dim, num_attention_heads)
```
**Purpose:** Computes attention biases from (b,e,d,a) tuples to preserve invariants.

### Multi-AI Architecture
```
User → Claude (orchestrate)
       ↓
    ┌──┴──┐
    ↓     ↓
  QALM  Codex
    ↓     ↓
    └──┬──┘
       ↓
    Claude (synthesize)
       ↓
     Result
```

---

## 🔗 Useful Links

- **QALM Project:** `qa_lab/projects/prj-002-qa-language-model.yaml`
- **Tasks:** `qa_lab/tasks/active/`
- **Architecture:** `qa_lab/qa_model_architecture.py`
- **Rules:** `qa_lab/context/QA_RULES.yaml`
- **Dataset:** `qa_training_dataset.jsonl`

---

## ⚡ One-Liners

```bash
# Task status
./opencode_cli.sh task T-006

# Dataset preview
head -1 qa_training_dataset.jsonl | python -m json.tool

# Count examples by domain
python -c "import json; print(set(json.loads(line)['domain'] for line in open('qa_training_dataset.jsonl')))"

# View QA rules
./opencode_cli.sh rules

# List all agents
./opencode_cli.sh agents

# Architecture summary
./opencode_cli.sh arch | head -30

# Help
./opencode_cli.sh help
```

---

## 📞 Troubleshooting

### OpenCode not responding
```bash
# Check if installed
which opencode

# Test directly
opencode --help
```

### Codex not found
```bash
# Check installation
which codex
codex --help
```

### Dataset issues
```bash
# Regenerate dataset
python collect_qa_training_data.py \
    --vault QAnotes/ \
    --output qa_training_dataset.jsonl

# Verify format
head -1 qa_training_dataset.jsonl | python -m json.tool
```

---

**Version:** 1.0
**Last Updated:** 2025-10-30
**Status:** Integration complete, T-006 finished, ready for training
