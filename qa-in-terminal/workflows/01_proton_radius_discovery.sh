#!/bin/bash
# Workflow: QA Proton Radius Discovery
# Network Chuck-inspired multi-AI collaboration workflow
#
# This workflow demonstrates how to use the QA Terminal Agent to:
# 1. Generate QA tuples using MCP tools
# 2. Analyze with multiple AI providers
# 3. Maintain persistent context across sessions

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QA_AGENT="${SCRIPT_DIR}/../qa_terminal_agent.py"
CONTEXT_FILE="qa_lab/qa_contexts/proton_radius.yaml"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  QA Proton Radius Discovery Workflow                      ║"
echo "║  Multi-AI Collaboration via Terminal Agent                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# Step 1: Verify MCP servers and context
echo "📋 Step 1: Verify Environment"
echo "----------------------------------------"
python3 "${QA_AGENT}" --show-context -c "${CONTEXT_FILE}"
echo

# Step 2: Use MCP to compute Fibonacci QA tuple (1,2)
echo "🔧 Step 2: Compute Fibonacci Seed Tuple via MCP"
echo "----------------------------------------"
echo "Computing QA tuple (1, 2, 3, 5)..."
python3 "${QA_AGENT}" --mcp qa_compute_triangle --mcp-args '{"b": 1.0, "e": 2.0}' -c "${CONTEXT_FILE}"
echo

# Step 3: Use MCP to compute high-resonance tuple (3,5)
echo "🔧 Step 3: Compute High-Resonance Tuple via MCP"
echo "----------------------------------------"
echo "Computing QA tuple (3, 5, 8, 13)..."
python3 "${QA_AGENT}" --mcp qa_compute_triangle --mcp-args '{"b": 3.0, "e": 5.0}' -c "${CONTEXT_FILE}"
echo

# Step 4: Analyze with Claude Code
echo "🤖 Step 4: Analyze Ellipse Properties (Claude)"
echo "----------------------------------------"
cat << 'EOF'
You can now run an interactive session with Claude to analyze the results:

python3 qa-in-terminal/qa_terminal_agent.py -c qa_lab/qa_contexts/proton_radius.yaml -p claude

Then ask:
"Based on the QA tuples (1,2,3,5) and (3,5,8,13), analyze their ellipse properties
and compare the scaled radii to the CODATA proton charge radius of 0.8414 fm."

The agent will:
- Load the proton_radius.yaml context with all previous work
- Provide QA-aware analysis using the system prompt
- Save the response back to the context file
EOF
echo

# Step 5: Validation workflow suggestion
echo "✅ Step 5: Next Steps for Multi-AI Validation"
echo "----------------------------------------"
cat << 'EOF'
To complete the workflow, you can:

1. **Gemini Analysis** - Review the mathematical rigor:
   python3 qa-in-terminal/qa_terminal_agent.py \
     -c qa_lab/qa_contexts/proton_radius.yaml -p gemini \
     "Review the QA ellipse quantization approach. Is the math sound?"

2. **QALM Verification** - Cross-check with local QA model:
   python3 qa-in-terminal/qa_terminal_agent.py \
     -c qa_lab/qa_contexts/proton_radius.yaml -p qalm \
     "Verify that tuples (1,2,3,5) and (3,5,8,13) satisfy all QA invariants"

3. **Codex Implementation** - Generate validation code:
   python3 qa-in-terminal/qa_terminal_agent.py \
     -c qa_lab/qa_contexts/proton_radius.yaml -p codex \
     "Generate Python code to systematically scan all mod-24 QA tuples
      and rank by proximity to proton radius when scaled appropriately"

All interactions are saved to: qa_lab/qa_contexts/proton_radius.yaml
EOF
echo

# Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Workflow Complete!                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo
echo "Context file: ${CONTEXT_FILE}"
echo "Chat history preserved for future sessions."
echo "Run 'python3 qa-in-terminal/qa_terminal_agent.py -c ${CONTEXT_FILE}' to continue."
