#!/bin/bash
# QA Theorem Discovery - Simple Launcher
# Makes it easy to run any component of the system

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_menu() {
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║           QA AUTOMATED THEOREM DISCOVERY SYSTEM                  ║"
    echo "║                          v2.0                                    ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Choose what to run:"
    echo ""
    echo "  QUICK OPTIONS:"
    echo "    1) Full pipeline (quick mode, ~15 min)"
    echo "    2) Full pipeline (production mode, ~30 min)"
    echo "    3) Test multi-AI collaboration"
    echo ""
    echo "  INDIVIDUAL AGENTS:"
    echo "    4) Graph Builder only"
    echo "    5) GNN Trainer only"
    echo "    6) Conjecture Miner only"
    echo "    7) Lean Verifier only"
    echo ""
    echo "  MULTI-AI COLLABORATION:"
    echo "    8) Run multi-AI orchestrator"
    echo "    9) Test Codex + Gemini integration"
    echo ""
    echo "  DOCUMENTATION:"
    echo "    d) View documentation (opens in less)"
    echo "    h) Show help"
    echo ""
    echo "    q) Quit"
    echo ""
    echo -n "Enter choice: "
}

run_full_quick() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Running FULL PIPELINE (Quick Mode)"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python qa_theorem_discovery_orchestrator.py --quick
}

run_full_production() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Running FULL PIPELINE (Production Mode)"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python qa_theorem_discovery_orchestrator.py
}

run_graph_builder() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Running GRAPH BUILDER"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python qa_graph_builder_v2.py
}

run_gnn_trainer() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Running GNN TRAINER"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python qa_gnn_trainer_v2.py --epochs 100
}

run_miner() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Running CONJECTURE MINER"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python qa_symbolic_miner_v2.py
}

run_verifier() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Running LEAN VERIFIER"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python qa_lean_verifier_v2.py
}

run_multi_ai() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Running MULTI-AI ORCHESTRATOR"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python qa_multi_ai_orchestrator.py
}

test_multi_ai() {
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Testing MULTI-AI COLLABORATION"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python test_multi_ai_collaboration.py
}

show_docs() {
    echo "Select documentation to view:"
    echo "  1) Complete System Overview"
    echo "  2) Quick Start Guide"
    echo "  3) Pipeline README"
    echo "  4) Multi-AI Collaboration Guide"
    echo "  5) Chronological Sweep (research history)"
    echo ""
    echo -n "Enter choice (or q to cancel): "
    read -r doc_choice

    case $doc_choice in
        1) less COMPLETE_SYSTEM_OVERVIEW.md ;;
        2) less QUICKSTART.md ;;
        3) less QA_PIPELINE_README.md ;;
        4) less MULTI_AI_COLLABORATION_GUIDE.md ;;
        5) less QA_VAULT_CHRONOLOGICAL_SWEEP.md ;;
        q|Q) return ;;
        *) echo "Invalid choice" ;;
    esac
}

show_help() {
    cat << 'EOF'
QA THEOREM DISCOVERY SYSTEM - HELP

QUICK START:
  Option 1 or 2 runs the complete pipeline from start to finish

INDIVIDUAL AGENTS:
  Run specific components if you want to test or debug them separately

MULTI-AI:
  Demonstrates Claude + Codex + Gemini collaboration
  Requires Codex and Gemini CLI to be installed

FILES CREATED:
  qa_discovery_workspace/  - Full pipeline outputs
  multi_ai_workspace/      - Multi-AI collaboration outputs

LOGS:
  All components log to stdout in real-time
  Training history saved to checkpoints/training_history.json

REQUIREMENTS:
  - Python 3.10+
  - PyTorch
  - torch-geometric
  - pandas, numpy, scikit-learn
  - tqdm (for progress bars)
  - sympy (for symbolic math)

OPTIONAL:
  - Lean 4 (for formal verification)
  - Codex CLI (for multi-AI mode)
  - Gemini CLI (for multi-AI mode)

For detailed documentation, choose option 'd' from main menu.

Press Enter to continue...
EOF
    read -r
}

# Main loop
while true; do
    clear
    show_menu
    read -r choice

    case $choice in
        1) run_full_quick ;;
        2) run_full_production ;;
        3) test_multi_ai ;;
        4) run_graph_builder ;;
        5) run_gnn_trainer ;;
        6) run_miner ;;
        7) run_verifier ;;
        8) run_multi_ai ;;
        9) test_multi_ai ;;
        d|D) show_docs ;;
        h|H) show_help ;;
        q|Q) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid choice. Press Enter to continue..."; read -r ;;
    esac

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Press Enter to return to menu..."
    read -r
done
