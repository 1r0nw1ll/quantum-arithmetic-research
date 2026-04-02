#!/bin/bash
# OpenCode CLI Helper
# Quick commands for interacting with OpenCode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT="$SCRIPT_DIR/opencode_agent.py"

case "$1" in
    status|s)
        echo "🔍 Checking OpenCode status..."
        python - <<'PY'
import json
from pathlib import Path

log_path = Path("qa_lab/logs/agent_runs.jsonl")
tasks_dir = Path("qa_lab/tasks/active")

if log_path.exists():
    print("\n🗂️  Recent agent activity:")
    events = []
    for line in log_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for event in events[-5:]:
        timestamp = event.get("timestamp", "unknown")
        agent = event.get("agent", "unknown")
        action = event.get("action", "unknown")
        details = [
            f"{key}={value}"
            for key, value in event.items()
            if key not in {"timestamp", "agent", "action"}
        ]
        detail_str = ", ".join(details) if details else "no additional details"
        print(f"  - {timestamp} | {agent} | {action} | {detail_str}")
else:
    print("No agent activity log found (qa_lab/logs/agent_runs.jsonl)")

print("\n📌 Active tasks:")
if tasks_dir.exists():
    active = sorted(p.stem for p in tasks_dir.glob("*.yaml"))
    if active:
        for task in active:
            print(f"  - {task}")
    else:
        print("  (none)")
else:
    print("  Task directory missing (qa_lab/tasks/active)")
PY
        ;;

    list|ls|l)
        echo "📋 Listing OpenCode work..."
        python "$AGENT" --list
        ;;

    tasks|t)
        echo "📝 Active QA Lab tasks:"
        echo
        find qa_lab/tasks/active -name "*.yaml" -exec basename {} .yaml \; | sort
        echo
        echo "To view a task: opencode task <ID>"
        ;;

    task)
        if [ -z "$2" ]; then
            echo "Usage: opencode task <ID>"
            exit 1
        fi
        task_file="qa_lab/tasks/active/$2.yaml"
        if [ -f "$task_file" ]; then
            cat "$task_file"
        else
            echo "❌ Task $2 not found in active tasks"
        fi
        ;;

    project|p)
        if [ -z "$2" ]; then
            echo "📁 Available projects:"
            ls qa_lab/projects/
        else
            cat "qa_lab/projects/$2"
        fi
        ;;

    ask|query|q)
        shift
        if [ -z "$1" ]; then
            echo "Usage: opencode ask <question>"
            exit 1
        fi
        python "$AGENT" "$@"
        ;;

    arch|architecture)
        echo "🏗️  QA Model Architecture:"
        head -150 qa_lab/qa_model_architecture.py | grep -E "^(class|def|    #)" | head -50
        ;;

    eval|evaluation)
        echo "📊 Latest Evaluation Results:"
        if [ -f "qa_lab/evaluation_report.json" ]; then
            python -c "import json; r=json.load(open('qa_lab/evaluation_report.json')); print(json.dumps(r, indent=2))" | head -50
        else
            echo "No evaluation report found"
        fi
        ;;

    rules)
        echo "📜 QA Core Rules:"
        cat qa_lab/context/QA_RULES.yaml
        ;;

    agents)
        echo "🤖 Available QA Agents:"
        ls -1 qa_lab/qa_agents/prompts/*.yaml | xargs -I {} basename {} .system.yaml
        ;;

    graphrag|grag|rag)
        shift
        if [ -z "$1" ]; then
            echo "Usage: opencode graphrag <query>"
            echo "Example: opencode graphrag 'What is Harmonic Index?'"
            exit 1
        fi
        echo "🧠 Querying GraphRAG..."
        python "$AGENT" --graphrag "$@"
        ;;

    graphrag-entity|grag-entity|rag-entity)
        if [ -z "$2" ]; then
            echo "Usage: opencode graphrag-entity <entity_name>"
            echo "Example: opencode graphrag-entity 'Harmonic Index'"
            exit 1
        fi
        echo "🔍 Getting GraphRAG entity details..."
        python "$AGENT" --graphrag-entity "$2"
        ;;

    graphrag-stats|grag-stats|rag-stats)
        echo "📊 GraphRAG Statistics:"
        python "$AGENT" --graphrag-stats
        ;;

    codex|cx)
        shift
        if [ -z "$1" ]; then
            echo "Usage: opencode codex <prompt>"
            exit 1
        fi
        echo "🔧 Codex generating code..."
        python -c "from opencode_agent import CodexAgent; agent=CodexAgent(); print(agent.generate_code('$*'))"
        ;;

    dataset|data)
        if [ -f "qa_training_dataset.jsonl" ]; then
            echo "📊 Dataset Statistics:"
            wc -l qa_training_dataset.jsonl
            du -h qa_training_dataset.jsonl
        else
            echo "❌ Dataset not found: qa_training_dataset.jsonl"
        fi
        ;;

    help|h|--help|-h)
        cat << 'EOF'
OpenCode CLI Helper

Usage: opencode <command> [args]

Commands:
  status, s              - Check OpenCode status
  list, ls, l           - List recent work
  tasks, t              - List active tasks
  task <ID>             - View specific task details
  project, p [name]     - View projects
  ask, query, q <text>  - Ask OpenCode a question
  codex, cx <prompt>    - Generate code with Codex
  arch, architecture    - View model architecture
  eval, evaluation      - Show latest evaluation results
   dataset, data         - Show dataset statistics
   rules                 - View QA core rules
   agents                - List available agents
   graphrag, rag <query> - Query GraphRAG knowledge graph
   graphrag-entity <name>- Get entity details from GraphRAG
   graphrag-stats        - Show GraphRAG statistics
   help, h               - Show this help

Examples:
  opencode status
  opencode task T-006
  opencode ask "What is the QAAttention module?"
   opencode codex "Generate QA tuple validator"
   opencode dataset
   opencode project prj-002-qa-language-model.yaml
   opencode graphrag "What is Harmonic Index?"
   opencode graphrag-entity "QA tuple"
   opencode graphrag-stats

For detailed OpenCode work, see:
  OPENCODE_INTEGRATION_SUMMARY.md
EOF
        ;;

    *)
        echo "Unknown command: $1"
        echo "Run 'opencode help' for usage"
        exit 1
        ;;
esac
