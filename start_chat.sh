#!/bin/bash
set -euo pipefail

cd /home/player2/signal_experiments/qa_lab

if [[ "${QA_ALLOW_NO_OPEN_BRAIN:-0}" != "1" ]]; then
  python3 open_brain_bootstrap.py --quiet
fi

python3 collab_chat_simple.py "$@"
