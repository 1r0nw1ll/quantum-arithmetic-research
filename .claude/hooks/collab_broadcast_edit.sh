#!/usr/bin/env bash
# PostToolUse hook (Edit|Write): broadcast file changes to collab bus
#
# Non-blocking — fires after every edit but never blocks.
# Other sessions see the broadcast and can react.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0

REPO="/home/player2/signal_experiments"
VENV_PYTHON="${REPO}/.venv/bin/python"
MARKER="/tmp/qa_collab_session_registered"

# Skip if no bus
if [ -f "${MARKER}.nobus" ] || ! ss -tlnp 2>/dev/null | grep -q ":5556 "; then
  exit 0
fi

SESSION_NAME=$(cat "$MARKER" 2>/dev/null || echo "unknown")
REL=$(realpath --relative-to="$REPO" "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# Skip files outside repo
case "$REL" in
  /*|..) exit 0 ;;
esac

# Broadcast (fire and forget, non-blocking)
"$VENV_PYTHON" -c "
import zmq, json, time
ctx = zmq.Context()
req = ctx.socket(zmq.REQ)
req.connect('tcp://127.0.0.1:5556')
req.setsockopt(zmq.RCVTIMEO, 1000)
req.send_string(json.dumps({
    'action': 'publish',
    'event_type': 'file_updated',
    'data': {
        'session': '${SESSION_NAME}',
        'file': '${REL}',
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
}))
req.recv_string()
req.close()
ctx.term()
" 2>/dev/null || true

exit 0
