#!/usr/bin/env bash
# SessionStart hook: enforce collab bus registration
#
# HARD GATE: Session cannot proceed without registering on the collab bus.
# Creates a session marker file that other hooks check.

set -euo pipefail

REPO="/home/player2/signal_experiments"
MARKER="/tmp/qa_collab_session_registered"
VENV_PYTHON="${REPO}/.venv/bin/python"

# Check if collab bus is running
if ! ss -tlnp 2>/dev/null | grep -q ":5555 "; then
  echo "COLLAB BUS NOT RUNNING — start it: cd qa_lab && bash start_collab_bus.sh"
  echo "Proceeding without bus (hooks will skip bus operations)"
  touch "${MARKER}.nobus"
  exit 0
fi

# Generate session name from branch + timestamp
BRANCH=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
SESSION_NAME="claude-${BRANCH}-$(date +%H%M)"

# Register session on collab bus
REGISTER_RESULT=$("$VENV_PYTHON" -c "
import zmq, json, time
ctx = zmq.Context()
req = ctx.socket(zmq.REQ)
req.connect('tcp://127.0.0.1:5557')
req.setsockopt(zmq.RCVTIMEO, 3000)
payload = json.dumps({
    'action': 'set',
    'key': 'session:${SESSION_NAME}',
    'value': {
        'scope': '${REPO}',
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'pid': $$
    }
})
req.send_string(payload)
reply = json.loads(req.recv_string())
req.close()
ctx.term()
print(reply.get('status', 'error'))
" 2>&1 || echo "error")

if [ "$REGISTER_RESULT" = "ok" ]; then
  echo "$SESSION_NAME" > "$MARKER"
  echo "Collab bus: registered as ${SESSION_NAME}"
else
  echo "Collab bus: registration failed (${REGISTER_RESULT}), proceeding anyway"
  echo "nobus" > "${MARKER}.nobus"
fi

exit 0
