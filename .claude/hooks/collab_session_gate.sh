#!/usr/bin/env bash
# SessionStart hook: collab bus heartbeat + optional registration
#
# HEARTBEAT ONLY (2026-04-12): report bus status, never block.
# If bus is up, register. If bus is down, say so and move on.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARKER="/tmp/qa_collab_session_registered"
VENV_PYTHON="${REPO}/.venv/bin/python"

# Heartbeat: is the bus listening?
if ! ss -tlnp 2>/dev/null | grep -q ":5555 "; then
  echo "Collab bus: not running (port 5555 down)"
  rm -f "${MARKER}"
  exit 0
fi

# Bus is up — try to register
BRANCH=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
SESSION_NAME="claude-${BRANCH}-$(date +%H%M)"

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
  echo "Collab bus: registration failed (${REGISTER_RESULT}), continuing"
fi

exit 0
