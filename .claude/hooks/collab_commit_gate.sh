#!/usr/bin/env bash
# PreToolUse hook (Bash): broadcast commit_intent before git commit
#
# cert_gate_hook.py blocks git commit unless a collab session marker
# exists. This hook then broadcasts commit_intent and blocks only if a
# veto is observed.

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only gate git commit commands
case "$COMMAND" in
  *"git commit"*|*"git add"*"&&"*"git commit"*)
    ;;
  *)
    exit 0  # Not a commit, pass through
    ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="${REPO}/.venv/bin/python"
MARKER="/tmp/qa_collab_session_registered"

# Skip if no bus
if [ -f "${MARKER}.nobus" ] || ! ss -tlnp 2>/dev/null | grep -q ":5556 "; then
  exit 0
fi

SESSION_NAME=$(cat "$MARKER" 2>/dev/null || echo "unknown")

# Broadcast commit intent
INTENT_RESULT=$("$VENV_PYTHON" -c "
import zmq, json, time
ctx = zmq.Context()
req = ctx.socket(zmq.REQ)
req.connect('tcp://127.0.0.1:5556')
req.setsockopt(zmq.RCVTIMEO, 3000)
req.send_string(json.dumps({
    'action': 'publish',
    'event_type': 'commit_intent',
    'data': {
        'session': '${SESSION_NAME}',
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
}))
reply = json.loads(req.recv_string())
req.close()
ctx.term()
print(reply.get('status', 'error'))
" 2>&1 || echo "error")

if [ "$INTENT_RESULT" = "ok" ]; then
  # Wait 2s for veto (non-blocking check)
  sleep 2
  VETO=$("$VENV_PYTHON" -c "
import zmq, json
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect('tcp://127.0.0.1:5555')
sub.setsockopt_string(zmq.SUBSCRIBE, '')
sub.setsockopt(zmq.RCVTIMEO, 100)
try:
    msg = sub.recv_string()
    data = json.loads(msg.split(' ', 1)[1]) if ' ' in msg else json.loads(msg)
    if data.get('payload', {}).get('data', {}).get('event_type') == 'commit_veto':
        print('VETOED')
    else:
        print('ok')
except:
    print('ok')
sub.close()
ctx.term()
" 2>&1 || echo "ok")

  if [ "$VETO" = "VETOED" ]; then
    echo "COMMIT VETOED by another session — check collab bus" >&2
    exit 2
  fi
fi

exit 0
