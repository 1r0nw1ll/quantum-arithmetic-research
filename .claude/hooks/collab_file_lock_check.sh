#!/usr/bin/env bash
# PreToolUse hook (Edit|Write): check file locks on shared files
#
# HARD GATE on shared files listed in CLAUDE.md:
# - CLAUDE.md, MEMORY.md, AGENTS.md
# - qa_alphageometry_ptolemy/qa_meta_validator.py
# - docs/families/README.md
#
# If another session holds a lock < 5 min old, BLOCK.
# If lock is stale (> 5 min) or no lock, proceed.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0

REPO="/home/player2/signal_experiments"
VENV_PYTHON="${REPO}/.venv/bin/python"
MARKER="/tmp/qa_collab_session_registered"
SESSION_NAME=$(cat "$MARKER" 2>/dev/null || echo "unknown")

# Only check shared files that ALWAYS require locking
REL=$(realpath --relative-to="$REPO" "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")
case "$REL" in
  CLAUDE.md|MEMORY.md|AGENTS.md|\
  qa_alphageometry_ptolemy/qa_meta_validator.py|\
  docs/families/README.md)
    # These require lock checking
    ;;
  *)
    exit 0  # Not a shared file, no lock needed
    ;;
esac

# Check if bus is running
if [ -f "${MARKER}.nobus" ]; then
  exit 0  # No bus, skip lock check
fi

if ! ss -tlnp 2>/dev/null | grep -q ":5557 "; then
  exit 0  # State port not listening
fi

# Check lock
LOCK_CHECK=$("$VENV_PYTHON" -c "
import zmq, json, time
ctx = zmq.Context()
req = ctx.socket(zmq.REQ)
req.connect('tcp://127.0.0.1:5557')
req.setsockopt(zmq.RCVTIMEO, 2000)
req.send_string(json.dumps({'action': 'get', 'key': 'file_locks'}))
reply = json.loads(req.recv_string())
req.close()
ctx.term()

locks = reply.get('value') or {}
rel = '${REL}'
if rel not in locks:
    print('ok')
else:
    lock = locks[rel]
    lock_session = lock.get('session', '')
    lock_ts = lock.get('ts', '')
    if lock_session == '${SESSION_NAME}':
        print('ok')  # Our own lock
    else:
        # Check staleness (> 5 min = stale)
        try:
            from datetime import datetime
            lock_time = datetime.fromisoformat(lock_ts.replace('Z', '+00:00'))
            now = datetime.now(lock_time.tzinfo)
            age_s = (now - lock_time).total_seconds()
            if age_s > 300:
                print('ok')  # Stale lock
            else:
                print(f'LOCKED by {lock_session} ({int(age_s)}s ago)')
        except Exception:
            print('ok')  # Can't parse, allow
" 2>&1 || echo "ok")

if [[ "$LOCK_CHECK" == LOCKED* ]]; then
  echo "FILE LOCK: $REL is $LOCK_CHECK" >&2
  echo "Wait for the other session to release, or check collab bus state" >&2
  exit 2  # BLOCK
fi

# Acquire lock for this file
"$VENV_PYTHON" -c "
import zmq, json, time
ctx = zmq.Context()
req = ctx.socket(zmq.REQ)
req.connect('tcp://127.0.0.1:5557')
req.setsockopt(zmq.RCVTIMEO, 2000)

# Get current locks
req.send_string(json.dumps({'action': 'get', 'key': 'file_locks'}))
reply = json.loads(req.recv_string())
locks = reply.get('value') or {}

# Add our lock
locks['${REL}'] = {
    'session': '${SESSION_NAME}',
    'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
}

req.send_string(json.dumps({'action': 'set', 'key': 'file_locks', 'value': locks}))
req.recv_string()
req.close()
ctx.term()
" 2>/dev/null || true

exit 0
