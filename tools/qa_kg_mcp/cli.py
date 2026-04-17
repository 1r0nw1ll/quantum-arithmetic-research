#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 MCP operator CLI -->
"""QA-KG Phase 6 MCP operator CLI.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Operator-facing commands:

  list-writes                  Dump _agent_writes.json contents.
  reset-writes <session>       Clear the counter for a crashed session.

`reset-writes` is the human-authorized recovery path for sessions that
crashed before broadcasting session_done. See README.md.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.qa_kg_mcp import rate_limit as _rl


def _cmd_list_writes(args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else _rl._default_path()
    if not path.exists():
        print(f"{path} does not exist — counter is empty")
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def _cmd_reset_writes(args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else None
    found = _rl.reset_session(args.session, ledger_path=path)
    if found:
        print(f"reset session {args.session!r}")
        return 0
    print(f"session {args.session!r} not found in counter")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA-KG Phase 6 MCP CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-writes", help="Dump _agent_writes.json.")
    p_list.add_argument("--path", default=None)
    p_list.set_defaults(func=_cmd_list_writes)

    p_reset = sub.add_parser("reset-writes",
                             help="Clear the counter for a crashed session.")
    p_reset.add_argument("session")
    p_reset.add_argument("--path", default=None)
    p_reset.set_defaults(func=_cmd_reset_writes)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
