"""Claim predicate runtime.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Canonical claims must be runnable. A predicate_ref is a dotted path
"module.path:callable" resolving to a zero-arg callable returning
bool or (bool, str). Results are cached in nodes.last_check_* columns.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import importlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class CheckResult:
    ok: bool
    msg: str


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(predicate_ref: str) -> Callable[[], object]:
    """Resolve 'module.sub:callable' to the callable."""
    if ":" not in predicate_ref:
        raise ValueError(f"predicate_ref missing ':' — got {predicate_ref!r}")
    mod_path, attr = predicate_ref.split(":", 1)
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{predicate_ref} is not callable")
    return fn


def run(predicate_ref: str) -> CheckResult:
    try:
        fn = resolve(predicate_ref)
        out = fn()
    except Exception as exc:  # noqa: BLE001 — predicate failures must not crash runtime
        return CheckResult(False, f"{type(exc).__name__}: {exc}")
    if isinstance(out, tuple) and len(out) == 2:
        ok, msg = out
        return CheckResult(bool(ok), str(msg))
    return CheckResult(bool(out), "")


def check_node(conn: sqlite3.Connection, node_id: str) -> CheckResult:
    row = conn.execute(
        "SELECT predicate_ref FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if row is None:
        return CheckResult(False, f"node {node_id!r} not found")
    ref = row["predicate_ref"]
    if not ref:
        return CheckResult(True, "no predicate (n/a)")
    result = run(ref)
    conn.execute(
        "UPDATE nodes SET last_check_ts=?, last_check_ok=?, last_check_msg=?, updated_ts=? WHERE id=?",
        (_now(), 1 if result.ok else 0, result.msg, _now(), node_id),
    )
    conn.commit()
    return result


def check_all(conn: sqlite3.Connection, tier: str | None = None) -> dict[str, CheckResult]:
    q = "SELECT id FROM nodes WHERE predicate_ref != ''"
    args: tuple = ()
    if tier:
        q += " AND tier = ?"
        args = (tier,)
    ids = [r["id"] for r in conn.execute(q, args).fetchall()]
    return {nid: check_node(conn, nid) for nid in ids}
