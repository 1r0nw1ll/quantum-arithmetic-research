# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 1 infrastructure validator -->
"""QA-KG Epistemic Fields Cert [252] v1 validator.

QA_COMPLIANCE = "cert_validator — validates KG epistemic field invariants, no empirical QA state machine"

Gates:
  EF1  (HARD) every node has non-null authority.
  EF2  (HARD) every node has non-null epistemic_status.
  EF3  (HARD) (authority, epistemic_status) ∈ allowed_matrix.json.
  EF4  (HARD) authority=primary ⇒ source_locator present, scheme ∈ {file,pdf,cert},
       and file: scheme resolves to an on-disk file.
  EF5  (WARN) count of authority=agent nodes (pressure sensor for Phase 2).
  EF6  (HARD) node_type=Axiom ⇒ authority=primary ∧ epistemic_status=axiom.
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates KG epistemic field invariants, no empirical QA state machine"

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_CERT_DIR = Path(__file__).resolve().parent
_MATRIX_FILE = _CERT_DIR / "allowed_matrix.json"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _db_path_default() -> Path:
    override = os.environ.get("QA_KG_DB")
    if override:
        return Path(override)
    return _REPO / "tools" / "qa_kg" / "qa_kg.db"


def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise FileNotFoundError(f"QA-KG DB not found at {db}")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _load_allowed_matrix() -> dict[str, list[str]]:
    data = json.loads(_MATRIX_FILE.read_text(encoding="utf-8"))
    return data["allowed"]


def _resolve_source_locator(loc: str) -> tuple[bool, str]:
    """Check that a source_locator resolves.
    Schemes: file:<path>[:<line>], pdf:<path>#page=N, cert:<id>."""
    if not loc:
        return False, "empty source_locator"
    if loc.startswith("file:"):
        path_part = loc[5:].split(":")[0].split("#")[0]
        full = _REPO / path_part
        if full.exists():
            return True, f"file exists: {path_part}"
        return False, f"file not found: {full}"
    if loc.startswith("pdf:"):
        path_part = loc[4:].split("#")[0]
        full = _REPO / path_part
        if full.exists():
            return True, f"pdf exists: {path_part}"
        return False, f"pdf not found: {full}"
    if loc.startswith("cert:"):
        return True, "cert scheme (not file-resolved)"
    return False, f"unknown scheme in {loc!r}"


def check_ef1(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols = [r["id"] for r in conn.execute(
        "SELECT id FROM nodes WHERE authority IS NULL"
    ).fetchall()]
    return len(viols) == 0, f"{len(viols)} node(s) with NULL authority", viols


def check_ef2(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols = [r["id"] for r in conn.execute(
        "SELECT id FROM nodes WHERE epistemic_status IS NULL"
    ).fetchall()]
    return len(viols) == 0, f"{len(viols)} node(s) with NULL epistemic_status", viols


def check_ef3(conn: sqlite3.Connection, matrix: dict[str, list[str]]) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    for r in conn.execute(
        "SELECT id, authority, epistemic_status FROM nodes "
        "WHERE authority IS NOT NULL AND epistemic_status IS NOT NULL"
    ).fetchall():
        auth = r["authority"]
        eps = r["epistemic_status"]
        allowed = matrix.get(auth, [])
        if eps not in allowed:
            viols.append(f"{r['id']}: ({auth},{eps}) not in allowed matrix")
    return len(viols) == 0, f"{len(viols)} matrix violation(s)", viols


def check_ef4(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    for r in conn.execute(
        "SELECT id, source_locator FROM nodes WHERE authority = 'primary'"
    ).fetchall():
        loc = r["source_locator"]
        if not loc:
            viols.append(f"{r['id']}: primary node with no source_locator")
            continue
        scheme = loc.split(":")[0] if ":" in loc else ""
        if scheme not in ("file", "pdf", "cert"):
            viols.append(f"{r['id']}: primary node source_locator scheme={scheme!r} not in {{file,pdf,cert}}")
            continue
        ok, msg = _resolve_source_locator(loc)
        if not ok:
            viols.append(f"{r['id']}: {msg}")
    return len(viols) == 0, f"{len(viols)} primary source_locator failure(s)", viols


def check_ef5(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    count = conn.execute(
        "SELECT COUNT(*) n FROM nodes WHERE authority = 'agent'"
    ).fetchone()["n"]
    return True, f"{count} agent-authority node(s)", []


def check_ef6(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    for r in conn.execute(
        "SELECT id, authority, epistemic_status FROM nodes WHERE node_type = 'Axiom'"
    ).fetchall():
        problems: list[str] = []
        if r["authority"] != "primary":
            problems.append(f"authority={r['authority']!r}≠primary")
        if r["epistemic_status"] != "axiom":
            problems.append(f"epistemic_status={r['epistemic_status']!r}≠axiom")
        if problems:
            viols.append(f"{r['id']}: {'; '.join(problems)}")
    return len(viols) == 0, f"{len(viols)} Axiom-node epistemic violation(s)", viols


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--show-details", action="store_true")
    args = p.parse_args(argv)

    db = args.db or _db_path_default()
    try:
        conn = _connect(db)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr); return 2

    matrix = _load_allowed_matrix()
    hard_fail = False

    def run_bool(code: str, desc: str, fn, is_hard: bool) -> None:
        nonlocal hard_fail
        ok, msg, detail = fn(conn) if not callable(getattr(fn, '__code__', None)) else fn(conn)
        flag = "PASS" if ok else ("FAIL" if is_hard or args.strict else "WARN")
        print(f"[{flag}] {code}  {desc} — {msg}")
        if args.show_details and detail:
            for d in detail[:10]:
                print(f"         {d}")
            if len(detail) > 10:
                print(f"         ... +{len(detail)-10} more")
        if not ok and (is_hard or args.strict):
            hard_fail = True

    run_bool("EF1", "authority non-null", check_ef1, True)
    run_bool("EF2", "epistemic_status non-null", check_ef2, True)
    run_bool("EF3", "authority × epistemic_status matrix",
             lambda c: check_ef3(c, matrix), True)
    run_bool("EF4", "primary source_locator resolves", check_ef4, True)

    ok5, msg5, _ = check_ef5(conn)
    print(f"[INFO] EF5  Agent-authority pressure — {msg5}")

    run_bool("EF6", "Axiom ⇒ primary+axiom", check_ef6, True)

    if hard_fail:
        print("[FAIL] QA-KG epistemic fields cert [252] v1"); return 1
    print("[PASS] QA-KG epistemic fields cert [252] v1"); return 0


if __name__ == "__main__":
    sys.exit(main())
