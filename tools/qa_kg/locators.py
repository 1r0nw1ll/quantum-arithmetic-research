"""Shared source_locator resolver for QA-KG cert validators.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Phase 3 (2026-04-16). Extracted from the inline resolver in [252] EF4
(qa_kg_epistemic_fields_cert_validate.py) so [253] SC1 can share the
same resolution semantics without duplicating — preventing the two from
drifting when Phase 4.5 adds a URL scheme.

Locator grammar:
  file:<path>[:<line>][#<anchor>]   — on-disk file; fragments stripped
  pdf:<path>[#<anchor>]              — on-disk PDF; fragments stripped
  cert:<cert_id>                     — Cert node; existence optionally
                                       verified against a DB connection

`conn` is optional on `resolve_cert_locator` / `resolve_any` so callers
that cannot afford to look up DB nodes (or that want to preserve the
pre-Phase-3 vacuous cert: semantics, as EF4 does) can pass `conn=None`.
When `conn` is provided, cert:<id> resolves to a real node existence check.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sqlite3
from pathlib import Path


_FILE_SCHEME = "file:"
_PDF_SCHEME = "pdf:"
_CERT_SCHEME = "cert:"


def _strip_fragments(path_with_fragments: str) -> str:
    """Strip :line and #anchor suffixes from a file-path locator body."""
    return path_with_fragments.split(":")[0].split("#")[0]


def resolve_file_locator(loc: str, repo_root: Path) -> tuple[bool, str]:
    """Resolve `file:<path>[:<line>][#<anchor>]` to an on-disk file."""
    if not loc.startswith(_FILE_SCHEME):
        return False, f"scheme not 'file:' in {loc!r}"
    path_part = _strip_fragments(loc[len(_FILE_SCHEME):])
    full = repo_root / path_part
    if full.exists():
        return True, f"file exists: {path_part}"
    return False, f"file not found: {full}"


def resolve_pdf_locator(loc: str, repo_root: Path) -> tuple[bool, str]:
    """Resolve `pdf:<path>[#<anchor>]` to an on-disk PDF."""
    if not loc.startswith(_PDF_SCHEME):
        return False, f"scheme not 'pdf:' in {loc!r}"
    path_part = loc[len(_PDF_SCHEME):].split("#")[0]
    full = repo_root / path_part
    if full.exists():
        return True, f"pdf exists: {path_part}"
    return False, f"pdf not found: {full}"


def resolve_cert_locator(
    loc: str,
    conn: sqlite3.Connection | None = None,
) -> tuple[bool, str]:
    """Resolve `cert:<cert_id>` — optionally verifying node existence.

    When `conn` is None (the EF4 call site, preserved behavior), this
    passes vacuously. When `conn` is provided (the SC1 call site),
    checks that a `Cert` node with id `cert:<cert_id>` exists in the DB.
    """
    if not loc.startswith(_CERT_SCHEME):
        return False, f"scheme not 'cert:' in {loc!r}"
    cert_id = loc[len(_CERT_SCHEME):]
    if conn is None:
        return True, "cert scheme (not DB-verified, pass-through)"
    row = conn.execute(
        "SELECT 1 FROM nodes WHERE id = ? AND node_type = 'Cert'",
        (f"cert:{cert_id}",),
    ).fetchone()
    if row is not None:
        return True, f"cert node exists: cert:{cert_id}"
    return False, f"cert node not found: cert:{cert_id}"


def resolve_any(
    loc: str,
    *,
    repo_root: Path,
    conn: sqlite3.Connection | None = None,
) -> tuple[bool, str]:
    """Dispatch on scheme prefix. Returns (ok, detail_msg).

    Empty locator is an error. Unknown scheme is an error.
    """
    if not loc:
        return False, "empty source_locator"
    if loc.startswith(_FILE_SCHEME):
        return resolve_file_locator(loc, repo_root)
    if loc.startswith(_PDF_SCHEME):
        return resolve_pdf_locator(loc, repo_root)
    if loc.startswith(_CERT_SCHEME):
        return resolve_cert_locator(loc, conn)
    return False, f"unknown scheme in {loc!r}"
