#!/usr/bin/env python3
"""
qa_radionics_ledger_sanity.py

Stdlib-only sanity checker for qa_ledger__radionics_obstructions.v1.yaml

Validates:
  - obstruction_id present and unique
  - fail_type present and non-empty
  - class present and non-empty
  - (optional warnings) suggested_fix present

This is YAML-lite: it expects ledger entries like:
  - obstruction_id: "..."
    class: "..."
    fail_type: "..."

Usage:
    python qa_radionics_ledger_sanity.py --ledger qa_ledger__radionics_obstructions.v1.yaml

Exit codes:
    0  ok
    2  invalid
    3  runtime error
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

RE_OBS = re.compile(
    r'^\s*-?\s*obstruction_id:\s*"([^"]+)"\s*$|'
    r"^\s*-?\s*obstruction_id:\s*'([^']+)'\s*$|"
    r'^\s*-?\s*obstruction_id:\s*([A-Za-z0-9_\-\.]+)\s*$'
)
RE_FAIL = re.compile(
    r'^\s*fail_type:\s*"([^"]*)"\s*$|'
    r"^\s*fail_type:\s*'([^']*)'\s*$|"
    r'^\s*fail_type:\s*([A-Za-z0-9_\-\.]+)\s*$'
)
RE_CLASS = re.compile(
    r'^\s*class:\s*"([^"]*)"\s*$|'
    r"^\s*class:\s*'([^']*)'\s*$|"
    r'^\s*class:\s*([A-Za-z0-9_\-\.]+)\s*$'
)
RE_SUGGEST = re.compile(r'^\s*suggested_fix:\s*$')


@dataclass
class Entry:
    obstruction_id: str
    fail_type: Optional[str] = None
    cls: Optional[str] = None
    has_suggested_fix: bool = False
    line_no: int = 0


def _pick(m: re.Match) -> str:
    for i in range(1, 4):
        g = m.group(i)
        if g is not None:
            return g.strip()
    return ""


def parse_entries(text: str) -> List[Entry]:
    """
    Extremely small parser: detects a new entry when obstruction_id seen.
    Captures fail_type/class under that entry by subsequent lines until next obstruction_id.
    """
    entries: List[Entry] = []
    cur: Optional[Entry] = None

    for idx, line in enumerate(text.splitlines(), start=1):
        m_obs = RE_OBS.match(line)
        if m_obs:
            if cur is not None:
                entries.append(cur)
            cur = Entry(obstruction_id=_pick(m_obs), line_no=idx)
            continue

        if cur is None:
            continue

        m_fail = RE_FAIL.match(line)
        if m_fail:
            cur.fail_type = _pick(m_fail)
            continue

        m_cls = RE_CLASS.match(line)
        if m_cls:
            cur.cls = _pick(m_cls)
            continue

        if RE_SUGGEST.match(line):
            cur.has_suggested_fix = True

    if cur is not None:
        entries.append(cur)

    return entries


def sanity_check(text: str) -> Tuple[bool, List[str], List[str]]:
    """Returns (ok, errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []

    entries = parse_entries(text)

    if not entries:
        errors.append("No entries found (no obstruction_id detected).")
        return (False, errors, warnings)

    seen: Dict[str, int] = {}
    for e in entries:
        oid = e.obstruction_id
        if not oid:
            errors.append(f"Entry at line {e.line_no}: empty obstruction_id.")
            continue

        if oid in seen:
            errors.append(
                f"Duplicate obstruction_id '{oid}' (first at line {seen[oid]}, again at line {e.line_no})."
            )
        else:
            seen[oid] = e.line_no

        if e.fail_type is None or e.fail_type.strip() == "":
            errors.append(f"obstruction_id '{oid}' (line {e.line_no}): missing/empty fail_type.")

        if e.cls is None or e.cls.strip() == "":
            errors.append(f"obstruction_id '{oid}' (line {e.line_no}): missing/empty class.")

        if not e.has_suggested_fix:
            warnings.append(f"obstruction_id '{oid}' (line {e.line_no}): suggested_fix missing (recommended).")

    ok = len(errors) == 0
    return (ok, errors, warnings)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sanity check radionics obstruction ledger")
    ap.add_argument("--ledger", required=True, help="Path to obstruction ledger YAML")
    args = ap.parse_args(argv)

    try:
        with open(args.ledger, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"ERROR: cannot read ledger: {e}", file=sys.stderr)
        return 3

    ok, errors, warnings = sanity_check(text)

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if ok:
        print(f"OK: ledger sanity check passed ({len(parse_entries(text))} entries).")
        return 0

    print("INVALID: ledger sanity check failed.")
    for e in errors:
        print(f"  - {e}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
