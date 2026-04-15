"""Axiom predicates — invoke the QA axiom linter to re-verify each axiom.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Each check runs the linter and filters for rules matching the axiom code.
If the linter is absent or errors, returns (False, reason) — diagnostic,
not fatal.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LINTER = REPO / "tools" / "qa_axiom_linter.py"


def _run_linter() -> tuple[int, str]:
    if not LINTER.exists():
        return -1, f"linter missing at {LINTER}"
    try:
        proc = subprocess.run(
            ["python3", str(LINTER), "--all"],
            capture_output=True, text=True, cwd=str(REPO), timeout=180,
        )
        return proc.returncode, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        return -2, "linter timeout"
    except Exception as exc:  # noqa: BLE001
        return -3, f"{type(exc).__name__}: {exc}"


def _check_rule(code: str) -> tuple[bool, str]:
    rc, out = _run_linter()
    if rc < 0:
        return False, out
    # Linter exits non-zero if ANY rule fails; we parse for this code's hits.
    hits = [ln for ln in out.splitlines() if f"[{code}" in ln or f" {code}-" in ln or f":{code}:" in ln]
    if hits:
        return False, f"{len(hits)} {code} violation(s)"
    return True, f"{code} clean"


def check_a1() -> tuple[bool, str]: return _check_rule("A1")
def check_a2() -> tuple[bool, str]: return _check_rule("A2")
def check_t2() -> tuple[bool, str]: return _check_rule("T2")
def check_s1() -> tuple[bool, str]: return _check_rule("S1")
def check_s2() -> tuple[bool, str]: return _check_rule("S2")
def check_t1() -> tuple[bool, str]: return _check_rule("T1")


def check_nt() -> tuple[bool, str]:
    """Theorem NT — union of T2 + T1 + A1 checks."""
    ok_t2, m_t2 = check_t2()
    ok_t1, m_t1 = check_t1()
    ok_a1, m_a1 = check_a1()
    ok = ok_t2 and ok_t1 and ok_a1
    return ok, f"T2:{m_t2}; T1:{m_t1}; A1:{m_a1}"
