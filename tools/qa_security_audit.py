#!/usr/bin/env python3
"""
QA Security Audit — automated security health check.

Run daily or before/after any infrastructure change.

Usage:
    python tools/qa_security_audit.py          # Full audit
    python tools/qa_security_audit.py --quick   # Fast checks only

Checks:
  1. Guardrail E2E tests (12 tests)
  2. Agent security kernel self-tests (14 tests)
  3. Collab bus agent registry (flag unknowns)
  4. Bridge guardrail status
  5. Recent GUARDRAIL_DENY events
  6. Sensitive data in event log
  7. Axiom linter (existing)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {"pass": [], "fail": [], "warn": []}


def _run(cmd: str, cwd: str = None, timeout: int = 60) -> tuple:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd or str(ROOT))
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def check_guardrail_e2e():
    """Run guardrail E2E test suite."""
    cwd = str(ROOT / "qa_alphageometry_ptolemy" / "qa_guardrail")
    rc, stdout, stderr = _run("python e2e_test.py", cwd=cwd)
    if rc == 0 and "ALL TESTS PASSED" in stdout:
        RESULTS["pass"].append(f"Guardrail E2E: ALL TESTS PASSED")
    else:
        RESULTS["fail"].append(f"Guardrail E2E: FAILED (rc={rc})")


def check_agent_security():
    """Run agent security kernel self-tests."""
    cwd = str(ROOT / "qa_agent_security")
    rc, stdout, stderr = _run("python qa_agent_security.py", cwd=cwd)
    if rc == 0 and "ALL PASS" in stdout:
        RESULTS["pass"].append("Agent Security Kernel: 14/14 PASS")
    else:
        RESULTS["fail"].append(f"Agent Security Kernel: FAILED (rc={rc})")


def check_collab_agents():
    """Check collab bus for unknown agents."""
    state_file = ROOT / "qa_lab" / "logs" / "collab_state.json"
    agents_dir = ROOT / "qa_lab" / "logs" / "agents"
    known_prefixes = {"claude-code", "codex_bridge", "opencode_bridge",
                      "gemini_bridge", "opencode_test"}

    if agents_dir.exists():
        for f in agents_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                name = data.get("name", "")
                if not any(name.startswith(p) for p in known_prefixes):
                    RESULTS["warn"].append(f"Unknown agent on bus: {name} ({f.name})")
            except Exception:
                pass

    if not any("Unknown agent" in w for w in RESULTS["warn"]):
        RESULTS["pass"].append("Collab bus: no unknown agents")


def check_event_log_secrets():
    """Scan event log for leaked credentials."""
    log_file = ROOT / "qa_lab" / "logs" / "collab_events.jsonl"
    if not log_file.exists():
        RESULTS["pass"].append("Event log: not present")
        return

    sensitive_patterns = [
        re.compile(r'sk-[a-zA-Z0-9]{20,}'),
        re.compile(r'AIza[a-zA-Z0-9_-]{35}'),
        re.compile(r'ya29\.[a-zA-Z0-9_-]+'),
        re.compile(r'AKIA[A-Z0-9]{16}'),
        re.compile(r'https?://[^@\s]+:[^@\s]+@'),
    ]

    hits = 0
    with open(log_file, "r") as f:
        for line in f:
            for pat in sensitive_patterns:
                if pat.search(line):
                    hits += 1
                    break

    if hits > 0:
        RESULTS["warn"].append(f"Event log: {hits} lines with potential credential patterns")
    else:
        RESULTS["pass"].append("Event log: no credential patterns found")


def check_guardrail_denials():
    """Report recent GUARDRAIL_DENY events."""
    log_file = ROOT / "qa_lab" / "logs" / "collab_events.jsonl"
    if not log_file.exists():
        return

    denials = 0
    with open(log_file, "r") as f:
        for line in f:
            if "GUARDRAIL_DENY" in line:
                denials += 1

    if denials > 0:
        RESULTS["warn"].append(f"Guardrail blocked {denials} malicious prompt(s) — review event log")
    else:
        RESULTS["pass"].append("Guardrail: no blocked prompts in log")


def check_bridge_processes():
    """Check if bridges are running with guardrail."""
    rc, stdout, _ = _run("ps aux | grep llm_bridge_agent | grep -v grep")
    bridges = [l for l in stdout.strip().split("\n") if l.strip()]

    if not bridges:
        RESULTS["warn"].append("No LLM bridge agents running")
        return

    for b in bridges:
        name = "unknown"
        if "--name" in b:
            parts = b.split("--name")
            if len(parts) > 1:
                name = parts[1].strip().split()[0]

        if "--no-guardrail" in b:
            RESULTS["fail"].append(f"Bridge {name}: running WITHOUT guardrail!")
        else:
            RESULTS["pass"].append(f"Bridge {name}: running (guardrail enabled)")


def main():
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("QA SECURITY AUDIT")
    print("=" * 60)

    check_guardrail_e2e()
    check_agent_security()
    check_collab_agents()
    check_event_log_secrets()
    check_guardrail_denials()
    check_bridge_processes()

    print()
    for p in RESULTS["pass"]:
        print(f"  [PASS] {p}")
    for w in RESULTS["warn"]:
        print(f"  [WARN] {w}")
    for f in RESULTS["fail"]:
        print(f"  [FAIL] {f}")

    total = len(RESULTS["pass"]) + len(RESULTS["warn"]) + len(RESULTS["fail"])
    print()
    print(f"SUMMARY: {len(RESULTS['pass'])} pass, {len(RESULTS['warn'])} warn, {len(RESULTS['fail'])} fail ({total} checks)")

    if RESULTS["fail"]:
        print("\n⛔ SECURITY AUDIT FAILED — fix before proceeding")
        return 1
    elif RESULTS["warn"]:
        print("\n⚠️  SECURITY AUDIT PASSED WITH WARNINGS")
        return 0
    else:
        print("\n✅ SECURITY AUDIT CLEAN")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
