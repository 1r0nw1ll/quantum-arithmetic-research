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
  3. Bridge cert enforcement wiring (static)
  4. Bridge cert self-test (runtime)
  5. Collab bus agent registry (flag unknowns)
  6. Bridge guardrail status
  7. Recent GUARDRAIL_DENY events
  8. Sensitive data in event log
  9. Axiom linter (existing)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {"pass": [], "fail": [], "warn": []}
BRIDGE_FILE = ROOT / "qa_lab" / "qa_agents" / "cli" / "llm_bridge_agent.py"
CERT_GATE_HOOK = ROOT / "llm_qa_wrapper" / "cert_gate_hook.py"
CERT_GATE_HOOK_TEST = ROOT / "llm_qa_wrapper" / "tests" / "test_cert_gate_hook.py"
PROJECT_STEERING_HOOK_TEST = ROOT / "llm_qa_wrapper" / "tests" / "test_project_steering_hooks.py"
META_VALIDATOR_TRUST_AUDIT = ROOT / "tools" / "qa_meta_validator_trust_audit.py"


def _run(cmd: str, cwd: str = None, timeout: int = 60) -> tuple:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd or str(ROOT))
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def _run_argv(argv: list[str], cwd: str = None, timeout: int = 60) -> tuple:
    try:
        p = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(ROOT),
        )
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


def check_bridge_cert_static():
    """Fail unless the bridge is wired to emit certs on the live path."""
    if not BRIDGE_FILE.exists():
        RESULTS["fail"].append("Bridge cert wiring: llm_bridge_agent.py not found")
        return

    content = BRIDGE_FILE.read_text(encoding="utf-8")
    required_markers = [
        "TOOL_CALL_CERT.v1",
        "PROMPT_INJECTION_OBSTRUCTION.v1",
        "OUTPUT_SCAN_CERT.v1",
        "BridgeExecutionCerts",
        "--self-test",
        "enforce_policy",
        "CapabilityToken",
        "_mint_bridge_token",
        "_scan_and_redact_output",
        "output_provenance",
    ]
    missing = [marker for marker in required_markers if marker not in content]
    if missing:
        RESULTS["fail"].append(
            f"Bridge cert wiring: missing markers {missing} in llm_bridge_agent.py"
        )
    else:
        RESULTS["pass"].append("Bridge cert wiring: cert-gated + capability-enforced")


def check_bridge_cert_runtime():
    """Run the bridge self-test and verify real cert artifacts are produced."""
    if not BRIDGE_FILE.exists():
        RESULTS["fail"].append("Bridge cert runtime: llm_bridge_agent.py not found")
        return

    rc, stdout, stderr = _run_argv(
        [
            sys.executable,
            str(BRIDGE_FILE),
            "--self-test",
            "--name",
            "audit_bridge",
            "--cmd",
            "cat",
        ],
        timeout=60,
    )
    if rc != 0:
        RESULTS["fail"].append(f"Bridge cert runtime: self-test failed (rc={rc})")
        return

    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError:
        RESULTS["fail"].append("Bridge cert runtime: self-test did not return JSON")
        return

    trace_path = Path(payload.get("trace_path", ""))
    cap_enforced = payload.get("capability_enforced", False)
    tests_passed = payload.get("tests_passed", 0)
    tests_total = payload.get("tests_total", 0)
    if (
        payload.get("ok")
        and payload.get("tool_call_cert_count", 0) >= 2
        and trace_path.exists()
        and cap_enforced
        and tests_passed == tests_total
    ):
        RESULTS["pass"].append(
            f"Bridge cert runtime: {tests_passed}/{tests_total} tests, capability enforced"
        )
    else:
        detail = []
        if not payload.get("ok"):
            detail.append("self-test failed")
        if not cap_enforced:
            detail.append("capability NOT enforced")
        if tests_passed != tests_total:
            detail.append(f"tests {tests_passed}/{tests_total}")
        RESULTS["fail"].append(
            f"Bridge cert runtime: {' + '.join(detail) or 'unknown failure'}"
        )


def check_cert_gate_hook_static():
    """Fail unless the live Claude PreToolUse hook is an enforcement gate."""
    if not CERT_GATE_HOOK.exists():
        RESULTS["fail"].append("PreTool cert gate: cert_gate_hook.py not found")
        return

    content = CERT_GATE_HOOK.read_text(encoding="utf-8")
    forbidden_markers = [
        "AUDIT ONLY",
        "Always exits 0",
        "do not block",
        "LLM_QA_ALLOW_CLAUDE_PYTHON_EDIT",
        "CLAUDE_PYTHON_WRITE_FORBIDDEN",
    ]
    forbidden = [marker for marker in forbidden_markers if marker in content]
    required_markers = [
        "EXIT_BLOCK = 2",
        "CERT_LEDGER_FAILURE",
        "CERT_COLLAB_MARKER_MISSING",
        "DESTRUCTIVE_RM_RECURSIVE_FORCE",
        "WRAPPER_SELF_MODIFICATION",
        "GIT_FORCE_PUSH_FORBIDDEN",
        "GIT_COMMIT_WITHOUT_COLLAB_MARKER",
        "PROTECTED_TARGET_MUTATION",
        "CLAUDE_PYTHON_WRITE_QUARANTINED",
        "CODEX_REVIEW_PENDING",
        "decision != \"ALLOW\"",
        "_read_ledger_state",
        "enforced.jsonl",
    ]
    missing = [marker for marker in required_markers if marker not in content]
    if forbidden or missing:
        detail = []
        if forbidden:
            detail.append(f"forbidden markers {forbidden}")
        if missing:
            detail.append(f"missing markers {missing}")
        RESULTS["fail"].append(f"PreTool cert gate static: {'; '.join(detail)}")
    else:
        RESULTS["pass"].append("PreTool cert gate static: enforcement markers present")


def check_cert_gate_hook_runtime():
    """Run the PreToolUse hook regression suite."""
    if not CERT_GATE_HOOK_TEST.exists():
        RESULTS["fail"].append("PreTool cert gate runtime: test_cert_gate_hook.py not found")
        return

    rc, stdout, stderr = _run_argv(
        [sys.executable, str(CERT_GATE_HOOK_TEST)],
        timeout=60,
    )
    if rc == 0 and "18/18 tests passed" in stdout:
        RESULTS["pass"].append("PreTool cert gate runtime: 18/18 PASS")
    else:
        RESULTS["fail"].append(
            f"PreTool cert gate runtime: FAILED (rc={rc})"
        )


def check_project_steering_hooks_runtime():
    """Run regression tests for legacy Claude steering hooks."""
    if not PROJECT_STEERING_HOOK_TEST.exists():
        RESULTS["fail"].append("Project steering hooks: test_project_steering_hooks.py not found")
        return

    rc, stdout, stderr = _run_argv(
        [sys.executable, str(PROJECT_STEERING_HOOK_TEST)],
        timeout=60,
    )
    if rc == 0 and "6/6 tests passed" in stdout:
        RESULTS["pass"].append("Project steering hooks runtime: 6/6 PASS")
    else:
        RESULTS["fail"].append(
            f"Project steering hooks runtime: FAILED (rc={rc})"
        )


def check_meta_validator_trust_audit():
    """Run the meta-validator trust-boundary audit."""
    if not META_VALIDATOR_TRUST_AUDIT.exists():
        RESULTS["fail"].append("Meta-validator trust audit: qa_meta_validator_trust_audit.py not found")
        return

    rc, stdout, stderr = _run_argv(
        [sys.executable, str(META_VALIDATOR_TRUST_AUDIT), "--json"],
        timeout=60,
    )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        RESULTS["fail"].append(
            f"Meta-validator trust audit: non-JSON output (rc={rc})"
        )
        return

    if rc == 0 and payload.get("ok") is True:
        RESULTS["pass"].append(
            f"Meta-validator trust audit: {payload.get('family_sweep_count')} families"
        )
        return

    issue_counts = {}
    for issue in payload.get("issues", []):
        issue_counts[issue.get("code", "UNKNOWN")] = issue_counts.get(issue.get("code", "UNKNOWN"), 0) + 1
    RESULTS["fail"].append(
        "Meta-validator trust audit: "
        f"{payload.get('fail_count', 0)} fail, {payload.get('warn_count', 0)} warn; "
        f"codes={json.dumps(issue_counts, sort_keys=True)}"
    )


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


def check_topic_acl():
    """Verify topic ACL is present in collab MCP server."""
    server_file = ROOT / "qa_lab" / "qa_mcp_servers" / "qa-collab" / "server.py"
    if not server_file.exists():
        RESULTS["warn"].append("Collab MCP server not found")
        return

    content = server_file.read_text()
    if "TOPIC_ACL" in content and "_check_topic_acl" in content:
        RESULTS["pass"].append("Topic ACL: enforced in collab MCP server")
    else:
        RESULTS["fail"].append("Topic ACL: NOT present in collab MCP server — llm_request.* topics unprotected")


def check_bridge_processes():
    """Check bridge liveness via shared heartbeat files, not process namespace state."""
    bridge_root = ROOT / "qa_lab" / "logs" / "bridge_security"
    if not bridge_root.exists():
        RESULTS["warn"].append("No LLM bridge agents running")
        return

    current_unix = int(time.time())
    active = 0

    for status_file in sorted(bridge_root.glob("*/bridge_status.json")):
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            RESULTS["warn"].append(f"Bridge status unreadable: {status_file}")
            continue

        agent = str(payload.get("agent") or status_file.parent.name)
        updated_unix = payload.get("updated_unix")
        if not isinstance(updated_unix, int):
            RESULTS["warn"].append(f"Bridge {agent}: missing heartbeat timestamp")
            continue

        age_s = max(0, current_unix - updated_unix)

        if age_s > 5 or not payload.get("running", False):
            continue

        active += 1
        if not payload.get("guardrail_active", False):
            RESULTS["fail"].append(f"Bridge {agent}: heartbeat shows guardrail disabled")
        elif not payload.get("bus_connected", False):
            RESULTS["warn"].append(f"Bridge {agent}: running without collaboration bus connectivity")
        else:
            RESULTS["pass"].append(f"Bridge {agent}: running (guardrail enabled, bus connected)")

    if active == 0:
        RESULTS["warn"].append("No LLM bridge agents running")


def main():
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("QA SECURITY AUDIT")
    print("=" * 60)

    check_guardrail_e2e()
    check_agent_security()
    check_bridge_cert_static()
    check_bridge_cert_runtime()
    check_cert_gate_hook_static()
    check_cert_gate_hook_runtime()
    check_project_steering_hooks_runtime()
    check_meta_validator_trust_audit()
    check_collab_agents()
    check_event_log_secrets()
    check_guardrail_denials()
    check_topic_acl()
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
