#!/usr/bin/env python3
# noqa: DECL-1 (hook infrastructure — not empirical QA code)
"""
cert_gate_hook.py — PreToolUse hook for LLM QA Wrapper.

Reads the tool call JSON from stdin, evaluates it against the local
policy, creates a CertRecord, and appends it to the hash-chained
ledger under llm_qa_wrapper/ledger/. Runs on every Edit, Write, and
Bash tool call Claude makes.

This hook is an enforcement boundary. It exits 0 only when the policy
allows the call and the ledger append succeeds. It exits 2 on policy
denial, malformed hook input, or cert/ledger failure.

Runtime ledger: `enforced.jsonl`. The older `live.jsonl` file was an
audit-only ledger and may contain historical forked-chain records; the
enforcement ledger is verified from genesis before every append.

Usage (in .claude/settings.local.json):
    {"matcher": "Edit|Write|Bash", "hooks": [
        {"type": "command", "command": ".../cert_gate_hook.py"}
    ]}

The hook's stdin receives:
    {"tool_name": "Write", "tool_input": {"file_path": "...", ...}}
"""
import hashlib
import json
import os
import re
import sys
import time
import base64
from pathlib import Path

REPO = Path(os.environ.get(
    "LLM_QA_WRAPPER_REPO",
    str(Path(__file__).resolve().parent.parent),
)).resolve()
LEDGER_DIR = Path(os.environ.get(
    "LLM_QA_WRAPPER_LEDGER_DIR",
    str(Path(__file__).resolve().parent / "ledger"),
)).resolve()
LEDGER_BASENAME = os.environ.get("LLM_QA_WRAPPER_LEDGER_FILE", "enforced.jsonl")
COLLAB_MARKER = Path(os.environ.get(
    "LLM_QA_WRAPPER_COLLAB_MARKER",
    "/tmp/qa_collab_session_registered",
))
LEDGER_FILE = LEDGER_DIR / LEDGER_BASENAME
COUNTER_FILE = LEDGER_DIR / f".{LEDGER_BASENAME}.counter"
TAIL_HASH_FILE = LEDGER_DIR / f".{LEDGER_BASENAME}.tail_hash"
QUARANTINE_DIR = Path(os.environ.get(
    "LLM_QA_WRAPPER_QUARANTINE_DIR",
    str(Path(__file__).resolve().parent / "quarantine"),
)).resolve()
QUARANTINE_PENDING_DIR = QUARANTINE_DIR / "pending"
QUARANTINE_LEDGER_FILE = QUARANTINE_DIR / "codex_reviews.jsonl"

GENESIS = bytes(32)
AGENT = "claude"
EXIT_ALLOW = 0
EXIT_BLOCK = 2
POLICY_VERSION = "LLM_QA_WRAPPER_PRETOOL_POLICY.v2"

EDIT_TOOLS = {"Edit", "Write"}
SUPPORTED_TOOLS = EDIT_TOOLS | {"Bash"}

DANGEROUS_BASH_PATTERNS = [
    ("SHELL_PIPE_TO_INTERPRETER", re.compile(r"\b(curl|wget)\b[^|;&]*\|\s*(sh|bash|zsh|python|python3)\b", re.I)),
    ("DESTRUCTIVE_RM_RECURSIVE_FORCE", re.compile(r"(^|[;&|]\s*)rm\s+-[A-Za-z]*r[A-Za-z]*f[A-Za-z]*\b", re.I)),
    ("DESTRUCTIVE_FILESYSTEM_FORMAT", re.compile(r"\bmkfs(?:\.[A-Za-z0-9_+-]+)?\b|\bmkswap\b", re.I)),
    ("RAW_BLOCK_DEVICE_WRITE", re.compile(r"\bdd\b[^;&|]*\bof=/dev/", re.I)),
    ("DESTRUCTIVE_GIT_RESET", re.compile(r"\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[A-Za-z]*[xdf][A-Za-z]*\b", re.I)),
    ("GIT_FORCE_PUSH_FORBIDDEN", re.compile(r"\bgit\s+push\b(?=[^;&|]*(?:--force|-f\b|--force-with-lease))", re.I)),
    ("GLOBAL_CHMOD_777", re.compile(r"\bchmod\s+(?:-R\s+)?777\s+(?:/|~|\$HOME|/home/player2)\b", re.I)),
    ("PRIVILEGE_ESCALATION", re.compile(r"(^|[;&|]\s*)sudo\b", re.I)),
    ("FORK_BOMB", re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:", re.I)),
    ("SECRET_FILE_READ", re.compile(r"(?<![A-Za-z0-9_])(?:cat|less|more|head|tail|sed|awk)\b[^;&|]*(?:\.open_brain_mcp_key|id_rsa|\.ssh/|\.env)(?![A-Za-z0-9_])", re.I)),
]

PROTECTED_BASH_TARGETS = (
    ".claude/hooks/",
    ".claude/settings.local.json",
    "archive/",
    "QAnotes/",
    "llm_qa_wrapper/cert_gate_hook.py",
    "llm_qa_wrapper/kernel/",
    "llm_qa_wrapper/ledger/",
)
DOCUMENTS_PDF_INGRESS_PREFIXES = (
    "Documents/wildberger_corpus/",
    "Documents/haramein_rsf/",
)
CERT_ADJACENT_PREFIXES = (
    "qa_alphageometry_ptolemy/",
    "tools/qa_",
    "tools/tests/",
    "qa_agent_security/",
    "qa_guardrail/",
    "qa_mapping/",
    "qa_predict/",
    "llm_qa_wrapper/",
    ".claude/hooks/",
)
CERT_ADJACENT_EXACT = {
    "AGENTS.md",
    "CLAUDE.md",
    "MEMORY.md",
    "docs/families/README.md",
}
PYTHON_INLINE_MUTATION_PATTERN = re.compile(
    r"(^|[;&|]\s*)(?:python3?|perl|ruby)\b.*"
    r"(?:write_text|open\s*\([^)]*,\s*['\"][wa]|os\.remove|unlink|shutil\.rmtree)",
    re.I,
)
# Shell-token mutation patterns. Run against a quote-stripped view so that
# `>` / `mv` / `rm` etc. inside quoted string payloads do not false-positive.
SHELL_MUTATION_PATTERN = re.compile(
    r"(^|[;&|]\s*)(?:sed|perl)\s+-i\b|"
    r"(^|[;&|]\s*)(?:mv|cp|rm|truncate|chmod|chown)\b|"
    r"(?<![\w<])>>?",
    re.I,
)
# Kept as the union for callers / tests that want a single-check form.
MUTATING_BASH_PATTERN = re.compile(
    PYTHON_INLINE_MUTATION_PATTERN.pattern + "|" + SHELL_MUTATION_PATTERN.pattern,
    re.I,
)
QUOTED_STRING_PATTERN = re.compile(r"'[^']*'|\"[^\"]*\"")
HEREDOC_START_PATTERN = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
PYTHON_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_./-])[^;&|<>\s'\"]+\.py(?:\b|$)", re.I)
# Fd-to-fd duplications and /dev/null sinks are not file mutations. Strip
# them before running MUTATING_BASH_PATTERN so stderr redirects like
# `2>&1` and `2>/dev/null` don't false-positive the Python-write gate.
FD_REDIRECT_PATTERN = re.compile(
    r"\d*>&[\d-]"           # 2>&1, >&2, 1>&-
    r"|&>\s*/dev/null"      # &>/dev/null
    r"|\d*>\s*/dev/null",   # 2>/dev/null, >/dev/null
)


def _strip_heredoc_bodies(command: str) -> str:
    """Remove literal heredoc payloads so policy scans shell syntax, not delegated prompts."""
    lines = command.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        match = HEREDOC_START_PATTERN.search(line)
        if not match:
            i += 1
            continue
        delimiter = match.group(1)
        i += 1
        while i < len(lines) and lines[i].strip() != delimiter:
            i += 1
        if i < len(lines):
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _bash_mutates_python_path(scan_command: str) -> bool:
    """Return true only for Bash commands that plausibly write Python files."""
    normalized = scan_command.replace("\\", "/")
    if ".py" not in normalized:
        return False
    fd_stripped = FD_REDIRECT_PATTERN.sub("", scan_command)
    shell_scan = QUOTED_STRING_PATTERN.sub("", fd_stripped)
    has_python_inline_mutation = bool(PYTHON_INLINE_MUTATION_PATTERN.search(scan_command))
    has_shell_mutation = bool(SHELL_MUTATION_PATTERN.search(shell_scan))
    if not (has_python_inline_mutation or has_shell_mutation):
        return False
    return PYTHON_PATH_PATTERN.search(normalized) is not None


def _is_allowed_documents_pdf(rel_posix: str) -> bool:
    """Allow primary-source PDF ingress only in the two corpus directories.

    The rest of Documents/ remains protected. This preserves the historical
    "Do Not Touch" rule for drafts/exports/chat migrations while allowing the
    Phase 4.5 primary-source workflow to add or replace canonical PDFs.
    """
    return (
        rel_posix.lower().endswith(".pdf")
        and any(rel_posix.startswith(prefix) for prefix in DOCUMENTS_PDF_INGRESS_PREFIXES)
    )


def _bash_mentions_protected_project_target(scan_command: str) -> bool:
    """Detect protected project-directory mutation targets in Bash text.

    Existing hook logic intentionally avoids a full shell parser. For
    Documents/, strip the narrow allowed PDF targets before checking the
    remaining command text so `cp /tmp/p.pdf Documents/haramein_rsf/p.pdf`
    can pass, while `rm Documents/foo.md` and directory-level operations stay
    denied.
    """
    normalized = scan_command.replace("\\", "/")
    if any(target in normalized for target in PROTECTED_BASH_TARGETS):
        return True
    docs_stripped = normalized
    for prefix in DOCUMENTS_PDF_INGRESS_PREFIXES:
        pattern = re.compile(
            re.escape(prefix) + r"[^;&|<>\s'\"\)]*\.pdf\b",
            re.I,
        )
        docs_stripped = pattern.sub("", docs_stripped)
    return "Documents/" in docs_stripped


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _hex_to_hash(value: str, field_name: str) -> bytes:
    try:
        raw = bytes.fromhex(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} is not valid hex") from e
    if len(raw) != 32:
        raise ValueError(f"{field_name} is not a 32-byte hash")
    return raw


def _read_ledger_state() -> tuple[int, bytes]:
    """Verify the on-disk ledger and return (next_counter, tail_hash)."""
    if not LEDGER_FILE.exists():
        return 0, GENESIS

    expected_prev = GENESIS
    seen: set[bytes] = set()
    count = 0
    with LEDGER_FILE.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                agent = str(obj["agent"])
                tool = str(obj["tool"])
                payload_hash_bytes = _hex_to_hash(obj["payload_hash"], "payload_hash")
                prev = _hex_to_hash(obj["prev"], "prev")
                counter = int(obj["counter"])
                self_hash = _hex_to_hash(obj["self_hash"], "self_hash")
            except Exception as e:
                raise RuntimeError(f"ledger malformed at line {line_no}: {e}") from e

            if prev != expected_prev:
                raise RuntimeError(
                    f"ledger chain broken at line {line_no}: "
                    f"expected prev={expected_prev.hex()}, got prev={prev.hex()}"
                )
            if counter != count:
                raise RuntimeError(
                    f"ledger counter broken at line {line_no}: "
                    f"expected {count}, got {counter}"
                )
            expected_self = _compute_self_hash(agent, tool, payload_hash_bytes, prev, counter)
            if self_hash != expected_self:
                raise RuntimeError(f"ledger self_hash mismatch at line {line_no}")
            if self_hash in seen:
                raise RuntimeError(f"ledger duplicate self_hash at line {line_no}")
            seen.add(self_hash)
            expected_prev = self_hash
            count += 1
    return count, expected_prev


def _compute_self_hash(agent, tool, payload_hash, prev, counter):
    h = hashlib.sha256()
    h.update(agent.encode("utf-8"))
    h.update(b"\x00")
    h.update(tool.encode("utf-8"))
    h.update(b"\x00")
    h.update(payload_hash)
    h.update(b"\x00")
    h.update(prev)
    h.update(b"\x00")
    h.update(counter.to_bytes(8, "big"))
    return h.digest()


LOCK_FILE = LEDGER_DIR / ".lock"


def _repo_relative(file_path: str) -> tuple[Path, str]:
    raw_path = Path(file_path)
    if not raw_path.is_absolute():
        raw_path = REPO / raw_path
    canonical = raw_path.resolve(strict=False)
    try:
        rel = str(canonical.relative_to(REPO))
    except ValueError:
        rel = str(canonical)
    return canonical, rel


def _quarantine_payload_hash(packet_payload: dict) -> str:
    return hashlib.sha256(
        b"QA_CLAUDE_PYTHON_QUARANTINE.v1\x00" + _canonical_json(packet_payload)
    ).hexdigest()


def _write_quarantine_packet(tool_name: str, tool_input: dict, reasons: list[str]) -> str | None:
    """Persist a Python mutation attempt for batched Codex review."""
    packet_payload = {
        "agent": AGENT,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "deny_reasons": sorted(set(reasons)),
    }
    digest = _quarantine_payload_hash(packet_payload)
    packet = {
        "schema_version": "QA_CLAUDE_PYTHON_QUARANTINE.v1",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": AGENT,
        "tool_name": tool_name,
        "payload_sha256": digest,
        "review_status": "pending_codex_review",
        "deny_reasons": sorted(set(reasons)),
        "tool_input": tool_input,
    }
    file_path = tool_input.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        canonical, rel = _repo_relative(file_path)
        packet["target_path"] = str(canonical)
        packet["target_rel"] = rel.replace(os.sep, "/")
        packet["original_exists"] = canonical.exists()
        packet["original_snapshot_available"] = False
        if canonical.exists() and canonical.is_file():
            try:
                original = canonical.read_bytes()
            except OSError as exc:
                packet["original_snapshot_error"] = str(exc)
            else:
                packet["original_snapshot_available"] = True
                packet["original_sha256"] = hashlib.sha256(original).hexdigest()
                packet["original_content_b64"] = base64.b64encode(original).decode("ascii")
        elif not canonical.exists():
            packet["original_snapshot_available"] = True

    QUARANTINE_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time() * 1000000)}_{digest[:16]}.json"
    out = QUARANTINE_PENDING_DIR / filename
    out.write_text(
        json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(out)


def _quarantine_attempt(tool_name: str, tool_input: dict, reasons: list[str]) -> None:
    try:
        _write_quarantine_packet(tool_name, tool_input, reasons)
    except Exception:
        reasons.append("QUARANTINE_WRITE_FAILURE")


def _pending_quarantine_count() -> int:
    try:
        return sum(1 for p in QUARANTINE_PENDING_DIR.glob("*.json") if p.is_file())
    except OSError:
        return 0


def _deny_file_path(tool_name: str, tool_input: dict) -> list[str]:
    reasons: list[str] = []
    quarantine_needed = False
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        return [f"{tool_name.upper()}_MISSING_FILE_PATH"]

    canonical, rel = _repo_relative(file_path)
    rel_posix = rel.replace(os.sep, "/")
    canonical_text = str(canonical)

    if canonical_text.startswith("/home/player2/Desktop/qa_finance/"):
        reasons.append("FROZEN_QA_FINANCE")
    if rel_posix.startswith(("archive/", "QAnotes/")):
        reasons.append("PROTECTED_PROJECT_DIRECTORY")
    if rel_posix.startswith("Documents/") and not _is_allowed_documents_pdf(rel_posix):
        reasons.append("PROTECTED_PROJECT_DIRECTORY")
    if rel_posix.lower().endswith(".png"):
        reasons.append("GENERATED_BINARY_OUTPUT")
    if rel_posix.startswith("llm_qa_wrapper/ledger/"):
        reasons.append("LEDGER_DIRECT_EDIT")
    if rel_posix.endswith(".py"):
        quarantine_needed = True
    if rel_posix in {
        ".claude/settings.local.json",
        "llm_qa_wrapper/cert_gate_hook.py",
    } or rel_posix.startswith((".claude/hooks/", "llm_qa_wrapper/kernel/")):
        reasons.append("WRAPPER_SELF_MODIFICATION")
    if (
        rel_posix in CERT_ADJACENT_EXACT
        or rel_posix.startswith(CERT_ADJACENT_PREFIXES)
    ):
        try:
            marker_ok = COLLAB_MARKER.read_text(encoding="utf-8").strip() != ""
        except OSError:
            marker_ok = False
        if not marker_ok:
            reasons.append("CERT_COLLAB_MARKER_MISSING")
    if quarantine_needed:
        _quarantine_attempt(tool_name, tool_input, reasons + ["CLAUDE_PYTHON_WRITE_QUARANTINED"])
    return reasons


def _deny_bash(tool_input: dict) -> list[str]:
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return ["BASH_MISSING_COMMAND"]

    scan_command = _strip_heredoc_bodies(command)
    quarantine_needed = False
    reasons = [
        reason for reason, pattern in DANGEROUS_BASH_PATTERNS
        if pattern.search(scan_command)
    ]
    shell_scan = QUOTED_STRING_PATTERN.sub("", FD_REDIRECT_PATTERN.sub("", scan_command))
    if PYTHON_INLINE_MUTATION_PATTERN.search(scan_command) or SHELL_MUTATION_PATTERN.search(shell_scan):
        normalized = scan_command.replace("\\", "/")
        if _bash_mentions_protected_project_target(scan_command):
            reasons.append("PROTECTED_TARGET_MUTATION")
        if _bash_mutates_python_path(scan_command):
            quarantine_needed = True
        if re.search(r"(^|/)[^/ ]+\.png(?:\s|$)", normalized, re.I):
            reasons.append("GENERATED_BINARY_OUTPUT")
    if re.search(r"\bgit\s+commit\b", scan_command, re.I):
        try:
            marker_ok = COLLAB_MARKER.read_text(encoding="utf-8").strip() != ""
        except OSError:
            marker_ok = False
        if not marker_ok:
            reasons.append("GIT_COMMIT_WITHOUT_COLLAB_MARKER")
        if _pending_quarantine_count() > 0:
            reasons.append("CODEX_REVIEW_PENDING")
    if quarantine_needed:
        _quarantine_attempt("Bash", tool_input, reasons + ["CLAUDE_PYTHON_WRITE_QUARANTINED"])
    return reasons


def _policy_decision(hook_input: dict) -> tuple[str, list[str], str, dict]:
    if not isinstance(hook_input, dict):
        return "DENY", ["HOOK_INPUT_NOT_OBJECT"], "unknown", {}

    tool_name = hook_input.get("tool_name", "unknown")
    tool_input = hook_input.get("tool_input", {})
    if not isinstance(tool_name, str) or not tool_name:
        tool_name = "unknown"
    if not isinstance(tool_input, dict):
        return "DENY", ["TOOL_INPUT_NOT_OBJECT"], tool_name, {}

    if tool_name not in SUPPORTED_TOOLS:
        return "DENY", [f"UNSUPPORTED_TOOL:{tool_name}"], tool_name, tool_input
    if tool_name in EDIT_TOOLS:
        reasons = _deny_file_path(tool_name, tool_input)
    else:
        reasons = _deny_bash(tool_input)
    return ("DENY" if reasons else "ALLOW"), reasons, tool_name, tool_input


def _enforcement_markers(tool_name: str, tool_input: dict) -> list[str]:
    if tool_name in EDIT_TOOLS:
        file_path = tool_input.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            _, rel = _repo_relative(file_path)
            if rel.replace(os.sep, "/").endswith(".py"):
                return ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
    if tool_name == "Bash":
        command = tool_input.get("command")
        if isinstance(command, str) and _bash_mutates_python_path(_strip_heredoc_bodies(command)):
            return ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
    return []


def _append_record(tool_name: str, policy_payload: dict) -> None:
    ph = _sha256(_canonical_json(policy_payload))
    counter, prev = _read_ledger_state()
    self_hash = _compute_self_hash(AGENT, tool_name, ph, prev, counter)

    record = {
        "agent": AGENT,
        "tool": tool_name,
        "payload_hash": ph.hex(),
        "prev": prev.hex(),
        "counter": counter,
        "self_hash": self_hash.hex(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy_version": POLICY_VERSION,
        "decision": policy_payload["decision"],
        "deny_reasons": policy_payload["deny_reasons"],
        "enforcement_markers": policy_payload.get("enforcement_markers", []),
    }

    fd = os.open(str(LEDGER_FILE), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        line = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)

    COUNTER_FILE.write_text(str(counter + 1))
    TAIL_HASH_FILE.write_text(self_hash.hex())


def main():
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        hook_input = {}
        parse_error = f"HOOK_JSON_PARSE_ERROR:{type(e).__name__}"
    else:
        parse_error = ""

    decision, deny_reasons, tool_name, tool_input = _policy_decision(hook_input)
    enforcement_markers = _enforcement_markers(tool_name, tool_input)
    if parse_error:
        decision = "DENY"
        deny_reasons = [parse_error]
        enforcement_markers = []

    # Acquire file lock to prevent race between parallel hook invocations.
    # Two parallel tool calls previously read the same counter+tail_hash,
    # producing duplicate counters and chain breaks (2026-04-12 incident).
    import fcntl
    lock_fd = os.open(str(LOCK_FILE), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        policy_payload = {
            "policy_version": POLICY_VERSION,
            "decision": decision,
            "deny_reasons": deny_reasons,
            "enforcement_markers": enforcement_markers,
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        _append_record(tool_name, policy_payload)
    except Exception as e:
        print(f"cert_gate_hook: BLOCKED ledger/cert failure: {e}", file=sys.stderr)
        decision = "DENY"
        deny_reasons = ["CERT_LEDGER_FAILURE"]
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    if decision != "ALLOW":
        print(
            "cert_gate_hook: BLOCKED "
            + ",".join(deny_reasons or ["POLICY_DENY"]),
            file=sys.stderr,
        )
        sys.exit(EXIT_BLOCK)

    sys.exit(EXIT_ALLOW)


if __name__ == "__main__":
    main()
