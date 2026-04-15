#!/usr/bin/env python3
# noqa: DECL-1 (hook infrastructure)
"""
primary_source_gate.py -- PreToolUse hook for primary-source enforcement.

Reads Claude hook JSON from stdin and blocks Write/Edit calls to in-scope
theory, experiment, cert, and paper files unless the post-write content
contains a primary-source citation or an explicit exemption marker.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


REPO = Path(os.environ.get(
    "LLM_QA_WRAPPER_REPO",
    str(Path(__file__).resolve().parents[2]),
)).resolve()
LEDGER_FILE = Path(os.environ.get(
    "LLM_QA_WRAPPER_PRIMARY_SOURCE_LEDGER",
    str(REPO / "llm_qa_wrapper" / "ledger" / "primary_source_gate.jsonl"),
)).resolve()

EXIT_ALLOW = 0
EXIT_BLOCK = 2
SUPPORTED_TOOLS = {"Write", "Edit"}

CITATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "explicit_primary_source",
        re.compile(r"(?:^|\n)\s*\*?\*?\s*Primary\s+source[s]?\s*:\s*\*?\*?", re.I),
    ),
    (
        "arxiv_id",
        re.compile(r"arxiv(?:\.org/(?:abs|pdf))?[:/\s]\d{4}\.\d{4,5}", re.I),
    ),
    (
        "old_arxiv_id",
        re.compile(r"arxiv[:/\s](?:math|cs|stat|physics|quant-ph)/\d{7}", re.I),
    ),
    (
        "doi",
        re.compile(r"doi\.org/\S+|DOI:\s*\S+", re.I),
    ),
    (
        "isbn",
        re.compile(r"ISBN(?:-10|-13)?[:\s]\s*[\d\-X]{10,17}", re.I),
    ),
    (
        "inline_citation",
        re.compile(
            r"\([A-Z][A-Za-z\-']+"
            r"(?:\s+(?:&|and|et\s+al\.?)\s+[A-Z][A-Za-z\-']+)?,?"
            r"\s+\d{4}\)"
        ),
    ),
    (
        "companion_file_primary",
        re.compile(r"companion\s+files?\s*:.*\.(?:pdf|tex)", re.I),
    ),
]

REFERENCES_SECTION = re.compile(
    r"^##\s+(?:References|Bibliography|Sources|Citations)\s*$",
    re.I,
)
NEXT_SECTION = re.compile(r"^##\s+\S+")
EXEMPTION = re.compile(
    r"<!--\s*PRIMARY-SOURCE-EXEMPT:"
    r"(?P<body>.*?)"
    r"-->"
    ,
    re.I | re.S,
)
EXEMPT_REASON = re.compile(r"""reason\s*=\s*["'](?P<reason>[^"']+)["']""", re.I)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _repo_relative(file_path: str) -> tuple[Path, str]:
    raw_path = Path(file_path)
    if not raw_path.is_absolute():
        raw_path = REPO / raw_path
    canonical = raw_path.resolve(strict=False)
    try:
        rel = str(canonical.relative_to(REPO))
    except ValueError:
        rel = str(canonical)
    return canonical, rel.replace(os.sep, "/")


def _name_matches(pattern: str, text: str) -> bool:
    regex = re.escape(pattern).replace(r"\*", r"[^/]*")
    return re.fullmatch(regex, text) is not None


def _is_in_scope(rel_path: str) -> bool:
    if _name_matches(r"docs/theory/QA_*.md", rel_path):
        return True
    if _name_matches(r"docs/theory/QA_*.tex", rel_path):
        return True
    if re.fullmatch(r"qa_[^/]*_experiments/.*", rel_path):
        return True
    if re.fullmatch(r"qa_alphageometry_ptolemy/qa_[^/]*_cert_v1/.*", rel_path):
        return True
    if rel_path.startswith("qa_alphageometry_ptolemy/docs/"):
        return True
    if rel_path.startswith("papers/") and rel_path.endswith((".md", ".tex")):
        return True
    if rel_path == "docs/families/README.md":
        return True
    if _name_matches(r"docs/specs/QA_*.md", rel_path):
        return True
    return False


def _references_section_has_entry(content: str) -> bool:
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if not REFERENCES_SECTION.match(line):
            continue
        for following in lines[idx + 1:]:
            if NEXT_SECTION.match(following):
                break
            if following.strip():
                return True
    return False


def _matched_citation_pattern(content: str) -> str | None:
    for name, pattern in CITATION_PATTERNS:
        if name == "inline_citation":
            if len(pattern.findall(content)) >= 2:
                return name
            continue
        if pattern.search(content):
            return name
    if _references_section_has_entry(content):
        return "references_section"
    return None


def _exemption_reason(content: str) -> str | None:
    prefix = content.encode("utf-8")[:500].decode("utf-8", errors="ignore")
    marker = EXEMPTION.search(prefix)
    if not marker:
        return None
    reason = EXEMPT_REASON.search(marker.group("body"))
    return reason.group("reason") if reason else ""


def _post_write_content(tool_input: dict) -> str:
    content = tool_input.get("content", "")
    return content if isinstance(content, str) else str(content)


def _post_edit_content(file_path: Path, tool_input: dict) -> str:
    old_string = tool_input.get("old_string")
    new_string = tool_input.get("new_string")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        raise ValueError("Edit payload missing old_string/new_string")
    current = file_path.read_text(encoding="utf-8")
    if old_string not in current:
        raise ValueError("old_string not found in on-disk file")
    return current.replace(old_string, new_string, 1)


def _append_ledger(
    *,
    tool_name: str,
    rel_path: str,
    decision: str,
    matched_pattern: str | None,
    exempt_reason: str | None,
    content: str,
) -> None:
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool_name,
        "file": rel_path,
        "decision": decision,
        "matched_pattern": matched_pattern,
        "exempt_reason": exempt_reason,
        "content_sha256": _sha256_text(content),
    }
    with LEDGER_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _print_block_message(rel_path: str) -> None:
    print(
        "\n".join([
            "cert_gate_hook: BLOCKED PRIMARY_SOURCE_REQUIRED",
            f"file: {rel_path}",
            "reason: no primary-source citation matched.",
            (
                "required: Primary source: ... / arxiv.org/abs/... / DOI / "
                "## References / ISBN / \u22652 (Author, Year) citations / companion file"
            ),
            "authority: memory/feedback_map_best_to_qa.md",
            (
                "exempt: add <!-- PRIMARY-SOURCE-EXEMPT: reason=... --> in the "
                "first 500 bytes if this is an intentional scaffold write"
            ),
        ]),
        file=sys.stderr,
    )


def _print_malformed(reason: str) -> None:
    print(f"cert_gate_hook: BLOCKED PRIMARY_SOURCE_GATE_MALFORMED {reason}", file=sys.stderr)


def main() -> None:
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        _print_malformed(f"HOOK_JSON_PARSE_ERROR:{type(e).__name__}")
        sys.exit(EXIT_BLOCK)

    if not isinstance(hook_input, dict):
        _print_malformed("HOOK_INPUT_NOT_OBJECT")
        sys.exit(EXIT_BLOCK)

    tool_name = hook_input.get("tool_name")
    tool_input = hook_input.get("tool_input", {})
    if tool_name not in SUPPORTED_TOOLS:
        sys.exit(EXIT_ALLOW)
    if not isinstance(tool_input, dict):
        _print_malformed("TOOL_INPUT_NOT_OBJECT")
        sys.exit(EXIT_BLOCK)

    file_path_value = tool_input.get("file_path")
    if not isinstance(file_path_value, str) or not file_path_value.strip():
        _print_malformed("MISSING_FILE_PATH")
        sys.exit(EXIT_BLOCK)

    canonical_path, rel_path = _repo_relative(file_path_value)
    if not _is_in_scope(rel_path):
        _append_ledger(
            tool_name=tool_name,
            rel_path=rel_path,
            decision="ALLOW",
            matched_pattern=None,
            exempt_reason=None,
            content="",
        )
        sys.exit(EXIT_ALLOW)

    try:
        content = (
            _post_write_content(tool_input)
            if tool_name == "Write"
            else _post_edit_content(canonical_path, tool_input)
        )
    except Exception as e:
        _append_ledger(
            tool_name=tool_name,
            rel_path=rel_path,
            decision="BLOCK",
            matched_pattern=None,
            exempt_reason=None,
            content="",
        )
        _print_malformed(f"EDIT_POST_IMAGE_FAILED:{type(e).__name__}")
        sys.exit(EXIT_BLOCK)

    exempt_reason = _exemption_reason(content)
    if exempt_reason is not None:
        _append_ledger(
            tool_name=tool_name,
            rel_path=rel_path,
            decision="EXEMPT",
            matched_pattern=None,
            exempt_reason=exempt_reason,
            content=content,
        )
        print(
            "WARN: PRIMARY-SOURCE-EXEMPT used \u2014 ensure sources are added before cert/paper submission",
            file=sys.stderr,
        )
        sys.exit(EXIT_ALLOW)

    matched_pattern = _matched_citation_pattern(content)
    if matched_pattern is not None:
        _append_ledger(
            tool_name=tool_name,
            rel_path=rel_path,
            decision="ALLOW",
            matched_pattern=matched_pattern,
            exempt_reason=None,
            content=content,
        )
        sys.exit(EXIT_ALLOW)

    _append_ledger(
        tool_name=tool_name,
        rel_path=rel_path,
        decision="BLOCK",
        matched_pattern=None,
        exempt_reason=None,
        content=content,
    )
    _print_block_message(rel_path)
    sys.exit(EXIT_BLOCK)


if __name__ == "__main__":
    main()
