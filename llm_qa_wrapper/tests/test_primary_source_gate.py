# noqa: DECL-1 (hook test)
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "llm_qa_wrapper" / "kernel" / "primary_source_gate.py"


def _run_hook(tmp_path: Path, payload: dict) -> subprocess.CompletedProcess[str]:
    test_repo = tmp_path / "repo"
    test_repo.mkdir()
    ledger = tmp_path / "ledger" / "primary_source_gate.jsonl"
    env = os.environ.copy()
    env["LLM_QA_WRAPPER_REPO"] = str(test_repo)
    env["LLM_QA_WRAPPER_PRIMARY_SOURCE_LEDGER"] = str(ledger)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(test_repo),
        env=env,
    )


def _write_payload(rel_path: str, content: str) -> dict:
    return {
        "tool_name": "Write",
        "tool_input": {
            "file_path": rel_path,
            "content": content,
        },
    }


def test_allow_with_primary_source(tmp_path: Path) -> None:
    proc = _run_hook(
        tmp_path,
        _write_payload("docs/theory/QA_TEST.md", "Primary source: Wildberger 2025 AMM"),
    )
    assert proc.returncode == 0, proc.stderr


def test_allow_with_arxiv_url(tmp_path: Path) -> None:
    proc = _run_hook(
        tmp_path,
        _write_payload("docs/theory/QA_TEST.md", "See arxiv.org/abs/1602.07576."),
    )
    assert proc.returncode == 0, proc.stderr


def test_allow_with_references_section(tmp_path: Path) -> None:
    proc = _run_hook(
        tmp_path,
        _write_payload(
            "docs/theory/QA_TEST.md",
            "# Note\n\n## References\n- Cohen & Welling (2016). Group equivariant CNNs.",
        ),
    )
    assert proc.returncode == 0, proc.stderr


def test_block_empty_scaffold(tmp_path: Path) -> None:
    proc = _run_hook(tmp_path, _write_payload("docs/theory/QA_TEST.md", "# Untitled\n"))
    assert proc.returncode == 2
    assert "PRIMARY_SOURCE_REQUIRED" in proc.stderr


def test_block_prose_only_no_citation(tmp_path: Path) -> None:
    proc = _run_hook(
        tmp_path,
        _write_payload(
            "docs/theory/QA_TEST.md",
            "This theory note proposes a mapping without citing any source.",
        ),
    )
    assert proc.returncode == 2
    assert "no primary-source citation matched" in proc.stderr


def test_exempt_allow_with_marker(tmp_path: Path) -> None:
    proc = _run_hook(
        tmp_path,
        _write_payload(
            "docs/theory/QA_TEST.md",
            '<!-- PRIMARY-SOURCE-EXEMPT: reason="staged write" approver="Will" ts="2026-04-14T00:00:00Z" -->\n# Draft',
        ),
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARN: PRIMARY-SOURCE-EXEMPT used" in proc.stderr


def test_out_of_scope_pass_through(tmp_path: Path) -> None:
    proc = _run_hook(
        tmp_path,
        _write_payload("qa_observer/qa_observer/core.py", "print('no citation needed')\n"),
    )
    assert proc.returncode == 0, proc.stderr
