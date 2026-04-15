# noqa: DECL-1 (test file)
"""
Regression tests for Claude project-steering hooks outside the cert gate.

Run:
    python llm_qa_wrapper/tests/test_project_steering_hooks.py
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

_results: list[tuple[str, bool, str]] = []


def test(name: str):
    def decorator(fn):
        def runner():
            try:
                fn()
                _results.append((name, True, ""))
                print(f"[PASS] {name}")
            except AssertionError as e:
                _results.append((name, False, str(e)))
                print(f"[FAIL] {name}: {e}")
            except Exception as e:
                _results.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return runner
    return decorator


def _run_hook(path: Path, payload: dict, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        [str(path)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(REPO),
        env=full_env,
    )


@test("pretool_guard blocks protected Documents edits")
def t_pretool_guard_documents():
    hook = REPO / ".claude" / "hooks" / "pretool_guard.sh"
    proc = _run_hook(
        hook,
        {"tool_input": {"file_path": str(REPO / "Documents" / "draft.md")}},
    )
    assert proc.returncode == 2
    assert "protected directory" in proc.stderr


@test("pretool_guard blocks generated root PNG edits")
def t_pretool_guard_png():
    hook = REPO / ".claude" / "hooks" / "pretool_guard.sh"
    proc = _run_hook(
        hook,
        {"tool_input": {"file_path": str(REPO / "figure.png")}},
    )
    assert proc.returncode == 2
    assert "PNG outputs" in proc.stderr


@test("qa_dismissal_guard blocks QA dismissal language")
def t_dismissal_guard_blocks():
    hook = REPO / ".claude" / "hooks" / "qa_dismissal_guard.sh"
    proc = _run_hook(
        hook,
        {"tool_output": "QA is not applicable here; use a standard method instead."},
    )
    assert proc.returncode == 2
    assert "QA DISMISSAL DETECTED" in proc.stderr


@test("qa_dismissal_guard allows neutral QA investigation language")
def t_dismissal_guard_allows():
    hook = REPO / ".claude" / "hooks" / "qa_dismissal_guard.sh"
    proc = _run_hook(
        hook,
        {"tool_output": "Investigate the observer projection and verify the b,e mapping."},
    )
    assert proc.returncode == 0, proc.stderr


@test("collab_file_lock_check blocks shared file when bus state is unavailable")
def t_file_lock_blocks_no_bus():
    hook = REPO / ".claude" / "hooks" / "collab_file_lock_check.sh"
    with tempfile.TemporaryDirectory(prefix="qa_lock_hook_") as tmp:
        marker = Path(tmp) / "marker"
        marker.write_text("claude-test-session", encoding="utf-8")
        proc = _run_hook(
            hook,
            {"tool_input": {"file_path": str(REPO / "CLAUDE.md")}},
            {
                "QA_COLLAB_REPO": str(REPO),
                "QA_COLLAB_MARKER": str(marker),
                "QA_COLLAB_FORCE_NO_BUS": "1",
                "QA_COLLAB_STATE_PORT": "65534",
            },
        )
    assert proc.returncode == 2
    assert "state port" in proc.stderr


@test("collab_file_lock_check ignores non-shared files")
def t_file_lock_ignores_nonshared():
    hook = REPO / ".claude" / "hooks" / "collab_file_lock_check.sh"
    proc = _run_hook(
        hook,
        {"tool_input": {"file_path": str(REPO / "scratch.txt")}},
        {
            "QA_COLLAB_REPO": str(REPO),
            "QA_COLLAB_FORCE_NO_BUS": "1",
            "QA_COLLAB_STATE_PORT": "65534",
        },
    )
    assert proc.returncode == 0, proc.stderr


def main() -> int:
    tests = [
        t_pretool_guard_documents,
        t_pretool_guard_png,
        t_dismissal_guard_blocks,
        t_dismissal_guard_allows,
        t_file_lock_blocks_no_bus,
        t_file_lock_ignores_nonshared,
    ]
    print("=== Claude project steering hook tests ===\n")
    for t in tests:
        t()
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
