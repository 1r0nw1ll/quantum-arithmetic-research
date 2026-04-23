# noqa: DECL-1 (test file)
"""
Regression tests for cert_gate_hook.py as the live Claude PreToolUse gate.

Run:
    python llm_qa_wrapper/tests/test_cert_gate_hook.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "llm_qa_wrapper" / "cert_gate_hook.py"
QUARANTINE_REVIEW = REPO / "tools" / "qa_codex_quarantine_review.py"
GENESIS = bytes(32)

_results: list[tuple[str, bool, str]] = []


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _compute_self_hash(
    agent: str,
    tool: str,
    payload_hash: bytes,
    prev: bytes,
    counter: int,
) -> bytes:
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


def _records(ledger_dir: Path) -> list[dict[str, Any]]:
    path = ledger_dir / "enforced.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _verify_chain(rows: list[dict[str, Any]]) -> None:
    prev = GENESIS
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        payload_hash = bytes.fromhex(row["payload_hash"])
        self_hash = bytes.fromhex(row["self_hash"])
        assert row["counter"] == idx, f"counter mismatch at row {idx}"
        assert bytes.fromhex(row["prev"]) == prev, f"prev mismatch at row {idx}"
        expected = _compute_self_hash(
            row["agent"],
            row["tool"],
            payload_hash,
            prev,
            idx,
        )
        assert self_hash == expected, f"self_hash mismatch at row {idx}"
        assert row["self_hash"] not in seen, f"duplicate self_hash at row {idx}"
        seen.add(row["self_hash"])
        prev = self_hash


def _run_hook(
    ledger_dir: Path,
    payload: str,
    *,
    marker_text: str | None = None,
    extra_env: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LLM_QA_WRAPPER_LEDGER_DIR"] = str(ledger_dir)
    repo = repo_root or REPO
    env["LLM_QA_WRAPPER_REPO"] = str(repo)
    env["LLM_QA_WRAPPER_QUARANTINE_DIR"] = str(ledger_dir / "quarantine")
    marker = ledger_dir / "collab_marker"
    if marker_text is not None:
        marker.write_text(marker_text, encoding="utf-8")
    env["LLM_QA_WRAPPER_COLLAB_MARKER"] = str(marker)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        cwd=str(repo),
    )


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


@test("allowed Bash exits 0 and appends ALLOW cert")
def t_allow_bash():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "sed -n '1,10p' README.md"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        _verify_chain(rows)


@test("dangerous Bash exits 2 and appends DENY cert")
def t_deny_bash():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "rm -rf /"},
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "DESTRUCTIVE_RM_RECURSIVE_FORCE" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("protected wrapper edit exits 2")
def t_deny_wrapper_edit():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {"file_path": str(HOOK)},
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "WRAPPER_SELF_MODIFICATION" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("cert-adjacent edit without collab marker exits 2")
def t_deny_cert_adjacent_without_marker():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO / "qa_alphageometry_ptolemy" / "qa_meta_validator.py")
                },
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "CERT_COLLAB_MARKER_MISSING" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("Outside repo write reaches allow path")
def t_allow_outside_repo_write():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        outside_file = ledger_dir / "outside" / "notes.md"
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {"file_path": str(outside_file)},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        _verify_chain(rows)


@test("Frozen finance outside repo write remains blocked")
def t_deny_frozen_finance_outside_repo_write():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {"file_path": "/home/player2/Desktop/qa_finance/report.md"},
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "FROZEN_QA_FINANCE" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("Python edit quarantines and allows write")
def t_deny_python_edit():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO / "some_experiment.py")
                },
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        packet = json.loads(packets[0].read_text(encoding="utf-8"))
        assert packet["schema_version"] == "QA_CLAUDE_PYTHON_QUARANTINE.v1"
        assert packet["review_status"] == "pending_codex_review"
        assert packet["original_exists"] is False
        assert packet["original_snapshot_available"] is True
        _verify_chain(rows)


@test("TLA edit quarantines and allows write")
def t_quarantine_tla_edit():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO / "qa_alphageometry_ptolemy" / "Spec.tla")
                },
            }),
            marker_text="claude-test-session",
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["enforcement_markers"] == ["CLAUDE_FORMAL_PUBLICATION_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        packet = json.loads(packets[0].read_text(encoding="utf-8"))
        assert packet["schema_version"] == "QA_CLAUDE_FORMAL_PUBLICATION_QUARANTINE.v1"
        assert packet["review_marker"] == "CLAUDE_FORMAL_PUBLICATION_WRITE_QUARANTINED"
        _verify_chain(rows)


@test("nested README above descendant TLA quarantines and allows write")
def t_quarantine_nested_formal_readme():
    with tempfile.TemporaryDirectory(prefix="qa_hook_repo_") as tmp_repo, tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        repo_root = Path(tmp_repo)
        ledger_dir = Path(tmp)
        nested_dir = repo_root / "docs" / "examples" / "formal_case"
        (nested_dir / "spec").mkdir(parents=True, exist_ok=True)
        (nested_dir / "spec" / "Spec.tla").write_text("---- MODULE Spec ----\n====\n", encoding="utf-8")
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {"file_path": str(nested_dir / "README.md")},
            }),
            marker_text="claude-test-session",
            repo_root=repo_root,
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert rows[0]["enforcement_markers"] == ["CLAUDE_FORMAL_PUBLICATION_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        _verify_chain(rows)


@test("Python edit env override also quarantines and allows write")
def t_deny_python_edit_env_override():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO / "some_experiment.py")
                },
            }),
            extra_env={"LLM_QA_ALLOW_CLAUDE_PYTHON_EDIT": "1"},
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        _verify_chain(rows)


@test("Bash Python mutation quarantines and allows write")
def t_deny_bash_python_mutation():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "python -c \"open('x.py','w').write('bad')\""},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        _verify_chain(rows)


@test("Bash Python mutation env override also quarantines and allows write")
def t_deny_bash_python_mutation_env_override():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "python -c \"open('x.py','w').write('bad')\""},
            }),
            extra_env={"LLM_QA_ALLOW_CLAUDE_PYTHON_EDIT": "1"},
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        _verify_chain(rows)


@test("Bash git add and commit mentioning Python does not create extra quarantine packet")
def t_git_add_commit_python_path_blocks_without_extra_quarantine():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        pending = ledger_dir / "quarantine" / "pending"
        pending.mkdir(parents=True)
        (pending / "pending.json").write_text(
            json.dumps({
                "schema_version": "QA_CLAUDE_PYTHON_QUARANTINE.v1",
                "review_status": "pending_codex_review",
            }),
            encoding="utf-8",
        )
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {
                    "command": "git add tools/qa_guardrail_smoke.py && git commit -m smoke"
                },
            }),
            marker_text="claude-test-session",
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "CODEX_REVIEW_PENDING" in rows[0]["deny_reasons"]
        assert rows[0]["enforcement_markers"] == []
        assert len(list(pending.glob("*.json"))) == 1
        _verify_chain(rows)


def _init_temp_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "QA Hook Test"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "qa-hook@example.com"], cwd=repo_root, check=True, capture_output=True, text=True)
    (repo_root / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=repo_root, check=True, capture_output=True, text=True)


def _write_minimal_formal_bundle(bundle_root: Path) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "Spec.tla").write_text(
        "---- MODULE Spec ----\nVARIABLE x\nInit == x = 0\nNext == x' = x + 1\nInv_Semantic == x >= 0\n====\n",
        encoding="utf-8",
    )
    (bundle_root / "audience_translation.md").write_text(
        "What is modeled: a bounded counter state machine in TLA+.\n"
        "Why useful: it shows the modeled state and why an outsider should care.\n"
        "This is written for formal-methods readers and explains the mapping.\n",
        encoding="utf-8",
    )
    (bundle_root / "semantics_boundary.md").write_text(
        "Intrinsic semantics: the counter evolves by Next and the invariant constrains states.\n"
        "TLC bounds: model checking may bound the search depth or state cap, but those bounds are not semantics.\n",
        encoding="utf-8",
    )
    (bundle_root / "repo_fit_review.json").write_text(
        json.dumps({
            "target_repo": "tlaplus/examples",
            "comparables": ["Counter.tla", "Clock.tla"],
            "why_belongs": "Matches small illustrative examples",
            "maintainer_value": "Shows a minimal outsider-readable transition system",
        }, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (bundle_root / "skeptical_review.json").write_text(
        json.dumps({
            "reviewer": "hostile-maintainer-sim",
            "recommendation": "revise",
            "rejection_arguments": [
                "A maintainer may ask for clearer semantics wording and explicit outsider translation.",
                "Repository fit should be tied to example norms rather than asserted by style alone.",
            ],
        }, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (bundle_root / "source_grounding.json").write_text(
        json.dumps({
            "entries": [
                {
                    "claim": "The module models a bounded counter.",
                    "artifact_element": "variable:x",
                    "source_ref": "visible task statement::counter",
                    "source_excerpt": "The task describes a bounded counter value and its state.",
                    "interpretation": "The variable x tracks the current counter value.",
                    "modeled_consequence": "The model state includes the current counter value.",
                    "authority_tier": "formalism",
                },
                {
                    "claim": "The module uses a state transition.",
                    "artifact_element": "action:Next",
                    "source_ref": "visible task statement::counter",
                    "source_excerpt": "The task describes the counter transition relation.",
                    "interpretation": "The Next action advances the counter state.",
                    "modeled_consequence": "The model transition updates the counter.",
                    "authority_tier": "formalism",
                },
                {
                    "claim": "The intrinsic semantics come from the task statement.",
                    "artifact_element": "semantics:intrinsic",
                    "source_ref": "visible task statement::counter",
                    "source_excerpt": "The task states the meaning of the counter transitions.",
                    "interpretation": "The intrinsic semantics are the counter transitions given in the task.",
                    "modeled_consequence": "The model meaning comes from the visible task statement.",
                    "authority_tier": "formalism",
                },
            ],
        }, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (bundle_root / "repo_comparables.json").write_text(
        json.dumps({
            "target_repo": "tlaplus/examples",
            "candidate_scope": "Small outsider-readable bounded counter example",
            "in_scope_rationale": "The example matches the scope and audience of small public teaching examples.",
            "comparables": [
                {
                    "artifact_name": "Clock.tla",
                    "artifact_ref": "tlaplus/examples/Clock.tla",
                    "norm_supported": "Supports simple state-machine structure and outsider-readable README norms.",
                    "similarity_axes": ["structure", "audience", "readme"],
                },
                {
                    "artifact_name": "CounterExample.tla",
                    "artifact_ref": "tlaplus/examples/CounterExample.tla",
                    "norm_supported": "Supports bounded-state educational example scope.",
                    "similarity_axes": ["structure", "semantics"],
                },
            ],
        }, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


@test("git commit blocks formal changes when required artifacts are absent")
def t_git_commit_blocks_without_formal_gate():
    with tempfile.TemporaryDirectory(prefix="qa_hook_repo_") as tmp_repo, tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        repo_root = Path(tmp_repo)
        ledger_dir = Path(tmp)
        _init_temp_git_repo(repo_root)
        bundle_root = repo_root / "qa_alphageometry_ptolemy"
        bundle_root.mkdir(parents=True, exist_ok=True)
        (bundle_root / "Spec.tla").write_text("---- MODULE Spec ----\n====\n", encoding="utf-8")
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m formal"},
            }),
            marker_text="claude-test-session",
            extra_env={"LLM_QA_FORMAL_CHANGED_PATHS": "qa_alphageometry_ptolemy/Spec.tla"},
            repo_root=repo_root,
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert rows[0]["decision"] == "DENY"
        assert "FORMAL_PUBLICATION_GATE_REQUIRED" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("git push blocks formal changes when required artifacts are absent")
def t_git_push_blocks_without_formal_gate():
    with tempfile.TemporaryDirectory(prefix="qa_hook_repo_") as tmp_repo, tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        repo_root = Path(tmp_repo)
        ledger_dir = Path(tmp)
        _init_temp_git_repo(repo_root)
        bundle_root = repo_root / "qa_alphageometry_ptolemy"
        bundle_root.mkdir(parents=True, exist_ok=True)
        (bundle_root / "Spec.tla").write_text("---- MODULE Spec ----\n====\n", encoding="utf-8")
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git push origin HEAD"},
            }),
            marker_text="claude-test-session",
            extra_env={"LLM_QA_FORMAL_CHANGED_PATHS": "qa_alphageometry_ptolemy/Spec.tla"},
            repo_root=repo_root,
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert rows[0]["decision"] == "DENY"
        assert "FORMAL_PUBLICATION_GATE_REQUIRED" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("git commit blocks formal submission text changes when required artifacts are absent")
def t_git_commit_blocks_submission_text_without_gate():
    with tempfile.TemporaryDirectory(prefix="qa_hook_repo_") as tmp_repo, tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        repo_root = Path(tmp_repo)
        ledger_dir = Path(tmp)
        _init_temp_git_repo(repo_root)
        bundle_root = repo_root / "qa_alphageometry_ptolemy"
        bundle_root.mkdir(parents=True, exist_ok=True)
        (bundle_root / "submission.txt").write_text("publication-ready\n", encoding="utf-8")
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m formal submission"},
            }),
            marker_text="claude-test-session",
            extra_env={"LLM_QA_FORMAL_CHANGED_PATHS": "qa_alphageometry_ptolemy/submission.txt"},
            repo_root=repo_root,
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert rows[0]["decision"] == "DENY"
        assert "FORMAL_PUBLICATION_GATE_REQUIRED" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("git commit blocks formal changes while formal quarantine review is pending")
def t_git_commit_blocks_pending_formal_review():
    with tempfile.TemporaryDirectory(prefix="qa_hook_repo_") as tmp_repo, tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        repo_root = Path(tmp_repo)
        ledger_dir = Path(tmp)
        _init_temp_git_repo(repo_root)
        _write_minimal_formal_bundle(repo_root / "qa_alphageometry_ptolemy")
        pending = ledger_dir / "quarantine" / "pending"
        pending.mkdir(parents=True)
        (pending / "formal_pending.json").write_text(
            json.dumps({
                "schema_version": "QA_CLAUDE_FORMAL_PUBLICATION_QUARANTINE.v1",
                "review_status": "pending_codex_review",
                "review_marker": "CLAUDE_FORMAL_PUBLICATION_WRITE_QUARANTINED",
            }),
            encoding="utf-8",
        )
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m formal"},
            }),
            marker_text="claude-test-session",
            extra_env={"LLM_QA_FORMAL_CHANGED_PATHS": "qa_alphageometry_ptolemy/Spec.tla"},
            repo_root=repo_root,
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert rows[0]["decision"] == "DENY"
        assert "FORMAL_REVIEW_PENDING" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("Bash read-only Python path reference does not quarantine")
def t_allow_read_only_python_path_reference():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "sed -n '1,20p' tools/qa_guardrail_smoke.py"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == []
        assert not (ledger_dir / "quarantine" / "pending").exists()
        _verify_chain(rows)


@test("Bash fd-to-fd redirect on Python read does not quarantine")
def t_allow_fd_redirect_python_read():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "head foo.py 2>&1"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == []
        assert not (ledger_dir / "quarantine" / "pending").exists()
        _verify_chain(rows)


@test("Bash stderr /dev/null redirect does not quarantine")
def t_allow_stderr_devnull_redirect():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "ls dir/ 2>/dev/null"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == []
        assert not (ledger_dir / "quarantine" / "pending").exists()
        _verify_chain(rows)


@test("Bash read-only python command with fd redirect does not quarantine")
def t_allow_python_command_fd_redirect_read_only():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "python -c \"print(1)\" foo.py 2>&1"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == []
        assert not (ledger_dir / "quarantine" / "pending").exists()
        _verify_chain(rows)


@test("Bash redirection to Python path still quarantines")
def t_bash_redirect_to_python_path_quarantines():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "printf 'x = 1\\n' > tools/qa_guardrail_smoke.py"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        _verify_chain(rows)


@test("Bash cat redirect to Python path still quarantines")
def t_bash_cat_redirect_to_python_path_quarantines():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "cat src.py > dst.py"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        _verify_chain(rows)


@test("Bash Python path mutation with stderr redirect still quarantines")
def t_bash_python_path_mutation_with_stderr_redirect_quarantines():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "echo x > foo.py 2>&1"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        _verify_chain(rows)


@test("Bash git commit with <email> in message does not quarantine when body mentions .py")
def t_allow_git_commit_email_bracket_with_py_in_body():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        # Write the collab marker so GIT_COMMIT_WITHOUT_COLLAB_MARKER does not fire.
        (ledger_dir / "collab_marker").write_text("claude-main-test", encoding="utf-8")
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        "git commit -m \"docs: note about cert_gate_hook.py\n\n"
                        "Co-Authored-By: Claude <noreply@anthropic.com>\""
                    )
                },
            }),
            extra_env={
                "QA_COLLAB_MARKER_PATH": str(ledger_dir / "collab_marker"),
                "LLM_QA_FORMAL_CHANGED_PATHS": "",
            },
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == []
        assert not (ledger_dir / "quarantine" / "pending").exists()
        _verify_chain(rows)


@test("Bash python -c with > inside string literal does not quarantine")
def t_allow_python_dash_c_with_redirect_in_string():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {
                    "command": "python3 -c \"print('a > b in foo.py'); print(1)\""
                },
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == []
        assert not (ledger_dir / "quarantine" / "pending").exists()
        _verify_chain(rows)


@test("Bash python -c with os.remove on .py still quarantines")
def t_python_dash_c_os_remove_still_quarantines():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "python3 -c \"import os; os.remove('foo.py')\""},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["enforcement_markers"] == ["CLAUDE_PYTHON_WRITE_QUARANTINED"]
        packets = sorted((ledger_dir / "quarantine" / "pending").glob("*.json"))
        assert len(packets) == 1
        _verify_chain(rows)


@test("Bash combined stdout stderr devnull redirect does not quarantine")
def t_allow_combined_devnull_redirect():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "command &>/dev/null"},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        assert rows[0]["enforcement_markers"] == []
        assert not (ledger_dir / "quarantine" / "pending").exists()
        _verify_chain(rows)


@test("Codex delegation heredoc with Python task text reaches allow path")
def t_allow_codex_delegation_heredoc_python_task_text():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        command = """python3 qa_lab/qa_agents/cli/collab_request.py --to codex <<'PROMPT_EOF'
Edit /home/player2/signal_experiments/tools/qa_retrieval/query.py to add sector rerank.
Use an affinity ladder: exact / role-diagonal / shared-axis / none -> 1.0 / 0.5 / 0.25 / 0.0.
PROMPT_EOF"""
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": command},
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        _verify_chain(rows)


@test("Bash protected directory mutation exits 2")
def t_deny_bash_protected_mutation():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "mv archive/old archive/new"},
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "PROTECTED_TARGET_MUTATION" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("Documents corpus PDF Write is allowed")
def t_allow_documents_corpus_pdf_write():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO / "Documents" / "haramein_rsf" / "new_primary.pdf"),
                    "content": "%PDF-1.4\n",
                },
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        _verify_chain(rows)


@test("Documents non-PDF Write remains denied")
def t_deny_documents_non_pdf_write():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO / "Documents" / "haramein_rsf" / "notes.md"),
                    "content": "not allowed",
                },
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "PROTECTED_PROJECT_DIRECTORY" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("Documents corpus PDF Bash mutation is allowed")
def t_allow_documents_corpus_pdf_bash_mutation():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {
                    "command": "cp /tmp/volk.pdf Documents/wildberger_corpus/volk_primary.pdf",
                },
            }),
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        assert rows[0]["deny_reasons"] == []
        _verify_chain(rows)


@test("Documents directory-level Bash mutation remains denied")
def t_deny_documents_directory_bash_mutation():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {
                    "command": "rm -rf Documents/haramein_rsf",
                },
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "PROTECTED_TARGET_MUTATION" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("git force push exits 2")
def t_deny_force_push():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git push --force-with-lease origin main"},
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "GIT_FORCE_PUSH_FORBIDDEN" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("git commit without collab marker exits 2")
def t_deny_commit_without_marker():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m test"},
            }),
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "GIT_COMMIT_WITHOUT_COLLAB_MARKER" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("git commit with collab marker reaches allow path")
def t_allow_commit_with_marker():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m test"},
            }),
            marker_text="claude-test-session",
            extra_env={"LLM_QA_FORMAL_CHANGED_PATHS": ""},
        )
        assert proc.returncode == 0, proc.stderr
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "ALLOW"
        _verify_chain(rows)


@test("git commit with pending quarantine exits 2")
def t_deny_commit_with_pending_quarantine():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        pending = ledger_dir / "quarantine" / "pending"
        pending.mkdir(parents=True)
        (pending / "pending.json").write_text(
            json.dumps({
                "schema_version": "QA_CLAUDE_PYTHON_QUARANTINE.v1",
                "review_status": "pending_codex_review",
            }),
            encoding="utf-8",
        )
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m test"},
            }),
            marker_text="claude-test-session",
        )
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert "CODEX_REVIEW_PENDING" in rows[0]["deny_reasons"]
        _verify_chain(rows)


@test("reject review restores original Python file")
def t_reject_review_restores_original_file():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        temp_repo = ledger_dir / "repo"
        target = temp_repo / "experiment.py"
        temp_repo.mkdir(parents=True)
        target.write_text("original = True\n", encoding="utf-8")
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {"file_path": str(target)},
            }),
            extra_env={"LLM_QA_WRAPPER_REPO": str(temp_repo)},
        )
        assert proc.returncode == 0, proc.stderr
        target.write_text("mutated = True\n", encoding="utf-8")
        review = subprocess.run(
            [
                sys.executable,
                str(QUARANTINE_REVIEW),
                "--quarantine-dir",
                str(ledger_dir / "quarantine"),
                "reject",
                "--all",
                "--notes",
                "test rejection",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert review.returncode == 0, review.stderr
        assert target.read_text(encoding="utf-8") == "original = True\n"
        assert not list((ledger_dir / "quarantine" / "pending").glob("*.json"))
        reviewed = json.loads(review.stdout)
        assert reviewed["reviewed"][0]["rollback"]["status"] == "restored_original"


@test("reject review removes new Python file")
def t_reject_review_removes_new_file():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        temp_repo = ledger_dir / "repo"
        target = temp_repo / "new_experiment.py"
        temp_repo.mkdir(parents=True)
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Write",
                "tool_input": {"file_path": str(target)},
            }),
            extra_env={"LLM_QA_WRAPPER_REPO": str(temp_repo)},
        )
        assert proc.returncode == 0, proc.stderr
        target.write_text("created = True\n", encoding="utf-8")
        review = subprocess.run(
            [
                sys.executable,
                str(QUARANTINE_REVIEW),
                "--quarantine-dir",
                str(ledger_dir / "quarantine"),
                "reject",
                "--all",
                "--notes",
                "test rejection",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert review.returncode == 0, review.stderr
        assert not target.exists()
        reviewed = json.loads(review.stdout)
        assert reviewed["reviewed"][0]["rollback"]["status"] == "removed_new_file"


@test("malformed JSON exits 2 and records parse denial")
def t_deny_malformed_json():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        proc = _run_hook(ledger_dir, "{not-json")
        assert proc.returncode == 2
        rows = _records(ledger_dir)
        assert len(rows) == 1
        assert rows[0]["decision"] == "DENY"
        assert rows[0]["tool"] == "unknown"
        assert any(r.startswith("HOOK_JSON_PARSE_ERROR") for r in rows[0]["deny_reasons"])
        _verify_chain(rows)


@test("corrupt existing ledger blocks even an otherwise allowed call")
def t_corrupt_ledger_blocks():
    with tempfile.TemporaryDirectory(prefix="qa_hook_test_") as tmp:
        ledger_dir = Path(tmp)
        ledger_dir.mkdir(parents=True, exist_ok=True)
        (ledger_dir / "enforced.jsonl").write_text('{"not":"a cert"}\n', encoding="utf-8")
        proc = _run_hook(
            ledger_dir,
            json.dumps({
                "tool_name": "Bash",
                "tool_input": {"command": "sed -n '1,10p' README.md"},
            }),
        )
        assert proc.returncode == 2
        assert "CERT_LEDGER_FAILURE" in proc.stderr


def main() -> int:
    tests = [
        t_allow_bash,
        t_deny_bash,
        t_deny_wrapper_edit,
        t_deny_cert_adjacent_without_marker,
        t_allow_outside_repo_write,
        t_deny_frozen_finance_outside_repo_write,
        t_deny_python_edit,
        t_quarantine_tla_edit,
        t_quarantine_nested_formal_readme,
        t_deny_python_edit_env_override,
        t_deny_bash_python_mutation,
        t_deny_bash_python_mutation_env_override,
        t_git_add_commit_python_path_blocks_without_extra_quarantine,
        t_allow_read_only_python_path_reference,
        t_allow_fd_redirect_python_read,
        t_allow_stderr_devnull_redirect,
        t_allow_python_command_fd_redirect_read_only,
        t_bash_redirect_to_python_path_quarantines,
        t_bash_cat_redirect_to_python_path_quarantines,
        t_bash_python_path_mutation_with_stderr_redirect_quarantines,
        t_allow_git_commit_email_bracket_with_py_in_body,
        t_allow_python_dash_c_with_redirect_in_string,
        t_python_dash_c_os_remove_still_quarantines,
        t_allow_combined_devnull_redirect,
        t_allow_codex_delegation_heredoc_python_task_text,
        t_deny_bash_protected_mutation,
        t_allow_documents_corpus_pdf_write,
        t_deny_documents_non_pdf_write,
        t_allow_documents_corpus_pdf_bash_mutation,
        t_deny_documents_directory_bash_mutation,
        t_deny_force_push,
        t_deny_commit_without_marker,
        t_allow_commit_with_marker,
        t_deny_commit_with_pending_quarantine,
        t_git_commit_blocks_without_formal_gate,
        t_git_push_blocks_without_formal_gate,
        t_git_commit_blocks_pending_formal_review,
        t_git_commit_blocks_submission_text_without_gate,
        t_reject_review_restores_original_file,
        t_reject_review_removes_new_file,
        t_deny_malformed_json,
        t_corrupt_ledger_blocks,
    ]
    print("=== LLM QA Wrapper PreToolUse hook tests ===\n")
    for t in tests:
        t()
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
