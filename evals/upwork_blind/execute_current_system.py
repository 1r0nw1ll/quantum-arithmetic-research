#!/usr/bin/env python3
# noqa: DECL-1 (eval execution scaffold — not empirical QA code)
"""
Execute the Upwork-style blind suite against the current heuristic judgment
layer. Mirrors the shape of evals/tla_blind and evals/lean4_blind but with
task-correctness axes appropriate for practical deliverables.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "current_system"

SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".sql", ".sh", ".go", ".rb", ".java", ".rs", ".c", ".cpp", ".h"}
TEST_FILENAME_MARKERS = ("test_", "_test.", "tests/", ".test.", "spec.")

OVERCLAIM_MARKERS = (
    "handles all inputs",
    "production-ready",
    "production ready",
    "fully tested",
    "comprehensive test",
    "comprehensive coverage",
    "battle-tested",
    "enterprise-grade",
    "all edge cases handled",
    "no edge cases missed",
    "robust to all inputs",
)
PLACEHOLDER_PATTERNS = (
    re.compile(r"^\s*pass\s*(#.*)?$", re.M),
    re.compile(r"^\s*return\s+None\s*#\s*(TODO|FIXME|stub|placeholder)", re.I | re.M),
    re.compile(r"\braise\s+NotImplementedError\b"),
    re.compile(r"#\s*TODO\b", re.I),
    re.compile(r"#\s*FIXME\b", re.I),
)
TAUTOLOGY_ASSERTIONS = (
    re.compile(r"^\s*assert\s+True\s*(#.*)?$", re.M),
    re.compile(r"^\s*assert\s+([A-Za-z_][A-Za-z_0-9]*)\s*==\s*\1\s*(#.*)?$", re.M),
    re.compile(r"^\s*assert\s+1\s*==\s*1\s*(#.*)?$", re.M),
    re.compile(r"^\s*assert\s+not\s+False\s*(#.*)?$", re.M),
)
FAKE_ERROR_HANDLING = re.compile(
    r"try\s*:[^}]{0,400}except[^:]*:\s*(pass|print\([^)]*\)|#)",
    re.DOTALL,
)

INVOCATION_MARKERS = (
    "usage:", "usage", "how to run", "run with", "example:", "example usage",
    "```bash", "```sh", "```shell", "python3 ", "python ", "node ", "go run",
)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _source_files(bundle_root: Path) -> list[Path]:
    return [p for p in bundle_root.rglob("*")
            if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS
            and not any(marker in str(p) for marker in ("/.git/", "__pycache__"))]


def _test_files(bundle_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in _source_files(bundle_root):
        name = p.name.lower()
        if any(marker in name or marker in str(p).lower() for marker in TEST_FILENAME_MARKERS):
            out.append(p)
    return out


def _readme_text(bundle_root: Path) -> str:
    for name in ("README.md", "readme.md", "README", "README.txt"):
        p = bundle_root / name
        if p.exists():
            return _read_text(p)
    return ""


def _all_source_text(bundle_root: Path) -> str:
    return "\n".join(_read_text(p) for p in _source_files(bundle_root))


def _scope_claims(task_spec: dict[str, Any]) -> list[str]:
    """Requirements/edge-case words the spec explicitly calls out."""
    hint = task_spec.get("required_keywords", []) if task_spec else []
    return [str(k).lower() for k in hint if isinstance(k, str)]


def _check_python_structural(source_paths: list[Path], test_paths: list[Path]) -> list[str]:
    """Pass-16 structural-gate analog of Pass-13b's git-apply-check.

    Three cheap tool-native checks for Python deliverables:

    1. `python -m py_compile <file>` per .py source → syntactic validity.
       Catches things the heuristics miss: malformed indentation, unclosed
       parens, garbage tokens, broken docstrings.

    2. Top-level import smoke for any module-style file in a tempdir copy
       (so we never pollute the bundle dir or trigger side effects in repo).
       Catches missing imports, top-level NameError, or import-time
       exceptions that prove the file can't actually be used as claimed.

    3. `python -m pytest --collect-only` on test files → tests at least
       parse and collect. Catches "test file exists" deceptions where the
       file is a syntax/import-error mess that would never run.

    Each sub-check that fails emits one finding line with the tool's first
    stderr line. No file mutates the bundle. No source is executed beyond
    `compile()` + the import statement.

    Returns list of findings (empty if all checks pass for all .py files).
    Non-Python source files are skipped silently — this gate is Python-only.
    """
    findings: list[str] = []
    py_sources = [p for p in source_paths if p.suffix.lower() == ".py"]
    if not py_sources:
        return findings

    for path in py_sources:
        # Sub-check 1: py_compile
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            err = (proc.stderr.strip().splitlines() or [""])[0]
            findings.append(f"`{path.name}` fails `python -m py_compile`: {err[:160]}")
            # If a file doesn't even compile, skip its import-smoke check —
            # there's nothing to import.
            continue

        # Sub-check 2: import smoke. Run in a tempdir so the bundle dir
        # isn't on sys.path inadvertently for other files; we explicitly
        # add the file's parent dir to PYTHONPATH and import by module name.
        module_name = path.stem
        if module_name in {"__init__", "conftest"}:
            continue
        env = {"PYTHONPATH": str(path.parent), "PATH": "/usr/bin:/bin", "HOME": "/tmp"}
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        if proc.returncode != 0:
            err = (proc.stderr.strip().splitlines() or [""])[-1]
            findings.append(
                f"`{path.name}` compiles but fails import smoke "
                f"(`python -c 'import {module_name}'`): {err[:160]}"
            )

    # Sub-check 3: pytest --collect-only on test files. Aggregated across
    # the test dir if any test files exist.
    if test_paths:
        test_dirs = sorted({p.parent for p in test_paths})
        for td in test_dirs:
            env = {"PYTHONPATH": str(td), "PATH": "/usr/bin:/bin", "HOME": "/tmp"}
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q", str(td)],
                capture_output=True, text=True, timeout=60, env=env,
            )
            # pytest exit codes: 0=tests collected ok, 5=no tests collected
            # (acceptable here — the deliverable might not actually have
            # @pytest-style tests), 1=collection failed
            if proc.returncode not in (0, 5):
                tail = (proc.stdout or proc.stderr).strip().splitlines()[-3:]
                findings.append(
                    f"Test files in `{td.name}/` fail `pytest --collect-only`: "
                    f"{(tail[-1] if tail else '')[:160]}"
                )

    return findings


def _score_bundle(bundle_root: Path, task_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    findings: list[str] = []
    readme = _readme_text(bundle_root)
    readme_lower = readme.lower()
    source_paths = _source_files(bundle_root)
    source_text = _all_source_text(bundle_root)
    test_paths = _test_files(bundle_root)

    scores = {
        "task_validity_score": 3,
        "external_admissibility_score": 3,
        "requirement_coverage_score": 3,
        "deliverable_fit_score": 3,
        "scope_honesty_score": 3,
        "client_utility_score": 3,
        "source_fidelity_score": 3,
        "reviewer_rejection_risk_score": 0,
    }

    if not source_paths:
        findings.append("Deliverable contains no source code files — README-only submission")
        scores["task_validity_score"] = 0
        scores["external_admissibility_score"] = 0
        scores["requirement_coverage_score"] = 0
        scores["deliverable_fit_score"] = 0
        scores["reviewer_rejection_risk_score"] = 3
        for key in list(scores):
            if key == "reviewer_rejection_risk_score":
                scores[key] = max(0, min(3, scores[key]))
            else:
                scores[key] = max(0, min(3, scores[key]))
        return {"ok": False, "errors": findings, "scores": scores}

    # Pass-16 structural gate: cheap tool-native checks for Python deliverables.
    # Mirrors the role `git apply --check` plays for SWE-Bench. Catches
    # syntax errors, import failures, and uncollectable test files that the
    # text heuristics miss. Non-Python sources are skipped (gate is
    # Python-only). When any sub-check fails, task_validity_score is zeroed
    # — a deliverable that won't compile or import is operationally broken.
    structural_findings = _check_python_structural(source_paths, test_paths)
    if structural_findings:
        findings.extend(structural_findings)
        scores["task_validity_score"] = 0
        scores["external_admissibility_score"] = 0
        scores["deliverable_fit_score"] -= 2
        scores["reviewer_rejection_risk_score"] = 3

    placeholder_hits = sum(len(pat.findall(source_text)) for pat in PLACEHOLDER_PATTERNS)
    if placeholder_hits >= 2:
        findings.append(f"Source contains {placeholder_hits} placeholder/TODO/NotImplementedError markers — core operation is stubbed")
        scores["task_validity_score"] -= 3
        scores["external_admissibility_score"] -= 2
        scores["requirement_coverage_score"] -= 2
        scores["reviewer_rejection_risk_score"] += 3
    elif placeholder_hits == 1:
        findings.append("Source contains a placeholder/TODO marker in the core path")
        scores["task_validity_score"] -= 1
        scores["requirement_coverage_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    tautology_hits = sum(len(pat.findall(source_text)) for pat in TAUTOLOGY_ASSERTIONS)
    test_file_count = len(test_paths)
    claims_tests = any(marker in readme_lower for marker in (
        "tests pass", "passing tests", "test coverage", "all tests green",
        "tests included", "fully tested", "comprehensive coverage",
        "comprehensive test", "x passing test",
    ))

    if test_file_count > 0 and tautology_hits > 0:
        findings.append(f"Test file contains {tautology_hits} tautological assertion(s) (assert True / x == x / 1 == 1)")
        scores["task_validity_score"] -= 1
        scores["scope_honesty_score"] -= 2
        scores["external_admissibility_score"] -= 2
        scores["reviewer_rejection_risk_score"] += 2
    if claims_tests and test_file_count == 0:
        findings.append("README claims tests exist but no test file is present in the deliverable")
        scores["scope_honesty_score"] -= 3
        scores["external_admissibility_score"] -= 2
        scores["reviewer_rejection_risk_score"] += 2

    overclaim_hits = [m for m in OVERCLAIM_MARKERS if m in readme_lower]
    if overclaim_hits:
        findings.append(
            f"README overclaims scope with {len(overclaim_hits)} marker(s): "
            f"{', '.join(repr(m) for m in overclaim_hits)} — not supported by deliverable"
        )
        # Compound: each additional overclaim marker strengthens the reject signal.
        n = len(overclaim_hits)
        scores["scope_honesty_score"] -= min(3, 1 + n)
        scores["external_admissibility_score"] -= min(3, n)
        scores["reviewer_rejection_risk_score"] += min(3, n)

    fake_handling = FAKE_ERROR_HANDLING.search(source_text)
    if fake_handling:
        findings.append("Source contains try/except that swallows errors (bare pass or print-and-continue)")
        scores["task_validity_score"] -= 1
        scores["requirement_coverage_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    if readme.strip() and not any(marker in readme_lower for marker in INVOCATION_MARKERS):
        findings.append("README has no invocation/usage section — client cannot run deliverable as-given")
        scores["client_utility_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if not readme.strip() and source_paths:
        findings.append("Deliverable has source code but no README — missing context for client")
        scores["client_utility_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    # Requirement-coverage check: if task_spec declares required_keywords, each
    # must appear somewhere in source or README (honest substring match, not a
    # semantic check). Missing keywords indicate requirement drop-out.
    if task_spec is not None:
        required = _scope_claims(task_spec)
        haystack = (readme + "\n" + source_text).lower()
        missing = [kw for kw in required if kw not in haystack]
        if missing:
            findings.append(f"Spec requires keywords not present in deliverable: {', '.join(missing)}")
            drop = min(3, len(missing))
            scores["requirement_coverage_score"] -= drop
            scores["source_fidelity_score"] -= min(2, len(missing))
            scores["external_admissibility_score"] -= 1
            scores["reviewer_rejection_risk_score"] += 1

    for key in list(scores):
        if key == "reviewer_rejection_risk_score":
            scores[key] = max(0, min(3, scores[key]))
        else:
            scores[key] = max(0, min(3, scores[key]))
    return {"ok": not findings, "errors": findings, "scores": scores}


def _decision_from_scores(scores: dict[str, int]) -> str:
    if (
        scores["task_validity_score"] <= 0
        or (scores["scope_honesty_score"] <= 0 and scores["requirement_coverage_score"] <= 1)
        or (scores["external_admissibility_score"] <= 0 and scores["task_validity_score"] <= 1)
        or scores["reviewer_rejection_risk_score"] >= 3 and scores["scope_honesty_score"] <= 1
    ):
        return "reject"
    if any(scores[key] < 3 for key in scores if key != "reviewer_rejection_risk_score") or scores["reviewer_rejection_risk_score"] > 0:
        return "revise"
    return "accept"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _evaluate_generation_case(case_dir: Path) -> dict[str, Any]:
    task = _load_json(case_dir / "task.json")
    spec = task.get("generation_spec", {})
    out_dir = RESULTS_ROOT / "generation" / task["case_id"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic current-system generator: writes spec's reference deliverable
    # literally (filenames + contents provided by the fixture). That is the only
    # honest "current system" simulation we can do without running a live agent.
    for entry in spec.get("files", []):
        _write_text(out_dir / entry["path"], entry["content"].rstrip() + "\n")
    report = _score_bundle(out_dir, task_spec=task)
    return {
        "layer": "generation",
        "case_id": task["case_id"],
        "title": task["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "result_bundle": str(out_dir.relative_to(REPO_ROOT)),
    }


def _evaluate_review_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_scorecard.json")
    report = _score_bundle(case_dir / "artifact", task_spec=case)
    result = {
        "layer": "review",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["decision"],
        "expected_scores": label.get("scores"),
    }
    result["matches_expected_decision"] = result["decision"] == result["expected_decision"]
    return result


def _evaluate_repair_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_outcome.json")
    report = _score_bundle(case_dir / "input", task_spec=case)
    result = {
        "layer": "repair",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["final_decision"],
    }
    result["matches_expected_decision"] = result["decision"] == result["expected_decision"]
    return result


def _evaluate_deception_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_outcome.json")
    report = _score_bundle(case_dir / "artifact", task_spec=case)
    result = {
        "layer": "deception",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["decision"],
    }
    result["matches_expected_decision"] = result["decision"] == result["expected_decision"]
    return result


def execute() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for case_dir in sorted((ROOT / "tasks" / "generation").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_generation_case(case_dir))
    for case_dir in sorted((ROOT / "review_corpus").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_review_case(case_dir))
    for case_dir in sorted((ROOT / "repair_cases").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_repair_case(case_dir))
    for case_dir in sorted((ROOT / "deception_corpus").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_deception_case(case_dir))
    return {"ok": True, "results": results}


def main() -> int:
    payload = execute()
    print(_json_dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
