#!/usr/bin/env python3
"""
tools/tests/test_latex_claim_linter.py — regression tests for qa_latex_claim_linter

Run from repo root:
    python tools/tests/test_latex_claim_linter.py

Exit 0 = all pass, exit 1 = any failure.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LINTER = [sys.executable, "tools/qa_latex_claim_linter.py"]

FULL_TRIPWIRES = """\
%% QA_CERT_REQUIRED: QA_PAC_BAYES_DPI_SCOPE_CERT.v1 (family-86)
%% QA_CERT_REQUIRED: QA_PAC_BAYES_CONSTANT_CERT.v1.1 (family-84)
%% QA_CERT_REQUIRED: QA_DQA_PAC_BOUND_KERNEL_CERT.v1 (family-85)
"""


def _tex(body, tripwires=""):
    return f"\\documentclass{{article}}\n\\begin{{document}}\n{body}\n{tripwires}\\end{{document}}\n"


def _run(content, extra_flags=None):
    """Write content to a temp .tex file, run linter, return (exit_code, stdout)."""
    flags = extra_flags or []
    with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as f:
        f.write(content)
        fname = f.name
    try:
        r = subprocess.run(LINTER + flags + [fname], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_file_passes():
    """File with no triggers at all → PASS."""
    code, out = _run(_tex("No sensitive phrases here."))
    assert code == 0, f"Expected exit 0, got {code}.\n{out}"


def test_dpi_trigger_with_tripwires_passes():
    """DPI trigger present + all tripwires → PASS."""
    body = "By the Data Processing Inequality, $D_{QA} \\leq D_{KL}$."
    code, out = _run(_tex(body, FULL_TRIPWIRES))
    assert code == 0, f"Expected exit 0, got {code}.\n{out}"


def test_dpi_trigger_missing_tripwires_fails():
    """DPI trigger present + NO tripwires → FAIL (exit 1)."""
    body = "By the Data Processing Inequality, we bound divergence."
    code, out = _run(_tex(body))
    assert code == 1, f"Expected exit 1 (missing tripwires), got {code}.\n{out}"
    assert "missing" in out.lower() or "tripwire" in out.lower() or "ERROR" in out, \
        f"Expected missing-tripwire error in output.\n{out}"


def test_overclaim_trigger_missing_tripwires_fails():
    """Overclaim phrase + NO tripwires → FAIL."""
    body = "This result is universal and proven for all distributions."
    code, out = _run(_tex(body))
    assert code == 1, f"Expected exit 1, got {code}.\n{out}"


def test_partial_tripwires_fails():
    """Only 1 of 3 tripwires present → FAIL."""
    partial = "%% QA_CERT_REQUIRED: QA_PAC_BAYES_DPI_SCOPE_CERT.v1\n"
    body = "By the Data Processing Inequality."
    code, out = _run(_tex(body, partial))
    assert code == 1, f"Expected exit 1 (partial tripwires), got {code}.\n{out}"


def test_trigger_in_comment_skipped():
    """LaTeX comment line with DPI phrase is NOT a trigger → PASS."""
    body = "% Data Processing Inequality is mentioned in a comment only.\nSafe sentence."
    code, out = _run(_tex(body))
    assert code == 0, f"Expected exit 0 (trigger in comment skipped), got {code}.\n{out}"


def test_strict_overclaim_without_claim_level_fails():
    """--strict: overclaim + tripwires but no QA_CLAIM_LEVEL → FAIL."""
    body = "This result holds for all distributions."
    code, out = _run(_tex(body, FULL_TRIPWIRES), extra_flags=["--strict"])
    assert code == 1, f"Expected exit 1 (strict, missing QA_CLAIM_LEVEL), got {code}.\n{out}"


def test_strict_overclaim_with_claim_level_passes():
    """--strict: overclaim + tripwires + QA_CLAIM_LEVEL: empirical_only → PASS."""
    marker = "%% QA_CLAIM_LEVEL: empirical_only\n"
    body = "This result holds for all distributions."
    code, out = _run(_tex(body, FULL_TRIPWIRES + marker), extra_flags=["--strict"])
    assert code == 0, f"Expected exit 0 (strict with claim level), got {code}.\n{out}"


def test_json_output_parseable():
    """--json flag produces valid JSON with expected keys."""
    import json
    body = "By the DPI, we bound things."
    with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as f:
        f.write(_tex(body, FULL_TRIPWIRES))
        fname = f.name
    try:
        r = subprocess.run(LINTER + ["--json", fname], capture_output=True, text=True)
        data = json.loads(r.stdout)
        assert "summary" in data, f"Missing 'summary' key.\n{r.stdout}"
        assert "files" in data, f"Missing 'files' key.\n{r.stdout}"
        assert data["summary"]["total"] == 1
        assert data["files"][0]["status"] == "PASS"
    finally:
        os.unlink(fname)


def test_real_pac_bayes_paper_passes():
    """The actual PAC-Bayes paper (with tripwires added) must PASS."""
    paper = Path("papers/in-progress/phase1-pac-bayes/phase1_workspace/pac_bayes_qa_theory.tex")
    if not paper.exists():
        print(f"  SKIP test_real_pac_bayes_paper_passes — {paper} not found")
        return
    r = subprocess.run(LINTER + [str(paper)], capture_output=True, text=True)
    assert r.returncode == 0, \
        f"Expected pac_bayes_qa_theory.tex to PASS, got exit {r.returncode}.\n{r.stdout}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_clean_file_passes,
    test_dpi_trigger_with_tripwires_passes,
    test_dpi_trigger_missing_tripwires_fails,
    test_overclaim_trigger_missing_tripwires_fails,
    test_partial_tripwires_fails,
    test_trigger_in_comment_skipped,
    test_strict_overclaim_without_claim_level_fails,
    test_strict_overclaim_with_claim_level_passes,
    test_json_output_parseable,
    test_real_pac_bayes_paper_passes,
]


def main():
    passed = 0
    failed = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERR]  {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  {passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
