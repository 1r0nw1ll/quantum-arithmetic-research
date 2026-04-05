#!/usr/bin/env python3
"""
qa_axiom_linter.py — QA Axiom Compliance Linter

Detects violations of the QA axiom set in Python source files.
Authority: QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1 + QA_AXIOMS_BLOCK.md

Usage:
    python tools/qa_axiom_linter.py [file1.py file2.py ...]
    python tools/qa_axiom_linter.py --all          # scan all .py files
    python tools/qa_axiom_linter.py --staged        # scan git-staged .py files

Exit codes:
    0 = clean (or no QA files found)
    1 = violations found
    2 = usage error

Rule groups:
    T2-b  — Theorem NT: continuous → discrete boundary violations (code-level)
    T2-D  — Theorem NT: design-level violations (stochastic/random graph generators,
              continuous distributions as QA data sources). Added 2026-04-02 after
              repeated violations using SBM/random models to test QA hypotheses.
              Use # noqa: T2-D-N if the random process is an explicit observer
              projection (null model, noise annealing, measurement simulation).
    A1    — No-Zero axiom
    A2    — Derived coordinates
    S1    — No x**2
    S2    — No float state
    DECL  — Missing QA_COMPLIANCE declaration
    ORBIT — Orbit-rule integrity (gaps identified 2026-03-28):
              ORBIT-1  v₃(f) equal to 1 is algebraically impossible → ERROR
              ORBIT-2  orbit_family assigned as string literal (abstract label) → WARNING
              ORBIT-3  orbit_family() called with array-indexed arg (numpy.int64 risk) → WARNING
              ORBIT-4  local orbit_family reimplementation (whole-file) → ERROR
              ORBIT-5  orbit_family() used without qa_orbit_rules import (whole-file) → ERROR
              ORBIT-6  STATE_ALPHABET declared without audit_alphabet call (whole-file) → WARNING
"""

import sys
import re
import os
import subprocess
from pathlib import Path
from typing import NamedTuple

# ── Violation definitions ──────────────────────────────────────────────────────

class ViolationRule(NamedTuple):
    id: str
    axiom: str
    description: str
    pattern: re.Pattern
    severity: str          # ERROR or WARNING
    qa_file_only: bool     # Only applies to files that use QA keywords

# Patterns that indicate a file is doing QA work (not just importing)
QA_INDICATOR_PATTERNS = [
    re.compile(r'orbit_famil|orbit_class|classify_orbit|qa_tuple|QASystem|QAEngine|'
               r'singularity|satellite|cosmos|v_3\(f\)|three_adic|norm_f\('),
]

RULES: list[ViolationRule] = [
    # T2-b: Continuous state injection — the primary violation
    ViolationRule(
        id="T2-b-1",
        axiom="T2-b",
        description="Float multiplication → int cast creates continuous (b,e) state (Theorem NT violation)",
        pattern=re.compile(r'int\s*\(\s*\w+\s*\*\s*\w*\s*(?:modulus|MODULUS|m\b|\d+)\s*\)'),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-b-2",
        axiom="T2-b",
        description="astype(int) on float array likely creates continuous (b,e) state",
        pattern=re.compile(r'\.astype\s*\(\s*int\s*\)'),
        severity="WARNING",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-b-3",
        axiom="T2-b",
        description="float × modulus pattern (amplitude * m → integer state)",
        pattern=re.compile(r'(?:normalized?|amplitude|sample|feature|return|price|signal)\s*[*×]\s*\(?\s*(?:m|modulus|MODULUS|24|9)\s*\)?'),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-b-4",
        axiom="T2-b",
        description="Signal injection into QA state variable (direct float into b or e)",
        pattern=re.compile(r'(?:self\.)?[Bb]\s*=\s*[^=].*\+.*(?:signal|inject|continuous|float)'),
        severity="ERROR",
        qa_file_only=True,
    ),

    # A1: Zero state violation
    ViolationRule(
        id="A1-1",
        axiom="A1",
        description="np.clip with lower bound 0 on QA state produces zero states (A1: states must be in {1,...,N})",
        pattern=re.compile(r'np\.clip\s*\([^)]+,\s*0\s*,'),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="A1-2",
        axiom="A1",
        description="range(0, ...) for QA state iteration starts at 0 (A1: states must be in {1,...,N})",
        pattern=re.compile(r'range\s*\(\s*0\s*,\s*(?:m|modulus|MODULUS|9|24)\s*\)'),
        severity="WARNING",
        qa_file_only=True,
    ),
    ViolationRule(
        id="A1-3",
        axiom="A1",
        description="(b+e) % m produces 0 when sum is multiple of m — use ((b+e-1)%m)+1 instead",
        pattern=re.compile(r'\(\s*[be]\s*\+\s*[be]\s*\)\s*%\s*(?:m|modulus|MODULUS|9|24)'),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="A1-4",
        axiom="A1",
        description="np.random.rand() * modulus initializes float states including 0 (A1 + S2 violation)",
        pattern=re.compile(r'np\.random\.rand\s*\([^)]*\)\s*\*\s*(?:m|modulus|MODULUS|9|24)'),
        severity="ERROR",
        qa_file_only=True,
    ),

    # S1: x**2 in QA arithmetic
    ViolationRule(
        id="S1-1",
        axiom="S1",
        description="x**2 in QA arithmetic — use x*x to avoid libm ULP drift (S1 rule)",
        pattern=re.compile(r'\b[beda]\s*\*\*\s*2\b'),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="S1-2",
        axiom="S1",
        description="pow(x, 2) in QA arithmetic — use x*x (S1 rule)",
        pattern=re.compile(r'pow\s*\(\s*[beda]\s*,\s*2\s*\)'),
        severity="ERROR",
        qa_file_only=True,
    ),

    # S2: Float in QA layer
    ViolationRule(
        id="S2-1",
        axiom="S2",
        description="numpy float array used as QA state (b or e must be int or Fraction)",
        pattern=re.compile(r'(?:self\.)?[be]\s*=\s*np\.(?:zeros|ones|random\.rand|full|empty)\s*\(.*\)(?!\s*\.astype\s*\(\s*int)'),
        severity="ERROR",
        qa_file_only=True,
    ),

    # A2: Independent assignment of derived coordinates
    ViolationRule(
        id="A2-1",
        axiom="A2",
        description="d assigned independently (A2: d must always be derived as d = b+e)",
        pattern=re.compile(r'(?<![beda])\bd\s*=\s*(?!\s*b\s*\+\s*e|\s*b\+e)'),
        severity="WARNING",
        qa_file_only=True,
    ),
    ViolationRule(
        id="A2-2",
        axiom="A2",
        description="a assigned independently (A2: a must always be derived as a = b+2e)",
        pattern=re.compile(r'(?<![beda])\ba\s*=\s*(?!\s*b\s*\+\s*2\s*\*?\s*e|\s*b\+2\*?e)'),
        severity="WARNING",
        qa_file_only=True,
    ),

    # ORBIT-1: v3()==1 is algebraically impossible — always 0, silently wrong
    # v₃(f(b,e)) ∈ {0, 2, ...}, never 1. Using ==1 produces dead code that
    # appears to work (returns empty) but silently mis-classifies orbits.
    ViolationRule(
        id="ORBIT-1",
        axiom="ORBIT",
        description="v\u2083(f)==1 is algebraically impossible (v\u2083(f) is never 1); "
                    "use orbit_family(b,e,m)=='satellite' instead",
        pattern=re.compile(r'v3\s*\([^)]*\)\s*==\s*1(?!\d)|(?<!\d)1\s*==\s*v3\s*\('),
        severity="ERROR",
        qa_file_only=True,
    ),

    # ORBIT-2: orbit_family assigned as string literal without arithmetic verification
    # Cert validators that hardcode "satellite" / "cosmos" strings skip the algebra.
    ViolationRule(
        id="ORBIT-2",
        axiom="ORBIT",
        description="orbit_family assigned as string literal — use orbit_family(b,e,m) "
                    "arithmetic instead of hardcoded label",
        pattern=re.compile(
            r'\borbit_family\s*=\s*["\'](?:satellite|cosmos|singularity)["\']'
            r'|["\']orbit_family["\']\s*:\s*["\'](?:satellite|cosmos|singularity)["\']'
        ),
        severity="WARNING",
        qa_file_only=True,
    ),

    # ORBIT-3: orbit_family() called with array-indexed argument
    # Array indexing returns numpy.int64, not Python int. orbit_family() may raise
    # or silently misbehave. Wrap with int(): orbit_family(int(arr[i]), int(arr[j]), m)
    ViolationRule(
        id="ORBIT-3",
        axiom="ORBIT",
        description="orbit_family() called with array-indexed arg — numpy.int64 is not "
                    "Python int; wrap with int(): orbit_family(int(arr[i]), ...)",
        pattern=re.compile(r'orbit_family\s*\(\s*(?!int\s*\()(\w+)\s*\['),
        severity="WARNING",
        qa_file_only=True,
    ),

    # T2-DESIGN: Stochastic/random graph generators used to produce QA test data.
    # Random graph models (SBM, Erdos-Renyi, Barabasi-Albert) are continuous
    # processes that violate Theorem NT at the DESIGN level: the graph has no QA
    # structure, so QA properties measured on it are meaningless noise.
    # These are ERRORS in QA experiment files, not just warnings.
    ViolationRule(
        id="T2-D-1",
        axiom="T2-D",
        description="stochastic_block_model generates random graphs — T2 design violation: "
                    "use QA-native graph construction (generator moves, harmonic edges) for "
                    "testing QA hypotheses. If used as a NULL MODEL, add # noqa: T2-D-1 and "
                    "declare as observer projection in QA_COMPLIANCE",
        pattern=re.compile(r'stochastic_block_model\s*\('),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-D-2",
        axiom="T2-D",
        description="erdos_renyi_graph generates random graphs — T2 design violation: "
                    "no QA structure to test. Use QA-native graph construction or declare "
                    "as observer projection with # noqa: T2-D-2",
        pattern=re.compile(r'erdos_renyi_graph\s*\(|gnp_random_graph\s*\(|gnm_random_graph\s*\('),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-D-3",
        axiom="T2-D",
        description="barabasi_albert_graph generates random graphs — T2 design violation: "
                    "preferential attachment is a continuous process. Declare as observer "
                    "projection with # noqa: T2-D-3",
        pattern=re.compile(r'barabasi_albert_graph\s*\('),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-D-4",
        axiom="T2-D",
        description="random_partition_graph generates random graphs — T2 design violation",
        pattern=re.compile(r'random_partition_graph\s*\(|planted_partition_graph\s*\('),
        severity="ERROR",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-D-5",
        axiom="T2-D",
        description="np.random used for QA state/graph generation — T2 design violation: "
                    "random processes are observer projections, not QA inputs. If used for "
                    "noise annealing or measurement simulation, add # noqa: T2-D-5",
        pattern=re.compile(r'np\.random\.(?:normal|randn|standard_normal|multivariate_normal)\s*\('),
        severity="WARNING",
        qa_file_only=True,
    ),
    ViolationRule(
        id="T2-D-6",
        axiom="T2-D",
        description="scipy.stats continuous distribution used in QA context — continuous "
                    "functions are observer projections only (Theorem NT). Declare as such "
                    "or add # noqa: T2-D-6",
        pattern=re.compile(r'scipy\.stats\.(?:norm|t|chi2|f|gamma|beta|uniform)\.(?:rvs|pdf|cdf)\s*\('),
        severity="WARNING",
        qa_file_only=True,
    ),
]

# Required declaration block pattern — all QA empirical scripts must declare their observer
QA_COMPLIANCE_DECLARATION = re.compile(r'QA_COMPLIANCE\s*=\s*[\'"{]')

# ── Whole-file orbit-rule patterns ────────────────────────────────────────────

# ORBIT-4: local orbit_family reimplementation
# Any def that shadows the canonical function — but NOT in qa_orbit_rules.py itself.
# Negative lookahead excludes class methods (first param = self or cls).
# Also excludes classify_orbit — that name is allowed for orbit-length helpers that
# take (orbit_length, max_orbit_length), not the canonical (b, e, m) signature.
_ORBIT_REDEF_PATTERN = re.compile(
    r'^\s*def\s+(?:orbit_family|recompute_orbit_family|orbit_class)\s*\(\s*(?!self\b|cls\b)',
    re.MULTILINE,
)

# ORBIT-5: orbit_family used without importing from qa_orbit_rules or qa_arithmetic
_ORBIT_CALL_PATTERN   = re.compile(r'\borbit_family\s*\(')
_ORBIT_IMPORT_PATTERN = re.compile(r'from\s+(qa_orbit_rules|qa_arithmetic)\s+import')

# ORBIT-6: STATE_ALPHABET / MICROSTATE_STATES / WAVE_CLASS_STATES declared
# without a corresponding audit_alphabet() call anywhere in the file.
_ALPHABET_DECL_PATTERN  = re.compile(
    r'\b(?:STATE_ALPHABET|MICROSTATE_STATES|WAVE_CLASS_STATES|FINANCE_STATES)\s*[=:]'
)
_ALPHABET_AUDIT_PATTERN = re.compile(r'\baudit_alphabet\s*\(')

# Canonical orbit-rule files — never flag ORBIT-4/5 here.
# qa_orbit_rules.py is the legacy canonical module; qa_arithmetic/qa_arithmetic/* is
# the new canonical package home (same role, cleaner packaging).
_ORBIT_RULES_FILENAME = "qa_orbit_rules.py"
_CANONICAL_ORBIT_PATH_PARTS = ("qa_arithmetic", "qa_arithmetic")  # matches qa_arithmetic/qa_arithmetic/*

def _is_canonical_orbit_source(path: Path) -> bool:
    """True if this file IS a canonical orbit_family source (legacy or new package)."""
    if path.name == _ORBIT_RULES_FILENAME:
        return True
    parts = path.parts
    # Match any path containing .../qa_arithmetic/qa_arithmetic/... (the package interior)
    for i in range(len(parts) - 1):
        if parts[i] == _CANONICAL_ORBIT_PATH_PARTS[0] and parts[i+1] == _CANONICAL_ORBIT_PATH_PARTS[1]:
            return True
    return False

# ── Core linting logic ────────────────────────────────────────────────────────

def is_qa_file(lines: list[str]) -> bool:
    """Return True if the file contains QA-specific keywords indicating it does QA work."""
    content = "\n".join(lines)
    # A file with QA_COMPLIANCE declaration is definitively a QA file
    if QA_COMPLIANCE_DECLARATION.search(content):
        return True
    return any(p.search(content) for p in QA_INDICATOR_PATTERNS)

def is_comment_or_string(line: str, match_start: int) -> bool:
    """Heuristic: is the match inside a comment or string?"""
    stripped = line[:match_start]
    # Check if preceded by # (comment)
    if "#" in stripped:
        return True
    return False

def lint_file(path: Path) -> list[tuple[int, str, str, str]]:
    """
    Lint a single Python file.
    Returns list of (line_no, rule_id, axiom, description) tuples.
    """
    violations = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [(0, "IO", "IO", f"Cannot read file: {e}")]

    lines = content.splitlines()
    file_is_qa = is_qa_file(lines)
    is_orbit_rules = _is_canonical_orbit_source(path)

    # Check for required declaration block in QA files (hard gate — ERROR)
    if file_is_qa:
        if not QA_COMPLIANCE_DECLARATION.search(content):
            violations.append((1, "DECL-1", "DECL",
                "Missing QA_COMPLIANCE declaration block — empirical QA scripts must declare "
                "their observer and state alphabet (see QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md)"))

    # ORBIT-4: local orbit_family reimplementation (whole-file check)
    # Skip the canonical file itself.
    if not is_orbit_rules and file_is_qa:
        for m in _ORBIT_REDEF_PATTERN.finditer(content):
            lineno = content[:m.start()].count("\n") + 1
            fn_name = re.search(r'def\s+(\w+)', m.group()).group(1)
            violations.append((lineno, "ORBIT-4", "ORBIT",
                f"Local reimplementation of '{fn_name}' — import from qa_orbit_rules instead; "
                "local copies diverge and reintroduce wrong orbit rules"))

    # ORBIT-5: orbit_family() used without importing from qa_orbit_rules (whole-file)
    # Strip def lines first to avoid false-positives from class method/property definitions
    # (e.g. `def orbit_family(self)` is a property, not a canonical orbit call).
    _content_no_defs = re.sub(r'^\s*def\s+\w+\s*\(.*$', '', content, flags=re.MULTILINE)
    if not is_orbit_rules and _ORBIT_CALL_PATTERN.search(_content_no_defs):
        if not _ORBIT_IMPORT_PATTERN.search(content):
            violations.append((1, "ORBIT-5", "ORBIT",
                "orbit_family() called but 'from qa_orbit_rules import' not found — "
                "all orbit classification must use the canonical implementation"))

    # ORBIT-6: STATE_ALPHABET declared without audit_alphabet call (whole-file)
    if file_is_qa and _ALPHABET_DECL_PATTERN.search(content):
        if not _ALPHABET_AUDIT_PATTERN.search(content):
            # Find line number of the first declaration
            m = _ALPHABET_DECL_PATTERN.search(content)
            lineno = content[:m.start()].count("\n") + 1
            # Allow inline suppression on the declaration line itself
            decl_line = lines[lineno - 1] if lineno <= len(lines) else ""
            if "# noqa: ORBIT-6" not in decl_line:
                violations.append((lineno, "ORBIT-6", "ORBIT",
                    "STATE_ALPHABET declared without audit_alphabet() call — run "
                    "audit_alphabet(STATE_ALPHABET, MODULUS) to verify satellite coverage "
                    "before this script can be considered QA-compliant"))

    # Per-line rules
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Support inline suppression: # noqa: RULE-ID[,RULE-ID]
        noqa_ids: set[str] = set()
        noqa_m = re.search(r'#\s*noqa:\s*([\w,-]+)', line)
        if noqa_m:
            noqa_ids = {s.strip() for s in noqa_m.group(1).split(",")}

        for rule in RULES:
            if rule.id in noqa_ids:
                continue
            if rule.qa_file_only and not file_is_qa:
                continue
            m = rule.pattern.search(line)
            if m and not is_comment_or_string(line, m.start()):
                violations.append((lineno, rule.id, rule.axiom, rule.description))

    return violations


def scan_files(paths: list[Path]) -> dict[Path, list]:
    results = {}
    for p in paths:
        if p.suffix == ".py" and p.exists() and p.name not in _EXCLUDE_FILES:
            v = lint_file(p)
            if v:
                results[p] = v
    return results


def get_staged_py_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            text=True
        )
        return [Path(f) for f in out.splitlines() if f.endswith(".py")]
    except subprocess.CalledProcessError:
        return []


_EXCLUDE_DIRS = frozenset({
    ".git", "__pycache__", "archive", ".venv", "venv", "env",
    "node_modules", "site-packages", "dist-packages", "qa_venv",
    ".tox", "build", "dist",
    # Non-cert infrastructure — analysis/ML/paper/lab scripts, not QA cert code.
    # The axiom linter gates cert families (qa_*_cert_v1/) and core tools only.
    # Legacy experiment scripts, qa_lab, and paper verify scripts are excluded.
    "qa_kayser", "qa_synthetic_data", "qa_kona_ebm_mnist_v1",
    "qa_kona_ebm_qa_native_v1", "qa_kona_ebm_qa_native_orbit_reg_v1",
    "papers", "qa_lab", "qa_core", "experiments", "gemini_qa_project",
    "qalm_2.0", "scratch_experiments",
})

# Files that are QA infrastructure, not empirical scripts — exempt from DECL-1
# and self-referential false positives (pattern strings in rule definitions).
_EXCLUDE_FILES = frozenset({
    "qa_axiom_linter.py",              # this file — pattern strings look like violations
    "qa_orbit_rules.py",               # canonical implementation — defines, not uses
    "qa_observer_alphabet_audit.py",   # audit tool itself
    "qa_finance_joint_transition.py",  # explicitly SUPERSEDED — header says noncompliant
    "qa_harmonicity_v2.py",            # legacy experiment script — not cert code
})

def get_all_py_files() -> list[Path]:
    root = Path(__file__).parent.parent
    return [
        p for p in root.rglob("*.py")
        if not any(part in _EXCLUDE_DIRS for part in p.parts)
        and p.parent != root  # skip root-level experiment scripts
    ]


def print_report(results: dict) -> int:
    """Print violations and return exit code."""
    if not results:
        print("qa_axiom_linter: CLEAN — no violations found")
        return 0

    error_count = 0
    warning_count = 0

    for path, violations in sorted(results.items()):
        print(f"\n{path}")
        for lineno, rule_id, axiom, description in violations:
            severity = next((r.severity for r in RULES if r.id == rule_id), "WARNING")
            if rule_id.startswith("DECL-"):
                severity = "ERROR"   # Hard gate — declaration is mandatory
            if rule_id in ("ORBIT-4", "ORBIT-5", "ORBIT-1"):
                severity = "ERROR"   # Orbit reimplementation + v3==1 are hard errors
            icon = "✗" if severity == "ERROR" else "⚠"
            print(f"  {icon} line {lineno:4d}  [{rule_id}] [{axiom}]  {description}")
            if severity == "ERROR":
                error_count += 1
            else:
                warning_count += 1

    print(f"\nqa_axiom_linter: {error_count} error(s), {warning_count} warning(s) in {len(results)} file(s)")
    print("Authority: QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1 | QA_AXIOMS_BLOCK.md")

    return 1 if error_count > 0 else 0


def main() -> int:
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    if "--staged" in args:
        files = get_staged_py_files()
        if not files:
            print("qa_axiom_linter: no staged .py files")
            return 0
    elif "--all" in args:
        files = get_all_py_files()
    else:
        files = [Path(a) for a in args if a.endswith(".py")]

    results = scan_files(files)
    return print_report(results)


if __name__ == "__main__":
    sys.exit(main())
