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
    ELEM  — Element computation enforcement (raw vs mod-reduced):
              ELEM-1  Modular reduction inside element (C/F/G) computation → ERROR
              ELEM-4  norm_f() or invariant from mod-reduced input → WARNING
              ELEM-5  Hardcoded (b,e) lookup table (CMAP/MICROSTATE_STATES) → WARNING
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
# Group A: orbit/state keywords (original indicators).
# Group B: cert-module / validator / schema keywords — added 2026-04-11 after
#          the FST v1 audit revealed that validator source files containing
#          observer/QA-layer firewall crossings were being skipped entirely
#          because they lacked Group-A keywords. Validator sources for
#          schema-registered cert modules must be linted regardless of
#          whether they mention orbit classification directly.
QA_INDICATOR_PATTERNS = [
    re.compile(r'orbit_famil|orbit_class|classify_orbit|qa_tuple|QASystem|QAEngine|'
               r'singularity|satellite|cosmos|v_3\(f\)|three_adic|norm_f\('),
    re.compile(r'QA_MAP_MODULE_SPINE|QA_CERT_BUNDLE|QA_SUBMISSION_PACKET_SPINE|'
               r'QA_RUN_ARTIFACT_BUNDLE|QA_SHA256_MANIFEST|'
               r'qa\.cert\.|qa\.map\.|qa\.manifest\.|'
               r'validator_contract|module_spine|cert_bundle|'
               r'fail_records|FAIL_RECORD|fail_record\(|add_fail\(|add_warning\(|'
               r'LOOP_TO_MEV|FST_PROTON|FST_STF_|FST_FERMION|FST_QUARK|FST_LOOP_MASS'),
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
        # Precision (2026-04-11): skip when the argument is clearly a boolean
        # mask expression (e.g. `(vals > thresh).astype(int)`), which is just
        # converting 0/1 cluster labels, not creating a float-derived QA state.
        pattern=re.compile(r'(?<![=!<>])\s*(?<!\))\.astype\s*\(\s*int\s*\)'),
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
    # Precision note (2026-04-11, tightened after qa_guardrail and qa_graph
    # triage): the original rule matched `d = <anything>` which produced false
    # positives on every Python idiom where `d` means "dict" or "distance" or
    # "denominator". Now the rule fires ONLY when the RHS looks like it could
    # be a QA coordinate value being set independently. Excluded patterns:
    #   d = {, d = [, d = (      dict / list / tuple literal
    #   d = obj.to_dict(), d = obj.dict, d = dict(   dict construction
    #   d = json.<anything>      JSON parsing
    #   d = math.<anything>      math library (sqrt, log, etc.)
    #   d = np.<anything>        numpy array construction/op
    #   d = torch.<anything>     torch tensor op
    #   d = <numeric literal>    scalar initializer like d = 0.0
    #   d = <identifier> =       chained assignment like d = a = 0
    #   d = <identifier>.<attr>  attribute access (d = W.sum, d = self.x)
    #   d = b + e, d = b+e       legitimate derivation
    # Same exclusions applied to the A2-2 rule for `a`.
    # Pragmatic precision notes (2026-04-11 round 2):
    #   - Accept d = b+e, d = b + e, a = b+2e, a = b + 2e, a = b + 2.0 * e
    #   - Skip subscript access: d = result[...], d = obj['key']
    #   - Skip tuple unpacking: `..., a = func(...)` where `a` is inside a
    #     comma-separated LHS list. Handled by requiring the A2 variable to
    #     be the FIRST non-whitespace token on the line (after optional
    #     `self.` prefix).
    ViolationRule(
        id="A2-1",
        axiom="A2",
        description="d assigned independently (A2: d must always be derived as d = b+e)",
        pattern=re.compile(
            r'^\s*(?:self\.)?d\s*=\s*'   # must be first token (no tuple unpack)
            r'(?!'
            r'\s*b\s*\+\s*e|'
            r'\s*b\+e|'
            r'\s*qa_mod\s*\(|'           # QA mod-wrapper of b+e (A2-compliant)
            r'\s*int\s*\(|'              # int cast wrapper
            r'\s*Fraction\s*\(|'         # Fraction wrapper
            r'\s*\{|'
            r'\s*\[|'
            r'\s*\(|'
            r'\s*\w+\s*\.\s*to_dict|'
            r'\s*\w+\s*\.\s*dict|'
            r'\s*dict\s*\(|'
            r'\s*json\.|'
            r'\s*math\.|'
            r'\s*np\.|'
            r'\s*numpy\.|'
            r'\s*torch\.|'
            r'\s*tf\.|'
            r'\s*-?[0-9]+(?:\.[0-9]*)?\s*$|'
            r'\s*-?[0-9]+(?:\.[0-9]*)?\s*[,)\]}]|'
            r'\s*\w+\s*=\s*|'
            r'\s*\w+\s*\.\s*\w+|'
            r'\s*\w+\s*\[|'                     # d = result[key]
            r'\s*True\b|\s*False\b|\s*None\b'
            r')'
        ),
        severity="WARNING",
        qa_file_only=True,
    ),
    ViolationRule(
        id="A2-2",
        axiom="A2",
        description="a assigned independently (A2: a must always be derived as a = b+2e)",
        pattern=re.compile(
            r'^\s*(?:self\.)?a\s*=\s*'
            r'(?!'
            r'\s*b\s*\+\s*2(?:\.0)?\s*\*?\s*e|'
            r'\s*b\+2(?:\.0)?\*?e|'
            r'\s*d\s*\+\s*e|'            # a = d+e is algebraically equivalent when d=b+e
            r'\s*d\+e|'
            r'\s*qa_mod\s*\(|'           # QA mod-wrapper (A2-compliant)
            r'\s*int\s*\(|'
            r'\s*Fraction\s*\(|'
            r'\s*\{|'
            r'\s*\[|'
            r'\s*\(|'
            r'\s*\w+\s*\.\s*to_dict|'
            r'\s*\w+\s*\.\s*dict|'
            r'\s*dict\s*\(|'
            r'\s*json\.|'
            r'\s*math\.|'
            r'\s*np\.|'
            r'\s*numpy\.|'
            r'\s*torch\.|'
            r'\s*tf\.|'
            r'\s*-?[0-9]+(?:\.[0-9]*)?\s*$|'
            r'\s*-?[0-9]+(?:\.[0-9]*)?\s*[,)\]}]|'
            r'\s*\w+\s*=\s*|'
            r'\s*\w+\s*\.\s*\w+|'
            r'\s*\w+\s*\[|'
            r'\s*True\b|\s*False\b|\s*None\b'
            r')'
        ),
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
    # ── ELEM rules: element computation enforcement ──────────────────────────
    ViolationRule(
        id="ELEM-1",
        axiom="ELEM",
        description="Modular reduction (% m) in element computation context — "
                    "QA elements (C, F, G, A, B, D, E) use RAW d=b+e, "
                    "NEVER mod-reduced. Use qa_elements() from qa_elements.py. "
                    "If this is a T-operator step, add # noqa: ELEM-1",
        pattern=re.compile(
            r'(?:^|[^#])\b(?:C|F|G)\s*=.*%\s*(?:m\b|M\b|modulus|MODULUS|\d{1,2}\b)'
        ),
        severity="ERROR",
        qa_file_only=True,
    ),

    # ELEM-4: QA invariants (norm_f, C=2de, F=ba) computed from mod-reduced
    # inputs. The invariants should use RAW values; mod reduction is T-operator
    # only. Catches: norm_f(x % m, ...) or norm_f(qa_residue(...), ...) or
    # computing f/C/F from e that was already mod-reduced.
    # If this IS a T-operator (orbit classification only), add # noqa: ELEM-4
    ViolationRule(
        id="ELEM-4",
        axiom="ELEM",
        description="norm_f() or QA invariant computed from mod-reduced input — "
                    "invariants should use RAW values (d=b+e, a=b+2e unmodulated). "
                    "Mod reduction is T-operator ONLY (orbit classification). "
                    "If this is orbit classification, add # noqa: ELEM-4",
        pattern=re.compile(
            r'norm_f\s*\(\s*(?:'
            r'[^,)]*%\s*(?:m\b|M\b|MOD|modulus|MODULUS|\d{1,2})\b'   # norm_f(x % m, ...)
            r'|[^,)]*qa_residue\s*\('                                  # norm_f(qa_residue(...), ...)
            r')'
        ),
        severity="WARNING",
        qa_file_only=True,
    ),

    # ELEM-5: Hardcoded state lookup tables (CMAP, MICROSTATE_STATES,
    # QUINTILE_TO_STATE) that assign (b,e) by analyst choice instead of
    # deriving from data. Superseded by [209] generator inference.
    # If the mapping is canonical (from QA books/Volk/Grant), add # noqa: ELEM-5
    ViolationRule(
        id="ELEM-5",
        axiom="ELEM",
        description="Hardcoded (b,e) lookup table — [209] generator inference "
                    "supersedes analyst-chosen state mappings. Derive (b,e) from "
                    "signal evolution instead. If mapping is canonical (Volk/Grant), "
                    "add # noqa: ELEM-5",
        pattern=re.compile(
            r'\bCMAP\s*=\s*\{[^}]*\d+\s*:\s*\d+'
            r'|MICROSTATE_STATES\s*[=:{]'
            r'|QUINTILE_TO_STATE\s*[=:{]'
        ),
        severity="WARNING",
        qa_file_only=True,
    ),

    # ── MAP rules: enforce domain-specific (b,e) mapping ─────────────────────
    # The generic mapping b=degree, e=core_number throws away domain structure
    # (e.g., edge signs, edge weights, temporal order). Every QA graph experiment
    # MUST declare its mapping rationale. A script that assigns b from degree()
    # without a QA_MAP comment is using the generic default, which has repeatedly
    # produced nulls on graphs where domain-specific mappings (signed-degree,
    # hub-distance, generator inference) would have worked.
    # The rule: any line that does `b = ...degree...` or `b = max(1, int(degree`
    # must have a `# QA_MAP:` comment on the same line or within 3 lines above
    # explaining WHY this mapping is correct for this domain.
    # Suppression: # noqa: MAP-1 if the generic mapping is intentionally used
    # as a BASELINE for comparison (not as the primary QA method).
    ViolationRule(
        id="MAP-1",
        axiom="MAP",
        description="Generic b=degree mapping without QA_MAP declaration — every QA graph "
                    "experiment must declare WHY this (b,e) mapping is appropriate for this "
                    "domain. Add a '# QA_MAP: <rationale>' comment within 3 lines, or use "
                    "a domain-specific mapping (signed-degree, hub-distance, generator "
                    "inference per [209]). Add # noqa: MAP-1 if generic is intentional baseline.",
        pattern=re.compile(
            r'b\s*=\s*(?:max\s*\(\s*1\s*,\s*)?int\s*\(\s*degree'
        ),
        severity="WARNING",
        qa_file_only=True,
    ),

    # ── FIREWALL rules (added 2026-04-11 after FST v1 audit) ────────────────
    # Theorem NT / T2-b at the validator-source level: a file that handles
    # both observer-layer values (MeV, frequency, continuous measurement)
    # AND QA-layer values (loop counts, integer states) must route the
    # crossing through an explicit observer projection Pi. A line that
    # arithmetically mixes a mev-named identifier and a loop-named
    # identifier without going through apply_Pi is a firewall crossing
    # with no declared direction, i.e. a T2-b violation at the code level.
    # Root cause: v1 FST validator had "abs(mev_ratio - loop_ratio)" which
    # subtracted an observer-layer float from a QA-layer int ratio inside
    # the same decision logic, never declaring Pi. The linter did not
    # catch this because (a) the file was not a recognized QA file and
    # (b) no rule pattern matched this specific construction. Both
    # problems are now fixed.
    ViolationRule(
        id="FIREWALL-1",
        axiom="T2-b",
        description="MeV and loop identifiers arithmetically combined without declared Pi projection (observer/QA firewall crossing). If this line is the observer projection itself, mark with # noqa: FIREWALL-1 and ensure the enclosing function is named apply_Pi.",
        pattern=re.compile(
            r'(?i)'
            r'(?:\b\w*mev\w*\s*[+\-*/]\s*\w*loop\w*)'
            r'|(?:\b\w*loop\w*\s*[+\-*/]\s*\w*mev\w*)'
            r'|(?:abs\s*\([^)]*\b\w*mev\w*[^)]*\b\w*loop\w*[^)]*\))'
            r'|(?:abs\s*\([^)]*\b\w*loop\w*[^)]*\b\w*mev\w*[^)]*\))'
        ),
        severity="ERROR",
        qa_file_only=True,
    ),
]

# ── FIREWALL-2: whole-file check ──────────────────────────────────────────────
# A cert-validator file that:
#   (a) defines or reads both mev_* and loop_* identifiers (both layers appear)
#   (b) does NOT declare an apply_Pi() function (or import one from qa_cert_core)
# is a firewall violation at the module level. Either the file is QA-only
# (no mev_*), observer-only (no loop_*), or it must route crossings through Pi.
_MEV_IDENT_PATTERN = re.compile(r'\b\w*[mM][eE][vV]\w*\b')
_LOOP_IDENT_PATTERN = re.compile(r'\b\w*[lL]oop\w*\b')
_APPLY_PI_DECL_PATTERN = re.compile(
    r'^\s*def\s+(?:apply_Pi|apply_pi|observer_project|project_Pi)\s*\(',
    re.MULTILINE,
)
_APPLY_PI_IMPORT_PATTERN = re.compile(
    r'from\s+\S+\s+import[^#\n]*\b(?:apply_Pi|apply_pi|observer_project)\b'
)

# Required declaration block pattern — all QA empirical scripts must declare their observer
# Matches both single-line: QA_COMPLIANCE = "..."
# and multi-line:           QA_COMPLIANCE = (\n  "..."\n)
QA_COMPLIANCE_DECLARATION = re.compile(r'QA_COMPLIANCE\s*=\s*[\'\"{(]')

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
    """Heuristic: is the match inside a comment or string?

    Checks:
      (a) a `#` comment earlier on the same line
      (b) the match position lies inside a single-line string literal
          (either single or double-quoted). This is NOT a full parser —
          it doesn't handle cross-line triple-quoted strings; those are
          filtered separately by strip_triple_quoted_strings().
    """
    prefix = line[:match_start]
    if "#" in prefix.split('"')[0].split("'")[0]:
        return True
    # Count unescaped quote pairs before the match. Odd count = inside string.
    single = prefix.count("'") - prefix.count("\\'")
    double = prefix.count('"') - prefix.count('\\"')
    if single % 2 == 1 or double % 2 == 1:
        return True
    return False


def strip_triple_quoted_strings(content: str) -> str:
    """Return content with all triple-quoted string bodies replaced by blank
    lines (preserving line numbers). Prevents false positives inside
    docstrings, multi-line string literals, and negative-example code samples.
    """
    out: list[str] = []
    i = 0
    in_triple = False
    triple_char = ''
    while i < len(content):
        if not in_triple:
            # Look for an opening triple quote
            if content[i:i+3] == '"""' or content[i:i+3] == "'''":
                triple_char = content[i:i+3]
                in_triple = True
                out.append(content[i:i+3])
                i += 3
                continue
            out.append(content[i])
            i += 1
        else:
            # Inside a triple-quoted string: keep newlines, mask everything else
            if content[i:i+3] == triple_char:
                in_triple = False
                out.append(triple_char)
                i += 3
                continue
            if content[i] == '\n':
                out.append('\n')
            else:
                out.append(' ')
            i += 1
    return ''.join(out)

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

    # Strip triple-quoted string bodies for per-line rule matching so that
    # docstrings documenting bad patterns (e.g. `S1: use x*x, not b**2`) do
    # not trip the very rules they are explaining. Line numbers are preserved.
    masked_content = strip_triple_quoted_strings(content)
    masked_lines = masked_content.splitlines()

    # Check for required declaration block in QA files (hard gate — ERROR)
    # Exemptions (2026-04-11):
    #   (a) test_*.py files — test suites don't carry empirical observer
    #       declarations; the code they test does.
    #   (b) top-of-file `# noqa: DECL-1 (reason)` suppression — for
    #       infrastructure files that trip the QA indicator but are not
    #       empirical scripts (data models, schema validators, etc.).
    if file_is_qa:
        is_test_file = path.name.startswith("test_") or "/tests/" in str(path)
        first_15 = "\n".join(lines[:15])
        has_top_noqa = "noqa: DECL-1" in first_15
        if not QA_COMPLIANCE_DECLARATION.search(content) \
                and not is_test_file \
                and not has_top_noqa:
            violations.append((1, "DECL-1", "DECL",
                "Missing QA_COMPLIANCE declaration block — empirical QA scripts must declare "
                "their observer and state alphabet (see QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md)"))

    # ORBIT-4: local orbit_family reimplementation (whole-file check)
    # Skip the canonical file itself.  Respect inline `# noqa: ORBIT-4`
    # on the def line — a marked fallback is an intentional decision.
    if not is_orbit_rules and file_is_qa:
        for m in _ORBIT_REDEF_PATTERN.finditer(content):
            lineno = content[:m.start()].count("\n") + 1
            fn_name = re.search(r'def\s+(\w+)', m.group()).group(1)
            def_line = lines[lineno - 1] if lineno <= len(lines) else ""
            if "noqa: ORBIT-4" in def_line or "noqa:ORBIT-4" in def_line:
                continue
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

    # FIREWALL-2: file defines both mev_* and loop_* identifiers without
    # declaring apply_Pi (whole-file Theorem NT firewall check)
    if file_is_qa:
        has_mev = bool(_MEV_IDENT_PATTERN.search(content))
        has_loop = bool(_LOOP_IDENT_PATTERN.search(content))
        has_apply_pi = bool(_APPLY_PI_DECL_PATTERN.search(content) or
                             _APPLY_PI_IMPORT_PATTERN.search(content))
        if has_mev and has_loop and not has_apply_pi:
            # Allow whole-file suppression at top of file
            first_non_blank = next(
                (ln for ln in lines[:15] if ln.strip()), "")
            if "noqa: FIREWALL-2" not in "\n".join(lines[:15]):
                # Find first line that mentions mev for precise pointer
                mev_lineno = 1
                for i, ln in enumerate(lines, 1):
                    if _MEV_IDENT_PATTERN.search(ln):
                        mev_lineno = i
                        break
                violations.append((mev_lineno, "FIREWALL-2", "T2-b",
                    "File references both mev_* and loop_* identifiers but "
                    "does not declare an apply_Pi() function or import one. "
                    "Cert-validator files that bridge observer-layer MeV "
                    "and QA-layer loop counts must route crossings through "
                    "an explicit Pi projection (Theorem NT). Add "
                    "def apply_Pi(loop_count: int) -> float, or suppress "
                    "with # noqa: FIREWALL-2 in the first 15 lines if the "
                    "file is pure observer-layer or pure QA-layer."))

    # ELEM-2: validator reimplements element computation without importing qa_elements
    # Detect: def compute_C, compute_F, compute_G, compute_all, compute_elements,
    # qa_derived (in element context), qa_tuple (with modulus for elements).
    # Skip qa_elements.py itself and canonical modules.
    _ELEM_REDEF_PATTERN = re.compile(
        r'^\s*def\s+(?:compute_C|compute_F|compute_G|compute_all|'
        r'compute_elements|compute_16|qa_derived|qa_raw_derived)\s*\(',
        re.MULTILINE,
    )
    _ELEM_IMPORT_PATTERN = re.compile(
        r'from\s+qa_elements\s+import|from\s+qa_arithmetic'
    )
    _is_qa_elements_file = (path.name == "qa_elements.py")
    _is_canonical_elem = _is_qa_elements_file or _is_canonical_orbit_source(path)

    if not _is_canonical_elem and file_is_qa:
        if _ELEM_REDEF_PATTERN.search(content) and not _ELEM_IMPORT_PATTERN.search(content):
            m_elem = _ELEM_REDEF_PATTERN.search(content)
            lineno_elem = content[:m_elem.start()].count("\n") + 1
            def_line_elem = lines[lineno_elem - 1] if lineno_elem <= len(lines) else ""
            if "# noqa: ELEM-2" not in def_line_elem:
                violations.append((lineno_elem, "ELEM-2", "ELEM",
                    "Local reimplementation of element computation — import from "
                    "qa_elements.py instead. Local copies diverge and cause axiom "
                    "violations (C>=4 bound, raw d=b+e, F identity). "
                    "Add # noqa: ELEM-2 if intentional."))

    # Per-line rules
    # Use masked_lines (triple-quoted string bodies replaced with spaces)
    # for matching so docstrings cannot trip rules they describe.
    for lineno, (line, masked) in enumerate(zip(lines, masked_lines), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Support inline suppression: # noqa: RULE-ID[,RULE-ID]
        # The suppression comment is read from the ORIGINAL line (not masked)
        # so comments inside strings don't activate suppression.
        noqa_ids: set[str] = set()
        noqa_m = re.search(r'#\s*noqa:\s*([\w,-]+)', line)
        if noqa_m:
            noqa_ids = {s.strip() for s in noqa_m.group(1).split(",")}

        for rule in RULES:
            if rule.id in noqa_ids:
                continue
            if rule.qa_file_only and not file_is_qa:
                continue
            m = rule.pattern.search(masked)
            if m and not is_comment_or_string(masked, m.start()):
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
    # Linter test fixtures — intentionally bad files that exercise the
    # FIREWALL rules via the test harness test_qa_axiom_linter_firewall.py.
    # Excluded from bulk --all scan so they don't appear as errors in repo
    # audits; they ARE scanned directly by the test harness, which asserts
    # on expected violations.
    "linter_fixtures",
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
