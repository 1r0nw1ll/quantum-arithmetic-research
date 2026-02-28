#!/usr/bin/env python3
"""
qa_latex_claim_linter.py — QA LaTeX Claim Linter

Scans .tex files for DPI-anchored or overclaim trigger phrases and verifies
that the required QA_CERT_REQUIRED tripwire bundle is present.

Trigger classes (from rules config):
  dpi_anchors   — phrases that invoke the data processing inequality
  overclaims    — universality claims that need empirical qualification

If ANY trigger fires in a file, ALL tripwires in the bundle must be present
(as %% QA_CERT_REQUIRED: <cert_id> comment lines).

Strict mode (--strict):
  Overclaim phrase matches are only permitted if the file also contains
  %% QA_CLAIM_LEVEL: empirical_only  OR  %% QA_CLAIM_LEVEL: proof_sketch

Usage:
  python tools/qa_latex_claim_linter.py [paths...]      # paths = .tex files or dirs
  python tools/qa_latex_claim_linter.py --json          # JSON output
  python tools/qa_latex_claim_linter.py --strict        # strict claim-level check
  python tools/qa_latex_claim_linter.py --config FILE   # custom rules

Exit codes:
  0 — all files pass
  1 — one or more failures detected
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default rules (also written to qa_latex_claim_linter_rules.json)
# ---------------------------------------------------------------------------

DEFAULT_RULES = {
    "version": "1.1",
    "trigger_classes": {
        "dpi_anchors": [
            r"\bDPI\b",
            r"data processing inequality",
            r"processing inequality"
        ],
        "overclaims": [
            r"\buniversal\b",
            r"\bproven\b",
            r"for all distributions",
            r"for any distribution",
            r"holds for all",
            r"always holds"
        ],
        "kernel_conformity": [
            r"\\ln\(m/\\delta\)",
            r"\\log\(m/\\delta\)"
        ]
    },
    "required_tripwires": [
        "QA_PAC_BAYES_DPI_SCOPE_CERT.v1",
        "QA_PAC_BAYES_CONSTANT_CERT.v1.1",
        "QA_DQA_PAC_BOUND_KERNEL_CERT.v1"
    ],
    "allowed_claim_levels": [
        "empirical_only",
        "proof_sketch"
    ]
}


# ---------------------------------------------------------------------------
# Core linting logic
# ---------------------------------------------------------------------------

def _load_rules(config_path=None):
    if config_path is None:
        return DEFAULT_RULES
    with open(config_path) as f:
        return json.load(f)


def _compile_patterns(rules):
    """Return compiled regexes for (dpi_anchors, overclaims, kernel_conformity)."""
    tc = rules["trigger_classes"]
    dpi = [re.compile(p, re.IGNORECASE) for p in tc["dpi_anchors"]]
    oc  = [re.compile(p, re.IGNORECASE) for p in tc["overclaims"]]
    kc  = [re.compile(p) for p in tc.get("kernel_conformity", [])]
    return dpi, oc, kc


def _extract_tripwires(text):
    """Return set of cert_ids found in %% QA_CERT_REQUIRED: <id> lines."""
    found = set()
    for m in re.finditer(r"%%\s*QA_CERT_REQUIRED:\s*(\S+)", text):
        found.add(m.group(1))
    return found


def _extract_claim_levels(text):
    """Return set of claim levels found in %% QA_CLAIM_LEVEL: <level> lines."""
    found = set()
    for m in re.finditer(r"%%\s*QA_CLAIM_LEVEL:\s*(\S+)", text):
        found.add(m.group(1))
    return found


def _find_trigger_hits(text, patterns, label):
    """
    Return list of {class, pattern, line_no, line_text} for each match,
    skipping comment-only lines (lines whose first non-whitespace is %).
    """
    hits = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue  # skip LaTeX comments
        for pat in patterns:
            if pat.search(line):
                hits.append({
                    "class": label,
                    "pattern": pat.pattern,
                    "line_no": lineno,
                    "line_text": line.rstrip()
                })
    return hits


def lint_file(filepath, rules, strict=False):
    """
    Lint a single .tex file.

    Returns a dict:
      {
        "file": str,
        "status": "PASS" | "FAIL",
        "triggers": [...],
        "missing_tripwires": [...],
        "strict_violation": bool,
        "errors": [...]
      }
    """
    path = Path(filepath)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {
            "file": str(filepath),
            "status": "FAIL",
            "triggers": [],
            "missing_tripwires": [],
            "strict_violation": False,
            "errors": [f"Cannot read file: {e}"]
        }

    dpi_patterns, oc_patterns, kc_patterns = _compile_patterns(rules)

    dpi_hits = _find_trigger_hits(text, dpi_patterns, "dpi_anchor")
    oc_hits  = _find_trigger_hits(text, oc_patterns,  "overclaim")
    kc_hits  = _find_trigger_hits(text, kc_patterns,  "kernel_conformity")
    all_hits = dpi_hits + oc_hits + kc_hits

    errors = []
    missing_tripwires = []
    strict_violation = False

    # kernel_conformity hits are always errors (formula mismatch vs locked kernel)
    for h in kc_hits:
        errors.append(
            f"L{h['line_no']}: kernel_conformity: formula mismatch — "
            f"expected \\ln(1/\\delta) per QA_DQA_PAC_BOUND_KERNEL_CERT.v1 "
            f"(family-85), found pattern '{h['pattern']}'"
        )

    if dpi_hits or oc_hits:
        # Check tripwire bundle
        present = _extract_tripwires(text)
        required = set(rules["required_tripwires"])
        missing_tripwires = sorted(required - present)
        if missing_tripwires:
            errors.append(
                f"Trigger(s) found but {len(missing_tripwires)} tripwire(s) missing: "
                + ", ".join(missing_tripwires)
            )

        # Strict mode: overclaim hits require QA_CLAIM_LEVEL marker
        if strict and oc_hits:
            claim_levels = _extract_claim_levels(text)
            allowed = set(rules.get("allowed_claim_levels", []))
            if not (claim_levels & allowed):
                strict_violation = True
                errors.append(
                    "Strict: overclaim trigger found but no %% QA_CLAIM_LEVEL: "
                    + "/".join(sorted(allowed)) + " marker present"
                )

    status = "PASS" if not errors else "FAIL"

    return {
        "file": str(filepath),
        "status": status,
        "triggers": all_hits,
        "missing_tripwires": missing_tripwires,
        "strict_violation": strict_violation,
        "errors": errors
    }


def collect_tex_files(paths):
    """Expand paths: .tex files directly, directories searched recursively."""
    collected = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            collected.extend(sorted(p.rglob("*.tex")))
        elif p.suffix == ".tex":
            collected.append(p)
        else:
            # Treat as explicit file regardless of extension
            collected.append(p)
    return collected


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_human(results):
    lines = []
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = len(results) - pass_count

    for r in results:
        mark = "✓" if r["status"] == "PASS" else "✗"
        lines.append(f"  [{r['status']}] {mark} {r['file']}")
        if r["triggers"]:
            lines.append(f"         triggers: {len(r['triggers'])} hit(s)")
            for t in r["triggers"][:5]:  # show first 5
                lines.append(f"           L{t['line_no']:4d}: [{t['class']}] {t['line_text'][:80]}")
            if len(r["triggers"]) > 5:
                lines.append(f"           ... {len(r['triggers'])-5} more")
        for err in r["errors"]:
            lines.append(f"         ERROR: {err}")

    lines.append("")
    lines.append(f"  Summary: {pass_count} PASS, {fail_count} FAIL out of {len(results)} file(s)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QA LaTeX Claim Linter — checks DPI/overclaim triggers + tripwires"
    )
    parser.add_argument(
        "paths", nargs="*",
        help=".tex files or directories to scan (default: current directory)"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_out",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Overclaim phrases require %% QA_CLAIM_LEVEL: marker"
    )
    parser.add_argument(
        "--config", metavar="FILE",
        help="Path to custom rules JSON (default: built-in rules)"
    )
    args = parser.parse_args()

    rules = _load_rules(args.config)

    scan_paths = args.paths if args.paths else ["."]
    tex_files = collect_tex_files(scan_paths)

    if not tex_files:
        msg = {"error": "No .tex files found", "paths": [str(p) for p in scan_paths]}
        if args.json_out:
            print(json.dumps(msg, indent=2))
        else:
            print("qa_latex_claim_linter: no .tex files found in", scan_paths)
        sys.exit(0)

    results = [lint_file(f, rules, strict=args.strict) for f in tex_files]

    if args.json_out:
        out = {
            "tool": "qa_latex_claim_linter",
            "version": "1.0",
            "strict": args.strict,
            "summary": {
                "total": len(results),
                "pass": sum(1 for r in results if r["status"] == "PASS"),
                "fail": sum(1 for r in results if r["status"] == "FAIL")
            },
            "files": results
        }
        print(json.dumps(out, indent=2))
    else:
        print("qa_latex_claim_linter")
        print("=" * 60)
        print(_format_human(results))

    any_fail = any(r["status"] == "FAIL" for r in results)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
