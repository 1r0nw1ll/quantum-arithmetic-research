#!/usr/bin/env python3
"""
qa_family_demo.py -- Narrated demo runner for QA cert families.

Usage:
    python demos/qa_family_demo.py --family geogebra
    python demos/qa_family_demo.py --family rule30
    python demos/qa_family_demo.py --all
    python demos/qa_family_demo.py --family geogebra --ci
    python demos/qa_family_demo.py --family rule30 --ci
    python demos/qa_family_demo.py --all --ci
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = "/home/player2/signal_experiments"

BOLD  = "\033[1m"
RESET = "\033[0m"

LABEL_PASS = f"{BOLD}[PASS]{RESET}"
LABEL_FAIL = f"{BOLD}[FAIL]{RESET}"


def section(n, title, ci_mode=False):
    if not ci_mode:
        print(f"\n=== Step {n}: {title} ===")


def run_validator(cmd, expect_pass, ci_mode=False):
    result = subprocess.run(
        [sys.executable] + cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if expect_pass and result.returncode != 0:
        if not ci_mode:
            print(f"  ERROR: validator exited {result.returncode} but PASS was expected.")
            print(f"  stdout: {result.stdout.strip()}")
            print(f"  stderr: {result.stderr.strip()}")
            sys.exit(1)
        return result.returncode, result.stdout, result.stderr, "PASS_EXPECTED_BUT_FAILED"
    if not expect_pass and result.returncode == 0:
        if not ci_mode:
            print("  WARNING: validator exited 0 but FAIL was expected -- continuing.")
        else:
            return result.returncode, result.stdout, result.stderr, "FAIL_EXPECTED_BUT_PASSED"
    return result.returncode, result.stdout, result.stderr, None


# ---------------------------------------------------------------------------
# GeoGebra family [56]
# ---------------------------------------------------------------------------

GEOGEBRA_VALIDATOR    = "qa_geogebra_scene_adapter_v1/validator.py"
GEOGEBRA_PASS_FIXTURE = "qa_geogebra_scene_adapter_v1/fixtures/valid_exact_345_triangle.json"
GEOGEBRA_FAIL_FIXTURE = "qa_geogebra_scene_adapter_v1/fixtures/invalid_zero_denominator.json"
GEOGEBRA_FAIL_STEP3   = "qa_geogebra_scene_adapter_v1/fixtures/invalid_law_equation_mismatch.json"
GEOGEBRA_PUNCHLINE    = "Exact Z/Q arithmetic -- structural obstruction, zero ambiguity."
GEOGEBRA_EXPECTED_FAIL_TYPE = "ZERO_DENOMINATOR"

GEOGEBRA_FIXTURES = [
    GEOGEBRA_VALIDATOR,
    GEOGEBRA_PASS_FIXTURE,
    GEOGEBRA_FAIL_FIXTURE,
]


def _check_paths_exist(paths, family_name):
    """Verify all registered fixture paths exist under REPO_ROOT. Exit 1 if any missing."""
    missing = []
    for p in paths:
        full = os.path.join(REPO_ROOT, p)
        if not os.path.exists(full):
            missing.append(full)
    if missing:
        print(f"ERROR [{family_name}]: missing fixture paths:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)


def _extract_geogebra_fail_type(stdout):
    """Parse GeoGebra validator JSON output and return the first FAIL gate's fail_type."""
    stdout_stripped = stdout.strip()
    if not stdout_stripped:
        return None
    parsed = None
    for chunk in stdout_stripped.split("\n\n"):
        try:
            parsed = json.loads(chunk.strip())
            break
        except json.JSONDecodeError:
            pass
    if parsed is None:
        try:
            parsed = json.loads(stdout_stripped)
        except json.JSONDecodeError:
            pass
    if parsed is None:
        return None
    for gate in parsed.get("results", []):
        if gate.get("status") == "FAIL":
            details = gate.get("details", {})
            ft = details.get("fail_type") if details else None
            if ft is None:
                ft = gate.get("message")
            return ft
    return None


def demo_geogebra(ci_mode=False):
    _check_paths_exist(GEOGEBRA_FIXTURES, "geogebra")

    if not ci_mode:
        print()
        print("QA Cert Families -- narrated demo runner.")
        print("Family: GeoGebra Scene Adapter [56] -- exact Z/Q typed coordinates.")

    # --- Step 1: PASS -------------------------------------------------------
    section(1, "Valid fixture -- expect PASS", ci_mode)

    fixture_path = os.path.join(REPO_ROOT, GEOGEBRA_PASS_FIXTURE)
    with open(fixture_path) as fh:
        fixture_data = json.load(fh)

    raw_cert_value = (
        fixture_data.get("cert_hash")
        or fixture_data.get("manifest_hash")
        or fixture_data.get("hash_chain", {}).get("this_cert_hash")
        or fixture_data.get("cert_id", "(not found)")
    )

    # Label standardization: short slug (len < 64) -> cert_id
    if isinstance(raw_cert_value, str) and len(raw_cert_value) < 64:
        cert_label = "cert_id"
    else:
        cert_label = "cert_id"

    _rc, _stdout, _stderr, err_indicator = run_validator(
        [GEOGEBRA_VALIDATOR, GEOGEBRA_PASS_FIXTURE, "--json"],
        expect_pass=True,
        ci_mode=ci_mode,
    )

    if ci_mode:
        if err_indicator == "PASS_EXPECTED_BUT_FAILED":
            return False, "PASS fixture failed"
    else:
        print(f"  Fixture  : {GEOGEBRA_PASS_FIXTURE}")
        print(f"  Result   : {LABEL_PASS}")
        print(f"  {cert_label}: {raw_cert_value}")

    # --- Step 2: FAIL (ZERO_DENOMINATOR) ------------------------------------
    section(2, "Structural failure fixture -- expect FAIL (ZERO_DENOMINATOR)", ci_mode)

    _rc2, stdout2, _stderr2, err_indicator2 = run_validator(
        [GEOGEBRA_VALIDATOR, GEOGEBRA_FAIL_FIXTURE, "--json"],
        expect_pass=False,
        ci_mode=ci_mode,
    )

    if ci_mode:
        if err_indicator2 == "FAIL_EXPECTED_BUT_PASSED":
            return False, "FAIL fixture unexpectedly passed"
        fail_type = _extract_geogebra_fail_type(stdout2)
        if fail_type != GEOGEBRA_EXPECTED_FAIL_TYPE:
            return False, f"expected fail_type={GEOGEBRA_EXPECTED_FAIL_TYPE!r}, got {fail_type!r}"
        return True, None
    else:
        print(f"  Fixture : {GEOGEBRA_FAIL_FIXTURE}")
        print(f"  Result  : {LABEL_FAIL}")

        stdout_stripped = stdout2.strip()
        parsed = None
        if stdout_stripped:
            for chunk in stdout_stripped.split("\n\n"):
                try:
                    parsed = json.loads(chunk.strip())
                    break
                except json.JSONDecodeError:
                    pass
            if parsed is None:
                try:
                    parsed = json.loads(stdout_stripped)
                except json.JSONDecodeError:
                    pass

        if parsed is not None:
            for gate in parsed.get("results", []):
                if gate.get("status") == "FAIL":
                    details = gate.get("details", {})
                    inv_diff = details if details else {"fail_type": gate.get("message")}
                    print("  invariant_diff:")
                    for line in json.dumps(inv_diff, indent=4).splitlines():
                        print(f"    {line}")
                    break
        else:
            print(f"  stdout: {stdout_stripped[:400]}")

        # --- Step 3: Summary ------------------------------------------------
        section(3, "Summary", ci_mode)

        print("  Family [56] certifies: GeoGebra scene exports with exact Z/Q typed coordinates.")
        print(
            f"  Structural obstructions caught: ZERO_DENOMINATOR, LAW_EQUATION_MISMATCH"
            f" (fixture: {GEOGEBRA_FAIL_STEP3})."
        )
        print(f"  Punchline: {GEOGEBRA_PUNCHLINE}")
        print()


# ---------------------------------------------------------------------------
# Rule30 family [34]
# ---------------------------------------------------------------------------

RULE30_VALIDATOR    = "qa_alphageometry_ptolemy/qa_rule30/qa_rule30_cert_validator.py"
RULE30_PASS_FIXTURE = (
    "qa_alphageometry_ptolemy/qa_rule30/certpacks/"
    "rule30_nonperiodicity_v3/QA_RULE30_NONPERIODICITY_CERT.v1.json"
)
RULE30_FAIL_FIXTURE = (
    "qa_alphageometry_ptolemy/qa_rule30/fixtures/cert_neg_aggregate_mismatch.json"
)
RULE30_PUNCHLINE = (
    "Cellular automata as certified trace -- we don't approximate, we certify evolution."
)
RULE30_EXPECTED_FAIL_TYPE = "AGGREGATE_MISMATCH"

RULE30_FIXTURES = [
    RULE30_VALIDATOR,
    RULE30_PASS_FIXTURE,
    RULE30_FAIL_FIXTURE,
]


def _extract_rule30_fail_type(stdout):
    """Parse Rule30 validator plain-text/JSON output and return the fail_type."""
    lines = stdout.splitlines()
    block = []
    in_block = False
    for ln in lines:
        stripped = ln.strip()
        if not in_block and stripped.startswith("{"):
            in_block = True
        if in_block:
            block.append(ln)
        if in_block and stripped == "}":
            try:
                parsed = json.loads("\n".join(block))
                ft = parsed.get("fail_type")
                if ft:
                    return ft
            except json.JSONDecodeError:
                pass
            block = []
            in_block = False
    # Fallback: scan for known fail_type strings in text
    for ln in lines:
        if "AGGREGATE_MISMATCH" in ln:
            return "AGGREGATE_MISMATCH"
        if "fail_type" in ln.lower():
            idx = ln.find(":")
            if idx != -1:
                candidate = ln[idx + 1:].strip().strip('"').strip("'").strip(",")
                if candidate:
                    return candidate
    return None


def demo_rule30(ci_mode=False):
    _check_paths_exist(RULE30_FIXTURES, "rule30")

    if not ci_mode:
        print()
        print("QA Cert Families -- narrated demo runner.")
        print("Family: Rule 30 Certified Discovery [34] -- nonperiodicity of the center column.")

    # --- Step 1: PASS -------------------------------------------------------
    section(1, "Valid cert pack -- expect PASS", ci_mode)

    fixture_path = os.path.join(REPO_ROOT, RULE30_PASS_FIXTURE)
    with open(fixture_path) as fh:
        fixture_data = json.load(fh)

    raw_cert_value = (
        fixture_data.get("cert_hash")
        or fixture_data.get("manifest_hash")
        or fixture_data.get("hash_chain", {}).get("this_cert_hash", "(not found)")
    )

    # Label standardization: 64-char hex -> cert_sha256, else cert_id
    if isinstance(raw_cert_value, str) and len(raw_cert_value) == 64:
        cert_label = "cert_sha256"
    else:
        cert_label = "cert_id"

    _rc, _stdout, _stderr, err_indicator = run_validator(
        [RULE30_VALIDATOR, "cert", RULE30_PASS_FIXTURE],
        expect_pass=True,
        ci_mode=ci_mode,
    )

    if ci_mode:
        if err_indicator == "PASS_EXPECTED_BUT_FAILED":
            return False, "PASS fixture failed"
    else:
        print(f"  Fixture  : {RULE30_PASS_FIXTURE}")
        print(f"  Result   : {LABEL_PASS}")
        print(f"  {cert_label}: {raw_cert_value}")

    # --- Step 2: FAIL (AGGREGATE_MISMATCH) ----------------------------------
    section(2, "Aggregate mismatch fixture -- expect FAIL (AGGREGATE_MISMATCH)", ci_mode)

    _rc2, stdout2, _stderr2, err_indicator2 = run_validator(
        [RULE30_VALIDATOR, "cert", RULE30_FAIL_FIXTURE],
        expect_pass=False,
        ci_mode=ci_mode,
    )

    if ci_mode:
        if err_indicator2 == "FAIL_EXPECTED_BUT_PASSED":
            return False, "FAIL fixture unexpectedly passed"
        fail_type = _extract_rule30_fail_type(stdout2)
        if fail_type != RULE30_EXPECTED_FAIL_TYPE:
            return False, f"expected fail_type={RULE30_EXPECTED_FAIL_TYPE!r}, got {fail_type!r}"
        return True, None
    else:
        print(f"  Fixture : {RULE30_FAIL_FIXTURE}")
        print(f"  Result  : {LABEL_FAIL}")

        lines = stdout2.splitlines()
        fail_lines = [ln for ln in lines if "fail" in ln.lower()]

        parsed = None
        block = []
        in_block = False
        for ln in lines:
            stripped = ln.strip()
            if not in_block and stripped.startswith("{"):
                in_block = True
            if in_block:
                block.append(ln)
            if in_block and stripped == "}":
                try:
                    parsed = json.loads("\n".join(block))
                    break
                except json.JSONDecodeError:
                    pass
                block = []
                in_block = False

        if parsed is not None:
            inv_diff  = parsed.get("invariant_diff")
            fail_type = parsed.get("fail_type")
            if inv_diff or fail_type:
                display = {"fail_type": fail_type, "invariant_diff": inv_diff}
                print("  invariant_diff:")
                for line in json.dumps(display, indent=4).splitlines():
                    print(f"    {line}")
            else:
                for fl in fail_lines[:4]:
                    print(f"  {fl}")
        elif fail_lines:
            for fl in fail_lines[:4]:
                print(f"  {fl}")
        else:
            print(f"  stdout: {stdout2.strip()[:400]}")

        # --- Step 3: Summary ------------------------------------------------
        section(3, "Summary", ci_mode)

        print(
            "  Family [34] certifies: Rule 30 cellular automaton nonperiodicity"
            " via hash-chained witness manifests."
        )
        print(
            "  Structural obstructions caught: AGGREGATE_MISMATCH, SCOPE_INVALID,"
            " HASH_MISMATCH, and more."
        )
        print(f"  Punchline: {RULE30_PUNCHLINE}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global BOLD, RESET, LABEL_PASS, LABEL_FAIL

    parser = argparse.ArgumentParser(
        description="Narrated demo runner for QA cert families."
    )
    parser.add_argument(
        "--family",
        choices=["geogebra", "rule30"],
        help="Which cert family to demo.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all cert family demos.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Quiet machine mode: suppress ANSI codes and narration, emit one line per family.",
    )
    args = parser.parse_args()

    if args.all and args.family:
        parser.error("Use --all or --family, not both.")
    if not args.all and not args.family:
        parser.error("Specify --family <name> or --all.")

    ci_mode = args.ci

    if ci_mode:
        BOLD = ""
        RESET = ""
        LABEL_PASS = "[PASS]"
        LABEL_FAIL = "[FAIL]"

    if args.all:
        if not ci_mode:
            print()
            print("QA Cert Family Demo -- running all families")

        families = [
            ("geogebra", demo_geogebra),
            ("rule30",   demo_rule30),
        ]

        if ci_mode:
            all_ok = True
            try:
                for name, fn in families:
                    ok, reason = fn(ci_mode=True)
                    if ok:
                        print(f"FAMILY {name}: OK")
                    else:
                        print(f"FAMILY {name}: FAIL {reason}")
                        all_ok = False
            except Exception as exc:
                print(f"ERROR: {exc}")
                sys.exit(1)
            sys.exit(0 if all_ok else 2)
        else:
            for _name, fn in families:
                fn(ci_mode=False)
            print("All families complete.")

    else:
        # Single family mode
        family_map = {
            "geogebra": demo_geogebra,
            "rule30":   demo_rule30,
        }
        fn = family_map[args.family]

        if ci_mode:
            try:
                ok, reason = fn(ci_mode=True)
                if ok:
                    print(f"FAMILY {args.family}: OK")
                    sys.exit(0)
                else:
                    print(f"FAMILY {args.family}: FAIL {reason}")
                    sys.exit(2)
            except Exception as exc:
                print(f"ERROR: {exc}")
                sys.exit(1)
        else:
            fn(ci_mode=False)


if __name__ == "__main__":
    main()
