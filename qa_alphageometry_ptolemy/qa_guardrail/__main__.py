#!/usr/bin/env python3
"""
__main__.py - CLI entrypoint for qa_guardrail package

Allows clean execution via: python -m qa_alphageometry_ptolemy.qa_guardrail
"""

import argparse
import json
import os
import sys

from .qa_guardrail import (
    guard_from_stdin,
    validate_fixtures,
    run_self_tests,
)


def main():
    parser = argparse.ArgumentParser(description="QA Guardrail MVP")
    parser.add_argument("--validate", action="store_true", help="Output self-test results as JSON")
    parser.add_argument("--fixtures", action="store_true", help="Validate golden fixtures")
    parser.add_argument("--fixtures-dir", default=None, help="Directory containing fixtures")
    parser.add_argument("--guard", action="store_true",
                        help="Run guard on stdin JSON (GUARDRAIL_REQUEST.v1 -> GUARDRAIL_RESULT.v1)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Guard mode: read request from stdin, output result to stdout
    if args.guard:
        result = guard_from_stdin()
        print(json.dumps(result, indent=2))
        sys.exit(0)  # Always exit 0 (result carries allow/deny); exit 1 only on internal errors

    if args.fixtures:
        fixtures_dir = args.fixtures_dir or os.path.join(base_dir, "fixtures")
        if os.path.isdir(fixtures_dir):
            result = validate_fixtures(fixtures_dir)
            print(json.dumps(result, indent=2))
            sys.exit(0 if result["ok"] else 1)
        else:
            print(f"Fixtures directory not found: {fixtures_dir}")
            sys.exit(1)

    results = run_self_tests()

    if args.validate:
        print(json.dumps(results, indent=2))
    else:
        print("QA Guardrail MVP - Self Tests")
        print("=" * 40)
        for test in results["tests"]:
            print(f"  {test}")
        if results["errors"]:
            print()
            print("Errors:")
            for err in results["errors"]:
                print(f"  - {err}")
        print()
        print(f"Result: {'PASS' if results['ok'] else 'FAIL'}")

    sys.exit(0 if results["ok"] else 1)


if __name__ == "__main__":
    main()
