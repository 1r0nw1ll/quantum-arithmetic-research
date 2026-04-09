#!/usr/bin/env python3
"""
qa_fixture_builder.py — Generate correct QA cert fixture witness blocks.

Uses qa_elements() as the ONLY source of element values. Never hand-compute.

Usage:
    # Generate witness JSON for specific (b,e) pairs:
    python tools/qa_fixture_builder.py --be 1,1 --be 5,2 --be 9,9

    # Verify an existing fixture file:
    python tools/qa_fixture_builder.py --verify path/to/fixture.json

    # Generate + verify:
    python tools/qa_fixture_builder.py --be 2,1 --be 3,3 --verify-output
"""

QA_COMPLIANCE = "fixture_builder — canonical element generation; no empirical computation"

import json
import sys
from pathlib import Path

# Import the authoritative element oracle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qa_alphageometry_ptolemy"))
from qa_elements import qa_elements, verify_fixture_witness


def build_witness(b: int, e: int, extra: dict = None) -> dict:
    """Build a complete witness dict with canonical element values.

    All values computed by qa_elements(). Never hand-derived.
    """
    elems = qa_elements(b, e)
    witness = {
        "state_be": [b, e],
        "tuple_beda": [elems.b, elems.e, elems.d, elems.a],
        "d": elems.d,
        "e": elems.e,
        "a": elems.a,
        "C": elems.C,
        "F": elems.F,
        "G": elems.G,
        "I": elems.I,
        "conic_type": "hyperbola" if elems.I > 0 else ("parabola" if elems.I == 0 else "ellipse"),
    }
    # Add products in string form for [208]-style witnesses
    witness["products"] = {
        "B": f"{b}*{b}={elems.B}",
        "E": f"{e}*{e}={elems.E}",
        "C": f"2*{elems.d}*{e}={elems.C}",
        "F": f"{b}*{elems.a}={elems.F}",
        "J": f"{b}*{elems.d}={elems.J}",
    }
    if extra:
        witness.update(extra)
    return witness


def verify_fixture_file(path: str) -> list:
    """Verify all witnesses in a fixture JSON file.

    Returns list of (witness_index, error_list) tuples.
    """
    with open(path) as f:
        fixture = json.load(f)

    all_errors = []

    # Check top-level witnesses
    witnesses = fixture.get("witnesses", [])
    for i, w in enumerate(witnesses):
        errors = verify_fixture_witness(w)
        if errors:
            all_errors.append((i, errors))

    return all_errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA fixture witness builder")
    parser.add_argument("--be", action="append", metavar="B,E",
                        help="(b,e) pair to generate witness for (repeatable)")
    parser.add_argument("--verify", metavar="FILE",
                        help="Verify an existing fixture JSON file")
    parser.add_argument("--verify-output", action="store_true",
                        help="Verify generated witnesses against canonical values")
    args = parser.parse_args()

    if args.verify:
        errors = verify_fixture_file(args.verify)
        if errors:
            print(f"FAIL: {len(errors)} witnesses have errors:")
            for idx, errs in errors:
                print(f"  witness[{idx}]:")
                for err in errs:
                    print(f"    - {err}")
            sys.exit(1)
        else:
            print(f"PASS: all witnesses in {args.verify} are correct")
            sys.exit(0)

    if args.be:
        witnesses = []
        for pair in args.be:
            b, e = map(int, pair.split(","))
            w = build_witness(b, e)
            witnesses.append(w)

            if args.verify_output:
                errs = verify_fixture_witness(w)
                if errs:
                    print(f"SELF-CHECK FAILED for ({b},{e}): {errs}")
                    sys.exit(1)

        print(json.dumps(witnesses, indent=2))
        if args.verify_output:
            print(f"\n# All {len(witnesses)} witnesses verified against qa_elements()")
        sys.exit(0)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
