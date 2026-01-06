#!/usr/bin/env python3
"""
Generate Ptolemy theorem certificates from QA-AlphaGeometry SearchResult.

Usage:
    python generate_ptolemy_certificates.py --in ptolemy_searchresult.json

This will create:
    - artifacts/ptolemy_success.cert.json (if solved=True)
    - artifacts/ptolemy_ablated.obstruction.cert.json (if solved=False)
"""

import argparse
import json
from pathlib import Path

# Import the certificate adapter
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate


def main():
    parser = argparse.ArgumentParser(
        description="Generate Ptolemy certificates from SearchResult JSON"
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        required=True,
        help="Input SearchResult JSON file"
    )
    parser.add_argument(
        "--theorem",
        default="ptolemy_quadrance",
        help="Theorem ID (default: ptolemy_quadrance)"
    )
    parser.add_argument(
        "--tag",
        default="qa-alphageometry-ptolemy-v0.1",
        help="Repo tag for certificate"
    )
    parser.add_argument(
        "--commit",
        default="4064bce",
        help="Git commit hash"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=50,
        help="Max depth limit (for inferring stop reason)"
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        help="Output certificate file (auto-generated if not specified)"
    )

    args = parser.parse_args()

    # Load SearchResult
    print(f"Loading SearchResult from {args.input_file}...")
    with open(args.input_file) as f:
        sr = json.load(f)

    # Check if solved
    solved = sr.get("solved", False)
    print(f"  solved: {solved}")
    print(f"  states_expanded: {sr.get('states_expanded', 0)}")
    print(f"  depth_reached: {sr.get('depth_reached', 0)}")

    # Generate certificate
    print(f"\nGenerating certificate...")
    cert = wrap_searchresult_to_certificate(
        sr,
        theorem_id=args.theorem,
        max_depth_limit=args.max_depth,
        repo_tag=args.tag,
        commit=args.commit
    )

    # Determine output filename
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Auto-generate based on witness type
        suffix = "success" if cert.witness_type == "success" else "obstruction"
        output_path = Path("artifacts") / f"{args.theorem}.{suffix}.cert.json"

    # Ensure artifacts directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write certificate
    with open(output_path, "w") as f:
        json.dump(cert.to_json(), f, indent=2)

    print(f"\n✓ Certificate generated: {output_path}")
    print(f"  Type: {cert.witness_type}")

    if cert.witness_type == "success":
        print(f"  Proof length: {len(cert.success_path)} steps")
        print(f"  Generator set: {len(cert.generator_set)} generators")

        # Extract rule IDs
        rule_ids = [
            step["gen"]["name"].replace("AG:", "")
            for step in cert.to_json()["success_path"]
        ]
        print(f"  Rules used: {', '.join(rule_ids[:5])}" +
              (f" (+{len(rule_ids)-5} more)" if len(rule_ids) > 5 else ""))
    else:
        obs = cert.obstruction
        print(f"  Fail type: {obs.fail_type.value}")
        print(f"  Max depth reached: {obs.max_depth_reached}")
        print(f"  States explored: {obs.states_explored}")
        print(f"  Inferred reason: {cert.context.get('inferred_stop_reason', 'N/A')}")

    print(f"\n✓ Done! Certificate ready for paper submission.")
    return 0


if __name__ == "__main__":
    exit(main())
