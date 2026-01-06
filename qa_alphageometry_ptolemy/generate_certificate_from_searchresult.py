#!/usr/bin/env python3
"""
Generate QA certificate from AlphaGeometry SearchResult JSON

Usage:
    python3 generate_certificate_from_searchresult.py <searchresult.json> [output.cert.json]
"""

import json
import sys
from pathlib import Path

# Import certificate adapter
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable SearchResult files:")
        for p in Path("/home/player2/signal_experiments/qa_alphageometry/core").glob("*.searchresult.json"):
            print(f"  {p}")
        sys.exit(1)

    input_path = sys.argv[1]

    # Auto-generate output path if not provided
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Convert searchresult.json → cert.json
        output_path = input_path.replace(".searchresult.json", ".cert.json")
        if output_path == input_path:
            output_path = input_path.replace(".json", ".cert.json")

    # Load SearchResult
    print(f"📂 Loading SearchResult from: {input_path}")
    with open(input_path, 'r') as f:
        search_result = json.load(f)

    # Extract theorem ID from filename
    theorem_id = Path(input_path).stem.replace(".searchresult", "")

    print(f"🔍 Processing SearchResult:")
    print(f"   Theorem ID: {theorem_id}")
    print(f"   Solved: {search_result['solved']}")
    print(f"   States expanded: {search_result['states_expanded']}")
    print(f"   Depth reached: {search_result['depth_reached']}")

    if search_result['proof']:
        print(f"   Proof steps: {len(search_result['proof']['steps'])}")

    # Generate certificate
    print(f"\n🔨 Generating certificate...")
    cert = wrap_searchresult_to_certificate(
        search_result,
        theorem_id=theorem_id,
        max_depth_limit=50,
        repo_tag="qa-alphageometry-v0.1",
        commit="auto-generated"
    )

    # Serialize certificate
    cert_json = cert.to_json()

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(cert_json, f, indent=2)

    print(f"\n✅ Certificate generated: {output_path}")
    print(f"   Schema version: {cert_json['schema_version']}")
    print(f"   Witness type: {cert_json['witness_type']}")
    print(f"   Theorem ID: {cert_json['theorem_id']}")

    if cert_json['witness_type'] == 'success':
        print(f"   Success path length: {len(cert_json['success_path'])}")
        print(f"   Generator set: {[g['name'] for g in cert_json['generator_set']]}")
    else:
        print(f"   Obstruction type: {cert_json['obstruction']['fail_type']}")

    print(f"\n📊 Certificate statistics:")
    print(f"   States explored: {cert_json['search']['states_explored']}")
    print(f"   Max depth: {cert_json['search']['max_depth']}")


if __name__ == "__main__":
    main()
