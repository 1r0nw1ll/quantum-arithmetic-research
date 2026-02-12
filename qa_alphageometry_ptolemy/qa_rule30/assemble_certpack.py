#!/usr/bin/env python3
"""
Assemble a QA_RULE30_NONPERIODICITY_CERT from witness manifests.

Runs witness generation for each T value, validates manifests,
and emits the final cert with computed self-hash.

Usage:
  python assemble_certpack.py --outdir qa_rule30/certpacks/rule30_nonperiodicity_v1
  python assemble_certpack.py --outdir /tmp/certpack --T 4096 16384  # subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Frozen scope v1
# ---------------------------------------------------------------------------

DEFAULT_T_VALUES = [4096, 8192, 16384, 32768, 65536]
P_MIN = 1
DEFAULT_P_MAX = 256
K_RANGE = {"min": 4, "max": 16}
AGENT_ID = "qa-agent-ctrl-1"

CERT_SCHEMA_ID = "QA_RULE30_NONPERIODICITY_CERT_SCHEMA.v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assemble Rule 30 cert pack")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--T", type=int, nargs="+", default=None,
                        help="T values to use (default: all 5)")
    parser.add_argument("--P_max", type=int, default=DEFAULT_P_MAX,
                        help=f"Maximum period to check (default: {DEFAULT_P_MAX})")
    args = parser.parse_args()

    P_MAX = args.P_max
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_values = args.T or DEFAULT_T_VALUES
    t_values = sorted(set(t_values))
    if not t_values:
        print("FATAL: no T values provided", file=sys.stderr)
        sys.exit(2)

    min_t = min(t_values)
    safe_p_max = min_t // 2
    if P_MAX >= min_t:
        print(
            f"FATAL: invalid scope P_max={P_MAX} with min(T)={min_t}. "
            "Require P_max < min(T).",
            file=sys.stderr,
        )
        sys.exit(2)
    if P_MAX > safe_p_max:
        print(
            f"WARNING: P_max={P_MAX} exceeds recommended safe margin "
            f"min(T)//2={safe_p_max}.",
            file=sys.stderr,
        )
    witness_dir = outdir / "witnesses"

    t0 = time.time()
    print(f"=== Rule 30 Cert Pack Assembly ===", file=sys.stderr)
    print(f"Scope: P=[{P_MIN},{P_MAX}], T={t_values}, k=[{K_RANGE['min']},{K_RANGE['max']}]",
          file=sys.stderr)

    # Phase 1: generate witnesses for each T
    witness_refs: List[Dict[str, Any]] = []
    script_dir = Path(__file__).parent

    for T in t_values:
        t_dir = witness_dir / f"T{T}"
        print(f"\n--- Generating witnesses for T={T} ---", file=sys.stderr)
        result = subprocess.run(
            [sys.executable, str(script_dir / "generate_witnesses.py"),
             "--T", str(T), "--P_max", str(P_MAX), "--P_min", str(P_MIN),
             "--outdir", str(t_dir)],
            capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"FATAL: witness generation failed for T={T}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        manifest_path = t_dir / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))

        # Verify manifest via validator
        val_result = subprocess.run(
            [sys.executable, str(script_dir / "qa_rule30_cert_validator.py"),
             "manifest", str(manifest_path), "--verify-files", "--ci"],
            capture_output=True, text=True)

        if val_result.returncode != 0:
            print(f"FATAL: manifest validation failed for T={T}", file=sys.stderr)
            print(val_result.stdout, file=sys.stderr)
            sys.exit(1)

        print(f"  {val_result.stdout.strip()}", file=sys.stderr)

        manifest_hash = _sha256_bytes(manifest_path.read_bytes())
        witness_refs.append({
            "T": T,
            "P_min": P_MIN,
            "P_max": P_MAX,
            "manifest_path": str(manifest_path.relative_to(outdir)),
            "manifest_sha256": manifest_hash,
            "verified_periods": manifest["verified_periods"],
            "failure_count": manifest["failure_count"],
        })

    # Phase 2: compute aggregates
    total_verified = sum(r["verified_periods"] for r in witness_refs)
    total_failures = sum(r["failure_count"] for r in witness_refs)
    total_checked = total_verified + total_failures
    t_vals_sorted = sorted(set(r["T"] for r in witness_refs))

    T_max = max(t_vals_sorted)
    claim = (f"No period p in [{P_MIN},{P_MAX}] detected for Rule 30 center column "
             f"at any T in {t_vals_sorted}, up to T_max={T_max}. "
             f"Verified: {total_verified}/{total_checked}, failures: {total_failures}.")

    # Phase 3: assemble cert
    cert = {
        "schema_id": CERT_SCHEMA_ID,
        "cert_id": f"CERT-RULE30-NONPERIOD-V1-P{P_MAX}-T{T_max}",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent_id": AGENT_ID,
        "scope": {
            "rule": 30,
            "initial_condition": "single_1_at_origin",
            "P_min": P_MIN,
            "P_max": P_MAX,
            "T_max": T_max,
            "k_range": K_RANGE,
        },
        "witness_refs": witness_refs,
        "bundle_refs": [],
        "aggregate": {
            "total_periods_checked": total_checked,
            "total_verified": total_verified,
            "total_failures": total_failures,
            "T_values": t_vals_sorted,
            "claim": claim,
        },
        "hash_chain": {
            "prev_cert_hash": "0" * 64,
            "this_cert_hash": "",
        },
        "invariant_diff": {
            "scope": {"P_min": P_MIN, "P_max": P_MAX, "T_max": T_max},
            "total_verified": total_verified,
            "total_failures": total_failures,
            "generation_seconds": round(time.time() - t0, 1),
        },
    }

    # Compute self-hash
    self_hash = _sha256_str(_canonical(cert))
    cert["hash_chain"]["this_cert_hash"] = self_hash

    # Write cert
    cert_path = outdir / "QA_RULE30_NONPERIODICITY_CERT.v1.json"
    cert_path.write_text(_canonical(cert) + "\n", encoding="utf-8")

    # Validate cert
    val_result = subprocess.run(
        [sys.executable, str(script_dir / "qa_rule30_cert_validator.py"),
         "cert", str(cert_path), "--ci"],
        capture_output=True, text=True)

    total_time = time.time() - t0
    print(f"\n=== Assembly Complete ===", file=sys.stderr)
    print(f"Cert: {cert_path}", file=sys.stderr)
    print(f"Self-hash: {self_hash}", file=sys.stderr)
    print(f"Claim: {claim}", file=sys.stderr)
    print(f"Validation: {val_result.stdout.strip()}", file=sys.stderr)
    print(f"Total time: {total_time:.1f}s", file=sys.stderr)

    if val_result.returncode != 0:
        print("FATAL: cert validation failed!", file=sys.stderr)
        sys.exit(1)

    # Print cert path to stdout
    print(str(cert_path))


if __name__ == "__main__":
    main()
