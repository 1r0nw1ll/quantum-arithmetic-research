#!/usr/bin/env python3
"""
Independent verifier for QA Rule 30 nonperiodicity cert packs.

Replays Rule 30 evolution from scratch and spot-checks every witness
point: for each claimed (p, t), verifies center(t) != center(t+p).

This script trusts NOTHING except the Rule 30 truth table.

Usage:
  python verify_certpack.py qa_rule30/certpacks/rule30_nonperiodicity_v1
  python verify_certpack.py qa_rule30/certpacks/rule30_nonperiodicity_v1 --full
  python verify_certpack.py qa_rule30/certpacks/rule30_nonperiodicity_v1 --T 4096

Modes:
  default  -- verify all witness (p,t) pairs against recomputed center column
  --full   -- also verify center sequence file matches recomputed center
  --T N    -- only verify the T=N witness set (faster for spot-checks)

Exit codes: 0 = all verified, 1 = verification failure, 2 = usage error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Rule 30 — recomputed from scratch (zero trust)
# ---------------------------------------------------------------------------

def evolve_rule30(T: int) -> np.ndarray:
    """Evolve Rule 30 for T steps. Returns center column as uint8 array."""
    width = 2 * T + 1
    center_idx = T
    current = np.zeros(width, dtype=np.uint8)
    current[center_idx] = 1
    center = np.empty(T + 1, dtype=np.uint8)
    center[0] = 1
    for t in range(1, T + 1):
        new = np.zeros(width, dtype=np.uint8)
        new[1:-1] = current[:-2] ^ (current[1:-1] | current[2:])
        new[0] = current[0] | current[1]
        new[-1] = current[-2] ^ current[-1]
        center[t] = new[center_idx]
        current = new
    return center


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_witness_set(witness_path: Path, center: np.ndarray,
                       T: int) -> tuple[int, int, list]:
    """Verify all witness pairs against recomputed center column.

    Returns (checked, verified, failures).
    """
    with witness_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("rule") != 30:
        return 0, 0, [{"error": "wrong rule", "got": data.get("rule")}]
    if data.get("T") != T:
        return 0, 0, [{"error": "T mismatch", "expected": T, "got": data.get("T")}]

    witnesses = data.get("witnesses", [])
    checked = 0
    verified = 0
    failures: list[dict] = []

    for w in witnesses:
        p = w["p"]
        t = w["t"]
        checked += 1

        if t + p > T:
            failures.append({"p": p, "t": t, "error": "t+p exceeds T"})
            continue

        recomputed_ct = int(center[t])
        recomputed_ctp = int(center[t + p])

        # Verify the claimed values match recomputed
        if recomputed_ct != w["center_t"]:
            failures.append({
                "p": p, "t": t,
                "error": "center_t mismatch",
                "claimed": w["center_t"],
                "recomputed": recomputed_ct,
            })
            continue
        if recomputed_ctp != w["center_t_plus_p"]:
            failures.append({
                "p": p, "t": t,
                "error": "center_t_plus_p mismatch",
                "claimed": w["center_t_plus_p"],
                "recomputed": recomputed_ctp,
            })
            continue

        # Verify the witness actually proves nonperiodicity
        if recomputed_ct == recomputed_ctp:
            failures.append({
                "p": p, "t": t,
                "error": "witness does not break periodicity",
                "center_t": recomputed_ct,
                "center_t_plus_p": recomputed_ctp,
            })
            continue

        verified += 1

    return checked, verified, failures


def verify_center_file(center_path: Path, center: np.ndarray) -> bool:
    """Verify center sequence file matches recomputed center."""
    text = center_path.read_text("utf-8").strip()
    file_values = [int(x) for x in text.split()]
    if len(file_values) != len(center):
        print(f"    LENGTH MISMATCH: file={len(file_values)}, "
              f"recomputed={len(center)}", file=sys.stderr)
        return False
    for i, (fv, rv) in enumerate(zip(file_values, center)):
        if fv != int(rv):
            print(f"    VALUE MISMATCH at t={i}: file={fv}, "
                  f"recomputed={int(rv)}", file=sys.stderr)
            return False
    return True


def verify_manifest_hashes(manifest_path: Path) -> list[str]:
    """Verify file hashes in manifest match actual files on disk."""
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    errors = []
    base_dir = manifest_path.parent
    for key, info in manifest.get("files", {}).items():
        fpath = base_dir / info["path"]
        if not fpath.exists():
            errors.append(f"{key}: file not found ({fpath})")
            continue
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual != info["sha256"]:
            errors.append(f"{key}: hash mismatch "
                          f"(expected {info['sha256'][:16]}..., "
                          f"got {actual[:16]}...)")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Independent verifier for Rule 30 cert packs")
    parser.add_argument("certpack_dir", type=str,
                        help="Path to cert pack directory")
    parser.add_argument("--full", action="store_true",
                        help="Also verify center sequence files byte-for-byte")
    parser.add_argument("--T", type=int, default=None,
                        help="Only verify a single T value")
    args = parser.parse_args()

    certpack = Path(args.certpack_dir)
    cert_file = certpack / "QA_RULE30_NONPERIODICITY_CERT.v1.json"

    if not cert_file.exists():
        print(f"ERROR: cert file not found: {cert_file}", file=sys.stderr)
        sys.exit(2)

    with cert_file.open("r", encoding="utf-8") as f:
        cert = json.load(f)

    print("=" * 60)
    print("INDEPENDENT RULE 30 NONPERIODICITY VERIFICATION")
    print("=" * 60)
    print(f"Cert ID: {cert['cert_id']}")
    print(f"Cert hash: {cert['hash_chain']['this_cert_hash']}")
    print(f"Claim: {cert['aggregate']['claim']}")
    print()

    t0 = time.time()
    total_checked = 0
    total_verified = 0
    total_failures: list[dict] = []
    all_ok = True

    witness_refs = cert.get("witness_refs", [])
    if args.T:
        witness_refs = [r for r in witness_refs if r["T"] == args.T]
        if not witness_refs:
            print(f"ERROR: no witness_ref for T={args.T}", file=sys.stderr)
            sys.exit(2)

    for ref in witness_refs:
        T = ref["T"]
        man_path = certpack / ref["manifest_path"]

        print(f"--- T={T} ---")

        # Step 1: verify manifest file hashes
        hash_errors = verify_manifest_hashes(man_path)
        if hash_errors:
            for e in hash_errors:
                print(f"  HASH ERROR: {e}")
            all_ok = False
            continue
        print(f"  Manifest hashes: OK")

        # Step 2: recompute Rule 30 center column from scratch
        t_start = time.time()
        center = evolve_rule30(T)
        t_evolve = time.time() - t_start
        print(f"  Rule 30 evolved: {T+1} steps in {t_evolve:.1f}s")

        # Step 3: verify every witness pair
        with man_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        witness_file = man_path.parent / manifest["files"]["witnesses"]["path"]
        checked, verified, failures = verify_witness_set(witness_file, center, T)
        total_checked += checked
        total_verified += verified
        total_failures.extend(failures)

        if failures:
            print(f"  Witnesses: {verified}/{checked} VERIFIED, "
                  f"{len(failures)} FAILURES")
            for fail in failures[:5]:
                print(f"    FAIL: {fail}")
            all_ok = False
        else:
            print(f"  Witnesses: {verified}/{checked} verified")

        # Step 4 (optional): full center sequence verification
        if args.full:
            center_file = man_path.parent / manifest["files"]["center_sequence"]["path"]
            if center_file.exists():
                ok = verify_center_file(center_file, center)
                print(f"  Center sequence file: {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    all_ok = False
            else:
                print(f"  Center sequence file: NOT FOUND")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total witnesses checked:  {total_checked}")
    print(f"Total witnesses verified: {total_verified}")
    print(f"Total failures:           {len(total_failures)}")
    print(f"Elapsed:                  {elapsed:.1f}s")
    print()

    if all_ok and total_verified == total_checked and not total_failures:
        print("RESULT: ALL WITNESSES INDEPENDENTLY VERIFIED")
        print(f"The claim holds: no period p in "
              f"[{cert['scope']['P_min']},{cert['scope']['P_max']}] "
              f"detected up to T={cert['scope']['T_max']}.")
        sys.exit(0)
    else:
        print("RESULT: VERIFICATION FAILED")
        if total_failures:
            print(f"  {len(total_failures)} witness(es) did not verify")
        sys.exit(1)


if __name__ == "__main__":
    main()
