#!/usr/bin/env python3
"""
Rule 30 center column nonperiodicity witness generator — numpy-vectorized.

For each period p in [P_MIN, P_MAX], finds the smallest t where
center(t) != center(t+p), proving center is NOT periodic with period p
within the time horizon [0, T].

Rule 30 update: new[i] = left XOR (center OR right)

Usage:
  python generate_witnesses.py --T 16384 --P_max 256 --outdir /tmp/rule30
  python generate_witnesses.py --T 65536 --P_max 256 --outdir /tmp/rule30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core computation (numpy-vectorized)
# ---------------------------------------------------------------------------

def evolve_rule30(T: int, *, progress_every: int = 4096) -> np.ndarray:
    """
    Evolve Rule 30 for T steps, extract center column.

    Uses numpy vectorized operations: new[1:-1] = left ^ (center | right)
    Memory: O(width) = O(2T+1).  Time: O(T * width).
    """
    width = 2 * T + 1
    center_idx = T

    current = np.zeros(width, dtype=np.uint8)
    current[center_idx] = 1

    center = np.empty(T + 1, dtype=np.uint8)
    center[0] = 1

    for t in range(1, T + 1):
        # Rule 30: left XOR (center OR right)
        new = np.zeros(width, dtype=np.uint8)
        new[1:-1] = current[:-2] ^ (current[1:-1] | current[2:])
        # Boundary: new[0] = 0 ^ (current[0] | current[1]), new[-1] = current[-2] ^ (current[-1] | 0)
        new[0] = current[0] | current[1]          # left=0, so XOR = identity
        new[-1] = current[-2] ^ current[-1]        # right=0, so OR = center
        center[t] = new[center_idx]
        current = new
        if progress_every and t % progress_every == 0:
            print(f"  evolve: t={t}/{T}", file=sys.stderr)

    return center


def find_witnesses(center: np.ndarray, P_min: int, P_max: int
                   ) -> Tuple[List[Dict[str, int]], List[int]]:
    """
    For each period p in [P_min, P_max], find smallest t where
    center[t] != center[t+p].

    Returns (witnesses, failures).
    """
    T = len(center) - 1
    witnesses: List[Dict[str, int]] = []
    failures: List[int] = []

    for p in range(P_min, P_max + 1):
        # Vectorized: find first index where center[0:T-p+1] != center[p:T+1]
        diff = center[:T - p + 1] != center[p:T + 1]
        idx = np.argmax(diff)
        if diff[idx]:
            witnesses.append({
                "p": int(p),
                "t": int(idx),
                "center_t": int(center[idx]),
                "center_t_plus_p": int(center[idx + p]),
            })
        else:
            failures.append(p)

    return witnesses, failures


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def emit_witness_bundle(T: int, P_min: int, P_max: int,
                        witnesses: List[Dict[str, int]],
                        failures: List[int],
                        center: np.ndarray,
                        outdir: Path) -> Dict[str, Any]:
    """Write witness data + manifest to outdir. Returns manifest dict."""
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Witness JSON (canonical)
    witness_data = {
        "rule": 30,
        "T": T,
        "P_min": P_min,
        "P_max": P_max,
        "initial_condition": "single_1_at_origin",
        "witnesses": witnesses,
        "failures": failures,
    }
    witness_path = outdir / f"witnesses_P{P_min}-{P_max}_T{T}.json"
    witness_canonical = _canonical(witness_data) + "\n"
    witness_path.write_text(witness_canonical, encoding="utf-8")
    witness_hash = _sha256_bytes(witness_path.read_bytes())

    # 2) Center sequence (binary, space-separated)
    center_path = outdir / f"center_T{T}.txt"
    center_text = " ".join(str(int(b)) for b in center) + "\n"
    center_path.write_text(center_text, encoding="utf-8")
    center_hash = _sha256_bytes(center_path.read_bytes())

    # 3) Manifest
    manifest = {
        "schema_id": "QA_RULE30_WITNESS_MANIFEST.v1",
        "rule": 30,
        "T": T,
        "P_min": P_min,
        "P_max": P_max,
        "initial_condition": "single_1_at_origin",
        "total_periods": P_max - P_min + 1,
        "verified_periods": len(witnesses),
        "failure_count": len(failures),
        "failures": failures,
        "files": {
            "witnesses": {
                "path": witness_path.name,
                "sha256": witness_hash,
            },
            "center_sequence": {
                "path": center_path.name,
                "sha256": center_hash,
            },
        },
    }
    manifest_path = outdir / "MANIFEST.json"
    manifest_path.write_text(_canonical(manifest) + "\n", encoding="utf-8")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule 30 center column nonperiodicity witness generator")
    parser.add_argument("--T", type=int, required=True,
                        help="Time horizon (e.g. 16384, 65536)")
    parser.add_argument("--P_min", type=int, default=1,
                        help="Minimum period to check (default: 1)")
    parser.add_argument("--P_max", type=int, required=True,
                        help="Maximum period to check (e.g. 256)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Rule 30 witness generator: T={args.T}, P=[{args.P_min},{args.P_max}]",
          file=sys.stderr)

    print("Phase 1: evolving Rule 30...", file=sys.stderr)
    center = evolve_rule30(args.T)
    t_evolve = time.time() - t0
    print(f"  evolution done in {t_evolve:.1f}s", file=sys.stderr)

    print("Phase 2: finding witnesses...", file=sys.stderr)
    witnesses, failures = find_witnesses(center, args.P_min, args.P_max)
    t_witness = time.time() - t0 - t_evolve
    print(f"  witnesses done in {t_witness:.1f}s", file=sys.stderr)

    if failures:
        print(f"WARNING: {len(failures)} periods without counterexample: {failures}",
              file=sys.stderr)

    print(f"Phase 3: writing output to {args.outdir}...", file=sys.stderr)
    manifest = emit_witness_bundle(
        args.T, args.P_min, args.P_max, witnesses, failures, center,
        Path(args.outdir))

    total = time.time() - t0
    print(f"Done: {manifest['verified_periods']}/{manifest['total_periods']} "
          f"periods verified, {manifest['failure_count']} failures, "
          f"{total:.1f}s total", file=sys.stderr)

    # Print manifest path to stdout for piping
    print(str(Path(args.outdir) / "MANIFEST.json"))


if __name__ == "__main__":
    main()
