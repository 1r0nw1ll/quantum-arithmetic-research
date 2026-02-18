#!/usr/bin/env python3
"""
generate_multi_seed_report.py

Standalone multi-seed report for qa_kona_ebm_qa_native_v1.

Runs train_qa_rbm for 5 seeds (0,1,2,3,4) x n_epochs epochs, then prints a
table of per-seed orbit coherence scores and permutation-gap z-scores, followed
by summary statistics (mean +/- std) across seeds.

Not a certificate; no JSON output.

Usage:
    python qa_kona_ebm_qa_native_v1/generate_multi_seed_report.py
    python qa_kona_ebm_qa_native_v1/generate_multi_seed_report.py --n-epochs 20
    python qa_kona_ebm_qa_native_v1/generate_multi_seed_report.py --n-epochs 20 --n-samples 1000 --lr 0.01
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from rbm_qa_native_train import train_qa_rbm

SEEDS = [0, 1, 2, 3, 4]


def run_report(n_epochs: int, n_samples: int, lr: float) -> None:
    print(f"Multi-seed report: n_epochs={n_epochs}, n_samples={n_samples}, lr={lr}")
    print(f"Seeds: {SEEDS}")
    print()

    header = f"{'seed':>4}  {'COSMOS_c':>10}  {'SAT_c':>10}  {'COSMOS_z':>10}  {'SAT_z':>10}  {'status':>18}"
    print(header)
    print("-" * len(header))

    cosmos_c_vals = []
    sat_c_vals = []
    cosmos_z_vals = []
    sat_z_vals = []

    for seed in SEEDS:
        result = train_qa_rbm(n_samples=n_samples, n_epochs=n_epochs, lr=lr, seed=seed)
        status = result["status"]
        oa = result["orbit_analysis"]
        ocs = oa["orbit_coherence_score"]
        cgs = oa.get("coherence_gap_stats", {})

        cosmos_c = ocs.get("COSMOS", float("nan"))
        sat_c = ocs.get("SATELLITE", float("nan"))

        cosmos_z = cgs.get("COSMOS", {}).get("z_score", float("nan")) if cgs else float("nan")
        sat_z = cgs.get("SATELLITE", {}).get("z_score", float("nan")) if cgs else float("nan")

        cosmos_c_vals.append(cosmos_c)
        sat_c_vals.append(sat_c)
        cosmos_z_vals.append(cosmos_z)
        sat_z_vals.append(sat_z)

        print(
            f"{seed:>4}  {cosmos_c:>10.6f}  {sat_c:>10.6f}  "
            f"{cosmos_z:>10.6f}  {sat_z:>10.6f}  {status:>18}"
        )

    print()
    print("Summary (mean +/- std across seeds):")

    def _fmt(vals: list) -> str:
        arr = [v for v in vals if not (isinstance(v, float) and v != v)]  # drop NaN
        if not arr:
            return "N/A"
        m = float(np.mean(arr))
        s = float(np.std(arr))
        return f"{m:.6f} +/- {s:.6f}"

    print(f"  COSMOS coherence  : {_fmt(cosmos_c_vals)}")
    print(f"  SATELLITE coherence: {_fmt(sat_c_vals)}")
    print(f"  COSMOS z-score    : {_fmt(cosmos_z_vals)}")
    print(f"  SATELLITE z-score : {_fmt(sat_z_vals)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed orbit coherence + permutation-gap report for QA-native EBM."
    )
    parser.add_argument("--n-epochs",  type=int,   default=20,   help="Epochs per run (default: 20)")
    parser.add_argument("--n-samples", type=int,   default=1000, help="MNIST samples (default: 1000)")
    parser.add_argument("--lr",        type=float, default=0.01, help="Learning rate (default: 0.01)")
    args = parser.parse_args()
    run_report(n_epochs=args.n_epochs, n_samples=args.n_samples, lr=args.lr)


if __name__ == "__main__":
    main()
