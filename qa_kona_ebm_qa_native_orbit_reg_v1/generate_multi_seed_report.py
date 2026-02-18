#!/usr/bin/env python3
"""
generate_multi_seed_report.py

Multi-seed report for qa_kona_ebm_qa_native_orbit_reg_v1 (family [64]).

Runs train_qa_orbit_reg_rbm for 5 seeds (0,1,2,3,4), prints a table of
per-seed orbit coherence scores and permutation-gap z-scores, followed by
summary statistics (mean +/- std) across seeds.

Optionally writes a deterministic JSON report with sha256 digest.

Usage:
    python qa_kona_ebm_qa_native_orbit_reg_v1/generate_multi_seed_report.py
    python qa_kona_ebm_qa_native_orbit_reg_v1/generate_multi_seed_report.py --n-epochs 20
    python qa_kona_ebm_qa_native_orbit_reg_v1/generate_multi_seed_report.py \
        --n-epochs 20 --n-samples 1000 --lr 0.01 --lambda-orbit 1e-3
    python qa_kona_ebm_qa_native_orbit_reg_v1/generate_multi_seed_report.py \
        --json-out /tmp/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from rbm_qa_orbit_reg_train import train_qa_orbit_reg_rbm

SEEDS = [0, 1, 2, 3, 4]
N_PERM = 500


def _round6(v: float) -> float:
    return round(float(v), 6)


def run_report(
    n_epochs: int,
    n_samples: int,
    lr: float,
    lambda_orbit: float,
    json_out,
) -> None:
    print(
        f"Multi-seed report: n_epochs={n_epochs}, n_samples={n_samples}, "
        f"lr={lr}, lambda_orbit={lambda_orbit}"
    )
    print(f"Seeds: {SEEDS}")
    print()

    header = (
        f"{'seed':>4}  {'COSMOS_c':>10}  {'SAT_c':>10}  "
        f"{'COSMOS_z':>10}  {'SAT_z':>10}  {'reg_norm_final':>15}  {'status':>30}"
    )
    print(header)
    print("-" * len(header))

    cosmos_c_vals = []
    sat_c_vals = []
    cosmos_z_vals = []
    sat_z_vals = []

    per_seed_records = []

    for seed in SEEDS:
        result = train_qa_orbit_reg_rbm(
            n_samples=n_samples,
            n_epochs=n_epochs,
            lr=lr,
            lambda_orbit=lambda_orbit,
            seed=seed,
        )
        status = result["status"]
        oa = result["orbit_analysis"]
        ocs = oa["orbit_coherence_score"]
        cgs = oa.get("coherence_gap_stats", {})

        cosmos_c = ocs.get("COSMOS", float("nan"))
        sat_c    = ocs.get("SATELLITE", float("nan"))

        cosmos_z = cgs.get("COSMOS",   {}).get("z_score", float("nan")) if cgs else float("nan")
        sat_z    = cgs.get("SATELLITE", {}).get("z_score", float("nan")) if cgs else float("nan")

        reg_norms = result.get("reg_norm_per_epoch", [])
        reg_final = reg_norms[-1] if reg_norms else float("nan")

        cosmos_c_vals.append(cosmos_c)
        sat_c_vals.append(sat_c)
        cosmos_z_vals.append(cosmos_z)
        sat_z_vals.append(sat_z)

        print(
            f"{seed:>4}  {cosmos_c:>10.6f}  {sat_c:>10.6f}  "
            f"{cosmos_z:>10.6f}  {sat_z:>10.6f}  "
            f"{reg_final:>15.6f}  {status:>30}"
        )

        final_energy = result["energy_per_epoch"][-1] if result["energy_per_epoch"] else float("nan")
        cosmos_cgs = cgs.get("COSMOS",   {}) if cgs else {}
        sat_cgs    = cgs.get("SATELLITE", {}) if cgs else {}

        per_seed_records.append({
            "seed":   seed,
            "status": status,
            "final_energy":        _round6(final_energy),
            "reg_norm_per_epoch_last": _round6(reg_final),
            "orbit_coherence_score": {
                "COSMOS":    _round6(cosmos_c),
                "SATELLITE": _round6(sat_c),
            },
            "coherence_gap_stats": {
                "COSMOS": {
                    "c_real":      _round6(cosmos_cgs.get("c_real",      float("nan"))),
                    "c_perm_mean": _round6(cosmos_cgs.get("c_perm_mean", float("nan"))),
                    "z_score":     _round6(cosmos_z),
                    "p_value":     _round6(cosmos_cgs.get("p_value",     float("nan"))),
                },
                "SATELLITE": {
                    "c_real":      _round6(sat_cgs.get("c_real",      float("nan"))),
                    "c_perm_mean": _round6(sat_cgs.get("c_perm_mean", float("nan"))),
                    "z_score":     _round6(sat_z),
                    "p_value":     _round6(sat_cgs.get("p_value",     float("nan"))),
                },
            },
        })

    print()
    print("Summary (mean +/- std across seeds):")

    def _fmt(vals: list) -> str:
        arr = [v for v in vals if not (isinstance(v, float) and v != v)]
        if not arr:
            return "N/A"
        m = float(np.mean(arr))
        s = float(np.std(arr))
        return f"{m:.6f} +/- {s:.6f}"

    print(f"  COSMOS coherence   : {_fmt(cosmos_c_vals)}")
    print(f"  SATELLITE coherence: {_fmt(sat_c_vals)}")
    print(f"  COSMOS z-score     : {_fmt(cosmos_z_vals)}")
    print(f"  SATELLITE z-score  : {_fmt(sat_z_vals)}")

    if json_out is not None:
        def _safe_mean(vals):
            arr = [v for v in vals if not (isinstance(v, float) and v != v)]
            return _round6(float(np.mean(arr))) if arr else float("nan")

        def _safe_std(vals):
            arr = [v for v in vals if not (isinstance(v, float) and v != v)]
            return _round6(float(np.std(arr))) if arr else float("nan")

        report_body = {
            "config": {
                "n_epochs":     n_epochs,
                "n_samples":    n_samples,
                "lr":           float(lr),
                "lambda_orbit": float(lambda_orbit),
                "seeds":        list(SEEDS),
                "n_perm":       N_PERM,
            },
            "per_seed": per_seed_records,
            "aggregate": {
                "COSMOS_coherence_mean":    _safe_mean(cosmos_c_vals),
                "COSMOS_coherence_std":     _safe_std(cosmos_c_vals),
                "COSMOS_zscore_mean":       _safe_mean(cosmos_z_vals),
                "COSMOS_zscore_std":        _safe_std(cosmos_z_vals),
                "SATELLITE_coherence_mean": _safe_mean(sat_c_vals),
                "SATELLITE_coherence_std":  _safe_std(sat_c_vals),
                "SATELLITE_zscore_mean":    _safe_mean(sat_z_vals),
                "SATELLITE_zscore_std":     _safe_std(sat_z_vals),
                "baseline_family":          "qa_kona_ebm_qa_native_v1",
            },
        }

        body_bytes = json.dumps(
            report_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        digest = hashlib.sha256(body_bytes).hexdigest()
        report_body["report_sha256"] = digest

        with open(json_out, "w") as fh:
            json.dump(report_body, fh, indent=2)
        print(f"JSON report written to: {json_out}")
        print(f"report_sha256: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed orbit coherence report for QA-native EBM with orbit regularizer."
    )
    parser.add_argument("--n-epochs",     type=int,   default=20,   help="Epochs per run (default: 20)")
    parser.add_argument("--n-samples",    type=int,   default=1000, help="MNIST samples (default: 1000)")
    parser.add_argument("--lr",           type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--lambda-orbit", type=float, default=1e-3, dest="lambda_orbit",
                        help="Orbit regularizer strength (default: 1e-3)")
    parser.add_argument("--json-out",     type=str,   default=None, dest="json_out",
                        help="Optional path to write JSON report")
    args = parser.parse_args()
    run_report(
        n_epochs=args.n_epochs,
        n_samples=args.n_samples,
        lr=args.lr,
        lambda_orbit=args.lambda_orbit,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
