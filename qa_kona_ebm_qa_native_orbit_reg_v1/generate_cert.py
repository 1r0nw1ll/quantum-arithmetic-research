#!/usr/bin/env python3
"""
generate_cert.py

Generate a QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1 certificate JSON.

n_hidden is fixed at 81 (one per QA state) and is NOT a user parameter.

Usage:
    python generate_cert.py --n-samples 1000 --n-epochs 5 --lr 0.01 \
        --lambda-orbit 1e-3 --seed 42 --cert-id kona_ebm_orbit_reg_stable_001

Prints cert JSON to stdout.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rbm_qa_orbit_reg_train import train_qa_orbit_reg_rbm


def build_cert(cert_id: str, run: dict) -> dict:
    status = run["status"]

    if status == "GRADIENT_EXPLOSION":
        invariant_diff = {
            "fail_type":   "GRADIENT_EXPLOSION",
            "target_path": "result.grad_norm_per_epoch[0]",
            "reason": (
                f"gradient norm exceeded threshold 1000.0 "
                f"-- training unstable under lr={run['lr']}"
            ),
        }
    elif status == "REGULARIZER_NUMERIC_INSTABILITY":
        invariant_diff = {
            "fail_type":   "REGULARIZER_NUMERIC_INSTABILITY",
            "target_path": "result.status",
            "reason":      "NaN/Inf detected in model parameters during regularized training",
        }
    else:
        invariant_diff = None

    return {
        "cert_type":      "QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1",
        "schema_version": 1,
        "cert_id":        cert_id,
        "issued_utc":     datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_config": {
            "n_visible":    run["n_visible"],
            "n_samples":    run["n_samples"],
            "n_epochs":     run["n_epochs"],
            "lr":           run["lr"],
            "lambda_orbit": run["lambda_orbit"],
            "seed":         run["seed"],
            "algorithm":    "rbm_qa_native_orbit_reg_cd1_numpy",
        },
        "result": {
            "status":                          status,
            "energy_per_epoch":                run["energy_per_epoch"],
            "reconstruction_error_per_epoch":  run["reconstruction_error_per_epoch"],
            "grad_norm_per_epoch":             run["grad_norm_per_epoch"],
            "reg_norm_per_epoch":              run["reg_norm_per_epoch"],
            "lr_per_epoch":                    run["lr_per_epoch"],
            "final_weights_norm":              run["final_weights_norm"],
            "reg_trace_hash":                  run["reg_trace_hash"],
            "invariant_diff":                  invariant_diff,
            "orbit_analysis":                  run["orbit_analysis"],
            "generator_curvature": {
                "definition":         "kappa_hat = 1 - abs(1 - lr * lambda_orbit)",
                "kappa_hat_per_epoch": run["kappa_hat_per_epoch"],
                "min_kappa_hat":      run["min_kappa_hat"],
                "min_kappa_epoch":    run["min_kappa_epoch"],
                "kappa_hash":         run["kappa_hash"],
                "max_dev_norm":        run["max_dev_norm"],
                "max_dev_epoch":       run["max_dev_epoch"],
            },
        },
        "trace": {
            "trace_hash": run["trace_hash"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1 JSON"
    )
    parser.add_argument("--n-samples",    type=int,   default=1000)
    parser.add_argument("--n-epochs",     type=int,   default=5)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--lambda-orbit", type=float, default=1e-3, dest="lambda_orbit")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--cert-id",      type=str,   default="generated_run")
    args = parser.parse_args()

    run = train_qa_orbit_reg_rbm(
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        lr=args.lr,
        lambda_orbit=args.lambda_orbit,
        seed=args.seed,
    )

    cert = build_cert(args.cert_id, run)
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
