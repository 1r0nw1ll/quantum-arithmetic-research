#!/usr/bin/env python3
"""
generate_cert.py

Generate a QA_KONA_EBM_QA_NATIVE_CERT.v1 certificate JSON from a QA-indexed RBM training run.

n_hidden is fixed at 81 (one per QA state) and is NOT a user parameter.

Usage:
    python generate_cert.py --n-samples 1000 --n-epochs 5 --lr 0.01 --seed 42 --cert-id kona_ebm_qa_native_001

Prints cert JSON to stdout.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rbm_qa_native_train import train_qa_rbm


def build_cert(cert_id: str, run: dict) -> dict:
    status = run["status"]

    if status == "GRADIENT_EXPLOSION":
        invariant_diff = {
            "fail_type": "GRADIENT_EXPLOSION",
            "target_path": "result.grad_norm_per_epoch[0]",
            "reason": (
                f"gradient norm exceeded threshold 1000.0 "
                f"-- training unstable under lr={run['lr']}"
            ),
        }
    else:
        invariant_diff = None

    return {
        "cert_type": "QA_KONA_EBM_QA_NATIVE_CERT.v1",
        "schema_version": 1,
        "cert_id": cert_id,
        "issued_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_config": {
            "n_visible": run["n_visible"],
            "n_samples": run["n_samples"],
            "n_epochs": run["n_epochs"],
            "lr": run["lr"],
            "seed": run["seed"],
            "algorithm": "rbm_qa_native_cd1_numpy",
        },
        "result": {
            "status": status,
            "energy_per_epoch": run["energy_per_epoch"],
            "reconstruction_error_per_epoch": run["reconstruction_error_per_epoch"],
            "grad_norm_per_epoch": run["grad_norm_per_epoch"],
            "final_weights_norm": run["final_weights_norm"],
            "invariant_diff": invariant_diff,
            "orbit_analysis": run["orbit_analysis"],
        },
        "trace": {
            "trace_hash": run["trace_hash"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA_KONA_EBM_QA_NATIVE_CERT.v1 JSON")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cert-id", type=str, default="generated_run")
    args = parser.parse_args()

    run = train_qa_rbm(
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed,
    )

    cert = build_cert(args.cert_id, run)
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
