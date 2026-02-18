#!/usr/bin/env python3
"""
generate_cert.py

Generate a QA_KONA_EBM_MNIST_CERT.v1 certificate JSON from a training run.

Usage:
    python generate_cert.py --n-hidden 64 --n-samples 1000 --n-epochs 5 \
        --lr 0.01 --seed 42 --cert-id my_run
    python generate_cert.py --n-hidden 64 --n-samples 1000 --n-epochs 5 \
        --lr 50.0 --seed 42 --cert-id explosion_test

Prints cert JSON to stdout.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import os

# Make sibling imports work when called from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rbm_train import train_rbm


def build_cert(cert_id: str, run: dict) -> dict:
    """Construct cert dict from a train_rbm() result."""
    status = run["status"]

    if status == "GRADIENT_EXPLOSION":
        invariant_diff = {
            "fail_type": "GRADIENT_EXPLOSION",
            "target_path": "result.grad_norm_per_epoch[0]",
            "reason": (
                f"gradient norm exceeded threshold 1000.0 "
                f"— training unstable under lr={run['lr']}"
            ),
        }
    else:
        invariant_diff = None

    return {
        "cert_type": "QA_KONA_EBM_MNIST_CERT.v1",
        "schema_version": 1,
        "cert_id": cert_id,
        "issued_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_config": {
            "n_visible": run["n_visible"],
            "n_hidden": run["n_hidden"],
            "n_samples": run["n_samples"],
            "n_epochs": run["n_epochs"],
            "lr": run["lr"],
            "seed": run["seed"],
            "algorithm": "rbm_cd1_numpy",
        },
        "result": {
            "status": status,
            "energy_per_epoch": run["energy_per_epoch"],
            "reconstruction_error_per_epoch": run["reconstruction_error_per_epoch"],
            "grad_norm_per_epoch": run["grad_norm_per_epoch"],
            "final_weights_norm": run["final_weights_norm"],
            "invariant_diff": invariant_diff,
        },
        "trace": {
            "trace_hash": run["trace_hash"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA_KONA_EBM_MNIST_CERT.v1 JSON")
    parser.add_argument("--n-visible", type=int, default=784)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cert-id", type=str, default="generated_run")
    args = parser.parse_args()

    run = train_rbm(
        n_visible=args.n_visible,
        n_hidden=args.n_hidden,
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed,
    )

    cert = build_cert(args.cert_id, run)
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
