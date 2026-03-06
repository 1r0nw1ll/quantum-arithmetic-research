#!/usr/bin/env python3
"""
Validator for QA_CONGRUENCE_LABEL_FACTOR_CERT.v1

Usage:
  python qa_congruence_label_factor_cert_v1/validator.py --demo
  python qa_congruence_label_factor_cert_v1/validator.py <cert.json>

Certificate modes:
  - status == "fail": witness of non-factorization through rho_N (must verify)
  - status == "bounded_pass": bounded adversarial search with fixed parameters and seed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import sys

# Ensure repo root is importable when running from this subdirectory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from qa_resonance_spectral_experiment import (
    Mat2,
    LABEL_FNS,
    enumerate_psl2z_ball,
    forced_collision_factor_test,
    is_hyperbolic,
    psl_normal_form,
    reduce_mod_n,
)


SCHEMA_ID = "QA_CONGRUENCE_LABEL_FACTOR_CERT.v1"


def _load(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"✗ {msg}")


def verify_fail(cert: Dict[str, Any], path: Path) -> None:
    label_name = cert["label_name"]
    modulus_n = int(cert["modulus_n"])
    _require(label_name in LABEL_FNS, f"unknown label_name: {label_name}")

    label_fn = LABEL_FNS[label_name]
    base = Mat2(*cert["base_mat"])
    collided = Mat2(*cert["collided_mat"])

    base_red = reduce_mod_n(base, modulus_n)
    col_red = reduce_mod_n(collided, modulus_n)
    _require(base_red == tuple(cert["base_reduction"]), "base_reduction mismatch")
    _require(col_red == tuple(cert["collided_reduction"]), "collided_reduction mismatch")
    _require(base_red == col_red, "reductions differ; not a forced collision")

    base_label = label_fn(base)
    col_label = label_fn(collided)
    _require(tuple(base_label) == tuple(cert["base_label"]), "base_label mismatch")
    _require(tuple(col_label) == tuple(cert["collided_label"]), "collided_label mismatch")
    _require(base_label != col_label, "certificate claims FAIL but labels match")

    # Optional constructive lineage: rebuild k from kernel_specs and verify collided = k * base in PSL normal form.
    kernel_specs = cert.get("kernel_specs")
    if kernel_specs:
        k = Mat2(1, 0, 0, 1)
        for spec in kernel_specs:
            kind = spec["kind"]
            params = tuple(spec["params"])
            if kind == "upper_unipotent":
                (m,) = params
                step = Mat2(1, modulus_n * int(m), 0, 1)
            elif kind == "lower_unipotent":
                (m,) = params
                step = Mat2(1, 0, modulus_n * int(m), 1)
            elif kind == "balanced_hyperbolic":
                (x,) = params
                x = int(x)
                step = Mat2(1 + modulus_n * x, modulus_n * x, -modulus_n * x, 1 - modulus_n * x)
            else:
                raise SystemExit(f"✗ unknown kernel spec kind: {kind}")
            k = psl_normal_form(k @ step)

        recomputed = psl_normal_form(k @ base)
        _require(recomputed == psl_normal_form(collided), "kernel_specs do not reproduce collided = k*base")

    print(f"✓ OK: {path} verifies FAIL witness (label does not factor through rho_N)")


def verify_bounded_pass(cert: Dict[str, Any], path: Path) -> None:
    label_name = cert["label_name"]
    modulus_n = int(cert["modulus_n"])
    _require(label_name in LABEL_FNS, f"unknown label_name: {label_name}")

    max_word_len = int(cert["max_word_len"])
    include_t_inv = bool(cert["include_t_inv"])
    limit_hyperbolic_rows = int(cert["limit_hyperbolic_rows"])
    seed = int(cert["seed"])

    kernel_collision_trials = int(cert["kernel_collision_trials"])
    kernel_steps = int(cert["kernel_steps"])
    kernel_x_bound = int(cert["kernel_x_bound"])

    elements = enumerate_psl2z_ball(max_word_len=max_word_len, include_t_inv=include_t_inv)
    hyperbolic = [(m, w) for (m, w) in elements.items() if is_hyperbolic(m)]
    hyperbolic.sort(key=lambda mw: (abs(mw[0].tr()), len(mw[1]), mw[1]))
    hyperbolic = hyperbolic[:limit_hyperbolic_rows]

    # Use the same deterministic PRNG behavior as the experiment harness.
    import random

    rng = random.Random(seed)
    report = forced_collision_factor_test(
        items=hyperbolic,
        modulus_n=modulus_n,
        label_name=label_name,
        label_fn=LABEL_FNS[label_name],
        trials=kernel_collision_trials,
        kernel_steps=kernel_steps,
        kernel_x_bound=kernel_x_bound,
        rng=rng,
    )
    _require(report["status"] == "pass", f"bounded_pass disproved: got status={report['status']}")

    print(f"✓ OK: {path} verifies bounded-pass (no counterexample found under stated bounds)")


def verify(cert_path: Path) -> None:
    cert = _load(cert_path)
    _require(cert.get("schema_id") == SCHEMA_ID, f"schema_id must be {SCHEMA_ID}")

    status = cert.get("status")
    if status == "fail":
        verify_fail(cert, cert_path)
        return
    if status == "bounded_pass":
        verify_bounded_pass(cert, cert_path)
        return
    raise SystemExit(f"✗ unsupported status: {status!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cert", nargs="?", default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        base = Path(__file__).resolve().parent / "examples"
        verify(base / "fail_cf_period_sum.json")
        verify(base / "bounded_pass_trace2_mod72.json")
        verify(base / "bounded_pass_disc_mod72.json")
        return 0

    if not args.cert:
        raise SystemExit("usage: validator.py <cert.json> OR --demo")

    verify(Path(args.cert))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
