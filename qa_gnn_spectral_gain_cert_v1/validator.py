#!/usr/bin/env python3
"""QA GNN Spectral Gain Cert v1 — family [98]

Derives the curvature gain from the spectral norm sigma_max(W) of the GNN
weight matrix W via power iteration on W^T W. This makes gain a native
structural object (determined by the learned weights) rather than a free
witness.

Gates:
  A — Recompute H_QA from substrate; verify claimed.H_QA.
  B — Derive sigma_max = spectral_norm(W) via power iteration;
      verify claimed.sigma_max; pin update rule:
        p_after = p_before - lr * sigma_max * H_QA * grad
  C — Recompute kappa = 1 - |1 - lr * sigma_max * H_QA|;
      verify claimed.kappa.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

TOL  = 1e-9
EPS  = 1e-12
POWER_ITER = 300

FAIL_SCHEMA          = "SCHEMA_INVALID"
FAIL_H_QA_MISMATCH   = "H_QA_MISMATCH"
FAIL_SIGMA_MISMATCH  = "SIGMA_MAX_MISMATCH"
FAIL_UPDATE_RULE     = "UPDATE_RULE_MISMATCH"
FAIL_KAPPA_MISMATCH  = "KAPPA_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# ── QA substrate maths ───────────────────────────────────────────────────────

def recompute_h_qa(sub: Dict[str, Any]) -> float:
    b = float(sub["b"]); e = float(sub["e"])
    d = float(sub["d"]); a = float(sub["a"])
    G = e * e + d * d          # use * not **
    F = b * a
    h_raw = 0.25 * (F / (G + EPS) + (e * d) / (a + b + EPS))
    return abs(h_raw) / (1.0 + abs(h_raw))


# ── Spectral norm via power iteration on W^T W ───────────────────────────────

def _matvec(M: List[List[float]], v: List[float]) -> List[float]:
    """M @ v for 2D list M and vector v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _transpose(W: List[List[float]]) -> List[List[float]]:
    rows, cols = len(W), len(W[0])
    return [[W[r][c] for r in range(rows)] for c in range(cols)]


def spectral_norm(W: List[List[float]], iters: int = POWER_ITER) -> float:
    """sigma_max(W) = sqrt(lambda_max(W^T W)) via power iteration."""
    Wt  = _transpose(W)
    WtW = [[sum(Wt[i][k] * W[k][j] for k in range(len(W)))
            for j in range(len(W[0]))]
           for i in range(len(W[0]))]
    n   = len(WtW)
    v   = [1.0 / math.sqrt(n)] * n
    lam = 0.0
    for _ in range(iters):
        v2  = _matvec(WtW, v)
        lam = math.sqrt(sum(x * x for x in v2))
        if lam < 1e-15:
            break
        v = [x / lam for x in v2]
    return math.sqrt(lam)


# ── Gates ────────────────────────────────────────────────────────────────────

def gate_a_h_qa(cert: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
    h_qa = recompute_h_qa(cert["substrate"])
    claimed = float(cert["claimed"]["H_QA"])
    if not _close(claimed, h_qa):
        return ({
            "ok": False, "fail_type": FAIL_H_QA_MISMATCH,
            "invariant_diff": {"claimed.H_QA": {"claimed": claimed, "recomputed": h_qa}},
            "details": {}, "witnesses": [],
        }, 0.0)
    return None, h_qa


def gate_b_sigma_and_update(
    cert: Dict[str, Any], h_qa: float
) -> Tuple[Optional[Dict[str, Any]], float]:
    W = [[float(x) for x in row] for row in cert["weight_matrix"]]
    sigma = spectral_norm(W)
    claimed_sigma = float(cert["claimed"]["sigma_max"])

    if not _close(claimed_sigma, sigma, tol=1e-6):
        return ({
            "ok": False, "fail_type": FAIL_SIGMA_MISMATCH,
            "invariant_diff": {"claimed.sigma_max": {"claimed": claimed_sigma, "recomputed": sigma}},
            "details": {}, "witnesses": [],
        }, 0.0)

    # Pin update rule: p_after = p_before - lr * sigma_max * H_QA * grad
    ex   = cert["claimed"]["update_example"]
    lr   = float(ex["lr"])
    g    = float(ex["grad"])
    pb   = float(ex["p_before"])
    pa_c = float(ex["p_after"])
    pa_r = pb - lr * sigma * h_qa * g

    if not _close(pa_c, pa_r):
        return ({
            "ok": False, "fail_type": FAIL_UPDATE_RULE,
            "invariant_diff": {
                "claimed.update_example.p_after": {
                    "claimed": pa_c, "recomputed": pa_r,
                    "p_before": pb, "grad": g, "lr": lr,
                    "sigma_max": sigma, "H_QA": h_qa,
                }
            },
            "details": {}, "witnesses": [],
        }, 0.0)

    return None, sigma


def gate_c_kappa(
    cert: Dict[str, Any], sigma: float, h_qa: float
) -> Tuple[Optional[Dict[str, Any]], float]:
    lr    = float(cert["optimizer"]["lr"])
    kappa = 1.0 - abs(1.0 - lr * sigma * h_qa)
    claimed = float(cert["claimed"]["kappa"])
    if not _close(claimed, kappa):
        return ({
            "ok": False, "fail_type": FAIL_KAPPA_MISMATCH,
            "invariant_diff": {"claimed.kappa": {"claimed": claimed, "recomputed": kappa}},
            "details": {"lr": lr, "sigma_max": sigma, "H_QA": h_qa},
            "witnesses": [],
        }, 0.0)
    return None, kappa


# ── Top-level ────────────────────────────────────────────────────────────────

def gate_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
    except jsonschema.ValidationError as exc:
        return {
            "ok": False, "fail_type": FAIL_SCHEMA,
            "invariant_diff": {}, "details": {"error": str(exc)}, "witnesses": [],
        }
    return None


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    err = gate_schema(cert, schema)
    if err is not None:
        return err

    err, h_qa = gate_a_h_qa(cert)
    if err is not None:
        return err

    err, sigma = gate_b_sigma_and_update(cert, h_qa)
    if err is not None:
        return err

    err, kappa = gate_c_kappa(cert, sigma, h_qa)
    if err is not None:
        return err

    W    = cert["weight_matrix"]
    rows = len(W); cols = len(W[0])
    return {
        "ok": True,
        "family": 98,
        "cert_sha256": sha256_hex(canonical_json_bytes(cert)),
        "recomputed": {
            "H_QA":      h_qa,
            "sigma_max": sigma,
            "kappa":     kappa,
            "weight_shape": [rows, cols],
        },
        "witnesses": [],
    }


# ── Self-test ────────────────────────────────────────────────────────────────

def run_self_test() -> int:
    base   = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_gnn_weight.json",    True,  None),
        ("fail_sigma_mismatch.json", False, FAIL_SIGMA_MISMATCH),
        ("fail_h_qa_mismatch.json", False, FAIL_H_QA_MISMATCH),
        ("fail_schema.json",        False, FAIL_SCHEMA),
    ]

    failed: List[str] = []
    results: List[Dict[str, Any]] = []

    for name, expected_ok, expected_fail in fixtures:
        cert    = load_json(base / "fixtures" / name)
        out     = validate_cert(cert, schema)
        got_ok  = bool(out.get("ok"))
        got_fail = out.get("fail_type")
        ok_match   = got_ok == expected_ok
        fail_match = expected_fail is None or got_fail == expected_fail
        if not ok_match or not fail_match:
            failed.append(name)
        results.append({
            "fixture":            name,
            "expected_ok":        expected_ok,
            "expected_fail_type": expected_fail,
            "got_ok":             got_ok,
            "got_fail_type":      got_fail,
        })

    payload = {"ok": len(failed) == 0, "failed_fixtures": failed, "fixtures": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA GNN Spectral Gain Cert v1")
    parser.add_argument("--schema", default="schema.json")
    parser.add_argument("--cert")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    if not args.cert:
        parser.error("provide --cert or --self-test")

    schema = load_json(Path(args.schema))
    cert   = load_json(Path(args.cert))
    print(json.dumps(validate_cert(cert, schema), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
