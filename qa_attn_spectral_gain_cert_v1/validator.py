#!/usr/bin/env python3
"""QA Attention Spectral Gain Cert v1 — family [99]

Derives the curvature gain from sigma_max(Q K^T / sqrt(d_k)) — the natural
Lipschitz constant of the scaled-dot-product attention score map — via power
iteration. This makes gain a native structural object determined by the
attention weight geometry, not a free scalar witness.

Gates:
  A — Recompute H_QA from substrate; verify claimed.H_QA.
  B — Compute attention score matrix A = Q K^T / sqrt(d_k);
      derive sigma_max(A) via power iteration on A^T A;
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

TOL        = 1e-9
EPS        = 1e-12
POWER_ITER = 300

FAIL_SCHEMA         = "SCHEMA_INVALID"
FAIL_H_QA_MISMATCH  = "H_QA_MISMATCH"
FAIL_SIGMA_MISMATCH = "SIGMA_MAX_MISMATCH"
FAIL_UPDATE_RULE    = "UPDATE_RULE_MISMATCH"
FAIL_KAPPA_MISMATCH = "KAPPA_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# ── QA substrate ─────────────────────────────────────────────────────────────

def recompute_h_qa(sub: Dict[str, Any]) -> float:
    b = float(sub["b"]); e = float(sub["e"])
    d = float(sub["d"]); a = float(sub["a"])
    G    = e * e + d * d          # use * not **
    F    = b * a
    h_raw = 0.25 * (F / (G + EPS) + (e * d) / (a + b + EPS))
    return abs(h_raw) / (1.0 + abs(h_raw))


# ── Linear algebra (pure Python, no numpy required) ──────────────────────────

def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, k, m = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(m)]
            for i in range(n)]


def _transpose(M: List[List[float]]) -> List[List[float]]:
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def _sigma_max(M: List[List[float]], iters: int = POWER_ITER) -> float:
    """sigma_max(M) = sqrt(lambda_max(M^T M)) via power iteration."""
    Mt  = _transpose(M)
    MtM = _matmul(Mt, M)
    n   = len(MtM)
    v   = [1.0 / math.sqrt(n)] * n
    lam = 0.0
    for _ in range(iters):
        v2  = [sum(MtM[i][j] * v[j] for j in range(n)) for i in range(n)]
        lam = math.sqrt(sum(x * x for x in v2))
        if lam < 1e-15:
            break
        v = [x / lam for x in v2]
    return math.sqrt(lam)


def compute_attn_sigma_max(attn: Dict[str, Any]) -> float:
    """sigma_max(Q K^T / sqrt(d_k))."""
    Q   = [[float(x) for x in row] for row in attn["Q"]]
    K   = [[float(x) for x in row] for row in attn["K"]]
    d_k = float(attn["d_k"])
    Kt  = _transpose(K)
    QKt = _matmul(Q, Kt)
    scale = 1.0 / math.sqrt(d_k)
    A_score = [[v * scale for v in row] for row in QKt]
    return _sigma_max(A_score)


# ── Gates ────────────────────────────────────────────────────────────────────

def gate_a_h_qa(cert: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
    h_qa    = recompute_h_qa(cert["substrate"])
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
    sigma        = compute_attn_sigma_max(cert["attention"])
    claimed_sigma = float(cert["claimed"]["sigma_max"])

    if not _close(claimed_sigma, sigma, tol=1e-6):
        return ({
            "ok": False, "fail_type": FAIL_SIGMA_MISMATCH,
            "invariant_diff": {
                "claimed.sigma_max": {"claimed": claimed_sigma, "recomputed": sigma}
            },
            "details": {}, "witnesses": [],
        }, 0.0)

    # Pin update rule
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

    attn = cert["attention"]
    return {
        "ok": True,
        "family": 99,
        "cert_sha256": sha256_hex(canonical_json_bytes(cert)),
        "recomputed": {
            "H_QA":      h_qa,
            "sigma_max": sigma,
            "kappa":     kappa,
            "Q_shape": [len(attn["Q"]), len(attn["Q"][0])],
            "K_shape": [len(attn["K"]), len(attn["K"][0])],
            "d_k":     attn["d_k"],
        },
        "witnesses": [],
    }


# ── Self-test ────────────────────────────────────────────────────────────────

def run_self_test() -> int:
    base   = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_attn_score.json",    True,  None),
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
    parser = argparse.ArgumentParser(description="Validate QA Attention Spectral Gain Cert v1")
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
