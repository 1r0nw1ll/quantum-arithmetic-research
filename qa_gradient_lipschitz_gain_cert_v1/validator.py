#!/usr/bin/env python3
"""QA Gradient Lipschitz Gain Cert v1 — family [101]

Derives the curvature gain from the L2 norm of the gradient vector, capped
at 2.0. This is the natural local Lipschitz constant of the update step:
the gradient magnitude determines how much the parameter moves per unit of
curvature, not a free scalar witness.

  gain = min(||grad_vector||_2, 2.0)

Together with [98] (GNN weight spectral norm) and [99] (attention score
spectral norm), this establishes derived gain across three architecture
classes: graph/message-passing, sequence/attention, and gradient descent.

Gates:
  A — Recompute H_QA from substrate; verify claimed.H_QA.
  B — Recompute ||grad_vector||_2; verify claimed.grad_norm and
      claimed.gain = min(grad_norm, 2.0); pin update rule:
        p_after = p_before - lr * gain * H_QA * grad
  C — Recompute kappa = 1 - |1 - lr * gain * H_QA|; verify claimed.kappa.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

TOL = 1e-9
EPS = 1e-12

FAIL_SCHEMA         = "SCHEMA_INVALID"
FAIL_H_QA_MISMATCH  = "H_QA_MISMATCH"
FAIL_GRAD_NORM      = "GRAD_NORM_MISMATCH"
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


# ── Gradient norm + derived gain ─────────────────────────────────────────────

def grad_l2_and_gain(grad_vector: List[float]) -> Tuple[float, float]:
    """||grad_vector||_2 and gain = min(||·||, 2.0)."""
    norm = math.sqrt(sum(x * x for x in grad_vector))
    gain = min(norm, 2.0)
    return norm, gain


# ── Gates ─────────────────────────────────────────────────────────────────────

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


def gate_b_grad_and_update(
    cert: Dict[str, Any], h_qa: float
) -> Tuple[Optional[Dict[str, Any]], float, float]:
    gv            = [float(x) for x in cert["grad_vector"]]
    norm, gain    = grad_l2_and_gain(gv)
    claimed_norm  = float(cert["claimed"]["grad_norm"])
    claimed_gain  = float(cert["claimed"]["gain"])

    if not _close(claimed_norm, norm):
        return ({
            "ok": False, "fail_type": FAIL_GRAD_NORM,
            "invariant_diff": {"claimed.grad_norm": {"claimed": claimed_norm, "recomputed": norm}},
            "details": {}, "witnesses": [],
        }, 0.0, 0.0)

    if not _close(claimed_gain, gain):
        return ({
            "ok": False, "fail_type": FAIL_GRAD_NORM,
            "invariant_diff": {"claimed.gain": {"claimed": claimed_gain, "recomputed": gain,
                                                 "note": "gain = min(grad_norm, 2.0)"}},
            "details": {}, "witnesses": [],
        }, 0.0, 0.0)

    # Pin update rule: p_after = p_before - lr * gain * H_QA * grad
    ex   = cert["claimed"]["update_example"]
    lr   = float(ex["lr"])
    g    = float(ex["grad"])
    pb   = float(ex["p_before"])
    pa_c = float(ex["p_after"])
    pa_r = pb - lr * gain * h_qa * g

    if not _close(pa_c, pa_r):
        return ({
            "ok": False, "fail_type": FAIL_UPDATE_RULE,
            "invariant_diff": {
                "claimed.update_example.p_after": {
                    "claimed": pa_c, "recomputed": pa_r,
                    "p_before": pb, "grad": g, "lr": lr, "gain": gain, "H_QA": h_qa,
                }
            },
            "details": {}, "witnesses": [],
        }, 0.0, 0.0)

    return None, norm, gain


def gate_c_kappa(
    cert: Dict[str, Any], gain: float, h_qa: float
) -> Tuple[Optional[Dict[str, Any]], float]:
    lr    = float(cert["optimizer"]["lr"])
    kappa = 1.0 - abs(1.0 - lr * gain * h_qa)
    claimed = float(cert["claimed"]["kappa"])
    if not _close(claimed, kappa):
        return ({
            "ok": False, "fail_type": FAIL_KAPPA_MISMATCH,
            "invariant_diff": {"claimed.kappa": {"claimed": claimed, "recomputed": kappa}},
            "details": {"lr": lr, "gain": gain, "H_QA": h_qa},
            "witnesses": [],
        }, 0.0)
    return None, kappa


# ── Top-level ─────────────────────────────────────────────────────────────────

def gate_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
    except jsonschema.ValidationError as exc:
        return {"ok": False, "fail_type": FAIL_SCHEMA,
                "invariant_diff": {}, "details": {"error": str(exc)}, "witnesses": []}
    return None


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    err = gate_schema(cert, schema)
    if err is not None:
        return err

    err, h_qa = gate_a_h_qa(cert)
    if err is not None:
        return err

    err, norm, gain = gate_b_grad_and_update(cert, h_qa)
    if err is not None:
        return err

    err, kappa = gate_c_kappa(cert, gain, h_qa)
    if err is not None:
        return err

    return {
        "ok": True,
        "family": 101,
        "cert_sha256": sha256_hex(canonical_json_bytes(cert)),
        "recomputed": {
            "H_QA":      h_qa,
            "grad_norm": norm,
            "gain":      gain,
            "kappa":     kappa,
            "grad_dim":  len(cert["grad_vector"]),
        },
        "witnesses": [],
    }


# ── Self-test ──────────────────────────────────────────────────────────────────

def run_self_test() -> int:
    base   = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_grad_l2.json",           True,  None),
        ("fail_grad_norm_mismatch.json", False, FAIL_GRAD_NORM),
        ("fail_h_qa_mismatch.json",     False, FAIL_H_QA_MISMATCH),
        ("fail_schema.json",            False, FAIL_SCHEMA),
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
            "fixture": name, "expected_ok": expected_ok,
            "expected_fail_type": expected_fail,
            "got_ok": got_ok, "got_fail_type": got_fail,
        })

    payload = {"ok": len(failed) == 0, "failed_fixtures": failed, "fixtures": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA Gradient Lipschitz Gain Cert v1")
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
