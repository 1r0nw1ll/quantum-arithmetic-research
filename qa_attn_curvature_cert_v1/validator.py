#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema


SCHEMA_ID = "QA_ATTN_CURVATURE_CERT"
TOL = 1e-9

FAIL_SCHEMA = "SCHEMA_INVALID"
FAIL_EPS_MISMATCH = "EPS_MISMATCH"
FAIL_H_QA_MISMATCH = "H_QA_MISMATCH"
FAIL_LOSS_HAT_MISMATCH = "LOSS_HAT_MISMATCH"
FAIL_UPDATE_RULE_MISMATCH = "UPDATE_RULE_MISMATCH"
FAIL_ATTN_GAIN_OUT_OF_RANGE = "ATTN_GAIN_OUT_OF_RANGE"
FAIL_KAPPA_MISMATCH = "KAPPA_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def recompute_h_qa_and_loss(b: float, e: float, d: float, a: float, eps: float) -> Tuple[float, float, float]:
    # Identical to family [89] substrate math, except uses e*e + d*d (no exponentiation operator).
    G = e * e + d * d
    F = b * a
    h_raw = 0.25 * ((F) / (G + eps) + (e * d) / (a + b + eps))
    h = abs(h_raw) / (1 + abs(h_raw))
    loss_hat = float(F / (G + eps))
    return float(h_raw), float(h), loss_hat


def recompute_kappa(lr: float, attn_gain: float, h_qa: float) -> float:
    return float(1 - abs(1 - lr * attn_gain * h_qa))


def gate_1_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
    except jsonschema.ValidationError as exc:
        return {
            "ok": False,
            "fail_type": FAIL_SCHEMA,
            "invariant_diff": {},
            "details": {"error": str(exc)},
        }
    return None


def gate_2_gates(cert: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    # Gate 2A (substrate): recompute h_raw, H_QA, loss_hat (matches family [89]).
    eps = float(cert["inputs"]["eps"])
    if not _close(eps, 1e-9):
        return {
            "ok": False,
            "fail_type": FAIL_EPS_MISMATCH,
            "invariant_diff": {"inputs.eps": {"claimed": eps, "expected": 1e-9}},
            "details": {},
        }, {}

    t = cert["inputs"]["tuple"]
    b = float(t["b"])
    e = float(t["e"])
    d = float(t["d"])
    a = float(t["a"])

    h_raw, h_qa, loss_hat = recompute_h_qa_and_loss(b=b, e=e, d=d, a=a, eps=eps)

    claimed = cert["claimed"]
    if not _close(float(claimed["H_QA_raw"]), h_raw):
        return {
            "ok": False,
            "fail_type": FAIL_H_QA_MISMATCH,
            "invariant_diff": {
                "claimed.H_QA_raw": {
                    "claimed": float(claimed["H_QA_raw"]),
                    "recomputed": h_raw,
                }
            },
            "details": {},
        }, {}

    if not _close(float(claimed["H_QA"]), h_qa):
        return {
            "ok": False,
            "fail_type": FAIL_H_QA_MISMATCH,
            "invariant_diff": {
                "claimed.H_QA": {
                    "claimed": float(claimed["H_QA"]),
                    "recomputed": h_qa,
                }
            },
            "details": {},
        }, {}

    if not _close(float(claimed["loss_hat"]), loss_hat):
        return {
            "ok": False,
            "fail_type": FAIL_LOSS_HAT_MISMATCH,
            "invariant_diff": {
                "claimed.loss_hat": {
                    "claimed": float(claimed["loss_hat"]),
                    "recomputed": loss_hat,
                }
            },
            "details": {},
        }, {}

    # Gate 2B (optimizer): strict attn_gain range check + update rule pin.
    lr = float(cert["optimizer"]["lr"])
    attn_gain = float(cert["optimizer"]["attn_gain"])

    if attn_gain <= 0 or attn_gain > 2:
        return {
            "ok": False,
            "fail_type": FAIL_ATTN_GAIN_OUT_OF_RANGE,
            "invariant_diff": {"optimizer.attn_gain": {"claimed": attn_gain, "expected_range": "(0,2]"}},
            "details": {},
        }, {}

    ex = cert["claimed"]["update_example"]
    p_before = float(ex["p_before"])
    grad = float(ex["grad"])
    p_after_claimed = float(ex["p_after"])

    p_after_recomputed = p_before - lr * attn_gain * grad
    if not _close(p_after_claimed, p_after_recomputed):
        return {
            "ok": False,
            "fail_type": FAIL_UPDATE_RULE_MISMATCH,
            "invariant_diff": {
                "claimed.update_example.p_after": {
                    "claimed": p_after_claimed,
                    "recomputed": p_after_recomputed,
                    "p_before": p_before,
                    "grad": grad,
                    "lr": lr,
                    "attn_gain": attn_gain,
                }
            },
            "details": {},
        }, {}

    # Gate 2C (kappa): recompute kappa and pin it to claimed.kappa.
    kappa_recomputed = recompute_kappa(lr=lr, attn_gain=attn_gain, h_qa=h_qa)
    if not _close(float(claimed["kappa"]), kappa_recomputed):
        return {
            "ok": False,
            "fail_type": FAIL_KAPPA_MISMATCH,
            "invariant_diff": {
                "claimed.kappa": {
                    "claimed": float(claimed["kappa"]),
                    "recomputed": kappa_recomputed,
                    "lr": lr,
                    "attn_gain": attn_gain,
                    "H_QA": h_qa,
                }
            },
            "details": {},
        }, {}

    return None, {"H_QA_raw": h_raw, "H_QA": h_qa, "loss_hat": loss_hat, "kappa": kappa_recomputed}


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    g1 = gate_1_schema(cert, schema)
    if g1 is not None:
        return g1

    g2, recomputed = gate_2_gates(cert)
    if g2 is not None:
        return g2

    return {
        "ok": True,
        "family": 94,
        "recomputed": recomputed,
        "cert_sha256": sha256_hex_bytes(canonical_json_bytes(cert)),
        "witnesses": [],
    }


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_default_attn.json", True, None),
        ("fail_attn_gain_mismatch.json", False, FAIL_UPDATE_RULE_MISMATCH),
        ("fail_h_qa_mismatch.json", False, FAIL_H_QA_MISMATCH),
        ("fail_seq_len_invalid.json", False, FAIL_SCHEMA),
    ]

    failed: List[str] = []
    results: List[Dict[str, Any]] = []

    for name, expected_ok, expected_fail_type in fixtures:
        cert = load_json(base / "fixtures" / name)
        out = validate_cert(cert, schema)
        got_ok = bool(out.get("ok"))
        got_fail_type = out.get("fail_type")
        ok_match = got_ok == expected_ok
        fail_match = expected_fail_type is None or got_fail_type == expected_fail_type
        if not ok_match or not fail_match:
            failed.append(name)
        results.append(
            {
                "fixture": name,
                "expected_ok": expected_ok,
                "expected_fail_type": expected_fail_type,
                "got_ok": got_ok,
                "got_fail_type": got_fail_type,
                "result": out,
            }
        )

    payload = {
        "ok": len(failed) == 0,
        "failed_fixtures": failed,
        "fixtures": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if len(failed) == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA Attention Layer Curvature Cert v1")
    parser.add_argument("--schema", default="schema.json")
    parser.add_argument("--cert")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    if not args.cert:
        parser.error("provide --cert or use --self-test")

    schema = load_json(Path(args.schema))
    cert = load_json(Path(args.cert))
    out = validate_cert(cert, schema)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
