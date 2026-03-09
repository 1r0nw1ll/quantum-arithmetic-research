#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

TOL = 1e-9

FAIL_SCHEMA = "SCHEMA_INVALID"
FAIL_H_QA_MISMATCH = "H_QA_MISMATCH"
FAIL_UPDATE_RULE_MISMATCH = "UPDATE_RULE_MISMATCH"
FAIL_KAPPA_MISMATCH = "KAPPA_MISMATCH"
FAIL_GAIN_DERIVATION_MISMATCH = "GAIN_DERIVATION_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def recompute_h_qa(substrate: Dict[str, Any]) -> Tuple[float, float]:
    b = float(substrate["b"])
    e = float(substrate["e"])
    d = float(substrate["d"])
    a = float(substrate["a"])

    eps = 1e-12
    G = e * e + d * d  # use * not **
    F = b * a
    h_raw = 0.25 * (F / (G + eps) + (e * d) / (a + b + eps))
    h_qa = abs(h_raw) / (1 + abs(h_raw))
    return float(h_raw), float(h_qa)


def gate_d_gain_derivation(cert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    opt = cert["optimizer"]
    sym_gain   = float(opt["sym_gain"])
    beam_width  = int(opt["beam_width"])
    search_depth = int(opt["search_depth"])
    rule_count  = int(opt["rule_count"])
    derived_gain = min(float(rule_count) / (float(beam_width) * float(search_depth)), 2.0)
    if not _close(sym_gain, derived_gain):
        return {
            "ok": False,
            "fail_type": FAIL_GAIN_DERIVATION_MISMATCH,
            "invariant_diff": {
                "optimizer.sym_gain": {
                    "claimed":  sym_gain,
                    "derived":  derived_gain,
                    "formula":  "min(rule_count / (beam_width * search_depth), 2.0)",
                    "beam_width":   beam_width,
                    "search_depth": search_depth,
                    "rule_count":   rule_count,
                }
            },
            "details": {},
            "witnesses": [],
        }
    return None


def gate_a_substrate(cert: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    h_raw, h_qa = recompute_h_qa(cert["substrate"])
    claimed_h = float(cert["claimed"]["H_QA"])
    if not _close(claimed_h, h_qa):
        return {
            "ok": False,
            "fail_type": FAIL_H_QA_MISMATCH,
            "invariant_diff": {"claimed.H_QA": {"claimed": claimed_h, "recomputed": h_qa}},
            "details": {},
            "witnesses": [],
        }, {}
    return None, {"H_raw": h_raw, "H_QA": h_qa}


def gate_b_update_rule_and_gain(cert: Dict[str, Any], h_qa: float) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    sym_gain = float(cert["optimizer"]["sym_gain"])
    if not (0 < sym_gain <= 2.0):
        return {
            "ok": False,
            "fail_type": FAIL_UPDATE_RULE_MISMATCH,
            "invariant_diff": {"optimizer.sym_gain": {"claimed": sym_gain, "expected_range": "(0,2]"}},
            "details": {},
            "witnesses": [],
        }, {}

    ex = cert["claimed"]["update_example"]
    lr = float(ex["lr"])
    g = float(ex["grad"])
    p_before = float(ex["p_before"])
    p_after_claimed = float(ex["p_after"])

    p_after_recomputed = p_before - lr * sym_gain * h_qa * g
    if not _close(p_after_claimed, p_after_recomputed):
        return {
            "ok": False,
            "fail_type": FAIL_UPDATE_RULE_MISMATCH,
            "invariant_diff": {
                "claimed.update_example.p_after": {
                    "claimed": p_after_claimed,
                    "recomputed": p_after_recomputed,
                    "p_before": p_before,
                    "grad": g,
                    "lr": lr,
                    "sym_gain": sym_gain,
                    "H_QA": h_qa,
                }
            },
            "details": {},
            "witnesses": [],
        }, {}

    return None, {"p_after": float(p_after_recomputed), "lr": lr, "sym_gain": sym_gain}


def gate_c_kappa(cert: Dict[str, Any], lr: float, sym_gain: float, h_qa: float) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    kappa_recomputed = float(1 - abs(1 - lr * sym_gain * h_qa))
    claimed_kappa = float(cert["claimed"]["kappa"])
    if not _close(claimed_kappa, kappa_recomputed):
        return {
            "ok": False,
            "fail_type": FAIL_KAPPA_MISMATCH,
            "invariant_diff": {"claimed.kappa": {"claimed": claimed_kappa, "recomputed": kappa_recomputed}},
            "details": {"lr": lr, "sym_gain": sym_gain, "H_QA": h_qa},
            "witnesses": [],
        }, {}
    return None, {"kappa": kappa_recomputed}


def gate_1_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
    except jsonschema.ValidationError as exc:
        return {
            "ok": False,
            "fail_type": FAIL_SCHEMA,
            "invariant_diff": {},
            "details": {"error": str(exc)},
            "witnesses": [],
        }
    return None


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    g1 = gate_1_schema(cert, schema)
    if g1 is not None:
        return g1

    gd = gate_d_gain_derivation(cert)
    if gd is not None:
        return gd

    ga, recomputed_a = gate_a_substrate(cert)
    if ga is not None:
        return ga

    h_qa = float(recomputed_a["H_QA"])
    gb, recomputed_b = gate_b_update_rule_and_gain(cert, h_qa=h_qa)
    if gb is not None:
        return gb

    gc, recomputed_c = gate_c_kappa(cert, lr=float(recomputed_b["lr"]), sym_gain=float(recomputed_b["sym_gain"]), h_qa=h_qa)
    if gc is not None:
        return gc

    recomputed = {}
    recomputed.update(recomputed_a)
    recomputed.update({"p_after": recomputed_b["p_after"]})
    recomputed.update(recomputed_c)

    return {
        "ok": True,
        "family": 96,
        "cert_sha256": sha256_hex_bytes(canonical_json_bytes(cert)),
        "recomputed": recomputed,
        "witnesses": [],
    }


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_default_sym.json", True, None),
        ("fail_sym_gain_mismatch.json", False, FAIL_GAIN_DERIVATION_MISMATCH),
        ("fail_h_qa_mismatch.json", False, FAIL_H_QA_MISMATCH),
        ("fail_beam_width_invalid.json", False, FAIL_SCHEMA),
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

    payload = {"ok": len(failed) == 0, "failed_fixtures": failed, "fixtures": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if len(failed) == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA Symbolic Search Curvature Cert v1")
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

