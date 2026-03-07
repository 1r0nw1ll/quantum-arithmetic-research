#!/usr/bin/env python3
"""QA Orbit Curvature Cert v1 — family [97]

Gates:
  A — Enumerate full orbit from (orbit_start.b, orbit_start.e) under modulus;
      verify claimed.orbit_length matches.
  B — Compute H_QA at every orbit state; compute κ_t = 1 - |1 - lr·gain·H_QA|;
      find κ_min = min over all orbit states.
  C — Verify claimed.kappa_min matches recomputed κ_min.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

TOL = 1e-9
EPS = 1e-12

FAIL_SCHEMA              = "SCHEMA_INVALID"
FAIL_ORBIT_LENGTH        = "ORBIT_LENGTH_MISMATCH"
FAIL_KAPPA_MIN_MISMATCH  = "KAPPA_MIN_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# ── QA core maths ────────────────────────────────────────────────────────────

def qa_step(b: int, e: int, mod: int) -> Tuple[int, int]:
    """One QA update: (b,e) → (d,a) where d=b+e mod*, a=b+2e mod*."""
    d = (b + e) % mod or mod
    a = (b + 2 * e) % mod or mod
    return d, a


def compute_h_qa(b: int, e: int, d: int, a: int) -> float:
    G = e * e + d * d          # use * not **
    F = b * a
    h_raw = 0.25 * (F / (G + EPS) + (e * d) / (a + b + EPS))
    return abs(h_raw) / (1.0 + abs(h_raw))


def enumerate_orbit(b0: int, e0: int, mod: int) -> List[Tuple[int, int, int, int]]:
    """Return list of (b,e,d,a) tuples for the full orbit from (b0,e0)."""
    orbit: List[Tuple[int, int, int, int]] = []
    b, e = b0, e0
    seen: set = set()
    while (b, e) not in seen:
        seen.add((b, e))
        d, a = qa_step(b, e, mod)
        orbit.append((b, e, d, a))
        b, e = d, a
    return orbit


# ── Gates ────────────────────────────────────────────────────────────────────

def gate_a_orbit_length(
    cert: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Tuple[int, int, int, int]]]:
    b0 = int(cert["orbit_start"]["b"])
    e0 = int(cert["orbit_start"]["e"])
    mod = int(cert["modulus"])
    orbit = enumerate_orbit(b0, e0, mod)
    claimed_len = int(cert["claimed"]["orbit_length"])
    if len(orbit) != claimed_len:
        return (
            {
                "ok": False,
                "fail_type": FAIL_ORBIT_LENGTH,
                "invariant_diff": {
                    "claimed.orbit_length": {
                        "claimed": claimed_len,
                        "recomputed": len(orbit),
                    }
                },
                "details": {"b0": b0, "e0": e0, "modulus": mod},
                "witnesses": [],
            },
            [],
        )
    return None, orbit


def gate_b_h_qa_series(
    orbit: List[Tuple[int, int, int, int]],
) -> List[float]:
    return [compute_h_qa(*state) for state in orbit]


def gate_c_kappa_min(
    cert: Dict[str, Any],
    h_series: List[float],
) -> Tuple[Optional[Dict[str, Any]], float]:
    lr   = float(cert["optimizer"]["lr"])
    gain = float(cert["optimizer"]["gain"])
    ks   = [1.0 - abs(1.0 - lr * gain * h) for h in h_series]
    kmin = min(ks)
    claimed_kmin = float(cert["claimed"]["kappa_min"])
    if not _close(claimed_kmin, kmin):
        return (
            {
                "ok": False,
                "fail_type": FAIL_KAPPA_MIN_MISMATCH,
                "invariant_diff": {
                    "claimed.kappa_min": {
                        "claimed": claimed_kmin,
                        "recomputed": kmin,
                    }
                },
                "details": {"lr": lr, "gain": gain, "orbit_length": len(h_series)},
                "witnesses": [],
            },
            0.0,
        )
    return None, kmin


# ── Top-level ────────────────────────────────────────────────────────────────

def gate_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
    err = gate_schema(cert, schema)
    if err is not None:
        return err

    err, orbit = gate_a_orbit_length(cert)
    if err is not None:
        return err

    h_series = gate_b_h_qa_series(orbit)

    err, kmin = gate_c_kappa_min(cert, h_series)
    if err is not None:
        return err

    return {
        "ok": True,
        "family": 97,
        "cert_sha256": sha256_hex(canonical_json_bytes(cert)),
        "recomputed": {
            "orbit_length": len(orbit),
            "kappa_min": kmin,
            "h_qa_min": min(h_series),
            "h_qa_max": max(h_series),
        },
        "witnesses": [],
    }


# ── Self-test ────────────────────────────────────────────────────────────────

def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_orbit_12.json",           True,  None),
        ("fail_orbit_length_mismatch.json", False, FAIL_ORBIT_LENGTH),
        ("fail_kappa_min_mismatch.json", False, FAIL_KAPPA_MIN_MISMATCH),
        ("fail_schema.json",             False, FAIL_SCHEMA),
    ]

    failed: List[str] = []
    results: List[Dict[str, Any]] = []

    for name, expected_ok, expected_fail_type in fixtures:
        cert = load_json(base / "fixtures" / name)
        out  = validate_cert(cert, schema)
        got_ok   = bool(out.get("ok"))
        got_fail = out.get("fail_type")
        ok_match   = got_ok == expected_ok
        fail_match = expected_fail_type is None or got_fail == expected_fail_type
        if not ok_match or not fail_match:
            failed.append(name)
        results.append({
            "fixture":            name,
            "expected_ok":        expected_ok,
            "expected_fail_type": expected_fail_type,
            "got_ok":             got_ok,
            "got_fail_type":      got_fail,
        })

    payload = {"ok": len(failed) == 0, "failed_fixtures": failed, "fixtures": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA Orbit Curvature Cert v1")
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
