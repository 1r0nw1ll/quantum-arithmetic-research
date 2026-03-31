#!/usr/bin/env python3
"""QA Lojasiewicz Orbit Descent Cert v1 — family [102]

Certifies per-orbit-window phi-contraction for smooth, Lojasiewicz-conditioned
losses under QA cosmos orbit update. The proved theorem (paper §8.3, B2a):

  If L is beta-smooth, satisfies the Lojasiewicz condition
      ||nabla L(w)||^2 >= 2*mu * V^alpha,  alpha in (0,1), mu > 0,
  along a QA orbit of length L with no critical points (H-crit), then:

      phi_{t+L} <= phi_t - (1-alpha) * C(O)

  where phi_t = V_t^{1-alpha} and C(O) = sum_t 2*mu*eta_eff_t*(1 - beta/2*eta_eff_t).

Gates:
  Schema  — Validate against QA_LOJASIEWICZ_ORBIT_CERT.v1 schema.
  A       — Verify orbit feasibility: eta_eff_t in (0, 2/beta) for all t,
             ensuring c_t = 2*mu*eta_eff_t*(1 - beta/2*eta_eff_t) > 0.
  2D      — Recompute C(O) from orbit.eta_eff, orbit.mu, orbit.beta;
             verify claimed.C_O matches to within tolerance.
             FAIL_TYPE: CO_MISMATCH
  B       — Verify h_crit_witnessed = true (H-crit is explicit hypothesis).
             FAIL_TYPE: HCRIT_NOT_WITNESSED
  C       — Verify phi_tL_bound <= phi_t - (1-alpha)*C(O).
             FAIL_TYPE: PHI_BOUND_INVALID
  D       — Verify convergence_orbits_bound = ceil(phi_t / ((1-alpha)*C(O))).
             FAIL_TYPE: ORBITS_BOUND_MISMATCH
"""
from __future__ import annotations

QA_COMPLIANCE = "observer=cert_validator, state_alphabet=integer_fixtures"

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

TOL = 1e-9

FAIL_SCHEMA          = "SCHEMA_INVALID"
FAIL_ORBIT_INFEAS    = "ORBIT_INFEASIBLE"
FAIL_CO_MISMATCH     = "CO_MISMATCH"
FAIL_HCRIT           = "HCRIT_NOT_WITNESSED"
FAIL_PHI_BOUND       = "PHI_BOUND_INVALID"
FAIL_ORBITS_BOUND    = "ORBITS_BOUND_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# ── C(O) recomputation ────────────────────────────────────────────────────────

def recompute_co(eta_eff: List[float], mu: float, beta: float) -> float:
    """C(O) = sum_t 2*mu*eta_eff_t*(1 - beta/2*eta_eff_t)."""
    total = 0.0
    for eta in eta_eff:
        total += 2.0 * mu * eta * (1.0 - beta / 2.0 * eta)
    return total


def orbit_infeasible_steps(eta_eff: List[float], beta: float) -> List[int]:
    """Return indices where eta_eff_t is not in (0, 2/beta)."""
    upper = 2.0 / beta
    return [i for i, eta in enumerate(eta_eff) if eta <= 0.0 or eta >= upper]


# ── Gates ──────────────────────────────────────────────────────────────────────

def gate_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
    except jsonschema.ValidationError as exc:
        return {"ok": False, "fail_type": FAIL_SCHEMA,
                "invariant_diff": {}, "details": {"error": str(exc)}, "witnesses": []}
    return None


def gate_a_orbit_feasibility(cert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eta_eff = [float(x) for x in cert["orbit"]["eta_eff"]]
    beta    = float(cert["orbit"]["beta"])
    bad     = orbit_infeasible_steps(eta_eff, beta)
    if bad:
        return {
            "ok": False, "fail_type": FAIL_ORBIT_INFEAS,
            "invariant_diff": {"orbit.eta_eff": {
                "bad_indices": bad,
                "requirement": f"each eta_eff_t in (0, 2/beta={2.0/beta:.6g})",
            }},
            "details": {}, "witnesses": [],
        }
    return None


def gate_2d_co(cert: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
    """Gate 2D: recompute C(O) and verify claimed.C_O."""
    eta_eff  = [float(x) for x in cert["orbit"]["eta_eff"]]
    mu       = float(cert["orbit"]["mu"])
    beta     = float(cert["orbit"]["beta"])
    co       = recompute_co(eta_eff, mu, beta)
    claimed  = float(cert["claimed"]["C_O"])
    if not _close(claimed, co):
        return ({
            "ok": False, "fail_type": FAIL_CO_MISMATCH,
            "invariant_diff": {"claimed.C_O": {
                "claimed": claimed, "recomputed": co,
                "formula": "sum_t 2*mu*eta_eff_t*(1 - beta/2*eta_eff_t)",
            }},
            "details": {"orbit_length": len(eta_eff), "mu": mu, "beta": beta},
            "witnesses": [],
        }, 0.0)
    return None, co


def gate_b_hcrit(cert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if cert["h_crit_witnessed"] is not True:
        return {
            "ok": False, "fail_type": FAIL_HCRIT,
            "invariant_diff": {"h_crit_witnessed": {
                "got": cert["h_crit_witnessed"],
                "required": True,
                "note": "H-crit (nabla L(w_t) != 0 for all t in orbit) must be witnessed true",
            }},
            "details": {}, "witnesses": [],
        }
    return None


def gate_c_phi_bound(cert: Dict[str, Any], co: float) -> Optional[Dict[str, Any]]:
    alpha     = float(cert["alpha"])
    phi_t     = float(cert["phi_t"])
    phi_bound = float(cert["claimed"]["phi_tL_bound"])
    rhs       = phi_t - (1.0 - alpha) * co
    if phi_bound > rhs + TOL:
        return {
            "ok": False, "fail_type": FAIL_PHI_BOUND,
            "invariant_diff": {"claimed.phi_tL_bound": {
                "claimed": phi_bound,
                "max_allowed": rhs,
                "rhs_formula": "phi_t - (1-alpha)*C(O)",
                "phi_t": phi_t, "alpha": alpha, "C_O": co,
            }},
            "details": {}, "witnesses": [],
        }
    return None


def gate_d_orbits_bound(cert: Dict[str, Any], co: float) -> Optional[Dict[str, Any]]:
    alpha    = float(cert["alpha"])
    phi_t    = float(cert["phi_t"])
    denom    = (1.0 - alpha) * co
    expected = math.ceil(phi_t / denom)
    claimed  = int(cert["claimed"]["convergence_orbits_bound"])
    if claimed != expected:
        return {
            "ok": False, "fail_type": FAIL_ORBITS_BOUND,
            "invariant_diff": {"claimed.convergence_orbits_bound": {
                "claimed": claimed, "recomputed": expected,
                "formula": "ceil(phi_t / ((1-alpha)*C(O)))",
                "phi_t": phi_t, "alpha": alpha, "C_O": co,
            }},
            "details": {}, "witnesses": [],
        }
    return None


# ── Top-level ──────────────────────────────────────────────────────────────────

def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    err = gate_schema(cert, schema)
    if err is not None:
        return err

    err = gate_a_orbit_feasibility(cert)
    if err is not None:
        return err

    err, co = gate_2d_co(cert)
    if err is not None:
        return err

    err = gate_b_hcrit(cert)
    if err is not None:
        return err

    err = gate_c_phi_bound(cert, co)
    if err is not None:
        return err

    err = gate_d_orbits_bound(cert, co)
    if err is not None:
        return err

    eta_eff = [float(x) for x in cert["orbit"]["eta_eff"]]
    return {
        "ok": True,
        "family": 102,
        "cert_sha256": sha256_hex(canonical_json_bytes(cert)),
        "recomputed": {
            "C_O":                     co,
            "phi_tL_bound_max":        float(cert["phi_t"]) - (1.0 - float(cert["alpha"])) * co,
            "convergence_orbits_bound": math.ceil(float(cert["phi_t"]) / ((1.0 - float(cert["alpha"])) * co)),
            "orbit_length":            len(eta_eff),
            "alpha":                   float(cert["alpha"]),
        },
        "witnesses": [],
    }


# ── Self-test ──────────────────────────────────────────────────────────────────

def run_self_test() -> int:
    base   = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_default.json",        True,  None),
        ("fail_co_mismatch.json",    False, FAIL_CO_MISMATCH),
        ("fail_hcrit.json",          False, FAIL_HCRIT),
        ("fail_schema.json",         False, FAIL_SCHEMA),
    ]

    failed: List[str] = []
    results: List[Dict[str, Any]] = []

    for name, expected_ok, expected_fail in fixtures:
        cert     = load_json(base / "fixtures" / name)
        out      = validate_cert(cert, schema)
        got_ok   = bool(out.get("ok"))
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
    parser = argparse.ArgumentParser(description="Validate QA Lojasiewicz Orbit Descent Cert v1")
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
