"""
QA Four-Wave-Mixing Phase Conjugate Cert [518]

Primary sources:
  (Hellwarth, 1977) "Generation of time-reversed wave fronts by nonlinear
    refraction." J. Opt. Soc. Am. 67(1):1-3. DOI 10.1364/JOSA.67.000001
  (Yariv, 1978) "Phase conjugate optics and real-time holography." IEEE J.
    Quantum Electron. 14(9):650-660. DOI 10.1109/JQE.1978.1069870
  (Zel'dovich, Pilipetsky, Shkunov, 1985) "Principles of Phase Conjugation."
    Springer. ISBN 978-3-540-13458-4
  (Agarwal & Friberg) "Scattering theory of distortion correction by phase
    conjugation." J. Opt. Soc. Am. (distortion-correction theorem)

CLAIM (exact, falsifiable): the degenerate four-wave-mixing phase-sum relation
theta_c = theta_f + theta_b - theta_s is realized EXACTLY in the QA additive
group on the A1 alphabet {1,...,m}:

    qa_mod(x)      = ((x - 1) % m) + 1          # A1: values in {1,...,m}
    qa_add(a,b)    = qa_mod(a + b)              # identity element = m (No-Zero 0)
    qa_neg(a)      = qa_mod(-a)                 # phase conjugation (involution)
    fwm(pf,pb,s)   = qa_mod(pf + pb - s)        # FWM phase-sum relation

With conjugate pumps pb = qa_neg(pf), fwm(pf,pb,s) = qa_neg(s) exactly. The
distortion-correction theorem then holds exactly: aberrate by phase screen phi,
conjugate, return through the SAME phi -> qa_neg(s) exactly (same-medium
recovery); return through phi' != phi leaves residual qa_mod(-s + phi' - phi).

Checks (all exhaustive over {1,...,m}, deterministic, integer-only):
  FWM_CONJUGATE   fwm(pf, qa_neg(pf), s) == qa_neg(s) for all pf,s
  DC_SAME_MEDIUM  recover(s, phi, phi, pf) == qa_neg(s) for all s,phi (any pf)
  DC_DIFF_RESID   recover(s, phi, phi2, pf) == qa_mod(-s + phi2 - phi) for all
                  s,phi,phi2 -> exact only where phi2==phi (same-medium
                  specificity)
  GROUP_IDENTITY  qa_add(a, m) == a for all a; m is the UNIQUE identity; != 0
  CONJ_INVOLUTION qa_neg(qa_neg(a)) == a for all a; exactly 2 fixed points
  CONTROL_NONCONJ a non-conjugate second pump does NOT reconstruct for all s
                  (conjugation is load-bearing)
  A1_RANGE        every operation output lies in {1,...,m}, never 0
  SRC             mapping_protocol_ref.json present + correct protocol_version
  F               pass/fail fixtures behave as declared

Companion: cert [155] (weak emergent phase-conjugate signature this explicit
operator supersedes). Reference impl: qa_fwm_conjugator.py (repo root).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

QA_COMPLIANCE = (
    "cert_validator -- integer phase arithmetic on the QA additive group over "
    "the A1 alphabet {1,...,m}: qa_mod(x)=((x-1)%m)+1, additive identity is m "
    "(the No-Zero representative of 0), never 0. No float state; the observer "
    "boundary (intensity<->phase) appears only in the reference demo, not here."
)

SCHEMA_VERSION = "QA_FWM_PHASE_CONJUGATE_CERT.v1"
FAMILY_ID = 518
SLUG = "qa_fwm_phase_conjugate_cert_v1"

REQUIRED_FIELDS = {
    "schema_version", "fixture_kind", "primary_source",
    "m", "signal_state", "phase_screen", "pump_forward",
    "return_screen", "expected_recovered",
}


# ---------------------------------------------------------------------------
# QA additive group + FWM conjugator (integer only, A1-compliant)
# ---------------------------------------------------------------------------
def qa_mod(x: int, m: int) -> int:
    return ((int(x) - 1) % m) + 1


def qa_add(a: int, b: int, m: int) -> int:
    return qa_mod(a + b, m)


def qa_neg(a: int, m: int) -> int:
    return qa_mod(-a, m)


def fwm_conjugate(p_f: int, p_b: int, s: int, m: int) -> int:
    return qa_mod(p_f + p_b - s, m)


def aberrate(s: int, phi: int, m: int) -> int:
    return qa_add(s, phi, m)


def recover(s: int, phi_forward: int, phi_return: int, p_f: int, m: int,
            p_b: Optional[int] = None) -> int:
    """Full pipeline: aberrate -> FWM conjugate (conjugate pumps unless p_b
    given) -> return through phi_return."""
    if p_b is None:
        p_b = qa_neg(p_f, m)
    distorted = aberrate(s, phi_forward, m)
    c = fwm_conjugate(p_f, p_b, distorted, m)
    return qa_add(c, phi_return, m)


def _in_range(x: int, m: int) -> bool:
    return isinstance(x, int) and 1 <= x <= m


# ---------------------------------------------------------------------------
# Per-fixture checks
# ---------------------------------------------------------------------------
def _check_pass_fixture(data: dict) -> Optional[str]:
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    if data["schema_version"] != SCHEMA_VERSION:
        return f"WRONG_SCHEMA: {data['schema_version']}"
    if data["fixture_kind"] != "pass":
        return f"WRONG_KIND: {data['fixture_kind']}"

    m = data["m"]
    if not (isinstance(m, int) and m >= 2):
        return f"OUT_OF_RANGE: m={m}"
    for key in ("signal_state", "phase_screen", "pump_forward",
                "return_screen", "expected_recovered"):
        if not _in_range(data[key], m):
            return f"OUT_OF_RANGE: {key}={data[key]} not in 1..{m}"

    s = data["signal_state"]
    phi = data["phase_screen"]
    phi2 = data["return_screen"]
    p_f = data["pump_forward"]

    recovered = recover(s, phi, phi2, p_f, m)
    if recovered != data["expected_recovered"]:
        return (f"WRONG_RECOVERY: recover={recovered} != "
                f"expected_recovered={data['expected_recovered']}")

    # Closed-form residual cross-check (independent of recover()):
    residual = qa_mod(-s + phi2 - phi, m)
    if recovered != residual:
        return (f"WRONG_RECOVERY: recover={recovered} != "
                f"closed-form residual qa_mod(-s+phi'-phi)={residual}")

    # Same-medium theorem: when the return screen matches, recovery must be
    # the exact phase conjugate of the clean signal.
    if phi2 == phi and recovered != qa_neg(s, m):
        return (f"WRONG_RECOVERY: same-medium recover={recovered} != "
                f"qa_neg(s)={qa_neg(s, m)}")

    return None


def _check_fail_fixture(data: dict) -> Optional[str]:
    if "expected_fail_type" not in data:
        return "FAIL_FIXTURE_MISSING_expected_fail_type"
    fail_type = data["expected_fail_type"]
    if fail_type not in ("MISSING_FIELD", "WRONG_RECOVERY", "OUT_OF_RANGE"):
        return f"UNKNOWN_expected_fail_type: {fail_type}"

    if fail_type == "MISSING_FIELD":
        missing = REQUIRED_FIELDS - set(data.keys())
        if not missing:
            return "FAIL_FIXTURE_DID_NOT_FAIL: expected MISSING_FIELD but all present"
        return None

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {fail_type} but missing {sorted(missing)}"

    err = _check_pass_fixture({**data, "fixture_kind": "pass"})
    if err is None:
        return f"FAIL_FIXTURE_DID_NOT_FAIL: expected {fail_type} but fixture is consistent"
    if not err.startswith(fail_type):
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {fail_type} but got {err}"
    return None


def validate_fixture(path: str) -> tuple[bool, str]:
    with open(path) as f:
        data = json.load(f)
    kind = data.get("fixture_kind")
    if kind == "pass":
        err = _check_pass_fixture(data)
        return (False, f"FAIL (expected PASS): {err}") if err else (True, "PASS")
    elif kind == "fail":
        err = _check_fail_fixture(data)
        return (False, f"FAIL (fail-fixture check): {err}") if err else (True, "PASS (expected FAIL)")
    return False, f"UNKNOWN fixture_kind: {kind}"


# ---------------------------------------------------------------------------
# Exhaustive theorem self-test
# ---------------------------------------------------------------------------
def self_test() -> dict:
    errors: list[str] = []
    m = 24  # applied QA modulus

    # FWM_CONJUGATE: conjugate pumps give the exact conjugate, for all pf, s
    for p_f in range(1, m + 1):
        p_b = qa_neg(p_f, m)
        for s in range(1, m + 1):
            if fwm_conjugate(p_f, p_b, s, m) != qa_neg(s, m):
                errors.append(f"FWM_CONJUGATE FAILED: pf={p_f}, s={s}")

    # DC_SAME_MEDIUM: same-medium recovery == qa_neg(s), all s, phi, sample pumps
    for p_f in (1, 7, 13, 24):
        for s in range(1, m + 1):
            for phi in range(1, m + 1):
                if recover(s, phi, phi, p_f, m) != qa_neg(s, m):
                    errors.append(f"DC_SAME_MEDIUM FAILED: pf={p_f}, s={s}, phi={phi}")

    # DC_DIFF_RESID: residual formula exact for all s, phi, phi2 (exact iff phi2==phi)
    exact_only_when_matched = True
    for s in range(1, m + 1):
        for phi in range(1, m + 1):
            for phi2 in range(1, m + 1):
                rec = recover(s, phi, phi2, 7, m)
                if rec != qa_mod(-s + phi2 - phi, m):
                    errors.append(f"DC_DIFF_RESID FAILED: s={s},phi={phi},phi2={phi2}")
                is_exact = (rec == qa_neg(s, m))
                if is_exact and phi2 != phi:
                    exact_only_when_matched = False
    if not exact_only_when_matched:
        errors.append("DC_DIFF_RESID FAILED: exact recovery occurred for a mismatched medium")

    # GROUP_IDENTITY: m is the unique additive identity, and it is never 0
    identities = [e for e in range(1, m + 1)
                  if all(qa_add(a, e, m) == a for a in range(1, m + 1))]
    if identities != [m]:
        errors.append(f"GROUP_IDENTITY FAILED: identities={identities} (expected [{m}])")
    if 0 in identities:
        errors.append("GROUP_IDENTITY FAILED: zero appeared as a state (A1 violation)")

    # CONJ_INVOLUTION: involution with exactly two fixed points (m and m/2)
    if any(qa_neg(qa_neg(a, m), m) != a for a in range(1, m + 1)):
        errors.append("CONJ_INVOLUTION FAILED: qa_neg is not an involution")
    fixed = [a for a in range(1, m + 1) if qa_neg(a, m) == a]
    if sorted(fixed) != sorted({m, m // 2}):
        errors.append(f"CONJ_INVOLUTION FAILED: fixed points {fixed} != {sorted({m, m//2})}")

    # CONTROL_NONCONJ: a non-conjugate second pump must NOT reconstruct for all s
    p_f = 7
    p_b_bad = qa_add(qa_neg(p_f, m), 1, m)  # off by one from the true conjugate
    matches = sum(1 for s in range(1, m + 1)
                  if recover(s, 5, 5, p_f, m, p_b=p_b_bad) == qa_neg(s, m))
    if matches == m:
        errors.append("CONTROL_NONCONJ FAILED: non-conjugate mixer reconstructed all states")

    # A1_RANGE: every operation output stays in {1,...,m}
    for a in range(1, m + 1):
        for b in range(1, m + 1):
            if not _in_range(qa_add(a, b, m), m) or not _in_range(qa_neg(a, m), m):
                errors.append(f"A1_RANGE FAILED: a={a}, b={b}")
                break

    # SRC gate
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "mapping_protocol_ref.json")
    if not os.path.exists(src_path):
        errors.append("SRC FAILED: mapping_protocol_ref.json missing")
    else:
        with open(src_path) as f:
            ref = json.load(f)
        if ref.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
            errors.append("SRC FAILED: wrong protocol_version in mapping_protocol_ref.json")

    # F gate: fixtures
    fixture_dir = os.path.join(here, "fixtures")
    pass_fixtures, fail_fixtures = [], []
    if os.path.isdir(fixture_dir):
        for fname in sorted(os.listdir(fixture_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(fixture_dir, fname)
            with open(fpath) as f:
                fx = json.load(f)
            kind = fx.get("fixture_kind")
            ok, msg = validate_fixture(fpath)
            if kind == "pass":
                pass_fixtures.append(fname)
                if not ok:
                    errors.append(f"F FAILED (pass fixture {fname}): {msg}")
            elif kind == "fail":
                fail_fixtures.append(fname)
                if not ok:
                    errors.append(f"F FAILED (fail fixture {fname}): {msg}")
    else:
        errors.append("F FAILED: fixtures/ directory missing")

    return {
        "ok": len(errors) == 0,
        "family_id": FAMILY_ID,
        "slug": SLUG,
        "schema_version": SCHEMA_VERSION,
        "pass_fixtures": len(pass_fixtures),
        "fail_fixtures": len(fail_fixtures),
        "errors": errors,
    }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        result = self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <fixture.json>  |  --self-test", file=sys.stderr)
        sys.exit(1)
    ok, msg = validate_fixture(sys.argv[1])
    print(msg)
    sys.exit(0 if ok else 1)
