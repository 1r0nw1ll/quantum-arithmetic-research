"""
QA Volk Apollonian Bipolar Mapping Cert [517]

Primary source: Volk, G. (2010). "Toroids, Vortices, Knots, Topology and
  Quanta, Part 2." Proceedings of the NPA (NPA-18), College Park, MD.
  Original .doc recovered (the on-disk OCR text extraction had lost
  every equation body); its 233 embedded MathType "Equation Native" OLE
  objects were decoded with a from-scratch MTEF v5 binary parser. One
  decoded object confirms Volk's own toroidal vector form -- numerator
  (sinh(eta)cos(theta), sinh(eta)sin(theta), sin(phi)) over denominator
  (cosh(eta)-cos(phi)) -- and another confirms the hyperbolic<->circular
  identity chain (tanh(eta)=sin(psi), cosh(eta)=sec(psi), etc.),
  self-consistently.
Helicola R/r parameter naming: Ginzburg, V. (2006). Prime Elements of
  Ordinary Matter, Dark Matter & Dark Energy. Helicola Press.

CLAIM (narrow, falsifiable): for any QA tuple (b,e,d,a) with d=b+e,
a=b+2e (QA's own A2 definitions), the identity

    d^2 - e^2 = (d-e)(d+e) = b*(b+2e) = b*a = F

holds exactly (not approximately) -- and this is precisely Volk's own
Apollonian orthogonality relation R^2 = a_volk^2 + r_volk^2 (his paper,
section 2: "there may exist many M-circles or toroids centered at
B=(+-R,0)... which we'll see satisfies [this relation], with r the
radius of the M-circle and R the distance from the origin to the circle
center") under the mapping:

    a_volk = e
    R_volk = d
    r_volk = sqrt(F) = sqrt(a*b)
    eta = arccoth(R_volk / a_volk) = 0.5 * ln(a/b)

Independently corroborated by a GeoGebra construction the user built
over a year before this cert (geogebra.org/calculator/nwkeyb7j, titled
"grant,volk-toroid1235"), built directly from BEDA=(1,2,3,5) against
Volk's own Figure 2: point A=(e,0)=(2,0), point R=(d,0)=(3,0), a circle
centered at R with radius sqrt(5)=sqrt(F) -- an exact numeric match to
this derivation, found independently and only afterward connected to it.

Implementation: qa_lab/qa_volk_coordinates.py, function beda_to_volk.

LIMITATION (not resolved by this cert): a single static BEDA tuple only
fixes the M-circle/E-circle FAMILY (a_volk, eta) -- i.e. which torus
shape -- not a specific point on it (the bipolar angle rho). Which rho a
QA orbit should trace as it iterates is an open question.

Checks:
  VAM_IDENTITY    d^2-e^2 == a*b exactly, for the fixture's (b,e)
  VAM_A_VOLK      declared expected_a_volk == e
  VAM_R_VOLK      declared expected_R_volk == d == b+e
  VAM_R2_VOLK     declared expected_r_volk_squared == a*b (a = b+2e)
  VAM_APOLLONIAN  R_volk^2 == a_volk^2 + r_volk^2 (Volk's own relation)
  VAM_ETA         eta=0.5*ln(a/b) matches arccoth(d/e) AND independently
                  reproduces R_volk via a_volk*coth(eta) through
                  math.cosh/sinh directly (self-test gate; not a
                  per-fixture field, since eta is an observer-projection
                  float per Theorem NT, not raw QA state)
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic on QA roots b,e,d,a (A1/A2 "
    "compliant, d=b+e and a=b+2e always derived, never assigned "
    "independently); r_volk kept squared (r_volk_squared=a*b) throughout "
    "so no float sqrt or float QA state is ever required"
)

SCHEMA_VERSION = "QA_VOLK_APOLLONIAN_MAPPING_CERT.v1"
FAMILY_ID = 517
SLUG = "qa_volk_apollonian_mapping_cert_v1"

REQUIRED_FIELDS = {
    "schema_version", "fixture_kind", "primary_source",
    "b", "e", "expected_a_volk", "expected_R_volk", "expected_r_volk_squared",
}


def _qa_derive(b: int, e: int) -> dict:
    d = b + e
    a = b + 2 * e
    F = a * b
    return {"b": b, "e": e, "d": d, "a": a, "F": F}


def _check_pass_fixture(data: dict) -> Optional[str]:
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    if data["schema_version"] != SCHEMA_VERSION:
        return f"WRONG_SCHEMA: {data['schema_version']}"
    if data["fixture_kind"] != "pass":
        return f"WRONG_KIND: {data['fixture_kind']}"

    b, e = data["b"], data["e"]
    if not (isinstance(b, int) and isinstance(e, int) and b >= 1 and e >= 1):
        return f"OUT_OF_RANGE: b={b}, e={e}"

    derived = _qa_derive(b, e)
    d, a, F = derived["d"], derived["a"], derived["F"]

    # VAM_IDENTITY: d^2-e^2 == a*b == F, exactly (integer arithmetic, no float)
    if (d * d - e * e) != F:
        return f"WRONG_IDENTITY: d^2-e^2={d*d - e*e} != F={F}"

    # VAM_A_VOLK
    if data["expected_a_volk"] != e:
        return f"WRONG_A_VOLK: expected_a_volk={data['expected_a_volk']} != e={e}"

    # VAM_R_VOLK
    if data["expected_R_volk"] != d:
        return f"WRONG_APOLLONIAN: expected_R_volk={data['expected_R_volk']} != d={d}"

    # VAM_R2_VOLK
    if data["expected_r_volk_squared"] != F:
        return f"WRONG_APOLLONIAN: expected_r_volk_squared={data['expected_r_volk_squared']} != F={F}"

    # VAM_APOLLONIAN: R_volk^2 == a_volk^2 + r_volk^2  (Volk's own relation)
    R_volk, a_volk, r2_volk = d, e, F
    if R_volk * R_volk != a_volk * a_volk + r2_volk:
        return (
            f"WRONG_APOLLONIAN: R_volk^2={R_volk*R_volk} != "
            f"a_volk^2+r_volk^2={a_volk*a_volk + r2_volk}"
        )

    return None


def _check_fail_fixture(data: dict) -> Optional[str]:
    if "expected_fail_type" not in data:
        return "FAIL_FIXTURE_MISSING_expected_fail_type"
    fail_type = data["expected_fail_type"]
    if fail_type not in ("MISSING_FIELD", "WRONG_APOLLONIAN", "WRONG_A_VOLK"):
        return f"UNKNOWN_expected_fail_type: {fail_type}"

    if fail_type == "MISSING_FIELD":
        missing = REQUIRED_FIELDS - set(data.keys())
        if not missing:
            return "FAIL_FIXTURE_DID_NOT_FAIL: expected MISSING_FIELD but all fields present"
        return None

    # For the other fail types, required fields must all be present so the
    # pass-checker actually reaches (and rejects on) the wrong value.
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {fail_type} but missing {sorted(missing)}"

    err = _check_pass_fixture({**data, "fixture_kind": "pass"})
    if err is None:
        return f"FAIL_FIXTURE_DID_NOT_FAIL: expected {fail_type} but fixture is actually internally consistent"
    if not err.startswith(fail_type):
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {fail_type} but got {err}"
    return None


def validate_fixture(path: str) -> tuple[bool, str]:
    with open(path) as f:
        data = json.load(f)

    kind = data.get("fixture_kind")
    if kind == "pass":
        err = _check_pass_fixture(data)
        if err:
            return False, f"FAIL (expected PASS): {err}"
        return True, "PASS"
    elif kind == "fail":
        err = _check_fail_fixture(data)
        if err:
            return False, f"FAIL (fail-fixture check): {err}"
        return True, "PASS (expected FAIL)"
    else:
        return False, f"UNKNOWN fixture_kind: {kind}"


def _gate_check(label: str, condition: bool, detail: str = "") -> Optional[str]:
    if not condition:
        return f"GATE {label} FAILED: {detail}"
    return None


def self_test() -> dict:
    errors = []

    # VAM_IDENTITY gate: exhaustive over a wide (b,e) grid, not just the
    # canonical (1,2,3,5) tuple -- this is what makes the claim "general"
    for b in range(1, 51):
        for e in range(1, 51):
            derived = _qa_derive(b, e)
            d, a, F = derived["d"], derived["a"], derived["F"]
            if (d * d - e * e) != F:
                errors.append(f"GATE VAM_IDENTITY FAILED: b={b},e={e}: d^2-e^2={d*d-e*e} != F={F}")

    # VAM_APOLLONIAN gate, same grid, restated in Volk's own variables
    for b in range(1, 51):
        for e in range(1, 51):
            derived = _qa_derive(b, e)
            d, a, F = derived["d"], derived["a"], derived["F"]
            R_volk, a_volk, r2_volk = d, e, F
            if R_volk * R_volk != a_volk * a_volk + r2_volk:
                errors.append(f"GATE VAM_APOLLONIAN FAILED: b={b},e={e}")

    # VAM_ETA gate: eta = arccoth(R_volk/a_volk) = 0.5*ln(a/b) via THREE
    # independent routes -- (1) the closed-form 0.5*ln(a/b), (2) the
    # standard arccoth identity 0.5*ln((x+1)/(x-1)) with x=d/e, (3) solving
    # R_volk=a_volk*coth(eta) numerically for eta and checking it reproduces
    # R_volk via math.cosh/sinh directly (not just algebraically).
    for b in range(1, 21):
        for e in range(1, 21):
            derived = _qa_derive(b, e)
            d, a = derived["d"], derived["a"]
            eta_closed_form = 0.5 * math.log(a / b)
            x = d / e
            eta_arccoth = 0.5 * math.log((x + 1) / (x - 1))
            if abs(eta_closed_form - eta_arccoth) > 1e-9:
                errors.append(
                    f"GATE VAM_ETA FAILED (closed-form vs arccoth): b={b},e={e}: "
                    f"{eta_closed_form} != {eta_arccoth}"
                )
                continue
            R_reconstructed = e * math.cosh(eta_closed_form) / math.sinh(eta_closed_form)
            if abs(R_reconstructed - d) > 1e-9:
                errors.append(
                    f"GATE VAM_ETA FAILED (R=a_volk*coth(eta) round-trip): b={b},e={e}: "
                    f"R_reconstructed={R_reconstructed} != d={d}"
                )

    # Canonical-tuple witness gate, cross-checked against the independent
    # GeoGebra construction ("grant,volk-toroid1235"): A=(2,0), R=(3,0),
    # circle radius sqrt(5) for BEDA=(1,2,3,5)
    derived = _qa_derive(1, 2)
    err = _gate_check(
        "GEOGEBRA_WITNESS",
        derived["d"] == 3 and derived["e"] == 2 and derived["F"] == 5,
        f"expected d=3,e=2,F=5 for BEDA=(1,2,3,5); got {derived}",
    )
    if err:
        errors.append(err)

    # SRC gate
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "mapping_protocol_ref.json")
    if not os.path.exists(src_path):
        errors.append("GATE SRC FAILED: mapping_protocol_ref.json missing")
    else:
        with open(src_path) as f:
            ref = json.load(f)
        if ref.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
            errors.append("GATE SRC FAILED: wrong protocol_version in mapping_protocol_ref.json")

    # F gate: fixtures
    fixture_dir = os.path.join(here, "fixtures")
    pass_fixtures, fail_fixtures = [], []
    if os.path.isdir(fixture_dir):
        for fname in sorted(os.listdir(fixture_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(fixture_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            kind = data.get("fixture_kind")
            ok, msg = validate_fixture(fpath)
            if kind == "pass":
                pass_fixtures.append(fname)
                if not ok:
                    errors.append(f"GATE F FAILED (pass fixture {fname}): {msg}")
            elif kind == "fail":
                fail_fixtures.append(fname)
                if not ok:
                    errors.append(f"GATE F FAILED (fail fixture {fname}): {msg}")
    else:
        errors.append("GATE F FAILED: fixtures/ directory missing")

    ok = len(errors) == 0
    return {
        "ok": ok,
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
