#!/usr/bin/env python3
"""Validator for QA Self-Pumped Phase Conjugate Cert [523].

Primary source: (Feinberg, 1982) DOI 10.1364/OL.7.000486 (self-pumped phase
conjugator using internal reflection); (Cronin-Golomb et al., 1984)
DOI 10.1109/JQE.1984.1072018; (Zel'dovich, 1985) ISBN 978-3-540-13458-4.

The self-pumped ("cat") phase conjugator supplies NO external pump: internal
reflection self-generates the counter-pump qa_neg(p) from the loop field, and the
four-wave-mixing output fwm(p, qa_neg(p), s) = qa_neg(s) is INDEPENDENT of the
self-pump p -- which is exactly why the mirror needs no external reference.

Checks (integer-only, A1-compliant; gain/amplitude are observer-layer reals):
  C1 PUMP_INDEPENDENCE   fwm(p, qa_neg(p), s) == qa_neg(s) for ALL p,s  (exhaustive)
  C2 SELF_STARTING       output stays qa_neg(s) as the self-pump wanders (== C1)
  C3 THRESHOLD           loop A' = g*A/(1+A): self-oscillates iff g>1 (g_c=1)
  C4 SELF_PUMPED_DC      aberrate(phi) -> self-conjugate -> return(phi) == qa_neg(s)
                         for all s,phi; different screen leaves qa_mod(-s+phi'-phi)
  C5 A1_RANGE            every phase operation stays in {1,...,m}

Self-starting cousin of cert [518] (which supplies the pumps); distinct from cert
[519] (a resonator driven by stored patterns) in that the pump is internal.

Usage: python qa_self_pumped_conjugate_cert_validate.py --self-test
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

SCHEMA_VERSION = "QA_SELF_PUMPED_CONJUGATE_CERT.v1"
FAMILY_ID = 523
SLUG = "qa_self_pumped_conjugate_cert_v1"

REQUIRED_FIELDS = {
    "schema_version", "fixture_kind", "primary_source",
    "m", "signal_state", "self_pump", "phase_screen",
    "return_screen", "expected_conjugate", "expected_recovered",
}

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


# --------------------------------------------------------------------------- #
# QA additive group + self-pumped conjugator (integer only, A1-compliant)
# --------------------------------------------------------------------------- #
def qa_mod(x: int, m: int) -> int:
    return ((int(x) - 1) % m) + 1


def qa_add(a: int, b: int, m: int) -> int:
    return qa_mod(a + b, m)


def qa_neg(a: int, m: int) -> int:
    return qa_mod(-a, m)


def fwm(pf: int, pb: int, s: int, m: int) -> int:
    return qa_mod(pf + pb - s, m)


def self_pumped(p_loop: int, s: int, m: int) -> int:
    """Self-pumped output: internal reflection makes the counter-pump qa_neg(p_loop);
    FWM of the self-generated conjugate pump pair with the signal. No external pump."""
    return fwm(p_loop, qa_neg(p_loop, m), s, m)


def self_pumped_recover(s: int, phi_fwd: int, phi_ret: int, p_loop: int, m: int) -> int:
    """Aberrate by phi_fwd -> self-pumped conjugate (any self-pump) -> return phi_ret."""
    distorted = qa_add(s, phi_fwd, m)
    c = self_pumped(p_loop, distorted, m)
    return qa_add(c, phi_ret, m)


def mirror_state(g: float) -> str:
    """Self-oscillation classification by the analytic fixed point of A'=g*A/(1+A):
    A*=g-1 for g>1 (ON), A=0 for g<1 (off), g=1 marginal edge (threshold)."""
    if g > 1.0:
        return "on"
    if g == 1.0:
        return "threshold"
    return "off"


def _in_range(x, m: int) -> bool:
    return isinstance(x, int) and 1 <= x <= m


# --------------------------------------------------------------------------- #
# Per-fixture checks
# --------------------------------------------------------------------------- #
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
    for key in ("signal_state", "self_pump", "phase_screen",
                "return_screen", "expected_conjugate", "expected_recovered"):
        if not _in_range(data[key], m):
            return f"OUT_OF_RANGE: {key}={data[key]} not in 1..{m}"

    s = data["signal_state"]
    p = data["self_pump"]
    phi = data["phase_screen"]
    phi2 = data["return_screen"]

    # C1/C2: self-pumped output is qa_neg(s), independent of the self-pump p
    conj = self_pumped(p, s, m)
    if conj != data["expected_conjugate"]:
        return f"WRONG_CONJUGATE: self_pumped={conj} != expected={data['expected_conjugate']}"
    if conj != qa_neg(s, m):
        return f"WRONG_CONJUGATE: self_pumped={conj} != qa_neg(s)={qa_neg(s, m)}"

    # C4: self-referenced distortion correction
    recovered = self_pumped_recover(s, phi, phi2, p, m)
    if recovered != data["expected_recovered"]:
        return (f"WRONG_RECOVERY: recover={recovered} != "
                f"expected_recovered={data['expected_recovered']}")
    residual = qa_mod(-s + phi2 - phi, m)          # closed form, independent of pipeline
    if recovered != residual:
        return f"WRONG_RECOVERY: recover={recovered} != closed-form residual={residual}"
    if phi2 == phi and recovered != qa_neg(s, m):
        return f"WRONG_RECOVERY: same-medium recover={recovered} != qa_neg(s)={qa_neg(s, m)}"
    return None


def _check_fail_fixture(data: dict) -> Optional[str]:
    """A fail fixture must FAIL _check_pass_fixture (i.e. be genuinely invalid)."""
    if data.get("fixture_kind") != "fail":
        return f"WRONG_KIND: expected fail, got {data.get('fixture_kind')}"
    err = _check_pass_fixture({**data, "fixture_kind": "pass"})
    if err is None:
        return "FAIL_FIXTURE_UNEXPECTEDLY_VALID: a fail fixture validated as pass"
    return None


# --------------------------------------------------------------------------- #
# Exhaustive self-consistency (not fixture-trusting)
# --------------------------------------------------------------------------- #
def _exhaustive(m: int = 24) -> Optional[str]:
    # C1 PUMP_INDEPENDENCE / C2 SELF_STARTING: output == qa_neg(s) for all p,s
    for p in range(1, m + 1):
        for s in range(1, m + 1):
            if self_pumped(p, s, m) != qa_neg(s, m):
                return f"C1 PUMP_INDEPENDENCE failed at p={p}, s={s}"
    # C4 SELF_PUMPED_DC same-medium: qa_neg(s) for all s,phi and any self-pump
    for s in range(1, m + 1):
        for phi in range(1, m + 1):
            for p in (1, 7, m):                      # a few self-pumps: must not matter
                if self_pumped_recover(s, phi, phi, p, m) != qa_neg(s, m):
                    return f"C4 SELF_PUMPED_DC failed at s={s}, phi={phi}, p={p}"
    # C3 THRESHOLD: g>1 -> A*=g-1>0 ; g<=1 -> A->0
    for g in (0.5, 0.9, 1.1, 1.5, 2.0):
        A = 0.01
        for _ in range(4000):
            A = g * A / (1.0 + A)
        want = max(g - 1.0, 0.0)
        if abs(A - want) > 1e-3:
            return f"C3 THRESHOLD failed at g={g}: A={A:.5f} != A*={want:.2f}"
        if (g > 1.0) != (mirror_state(g) == "on"):
            return f"C3 THRESHOLD state mismatch at g={g}"
    return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def self_test() -> int:
    errors = []

    ex = _exhaustive()
    if ex is not None:
        errors.append(f"exhaustive_self_consistency: {ex}")

    n_pass = n_fail = 0
    for fn in sorted(os.listdir(FIXTURES)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(FIXTURES, fn)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("fixture_kind") == "fail":
            n_fail += 1
            err = _check_fail_fixture(data)
        else:
            n_pass += 1
            err = _check_pass_fixture(data)
        if err is not None:
            errors.append(f"{fn}: {err}")

    ok = not errors
    print(json.dumps({
        "ok": ok,
        "family_id": FAMILY_ID,
        "slug": SLUG,
        "schema_version": SCHEMA_VERSION,
        "pass_fixtures": n_pass,
        "fail_fixtures": n_fail,
        "errors": errors,
    }, indent=2))
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    print(__doc__)
    print("Run with --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
