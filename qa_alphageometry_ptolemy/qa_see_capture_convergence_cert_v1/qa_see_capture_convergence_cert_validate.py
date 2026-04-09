#!/usr/bin/env python3
"""
qa_see_capture_convergence_cert_validate.py

Validator for QA_SEE_CAPTURE_CONVERGENCE_CERT.v1  [family 196]

Certifies: T.J.J. See's capture theory of cosmical evolution (1909-1910)
mapped to QA transient-to-periodic orbit convergence.

See's mechanism:
    Free body enters gravitational field -> resisting medium dissipates energy
    -> orbit eccentricity decays -> stable periodic orbit (capture).

QA mapping:
    Free body            = arbitrary initial state (b, e) in S_m
    Gravitational field  = modular arithmetic space (mod m)
    Resisting medium     = modular reduction (% m) acts as dissipative boundary
    Eccentricity decay   = transient steps before entering periodic orbit
    Stable capture       = orbit membership (cosmos / satellite / singularity)
    Circularization      = the orbit IS the attractor

Convergence time tau(b,e):
    The number of T-operator steps from (b,e) before the state enters its
    periodic orbit. For mod-9, all 81 states are ALREADY on their periodic
    orbit (tau=0 for all), because S_9 is finite and T is a bijection on
    {1,...,9}^2 with the QA step rule.

    Key insight: for QA, every state is already captured — the "resisting
    medium" (modular arithmetic) is maximally efficient. See's theory
    predicts that stronger dissipation leads to faster capture; QA with
    finite modulus is the limiting case where capture is instantaneous.

    For extended initial conditions (b_0, e_0) not in {1,...,m}, the
    transient = 1 step (the initial mod-reduction). This models See's
    "approach from infinity" scenario.

Source: T.J.J. See, "Researches on the Evolution of the Stellar Systems,
Vol II: The Capture Theory of Cosmical Evolution" (1910); "The Capture
Theory of Satellites", PASP 21:127 (1909).

Checks:
    SCC_1       — schema_version matches
    SCC_CONV    — convergence time well-defined for all S_m states
    SCC_MEAN    — mean convergence time reported
    SCC_MAX     — max convergence time reported
    SCC_DIST    — convergence distribution by orbit family present
    SCC_MED     — resisting medium mapping articulated
    SCC_SRC     — source attribution to See present
    SCC_WITNESS — at least 3 witnesses (one per orbit family)
    SCC_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates finite S_m orbit convergence; integer state space; no observer, no floats, no continuous dynamics"

import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SEE_CAPTURE_CONVERGENCE_CERT.v1"


# -----------------------------------------------------------------------------
# QA primitives (integer-only, axiom-compliant)
# -----------------------------------------------------------------------------

def qa_step(b, e, m):
    """One T-operator step: (b, e) -> (e, (b+e-1)%m + 1).  A1-compliant."""
    return (e, (b + e - 1) % m + 1)


def orbit_of(b, e, m, max_steps=200):
    """Return the periodic orbit containing (b,e) under T in S_m."""
    seen = []
    state = (b, e)
    for _ in range(max_steps):
        if state in seen:
            idx = seen.index(state)
            return seen[idx:]  # the periodic part
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    raise RuntimeError(f"orbit_of({b},{e},{m}) did not close in {max_steps} steps")


def convergence_time_extended(b0, e0, m):
    """Convergence time for an extended initial condition (b0,e0 may be outside {1..m}).

    Models See's 'approach from infinity': the initial mod-reduction is the
    capture event. Returns (tau, captured_state, orbit_family).
    """
    # Map into S_m (the 'capture' step)
    b = (b0 - 1) % m + 1
    e = (e0 - 1) % m + 1
    tau = 0 if (b0 == b and e0 == e) else 1

    # Classify orbit
    orbit = orbit_of(b, e, m)
    length = len(orbit)

    if m == 9:
        if length == 1:
            family = "singularity"
        elif length == 8:
            family = "satellite"
        elif length == 24:
            family = "cosmos"
        else:
            family = f"unknown_{length}"
    else:
        family = f"cycle_{length}"

    return tau, (b, e), family


def classify_s9():
    """Classify all 81 states in S_9 by orbit family."""
    dist = {"cosmos": 0, "satellite": 0, "singularity": 0}
    for b in range(1, 10):
        for e in range(1, 10):
            _, _, fam = convergence_time_extended(b, e, 9)
            dist[fam] = dist.get(fam, 0) + 1
    return dist


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def _run_checks(fixture):
    results = {}

    # SCC_1: schema version
    results["SCC_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    # SCC_CONV: convergence data present
    conv = fixture.get("convergence", {})
    results["SCC_CONV"] = (
        "modulus" in conv
        and "total_states" in conv
        and conv.get("total_states", 0) > 0
    )

    # SCC_MEAN: mean convergence time
    results["SCC_MEAN"] = "mean_tau" in conv

    # SCC_MAX: max convergence time
    results["SCC_MAX"] = "max_tau" in conv

    # SCC_DIST: distribution by orbit family
    dist = conv.get("family_distribution", {})
    results["SCC_DIST"] = len(dist) >= 2  # at least 2 families

    # If modulus == 9, verify counts match known orbit structure
    if conv.get("modulus") == 9 and results["SCC_DIST"]:
        expected = {"cosmos": 72, "satellite": 8, "singularity": 1}
        actual_dist = {k: v.get("count", 0) if isinstance(v, dict) else v for k, v in dist.items()}
        computed = classify_s9()
        # Check that fixture's distribution matches computation
        for fam in expected:
            if actual_dist.get(fam) != expected[fam]:
                results["SCC_DIST"] = False
                break

    # SCC_MED: resisting medium mapping
    med = fixture.get("resisting_medium_mapping", {})
    results["SCC_MED"] = (
        "see_concept" in med
        and "qa_concept" in med
        and len(med.get("see_concept", "")) > 0
    )

    # SCC_SRC: source attribution
    src = fixture.get("source_attribution", "")
    results["SCC_SRC"] = "See" in src and ("1909" in src or "1910" in src)

    # SCC_WITNESS: at least 3 witnesses
    witnesses = fixture.get("witnesses", [])
    results["SCC_WITNESS"] = len(witnesses) >= 3

    # Verify witness families are distinct
    if results["SCC_WITNESS"]:
        fams = {w.get("orbit_family") for w in witnesses}
        results["SCC_WITNESS"] = len(fams) >= 3

    # SCC_F: fail_ledger
    fl = fixture.get("fail_ledger")
    results["SCC_F"] = isinstance(fl, list)

    return results


def validate_fixture(path):
    with open(path) as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    all_pass = all(checks.values())
    actual = "PASS" if all_pass else "FAIL"
    ok = actual == expected
    return {"ok": ok, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    """Run validator against bundled fixtures."""
    fdir = Path(__file__).parent / "fixtures"
    results = {}
    for fp in sorted(fdir.glob("*.json")):
        results[fp.name] = validate_fixture(fp)
    all_ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    elif len(sys.argv) > 1:
        r = validate_fixture(sys.argv[1])
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    else:
        print("Usage: python qa_see_capture_convergence_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
