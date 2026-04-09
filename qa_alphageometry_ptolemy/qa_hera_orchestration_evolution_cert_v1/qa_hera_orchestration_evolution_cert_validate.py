#!/usr/bin/env python3
"""
qa_hera_orchestration_evolution_cert_validate.py

Validator for QA_HERA_ORCHESTRATION_EVOLUTION_CERT.v1  [family 206]

Certifies: Structural correspondence between HERA's multi-agent
orchestration evolution (Li & Ramakrishnan, VT 2026) and QA orbit dynamics.

Claims:
  ROPE    — RoPE dual-axes (op/bp) = Bateson [191] L1/L2a filtration
  PHASE   — Four-phase topology evolution = orbit descent with rebound
  ENTROPY — Entropy plateau at intermediate level = Satellite convergence
  SPARSE  — Sparse exploration = orbit discovery (invariant subsets)
  NT      — Theorem NT compliance (semantic insights = observer projections)
  PERF    — 38.69% improvement validates orbit-aware orchestration

Checks:
  HOE_1      — schema_version matches
  HOE_ROPE   — dual-axes present with Bateson level mapping
  HOE_PHASE  — 4 phases present; final phase NOT Singularity
  HOE_ENT    — entropy plateau at intermediate (not max, not zero)
  HOE_PERF   — improvement > 0 over baselines
  HOE_W      — at least 4 witnesses
  HOE_F      — falsifier well-formed
"""

QA_COMPLIANCE = "cert_validator — validates HERA orchestration evolution claims; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_HERA_ORCHESTRATION_EVOLUTION_CERT.v1"


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"HOE_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    if cert.get("expect_fail"):
        if not cert.get("fail_reason"):
            errors.append("HOE_F: expect_fail fixture missing fail_reason")
        # Check that the phase claim shows Singularity convergence (which should fail)
        claims = cert.get("claims", [])
        for claim in claims:
            if claim.get("id") == "PHASE":
                w = claim.get("witnesses", {})
                phases = w.get("phases", [])
                if phases:
                    final = phases[-1]
                    orbit = final.get("qa_orbit", "")
                    if "singularity" not in orbit.lower():
                        errors.append("HOE_F: expect_fail fixture final phase should be Singularity")
        return errors, warnings

    claims = cert.get("claims", [])

    if len(claims) < 4:
        warnings.append(f"HOE_W: need >= 4 claims, got {len(claims)}")

    for claim in claims:
        cid = claim.get("id", "?")
        w = claim.get("witnesses", {})

        if cid == "ROPE":
            axes = w.get("hera_axes", {})
            if not axes.get("operational_rules"):
                warnings.append("HOE_ROPE: missing operational_rules description")
            if not axes.get("behavioral_principles"):
                warnings.append("HOE_ROPE: missing behavioral_principles description")
            bm = w.get("bateson_mapping", {})
            if not bm.get("op_rules") or not bm.get("bp_principles"):
                warnings.append("HOE_ROPE: missing Bateson level mapping")

        elif cid == "PHASE":
            phases = w.get("phases", [])
            if len(phases) != 4:
                warnings.append(f"HOE_PHASE: expected 4 phases, got {len(phases)}")
            # Final phase must NOT be Singularity
            if phases:
                final_orbit = phases[-1].get("qa_orbit", "")
                if "singularity" in final_orbit.lower():
                    errors.append("HOE_PHASE: final phase is Singularity — system should converge to Satellite, not Singularity")
            # Check not_singularity flag
            ns = w.get("not_singularity")
            if not ns:
                warnings.append("HOE_PHASE: missing not_singularity explanation")

        elif cid == "ENTROPY":
            obs = w.get("observation", "")
            if "intermediate" not in obs.lower() and "stabiliz" not in obs.lower():
                warnings.append("HOE_ENT: entropy observation should mention intermediate/stabilization")

        elif cid == "PERF":
            imp = w.get("improvement")
            if imp is not None and imp <= 0:
                errors.append(f"HOE_PERF: improvement = {imp}, expected > 0")
            benchmarks = w.get("benchmarks")
            if benchmarks is not None and benchmarks < 1:
                errors.append(f"HOE_PERF: benchmarks = {benchmarks}, expected >= 1")

    # Numerical checks
    num = cert.get("numerical_checks", {})
    if num:
        perf = num.get("performance_improvement", {})
        if perf:
            imp = perf.get("hera_avg_improvement_pct")
            if imp is not None and imp <= 0:
                errors.append(f"HOE_NUM: improvement {imp}% <= 0")
            beaten = perf.get("single_agent_beaten")
            if beaten is not None and not beaten:
                errors.append("HOE_NUM: single agent not beaten — orbit descent analogy fails")

        phases = num.get("phase_count", {})
        if phases:
            pc = phases.get("phases")
            if pc is not None and pc != 4:
                warnings.append(f"HOE_NUM: phase count = {pc}, expected 4")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("hoe_pass_core.json", True),
        ("hoe_pass_numerical.json", True),
        ("hoe_fail_singularity.json", True),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs, "warnings": warns})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA HERA Orchestration Evolution Cert [206] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
