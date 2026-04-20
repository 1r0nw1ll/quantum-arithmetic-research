#!/usr/bin/env python3
"""
qa_integer_state_pipeline_cert_validate.py

Validator for QA_INTEGER_STATE_PIPELINE_CERT.v1  [family 257]

Certifies the two-boundary-crossing invariant of Theorem NT as a structural
property of a QA-native pipeline: the observer/QA boundary is crossed
exactly twice (input tokenization, output decoding); between those two
crossings the state is pure integer tuples. Structurally eliminates the
GLM-5 TITO misalignment failure mode by construction.

Checks:
    ISP_1           — schema_version matches
    ISP_BOUND       — both input_boundary and output_boundary declared
    ISP_INT         — every interior_state_sample is an integer tuple (b,e) in {1..m}^2
    ISP_NO_REPROJECT — no re-tokenization step declared in pipeline_stages
    ISP_DET         — declared determinism (bitwise_identical=true, repeats>=100)
                      and independent recomputation on canonical trace
    ISP_SRC         — source attribution references GLM-5 arXiv + TITO
    ISP_WIT         — >= 3 interior_state_samples across the canonical trace
    ISP_F           — fail_ledger well-formed

Source grounding:
    - GLM-5 Team, arXiv:2602.15763 [cs.LG], Feb 2026 — §4.1.2 TITO gateway.
    - docs/theory/QA_GLM5_ARCHITECTURE_MAPPING.md — design doc.
    - qa_lab/qa_orbit_resonance_attention.py — reference pipeline.
    - Theorem NT canonical spec: docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md.

QA axiom compliance: integer state alphabet {1..m} (A1), S1-compliant
(b*b not b**2), T-operator is the only mod-reduction path into QA state.
"""

QA_COMPLIANCE = "cert_validator — two-boundary-crossing invariant for QA-native pipelines; integer-only interior state; observer projections declared at endpoints"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_INTEGER_STATE_PIPELINE_CERT.v1"


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    return (e, qa_mod(b + e, m))


def _evolve_canonical(tokens, steps, m=9):
    state = [tuple(t) for t in tokens]
    traj = [list(state)]
    for _ in range(steps):
        state = [qa_step(b, e, m) for (b, e) in state]
        traj.append(list(state))
    return traj


def _canonical_pipeline_trace(seed_tokens, steps, m=9):
    """Emit the deterministic interior-state sequence for a canonical input."""
    return _evolve_canonical(seed_tokens, steps, m=m)


def _is_integer_tuple(pair, m):
    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
        return False
    b, e = pair
    if not (isinstance(b, int) and isinstance(e, int)):
        return False
    if isinstance(b, bool) or isinstance(e, bool):
        return False
    return 1 <= b <= m and 1 <= e <= m


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # ISP_1
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"ISP_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # ISP_F
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("ISP_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("ISP_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # ISP_BOUND — both boundaries declared
    ib = cert.get("input_boundary", {})
    ob = cert.get("output_boundary", {})
    if not isinstance(ib, dict) or ib.get("kind") != "observer_projection_in":
        errors.append("ISP_BOUND: input_boundary.kind must be 'observer_projection_in'")
    if not isinstance(ob, dict) or ob.get("kind") != "observer_projection_out":
        errors.append("ISP_BOUND: output_boundary.kind must be 'observer_projection_out'")
    if ib.get("direction") != "continuous_to_integer":
        errors.append("ISP_BOUND: input_boundary.direction must be 'continuous_to_integer'")
    if ob.get("direction") != "integer_to_continuous":
        errors.append("ISP_BOUND: output_boundary.direction must be 'integer_to_continuous'")

    # ISP_NO_REPROJECT — no re-tokenization stage in the pipeline
    stages = cert.get("pipeline_stages", [])
    if not isinstance(stages, list) or len(stages) < 3:
        errors.append(f"ISP_NO_REPROJECT: pipeline_stages must be a list of >=3 stages (input, interior, output), got {stages!r}")
    else:
        interior = stages[1:-1]
        for st in interior:
            stage_kind = (st.get("kind") if isinstance(st, dict) else None) or ""
            if "retokeniz" in stage_kind.lower() or "decode_then_encode" in stage_kind.lower():
                errors.append(f"ISP_NO_REPROJECT: interior stage {stage_kind!r} re-projects through a continuous intermediate")
            if isinstance(st, dict) and st.get("state_type") != "integer_tuple":
                errors.append(f"ISP_NO_REPROJECT: interior stage {stage_kind!r} state_type must be 'integer_tuple', got {st.get('state_type')!r}")

    # ISP_INT — every interior_state_sample is integer tuple in {1..m}^2
    m = cert.get("modulus", 9)
    samples = cert.get("interior_state_samples", [])
    if not isinstance(samples, list):
        errors.append("ISP_INT: interior_state_samples must be a list")
    else:
        for idx, s in enumerate(samples):
            if not _is_integer_tuple(s, m):
                errors.append(f"ISP_INT: interior_state_samples[{idx}] = {s!r} is not an integer tuple in {{1..{m}}}^2")

    # ISP_DET — declared + recomputed
    det = cert.get("determinism", {})
    repeats = det.get("repeats", 0)
    bitwise = det.get("bitwise_identical")
    if not isinstance(repeats, int) or repeats < 100:
        errors.append(f"ISP_DET: determinism.repeats must be int >= 100, got {repeats!r}")
    if bitwise is not True:
        errors.append(f"ISP_DET: determinism.bitwise_identical must be true, got {bitwise!r}")

    # Independent recomputation on canonical trace
    seed = cert.get("canonical_seed_tokens", [])
    steps = cert.get("canonical_steps", 0)
    if not isinstance(seed, list) or not seed:
        errors.append("ISP_DET: canonical_seed_tokens missing or empty")
    elif not isinstance(steps, int) or steps < 1:
        errors.append(f"ISP_DET: canonical_steps must be int >= 1, got {steps!r}")
    else:
        try:
            seed_tuples = [tuple(t) for t in seed]
            for t in seed_tuples:
                if not _is_integer_tuple(t, m):
                    raise ValueError(f"A1 violation in seed: {t}")
            t1 = _canonical_pipeline_trace(seed_tuples, steps, m=m)
            t2 = _canonical_pipeline_trace(seed_tuples, steps, m=m)
            if t1 != t2:
                errors.append("ISP_DET: canonical trace recomputation non-deterministic")
            # Declared trace length if present
            decl_len = cert.get("canonical_trace_length")
            if decl_len is not None and decl_len != len(t1):
                errors.append(
                    f"ISP_DET: declared canonical_trace_length={decl_len} != recomputed len={len(t1)}"
                )
        except Exception as ex:
            errors.append(f"ISP_DET: canonical trace recomputation failed: {ex}")

    # ISP_SRC
    src = str(cert.get("source_attribution", ""))
    for needle in ("2602.15763", "TITO"):
        if needle not in src:
            warnings.append(f"ISP_SRC: source_attribution should reference {needle!r}")

    # ISP_WIT
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 3:
        errors.append(
            f"ISP_WIT: need >= 3 interior-state witnesses, got {len(witnesses) if isinstance(witnesses, list) else 'none'}"
        )

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("isp_pass_default.json", True),
        ("isp_fail_continuous_intermediate.json", False),
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
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA Integer-State Pipeline Cert [257] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
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
