#!/usr/bin/env python3
"""
qa_generator_pattern_training_cert_validate.py

Validator for QA_GENERATOR_PATTERN_TRAINING_CERT.v1  [family 258]

Certifies QA-native training: discrete-search identification of an integer
generator pattern on integer path time. No gradients, no float optimizer
state, no importance sampling, no staleness bounds — the trained object is
an integer tuple (b_0, e_0) whose T-orbit best matches a target trace
under exact-match count.

Checks:
    GPT_1           — schema_version matches
    GPT_INT_GEN     — declared and recomputed trained_generator is an integer
                      tuple (b,e) in {1..m}^2
    GPT_NO_GRAD     — training_path.uses_gradients must be false;
                      optimizer.kind must be 'discrete_search'
    GPT_DISCRETE    — training_path.search_space_size equals m^2 (81 on S_9);
                      training_path.steps_enumerated within a declared bound
    GPT_ORBIT_EVAL  — evaluation.metric must be 'exact_match_count' or similar
                      integer-valued orbit-match metric
    GPT_DET         — repeated training on identical target produces bitwise-identical
                      trained_generator (independently recomputed)
    GPT_SRC         — source attribution includes GLM-5 arXiv + cert [209] + [256]
    GPT_WIT         — >= 5 (target_trace, trained_generator) witnesses covering
                      all five T-orbit families
    GPT_F           — fail_ledger well-formed

Source grounding:
    - GLM-5 Team, arXiv:2602.15763 [cs.LG], Feb 2026 — §4.1.2 async RL
      policy staleness, IS corrections, optimizer reset (the failure modes
      eliminated here by construction).
    - Cert [209] QA Signal Generator Inference: e_t = ((b_{t+1}-b_t-1) % m)+1 —
      the canonical integer generator whose identification IS training.
    - Cert [256] QA Orbit-Resonance Attention: uses the identified generator
      as its operating pattern.
    - Cert [257] QA Integer-State Pipeline: training stays inside the
      two-boundary invariant.
    - docs/theory/QA_GLM5_ARCHITECTURE_MAPPING.md — design doc.
    - qa_lab/qa_orbit_resonance_attention.py — reference implementation
      (identify_generator, identify_family).

QA axiom compliance: integer state alphabet {1..m} (A1), integer path time
(T1), no float state crossing discrete transitions (S2), S1-compliant.
"""

QA_COMPLIANCE = "cert_validator — generator-pattern training as discrete search over integer generator space; no gradients; no float optimizer state; evaluation by exact orbit-trace match"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_GENERATOR_PATTERN_TRAINING_CERT.v1"


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    return (e, qa_mod(b + e, m))


def _identify_generator(target_trace, m=9):
    """Vendored reference implementation — matches qa_lab prototype."""
    if not target_trace:
        raise ValueError("target_trace must be non-empty")
    n = len(target_trace)
    target = [tuple(t) for t in target_trace]
    best_start = (1, 1)
    best_score = -1
    for b0 in range(1, m + 1):
        for e0 in range(1, m + 1):
            state = (b0, e0)
            score = 0
            for k in range(n):
                if state == target[k]:
                    score += 1
                state = qa_step(state[0], state[1], m)
            if score > best_score:
                best_score = score
                best_start = (b0, e0)
    return best_start, best_score


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

    # GPT_1
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"GPT_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # GPT_F
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("GPT_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("GPT_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    m = cert.get("modulus", 9)

    # GPT_INT_GEN — declared trained_generator is integer tuple
    tg = cert.get("trained_generator")
    if not _is_integer_tuple(tg, m):
        errors.append(f"GPT_INT_GEN: trained_generator={tg!r} must be integer tuple in {{1..{m}}}^2")

    # GPT_NO_GRAD — gradients and optimizer kind
    tp = cert.get("training_path", {})
    uses_grad = tp.get("uses_gradients")
    if uses_grad is not False:
        errors.append(f"GPT_NO_GRAD: training_path.uses_gradients must be false, got {uses_grad!r}")
    opt = cert.get("optimizer", {})
    if opt.get("kind") != "discrete_search":
        errors.append(f"GPT_NO_GRAD: optimizer.kind must be 'discrete_search', got {opt.get('kind')!r}")
    # Must not carry float state
    float_state = opt.get("float_state")
    if float_state not in (None, [], False):
        errors.append(f"GPT_NO_GRAD: optimizer.float_state must be empty/false, got {float_state!r}")

    # GPT_DISCRETE — search space matches m^2
    ss_size = tp.get("search_space_size")
    expected_size = m * m
    if ss_size != expected_size:
        errors.append(f"GPT_DISCRETE: training_path.search_space_size must be {expected_size} (m*m), got {ss_size!r}")
    steps_enum = tp.get("steps_enumerated")
    if not isinstance(steps_enum, int) or steps_enum < 1 or steps_enum > expected_size:
        errors.append(f"GPT_DISCRETE: training_path.steps_enumerated must be int in [1, {expected_size}], got {steps_enum!r}")

    # GPT_ORBIT_EVAL — evaluation metric
    ev = cert.get("evaluation", {})
    metric = ev.get("metric", "")
    allowed_metrics = {"exact_match_count", "orbit_trace_hamming", "family_plurality"}
    if metric not in allowed_metrics:
        errors.append(f"GPT_ORBIT_EVAL: evaluation.metric must be one of {sorted(allowed_metrics)}, got {metric!r}")
    # Scalar loss forbidden
    if ev.get("scalar_loss_used") is True:
        errors.append("GPT_ORBIT_EVAL: evaluation.scalar_loss_used must not be true")

    # GPT_DET — recomputed training result
    target = cert.get("canonical_target_trace", [])
    if not isinstance(target, list) or not target:
        errors.append("GPT_DET: canonical_target_trace missing or empty")
    else:
        try:
            for t in target:
                if not _is_integer_tuple(t, m):
                    raise ValueError(f"A1 violation in target: {t}")
            r1, s1 = _identify_generator(target, m=m)
            r2, s2 = _identify_generator(target, m=m)
            if (r1, s1) != (r2, s2):
                errors.append("GPT_DET: identify_generator non-deterministic under recomputation")
            if tg is not None and _is_integer_tuple(tg, m):
                if tuple(tg) != r1:
                    errors.append(
                        f"GPT_DET: declared trained_generator={tuple(tg)} != recomputed {r1}"
                    )
            decl_score = cert.get("declared_match_score")
            if decl_score is not None and decl_score != s1:
                errors.append(f"GPT_DET: declared_match_score={decl_score} != recomputed {s1}")
        except Exception as ex:
            errors.append(f"GPT_DET: training recomputation failed: {ex}")

    # GPT_SRC
    src = str(cert.get("source_attribution", ""))
    for needle in ("2602.15763", "[209]", "[256]"):
        if needle not in src:
            warnings.append(f"GPT_SRC: source_attribution should reference {needle!r}")

    # GPT_WIT — >= 5 witnesses covering all 5 T-orbit families
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 5:
        errors.append(f"GPT_WIT: need >= 5 (target,generator) witnesses, got {len(witnesses) if isinstance(witnesses, list) else 'none'}")
    else:
        families_seen = set()
        for w in witnesses:
            fam = w.get("family") if isinstance(w, dict) else None
            if fam:
                families_seen.add(fam)
        required = {"fibonacci", "lucas", "phibonacci", "tribonacci", "ninbonacci"}
        missing = required - families_seen
        if missing:
            errors.append(f"GPT_WIT: witnesses missing families {sorted(missing)}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("gpt_pass_default.json", True),
        ("gpt_fail_gradient_optimizer.json", False),
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
    parser = argparse.ArgumentParser(description="QA Generator-Pattern Training Cert [258] validator")
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
