#!/usr/bin/env python3
"""qa_runtime_odd_monitor_cert_validate.py

Validator for QA_RUNTIME_ODD_MONITOR_CERT.v1  [family 264]

Second sharp-claim cert derived from the Kochenderfer 2026 'Algorithms for
Validation' bridge (Kochenderfer, 2026; docs/specs/QA_KOCHENDERFER_BRIDGE.md
§5 ODD-monitor row). Anchored at Kochenderfer Validation §12.1 'Operational
Design Domain Monitoring' superlevel-set ODD construction.

CLAIM (narrow): For QA finite orbit-class regimes on S_9, ODD membership can
be monitored by deterministic orbit-family membership with FP=0 and FN=0
relative to the declared discrete ODD. On continuous observer projections
(b, e) → ((b-1)/8, (e-1)/8) ∈ [0,1]² of QA-discrete inputs, a
Kochenderfer-style classifier-superlevel-set baseline (Validation §12.1)
produces non-zero classification error near orbit-class boundaries that
scales with input-noise σ.

Claim does NOT generalize to all runtime ODD monitoring; claim does NOT say
continuous classifiers are bad globally. The exactness holds only on the
QA-discrete side of the Theorem NT firewall.

Source attribution: (Kochenderfer, 2026) Algorithms for Validation §12.1
ODD monitoring; (Dale, 2026) cert [263] qa_failure_density_enumeration_cert
+ tools/qa_kg/orbit_failure_enumeration.py utility.

Checks:
    ODD_1     — schema_version matches
    ODD_DECL  — declared_odd is a non-empty subset of {singularity, satellite, cosmos}
    ODD_EXACT — deterministic monitor: enumerate all 81 states, FP=0 and FN=0
                relative to declared ODD (trivially true since the monitor IS
                the orbit-family-membership check; verified by independent
                recomputation against utility.orbit_family_s9)
    ODD_CLF   — for each declared (sigma, seed) test case, run 1-NN
                classifier baseline on (b,e)→((b-1)/8,(e-1)/8) noisy
                projection; declared expected_classifier_fp /
                expected_classifier_fn match recomputation bit-exactly
                (seeded determinism)
    ODD_LEAK  — for at least one declared (sigma, seed) case, the classifier
                baseline produces non-zero classification error
                (FP + FN > 0); demonstrates the leakage claim
    ODD_SRC   — source_attribution cites Kochenderfer + cert [263]
    ODD_WIT   — at least 3 witnesses (one per orbit class)
    ODD_F     — fail_ledger well-formed

Theorem NT compliance: deterministic monitor stays integer-only (orbit_family_s9
returns string label from integer (b,e)). Classifier baseline operates on
continuous embedding — declared as observer projection at the input boundary;
single noise injection via random.Random(seed).gauss is the only continuous
operation, and the comparison metrics (FP/FN counts) are integer-valued.
"""

QA_COMPLIANCE = "cert_validator — verifies deterministic orbit-family ODD membership (FP=0, FN=0) vs Kochenderfer §12.1 classifier-superlevel-set baseline; classifier helper is observer projection at input boundary (single random.Random.gauss noise injection per (state, seed)); QA state path stays integer; FP/FN counts integer-valued"

import json
import math
import random
import sys
from pathlib import Path

# Make the repo root importable so tools/qa_kg/orbit_failure_enumeration.py
# is reachable regardless of CWD when meta_validator runs us. Mirrors the
# pattern used by certs [263] and [194].
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.qa_kg.orbit_failure_enumeration import (  # noqa: E402
    ORBIT_CLASSES,
    orbit_family_s9,
    qa_mod,
    qa_step,
)

SCHEMA_VERSION = "QA_RUNTIME_ODD_MONITOR_CERT.v1"


# -----------------------------------------------------------------------------
# Continuous observer-projection embedding (input boundary; Theorem NT firewall
# crosses HERE — once at projection-in, once at FP/FN-count-out).
# -----------------------------------------------------------------------------

def embed_state(b: int, e: int) -> tuple[float, float]:
    """Canonical observer-projection embedding (b, e) → ((b-1)/8, (e-1)/8) ∈ [0,1]².

    This is the OUTPUT-side observer projection of the QA-discrete state space
    onto a continuous 2D embedding, used only to construct the Kochenderfer
    classifier-superlevel-set baseline. Returns floats; not allowed back into
    QA-discrete state evolution (Theorem NT firewall).
    """
    return ((int(b) - 1) / 8.0, (int(e) - 1) / 8.0)


def deterministic_in_odd(b: int, e: int, declared_odd: list[str]) -> bool:
    """Deterministic orbit-class ODD membership monitor.

    Returns True iff orbit_family_s9(b, e) ∈ declared_odd. Integer-only;
    no noise; FP=0 and FN=0 by construction relative to the declared ODD.
    """
    return orbit_family_s9(int(b), int(e)) in declared_odd


def _classifier_baseline_predict_label(
    test_embed: tuple[float, float],
    training_set: list[tuple[tuple[float, float], str]],
) -> str:
    """1-nearest-neighbor classifier baseline on continuous embedding.

    Per Kochenderfer Validation §12.1, the ODD is the superlevel set of a
    classifier trained on validated regime data. We use 1-NN on the
    noiseless training embedding for simplicity — a cleaner baseline would
    use a probability classifier with threshold > 0.5 (matching Kochenderfer
    §12.1 exactly), but 1-NN suffices to establish the leakage claim and
    keeps the cert dependency-light.
    """
    best_dist_sq = float("inf")
    best_label = ""
    for train_embed, train_label in training_set:
        d2 = (test_embed[0] - train_embed[0]) ** 2 + (test_embed[1] - train_embed[1]) ** 2
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_label = train_label
    return best_label


def classifier_in_odd_baseline(
    declared_odd: list[str],
    sigma: float,
    seed: int,
) -> dict:
    """Run the Kochenderfer-style classifier-superlevel-set baseline on a
    noisy projection of S_9 at given sigma + seed. Returns FP/FN counts.

    Procedure:
    1. Build training set: for each (b,e) ∈ {1..9}², compute clean embedding
       and orbit-family label → 81 (clean_embed, label) pairs.
    2. For each (b,e), inject Gaussian noise σ at given seed → noisy_embed.
    3. Predict label via 1-NN against training set.
    4. Compare predicted_in_odd (predicted_label ∈ declared_odd) to
       true_in_odd (orbit_family_s9(b,e) ∈ declared_odd).
    5. Count FP (predicted in, true out) + FN (predicted out, true in).

    Returns: {'fp': int, 'fn': int, 'total': int (=81), 'sigma': float, 'seed': int,
              'fp_witnesses': list of (b,e) where FP, 'fn_witnesses': list of (b,e) where FN}
    """
    training_set = [
        (embed_state(b, e), orbit_family_s9(b, e))
        for b in range(1, 10)
        for e in range(1, 10)
    ]
    rng = random.Random(int(seed))  # noqa: T2-D — seeded observer-projection noise at input boundary
    fp = 0
    fn = 0
    fp_witnesses: list[tuple[int, int]] = []
    fn_witnesses: list[tuple[int, int]] = []
    for b in range(1, 10):
        for e in range(1, 10):
            clean = embed_state(b, e)
            noisy = (
                clean[0] + rng.gauss(0.0, float(sigma)),
                clean[1] + rng.gauss(0.0, float(sigma)),
            )
            predicted_label = _classifier_baseline_predict_label(noisy, training_set)
            true_label = orbit_family_s9(b, e)
            predicted_in = predicted_label in declared_odd
            true_in = true_label in declared_odd
            if predicted_in and not true_in:
                fp += 1
                fp_witnesses.append((b, e))
            elif not predicted_in and true_in:
                fn += 1
                fn_witnesses.append((b, e))
    return {
        "fp": int(fp),
        "fn": int(fn),
        "total": 81,
        "sigma": float(sigma),
        "seed": int(seed),
        "fp_witnesses": fp_witnesses,
        "fn_witnesses": fn_witnesses,
    }


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors: list[str] = []
    warnings: list[str] = []

    # ODD_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(
            f"ODD_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}"
        )

    # ODD_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("ODD_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("ODD_F: fail_ledger must be a list")

    # Documented FAIL fixtures short-circuit gates after schema + ledger
    # (mirrors cert [194] / [263] convention).
    if cert.get("result") == "FAIL":
        return errors, warnings

    # ODD_DECL: declared_odd is a non-empty subset of ORBIT_CLASSES
    declared_odd = cert.get("declared_odd")
    if not isinstance(declared_odd, list) or not declared_odd:
        errors.append(
            f"ODD_DECL: declared_odd must be non-empty list; got {declared_odd!r}"
        )
    else:
        for cls in declared_odd:
            if cls not in ORBIT_CLASSES:
                errors.append(
                    f"ODD_DECL: declared_odd entry {cls!r} not in {ORBIT_CLASSES!r}"
                )

    # ODD_SRC: source attribution must mention Kochenderfer AND cert [263]
    src = str(cert.get("source_attribution", ""))
    if "Kochenderfer" not in src:
        errors.append(
            "ODD_SRC: source_attribution must cite Kochenderfer 2026 "
            "Algorithms for Validation §12.1"
        )
    if "263" not in src:
        errors.append(
            "ODD_SRC: source_attribution must cite cert [263] "
            "qa_failure_density_enumeration_cert (utility provider)"
        )

    # ODD_EXACT: deterministic monitor reproduces declared FP=0, FN=0 over S_9.
    if isinstance(declared_odd, list) and declared_odd:
        decl_exact = cert.get("exact_membership", {})
        if not isinstance(decl_exact, dict):
            errors.append("ODD_EXACT: exact_membership section missing/malformed")
        else:
            recomputed_fp = 0
            recomputed_fn = 0
            for b in range(1, 10):
                for e in range(1, 10):
                    monitor_says_in = deterministic_in_odd(b, e, declared_odd)
                    truth_in = orbit_family_s9(b, e) in declared_odd
                    if monitor_says_in and not truth_in:
                        recomputed_fp += 1
                    elif not monitor_says_in and truth_in:
                        recomputed_fn += 1
            decl_fp = int(decl_exact.get("fp", -1))
            decl_fn = int(decl_exact.get("fn", -1))
            if decl_fp != recomputed_fp:
                errors.append(
                    f"ODD_EXACT: declared deterministic fp={decl_fp}, recomputed {recomputed_fp}"
                )
            if decl_fn != recomputed_fn:
                errors.append(
                    f"ODD_EXACT: declared deterministic fn={decl_fn}, recomputed {recomputed_fn}"
                )
            if recomputed_fp != 0 or recomputed_fn != 0:
                errors.append(
                    f"ODD_EXACT: deterministic monitor produced fp={recomputed_fp}, "
                    f"fn={recomputed_fn} on S_9 (must be 0/0 by construction; "
                    f"orbit_family_s9 broken?)"
                )

    # ODD_CLF + ODD_LEAK: classifier-superlevel-set baseline at declared
    # (sigma, seed) test cases.
    classifier_cases = cert.get("classifier_test_cases", [])
    if not isinstance(classifier_cases, list) or len(classifier_cases) < 1:
        errors.append(
            "ODD_CLF: need >= 1 classifier_test_cases entries (sigma, seed pairs); "
            f"got {classifier_cases!r}"
        )
    else:
        any_leak_observed = False
        for i, case in enumerate(classifier_cases):
            sigma = case.get("sigma")
            seed = case.get("seed")
            if sigma is None or seed is None:
                errors.append(
                    f"ODD_CLF[{i}]: sigma and seed required (got sigma={sigma!r}, seed={seed!r})"
                )
                continue
            if isinstance(declared_odd, list) and declared_odd:
                recomputed = classifier_in_odd_baseline(declared_odd, float(sigma), int(seed))
                decl_fp = case.get("expected_classifier_fp")
                decl_fn = case.get("expected_classifier_fn")
                if decl_fp is not None and int(decl_fp) != recomputed["fp"]:
                    errors.append(
                        f"ODD_CLF[{i}]: declared expected_classifier_fp={decl_fp}, "
                        f"recomputed {recomputed['fp']} (seed={seed} not reproducible?)"
                    )
                if decl_fn is not None and int(decl_fn) != recomputed["fn"]:
                    errors.append(
                        f"ODD_CLF[{i}]: declared expected_classifier_fn={decl_fn}, "
                        f"recomputed {recomputed['fn']}"
                    )
                if recomputed["fp"] + recomputed["fn"] > 0:
                    any_leak_observed = True
        if not any_leak_observed:
            errors.append(
                "ODD_LEAK: no declared (sigma, seed) case produced classifier "
                "FP+FN > 0; the leakage claim requires at least one case where "
                "the continuous classifier baseline misclassifies"
            )

    # ODD_WIT: at least 3 witnesses, one per orbit class
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 3:
        errors.append(
            f"ODD_WIT: need >= 3 witnesses (one per orbit class), got "
            f"{len(witnesses) if isinstance(witnesses, list) else 'malformed'}"
        )
    else:
        seen_classes = {w.get("orbit_class") for w in witnesses}
        for cls in ORBIT_CLASSES:
            if cls not in seen_classes:
                errors.append(f"ODD_WIT: missing witness for class {cls!r}")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("pass_s9_declared_odd_exact_membership.json", True),
        ("pass_classifier_boundary_comparison.json", True),
        ("fail_bad_odd_label.json", True),
        ("fail_continuous_boundary_leakage.json", True),
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
            results.append({
                "fixture": fname, "ok": False,
                "error": f"expected PASS but got errors: {errs}",
            })
            all_ok = False
        elif not should_pass and passed:
            results.append({
                "fixture": fname, "ok": False,
                "error": "expected FAIL but got PASS",
            })
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})
    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Runtime ODD Monitor Cert [264] validator")
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
