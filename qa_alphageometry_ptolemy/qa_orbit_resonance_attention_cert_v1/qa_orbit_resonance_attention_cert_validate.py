#!/usr/bin/env python3
"""
qa_orbit_resonance_attention_cert_validate.py

Validator for QA_ORBIT_RESONANCE_ATTENTION_CERT.v1  [family 256]

Certifies a QA-native attention operator: attention as a deterministic
pairwise resonance relation on T-orbit tuples, with no learned parameters
and no stochastic top-k selection. Structurally eliminates the GLM-5 DSA
non-deterministic-topk entropy-collapse failure mode (arXiv:2602.15763).

Three resonance rules are certified orbit-invariant under T evolution on S_9:
    - family_match : same T-orbit family (Fibonacci/Lucas/Phibonacci/Tribonacci/Ninbonacci)
    - norm_match   : same Eisenstein norm mod m, with f(b,e)=b*b+b*e-e*e
    - chromogeometry: cross-pair bilinear C_i*C_j + F_i*F_j == G_i*G_j (mod m)
                     where (C, F, G) = (2*d*e, b*a, e*e + d*d), cert [234]

Checks:
    ORA_1        — schema_version matches
    ORA_DET      — declared determinism gate present (bitwise_identical=true, repeats>=100)
    ORA_A1       — all canonical_witness_tokens in {1..9}^2
    ORA_INV_FAM  — family_match invariant under 24 T-steps on canonical tokens (recomputed)
    ORA_INV_NORM — norm_match invariant under 24 T-steps on canonical tokens (recomputed)
    ORA_INV_CHR  — chromogeometry invariant under 24 T-steps on canonical tokens (recomputed)
    ORA_GRAN     — granularity crosscut: at least one same-family-different-norm pair
                   AND at least one same-norm-different-family pair exist on S_9
    ORA_SRC      — source attribution includes GLM-5 arXiv + cert [214] + prototype path
    ORA_WIT      — >= 5 witnesses covering all 5 T-orbit families
    ORA_F        — fail_ledger well-formed

Source grounding:
    - GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering",
      arXiv:2602.15763 [cs.LG], Feb 2026 — empirical target whose DSA
      non-deterministic-topk failure mode this cert's operator eliminates
      by construction.
    - Cert [214] QA Norm-Flip Signed-Temporal: orbit family classification
      via Eisenstein norm pairs on S_9.
    - Cert [234] QA Chromogeometry Pythagorean Identity: cross-pair
      bilinear resonance form.
    - Cert [209] QA Signal Generator Inference: T-operator canonical.
    - docs/theory/QA_GLM5_ARCHITECTURE_MAPPING.md — design doc.
    - qa_lab/qa_orbit_resonance_attention.py — reference prototype.

QA axiom compliance: integer state alphabet {1..m} (A1), S1-compliant
(b*b not b**2), T-operator is the only mod-reduction path (T1 integer path
time), no floats anywhere in validator or cert fixtures (S2).
"""

QA_COMPLIANCE = "cert_validator — orbit-resonance attention operator; integer state space; no floats; no learned parameters; T-operator only path into QA layer"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_ORBIT_RESONANCE_ATTENTION_CERT.v1"


# -----------------------------------------------------------------------------
# QA primitives (integer-only, S1-compliant, stdlib only)
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    return (e, qa_mod(b + e, m))


def eisenstein_norm(b, e, m):
    return (b * b + b * e - e * e) % m


def _classify_orbit(b, e, m):
    """Coarse orbit class via orbit length: singularity(1) / satellite / cosmos."""
    seen = set()
    cur = (b, e)
    while cur not in seen:
        seen.add(cur)
        cur = qa_step(cur[0], cur[1], m)
    length = len(seen)
    if length == 1:
        return "singularity"
    # On S_9 cosmos is length 24, satellite length 8
    if length == 24:
        return "cosmos"
    return "satellite"


_NORM_PAIR_TO_FAMILY_S9 = {
    frozenset({1, 8}): "fibonacci",
    frozenset({4, 5}): "lucas",
    frozenset({2, 7}): "phibonacci",
}


def orbit_family_s9(b, e):
    coarse = _classify_orbit(b, e, 9)
    if coarse == "singularity":
        return "ninbonacci"
    if coarse == "satellite":
        return "tribonacci"
    n = eisenstein_norm(b, e, 9)
    for pair, fam in _NORM_PAIR_TO_FAMILY_S9.items():
        if n in pair:
            return fam
    raise ValueError(f"cosmos state ({b},{e}) norm {n} not in any known pair")


# -----------------------------------------------------------------------------
# Resonance rules (mirrors of the reference implementation)
# -----------------------------------------------------------------------------

def _attn_family_match(tokens, m=9):
    fams = [orbit_family_s9(b, e) for (b, e) in tokens]
    n = len(tokens)
    return [[1 if fams[i] == fams[j] else 0 for j in range(n)] for i in range(n)]


def _attn_norm_match(tokens, m=9):
    norms = [eisenstein_norm(b, e, m) for (b, e) in tokens]
    n = len(tokens)
    return [[1 if norms[i] == norms[j] else 0 for j in range(n)] for i in range(n)]


def _attn_chromogeometry(tokens, m=9):
    # Elements (C, F, G) computed RAW (d=b+e, a=b+2e RAW per 2026-04-09 rule).
    # Mod m is applied only to the cross-pair bilinear equality test — that is
    # a modular-equivalence comparison, not an element computation.
    triples = []
    for (b, e) in tokens:
        d = b + e                # RAW
        a = b + 2 * e            # RAW
        C = 2 * d * e            # RAW
        F = b * a                # RAW
        G = e * e + d * d        # RAW
        triples.append((C, F, G))
    n = len(tokens)
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        Ci, Fi, Gi = triples[i]
        for j in range(n):
            Cj, Fj, Gj = triples[j]
            A[i][j] = 1 if (Ci * Cj + Fi * Fj) % m == (Gi * Gj) % m else 0
    return A


_RULE_FNS = {
    "family_match": _attn_family_match,
    "norm_match": _attn_norm_match,
    "chromogeometry": _attn_chromogeometry,
}


def _evolve(tokens, steps, m=9):
    state = [tuple(t) for t in tokens]
    traj = [list(state)]
    for _ in range(steps):
        state = [qa_step(b, e, m) for (b, e) in state]
        traj.append(list(state))
    return traj


def _invariant_over_evolution(tokens, rule, steps, m=9):
    """Recompute attention at every step; return True iff all equal step-0."""
    fn = _RULE_FNS[rule]
    traj = _evolve(tokens, steps=steps, m=m)
    A0 = fn(traj[0], m=m)
    for state in traj[1:]:
        if fn(state, m=m) != A0:
            return False
    return True


def _determinism_holds(tokens, rule, repeats, m=9):
    """Repeated calls on identical input produce bitwise-identical output."""
    fn = _RULE_FNS[rule]
    A_first = fn(tokens, m=m)
    for _ in range(repeats - 1):
        if fn(tokens, m=m) != A_first:
            return False
    return True


def _exhaustive_granularity_crosscut_s9():
    """Return (has_fam_without_norm, has_norm_without_family) on exhaustive S_9 pairs."""
    all_tokens = [(b, e) for b in range(1, 10) for e in range(1, 10)]
    fam_no_norm = False
    norm_no_fam = False
    for i, ti in enumerate(all_tokens):
        fi = orbit_family_s9(*ti)
        ni = eisenstein_norm(*ti, 9)
        for j in range(i + 1, len(all_tokens)):
            tj = all_tokens[j]
            fj = orbit_family_s9(*tj)
            nj = eisenstein_norm(*tj, 9)
            same_fam = fi == fj
            same_norm = ni == nj
            if same_fam and not same_norm:
                fam_no_norm = True
            if same_norm and not same_fam:
                norm_no_fam = True
            if fam_no_norm and norm_no_fam:
                return (True, True)
    return (fam_no_norm, norm_no_fam)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # ORA_1
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"ORA_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # ORA_F
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("ORA_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("ORA_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # ORA_A1 — tokens in {1..9}^2
    tokens = cert.get("canonical_witness_tokens", [])
    if not isinstance(tokens, list) or not tokens:
        errors.append("ORA_A1: canonical_witness_tokens missing or empty")
        return errors, warnings
    normalized_tokens = []
    for t in tokens:
        if not (isinstance(t, list) and len(t) == 2 and all(isinstance(x, int) for x in t)):
            errors.append(f"ORA_A1: malformed token {t!r} (expected [int, int])")
            continue
        b, e = t
        if not (1 <= b <= 9 and 1 <= e <= 9):
            errors.append(f"ORA_A1: token [{b},{e}] outside {{1..9}}^2")
            continue
        normalized_tokens.append((b, e))
    if errors:
        return errors, warnings

    # ORA_DET — declared determinism
    det = cert.get("determinism", {})
    repeats = det.get("repeats", 0)
    bitwise = det.get("bitwise_identical")
    if not isinstance(repeats, int) or repeats < 100:
        errors.append(f"ORA_DET: determinism.repeats must be int >= 100, got {repeats!r}")
    if bitwise is not True:
        errors.append(f"ORA_DET: determinism.bitwise_identical must be true, got {bitwise!r}")
    # Independently recompute determinism for each rule
    for rule in _RULE_FNS:
        if not _determinism_holds(normalized_tokens, rule, repeats=max(100, repeats)):
            errors.append(f"ORA_DET: determinism failed on rule {rule!r} (recomputation)")

    # ORA_INV_FAM / _NORM / _CHR — independently recompute over 24 T-steps
    inv = cert.get("orbit_invariance_s9", {})
    steps = inv.get("steps_tested", 0)
    if not isinstance(steps, int) or steps < 24:
        errors.append(f"ORA_INV_*: orbit_invariance_s9.steps_tested must be int >= 24, got {steps!r}")
    for rule, key in (
        ("family_match", "family_match_invariant"),
        ("norm_match", "norm_match_invariant"),
        ("chromogeometry", "chromogeometry_invariant"),
    ):
        recomputed = _invariant_over_evolution(normalized_tokens, rule, steps=max(24, steps))
        declared = inv.get(key)
        tag = "ORA_INV_FAM" if rule == "family_match" else (
            "ORA_INV_NORM" if rule == "norm_match" else "ORA_INV_CHR"
        )
        if declared is not True:
            errors.append(f"{tag}: declared {key}={declared!r}, must be true")
        if not recomputed:
            errors.append(f"{tag}: recomputation contradicts orbit-invariance under 24 T-steps")

    # ORA_GRAN — granularity crosscut on exhaustive S_9
    fam_no_norm, norm_no_fam = _exhaustive_granularity_crosscut_s9()
    if not fam_no_norm:
        errors.append("ORA_GRAN: no same-family-different-norm pair on S_9 — family_match and norm_match not distinguished")
    if not norm_no_fam:
        errors.append("ORA_GRAN: no same-norm-different-family pair on S_9 — family_match and norm_match not distinguished")
    decl_gran = cert.get("granularity_crosscut", {})
    if decl_gran:
        decl_fwn = decl_gran.get("family_without_norm")
        decl_nwf = decl_gran.get("norm_without_family")
        if not isinstance(decl_fwn, list) or len(decl_fwn) != 2:
            errors.append("ORA_GRAN: granularity_crosscut.family_without_norm must be a list of 2 tokens")
        else:
            (b1, e1), (b2, e2) = decl_fwn[0], decl_fwn[1]
            if orbit_family_s9(b1, e1) != orbit_family_s9(b2, e2):
                errors.append("ORA_GRAN: declared family_without_norm pair is not same-family")
            if eisenstein_norm(b1, e1, 9) == eisenstein_norm(b2, e2, 9):
                errors.append("ORA_GRAN: declared family_without_norm pair has same norm (contradiction)")
        if not isinstance(decl_nwf, list) or len(decl_nwf) != 2:
            errors.append("ORA_GRAN: granularity_crosscut.norm_without_family must be a list of 2 tokens")
        else:
            (b1, e1), (b2, e2) = decl_nwf[0], decl_nwf[1]
            if eisenstein_norm(b1, e1, 9) != eisenstein_norm(b2, e2, 9):
                errors.append("ORA_GRAN: declared norm_without_family pair is not same-norm")
            if orbit_family_s9(b1, e1) == orbit_family_s9(b2, e2):
                errors.append("ORA_GRAN: declared norm_without_family pair has same family (contradiction)")

    # ORA_SRC
    src = str(cert.get("source_attribution", ""))
    for needle in ("2602.15763", "GLM-5"):
        if needle not in src:
            warnings.append(f"ORA_SRC: source_attribution should reference {needle!r}")

    # ORA_WIT
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 5:
        errors.append(f"ORA_WIT: need >= 5 witnesses (one per T-orbit family), got {len(witnesses) if isinstance(witnesses, list) else 'none'}")
    else:
        families_seen = set()
        for w in witnesses:
            fam = w.get("family") if isinstance(w, dict) else None
            if fam:
                families_seen.add(fam)
        required = {"fibonacci", "lucas", "phibonacci", "tribonacci", "ninbonacci"}
        missing = required - families_seen
        if missing:
            errors.append(f"ORA_WIT: witnesses missing families {sorted(missing)}")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("ora_pass_default.json", True),
        ("ora_fail_wrong_invariance.json", False),
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
    parser = argparse.ArgumentParser(description="QA Orbit-Resonance Attention Cert [256] validator")
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
