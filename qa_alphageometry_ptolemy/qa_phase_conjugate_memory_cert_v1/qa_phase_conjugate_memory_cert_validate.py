"""
QA Phase-Conjugate Holographic Associative Memory Cert [519]

Primary sources:
  (Soffer, Dunning, Owechko, Marom, 1986) "Associative holographic memory with
    feedback using phase-conjugate mirrors." Opt. Lett. 11(2):118-120.
    DOI 10.1364/OL.11.000118
  (Owechko, 1987) "Nonlinear holographic associative memories." IEEE J. Quantum
    Electron. 25(3):619-634.
  (Hopfield, 1982) "Neural networks and physical systems..." PNAS 79(8):2554-2558.
    DOI 10.1073/pnas.79.8.2554
  (Levin, 2021) target morphology as an attractor reached from perturbed starts.

CLAIM: the exact QA phase conjugator of cert [518] composes into a
content-addressable associative memory. Patterns/probes are phase vectors in
{1,...,m}^N. The phase-conjugate overlap of probe x with stored pattern k is
    C_k = #{ i : qa_add(x_i, qa_neg(P_k_i)) == m }      (== phase-match count)
and recall is holographic playback: reconstruct each site by superposing stored
patterns weighted by sharpened overlap (argmax phase), iterated to a fixed point.
For sufficiently separated patterns (demonstrated empirically, NOT guaranteed for
arbitrary/near-identical memories) stored patterns are fixed points and a
corrupted probe flows to the nearest. A global phase screen breaks naive recall
but PHASE-LOCKED recall -- scan the global compensation phase that maximizes
overlap (the [518] distortion correction), recall in that frame, shift back --
recovers the pattern in the probe's (distorted) frame.

Checks (deterministic, integer-only, pure stdlib):
  PC_OVERLAP            overlap via qa_add(x,qa_neg(P))==m equals the match count
  FIXED_POINTS          stored patterns recall to themselves exactly
  RECALL                a corrupted probe recalls its exact stored pattern
  CONTENT_ADDRESSABLE   probe nearest to pattern s recalls exactly s
  DISTORTION_TOLERANT   phase-locked recall recovers through a global phase screen
  NAIVE_DISTORTION_FAILS naive recall does NOT recover through the screen (control)
  A1_RANGE              every output lies in {1,...,m}
  SRC / F               mapping_protocol_ref present; fixtures behave as declared

Builds on cert [518] (the exact FWM conjugator).
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

QA_COMPLIANCE = (
    "cert_validator -- integer phase arithmetic on the QA additive group over "
    "{1,...,m} (identity=m, never 0); correlation/vote scores are float "
    "observer-layer quantities, never fed back as QA state (Theorem NT). "
    "Composes cert [518]'s exact FWM conjugator."
)

SCHEMA_VERSION = "QA_PHASE_CONJUGATE_MEMORY_CERT.v1"
FAMILY_ID = 519
SLUG = "qa_phase_conjugate_memory_cert_v1"

REQUIRED_FIELDS = {
    "schema_version", "fixture_kind", "primary_source",
    "m", "patterns", "probe", "expected_recall",
}


# ---------------------------------------------------------------------------
# QA phase algebra + memory (pure stdlib)
# ---------------------------------------------------------------------------
def qa_mod(x: int, m: int) -> int:
    return ((int(x) - 1) % m) + 1


def qa_add(a: int, b: int, m: int) -> int:
    return qa_mod(a + b, m)


def qa_neg(a: int, m: int) -> int:
    return qa_mod(-a, m)


def overlap(x: List[int], pat: List[int], m: int) -> int:
    """Phase-conjugate overlap: #{i : qa_add(x_i, qa_neg(pat_i)) == m}."""
    return sum(1 for xi, pi in zip(x, pat) if qa_add(xi, qa_neg(pi, m), m) == m)


def _playback(x: List[int], P: List[List[int]], m: int, sharpen: float) -> List[int]:
    N = len(x)
    weights = [(overlap(x, pat, m) / N) ** sharpen for pat in P]  # observer-layer
    if sum(weights) <= 0:
        return list(x)
    out = []
    for i in range(N):
        votes = {}
        for k, pat in enumerate(P):
            votes[pat[i]] = votes.get(pat[i], 0.0) + weights[k]
        out.append(max(sorted(votes), key=lambda v: votes[v]))  # argmax, tie->low phase
    return out


def recall(probe: List[int], P: List[List[int]], m: int,
           sharpen: float = 6.0, iters: int = 25) -> List[int]:
    x = [qa_mod(v, m) for v in probe]
    for _ in range(iters):
        nxt = _playback(x, P, m, sharpen)
        if nxt == x:
            break
        x = nxt
    return x


def recall_phase_locked(probe: List[int], P: List[List[int]], m: int,
                        sharpen: float = 6.0, iters: int = 25) -> List[int]:
    probe = [qa_mod(v, m) for v in probe]
    best_psi, best_score = m, -1
    for psi in range(1, m + 1):
        shifted = [qa_add(v, psi, m) for v in probe]
        score = max(overlap(shifted, pat, m) for pat in P)
        if score > best_score:
            best_score, best_psi = score, psi
    shifted = [qa_add(v, best_psi, m) for v in probe]
    rec = recall(shifted, P, m, sharpen=sharpen, iters=iters)
    return [qa_add(v, qa_neg(best_psi, m), m) for v in rec]


def _in_range(seq, m: int) -> bool:
    return all(isinstance(v, int) and 1 <= v <= m for v in seq)


# ---------------------------------------------------------------------------
# Fixture checks
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
    P = data["patterns"]
    probe = data["probe"]
    exp = data["expected_recall"]
    N = len(P[0])
    if any(len(p) != N for p in P) or len(probe) != N or len(exp) != N:
        return "OUT_OF_RANGE: length mismatch among patterns/probe/expected_recall"
    for seq in P + [probe, exp]:
        if not _in_range(seq, m):
            return f"OUT_OF_RANGE: value not in 1..{m}"
    phi = data.get("global_phi")
    if phi is None:
        rec = recall(probe, P, m)
    else:
        rec = recall_phase_locked(probe, P, m)
    if rec != exp:
        return f"WRONG_RECALL: recall={rec} != expected_recall={exp}"
    return None


def _check_fail_fixture(data: dict) -> Optional[str]:
    if "expected_fail_type" not in data:
        return "FAIL_FIXTURE_MISSING_expected_fail_type"
    ft = data["expected_fail_type"]
    if ft not in ("MISSING_FIELD", "WRONG_RECALL", "OUT_OF_RANGE"):
        return f"UNKNOWN_expected_fail_type: {ft}"
    if ft == "MISSING_FIELD":
        if not (REQUIRED_FIELDS - set(data.keys())):
            return "FAIL_FIXTURE_DID_NOT_FAIL: expected MISSING_FIELD but all present"
        return None
    if REQUIRED_FIELDS - set(data.keys()):
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {ft} but a required field is missing"
    err = _check_pass_fixture({**data, "fixture_kind": "pass"})
    if err is None:
        return f"FAIL_FIXTURE_DID_NOT_FAIL: expected {ft} but fixture is consistent"
    if not err.startswith(ft):
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {ft} but got {err}"
    return None


def validate_fixture(path: str) -> tuple[bool, str]:
    with open(path) as f:
        data = json.load(f)
    kind = data.get("fixture_kind")
    if kind == "pass":
        err = _check_pass_fixture(data)
        return (False, f"FAIL (expected PASS): {err}") if err else (True, "PASS")
    if kind == "fail":
        err = _check_fail_fixture(data)
        return (False, f"FAIL (fail-fixture check): {err}") if err else (True, "PASS (expected FAIL)")
    return False, f"UNKNOWN fixture_kind: {kind}"


# ---------------------------------------------------------------------------
# Deterministic self-test
# ---------------------------------------------------------------------------
def self_test() -> dict:
    errors: list[str] = []
    m = 24

    # Fixed small memory: 3 patterns of length 12
    P = [
        [1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22],
        [3, 3, 3, 3, 12, 12, 12, 12, 20, 20, 20, 20],
        [24, 1, 24, 1, 24, 1, 24, 1, 24, 1, 24, 1],
    ]
    N = 12

    # PC_OVERLAP: overlap == exact match count, exhaustive over the stored set
    for pat in P:
        for other in P:
            oc = overlap(pat, other, m)
            mc = sum(1 for a, b in zip(pat, other) if a == b)
            if oc != mc:
                errors.append(f"PC_OVERLAP FAILED: {oc} != match count {mc}")

    # FIXED_POINTS: each stored pattern recalls to itself
    for k, pat in enumerate(P):
        if recall(pat, P, m) != pat:
            errors.append(f"FIXED_POINTS FAILED: pattern {k} not a fixed point")

    # RECALL: corrupt pattern 0 at 4 of 12 sites -> exact recall
    probe = list(P[0]); probe[1] = 7; probe[4] = 7; probe[7] = 7; probe[10] = 7
    if recall(probe, P, m) != P[0]:
        errors.append("RECALL FAILED: corrupted probe did not recall pattern 0")

    # CONTENT_ADDRESSABLE: probe nearest to pattern 1 recalls exactly pattern 1
    probe1 = list(P[1]); probe1[0] = 15; probe1[6] = 15; probe1[11] = 15
    rec1 = recall(probe1, P, m)
    if rec1 != P[1]:
        errors.append("CONTENT_ADDRESSABLE FAILED: did not recall nearest pattern 1")

    # DISTORTION_TOLERANT + NAIVE_DISTORTION_FAILS: global phase screen phi=5
    phi = 5
    distorted = [qa_add(v, phi, m) for v in probe]      # shifted corrupted P[0]
    target = [qa_add(v, phi, m) for v in P[0]]
    if recall_phase_locked(distorted, P, m) != target:
        errors.append("DISTORTION_TOLERANT FAILED: phase-locked recall did not recover P0+phi")
    if recall(distorted, P, m) == target:
        errors.append("NAIVE_DISTORTION_FAILS FAILED: naive recall unexpectedly recovered through the screen")

    # A1_RANGE: recall outputs stay in {1,...,m}
    if not _in_range(recall(probe, P, m), m):
        errors.append("A1_RANGE FAILED: recall produced a state outside {1,...,m}")

    # SRC gate
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "mapping_protocol_ref.json")
    if not os.path.exists(src_path):
        errors.append("SRC FAILED: mapping_protocol_ref.json missing")
    else:
        with open(src_path) as f:
            ref = json.load(f)
        if ref.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
            errors.append("SRC FAILED: wrong protocol_version")

    # F gate
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
                    errors.append(f"F FAILED (pass {fname}): {msg}")
            elif kind == "fail":
                fail_fixtures.append(fname)
                if not ok:
                    errors.append(f"F FAILED (fail {fname}): {msg}")
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
