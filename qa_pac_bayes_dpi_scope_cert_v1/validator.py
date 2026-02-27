#!/usr/bin/env python3
"""
validator.py

QA_PAC_BAYES_DPI_SCOPE_CERT.v1 validator (Machine Tract) — Family [86].

Locks the DPI scope claim for the Phase-1 PAC-Bayes work to structured-only
by certifying:
  - what "structured" means (generator + parameterization),
  - empirical evidence (random-trial violation rates, seeds),
  - allowed claim levels (no silent upgrade to "proven/universal"),
  - cross-family refs to [84]/[85].

Gates:
  1. JSON schema validity (cert_version == QA_PAC_BAYES_DPI_SCOPE_CERT.v1)
  2. Canonical SHA-256 digest integrity (self-referential)
  3. Evidence rate consistency — recompute violation_rate = n_violations/n_trials
     for structured + random (+ multistep if enabled); abs_tol=1e-12
  4. Scope separation assertion — structured.violation_rate <= pass_threshold
     AND random.violation_rate >= min_expected_violation_rate
  5. Claim policy enforcement — dpi_claim=="structured_only"; forbidden_phrases
     must include {"universal","proven","for all distributions"}

Failure modes:
  SCHEMA_INVALID, DIGEST_MISMATCH, EVIDENCE_RATE_MISMATCH,
  SCOPE_SEPARATION_VIOLATION, CLAIM_POLICY_VIOLATION
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Locked policy constants
# ---------------------------------------------------------------------------
_RATE_TOL               = 1e-12
_REQUIRED_FORBIDDEN     = frozenset(["universal", "proven", "for all distributions"])


# ---------------------------------------------------------------------------
# Gate scaffolding
# ---------------------------------------------------------------------------

class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _canonical_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    """SHA-256 over canonical JSON with digests.canonical_sha256 zeroed."""
    copy = json.loads(_canonical_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_compact(copy))


def _check_rate_block(
    label: str,
    block: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Recompute violation_rate = n_violations / n_trials; check bounds and match."""
    errs: List[Dict] = []
    n_viol  = int(block["n_violations"])
    n_trial = int(block["n_trials"])
    stored  = float(block["violation_rate"])

    if n_viol < 0 or n_viol > n_trial:
        errs.append({
            "path": f"evidence.{label}.n_violations",
            "expected": f"0 <= n_violations <= n_trials ({n_trial})",
            "got": n_viol,
            "delta": None,
        })
        return errs  # can't compute rate meaningfully

    computed = n_viol / n_trial
    delta    = abs(computed - stored)
    if delta > _RATE_TOL:
        errs.append({
            "path": f"evidence.{label}.violation_rate",
            "expected": computed,
            "got": stored,
            "delta": delta,
        })
    return errs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # ------------------------------------------------------------------
    # Gate 1 — Schema validity
    # ------------------------------------------------------------------
    try:
        import jsonschema
        schema = _load_json(_schema_path())
        jsonschema.validate(instance=obj, schema=schema)
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.PASS,
            "Schema valid; cert_version=QA_PAC_BAYES_DPI_SCOPE_CERT.v1",
        ))
    except Exception as exc:
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.FAIL,
            f"SCHEMA_INVALID: {exc}",
        ))
        return results

    # ------------------------------------------------------------------
    # Gate 2 — Canonical hash integrity
    # ------------------------------------------------------------------
    want = obj.get("digests", {}).get("canonical_sha256", "")
    got  = _compute_canonical_sha256(obj)
    if want == "0" * 64:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: canonical_sha256 is placeholder",
            {"got": got},
        ))
        return results
    if want != got:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: canonical_sha256 does not match",
            {"want": want, "got": got},
        ))
        return results
    results.append(GateResult(
        "gate_2_digest_integrity", GateStatus.PASS,
        "canonical_sha256 verified",
    ))

    # ------------------------------------------------------------------
    # Gate 3 — Evidence rate consistency
    # ------------------------------------------------------------------
    ev      = obj["evidence"]
    s_block = ev["structured_trials"]
    r_block = ev["random_trials"]

    rate_errors: List[Dict] = []
    rate_errors += _check_rate_block("structured_trials", s_block)
    rate_errors += _check_rate_block("random_trials",     r_block)

    # multistep_audit is optional; check only if present and enabled
    ms = ev.get("multistep_audit")
    if ms and ms.get("enabled"):
        rate_errors += _check_rate_block("multistep_audit", ms)

    if rate_errors:
        recomp = {
            "structured_rate": int(s_block["n_violations"]) / int(s_block["n_trials"]),
            "random_rate":     int(r_block["n_violations"]) / int(r_block["n_trials"]),
        }
        if ms and ms.get("enabled"):
            recomp["multistep_rate"] = int(ms["n_violations"]) / int(ms["n_trials"])
        results.append(GateResult(
            "gate_3_evidence_rate_consistency", GateStatus.FAIL,
            "EVIDENCE_RATE_MISMATCH",
            {"invariant_diff": {
                "gate": 3,
                "fields": rate_errors,
                "recomputed": recomp,
            }},
        ))
        return results

    results.append(GateResult(
        "gate_3_evidence_rate_consistency", GateStatus.PASS,
        (f"structured violation_rate={float(s_block['violation_rate']):.4f} "
         f"(n={s_block['n_trials']}, v={s_block['n_violations']}); "
         f"random violation_rate={float(r_block['violation_rate']):.4f} "
         f"(n={r_block['n_trials']}, v={r_block['n_violations']})"),
    ))

    # ------------------------------------------------------------------
    # Gate 4 — Scope separation assertion
    # ------------------------------------------------------------------
    s_rate      = float(s_block["violation_rate"])
    s_threshold = float(s_block["pass_threshold"])
    r_rate      = float(r_block["violation_rate"])
    r_min_exp   = float(r_block["min_expected_violation_rate"])

    sep_errors: List[Dict] = []

    if s_rate > s_threshold:
        sep_errors.append({
            "path": "evidence.structured_trials.violation_rate",
            "expected": f"<= pass_threshold ({s_threshold})",
            "got": s_rate,
            "delta": s_rate - s_threshold,
        })
    if r_rate < r_min_exp:
        sep_errors.append({
            "path": "evidence.random_trials.violation_rate",
            "expected": f">= min_expected_violation_rate ({r_min_exp})",
            "got": r_rate,
            "delta": r_rate - r_min_exp,
        })

    if sep_errors:
        results.append(GateResult(
            "gate_4_scope_separation_assertion", GateStatus.FAIL,
            "SCOPE_SEPARATION_VIOLATION",
            {"invariant_diff": {
                "gate": 4,
                "fields": sep_errors,
                "recomputed": {
                    "structured_rate":     s_rate,
                    "structured_threshold": s_threshold,
                    "random_rate":         r_rate,
                    "random_min_expected": r_min_exp,
                },
            }},
        ))
        return results

    results.append(GateResult(
        "gate_4_scope_separation_assertion", GateStatus.PASS,
        (f"structured {s_rate:.4f} <= threshold {s_threshold:.4f}; "
         f"random {r_rate:.4f} >= min_expected {r_min_exp:.4f}"),
    ))

    # ------------------------------------------------------------------
    # Gate 5 — Claim policy enforcement
    # ------------------------------------------------------------------
    claims       = obj["claims"]
    dpi_claim    = claims["dpi_claim"]
    claim_level  = claims["claim_level"]
    forbidden    = set(claims["forbidden_phrases"])

    policy_errors: List[Dict] = []

    if dpi_claim != "structured_only":
        policy_errors.append({
            "path": "claims.dpi_claim",
            "expected": "structured_only",
            "got": dpi_claim,
            "delta": None,
        })

    missing_phrases = _REQUIRED_FORBIDDEN - forbidden
    if missing_phrases:
        policy_errors.append({
            "path": "claims.forbidden_phrases",
            "expected": f"must include all of {sorted(_REQUIRED_FORBIDDEN)}",
            "got": sorted(forbidden),
            "delta": None,
        })

    if policy_errors:
        results.append(GateResult(
            "gate_5_claim_policy_enforcement", GateStatus.FAIL,
            "CLAIM_POLICY_VIOLATION",
            {"invariant_diff": {
                "gate": 5,
                "fields": policy_errors,
                "recomputed": {
                    "required_forbidden_phrases": sorted(_REQUIRED_FORBIDDEN),
                    "present_forbidden_phrases":  sorted(forbidden),
                    "missing_forbidden_phrases":  sorted(missing_phrases),
                },
            }},
        ))
        return results

    results.append(GateResult(
        "gate_5_claim_policy_enforcement", GateStatus.PASS,
        (f"dpi_claim='structured_only'; claim_level='{claim_level}'; "
         f"all required forbidden phrases present: {sorted(_REQUIRED_FORBIDDEN)}"),
    ))

    return results


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------

def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json_out(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx   = os.path.join(base, "fixtures")
    fixtures = [
        ("valid_pac_bayes_dpi_scope_v1.json",                   True,  None),
        ("invalid_random_trials_too_clean.json",                 False, "gate_4_scope_separation_assertion"),
        ("invalid_claim_policy_missing_forbidden_phrase.json",   False, "gate_5_claim_policy_enforcement"),
    ]
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        if not os.path.exists(path):
            details.append({
                "fixture": name, "ok": None,
                "expected_ok": should_pass, "failed_gates": [], "note": "MISSING",
            })
            ok = False
            continue
        obj = _load_json(path)
        res = validate_cert(obj)
        passed = _report_ok(res)
        if should_pass != passed:
            ok = False
        fail_gates = [r.gate for r in res if r.status == GateStatus.FAIL]
        if (not should_pass) and expected_fail_gate and expected_fail_gate not in fail_gates:
            ok = False
        details.append({
            "fixture": name, "ok": passed,
            "expected_ok": should_pass, "failed_gates": fail_gates,
        })

    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_PAC_BAYES_DPI_SCOPE_CERT.v1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QA_PAC_BAYES_DPI_SCOPE_CERT.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(as_json=args.json)
    if not args.file:
        ap.print_help()
        return 2
    obj = _load_json(args.file)
    results = validate_cert(obj)
    if args.json:
        _print_json_out(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
