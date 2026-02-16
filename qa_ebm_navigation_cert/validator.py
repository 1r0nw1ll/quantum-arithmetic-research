#!/usr/bin/env python3
"""
validator.py

QA_EBM_NAVIGATION_CERT.v1 validator (Machine Tract).

This family hardens "energy-based reasoning" into a QA-native navigation witness:
- exact (int/Fraction) energy scalars (no floats)
- deterministic min-energy selection over legal candidates
- total tie-break (lex(generator, state_after))
- per-step invariant_diff (delta_energy + delta_violations + witness)
- typed failures with obstruction witnesses

Hash spec:
canonical_sha256 = sha256(canonical_json_compact(cert_with_canonical_sha256_zeroed))
where canonical_json_compact uses sort_keys=True, separators=(',',':'), ensure_ascii=False.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple


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


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_exact_scalar(s: Dict[str, Any]) -> Tuple[bool, Optional[Fraction], str]:
    if not isinstance(s, dict):
        return False, None, "not an object"
    t = s.get("type")
    v = s.get("value")
    if t not in ("int", "rational"):
        return False, None, f"bad scalar type: {t}"
    if not isinstance(v, str) or not v.strip():
        return False, None, "scalar value missing/empty"
    if "." in v or "e" in v.lower():
        return False, None, "float/scientific notation forbidden"
    if "/" in v:
        if t != "rational":
            return False, None, "int scalar cannot contain '/'"
        num, den = v.split("/", 1)
        if not (num.lstrip("-").isdigit() and den.isdigit()):
            return False, None, "invalid rational format"
        if int(den) == 0:
            return False, None, "division by zero"
        return True, Fraction(int(num), int(den)), ""
    if not v.lstrip("-").isdigit():
        return False, None, "invalid int format"
    if t != "int":
        # rational may be given as integer string; accept but normalize as int
        return True, Fraction(int(v), 1), ""
    return True, Fraction(int(v), 1), ""


def _vec_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    keys = set(before.keys()) | set(after.keys())
    out: Dict[str, int] = {}
    for k in sorted(keys):
        out[k] = int(after.get(k, 0)) - int(before.get(k, 0))
    return out


def _validate_schema(obj: Dict[str, Any]) -> None:
    import jsonschema

    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    # Zero the canonical_sha256 field before hashing (no recursion).
    copy = json.loads(_canonical_json_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_json_compact(copy))


def _repo_root() -> str:
    # qa_ebm_navigation_cert/ lives at repo root; repo_root is its parent.
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _resolve_repo_rel_path(repo_root: str, rel_path: str) -> Tuple[bool, str, str]:
    """
    Resolve rel_path against repo_root and enforce that it stays within repo_root.
    Returns (ok, abs_path, error_message).
    """
    if not isinstance(rel_path, str) or not rel_path.strip():
        return False, "", "ref_name missing/empty"
    if os.path.isabs(rel_path):
        return False, "", "absolute ref_name forbidden"
    if "\x00" in rel_path:
        return False, "", "NUL byte forbidden in ref_name"
    abs_path = os.path.normpath(os.path.join(repo_root, rel_path))
    try:
        if os.path.commonpath([repo_root, abs_path]) != repo_root:
            return False, "", "ref_name escapes repo_root"
    except Exception:
        return False, "", "ref_name path check failed"
    return True, abs_path, ""


def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema Validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Valid QA_EBM_NAVIGATION_CERT.v1 schema"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL, f"Schema validation failed: {e}"))
        return results

    # Gate 2 — Canonical hash
    want = obj.get("digests", {}).get("canonical_sha256", "")
    got = _compute_canonical_sha256(obj)
    if not isinstance(want, str) or len(want) != 64:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL, "canonical_sha256 missing/invalid",
                                  {"want": want, "got": got}))
        return results
    if want == "0" * 64:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL, "canonical_sha256 is placeholder",
                                  {"got": got}))
        return results
    if want != got:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL, "canonical_sha256 mismatch",
                                  {"want": want, "got": got}))
        return results
    results.append(GateResult("gate_2_canonical_hash", GateStatus.PASS, "canonical_sha256 matches"))

    # Gate 3 — Exact energy + invariant_diff correctness
    energy_scalar_type = obj["navigation"]["energy"]["scalar_type"]
    bad_steps = []
    for i, step in enumerate(obj["trace"]["steps"]):
        ok_b, eb, msg_b = _parse_exact_scalar(step.get("energy_before"))
        ok_a, ea, msg_a = _parse_exact_scalar(step.get("energy_after")) if step.get("result") == "OK" else (True, Fraction(0, 1), "")
        if not ok_b:
            bad_steps.append({"step": i, "field": "energy_before", "error": msg_b})
            continue
        if step.get("result") == "OK" and not ok_a:
            bad_steps.append({"step": i, "field": "energy_after", "error": msg_a})
            continue

        # Enforce scalar type discipline (no mixing)
        if isinstance(step.get("energy_before"), dict) and step["energy_before"].get("type") != energy_scalar_type:
            bad_steps.append({"step": i, "field": "energy_before.type", "error": "scalar_type mismatch"})
            continue
        if step.get("result") == "OK":
            if isinstance(step.get("energy_after"), dict) and step["energy_after"].get("type") != energy_scalar_type:
                bad_steps.append({"step": i, "field": "energy_after.type", "error": "scalar_type mismatch"})
                continue

        inv = step.get("invariant_diff", {})
        ok_de, de, msg_de = _parse_exact_scalar(inv.get("delta_energy"))
        if not ok_de:
            bad_steps.append({"step": i, "field": "invariant_diff.delta_energy", "error": msg_de})
            continue

        # delta_energy must match (energy_after - energy_before) when OK
        if step.get("result") == "OK":
            if de != (ea - eb):
                bad_steps.append({
                    "step": i,
                    "field": "invariant_diff.delta_energy",
                    "error": "delta_energy mismatch",
                    "expected": str(ea - eb),
                    "got": str(de),
                })
                continue

        # delta_violations must match (after-before) when OK
        vb = step.get("violations_before", {})
        va = step.get("violations_after", {}) if step.get("result") == "OK" else {}
        if not isinstance(vb, dict) or (step.get("result") == "OK" and not isinstance(va, dict)):
            bad_steps.append({"step": i, "field": "violations_before/after", "error": "must be objects"})
            continue
        dv = inv.get("delta_violations", {})
        if not isinstance(dv, dict):
            bad_steps.append({"step": i, "field": "invariant_diff.delta_violations", "error": "must be object"})
            continue
        if step.get("result") == "OK":
            expected_dv = _vec_delta({k: int(v) for k, v in vb.items()}, {k: int(v) for k, v in va.items()})
            got_dv = {k: int(v) for k, v in dv.items()}
            if got_dv != expected_dv:
                bad_steps.append({
                    "step": i,
                    "field": "invariant_diff.delta_violations",
                    "error": "delta_violations mismatch",
                    "expected": expected_dv,
                    "got": got_dv,
                })
                continue

    if bad_steps:
        results.append(GateResult(
            "gate_3_exact_energy_and_invariant_diff",
            GateStatus.FAIL,
            "Exact energy / invariant_diff contract failed",
            {"bad_steps": bad_steps},
        ))
        return results
    results.append(GateResult(
        "gate_3_exact_energy_and_invariant_diff",
        GateStatus.PASS,
        "Exact energy + invariant_diff deltas consistent",
    ))

    # Gate 4 — Deterministic selection + tie-break
    bad_decisions = []
    tie_break = obj["navigation"]["policy"]["tie_break"]
    for i, step in enumerate(obj["trace"]["steps"]):
        candidates = step.get("candidates", [])
        chosen_idx = step.get("chosen_idx")
        if not isinstance(candidates, list) or not candidates:
            bad_decisions.append({"step": i, "error": "missing candidates"})
            continue
        if not isinstance(chosen_idx, int) or chosen_idx < 0 or chosen_idx >= len(candidates):
            bad_decisions.append({"step": i, "error": "chosen_idx out of range", "chosen_idx": chosen_idx})
            continue

        legal = []
        for c in candidates:
            if c.get("legal") is True:
                ok_e, e, msg = _parse_exact_scalar(c.get("energy_after"))
                if not ok_e:
                    bad_decisions.append({"step": i, "error": f"candidate energy_after invalid: {msg}", "candidate": c})
                    legal = []
                    break
                if c.get("energy_after", {}).get("type") != energy_scalar_type:
                    bad_decisions.append({"step": i, "error": "candidate scalar_type mismatch", "candidate": c})
                    legal = []
                    break
                legal.append((c.get("generator", ""), c.get("state_after", ""), e))
        if not legal:
            if step.get("result") != "FAIL":
                bad_decisions.append({"step": i, "error": "no legal candidates but result not FAIL"})
            continue

        min_e = min(e for _g, _s, e in legal)
        tied = [(g, s) for g, s, e in legal if e == min_e]
        if tie_break == "lex_generator_then_state_after":
            winner_g, winner_s = sorted(tied, key=lambda x: (x[0], x[1]))[0]
        else:
            bad_decisions.append({"step": i, "error": f"unsupported tie_break: {tie_break}"})
            continue

        chosen = candidates[chosen_idx]
        if chosen.get("legal") is not True:
            bad_decisions.append({"step": i, "error": "chosen candidate is illegal but legal candidates exist"})
            continue
        if (chosen.get("generator"), chosen.get("state_after")) != (winner_g, winner_s):
            bad_decisions.append({
                "step": i,
                "error": "chosen candidate does not match deterministic min-energy+tiebreak winner",
                "winner": {"generator": winner_g, "state_after": winner_s, "energy": str(min_e)},
                "chosen": {"generator": chosen.get("generator"), "state_after": chosen.get("state_after")},
            })
            continue

    if bad_decisions:
        results.append(GateResult(
            "gate_4_deterministic_tiebreak",
            GateStatus.FAIL,
            "Deterministic tie-break violated",
            {"bad_steps": bad_decisions},
        ))
        return results
    results.append(GateResult(
        "gate_4_deterministic_tiebreak",
        GateStatus.PASS,
        "Min-energy selection + deterministic tie-break satisfied",
    ))

    # Gate 5 — Failure completeness
    bad_failures = []
    for i, step in enumerate(obj["trace"]["steps"]):
        if step.get("result") == "FAIL":
            failure = step.get("failure")
            if not isinstance(failure, dict):
                bad_failures.append({"step": i, "error": "FAIL step missing failure object"})
                continue
            w = failure.get("invariant_diff", {}).get("witness", "")
            if not isinstance(w, str) or not w.strip():
                bad_failures.append({"step": i, "error": "failure witness missing/empty"})

    outcome = obj["trace"]["outcome"]
    if outcome.get("status") == "FAILED_TYPED":
        failure = outcome.get("failure")
        if not isinstance(failure, dict):
            bad_failures.append({"outcome": True, "error": "FAILED_TYPED outcome missing failure object"})
        else:
            w = failure.get("invariant_diff", {}).get("witness", "")
            if not isinstance(w, str) or not w.strip():
                bad_failures.append({"outcome": True, "error": "outcome failure witness missing/empty"})

    if bad_failures:
        results.append(GateResult(
            "gate_5_failure_completeness",
            GateStatus.FAIL,
            "Typed failures missing/invalid",
            {"issues": bad_failures},
        ))
        return results
    results.append(GateResult(
        "gate_5_failure_completeness",
        GateStatus.PASS,
        "Typed failure witnesses present when needed",
    ))

    # Gate 6 — Verifier-gated acceptance binding (optional)
    outcome = obj["trace"]["outcome"]
    if outcome.get("accepted_by_verifier") is True:
        vr = outcome.get("verifier_bridge_ref")
        if not isinstance(vr, dict):
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "accepted_by_verifier=true requires verifier_bridge_ref object",
            ))
            return results
        for field in ("ref_name", "sha256"):
            if field not in vr:
                results.append(GateResult(
                    "gate_6_verifier_acceptance_binding",
                    GateStatus.FAIL,
                    "verifier_bridge_ref missing field",
                    {"missing_field": field},
                ))
                return results

        refs = obj.get("digests", {}).get("refs")
        if not isinstance(refs, list) or len(refs) == 0:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "accepted_by_verifier=true requires digests.refs to include verifier bridge ref",
                {"verifier_bridge_ref": vr},
            ))
            return results
        matched = any(
            isinstance(r, dict)
            and r.get("sha256") == vr.get("sha256")
            and r.get("ref_name") == vr.get("ref_name")
            for r in refs
        )
        if not matched:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "digests.refs missing verifier bridge ref",
                {"verifier_bridge_ref": vr, "digests_refs": refs},
            ))
            return results

        # Transitive trust: referenced bridge cert must be repo-local and validator-clean,
        # and the ref sha256 must match bridge digests.canonical_sha256.
        repo_root = _repo_root()
        ok_path, abs_bridge_path, msg_path = _resolve_repo_rel_path(repo_root, str(vr.get("ref_name", "")))
        if not ok_path:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "verifier_bridge_ref.ref_name must be repo-relative and stay within repo_root",
                {"error": msg_path, "ref_name": vr.get("ref_name")},
            ))
            return results
        if not os.path.isfile(abs_bridge_path):
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "referenced verifier bridge cert not found",
                {"abs_path": abs_bridge_path},
            ))
            return results

        try:
            bridge_obj = _load_json(abs_bridge_path)
        except Exception as e:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "failed to load referenced verifier bridge cert JSON",
                {"abs_path": abs_bridge_path, "error": str(e)},
            ))
            return results

        bridge_sha = bridge_obj.get("digests", {}).get("canonical_sha256", "")
        if not isinstance(bridge_sha, str) or len(bridge_sha) != 64:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "referenced verifier bridge cert missing digests.canonical_sha256",
                {"abs_path": abs_bridge_path},
            ))
            return results
        if bridge_sha != vr.get("sha256"):
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "verifier_bridge_ref.sha256 must equal referenced bridge digests.canonical_sha256",
                {"expected": bridge_sha, "got": vr.get("sha256")},
            ))
            return results

        bridge_validator = os.path.join(repo_root, "qa_ebm_verifier_bridge_cert", "validator.py")
        if not os.path.isfile(bridge_validator):
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "verifier bridge validator missing",
                {"validator_path": bridge_validator},
            ))
            return results
        try:
            proc = subprocess.run(
                [sys.executable, bridge_validator, abs_bridge_path, "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_root,
            )
        except subprocess.TimeoutExpired:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "verifier bridge validator timed out",
                {"validator_path": bridge_validator, "abs_path": abs_bridge_path},
            ))
            return results
        if proc.returncode != 0:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "referenced verifier bridge cert failed its own validator",
                {
                    "validator_path": bridge_validator,
                    "abs_path": abs_bridge_path,
                    "stdout_head": (proc.stdout or "").strip()[:400],
                    "stderr_head": (proc.stderr or "").strip()[:200],
                },
            ))
            return results

        # Additional binding: the bridge cert must attest the specific terminal state + target
        # that this navigation cert claims was verifier-accepted.
        nav_target_ref = outcome.get("target_ref")
        if not isinstance(nav_target_ref, dict) or "ref_name" not in nav_target_ref or "sha256" not in nav_target_ref:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "accepted_by_verifier=true requires outcome.target_ref for bridge binding",
                {"target_ref": nav_target_ref},
            ))
            return results

        terminal_state: Optional[str] = None
        for step in reversed(obj.get("trace", {}).get("steps", [])):
            if step.get("result") == "OK":
                sa = step.get("state_after")
                if isinstance(sa, str) and sa:
                    terminal_state = sa
                    break
        if terminal_state is None:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "accepted_by_verifier=true requires at least one OK step with state_after to bind",
            ))
            return results

        terminal_state_sha256 = _sha256_hex(terminal_state)

        subject = bridge_obj.get("subject")
        verdict = bridge_obj.get("verdict")
        if not isinstance(subject, dict) or not isinstance(verdict, dict):
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "bridge cert missing subject/verdict objects for binding",
            ))
            return results

        if verdict.get("passed") is not True or verdict.get("fail_type") != "OK":
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "bridge cert verdict must be passed=true and fail_type=OK when accepted_by_verifier=true",
                {"passed": verdict.get("passed"), "fail_type": verdict.get("fail_type")},
            ))
            return results

        if subject.get("state_after") != terminal_state:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "bridge cert subject.state_after does not match navigation terminal state_after",
                {"nav_terminal_state": terminal_state, "bridge_state_after": subject.get("state_after")},
            ))
            return results

        if subject.get("state_sha256") != terminal_state_sha256:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "bridge cert subject.state_sha256 does not match sha256(navigation terminal state_after)",
                {"expected": terminal_state_sha256, "got": subject.get("state_sha256")},
            ))
            return results

        if subject.get("target_ref") != nav_target_ref:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "bridge cert subject.target_ref does not match navigation outcome.target_ref",
                {"nav_target_ref": nav_target_ref, "bridge_target_ref": subject.get("target_ref")},
            ))
            return results

        # Optional: mapping protocol consistency across nav/bridge when both declare it.
        nav_mp = obj.get("inputs", {}).get("mapping_protocol_ref") if isinstance(obj.get("inputs"), dict) else None
        bridge_mp = bridge_obj.get("inputs", {}).get("mapping_protocol_ref") if isinstance(bridge_obj.get("inputs"), dict) else None
        if nav_mp is not None and bridge_mp is not None and nav_mp != bridge_mp:
            results.append(GateResult(
                "gate_6_verifier_acceptance_binding",
                GateStatus.FAIL,
                "bridge inputs.mapping_protocol_ref must match navigation inputs.mapping_protocol_ref when both present",
                {"nav_mapping_protocol_ref": nav_mp, "bridge_mapping_protocol_ref": bridge_mp},
            ))
            return results
        results.append(GateResult(
            "gate_6_verifier_acceptance_binding",
            GateStatus.PASS,
            "Verifier bridge ref present, digest-linked, validator-clean, and terminal-state/target bound",
        ))
    else:
        results.append(GateResult(
            "gate_6_verifier_acceptance_binding",
            GateStatus.PASS,
            "No verifier acceptance claimed",
        ))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(base, "fixtures")

    fixtures = [
        ("valid_min.json", True, None),
        ("valid_accepted_with_verifier_bridge_ref.json", True, None),
        ("invalid_missing_invariant_diff.json", False, "gate_1_schema_validity"),
        ("invalid_accepted_missing_bridge_ref.json", False, "gate_1_schema_validity"),
        ("invalid_accepted_missing_digest_ref.json", False, "gate_6_verifier_acceptance_binding"),
        ("invalid_accepted_bridge_cert_invalid.json", False, "gate_6_verifier_acceptance_binding"),
        ("invalid_accepted_bridge_state_mismatch.json", False, "gate_6_verifier_acceptance_binding"),
        ("invalid_invariant_diff_mismatch.json", False, "gate_3_exact_energy_and_invariant_diff"),
        ("invalid_wrong_tiebreak.json", False, "gate_4_deterministic_tiebreak"),
        ("invalid_float_energy.json", False, "gate_1_schema_validity"),
        ("invalid_digest_mismatch.json", False, "gate_2_canonical_hash"),
    ]

    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        obj = _load_json(path)
        res = validate_cert(obj)
        passed = _report_ok(res)
        if should_pass != passed:
            ok = False
        fail_gates = [r.gate for r in res if r.status == GateStatus.FAIL]
        if (not should_pass) and expected_fail_gate and expected_fail_gate not in fail_gates:
            ok = False
        details.append({
            "fixture": name,
            "ok": passed,
            "expected_ok": should_pass,
            "failed_gates": fail_gates,
        })

    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_EBM_NAVIGATION_CERT.v1 SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_EBM_NAVIGATION_CERT.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test on fixtures")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = _load_json(args.file)
    results = validate_cert(obj)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
