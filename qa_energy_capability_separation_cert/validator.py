#!/usr/bin/env python3
"""
validator.py

QA_ENERGY_CAPABILITY_SEPARATION_CERT.v1 validator (Machine Tract).

Claim certified by this family (constructive separation witness):

  Capability = Reachability(S, Σ, I)
  Energy policy = deterministic ordering heuristic over legal successors

This cert proves, in a finite Caps(N,N) subuniverse, that:

  (1) target is reachable under the generator set (explicit witness path)
  (2) deterministic min-energy-legal policy fails to reach target within budget

Therefore: Energy ordering does not constitute (and does not expand) capability.

Hash spec:
canonical_sha256 = sha256(canonical_json_compact(cert_with_canonical_sha256_zeroed))
where canonical_json_compact uses sort_keys=True, separators=(',',':'), ensure_ascii=False.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    copy = json.loads(_canonical_json_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_json_compact(copy))


def _validate_schema(obj: Dict[str, Any]) -> None:
    import jsonschema

    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


def _state_to_str(s: Dict[str, Any]) -> str:
    return f"(b={int(s['b'])},e={int(s['e'])})"


def _in_caps(s: Dict[str, Any], N: int) -> bool:
    b = int(s["b"])
    e = int(s["e"])
    return (1 <= b <= N) and (1 <= e <= N)


def _apply_generator(name: str, s: Dict[str, Any], N: int) -> Tuple[bool, Dict[str, Any], str]:
    """
    Canonical QA generators (subset) per qa_canonical.md:
      sigma: (b,e+1) with caps legality
      mu:    (e,b) on Caps(N,N) always legal
    Returns (legal, next_state, fail_type)
    """
    b = int(s["b"])
    e = int(s["e"])
    if name == "sigma":
        if e + 1 > N:
            return False, {"b": b, "e": e}, "OUT_OF_BOUNDS"
        return True, {"b": b, "e": e + 1}, ""
    if name == "mu":
        # Always legal on square Caps(N,N)
        return True, {"b": e, "e": b}, ""
    raise ValueError(f"Unsupported generator: {name}")


def _energy(form: str, s: Dict[str, Any]) -> Fraction:
    if form == "e_coordinate":
        return Fraction(int(s["e"]), 1)
    raise ValueError(f"Unsupported energy form: {form}")


def _parse_int_scalar(obj: Dict[str, Any]) -> Tuple[bool, Optional[Fraction], str]:
    if not isinstance(obj, dict):
        return False, None, "not an object"
    if obj.get("type") != "int":
        return False, None, "type must be int"
    v = obj.get("value")
    if not isinstance(v, str) or not v.strip():
        return False, None, "value missing"
    if "." in v or "e" in v.lower():
        return False, None, "float/scientific forbidden"
    if not v.lstrip("-").isdigit():
        return False, None, "invalid int format"
    return True, Fraction(int(v), 1), ""


def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Schema valid"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL, f"Schema invalid: {e}"))
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

    N = int(obj["state_space"]["N"])
    start = obj["start_state"]
    target = obj["target_state"]
    gens = obj["generator_set"]["generators"]
    energy_form = obj["energy_function"]["form"]

    # Gate 3 — Witness path legality + reaches target within max_k
    max_k = int(obj["reachability_witness"]["max_k"])
    path = obj["reachability_witness"]["path_generators"]
    if len(path) > max_k:
        results.append(GateResult(
            "gate_3_witness_reachability",
            GateStatus.FAIL,
            "witness path length exceeds max_k",
            {"path_len": len(path), "max_k": max_k},
        ))
        return results

    cur = {"b": int(start["b"]), "e": int(start["e"])}
    if not _in_caps(cur, N) or not _in_caps(target, N):
        results.append(GateResult(
            "gate_3_witness_reachability",
            GateStatus.FAIL,
            "start/target out of Caps(N,N)",
            {"start": cur, "target": target, "N": N},
        ))
        return results
    illegal = []
    for i, g in enumerate(path):
        if g not in gens:
            illegal.append({"i": i, "generator": g, "error": "not in generator_set"})
            break
        ok, nxt, fail_type = _apply_generator(g, cur, N)
        if not ok:
            illegal.append({"i": i, "generator": g, "state": cur, "fail_type": fail_type})
            break
        cur = nxt
        if not _in_caps(cur, N):
            illegal.append({"i": i, "generator": g, "state": cur, "error": "out of caps"})
            break
    if illegal:
        results.append(GateResult(
            "gate_3_witness_reachability",
            GateStatus.FAIL,
            "witness path illegal",
            {"issues": illegal},
        ))
        return results

    if cur != {"b": int(target["b"]), "e": int(target["e"])}:
        results.append(GateResult(
            "gate_3_witness_reachability",
            GateStatus.FAIL,
            "witness path does not reach target",
            {"end_state": cur, "target": target},
        ))
        return results
    results.append(GateResult(
        "gate_3_witness_reachability",
        GateStatus.PASS,
        "target reachable under generators (witness path verified)",
        {"path_len": len(path)},
    ))

    # Gate 4 — Policy run follows min-energy-legal with deterministic tie-break, and invariant_diff energy deltas correct
    steps = obj["policy_run"]["steps"]
    max_steps = int(obj["policy_run"]["max_steps"])
    if len(steps) > max_steps:
        results.append(GateResult(
            "gate_4_policy_replay",
            GateStatus.FAIL,
            "policy_run.steps length exceeds max_steps",
            {"steps_len": len(steps), "max_steps": max_steps},
        ))
        return results

    policy_ok = True
    issues = []
    reached = False
    visited = []

    for st in steps:
        t = int(st["t"])
        sb = st["state_before"]
        sa = st["state_after"]
        if not _in_caps(sb, N) or not _in_caps(sa, N):
            policy_ok = False
            issues.append({"t": t, "error": "state out of caps", "state_before": sb, "state_after": sa})
            break

        # Candidates: verify each candidate is consistent with canonical generator semantics and energy form.
        candidates = st["candidates"]
        legal_with_energy: List[Tuple[Fraction, str, str, int]] = []  # (energy, generator, state_str, idx)
        for idx, c in enumerate(candidates):
            g = c["generator"]
            legal_flag = bool(c["legal"])
            if g not in gens:
                policy_ok = False
                issues.append({"t": t, "candidate": idx, "error": "generator not in generator_set", "generator": g})
                break
            ok, nxt, fail_type = _apply_generator(g, sb, N)
            if legal_flag is True and not ok:
                policy_ok = False
                issues.append({"t": t, "candidate": idx, "error": "marked legal but illegal", "fail_type": fail_type})
                break
            if legal_flag is False and ok:
                policy_ok = False
                issues.append({"t": t, "candidate": idx, "error": "marked illegal but legal"})
                break
            if legal_flag is False:
                if c.get("fail_type") != fail_type:
                    policy_ok = False
                    issues.append({"t": t, "candidate": idx, "error": "fail_type mismatch",
                                   "expected": fail_type, "got": c.get("fail_type")})
                    break
                continue

            # legal candidate: state_after must match
            if nxt != c.get("state_after"):
                policy_ok = False
                issues.append({"t": t, "candidate": idx, "error": "state_after mismatch",
                               "expected": nxt, "got": c.get("state_after")})
                break

            # energy_after must match exact energy of nxt
            ok_es, es, msg_es = _parse_int_scalar(c.get("energy_after"))
            if not ok_es:
                policy_ok = False
                issues.append({"t": t, "candidate": idx, "error": f"energy_after not exact int: {msg_es}"})
                break
            expected_energy = _energy(energy_form, nxt)
            if es != expected_energy:
                policy_ok = False
                issues.append({"t": t, "candidate": idx, "error": "energy_after mismatch",
                               "expected": str(expected_energy), "got": str(es)})
                break
            legal_with_energy.append((expected_energy, g, _state_to_str(nxt), idx))

        if not policy_ok:
            break

        if not legal_with_energy:
            policy_ok = False
            issues.append({"t": t, "error": "no legal candidates"})
            break

        # Compute deterministic winner
        min_e = min(x[0] for x in legal_with_energy)
        tied = [(g, s, idx) for e, g, s, idx in legal_with_energy if e == min_e]
        winner_g, winner_state_str, winner_idx = sorted(tied, key=lambda x: (x[0], x[1]))[0]

        chosen_idx = int(st["chosen_idx"])
        if chosen_idx != winner_idx:
            policy_ok = False
            issues.append({
                "t": t,
                "error": "chosen_idx does not match deterministic min-energy+tiebreak winner",
                "winner": {"generator": winner_g, "state_after": winner_state_str, "idx": winner_idx, "energy": str(min_e)},
                "chosen_idx": chosen_idx,
            })
            break

        # state_after must match winner candidate
        if sa != candidates[chosen_idx]["state_after"]:
            policy_ok = False
            issues.append({"t": t, "error": "state_after does not match chosen candidate", "state_after": sa})
            break

        # invariant_diff delta_energy must equal energy(sa)-energy(sb)
        ok_de, de, msg_de = _parse_int_scalar(st["invariant_diff"].get("delta_energy"))
        if not ok_de:
            policy_ok = False
            issues.append({"t": t, "error": f"delta_energy invalid: {msg_de}"})
            break
        expected_de = _energy(energy_form, sa) - _energy(energy_form, sb)
        if de != expected_de:
            policy_ok = False
            issues.append({"t": t, "error": "delta_energy mismatch", "expected": str(expected_de), "got": str(de)})
            break

        visited.append(sb)
        if sa == target:
            reached = True
        visited.append(sa)

    if not policy_ok:
        results.append(GateResult(
            "gate_4_policy_replay",
            GateStatus.FAIL,
            "policy replay invalid",
            {"issues": issues},
        ))
        return results

    results.append(GateResult(
        "gate_4_policy_replay",
        GateStatus.PASS,
        "policy trace consistent with min-energy legal + deterministic tie-break",
        {"steps": len(steps)},
    ))

    # Gate 5 — Separation claim consistency: reachable but not reached by policy
    claim = obj["claim"]
    if claim["target_reachable_under_generators"] is not True:
        results.append(GateResult(
            "gate_5_separation_claim",
            GateStatus.FAIL,
            "claim.target_reachable_under_generators must be true for this family",
        ))
        return results
    if claim["target_reached_under_policy"] is not False:
        results.append(GateResult(
            "gate_5_separation_claim",
            GateStatus.FAIL,
            "claim.target_reached_under_policy must be false for separation witness",
        ))
        return results
    if reached:
        results.append(GateResult(
            "gate_5_separation_claim",
            GateStatus.FAIL,
            "policy trace actually reached target, contradicting separation claim",
        ))
        return results

    # Optional cycle detection check (if present)
    if "cycle_detected" in obj["policy_run"]:
        cyc = bool(obj["policy_run"]["cycle_detected"])
        if cyc:
            cs = obj["policy_run"].get("cycle_state")
            if not isinstance(cs, dict) or not _in_caps(cs, N):
                results.append(GateResult(
                    "gate_5_separation_claim",
                    GateStatus.FAIL,
                    "cycle_detected=true but cycle_state missing/invalid",
                ))
                return results

    results.append(GateResult(
        "gate_5_separation_claim",
        GateStatus.PASS,
        "separation witness verified: reachable under generators, not reached by energy policy run",
    ))

    # Gate 6 — invariant_diff coherence (summary fields)
    inv = obj["invariant_diff"]
    expected_len = len(path)
    if inv.get("separation_mode") != "reachable_but_not_reached_by_policy":
        results.append(GateResult(
            "gate_6_invariant_diff",
            GateStatus.FAIL,
            "invariant_diff.separation_mode mismatch",
            {"got": inv.get("separation_mode")},
        ))
        return results
    if inv.get("witness_path_length") != expected_len:
        results.append(GateResult(
            "gate_6_invariant_diff",
            GateStatus.FAIL,
            "invariant_diff.witness_path_length mismatch",
            {"expected": expected_len, "got": inv.get("witness_path_length")},
        ))
        return results
    if inv.get("policy_steps_executed") != len(obj["policy_run"]["steps"]):
        results.append(GateResult(
            "gate_6_invariant_diff",
            GateStatus.FAIL,
            "invariant_diff.policy_steps_executed mismatch",
            {"expected": len(obj["policy_run"]["steps"]), "got": inv.get("policy_steps_executed")},
        ))
        return results
    results.append(GateResult(
        "gate_6_invariant_diff",
        GateStatus.PASS,
        "invariant_diff summary fields consistent",
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
        ("invalid_policy_wrong_choice.json", False, "gate_4_policy_replay"),
        ("invalid_witness_not_target.json", False, "gate_3_witness_reachability"),
        ("invalid_digest_mismatch.json", False, "gate_2_canonical_hash"),
    ]

    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        obj = _load_json(os.path.join(fx, name))
        res = validate_cert(obj)
        passed = _report_ok(res)
        if passed != should_pass:
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
        print("=== QA_ENERGY_CAPABILITY_SEPARATION_CERT.v1 SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_ENERGY_CAPABILITY_SEPARATION_CERT.v1 validator")
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

