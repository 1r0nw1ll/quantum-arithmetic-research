#!/usr/bin/env python3
"""
qa_self_improvement_cert_validate.py

**STUB VALIDATOR** for QA_SELF_IMPROVEMENT_CERT.v1.

Certificate family number: unassigned (session `lab-self-improvement`,
2026-04-05; not yet registered in the meta-validator — promotion to a live
cert number is explicitly deferred to a follow-up session per the kickoff
doc's guardrails).

Scope
-----
Certifies that a ``SelfImprovementAgentV2`` cycle trace respects the safety
invariants of the [191]/[192] framework:

    V1 LEVEL_TAGGED    — every proposal carries a ``level`` key and a
                         ``cert_ref`` pointing at QA_BATESON_LEARNING_LEVELS_CERT.v1
    V2 NO_L2B_APPLIED  — no L_2b or L_3 proposal ends up in ``applied``
    V3 LYAP_MONOTONE   — for every ``CONSISTENT``-verdict cycle,
                         ``lyap_post <= lyap_pre + tol``
    V4 FP_CANDIDATES   — ``fixed_point_candidates`` has the three required
                         keys (A/B/C) with numeric values
    V5 SELF_TEST       — pass fixture validates; fail fixture does not

Parent certs
------------
    [122] QA_EMPIRICAL_OBSERVATION_CERT.v1  — base observation format
    [191] QA_BATESON_LEARNING_LEVELS_CERT.v1 — filtration / level semantics
    [192] QA_DUAL_EXTREMALITY_24_CERT.v1    — Pisano operator grounding

Schema
------
A cert "trace" fixture is a JSON object::

    {
      "schema_version": "QA_SELF_IMPROVEMENT_CERT.v1",
      "agent": "self_improvement_agent_v2",
      "cycles": [
         {
           "verdict": "...",
           "proposals": [ { "level": ..., "cert_ref": ..., ... }, ... ],
           "applied":   [ { "level": ..., ... }, ... ],
           "deferred":  [ ... ],
           "lyapunov":  { "name": "...", "pre": ..., "post": ..., "tol": ... },
           "fixed_point_candidates": { "A_trace_compression": ..., ... }
         },
         ...
      ]
    }
"""
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator — validates SelfImprovementAgentV2 cycle traces; "
    "integer state space; no observer, no continuous dynamics"
)

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "QA_SELF_IMPROVEMENT_CERT.v1"
REQUIRED_FP_KEYS = (
    "A_trace_compression",
    "B_routing_barycenter",
    "C_pisano_periodicity",
)
ALLOWED_LEVELS = {None, "L_1", "L_2a", "L_2b", "L_3"}
FORBIDDEN_APPLIED_LEVELS = {"L_2b", "L_3"}


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(path: Path) -> Tuple[List[str], List[str]]:
    errs: List[str] = []
    warns: List[str] = []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ([f"failed to parse JSON: {exc}"], warns)

    if payload.get("schema_version") != SCHEMA_VERSION:
        errs.append(
            f"schema_version mismatch: got {payload.get('schema_version')!r}"
        )

    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        errs.append("cycles: expected non-empty list")
        return (errs, warns)

    for idx, cycle in enumerate(cycles):
        tag = f"cycle[{idx}]"

        # V1 LEVEL_TAGGED
        proposals = cycle.get("proposals")
        if not isinstance(proposals, list):
            errs.append(f"{tag}.proposals: expected list")
        else:
            for j, prop in enumerate(proposals):
                if "level" not in prop:
                    errs.append(f"{tag}.proposals[{j}]: missing 'level' (V1)")
                elif prop["level"] not in ALLOWED_LEVELS:
                    errs.append(
                        f"{tag}.proposals[{j}]: level={prop['level']!r} not in "
                        f"{sorted(str(x) for x in ALLOWED_LEVELS)} (V1)"
                    )
                if "cert_ref" not in prop:
                    errs.append(
                        f"{tag}.proposals[{j}]: missing 'cert_ref' (V1)"
                    )

        # V2 NO_L2B_APPLIED
        applied = cycle.get("applied")
        if not isinstance(applied, list):
            errs.append(f"{tag}.applied: expected list")
        else:
            for j, act in enumerate(applied):
                lvl = act.get("level")
                if lvl in FORBIDDEN_APPLIED_LEVELS:
                    errs.append(
                        f"{tag}.applied[{j}]: level={lvl!r} is forbidden in "
                        f"applied (V2) — must be deferred per [191]"
                    )

        # V3 LYAP_MONOTONE
        lyap = cycle.get("lyapunov")
        verdict = cycle.get("verdict")
        if not isinstance(lyap, dict):
            errs.append(f"{tag}.lyapunov: expected dict")
        else:
            pre = lyap.get("pre")
            post = lyap.get("post")
            tol = lyap.get("tol")
            if not all(_is_number(x) for x in (pre, post, tol)):
                errs.append(
                    f"{tag}.lyapunov: pre/post/tol must be numeric"
                )
            elif verdict == "CONSISTENT" and post > pre + tol:
                errs.append(
                    f"{tag}.lyapunov: CONSISTENT verdict but "
                    f"post={post} > pre+tol={pre + tol} (V3)"
                )

        # V4 FP_CANDIDATES
        fp = cycle.get("fixed_point_candidates")
        if not isinstance(fp, dict):
            errs.append(f"{tag}.fixed_point_candidates: expected dict")
        else:
            for key in REQUIRED_FP_KEYS:
                if key not in fp:
                    errs.append(
                        f"{tag}.fixed_point_candidates: missing {key!r} (V4)"
                    )
                elif not _is_number(fp[key]):
                    errs.append(
                        f"{tag}.fixed_point_candidates.{key}: "
                        f"expected number, got {type(fp[key]).__name__} (V4)"
                    )

    return (errs, warns)


def _self_test() -> Dict[str, Any]:
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("pass_trace.json", True),   # should validate cleanly
        ("fail_trace.json", False),  # should produce at least one error
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
            errs, _warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append(
                {"fixture": fname, "ok": False,
                 "error": f"expected PASS but got errors: {errs}"}
            )
            all_ok = False
        elif (not should_pass) and passed:
            results.append(
                {"fixture": fname, "ok": False,
                 "error": "expected FAIL but got PASS"}
            )
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="QA Self-Improvement Cert (stub) validator"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or [
        str(p) for p in (Path(__file__).parent / "fixtures").glob("*.json")
    ]
    total = 0
    for path in paths:
        p = Path(path)
        print(f"Validating {p.name}...")
        errs, warns = validate(p)
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total += len(errs)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
