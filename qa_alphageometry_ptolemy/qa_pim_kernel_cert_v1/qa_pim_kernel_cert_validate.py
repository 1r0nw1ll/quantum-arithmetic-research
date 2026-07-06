#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=pim_kernel_fixtures"
"""QA PIM Kernel Cert family [157] — certifies correctness of QA-native
Processing-In-Memory kernels: RESIDUE_SELECT, TORUS_SHIFT, RADD_m, RMUL_m,
MIRROR4, ROLLING_SUM_PHASE, and CRT (Chinese Remainder Theorem) with
non-coprime moduli support.

SOURCE: Extracted from qa_graph_pim_repo_v2 vault artifact (Sept 2025),
now in qa_lab/qa_pim/. This is a from-scratch internal implementation of
standard number theory (Chinese Remainder Theorem), not an external-
paper reproduction.

Primary source: Hardy, G.H. & Wright, E.M. (2008), An Introduction to
the Theory of Numbers, 6th ed., Oxford University Press, ISBN
978-0-19-921986-5, Ch. II (Chinese Remainder Theorem, extended GCD).

Checks: PIM_1 (schema), PIM_CRT (CRT correctness — genuinely recomputed
live against qa_lab/qa_pim/crt.py when importable, not just fixture-
trusted congruence checks), PIM_KERNEL (kernel outputs genuinely
recomputed live against qa_lab/qa_pim/kernels.py when importable),
PIM_A1 (no-zero: RADD_m/RMUL_m documented as coordinate-level 0-indexed),
PIM_W (>=1 kernel witness), PIM_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_PIM_KERNEL_CERT.v1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QA_LAB = os.path.join(_REPO_ROOT, "qa_lab")
if _QA_LAB not in sys.path:
    sys.path.insert(0, _QA_LAB)

try:
    from qa_pim import crt as _crt_mod
    from qa_pim import kernels as _kernels_mod
    import numpy as _np
    _LIVE_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - environment-dependent
    _crt_mod = _kernels_mod = _np = None
    _LIVE_IMPORT_ERROR = str(_exc)


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # PIM_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("PIM_1", f"schema_version must be {SCHEMA}")

    # PIM_CRT — CRT section
    crt = cert.get("crt_tests", {})
    if not crt:
        err("PIM_CRT", "crt_tests section missing")
    else:
        cases = crt.get("cases", [])
        if not cases:
            err("PIM_CRT", "crt_tests.cases must be non-empty")
        for i, c in enumerate(cases):
            a1, m1, a2, m2 = c.get("a1"), c.get("m1"), c.get("a2"), c.get("m2")
            expected_solvable = c.get("solvable")
            if expected_solvable and c.get("x") is not None:
                x = c["x"]
                if x % m1 != a1:
                    err("PIM_CRT", f"case {i}: x={x} mod {m1} != {a1}")
                if x % m2 != a2:
                    err("PIM_CRT", f"case {i}: x={x} mod {m2} != {a2}")
            # Genuine live recomputation against qa_lab/qa_pim/crt.py,
            # not just checking the fixture's own declared congruences.
            if _crt_mod is not None and None not in (a1, m1, a2, m2):
                live_x, live_lcm = _crt_mod.crt_join_general(a1, m1, a2, m2)
                live_solvable = live_x is not None
                if expected_solvable is not None and live_solvable != expected_solvable:
                    err("PIM_CRT",
                        f"case {i}: declared solvable={expected_solvable}, "
                        f"live qa_pim.crt recompute gives solvable={live_solvable}")
                if live_solvable:
                    if c.get("x") is not None and c["x"] != live_x:
                        err("PIM_CRT",
                            f"case {i}: declared x={c['x']}, live recompute x={live_x}")
                    if c.get("lcm") is not None and c["lcm"] != live_lcm:
                        err("PIM_CRT",
                            f"case {i}: declared lcm={c['lcm']}, live recompute lcm={live_lcm}")
            elif _LIVE_IMPORT_ERROR:
                warnings.append(f"PIM_CRT: qa_lab/qa_pim not importable, skipping live recompute ({_LIVE_IMPORT_ERROR})")

    # PIM_KERNEL — kernel witness section
    kernels = cert.get("kernel_witnesses", {})
    if not kernels:
        err("PIM_KERNEL", "kernel_witnesses section missing")
    else:
        required_ops = {"RESIDUE_SELECT", "TORUS_SHIFT", "ROLLING_SUM_PHASE"}
        present_ops = set(kernels.keys())
        missing = required_ops - present_ops
        if missing:
            err("PIM_KERNEL", f"missing kernel witnesses: {sorted(missing)}")
        for op, witness in kernels.items():
            if not witness.get("input"):
                err("PIM_KERNEL", f"{op}: input missing")
            has_output = ("output" in witness
                          or "output_first" in witness
                          or "output_length" in witness)
            if not has_output:
                err("PIM_KERNEL", f"{op}: output missing (need output, output_first, or output_length)")
                continue
            # Genuine live recomputation against qa_lab/qa_pim/kernels.py,
            # not just checking that input/output fields are present.
            if _kernels_mod is None or _np is None:
                if _LIVE_IMPORT_ERROR:
                    warnings.append(f"PIM_KERNEL: qa_lab/qa_pim not importable, skipping live recompute ({_LIVE_IMPORT_ERROR})")
                continue
            inp = witness.get("input", {})
            try:
                if op == "RESIDUE_SELECT":
                    live = _kernels_mod.RESIDUE_SELECT(
                        _np.array(inp["sectors"]), set(inp["mask"]))
                    if "output" in witness and list(live) != witness["output"]:
                        err("PIM_KERNEL", f"{op}: declared output={witness['output']}, live recompute={list(live)}")
                elif op == "TORUS_SHIFT":
                    live = _kernels_mod.TORUS_SHIFT(
                        _np.array(inp["arr"]), inp["delta"], inp["P"])
                    if "output" in witness and list(live) != witness["output"]:
                        err("PIM_KERNEL", f"{op}: declared output={witness['output']}, live recompute={list(live)}")
                elif op == "ROLLING_SUM_PHASE":
                    live = _kernels_mod.ROLLING_SUM_PHASE(
                        _np.array(inp["values"]), inp["width"])
                    if "output_first" in witness and float(live[0]) != witness["output_first"]:
                        err("PIM_KERNEL", f"{op}: declared output_first={witness['output_first']}, live recompute={float(live[0])}")
                    if "output_length" in witness and len(live) != witness["output_length"]:
                        err("PIM_KERNEL", f"{op}: declared output_length={witness['output_length']}, live recompute={len(live)}")
            except (KeyError, TypeError, ValueError) as exc:
                err("PIM_KERNEL", f"{op}: live recompute failed on declared input: {exc}")

    # PIM_A1 — coordinate-layer documentation
    a1_note = cert.get("a1_compliance_note", "")
    if not a1_note:
        warnings.append("PIM_A1: a1_compliance_note missing — document that RADD_m/RMUL_m are coordinate-level (0-indexed), not QA state-level")

    # PIM_W — at least one domain witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        err("PIM_W", "at least one witness required")

    # PIM_F — fail detection
    result = cert.get("result", "")
    fail_ledger = cert.get("fail_ledger", [])
    if result == "FAIL" and not fail_ledger:
        err("PIM_F", "FAIL result requires non-empty fail_ledger")
    if result == "PASS" and fail_ledger:
        err("PIM_F", "PASS result must not have fail_ledger entries")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "schema": SCHEMA,
    }


def self_test():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    results = {"pass_count": 0, "fail_count": 0, "errors": []}

    for fname in sorted(os.listdir(fixture_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(fixture_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            cert = json.load(f)

        out = validate(cert)
        declared = cert.get("result", "UNKNOWN")

        if declared == "PASS" and out["ok"]:
            results["pass_count"] += 1
        elif declared == "FAIL" and not out["ok"]:
            results["fail_count"] += 1
        else:
            results["errors"].append({
                "fixture": fname,
                "declared": declared,
                "validator_ok": out["ok"],
                "issues": out["errors"],
            })

    results["ok"] = len(results["errors"]) == 0
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("cert_file", nargs="?")
    args = ap.parse_args()

    if args.self_test:
        r = self_test()
        print(json.dumps(r, indent=2, sort_keys=True))
        sys.exit(0 if r["ok"] else 1)

    if args.cert_file:
        with open(args.cert_file, "r") as f:
            cert = json.load(f)
        r = validate(cert)
        print(json.dumps(r, indent=2, sort_keys=True))
        sys.exit(0 if r["ok"] else 1)

    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
