#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=phase_conjugate_fixtures"
"""QA Bearden Phase Conjugate Cert family [155] — certifies the structural
parallel between Bearden's pumped phase conjugate mirror theory and the
empirical QCI opposite-sign discovery.

STRUCTURAL PARALLEL:
- Bearden: "stress is a pumper" — stress acts as a pump beam creating
  phase conjugation (order at one level, scattered conjugate at another)
- QA: Global QCI rises during stress (pump beam = coupling tightens)
  while local QCI drops (conjugate response = trajectories scatter)
- QCI_gap = QCI_local - QCI_global = the phase conjugation signature

SOURCE: Will Dale insight 2026-04-01; Bearden scalar EM; SVP-adjacent
physics (Keely → Dale Pond → Bearden lineage).

Checks: BPC_1 (schema), BPC_MODEL (Bearden model declared with source),
BPC_MAP (QA mapping: pump→global, conjugate→local), BPC_SIGN (opposite-sign
pattern), BPC_EMP (empirical evidence), BPC_SVP (SVP lineage),
BPC_W (>=1 domain witness), BPC_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_BEARDEN_PHASE_CONJUGATE_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # BPC_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("BPC_1", f"schema_version must be {SCHEMA}")

    # BPC_MODEL — Bearden model declared
    model = cert.get("bearden_model", {})
    if not model.get("source"):
        err("BPC_MODEL", "bearden_model.source missing")
    if not model.get("core_claim"):
        err("BPC_MODEL", "bearden_model.core_claim missing")
    if not model.get("pump_beam"):
        err("BPC_MODEL", "bearden_model.pump_beam missing")
    if not model.get("conjugate_response"):
        err("BPC_MODEL", "bearden_model.conjugate_response missing")

    # BPC_MAP — QA mapping
    qa_map = cert.get("qa_mapping", {})
    if not qa_map:
        err("BPC_MAP", "qa_mapping section missing")
    else:
        if not qa_map.get("pump_beam_qa"):
            err("BPC_MAP", "qa_mapping.pump_beam_qa missing")
        if not qa_map.get("conjugate_response_qa"):
            err("BPC_MAP", "qa_mapping.conjugate_response_qa missing")
        if not qa_map.get("phase_conjugation_qa"):
            err("BPC_MAP", "qa_mapping.phase_conjugation_qa missing")

        # BPC_SIGN — opposite-sign pattern
        if qa_map.get("sign_opposition") is not True:
            err("BPC_SIGN", "qa_mapping.sign_opposition must be true")
        g_sign = qa_map.get("qci_global_sign", "")
        l_sign = qa_map.get("qci_local_sign", "")
        if g_sign and l_sign and g_sign == l_sign:
            err("BPC_SIGN", f"signs must be opposite: global={g_sign}, local={l_sign}")

    # BPC_EMP — empirical evidence
    emp = cert.get("empirical_evidence", {})
    if emp:
        local_r = emp.get("qci_local_partial_r")
        global_r = emp.get("qci_global_partial_r")
        if local_r is not None and global_r is not None:
            if local_r * global_r > 0:
                warnings.append("BPC_EMP: local and global partial r have same sign — check data")
    else:
        warnings.append("BPC_EMP: no empirical_evidence section")

    # BPC_SVP — SVP lineage
    svp = cert.get("svp_connection", {})
    if svp:
        if not svp.get("lineage"):
            warnings.append("BPC_SVP: svp_connection.lineage not specified")
    else:
        warnings.append("BPC_SVP: no svp_connection section")

    # BPC_W — at least one witness
    witnesses = cert.get("empirical_evidence", {})
    if not witnesses.get("scripts"):
        warnings.append("BPC_W: no script references in empirical_evidence")

    # Derive result from cert or from errors
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("BPC_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("BPC_F: declared FAIL but no fail_ledger entries and all checks pass")

    return {
        "ok": not has_errors,
        "errors": errors,
        "warnings": warnings,
        "schema": SCHEMA,
    }


def self_test():
    """Run validator against bundled fixtures."""
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
    parser = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    parser.add_argument("--self-test", action="store_true", help="Run self-test against fixtures")
    parser.add_argument("cert_file", nargs="?", help="Path to certificate JSON")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if args.cert_file:
        with open(args.cert_file, "r", encoding="utf-8") as f:
            cert = json.load(f)
        result = validate(cert)
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
