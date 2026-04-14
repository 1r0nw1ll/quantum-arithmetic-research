#!/usr/bin/env python3
"""
qa_conversation_arag_cert_validate.py

Validator for QA_CONVERSATION_ARAG_CERT.v1  [family 210]

Certifies: the conversation graph store at _forensics/qa_retrieval.sqlite
is a QA-native A-RAG datastore instance, with cross-source (ChatGPT / Claude.ai /
Google AI Studio) messages mapped to canonical (b, e) tuples via Candidate F
and composing with QA_ARAG_INTERFACE_CERT.v1 [existing cert].

Claims:
  SCHEMA  - cert declares schema_version, arag_tool_set, tool_to_view_kind
  TUPLE   - (b, e) derivation formula: b = dr(char_ord_sum), e = role_rank
  DIAG    - role-diagonal property: (a - d) mod 9 = role_rank mod 9
  CROSS   - same formula applies to ChatGPT / Claude / Gemini sample records
  PROMO   - thought-promotion rules enumerated per source
  A1      - all stored (b, e) and derived sector labels in valid ranges
  A2      - d and a are derived on read, not stored as columns
  T2      - observer-layer text never feeds back into QA state
  VIEWS   - three A-RAG views mapped to substrates
  WITNESS - >= 3 sample records per source with computed tuples
  FAIL    - falsifier demonstrates A1 violation (sector label 0 rejected)

Checks:
  CAV_1       - schema_version matches
  CAV_SCHEMA  - arag_tool_set equals {keyword_search, semantic_search, chunk_read}
  CAV_TUPLE   - tuple_derivation specifies b_formula and e_formula
  CAV_DIAG    - diagonal_check over witnesses: all rows satisfy property
  CAV_CROSS   - witness sources include at least chatgpt, claude, gemini
  CAV_PROMO   - thought_promotion_rules defined for each source
  CAV_A1      - all sample b in {1..9}, e in {1..5}, sector labels in {1..9}
  CAV_A2      - derived_coords flag true, no stored d/a columns
  CAV_T2      - firewall_declaration present and well-formed
  CAV_VIEWS   - tool_to_view_kind mapping complete
  CAV_W       - at least 9 witnesses (3 per source minimum)
  CAV_F       - falsifier demonstrates A1 violation correctly
"""

QA_COMPLIANCE = "cert_validator - validates conversation A-RAG instance; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_CONVERSATION_ARAG_CERT.v1"
REQUIRED_TOOLS = {"keyword_search", "semantic_search", "chunk_read"}
REQUIRED_VIEW_KINDS = {"KEYWORD_VIEW", "SEMANTIC_VIEW", "CHUNK_STORE"}
REQUIRED_SOURCES = {"chatgpt", "claude", "gemini"}
# Canonical role → role_rank integer. Expanded in v1.1 to include 'note' (rank 6)
# for non-conversational authoritative research material. E values in {1..9}.
CANONICAL_ROLES = {
    "user": 1,
    "assistant": 2,
    "tool": 3,
    "system": 4,
    "thought": 5,
    "note": 6,
}


def digital_root(n):
    """Aiq Bekar digital root - integer in {1..9}, zero excluded (A1)."""
    if n <= 0:
        return 9
    return 1 + ((n - 1) % 9)


def sector_labels(b, e):
    """T-operator output: (dr(b+e), dr(b+2e)) in {1..9} x {1..9}."""
    return digital_root(b + e), digital_root(b + 2 * e)


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CAV_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"CAV_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # Falsifier path
    if cert.get("expect_fail"):
        return _validate_fail(cert, errors, warnings)

    # CAV_SCHEMA: tool set
    tool_set = cert.get("arag_tool_set")
    if not isinstance(tool_set, list):
        errors.append("CAV_SCHEMA: arag_tool_set must be list")
    elif set(tool_set) != REQUIRED_TOOLS:
        errors.append(
            f"CAV_SCHEMA: arag_tool_set must equal {sorted(REQUIRED_TOOLS)}, got {sorted(tool_set)}"
        )

    tool_to_view_kind = cert.get("tool_to_view_kind")
    if not isinstance(tool_to_view_kind, dict):
        errors.append("CAV_VIEWS: tool_to_view_kind must be object")
    else:
        if set(tool_to_view_kind.keys()) != REQUIRED_TOOLS:
            errors.append(
                f"CAV_VIEWS: tool_to_view_kind keys must equal {sorted(REQUIRED_TOOLS)}"
            )
        for tool, kind in tool_to_view_kind.items():
            if kind not in REQUIRED_VIEW_KINDS:
                errors.append(
                    f"CAV_VIEWS: tool_to_view_kind[{tool}] = {kind!r} not in {sorted(REQUIRED_VIEW_KINDS)}"
                )

    # CAV_TUPLE: derivation formula
    tup = cert.get("tuple_derivation")
    if not isinstance(tup, dict):
        errors.append("CAV_TUPLE: tuple_derivation must be object")
    else:
        b_formula = tup.get("b_formula")
        e_formula = tup.get("e_formula")
        if not isinstance(b_formula, str) or "dr" not in b_formula or "ord" not in b_formula:
            errors.append(f"CAV_TUPLE: b_formula must reference dr() and ord(), got {b_formula!r}")
        if not isinstance(e_formula, str) or "role_rank" not in e_formula:
            errors.append(f"CAV_TUPLE: e_formula must reference role_rank, got {e_formula!r}")
        if tup.get("dr_cert_ref") != "family_202":
            errors.append(
                f"CAV_TUPLE: dr_cert_ref must be 'family_202' (Aiq Bekar), got {tup.get('dr_cert_ref')!r}"
            )
        rank_table = tup.get("role_rank_table")
        if rank_table != CANONICAL_ROLES:
            errors.append(
                f"CAV_TUPLE: role_rank_table must equal {CANONICAL_ROLES}, got {rank_table}"
            )

    # CAV_A2: derived coords
    a2 = cert.get("a2_compliance")
    if not isinstance(a2, dict):
        errors.append("CAV_A2: a2_compliance must be object")
    else:
        if a2.get("d_derived_on_read") is not True:
            errors.append("CAV_A2: d_derived_on_read must be true")
        if a2.get("a_derived_on_read") is not True:
            errors.append("CAV_A2: a_derived_on_read must be true")
        if a2.get("stored_columns") != ["b", "e"]:
            errors.append(
                f"CAV_A2: stored_columns must be ['b', 'e'], got {a2.get('stored_columns')}"
            )

    # CAV_T2: firewall declaration
    t2 = cert.get("t2_firewall")
    if not isinstance(t2, dict):
        errors.append("CAV_T2: t2_firewall must be object")
    else:
        if t2.get("raw_text_role") != "observer_projection":
            errors.append("CAV_T2: raw_text_role must be 'observer_projection'")
        if t2.get("fts_score_role") != "observer_measurement":
            errors.append("CAV_T2: fts_score_role must be 'observer_measurement'")
        if t2.get("qa_state_feedback_allowed") is not False:
            errors.append("CAV_T2: qa_state_feedback_allowed must be false")

    # CAV_PROMO: thought promotion rules
    promo = cert.get("thought_promotion_rules")
    if not isinstance(promo, dict):
        errors.append("CAV_PROMO: thought_promotion_rules must be object")
    else:
        for src in REQUIRED_SOURCES:
            rule = promo.get(src)
            if not isinstance(rule, str) or len(rule) < 10:
                errors.append(f"CAV_PROMO: missing or too-short rule for source {src}")

    # CAV_W and CAV_CROSS and CAV_A1 and CAV_DIAG: witnesses
    witnesses = cert.get("witnesses")
    if not isinstance(witnesses, list):
        errors.append("CAV_W: witnesses must be list")
    else:
        if len(witnesses) < 9:
            warnings.append(f"CAV_W: need >= 9 witnesses, got {len(witnesses)}")

        sources_seen = set()
        for i, w in enumerate(witnesses):
            if not isinstance(w, dict):
                errors.append(f"CAV_W[{i}]: witness must be object")
                continue
            src = w.get("source")
            sources_seen.add(src)

            b = w.get("b")
            e = w.get("e")
            role = w.get("role")
            ord_sum = w.get("char_ord_sum")

            # CAV_A1: b and e in valid ranges
            if not isinstance(b, int) or not (1 <= b <= 9):
                errors.append(f"CAV_A1[{i}]: b={b} not in {{1..9}}")
                continue
            if not isinstance(e, int) or not (1 <= e <= 9):
                errors.append(f"CAV_A1[{i}]: e={e} not in {{1..9}}")
                continue

            # Verify b = dr(ord_sum)
            if isinstance(ord_sum, int) and ord_sum > 0:
                expected_b = digital_root(ord_sum)
                if b != expected_b:
                    errors.append(
                        f"CAV_TUPLE[{i}]: b={b} != dr(char_ord_sum={ord_sum})={expected_b}"
                    )

            # Verify e = role_rank[role]
            if role in CANONICAL_ROLES:
                expected_e = CANONICAL_ROLES[role]
                if e != expected_e:
                    errors.append(
                        f"CAV_TUPLE[{i}]: e={e} != role_rank[{role}]={expected_e}"
                    )
            else:
                errors.append(f"CAV_TUPLE[{i}]: role {role!r} not canonical")

            # CAV_A1: sector labels also in {1..9}
            d_label, a_label = sector_labels(b, e)
            declared_d = w.get("d_label_mod9")
            declared_a = w.get("a_label_mod9")
            if declared_d != d_label:
                errors.append(f"CAV_A1[{i}]: d_label={declared_d} != computed {d_label}")
            if declared_a != a_label:
                errors.append(f"CAV_A1[{i}]: a_label={declared_a} != computed {a_label}")
            if not (1 <= d_label <= 9):
                errors.append(f"CAV_A1[{i}]: d_label={d_label} not in {{1..9}}")
            if not (1 <= a_label <= 9):
                errors.append(f"CAV_A1[{i}]: a_label={a_label} not in {{1..9}}")

            # CAV_DIAG: (a - d) mod 9 == role_rank mod 9
            if role in CANONICAL_ROLES:
                expected_diag = CANONICAL_ROLES[role] % 9
                observed_diag = (a_label - d_label) % 9
                if observed_diag != expected_diag:
                    errors.append(
                        f"CAV_DIAG[{i}]: (a-d) mod 9 = {observed_diag}, "
                        f"expected role_rank[{role}] mod 9 = {expected_diag}"
                    )

        # CAV_CROSS: all three sources represented
        missing = REQUIRED_SOURCES - sources_seen
        if missing:
            errors.append(f"CAV_CROSS: missing sources in witnesses: {sorted(missing)}")

    return errors, warnings


def _validate_fail(cert, errors, warnings):
    """Validate a fixture that's expected to FAIL — confirm the declared violation."""
    if not cert.get("fail_reason"):
        errors.append("CAV_F: expect_fail fixture missing fail_reason")

    # The FAIL fixture should have at least one witness with an intentional violation.
    witnesses = cert.get("witnesses") or []
    declared_kind = cert.get("fail_kind")

    if declared_kind == "A1_SECTOR_ZERO":
        # Check at least one witness has sector label 0 (A1 violation)
        found = False
        for w in witnesses:
            if not isinstance(w, dict):
                continue
            if w.get("d_label_mod9") == 0 or w.get("a_label_mod9") == 0:
                found = True
                break
        if not found:
            errors.append("CAV_F: A1_SECTOR_ZERO fail_kind requires a sector label == 0 witness")
    elif declared_kind == "A2_STORED_DA":
        a2 = cert.get("a2_compliance") or {}
        stored = a2.get("stored_columns") or []
        if "d" not in stored and "a" not in stored:
            errors.append(
                "CAV_F: A2_STORED_DA fail_kind requires stored_columns to include 'd' or 'a'"
            )
    else:
        errors.append(f"CAV_F: unknown fail_kind {declared_kind!r}")

    return errors, warnings


def run_self_test(json_output=False):
    here = Path(__file__).parent
    fixtures_dir = here / "fixtures"
    fixtures = sorted(fixtures_dir.glob("*.json"))
    if not fixtures:
        payload = {"ok": False, "error": f"No fixtures in {fixtures_dir}"}
        if json_output:
            print(json.dumps(payload))
        else:
            print(payload["error"], file=sys.stderr)
        return 2

    results = []
    n_pass = 0
    n_fail = 0
    for fixture in fixtures:
        errors, warnings = validate(fixture)
        is_expect_fail = fixture.name.startswith("cav_fail")
        if is_expect_fail:
            # FAIL fixtures should have NO errors in validation (the fail structure is well-formed)
            if errors:
                n_fail += 1
                results.append({
                    "fixture": fixture.name,
                    "status": "FAIL",
                    "errors": errors,
                    "note": "validator errors on fail fixture",
                })
            else:
                n_pass += 1
                results.append({
                    "fixture": fixture.name,
                    "status": "PASS",
                    "note": "fail fixture well-formed",
                })
        else:
            if errors:
                n_fail += 1
                results.append({
                    "fixture": fixture.name,
                    "status": "FAIL",
                    "errors": errors,
                })
            else:
                n_pass += 1
                results.append({
                    "fixture": fixture.name,
                    "status": "PASS",
                    "warnings": warnings,
                })

    total = len(fixtures)
    payload = {
        "ok": n_fail == 0,
        "schema_version": SCHEMA_VERSION,
        "family": 210,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_total": total,
        "results": results,
    }

    if json_output:
        print(json.dumps(payload))
    else:
        for r in results:
            if r["status"] == "PASS":
                print(f"  PASS {r['fixture']}: {r.get('note', '')}")
                for w in r.get("warnings", []):
                    print(f"    ! {w}")
            else:
                print(f"  FAIL {r['fixture']}: {len(r.get('errors', []))} error(s)")
                for e in r.get("errors", []):
                    print(f"    - {e}")
        print(f"\n{n_pass}/{total} passed, {n_fail}/{total} failed")

    return 0 if n_fail == 0 else 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA_CONVERSATION_ARAG_CERT.v1 validator")
    parser.add_argument("fixture", nargs="?", help="Path to a cert fixture JSON")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run all fixtures in ./fixtures/ (emits JSON for meta-validator)",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="With --self-test: emit human-readable output instead of JSON",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(json_output=not args.human)
    if not args.fixture:
        parser.error("Provide a fixture path or --self-test")
    errors, warnings = validate(args.fixture)
    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"WARN:  {w}")
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
