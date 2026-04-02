#!/usr/bin/env python3
"""QA Human Needs SDT Cert family [161] — certifies structural alignment
between Ryan & Deci Self-Determination Theory (3 validated needs: Autonomy,
Competence, Relatedness) and QA paired architecture.

Core claim:
  SDT's 3 basic needs decompose into Robbins' 6 human needs, each pair
  mapping to a structurally distinct QA pair type:
    Autonomy   = (b, e)       generators
    Competence = (d, DeltaT)  state + derivative
    Relatedness= (a, SigmaT)  reach + integral

Structural predictions (5/5 confirmed against SDT literature):
  PRED_1: Autonomy prerequisite (b,e must exist before d,a)
  PRED_2: Satisfaction/frustration partially independent (pair elements independent)
  PRED_3: Growth/contribution temporally downstream (operations need sequence)
  PRED_4: Sigma(DeltaT) = T (fundamental theorem — contribution integrates growth)
  PRED_5: Three needs independently predict well-being (3 distinct QA types)

Theorem NT compliant: observer projection framing.

Checks: HN_1 (schema), HN_MAP (mapping complete), HN_SDT (3 SDT pairings),
HN_TYPE (3 distinct QA types), HN_PRED (5/5 predictions confirmed),
HN_NT (Theorem NT), HN_SRC (>=3 sources incl peer-reviewed),
HN_W (>=5 witnesses), HN_F (fundamental QN present),
HN_DERIV (derivation witnesses correct), HN_DELTA (DeltaT preserves structure),
HN_SIGMA (SigmaT preserves structure), HN_FT (fundamental theorem witness).
"""

import json
import os
import sys


SCHEMA = "QA_HUMAN_NEEDS_SDT_CERT.v1"

REQUIRED_NEEDS = {"certainty", "variety", "significance", "connection", "growth", "contribution"}
REQUIRED_SDT = {"autonomy", "competence", "relatedness"}
REQUIRED_QA_TYPES = {"generators", "state_plus_derivative", "reach_plus_integral"}


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # HN_1 — schema
    if cert.get("schema_version") != SCHEMA:
        err("HN_1", f"schema_version must be {SCHEMA}")

    # HN_MAP — canonical mapping complete
    mapping = cert.get("canonical_mapping", {})
    mapped_needs = set(mapping.keys())
    missing_needs = REQUIRED_NEEDS - mapped_needs
    if missing_needs:
        err("HN_MAP", f"missing needs in canonical_mapping: {sorted(missing_needs)}")
    for need, info in mapping.items():
        if "qa_element" not in info:
            err("HN_MAP", f"need '{need}' missing qa_element")
        if "sdt_need" not in info:
            err("HN_MAP", f"need '{need}' missing sdt_need")

    # HN_SDT — 3 SDT needs, each paired with exactly 2 Robbins needs
    sdt_pairing = cert.get("sdt_pairing", [])
    sdt_needs_found = set()
    for sp in sdt_pairing:
        sdt_need = sp.get("sdt_need", "")
        sdt_needs_found.add(sdt_need)
        pair = sp.get("robbins_pair", [])
        if len(pair) != 2:
            err("HN_SDT", f"SDT need '{sdt_need}' must have exactly 2 Robbins pair members, got {len(pair)}")
        qa_pair = sp.get("qa_pair", [])
        if len(qa_pair) != 2:
            err("HN_SDT", f"SDT need '{sdt_need}' must have exactly 2 QA pair members, got {len(qa_pair)}")
    missing_sdt = REQUIRED_SDT - sdt_needs_found
    if missing_sdt:
        err("HN_SDT", f"missing SDT needs: {sorted(missing_sdt)}")

    # HN_TYPE — 3 distinct QA pair types
    qa_types_found = {sp.get("qa_type", "") for sp in sdt_pairing}
    missing_types = REQUIRED_QA_TYPES - qa_types_found
    if missing_types:
        err("HN_TYPE", f"missing QA pair types: {sorted(missing_types)}")

    # HN_PRED — structural predictions
    preds = cert.get("structural_predictions", [])
    confirmed_count = sum(1 for p in preds if p.get("confirmed"))
    if len(preds) < 5:
        err("HN_PRED", f"need >=5 structural predictions, got {len(preds)}")
    if confirmed_count < 5:
        err("HN_PRED", f"need >=5 confirmed predictions, got {confirmed_count}")

    # HN_NT — Theorem NT compliance
    nt = cert.get("theorem_nt_compliance", {})
    if nt.get("status") != "compliant":
        err("HN_NT", "theorem_nt_compliance.status must be 'compliant'")
    if nt.get("framing") != "observer_projection":
        err("HN_NT", "theorem_nt_compliance.framing must be 'observer_projection'")

    # HN_SRC — source grounding
    sources = cert.get("source_grounding", [])
    if len(sources) < 3:
        err("HN_SRC", f"need >=3 source groundings, got {len(sources)}")
    has_peer_reviewed = any("doi" in s for s in sources)
    if not has_peer_reviewed:
        err("HN_SRC", "need >=1 peer-reviewed source (with doi field)")

    # HN_W — witness count (derivation + delta + sigma + predictions)
    deriv_w = cert.get("derivation_witnesses", [])
    delta_w = cert.get("delta_witnesses", [])
    sigma_w = cert.get("sigma_witnesses", [])
    total_witnesses = len(deriv_w) + len(delta_w) + len(sigma_w) + len(preds)
    if total_witnesses < 5:
        err("HN_W", f"need >=5 total witnesses, got {total_witnesses}")

    # HN_F — fundamental QN (1,1,2,3) present
    has_fundamental = False
    for w in deriv_w:
        if w.get("b") == 1 and w.get("e") == 1:
            has_fundamental = True
            break
    if not has_fundamental:
        # Check in delta or sigma witnesses too
        for w in delta_w:
            t1 = w.get("T1", {})
            if t1.get("b") == 1 and t1.get("e") == 1:
                has_fundamental = True
                break
    if not has_fundamental:
        for w in sigma_w:
            for t in w.get("tuples", []):
                if t.get("b") == 1 and t.get("e") == 1:
                    has_fundamental = True
                    break
    if not has_fundamental:
        err("HN_F", "fundamental QN (b=1, e=1) not found in any witness")

    # HN_DERIV — derivation witnesses correct
    for w in deriv_w:
        b, e = w.get("b", 0), w.get("e", 0)
        d_exp = b + e
        a_exp = b + 2 * e
        if w.get("d_derived") != d_exp:
            err("HN_DERIV", f"witness {w.get('witness_id')}: d should be {d_exp}, got {w.get('d_derived')}")
        if w.get("a_derived") != a_exp:
            err("HN_DERIV", f"witness {w.get('witness_id')}: a should be {a_exp}, got {w.get('a_derived')}")

    # HN_DELTA — DeltaT preserves QA structure
    for w in delta_w:
        dt = w.get("DeltaT", {})
        db, de, dd, da = dt.get("db", 0), dt.get("de", 0), dt.get("dd", 0), dt.get("da", 0)
        if dd != db + de:
            err("HN_DELTA", f"witness {w.get('witness_id')}: delta-d ({dd}) != delta-b + delta-e ({db + de})")
        if da != db + 2 * de:
            err("HN_DELTA", f"witness {w.get('witness_id')}: delta-a ({da}) != delta-b + 2*delta-e ({db + 2 * de})")

    # HN_SIGMA — SigmaT preserves QA structure
    for w in sigma_w:
        st = w.get("SigmaT", {})
        sb, se, sd, sa = st.get("sb", 0), st.get("se", 0), st.get("sd", 0), st.get("sa", 0)
        if sd != sb + se:
            err("HN_SIGMA", f"witness {w.get('witness_id')}: sum-d ({sd}) != sum-b + sum-e ({sb + se})")
        if sa != sb + 2 * se:
            err("HN_SIGMA", f"witness {w.get('witness_id')}: sum-a ({sa}) != sum-b + 2*sum-e ({sb + 2 * se})")
        # Also verify the sums match actual tuple sums
        tuples = w.get("tuples", [])
        if tuples:
            actual_sb = sum(t.get("b", 0) for t in tuples)
            actual_se = sum(t.get("e", 0) for t in tuples)
            actual_sd = sum(t.get("d", 0) for t in tuples)
            actual_sa = sum(t.get("a", 0) for t in tuples)
            if sb != actual_sb or se != actual_se or sd != actual_sd or sa != actual_sa:
                err("HN_SIGMA", f"witness {w.get('witness_id')}: declared sums don't match tuple sums")

    # HN_FT — fundamental theorem witness
    ft = cert.get("fundamental_theorem_witness")
    if ft:
        if not ft.get("match"):
            err("HN_FT", "fundamental theorem witness: match must be true")
        sod = ft.get("sum_of_deltas", {})
        diff = ft.get("T3_minus_T0", {})
        for key in ("sb", "se", "sd", "sa"):
            dk = key.replace("s", "d")
            if sod.get(key) != diff.get(dk):
                err("HN_FT", f"fundamental theorem: sum_of_deltas.{key} ({sod.get(key)}) != T3_minus_T0.{dk} ({diff.get(dk)})")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def self_test():
    """Run validator on all fixtures in fixtures/ directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "hn_pass_structural_alignment.json": True,
        "hn_pass_derivation_chain.json": True,
    }
    results = []
    for fname, should_pass in expected.items():
        path = os.path.join(fix_dir, fname)
        with open(path) as f:
            cert = json.load(f)
        res = validate(cert)
        results.append({
            "fixture": fname,
            "expected_pass": should_pass,
            "actual_pass": res["ok"],
            "ok": res["ok"] == should_pass,
            "errors": res["errors"] if not res["ok"] else [],
        })
    return results


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        results = self_test()
        ok = all(r["ok"] for r in results)
        print(json.dumps({"ok": ok, "results": results}, indent=2))
        sys.exit(0 if ok else 1)
    elif len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cert = json.load(f)
        print(json.dumps(validate(cert), indent=2))
    else:
        print(f"Usage: python {os.path.basename(__file__)} [--self-test | <cert.json>]")
