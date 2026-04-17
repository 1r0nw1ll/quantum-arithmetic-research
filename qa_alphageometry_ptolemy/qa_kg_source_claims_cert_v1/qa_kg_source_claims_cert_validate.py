# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 3 infrastructure validator -->
"""QA-KG Source Claims Cert [253] v1 validator.

QA_COMPLIANCE = "cert_validator — validates KG SourceClaim / contradicts invariants, no empirical QA state machine"

Phase 3 + Phase 4.5. Validates that SourceClaim ingestion is well-formed
in the live qa_kg.db graph:

  SC1 (HARD) every SourceClaim has non-empty quote (body) AND
       source_locator that resolves via tools.qa_kg.locators.resolve_any.
  SC2 (HARD) every SourceClaim has exactly one quoted-from edge to an
       existing SourceWork node (epistemic_status=source_work).
  SC3 (HARD) extraction_method ∈ {manual, ocr, llm, script} on every
       SourceClaim.
  SC4 (HARD) no two SourceClaims with identical source_locator AND
       different body without a contradicts edge between them.
  SC5 (HARD) every contradicts edge's provenance JSON-parses and carries
       a `reason` key whose value ∈ {ocr, variant, typo, dispute, true}.
  SC6 (HARD) no contradicts cycles (scoped to populated graph).
  SC7 (WARN) unresolved contradicts with reason=dispute (triage sensor;
       value is surfaced into _meta_ledger.json via the Phase 3 WARN
       capture §12a).
  SC8 (HARD) no contradicts edge has endpoints with node_type=Axiom or
       authority=agent.
  SC9 (HARD, Phase 4.5) every SourceClaim/SourceWork's `confidence` equals
       `extraction_confidence.json[method]` OR the fixture entry declares
       a valid override (both `confidence_override` and
       `confidence_override_reason` present; override in [0,1]; reason
       non-empty). Scans every tools/qa_kg/fixtures/source_claims_*.json
       to verify the policy at the fixture level, then spot-checks DB
       nodes against the declared values.

Closed sets live in closed_sets.json (single source of truth). Locator
resolution reuses tools.qa_kg.locators.resolve_any (same as EF4) to avoid
divergence. Confidence-method map at tools/qa_kg/extraction_confidence.json
is the single source of truth for SC9 (same file is also used by
extractors/source_claims.populate).
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates KG SourceClaim / contradicts invariants, no empirical QA state machine"

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CERT_DIR = Path(__file__).resolve().parent
_CLOSED_SETS_FILE = _CERT_DIR / "closed_sets.json"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.qa_kg.locators import resolve_any as _resolve_any_locator


def _db_path_default() -> Path:
    override = os.environ.get("QA_KG_DB")
    if override:
        return Path(override)
    return _REPO / "tools" / "qa_kg" / "qa_kg.db"


def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise FileNotFoundError(f"QA-KG DB not found at {db}")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _load_closed_sets() -> dict:
    return json.loads(_CLOSED_SETS_FILE.read_text(encoding="utf-8"))


# --- SC1: quote + source_locator resolves ---

def check_sc1_quote_and_locator(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """Scope: SourceClaim nodes (authority=primary, epistemic_status=source_claim)."""
    viols: list[str] = []
    rows = conn.execute(
        "SELECT id, body, source_locator FROM nodes "
        "WHERE authority = 'primary' AND epistemic_status = 'source_claim'"
    ).fetchall()
    for r in rows:
        body = r["body"] or ""
        loc = r["source_locator"] or ""
        if not body.strip():
            viols.append(f"{r['id']}: empty body (quote) on SourceClaim")
            continue
        if not loc:
            viols.append(f"{r['id']}: empty source_locator on SourceClaim")
            continue
        ok, msg = _resolve_any_locator(loc, repo_root=_REPO, conn=conn)
        if not ok:
            viols.append(f"{r['id']}: {msg}")
    return len(viols) == 0, f"{len(viols)} SourceClaim quote/locator violation(s)", viols


# --- SC2: quoted-from FK to SourceWork ---

def check_sc2_source_work_fk(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """Each SourceClaim has exactly one quoted-from edge to a SourceWork."""
    viols: list[str] = []
    claims = conn.execute(
        "SELECT id FROM nodes "
        "WHERE authority = 'primary' AND epistemic_status = 'source_claim'"
    ).fetchall()
    for r in claims:
        edges = conn.execute(
            "SELECT dst_id FROM edges WHERE src_id=? AND edge_type='quoted-from'",
            (r["id"],),
        ).fetchall()
        if len(edges) == 0:
            viols.append(f"{r['id']}: 0 quoted-from edges")
            continue
        if len(edges) > 1:
            viols.append(f"{r['id']}: {len(edges)} quoted-from edges (>1)")
            continue
        tgt = conn.execute(
            "SELECT epistemic_status FROM nodes WHERE id=?",
            (edges[0]["dst_id"],),
        ).fetchone()
        if tgt is None:
            viols.append(
                f"{r['id']}: quoted-from target {edges[0]['dst_id']} missing"
            )
        elif tgt["epistemic_status"] != "source_work":
            viols.append(
                f"{r['id']}: quoted-from target {edges[0]['dst_id']} has "
                f"epistemic_status={tgt['epistemic_status']!r}≠source_work"
            )
    return len(viols) == 0, f"{len(viols)} SourceClaim FK violation(s)", viols


# --- SC3: extraction_method closed set ---

def check_sc3_extraction_method(
    conn: sqlite3.Connection, allowed: set[str],
) -> tuple[bool, str, list[str]]:
    """Scope: SourceClaim nodes. method ∈ allowed."""
    viols: list[str] = []
    rows = conn.execute(
        "SELECT id, method FROM nodes "
        "WHERE authority = 'primary' AND epistemic_status = 'source_claim'"
    ).fetchall()
    for r in rows:
        m = r["method"] or ""
        if m not in allowed:
            viols.append(f"{r['id']}: method={m!r} not in {sorted(allowed)}")
    return (
        len(viols) == 0,
        f"{len(viols)} SourceClaim extraction_method violation(s)",
        viols,
    )


# --- SC4: locator conflict rule ---

def check_sc4_locator_conflict_rule(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """Two SourceClaims with identical source_locator AND different body
    MUST be connected by a contradicts edge (either direction).

    Scope note: identical body + different locators is allowed and not
    a contradicts requirement — the same quote appearing across editions
    is expected and does not indicate a contradiction. Only same-position
    incompatible readings trigger SC4.
    """
    viols: list[str] = []
    rows = conn.execute(
        "SELECT id, source_locator, body FROM nodes "
        "WHERE authority = 'primary' AND epistemic_status = 'source_claim'"
    ).fetchall()
    by_loc: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        by_loc.setdefault(r["source_locator"] or "", []).append(
            (r["id"], r["body"] or "")
        )
    for loc, entries in by_loc.items():
        if len(entries) < 2:
            continue
        # All pairs with different body.
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a_id, a_body = entries[i]
                b_id, b_body = entries[j]
                if a_body == b_body:
                    continue
                edge = conn.execute(
                    """SELECT 1 FROM edges WHERE edge_type='contradicts'
                       AND ((src_id=? AND dst_id=?) OR (src_id=? AND dst_id=?))""",
                    (a_id, b_id, b_id, a_id),
                ).fetchone()
                if edge is None:
                    viols.append(
                        f"identical locator {loc!r} but different quote, "
                        f"no contradicts edge between {a_id} and {b_id}"
                    )
    return len(viols) == 0, f"{len(viols)} SC4 locator-conflict violation(s)", viols


# --- SC5: reason closed set on contradicts edges ---

def check_sc5_reason_closed_set(
    conn: sqlite3.Connection, allowed: set[str],
) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    rows = conn.execute(
        "SELECT src_id, dst_id, provenance FROM edges WHERE edge_type='contradicts'"
    ).fetchall()
    for r in rows:
        prov_raw = r["provenance"] or ""
        if not prov_raw:
            viols.append(f"{r['src_id']}→{r['dst_id']}: empty provenance")
            continue
        try:
            prov = json.loads(prov_raw)
        except json.JSONDecodeError:
            viols.append(f"{r['src_id']}→{r['dst_id']}: provenance not JSON")
            continue
        reason = prov.get("reason")
        if reason not in allowed:
            viols.append(
                f"{r['src_id']}→{r['dst_id']}: reason={reason!r} not in "
                f"{sorted(allowed)}"
            )
    return len(viols) == 0, f"{len(viols)} contradicts-reason violation(s)", viols


# --- SC6: no contradicts cycles (scoped) ---

def check_sc6_no_contradicts_cycles(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    edges = conn.execute(
        "SELECT src_id, dst_id FROM edges WHERE edge_type = 'contradicts'"
    ).fetchall()
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["src_id"], []).append(e["dst_id"])
    cycles: list[str] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in adj}

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for nxt in adj.get(node, []):
            c = color.get(nxt, WHITE)
            if c == GRAY:
                cycles.append(" → ".join(path + [node, nxt]))
            elif c == WHITE:
                color.setdefault(nxt, WHITE)
                dfs(nxt, path + [node])
        color[node] = BLACK

    for n in list(color.keys()):
        if color[n] == WHITE:
            dfs(n, [])
    return len(cycles) == 0, f"{len(cycles)} contradicts-cycle(s)", cycles


# --- SC7: WARN unresolved disputes ---

def check_sc7_warn_unresolved_disputes(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """WARN sensor. Always passes."""
    rows = conn.execute(
        "SELECT src_id, dst_id, provenance FROM edges WHERE edge_type='contradicts'"
    ).fetchall()
    disputes: list[str] = []
    for r in rows:
        prov_raw = r["provenance"] or ""
        if not prov_raw:
            continue
        try:
            prov = json.loads(prov_raw)
        except json.JSONDecodeError:
            continue
        if prov.get("reason") == "dispute":
            disputes.append(f"{r['src_id']}→{r['dst_id']}")
    return True, f"{len(disputes)} unresolved dispute(s)", disputes


# --- SC8: contradicts endpoint whitelist ---

def check_sc8_contradicts_endpoint_whitelist(
    conn: sqlite3.Connection,
    forbidden_node_types: set[str],
    forbidden_authorities: set[str],
) -> tuple[bool, str, list[str]]:
    """Every contradicts edge endpoint must satisfy:
      node_type ∉ forbidden_node_types
      authority ∉ forbidden_authorities
    per closed_sets.json. Preserves axiom-sanctity (axiom contradictions
    are linter bugs, not KG data) and sharp firewall (agent dissent must
    flow through kg.promote, not populate contradicts).
    """
    viols: list[str] = []
    rows = conn.execute(
        """SELECT e.src_id, e.dst_id,
                  s.node_type AS s_type, s.authority AS s_auth,
                  d.node_type AS d_type, d.authority AS d_auth
             FROM edges e
             LEFT JOIN nodes s ON s.id = e.src_id
             LEFT JOIN nodes d ON d.id = e.dst_id
            WHERE e.edge_type = 'contradicts'"""
    ).fetchall()
    for r in rows:
        problems: list[str] = []
        for side, nt, auth in (
            ("src", r["s_type"], r["s_auth"]),
            ("dst", r["d_type"], r["d_auth"]),
        ):
            if nt is None:
                problems.append(f"{side} node missing")
                continue
            if nt in forbidden_node_types:
                problems.append(f"{side} node_type={nt!r} forbidden")
            if auth in forbidden_authorities:
                problems.append(f"{side} authority={auth!r} forbidden")
        if problems:
            viols.append(
                f"{r['src_id']}→{r['dst_id']}: {'; '.join(problems)}"
            )
    return len(viols) == 0, f"{len(viols)} SC8 endpoint violation(s)", viols


# --- SC9: confidence-method map + override policy (Phase 4.5) ---

_EXTRACTION_CONFIDENCE_FILE = (
    _REPO / "tools" / "qa_kg" / "extraction_confidence.json"
)

_FIXTURES_DIR = _REPO / "tools" / "qa_kg" / "fixtures"
_FIXTURE_GLOB = "source_claims_*.json"


def _load_extraction_confidence_map() -> dict[str, float]:
    """Read extraction_confidence.json — single source of truth for SC9."""
    if not _EXTRACTION_CONFIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"extraction_confidence.json not found at "
            f"{_EXTRACTION_CONFIDENCE_FILE} — Phase 4.5 [253] SC9 requires it."
        )
    data = json.loads(_EXTRACTION_CONFIDENCE_FILE.read_text(encoding="utf-8"))
    methods = data.get("methods") or {}
    return {k: float(v) for k, v in methods.items()}


def check_sc9_confidence_map(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """SC9: fixture-level + DB-level confidence consistency.

    Rules:
      - Every fixture entry for a SourceClaim/SourceWork either omits
        `confidence` (accepts the map default) OR provides `confidence` ==
        map[method], UNLESS `confidence_override` + `confidence_override_reason`
        are BOTH present.
      - When override is present: value ∈ [0, 1]; reason is a non-empty
        string; no `confidence` field is required (override supersedes).
      - DB-level spot check: for every fixture entry without override, the
        corresponding DB node (by sc:<id> or work:<id>) has confidence ==
        map[method].
    """
    conf_map = _load_extraction_confidence_map()
    viols: list[str] = []
    total_entries = 0
    override_entries = 0

    if not _FIXTURES_DIR.exists():
        return True, f"fixtures dir {_FIXTURES_DIR} missing — 0 entries to check", []

    fixture_paths = sorted(_FIXTURES_DIR.glob(_FIXTURE_GLOB))
    for fp in fixture_paths:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            viols.append(f"{fp.name}: JSON parse error: {exc}")
            continue

        for kind, id_prefix in (("works", "work:"), ("claims", "sc:")):
            for entry in data.get(kind, []):
                total_entries += 1
                eid = entry.get("id", "<missing id>")
                full_id = f"{id_prefix}{eid}" if not eid.startswith(id_prefix) else eid
                method = entry.get("extraction_method", "manual")
                if method not in conf_map:
                    viols.append(
                        f"{fp.name}:{full_id}: method={method!r} not in "
                        f"extraction_confidence map (keys: {sorted(conf_map)})"
                    )
                    continue
                default = conf_map[method]
                has_override = "confidence_override" in entry
                has_reason = "confidence_override_reason" in entry

                if has_override or has_reason:
                    if not (has_override and has_reason):
                        viols.append(
                            f"{fp.name}:{full_id}: override partially declared "
                            f"(override={has_override}, reason={has_reason}) "
                            f"— both fields must be present together"
                        )
                        continue
                    override_entries += 1
                    ov = entry["confidence_override"]
                    if not isinstance(ov, (int, float)) or not (0.0 <= float(ov) <= 1.0):
                        viols.append(
                            f"{fp.name}:{full_id}: confidence_override={ov!r} "
                            f"not a float in [0.0, 1.0]"
                        )
                    rs = entry["confidence_override_reason"]
                    if not isinstance(rs, str) or not rs.strip():
                        viols.append(
                            f"{fp.name}:{full_id}: "
                            f"confidence_override_reason must be a non-empty string"
                        )
                    # DB spot-check: node confidence should equal the override
                    row = conn.execute(
                        "SELECT confidence FROM nodes WHERE id=?", (full_id,)
                    ).fetchone()
                    if row is not None:
                        db_conf = float(row["confidence"])
                        if abs(db_conf - float(ov)) > 1e-9:
                            viols.append(
                                f"{fp.name}:{full_id}: fixture override={ov} "
                                f"but DB confidence={db_conf} — extractor drift"
                            )
                else:
                    # No override → fixture `confidence` must match default
                    # OR be omitted
                    if "confidence" in entry:
                        fx = float(entry["confidence"])
                        if abs(fx - default) > 1e-9:
                            viols.append(
                                f"{fp.name}:{full_id}: confidence={fx} "
                                f"differs from method-default {default} "
                                f"(method={method!r}) but no "
                                f"confidence_override_reason present"
                            )
                    # DB spot-check: node confidence must equal default
                    row = conn.execute(
                        "SELECT confidence FROM nodes WHERE id=?", (full_id,)
                    ).fetchone()
                    if row is not None:
                        db_conf = float(row["confidence"])
                        if abs(db_conf - default) > 1e-9:
                            viols.append(
                                f"{fp.name}:{full_id}: no override in fixture "
                                f"but DB confidence={db_conf} ≠ default={default}"
                            )
    return (
        len(viols) == 0,
        f"{len(viols)} SC9 violation(s) across {total_entries} fixture entries "
        f"({override_entries} override(s))",
        viols,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--show-details", action="store_true")
    args = p.parse_args(argv)

    db = args.db or _db_path_default()
    try:
        conn = _connect(db)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 2

    closed = _load_closed_sets()
    reasons = set(closed["reasons"])
    methods = set(closed["extraction_methods"])
    forbidden_node_types = set(
        closed.get("contradicts_endpoints", {}).get("forbidden_node_types", [])
    )
    forbidden_authorities = set(
        closed.get("contradicts_endpoints", {}).get("forbidden_authorities", [])
    )

    hard_fail = False

    def run_bool(code: str, desc: str, fn, is_hard: bool) -> None:
        nonlocal hard_fail
        ok, msg, detail = fn()
        flag = "PASS" if ok else ("FAIL" if is_hard or args.strict else "WARN")
        print(f"[{flag}] {code}  {desc} — {msg}")
        if args.show_details and detail:
            for d in detail[:10]:
                print(f"         {d}")
            if len(detail) > 10:
                print(f"         ... +{len(detail)-10} more")
        if not ok and (is_hard or args.strict):
            hard_fail = True

    run_bool("SC1", "SourceClaim quote + locator resolves",
             lambda: check_sc1_quote_and_locator(conn), True)
    run_bool("SC2", "SourceClaim quoted-from FK to SourceWork",
             lambda: check_sc2_source_work_fk(conn), True)
    run_bool("SC3", "SourceClaim extraction_method closed set",
             lambda: check_sc3_extraction_method(conn, methods), True)
    run_bool("SC4", "Locator-conflict rule",
             lambda: check_sc4_locator_conflict_rule(conn), True)
    run_bool("SC5", "Contradicts reason closed set",
             lambda: check_sc5_reason_closed_set(conn, reasons), True)
    run_bool("SC6", "No contradicts cycles",
             lambda: check_sc6_no_contradicts_cycles(conn), True)

    # SC7 is WARN-only — always PASSes at the gate level, but emits a
    # visible [WARN] line for the Phase 3 §12a ledger capture to pick up
    # when disputes > 0.
    ok7, msg7, detail7 = check_sc7_warn_unresolved_disputes(conn)
    if detail7:
        print(f"[WARN] SC7  Unresolved disputes — {msg7}")
        if args.show_details:
            for d in detail7[:10]:
                print(f"         {d}")
    else:
        print(f"[PASS] SC7  Unresolved disputes — {msg7}")

    run_bool("SC8", "Contradicts endpoint whitelist",
             lambda: check_sc8_contradicts_endpoint_whitelist(
                 conn, forbidden_node_types, forbidden_authorities
             ), True)
    run_bool("SC9", "Confidence-method map + override policy (Phase 4.5)",
             lambda: check_sc9_confidence_map(conn), True)

    if hard_fail:
        print("[FAIL] QA-KG source claims cert [253] v1")
        return 1
    print("[PASS] QA-KG source claims cert [253] v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
