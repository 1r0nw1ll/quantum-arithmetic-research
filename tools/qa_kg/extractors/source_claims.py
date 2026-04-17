# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 3+4.5 infrastructure; populates SourceWork/SourceClaim nodes + contradicts/supersedes/derived-from edges from per-source fixtures -->
"""Phase 3 + Phase 4.5 Source-Claim ingestion extractor.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Reads every `tools/qa_kg/fixtures/source_claims_*.json` (deterministic
filename-asc glob) and populates:
  - SourceWork nodes       (node_type=Work, authority=primary, epistemic_status=source_work)
  - SourceClaim nodes      (node_type=Claim, authority=primary, epistemic_status=source_claim)
  - Internal Observations  (node_type=Claim, authority=internal, epistemic_status=observation)
  - quoted-from edges      (SourceClaim → SourceWork, structural, non-causal)
  - contradicts edges      (provenance JSON carries `reason` ∈ closed set)
  - supersedes edges       (newer → older, structural, non-causal)
  - derived-from edges     (Phase 4.5: cert → SourceClaim, causal, structural proof link)

Deterministic SourceClaim IDs: `sc:<claim_id>` (from fixture) OR derived from
`sha1(locator + quote)[:16]` if `id` is absent. 16 hex chars = 64 bits to
stay collision-safe at Phase 4.5+ corpus scale.

Phase 4.5 column population:
  - confidence: driven by `tools/qa_kg/extraction_confidence.json` map
    (manual=1.0, ocr=0.7, llm=0.5, script=0.9). Fixture entries may carry
    `confidence_override: <float>` + `confidence_override_reason: <str>`
    to replace the method default; both fields must be present together
    or the extractor raises ValueError early.
  - valid_from / valid_until / domain: copied verbatim from each fixture
    entry; defaults to '' when absent.

Idempotency: kg.upsert_node / kg.upsert_edge use ON CONFLICT DO UPDATE, so
running this extractor repeatedly does not duplicate nodes or edges. Missing
supersedes-target / derived-from endpoint nodes are skipped with a warning
— graceful degradation, not silent swallow.

Invoke after certs.populate so cert:fs:<family> nodes exist for supersedes
+ derived-from to attach to.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import hashlib
import json
import logging
from pathlib import Path

from tools.qa_kg.kg import KG, Edge, FirewallViolation, Node
from tools.qa_kg.schema import LIFECYCLE_STATES


_log = logging.getLogger("qa_kg.extractors.source_claims")

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
_DEFAULT_FIXTURE = _FIXTURES_DIR / "source_claims_phase3.json"  # Phase 3 frozen baseline
_FIXTURE_GLOB = "source_claims_*.json"

_EXTRACTION_CONFIDENCE_FILE = (
    Path(__file__).resolve().parents[1] / "extraction_confidence.json"
)

# Closed set of extraction methods accepted on SourceClaim nodes — mirrors
# [253] SC3. kept here so the extractor rejects bad fixtures early rather
# than letting SC3 catch them at cert time.
_VALID_EXTRACTION_METHODS = frozenset({"manual", "ocr", "llm", "script"})


def _load_extraction_confidence_map() -> dict[str, float]:
    """Load the Phase 4.5 extraction_confidence.json map.

    Single source of truth shared with [253] SC9. Raises FileNotFoundError
    when the config is missing (configuration error, not silent fallback).
    """
    if not _EXTRACTION_CONFIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"extraction_confidence.json not found at {_EXTRACTION_CONFIDENCE_FILE} — "
            f"Phase 4.5 [253] SC9 requires this single-source-of-truth map."
        )
    data = json.loads(_EXTRACTION_CONFIDENCE_FILE.read_text(encoding="utf-8"))
    methods = data.get("methods")
    if not isinstance(methods, dict):
        raise ValueError(
            f"extraction_confidence.json is malformed: 'methods' key missing or "
            f"not a dict (got {type(methods).__name__})"
        )
    out: dict[str, float] = {}
    for method, conf in methods.items():
        if method not in _VALID_EXTRACTION_METHODS:
            raise ValueError(
                f"extraction_confidence.json: method {method!r} not in "
                f"{sorted(_VALID_EXTRACTION_METHODS)}"
            )
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError(
                f"extraction_confidence.json: method {method!r} confidence "
                f"{conf!r} not a float in [0.0, 1.0]"
            )
        out[method] = float(conf)
    missing = _VALID_EXTRACTION_METHODS - set(out)
    if missing:
        raise ValueError(
            f"extraction_confidence.json: missing methods {sorted(missing)} — "
            f"map must be exhaustive across the SC3 closed set."
        )
    return out


def _resolve_confidence(spec: dict, method: str, conf_map: dict[str, float]) -> float:
    """Return the confidence value for this fixture entry.

    Rule:
      - If neither `confidence_override` nor `confidence_override_reason`
        is present, return conf_map[method]. A legacy `confidence` key in
        the fixture is honoured only when it equals conf_map[method]
        (Phase 3 backwards compat); any mismatch is rejected so SC9
        remains the single source of truth.
      - If either override field is present, BOTH must be present.
        Otherwise raise ValueError.
      - When both are present: `confidence_override` must be in [0.0, 1.0]
        and `confidence_override_reason` must be a non-empty string.
    """
    has_override = "confidence_override" in spec
    has_reason = "confidence_override_reason" in spec
    if has_override != has_reason:
        raise ValueError(
            f"source_claims fixture entry id={spec.get('id')!r}: "
            f"confidence_override and confidence_override_reason must be "
            f"present together (saw override={has_override}, reason={has_reason}). "
            f"[253] SC9 enforces this at cert time as well."
        )
    if has_override:
        override = spec["confidence_override"]
        if not isinstance(override, (int, float)) or not (0.0 <= float(override) <= 1.0):
            raise ValueError(
                f"source_claims fixture entry id={spec.get('id')!r}: "
                f"confidence_override={override!r} not a float in [0.0, 1.0]"
            )
        reason = spec["confidence_override_reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"source_claims fixture entry id={spec.get('id')!r}: "
                f"confidence_override_reason must be a non-empty string"
            )
        return float(override)
    # No override — default to the method map
    default = conf_map[method]
    if "confidence" in spec:
        fixture_conf = float(spec["confidence"])
        if abs(fixture_conf - default) > 1e-9:
            raise ValueError(
                f"source_claims fixture entry id={spec.get('id')!r}: "
                f"explicit confidence={fixture_conf} differs from method-default "
                f"{default} (method={method!r}) but no confidence_override_reason. "
                f"Either drop the `confidence` field, align it with the map, or "
                f"add a confidence_override + confidence_override_reason."
            )
    return default


def _claim_id(claim_spec: dict) -> str:
    """Compute the deterministic `sc:<...>` node id for a fixture claim entry.

    Explicit `id` in the fixture wins (human-readable slugs, stable across
    quote edits). Otherwise hash the locator + quote to 16 hex = 64 bits.
    """
    explicit = claim_spec.get("id")
    if explicit:
        return f"sc:{explicit}"
    h = hashlib.sha1()
    h.update((claim_spec.get("source_locator", "") + "\n"
              + claim_spec.get("quote", "")).encode("utf-8"))
    return f"sc:{h.hexdigest()[:16]}"


def _upsert_source_work(kg: KG, spec: dict, conf_map: dict[str, float]) -> str:
    """Create (or update) a SourceWork node from a fixture entry."""
    method = spec.get("extraction_method", "manual")
    if method not in _VALID_EXTRACTION_METHODS:
        raise ValueError(
            f"source_work extraction_method={method!r} not in "
            f"{sorted(_VALID_EXTRACTION_METHODS)} (fixture entry id={spec.get('id')!r})"
        )
    confidence = _resolve_confidence(spec, method, conf_map)
    node = Node.source_work(
        work_id=spec["id"],
        title=spec["title"],
        source_locator=spec["source_locator"],
        body=spec.get("body", ""),
        extraction_method=method,
        confidence=confidence,
        valid_from=spec.get("valid_from", ""),
        valid_until=spec.get("valid_until", ""),
        domain=spec.get("domain", ""),
    )
    kg.upsert_node(node)
    return node.id


def _upsert_source_claim(kg: KG, spec: dict, conf_map: dict[str, float]) -> str:
    """Create (or update) a SourceClaim node; emit quoted-from edge."""
    method = spec.get("extraction_method", "manual")
    if method not in _VALID_EXTRACTION_METHODS:
        raise ValueError(
            f"source_claim extraction_method={method!r} not in "
            f"{sorted(_VALID_EXTRACTION_METHODS)} (fixture entry id={spec.get('id')!r})"
        )
    confidence = _resolve_confidence(spec, method, conf_map)
    nid = _claim_id(spec)
    claim_slug = nid[len("sc:"):]
    node = Node.source_claim(
        claim_id=claim_slug,
        quote=spec["quote"],
        source_locator=spec["source_locator"],
        extraction_method=method,
        title=spec.get("title"),
        confidence=confidence,
        valid_from=spec.get("valid_from", ""),
        valid_until=spec.get("valid_until", ""),
        domain=spec.get("domain", ""),
    )
    kg.upsert_node(node)
    work_target = spec["work"]
    target_id = work_target if work_target.startswith("work:") else f"work:{work_target}"
    try:
        kg.upsert_edge(Edge(
            src_id=nid,
            dst_id=target_id,
            edge_type="quoted-from",
            confidence=1.0,
            method="source_claims_extractor",
            provenance=json.dumps({"extractor": "source_claims.v1"}),
        ))
    except FirewallViolation as exc:
        # quoted-from is structural (non-causal); firewall should never
        # fire here. If it does, something in orbit.STRUCTURAL_EDGE_TYPES
        # or edge_allowed has regressed — surface loudly.
        _log.error("quoted-from unexpectedly blocked by firewall: %s", exc)
        raise
    return nid


def _upsert_observation(kg: KG, spec: dict) -> str:
    """Internal-authority observation node (contradicts target for typos)."""
    obs_id = spec["id"]
    if not obs_id.startswith("obs:"):
        raise ValueError(f"observation id must start with 'obs:' — got {obs_id!r}")
    authority = spec.get("authority", "internal")
    epistemic = spec.get("epistemic_status", "observation")
    node = Node(
        id=obs_id,
        node_type="Claim",
        title=spec.get("title", obs_id),
        body=spec.get("body", ""),
        authority=authority,
        epistemic_status=epistemic,
        method=spec.get("method", "manual"),
        source_locator=spec.get("source_locator", ""),
        # Phase 4.5 fields — observations may carry domain / valid_from too.
        # confidence stays at the Node default (1.0) unless the fixture
        # overrides it explicitly; observations are not bound to the
        # method-confidence map since that map governs PRIMARY sources.
        confidence=float(spec.get("confidence", 1.0)),
        valid_from=spec.get("valid_from", ""),
        valid_until=spec.get("valid_until", ""),
        domain=spec.get("domain", ""),
    )
    kg.upsert_node(node)
    return node.id


def _emit_contradicts_edge(kg: KG, pair: dict) -> bool:
    """Emit one contradicts edge with reason + extractor provenance."""
    src_id = pair["src"]
    dst_id = pair["dst"]
    reason = pair["reason"]
    provenance = json.dumps({
        "reason": reason,
        "extractor": "source_claims.v1",
    })
    try:
        kg.upsert_edge(Edge(
            src_id=src_id, dst_id=dst_id,
            edge_type="contradicts",
            confidence=1.0,
            method="source_claims_extractor",
            provenance=provenance,
        ))
        return True
    except (FirewallViolation, ValueError) as exc:
        _log.warning("contradicts edge %s→%s skipped: %s", src_id, dst_id, exc)
        return False


def _emit_supersedes_edge(kg: KG, pair: dict) -> bool:
    """Emit one newer→older supersedes edge + set dst.lifecycle_state."""
    src_id = pair["src"]
    dst_id = pair["dst"]
    # Require both endpoints to exist — supersedes across non-existent
    # nodes would be a no-op landmine for KG13.
    if kg.conn.execute("SELECT 1 FROM nodes WHERE id=?", (src_id,)).fetchone() is None:
        _log.warning("supersedes source %s missing — skipping", src_id)
        return False
    if kg.conn.execute("SELECT 1 FROM nodes WHERE id=?", (dst_id,)).fetchone() is None:
        _log.warning("supersedes target %s missing — skipping", dst_id)
        return False
    try:
        kg.upsert_edge(Edge(
            src_id=src_id, dst_id=dst_id,
            edge_type="supersedes",
            confidence=1.0,
            method="source_claims_extractor",
            provenance=json.dumps({
                "reason": "schema_upgrade",
                "extractor": "source_claims.v1",
            }),
        ))
    except FirewallViolation as exc:
        _log.error("supersedes unexpectedly blocked by firewall: %s", exc)
        raise
    # Mark dst as superseded (unless it's already withdrawn/deprecated —
    # don't clobber more-specific lifecycle states).
    dst_row = kg.conn.execute(
        "SELECT lifecycle_state FROM nodes WHERE id=?", (dst_id,)
    ).fetchone()
    current_lc = dst_row["lifecycle_state"]
    if current_lc in ("current", None):
        kg.conn.execute(
            "UPDATE nodes SET lifecycle_state='superseded' WHERE id=?", (dst_id,)
        )
        kg.conn.commit()
    elif current_lc == "superseded":
        pass  # already correct
    else:
        _log.info(
            "dst %s already has lifecycle_state=%r — leaving untouched",
            dst_id, current_lc,
        )
    if current_lc is not None and current_lc not in LIFECYCLE_STATES:
        _log.warning("unexpected lifecycle_state=%r on %s", current_lc, dst_id)
    return True


def _emit_derived_from_edge(kg: KG, pair: dict) -> bool:
    """Emit one cert → SourceClaim derived-from edge (Phase 4.5).

    derived-from is CAUSAL in orbit.CAUSAL_EDGE_TYPES — the Theorem NT
    firewall applies. Cert nodes (authority=derived, tier=cosmos) → any
    non-Unassigned dst passes. Missing-endpoint degradation: skip with
    warning (graceful — see extractors/source_claims.py module docstring).
    """
    src_id = pair["src"]
    dst_id = pair["dst"]
    if kg.conn.execute("SELECT 1 FROM nodes WHERE id=?", (src_id,)).fetchone() is None:
        _log.warning("derived-from source %s missing — skipping", src_id)
        return False
    if kg.conn.execute("SELECT 1 FROM nodes WHERE id=?", (dst_id,)).fetchone() is None:
        _log.warning("derived-from target %s missing — skipping", dst_id)
        return False
    rationale = pair.get("rationale", "")
    provenance = json.dumps({
        "rationale": rationale,
        "extractor": "source_claims.v1",
    })
    try:
        kg.upsert_edge(Edge(
            src_id=src_id, dst_id=dst_id,
            edge_type="derived-from",
            confidence=0.9,
            method="structural",
            provenance=provenance,
        ))
    except FirewallViolation as exc:
        _log.warning("derived-from edge %s→%s blocked by firewall: %s",
                     src_id, dst_id, exc)
        return False
    return True


def _iter_fixture_paths(fixture_path: Path | str | None) -> list[Path]:
    """Return the ordered list of fixture files to ingest.

    When `fixture_path` is given: single-file mode (Phase 3 behavior +
    test override).

    When `fixture_path` is None: multi-file mode — Phase 3 baseline first
    (source_claims_phase3.json, frozen), then every other
    `source_claims_*.json` in deterministic filename-asc order. Missing
    baseline returns an empty list with a warning so legacy call sites
    that expect Phase 3 to exist surface cleanly.
    """
    if fixture_path is not None:
        return [Path(fixture_path)]
    if not _FIXTURES_DIR.exists():
        _log.warning("Phase 3 fixtures dir not found at %s", _FIXTURES_DIR)
        return []
    all_paths = sorted(_FIXTURES_DIR.glob(_FIXTURE_GLOB))
    if not all_paths:
        _log.warning("No source_claims_*.json fixtures under %s", _FIXTURES_DIR)
        return []
    # Phase 3 baseline first (frozen) — the glob sort already places it
    # ahead of alphabetically-later per-source fixtures (p < {i, k, p}).
    # If that ever changes, reorder explicitly. Today it's stable.
    return all_paths


def populate(kg: KG, fixture_path: Path | str | None = None) -> dict[str, int]:
    """Ingest Phase 3 + Phase 4.5 corpus. Returns counters for each category.

    Single-file mode (fixture_path given) preserves Phase 3 test behavior.
    Multi-file mode (fixture_path=None) is the Phase 4.5 default — iterates
    every `source_claims_*.json` in deterministic filename-asc order.
    """
    paths = _iter_fixture_paths(fixture_path)
    if not paths:
        return {
            "works": 0, "claims": 0, "observations": 0,
            "contradicts": 0, "supersedes": 0, "derived_from": 0,
        }

    # Phase 4.5: load the extraction-confidence map once per populate call.
    # Absence is a configuration error (not silent), matching SC9 discipline.
    conf_map = _load_extraction_confidence_map()

    n_works = 0
    n_claims = 0
    n_obs = 0
    n_contradicts = 0
    n_supersedes = 0
    n_derived_from = 0

    # Two-pass over the fixture list: first pass creates all works +
    # observations + claims + quoted-from edges; second pass emits the
    # cross-fixture edges (contradicts / supersedes / derived-from).
    # This ensures derived-from src (cert node, created by certs.populate
    # ahead of us) and dst (SourceClaim, created in first pass above) both
    # exist before the edge emit attempts.
    fixture_blobs: list[dict] = []
    for path in paths:
        if not path.exists():
            _log.warning("fixture %s vanished between glob and read — skipping", path)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        fixture_blobs.append(data)
        for w in data.get("works", []):
            _upsert_source_work(kg, w, conf_map)
            n_works += 1
        for o in data.get("observations", []):
            _upsert_observation(kg, o)
            n_obs += 1
        for c in data.get("claims", []):
            _upsert_source_claim(kg, c, conf_map)
            n_claims += 1

    for data in fixture_blobs:
        for pair in data.get("contradicts", []):
            if _emit_contradicts_edge(kg, pair):
                n_contradicts += 1
        for pair in data.get("supersedes", []):
            if _emit_supersedes_edge(kg, pair):
                n_supersedes += 1
        for pair in data.get("derived_from", []):
            if _emit_derived_from_edge(kg, pair):
                n_derived_from += 1

    return {
        "works":        n_works,
        "claims":       n_claims,
        "observations": n_obs,
        "contradicts":  n_contradicts,
        "supersedes":   n_supersedes,
        "derived_from": n_derived_from,
    }
