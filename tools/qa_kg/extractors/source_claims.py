# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 3 infrastructure; populates SourceWork/SourceClaim nodes + contradicts/supersedes edges from a fixture -->
"""Phase 3 Source-Claim ingestion extractor.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Reads `tools/qa_kg/fixtures/source_claims_phase3.json` and populates:
  - SourceWork nodes       (node_type=Work, authority=primary, epistemic_status=source_work)
  - SourceClaim nodes      (node_type=Claim, authority=primary, epistemic_status=source_claim)
  - Internal Observations  (node_type=Claim, authority=internal, epistemic_status=observation)
  - quoted-from edges      (SourceClaim → SourceWork, structural, non-causal)
  - contradicts edges      (provenance JSON carries `reason` ∈ closed set)
  - supersedes edges       (newer → older, structural, non-causal)

Deterministic SourceClaim IDs: `sc:<claim_id>` (from fixture) OR derived from
`sha1(locator + quote)[:16]` if `id` is absent. 16 hex chars = 64 bits to
stay collision-safe at Phase 4.5 corpus scale.

Idempotency: kg.upsert_node / kg.upsert_edge use ON CONFLICT DO UPDATE, so
running this extractor repeatedly does not duplicate nodes or edges. Missing
supersedes-target cert nodes (the case where certs.populate hasn't run yet)
are skipped with a warning — this is graceful degradation, not silent swallow.

Invoke after certs.populate so cert:fs:qa_kg_consistency_cert_v<N> nodes
exist for the supersedes seed to attach to.
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

_DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "source_claims_phase3.json"
)

# Closed set of extraction methods accepted on SourceClaim nodes — mirrors
# [253] SC3. kept here so the extractor rejects bad fixtures early rather
# than letting SC3 catch them at cert time.
_VALID_EXTRACTION_METHODS = frozenset({"manual", "ocr", "llm", "script"})


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


def _upsert_source_work(kg: KG, spec: dict) -> str:
    """Create (or update) a SourceWork node from a fixture entry."""
    node = Node.source_work(
        work_id=spec["id"],
        title=spec["title"],
        source_locator=spec["source_locator"],
        body=spec.get("body", ""),
        extraction_method=spec.get("extraction_method", "manual"),
    )
    kg.upsert_node(node)
    return node.id


def _upsert_source_claim(kg: KG, spec: dict) -> str:
    """Create (or update) a SourceClaim node; emit quoted-from edge."""
    method = spec.get("extraction_method", "manual")
    if method not in _VALID_EXTRACTION_METHODS:
        raise ValueError(
            f"source_claim extraction_method={method!r} not in "
            f"{sorted(_VALID_EXTRACTION_METHODS)} (fixture entry id={spec.get('id')!r})"
        )
    nid = _claim_id(spec)
    claim_slug = nid[len("sc:"):]
    node = Node.source_claim(
        claim_id=claim_slug,
        quote=spec["quote"],
        source_locator=spec["source_locator"],
        extraction_method=method,
        title=spec.get("title"),
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


def populate(kg: KG, fixture_path: Path | str | None = None) -> dict[str, int]:
    """Ingest Phase 3 seed. Returns counters for each node/edge category."""
    path = Path(fixture_path) if fixture_path else _DEFAULT_FIXTURE
    if not path.exists():
        _log.warning("Phase 3 fixture not found at %s — skipping", path)
        return {
            "works": 0, "claims": 0, "observations": 0,
            "contradicts": 0, "supersedes": 0,
        }
    data = json.loads(path.read_text(encoding="utf-8"))

    n_works = 0
    for w in data.get("works", []):
        _upsert_source_work(kg, w)
        n_works += 1

    n_obs = 0
    for o in data.get("observations", []):
        _upsert_observation(kg, o)
        n_obs += 1

    n_claims = 0
    for c in data.get("claims", []):
        _upsert_source_claim(kg, c)
        n_claims += 1

    n_contradicts = 0
    for pair in data.get("contradicts", []):
        if _emit_contradicts_edge(kg, pair):
            n_contradicts += 1

    n_supersedes = 0
    for pair in data.get("supersedes", []):
        if _emit_supersedes_edge(kg, pair):
            n_supersedes += 1

    return {
        "works": n_works,
        "claims": n_claims,
        "observations": n_obs,
        "contradicts": n_contradicts,
        "supersedes": n_supersedes,
    }
