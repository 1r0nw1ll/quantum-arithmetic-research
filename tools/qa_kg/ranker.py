"""QA-KG authority-tiered retrieval ranker — Phase 4.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Pure formula module. No DB access. No side effects. Loads constants from
qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/ranker_spec.json
(single source of truth; cert [254] R6 verifies the implementation matches
the spec, R9 verifies coverage of authority + epistemic_status against
schema enums + allowed_matrix.json).

Composed score (per node, per query):

    score = authority_weight[node.authority]          # primary 10, derived 8, internal 5, agent 1
          × bm25_norm                                 # min-max across candidate pool, FTS5-sign-inverted
          × node.confidence                           # measured signal; default 1.0
          × time_decay                                # 1.0 for axiom/source_*/certified; exp(-Δdays/365) else
          × contradiction_factor                      # 1.5 if contradicts edge present, else 1.0
          × provenance_decay                          # exp(-depth/3) if depth >= 0, no_path_factor (0.5) if -1

Tiebreak (deterministic): score DESC, authority_weight DESC, node_id ASC.

References:
  - tools/qa_kg/kg.py::KG.search_authority_ranked  (consumer)
  - qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/ranker_spec.json
  - qa_alphageometry_ptolemy/qa_kg_epistemic_fields_cert_v1/allowed_matrix.json (R9 axes)
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import datetime as _dt
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "qa_alphageometry_ptolemy"
    / "qa_kg_authority_ranker_cert_v1"
    / "ranker_spec.json"
)


@dataclass(frozen=True)
class RankerSpec:
    """Closed-form ranker constants. Mirrors ranker_spec.json exactly.

    R9 (cert [254]) enforces:
      decay_exempt_status ∪ decay_status == EPISTEMIC_STATUSES (schema.py)
      authority_weight.keys()           == AUTHORITIES (schema.py)
      lifecycle_factor.keys()           == LIFECYCLE_STATES \\ {'withdrawn'}
      decay_exempt_status ∩ decay_status == ∅
    """
    authority_weight:    dict[str, float]
    lifecycle_factor:    dict[str, float]
    decay_exempt_status: frozenset[str]
    decay_status:        frozenset[str]
    halflife_days:       float
    halflife_hops:       float
    no_path_factor:      float
    contradiction_prior: float
    candidate_pool_k:    int
    bm25_normalization:  str


@dataclass(frozen=True)
class RankedHit:
    """One ranked search result.

    `node` is the raw sqlite3.Row from the nodes table. `score_breakdown`
    carries each multiplicative factor so callers (and the cert validator)
    can see which signal dominated.
    """
    node: sqlite3.Row
    score: float
    authority: str
    contradiction_state: str          # 'none' | 'src' | 'dst' | 'both'
    provenance_depth: int             # -1 = no path to any axiom
    score_breakdown: dict[str, float] = field(default_factory=dict)


_CACHED_SPEC: RankerSpec | None = None


def _load_spec_from_disk(path: Path) -> RankerSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RankerSpec(
        authority_weight=dict(raw["authority_weight"]),
        lifecycle_factor=dict(raw["lifecycle_factor"]),
        decay_exempt_status=frozenset(raw["decay_exempt_status"]),
        decay_status=frozenset(raw["decay_status"]),
        halflife_days=float(raw["halflife_days"]),
        halflife_hops=float(raw["halflife_hops"]),
        no_path_factor=float(raw["no_path_factor"]),
        contradiction_prior=float(raw["contradiction_prior"]),
        candidate_pool_k=int(raw["candidate_pool_k"]),
        bm25_normalization=str(raw["bm25_normalization"]),
    )


def load_spec(path: Path | None = None) -> RankerSpec:
    """Return the canonical RankerSpec, cached at module level.

    Passing an explicit `path` bypasses and replaces the cache (useful when
    a test wants to load an alternate spec file).
    """
    global _CACHED_SPEC
    if path is not None:
        _CACHED_SPEC = _load_spec_from_disk(path)
        return _CACHED_SPEC
    if _CACHED_SPEC is None:
        _CACHED_SPEC = _load_spec_from_disk(DEFAULT_SPEC_PATH)
    return _CACHED_SPEC


def reload_spec() -> RankerSpec:
    """Drop the cache and reload from DEFAULT_SPEC_PATH. Test-only helper."""
    global _CACHED_SPEC
    _CACHED_SPEC = None
    return load_spec()


# --- per-factor functions -------------------------------------------------

def authority_weight(spec: RankerSpec, authority: str) -> float:
    """Return spec.authority_weight[authority] or raise KeyError.

    No silent default — R9 (cert [254]) requires every authority value in
    use to be in the spec. Forces a spec update in the same commit as any
    new authority value (defense-in-depth alongside the cert gate).
    """
    if authority is None:
        raise KeyError("authority_weight: authority is None")
    if authority not in spec.authority_weight:
        raise KeyError(
            f"authority_weight: {authority!r} not in spec keys "
            f"{sorted(spec.authority_weight)}"
        )
    return spec.authority_weight[authority]


def lifecycle_factor(spec: RankerSpec, lifecycle_state: str) -> float:
    """Return spec.lifecycle_factor[lifecycle_state] or raise KeyError.

    'withdrawn' is excluded from the candidate pool entirely (never reaches
    the ranker), so it is intentionally absent from the spec map. R9
    enforces lifecycle_factor.keys() == LIFECYCLE_STATES \\ {'withdrawn'}.
    """
    if lifecycle_state not in spec.lifecycle_factor:
        raise KeyError(
            f"lifecycle_factor: {lifecycle_state!r} not in spec keys "
            f"{sorted(spec.lifecycle_factor)} "
            f"(withdrawn is excluded from the pool, not factored)"
        )
    return spec.lifecycle_factor[lifecycle_state]


def _parse_iso(ts: str) -> _dt.datetime | None:
    """Parse an ISO-8601 timestamp; return None on empty/malformed input."""
    if not ts:
        return None
    s = ts.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = _dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed


def time_decay(
    spec: RankerSpec,
    epistemic_status: str | None,
    created_ts: str,
    valid_from: str,
    valid_at: _dt.datetime | None,
) -> float:
    """Exponential half-life decay, scoped to non-exempt epistemic_status.

    Precedence for the "claim age" timestamp: valid_from if non-empty,
    else created_ts. This keeps an OB observation captured today about a
    Pond statement from 2018 from looking fresh — once Phase 4.5 populates
    valid_from on those nodes, decay reflects claim age, not ingestion age.

    epistemic_status in spec.decay_exempt_status → 1.0 (no decay).
    epistemic_status not in spec.decay_status either → 1.0 (R9 prevents
    drift; this branch is the fail-safe for an unrecognized status).
    """
    if epistemic_status in spec.decay_exempt_status:
        return 1.0
    if epistemic_status not in spec.decay_status:
        return 1.0
    anchor_raw = valid_from if valid_from else created_ts
    anchor = _parse_iso(anchor_raw)
    if anchor is None:
        return 1.0
    now = valid_at if valid_at is not None else _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    delta_days = max(0.0, (now - anchor).total_seconds() / 86400.0)
    return math.exp(-delta_days / spec.halflife_days)


def contradiction_factor(spec: RankerSpec, contradiction_state: str) -> float:
    """1.5× boost when the node has any contradicts edge (either side).

    See plan D4: 1.5 surfaces contradicted material approximately one
    authority-tier higher in the score. Cert [254] R3 verifies top-3
    surfacing on the contradicted_material fixture queries.
    """
    if contradiction_state == "none":
        return 1.0
    return spec.contradiction_prior


def provenance_decay(spec: RankerSpec, depth: int) -> float:
    """exp(-depth / spec.halflife_hops) for depth >= 0; spec.no_path_factor if -1.

    See plan S1: exp half-life on hops gives d=0→1.0, d=1→0.717,
    d=3→0.368, d=5→0.189. depth = -1 means no axiom-rooted derivation
    chain; spec.no_path_factor (0.5) is a calibrated mid-penalty until
    Phase 4.5 populates more chains.
    """
    if depth < 0:
        return spec.no_path_factor
    return math.exp(-float(depth) / spec.halflife_hops)


def normalize_bm25(raw_scores: list[float]) -> list[float]:
    """Per-query min-max normalization, FTS5-sign-inverted.

    SQLite FTS5 returns NEGATIVE bm25 scores where SMALLER is better.
    This function inverts so larger = better, and rescales the candidate
    pool to [0, 1]. ε=1e-9 to avoid div-by-zero on single-candidate or
    all-tied pools. Empty input → empty output (caller handles).
    """
    if not raw_scores:
        return []
    bm_min = min(raw_scores)
    bm_max = max(raw_scores)
    span = bm_max - bm_min
    if span <= 0.0:
        return [1.0 for _ in raw_scores]
    return [(bm_max - s) / (span + 1e-9) for s in raw_scores]


def compose_score(
    *,
    authority: str,
    bm25_norm: float,
    confidence: float,
    epistemic_status: str | None,
    created_ts: str,
    valid_from: str,
    valid_at: _dt.datetime | None,
    contradiction_state: str,
    provenance_depth: int,
    lifecycle_state: str,
    spec: RankerSpec,
) -> tuple[float, dict[str, float]]:
    """Multiply the seven factors. Return (score, breakdown).

    breakdown carries each factor under a stable key so cert [254] R6 can
    inspect it for golden-fixture correctness.
    """
    aw = authority_weight(spec, authority)
    lf = lifecycle_factor(spec, lifecycle_state)
    td = time_decay(spec, epistemic_status, created_ts, valid_from, valid_at)
    cf = contradiction_factor(spec, contradiction_state)
    pd = provenance_decay(spec, provenance_depth)
    breakdown = {
        "authority":   aw,
        "lifecycle":   lf,
        "bm25_norm":   bm25_norm,
        "confidence":  confidence,
        "time_decay":  td,
        "contradiction": cf,
        "prov_decay":  pd,
    }
    score = aw * lf * bm25_norm * confidence * td * cf * pd
    return score, breakdown
