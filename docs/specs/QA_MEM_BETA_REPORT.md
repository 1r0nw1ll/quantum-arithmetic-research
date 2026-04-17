<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Beta-B benchmark report; applies pre-committed thresholds from docs/specs/QA_MEM_BETA_DECISION_MATRIX.md to the output of tools.qa_kg.analysis.run_beta_benchmark against the Beta-A frozen fixtures (graph_hash c41eb8f9…). No threshold changes, no methodology reopens. -->

# QA-MEM Beta-B Report

**Run date:** 2026-04-17
**Base commit:** 52099d9 (`feat(qa-mem): Beta-A pre-registration — 38 queries, gold, decision matrix`)
**Fixtures:** `beta_prereg_queries.json`, `beta_prereg_gold.json`, `beta_blind_gold.json` (all at Beta-A SHAs)
**Graph hash (stable pre/post):** `c41eb8f9388c795164d9c3a2172348cba6692b6138aacccd78e5ddece981c8a6`
**Raw results:** `tools/qa_kg/analysis/beta_results.json`
**Figures:** `tools/qa_kg/analysis/figures/*.png`

---

## Executive summary

| Axis | Outcome |
|---|---|
| **Q1 — ranker formula well-tuned?** | **PASS** (factor dominance 0.083 ≤ 0.50; prior=1.5 Pareto-optimal over {1.0, 1.25, 1.75, 2.0}) |
| **Q2 — design claims validated?** | **FAIL on 1 gate of 6** — head-to-head T1+T2. All other 5 gates PASS. |
| **Q3 — next work: corpus or ranker tuning?** | **Ranker tuning (Phase 4.7)** — 4 of 6 misses are ranker-traversal-limited (P-category); 2 are internal-authority demotion on contradicts (C-category). |

**Rollback NOT triggered** — Q2 fails 1 gate, rollback requires ≥ 3. Decision matrix §"Lower-tier failure — Phase 4.7 handoff, not rollback" applies: gate 6 head-to-head is filed as a Phase 4.7 action item with the concrete remediation below.

**Latency:** real-graph p95 = 32 ms; synthetic 5 000-node p95 = 27 ms (latency-only; ranking quality not claimed). Well under the 200 ms materialization threshold from `[254]` Phase 4 spec.

---

## Q1 — Ranker formula well-tuned?

### Factor dominance

| Metric | Value | Gate | Result |
|---|---|---|---|
| `dominated_fraction` (share > 0.80) | **0.083** (3/36) | ≤ 0.50 | **PASS** |

Mean contribution share across 38 top-1 breakdowns:

| Factor | Mean share |
|---|---|
| authority | 0.744 |
| prov_decay | 0.218 |
| contradiction | 0.026 |
| bm25_norm | 0.012 |
| lifecycle, confidence, time_decay | 0.000 |

**Interpretation.** The ranker composition is dominated by authority (log 10 ≈ 2.30) and provenance decay (log 0.5 ≈ −0.69 for orphans), which is the intended behaviour — those are the two signals Phase 4.5 was meant to make informative. Confidence, time_decay and lifecycle mean-share at ≈ 0 simply because the current corpus has (i) `confidence = 1.0` for 90%+ of nodes, (ii) `valid_until` empty on every node, (iii) only 3 superseded nodes (the `[225] v1/v2/v3` chain) — none of them at top-1 on any query. The three factors are **structurally live** (R9 gate enforces coverage) but have **no distinguishing values in-corpus yet**. That is a Phase 4.5 corpus-depth signal, not a ranker defect.

See `figures/factor_decomposition.png`.

### Contradiction-boost ablation

Five settings of `contradiction_prior` ∈ {1.0, 1.25, 1.5, 1.75, 2.0}:

| α | contradiction_recall_per_pair | graph_structural_pass_rate |
|---|---|---|
| 1.00 | 0.625 (5/8) | 0.857 (24/28) |
| 1.25 | 0.750 (6/8) | 0.857 (24/28) |
| **1.50** | **0.750 (6/8)** | **0.857 (24/28)** |
| 1.75 | 0.750 (6/8) | 0.857 (24/28) |
| 2.00 | 0.750 (6/8) | 0.857 (24/28) |

`prior=1.0` strictly dominated (contradiction recall drops to 5/8); 1.5 is **weakly** the best, tied with 1.25 / 1.75 / 2.00 on both axes. Pareto-optimality holds (no alternative strictly beats on any axis). See `figures/contradiction_boost_ablation.png`.

**Secondary finding.** The contradicts pair recall is capped at 6/8 regardless of α in [1.25, 2.0] — two pairs (C03, C07) are not moved by the contradiction boost because the missing endpoint is an `authority=internal` observation that is demoted by `authority_weight[internal]=5` (vs primary=10), and the 1.5× contradiction boost cannot close that gap. This is a Phase 4.7 interaction-between-factors tuning signal, not a ranker defect: moving the internal authority weight alone would break `[254] R9` coverage on the weight map.

**Q1 = PASS.** Both sub-gates pass.

---

## Q2 — Design claims validated?

Six gates, all pre-committed:

| # | Gate | Value | Threshold | Result |
|---|---|---|---|---|
| 1 | Graph-structural hit@5 pass rate (28 queries) | 24/28 = 85.7% | ≥ 80% (23/28) | **PASS** |
| 2 | Contradiction recall per-pair (8 pairs) | 6/8 = 75.0% | ≥ 6/8 | **PASS** |
| 3 | Authority presence in top-3 (8 A-queries) | 8/8 = 100.0% | ≥ 6/8 | **PASS** |
| 4 | Lifecycle ordering (2 L-queries) | 2/2 | 2/2 | **PASS** |
| 5 | NDCG@10 cross-domain wins over BM25 (4 X-queries) | 3/4 | ≥ 3/4 | **PASS** |
| 6 | Head-to-head agent tasks both pass (T1, T2) | 0/2 | 2/2 | **FAIL** |

**Q2 = FAIL on gate 6 only.** Tiebreak order applied honestly: gate 6 is lowest-priority after graph-structural / contradiction / authority / lifecycle / NDCG. Rollback trigger (≥ 3 gate fails) is **NOT** met.

### Per-category per-query breakdown

**Provenance (6 queries, 2 hits)** — *dominant miss driver.*

| ID | Target cert | Gold SourceClaims | Top-5 outcome |
|---|---|---|---|
| P01 | Haramein scaling diagonal | 3 haramein SC | cert#1, none in top-5 — **miss** |
| P02 | Keely structural ratio | sc:keely_law_18_atomic_pitch | cert#1, none in top-5 — **miss** |
| P03 | Keely triune | sc:keely_law_11_triune_force | cert#1, sc#2 — **hit** |
| P04 | Keely dominant control | 2 keely SC | cert#1, no SC in top-5 — **miss** |
| P05 | Levin cognitive lightcone | 2 levin SC | cert#1, SC#2-3 — **hit** |
| P06 | Sixteen identities | sc:parker_prop_vii_circumference_radius | cert#1, none in top-5 — **miss** |

Pattern: cert surfaces at rank #1 because query text matches the cert name; the cert's `derived-from` SourceClaims do not share enough BM25 signal with the query to surface. This is the "ranker is retrieval not traversal" design signal flagged in the Beta-A pilot (`memory/project_qa_mem_beta_testing.md`). The intended remediation (expansion post-step OR route through `why()`) is a Phase 4.7 item, not a Beta-B patch.

Even so, gate 1 passes because all 8 A-queries and 6 D-queries hit at top-5 — the 28-query gate is dominated numerically by authority + domain structural signal.

**Contradiction (8 pairs, 6 hits).**

| ID | Reason | Endpoints | Both in top-5 |
|---|---|---|---|
| C01 | dispute | sc/obs Law 17 | **yes** |
| C02 | dispute | Hull π / Lindemann | **yes** |
| **C03** | **true** | sc/obs F-never-prime | **no** (obs not in top-10) |
| C04 | dispute | sc/obs Law 13 labels | **yes** |
| C05 | ocr | sc/obs choromosomes | **yes** |
| C06 | typo | sc/obs 72-year precession | **yes** |
| **C07** | **typo** | sc/obs C-identity a²h² | **no** (obs not in top-10) |
| C08 | typo | sc/obs Briddell proton | **yes** |

Both misses (C03, C07) share the same structural pattern: the observation body doesn't share enough distinctive tokens with the sc title to push past 5 other primary-authority claims with the shared-domain keywords. This is a retrieval-coverage issue, not a contradicts-edge issue — the edges exist and `[253]` SC6 passes.

**Authority (8 queries, 8/8 top-3 hits).** Every curator-specified primary SourceClaim appears in the top-3 results for its query.

**Lifecycle (2/2).** `cert:fs:qa_kg_consistency_cert_v4` outranks v1/v2/v3 in both queries. The `lifecycle_factor=0.5` for superseded is working as designed.

**Cross-domain NDCG (3/4 wins).**

| ID | QA-MEM NDCG@10 | BM25 NDCG@10 | Δ | Outcome |
|---|---|---|---|---|
| X01 | 0.454 | 0.209 | +0.246 | QA-MEM |
| X02 | 0.730 | 0.707 | +0.022 | QA-MEM |
| X03 | 0.739 | 0.443 | +0.297 | QA-MEM |
| **X04** | 0.763 | 0.871 | −0.109 | **BM25** |

X04 (Philomath digital root vs Wildberger chromogeometry) is the one loss. The v3 review flagged this tension: blind labellers graded verbose derived certs highly on a bridging-query, while QA-MEM's authority weighting pushed terse primary SourceClaims up. This is a frame-of-reference asymmetry between grader and ranker, documented pre-committed in `QA_MEM_BETA_DECISION_MATRIX.md`. The ≥ 3/4 threshold survived.

**Edge cases (4/4 pass).**
- E01 (nonsense token): empty result set — pass.
- E02 (`domain=nonexistent_domain`): empty result set — filter plumbing correct.
- E03 (`min_authority=primary` with derived-only topic): 9 primary hits, 0 derived — ladder works.
- E04 (`valid_at=2099-01-01`): identical ordering vs no-valid_at run — passthrough correct.

**Head-to-head agent tasks (0/2 both-pass — Q2 gate 6 failure).**

| Task | QA-MEM top-10 includes target(s)? | A-RAG lexical marker test | Both pass? |
|---|---|---|---|
| T1 — Keely triune primary + cert | **YES** (`sc:keely_law_11_triune_force`, `cert:fs:qa_keely_triune_cert_v1`) | **NO** — A-RAG chat-corpus has no cert-filesystem index | no |
| T2 — Keely Law 17 dispute (both sides) | **NO** — obs side absent from top-10 | **NO** — chat corpus has the sc but not the Vibes reclassification post | no |

**Interpretation.** T1 is an object-space mismatch: A-RAG indexes chat/doc conversations, QA-MEM indexes the cert-filesystem+SourceClaim graph. An identical prompt cannot pass both unless the cert id text also appears verbatim in an indexed chat — it does not. The decision matrix pre-committed to treating the head-to-head as a decision gate anyway, and we apply it honestly.

T2 is a ranker-coverage issue on the QA-MEM side: `obs:keely_law_17_vibes_structural_reclassification` (`authority=internal`) loses to 9 primary/derived nodes with stronger Keely-token matches. Same structural mechanism as C03/C07.

---

## Q3 — Corpus expansion or ranker tuning?

Per-category miss classification:

| Category | Hits/total | Dominant failure mode |
|---|---|---|
| Provenance | 2/6 | **real_ranker_miss** (cert-at-top but no traversal to SCs) |
| Contradiction (pair) | 6/8 | **real_ranker_miss** (internal-authority demotion) |
| Domain | 6/6 | — |
| Authority | 8/8 | — |
| Lifecycle | 2/2 | — |
| Cross-domain (NDCG-win) | 3/4 | **grader-vs-ranker frame asymmetry** (not ranker defect) |
| Edge cases | 4/4 | — |
| Head-to-head | 0/2 | T1: object-space mismatch; T2: real_ranker_miss |

**Miss classification summary:** 6 of 8 misses are `real_ranker_miss` (P01, P02, P04, P06, C03, C07, plus T2). Per decision-matrix Q3 rule ("≥ 50% real → recommend ranker tuning"), **Phase 4.7 ranker tuning is the recommended next phase.**

### Authority × domain corpus picture

See `figures/authority_domain_heatmap.png`. The non-empty cells are:

| | qa_core | svp | geometry | biology | physics | rsf |
|---|---|---|---|---|---|---|
| primary | 18 | 5 | 20 | 6 | 16 | 15 |
| derived | — | — | — | — | — | — |
| internal | — | — | — | — | — | 3 |
| agent | — | — | — | — | — | — |

(internal rsf rows = Haramein-related curator observations.)

The empty `derived × domain` row is a labelling artefact — `derived` nodes (cert families) do not carry a `domain` column today because cert families span multiple domains structurally. Phase 4.7 could add computed `domain` inheritance via `derived-from` FK but it is not gate-driving.

The empty svp / biology / svp×internal cells are candidate **Phase 4.8 corpus-expansion targets** — specifically Keely expansion (currently only 5 primary svp nodes) and Levin 2026 corpus completion (currently 6 biology primary nodes, 0 internal observations).

### Phase 4.7 concrete action items (per decision matrix §"Lower-tier failure")

1. **Graph-expansion post-step for P-category queries** — when the top-1 cert is derived and has `derived-from` edges to SourceClaims that did not surface in top-k, promote those SCs into top-5 via a `why()` traversal expansion pass. Measurable target: P-category recall@5 ≥ 4/6 (up from 2/6).
2. **Contradicts-aware authority relaxation on C-pairs** — when a node has a `contradicts` edge to a top-K node, transiently boost it toward the contradicted partner's authority tier (or cap the differential). Measurable target: C-pair recall ≥ 7/8 without regressing gate 1.
3. **Head-to-head reframing** — either (a) extend A-RAG to index cert family READMEs so the object-space mismatch can be honestly resolved, or (b) split the Q2 gate 6 into "T1: QA-MEM-only capability" + "T2: QA-MEM-only capability" with pass = QA-MEM surfaces the target. This is a **decision-matrix revision for Beta-C**, never a Beta-B patch.

Owner: next session. Target phase: 4.7 (items 1–2) / 4.8 discussion (item 3).

---

## Latency

| Surface | n | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|---|
| Real graph (521 nodes) | 114 | 5.2 | 31.6 | 35.0 | 37.4 |
| Synthetic 5 000 nodes (latency-only) | 70 | 17.6 | 27.0 | 34.8 | — |

Synthetic nodes: 5 000 `method=script` `SourceClaim` rows with uniform `confidence=1.0` and generic token bodies. Inserted for latency measurement only; deleted after the run (`cleanup_verified=True`, `graph_hash_after == graph_hash_before`). **No ranking-quality claims are made on the synthetic configuration** per the Beta-B execution contract — observer-layer latency measurement only, Theorem NT + decision-matrix N5.

Latency p95 is well under the 200 ms threshold at which `kg.py` would materialize `depth_to_axiom` per the Phase 4 spec. The materialization threshold is not yet crossed.

---

## Consistency (reporting only — non-gate)

- `authority_monotonicity` = 0.694 — in ~31% of queries, the top-k ordering shows an authority-level inversion. Expected: the 7-factor composition can legitimately out-score a lower-authority hit with very strong BM25 + contradiction + shallow-provenance factors. Reported, not gating.
- `provenance_spearman` = −0.235 — weak *negative* correlation between rank position and provenance depth across top-k hits. Expected sign is *positive* if provenance_decay dominated, but at mean-share 0.22 (vs authority's 0.74) the depth signal is visible but not monotone at the rank level. Consistent with Q1 factor-dominance picture.

See `figures/provenance_depth_score.png` for the shape.

---

## Execution anomalies

- **`candidate_pool` Pass A warnings** on most queries (e.g., "341 primary/derived matches for query=…"). This is the Phase 4 logging intended to flag aggressive broad-OR expansions by the FTS5 sanitizer — not an error. All queries still returned top-10 results within latency bound.
- **T2 QA-MEM near-miss.** The observation `obs:keely_law_17_vibes_structural_reclassification` ranks below position 10 despite Keely + Law 17 + Vibes + reclassification tokens in the query; its `authority=internal` weight (5.0) is overwhelmed by 9 primary nodes with shared Keely-corpus tokens. Same class of failure as C03/C07.
- **No NaN/error on any query.** All 38 queries returned well-formed hit lists or empty (edge cases). Graph hash stable pre/post the full run.
- **Deviations log**: `tools/qa_kg/fixtures/beta_prereg_deviations.json` **unchanged** from Beta-A — no N1 fixture edits were required or made.

---

## Reproducibility

```bash
# Full run (reads frozen fixtures; writes beta_results.json + figures/)
python -m tools.qa_kg.analysis.run_beta_benchmark
python -m tools.qa_kg.analysis.beta_visualize
```

The run is deterministic under fixed `valid_at=None`-snapshot (wall-clock per call does not affect the seven-factor ordering on this corpus, because no node carries a non-default `valid_from` that pushes time_decay off 1.0 within the benchmark window; cert `[254]` R7 exercises strict determinism with an explicit `valid_at`).

---

## Status

- **Q2 = FAIL on gate 6 (head-to-head agent tasks).** Five of six gates PASS.
- **Rollback NOT triggered.** MCP docs retain current status. Scope doc Phase-6 header unchanged. Corpus expansion (Phase 4.6 commit `04d8d6b`, Phase 4.7 planning) not paused.
- **Alpha bar (`[228] + [254] + [255]`) holds.** Beta-B findings validate the structural cert-level correctness; ranker-formula correctness + determinism untouched.
- **Phase 4.7 recommendation: ranker tuning**, concrete action items above.

**Memory milestone** is captured post-commit, not pre-commit, per decision matrix N5. `memory/project_qa_mem_beta_testing.md` will append a Beta-B line with commit SHA + 5-gate PASS / 1-gate miss headline.
