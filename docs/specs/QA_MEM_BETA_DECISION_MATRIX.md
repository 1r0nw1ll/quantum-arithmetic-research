<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Beta-A decision matrix; pre-committed thresholds before Beta-B benchmark execution; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md, and the v3.1 review feedback cycle documented in this session. Reviewer-approved with four revisions (I1/I2/I4/I5) + one non-blocking (I6 pending implementation). -->

# QA-MEM Beta Decision Matrix

**Status**: pre-committed 2026-04-17. Thresholds below are binding for the Beta-B benchmark run. No threshold may change after Beta-A commit lands. Deviations are logged in `tools/qa_kg/fixtures/beta_prereg_deviations.json` (not silently adjusted).

**Authoring context**: Beta-A committed the 38-query pre-registered fixture, graph-structural gold (axioms stripped from provenance), curator-specified primary anchors for authority queries, and blind-labeled relevance grades for cross-domain queries. Beta-B executes the benchmark, computes metrics, applies thresholds, and produces the report.

**Pre-registration discipline**: Gold labels were derived BEFORE the ranker was run. Fixture changes after ranker execution are explicitly forbidden (N1 constraint). The decision matrix below was authored BEFORE the pilot results (8/10 hit; 2 P-queries surfaced cert-at-top but not SourceClaims-in-top-5) were interpreted. The pilot informs scope, not threshold.

---

## Three design-claim questions

These are the three questions Beta-B answers. They are NOT "does QA-MEM replace A-RAG" — A-RAG and QA-MEM index different object spaces (A-RAG: 58k chat/doc messages; QA-MEM: 521-node knowledge graph with epistemic-field invariants). The A-RAG comparison is an appendix observation, not decision-driving.

### Q1 — Is the ranker formula well-tuned?

**Metrics**

- **Factor dominance** (log-space contribution share): aggregated over all 28 graph-structural + 8 authority queries = 36 queries' top-1 breakdowns.
  - Per-query, `contribution_share_i = |log(factor_i)| / sum(|log(factor_j)|)` across the 7 factors.
  - Per-query dominance score = max(contribution_share_i).
  - Aggregate: `dominated_fraction = count(dominance > 0.80) / total_queries`.
- **Contradiction-boost ablation** (contradiction_prior ∈ {1.0, 1.25, 1.5, 1.75, 2.0}): run the benchmark five times at each setting; report per-setting contradiction_recall_per_pair + graph_structural_pass_rate.

**Thresholds**

- **Factor dominance** — PASS if `dominated_fraction ≤ 0.50`. Interpretation: no single factor drives >80% of the score for more than half the queries.
- **Contradiction-boost ablation** — PASS if `contradiction_prior=1.5` is Pareto-optimal: for every alternative (1.0, 1.25, 1.75, 2.0), at least one of {contradiction_recall_per_pair, graph_structural_pass_rate} is weakly worse at the alternative, AND at least one is strictly worse. If neither dominance relation holds, PASS only if 1.5 is on the Pareto frontier.

### Q2 — Does the authority + provenance structure deliver on its design claims?

**Metrics + thresholds**

| # | Metric | Denominator | Gate |
|---|---|---|---|
| 1 | Graph-structural hit@5 pass rate | 28 (P + C + D + A queries) | ≥ 80% (≥23/28 queries with ≥1 gold node in top-5) |
| 2 | Contradiction recall per-pair (both endpoints in top-5) | 8 C-queries | ≥ 6/8 |
| 3 | Authority presence in top-3 | 8 A-queries | ≥ 6/8 |
| 4 | Lifecycle ordering (current outranks all superseded in top-5) | 2 L-queries | 2/2 (both) |
| 5 | NDCG@10 cross-domain (QA-MEM ≥ BM25 baseline) | 4 X-queries | ≥ 3/4 |
| 6 | Head-to-head agent tasks | 2 (T1, T2) | both pass |

**PASS = all 6 gates pass**. Any single gate miss drops Q2 to the next tier in the tiebreak order.

**Tiebreak order** (if Q2 fails, diagnostic priority):
1. Graph-structural hit@5 pass rate
2. Contradiction recall per-pair
3. Authority presence in top-3
4. Lifecycle ordering
5. NDCG@10 cross-domain
6. Factor dominance

**Head-to-head rubric**: graders receive System A / System B output labels only; no system-of-origin labels. Pass = both the primary-source quote AND the cert family are identifiable (T1), or both sides of the dispute are identifiable (T2). Partial credit not awarded.

### Q3 — What should next work be: corpus expansion or ranker tuning?

**Editorial section**. No threshold. Beta-B report's Q3 section records:

- Per-category hit@5 breakdown (P, C, D, A, L, X, E)
- Per-category classification of misses (real_ranker_miss / graph_incomplete / gold_spec_bug)
- Narrative recommendation: if ≥50% of misses are graph-incompleteness false negatives → recommend corpus expansion as next phase; if ≥50% are real ranker misses → recommend ranker tuning (Phase 4.7).

---

## Lower-tier failure — Phase 4.7 handoff, not rollback

If Q1 fails factor-dominance but Q2 passes: file as Phase 4.7 ranker-tuning action item with owner + target phase. **Not automatic rollback.**

If Q1 passes but one Q2 gate fails: file the failing metric as Phase 4.7 action item with a concrete candidate remediation (e.g., "contradiction_recall 5/8 — prior=1.75 dominates in ablation → bump constant and rerun gate").

---

## Rollback operationalization — only if Q2 fails ≥ 3 gates

If Q2 fails ≥ 3 of its 6 gates, rollback triggers:

1. **MCP docs get `UNVALIDATED` banner** — `tools/qa_kg_mcp/README.md` top-of-file warning: "QA-MEM beta validation found <N> design-claim failures; the MCP tools remain functional but their output has not been empirically validated on pre-registered queries."
2. **Scope doc status revert** — `docs/specs/QA_MEM_SCOPE.md` Phase 6 header changes from "alpha achieved 2026-04-16" to "indexed catalog, beta-failed on [failing gates list]". The three-cert bar ([228] + [254] + [255]) still PASSes at the cert level — the reversion is about design-claim validation, not structural correctness.
3. **Corpus expansion pauses** — no new SourceClaims or SourceWorks ingested until the failing gates are addressed in Phase 4.7 (ranker tuning) or Phase 4.8 (graph-completeness expansion).
4. **Schema, firewall, certs REMAIN UNTOUCHED**. Rollback does not revert schema v4, the Phase 2 agent firewall, or any cert family. The alpha bar is about claim-validation, not infrastructure existence.

**Not triggered by Q1 or Q3**. Q1 is ranker-tuning territory; Q3 is editorial.

---

## Tiebreaker worked example

Hypothetical Beta-B outcome: Q1 PASS (dominated_fraction = 0.31); Q2 fails gate #5 only (NDCG@10 = 2/4 < 3/4); Q2 passes gates 1-4 and 6. Q3: misses concentrated in X-queries (cross-domain), classified as graph-incompleteness false negatives.

Decision:
- Q2 = **FAIL** (1 gate miss).
- Rollback = **NOT TRIGGERED** (< 3 gates failed).
- Phase 4.7 action item = "NDCG@10 cross-domain < 3/4; investigate whether graph-completeness or ranker's cross-domain semantic bridge is the cause." Owner: next session. Target phase: 4.7 or 4.8 depending on Q3 finding.
- Report framing: "QA-MEM alpha structural certification stands; cross-domain retrieval design claim failed on 1/6 gates and is deferred to Phase 4.7 investigation."

---

## Artifact SHAs (pre-committed fixture references)

| Artifact | Purpose |
|---|---|
| `tools/qa_kg/fixtures/beta_prereg_queries.json` | 38-query fixture, pre-committed |
| `tools/qa_kg/fixtures/beta_prereg_gold.json` | graph_structural + curator + mechanical gold |
| `tools/qa_kg/fixtures/beta_blind_gold.json` | Opus-4.7 blind grades for X01-X04 |
| `tools/qa_kg/fixtures/beta_prereg_contradiction_audit.json` | lexical pre-flight of 8 C-pairs |
| `tools/qa_kg/fixtures/beta_pilot_report.json` | 10-query pilot output (8 hit, 2 miss on P01/P02) |
| `tools/qa_kg/fixtures/beta_prereg_deviations.json` | N1 fixture-fix log (empty at Beta-A commit) |

All six are SHA-committed atomically. Reproducibility: `python -m tools.qa_kg.analysis.derive_gold` + `python -m tools.qa_kg.analysis.blind_label_prompt` + `python -m tools.qa_kg.analysis.pilot_validate` against the same `graph_hash` regenerates byte-identical artifacts (modulo LLM nondeterminism on blind grades; raw responses are committed for audit).
