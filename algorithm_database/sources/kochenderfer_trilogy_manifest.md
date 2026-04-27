<!-- PRIMARY-SOURCE-EXEMPT: reason="Manifest pointing at QA-MEM SourceWorks ingested 2026-04-26 / 2026-04-27 (Kochenderfer, 2026; Kochenderfer, 2022) Algorithms for Validation / Decision Making / Optimization. Catalog manifest, not a primary-source ingestion." -->

# Kochenderfer Trilogy — Source Manifest

The three textbooks plus the 2nd-edition supersedes-witness 1st edition. All four SourceWorks already ingested into `qa_kg.db` per QA-MEM Phase 4.x pattern. This manifest is the catalog-side pointer; the actual ingestion artifacts live in QA-MEM.

## Pointers

| SourceWork ID | Title | PDF | Excerpts | Fixture | Bridge §§ |
|---|---|---|---|---|---|
| `kochenderfer_wheeler_2026_algorithms_for_validation` | (Kochenderfer 2026) *Algorithms for Validation*, MIT Press, CC-BY-NC-ND, 441 pp | `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_validation.pdf` (13.9 MB) | `docs/theory/kochenderfer_validation_excerpts.md` (15 anchors) | `tools/qa_kg/fixtures/source_claims_kochenderfer.json` (1 SourceWork, 15 SourceClaims) | bridge §1-§5 |
| `kochenderfer_wheeler_wray_2022_algorithms_for_decision_making` | (Kochenderfer 2022) *Algorithms for Decision Making*, MIT Press, CC-BY-NC-ND, 700 pp | `Documents/kochenderfer_corpus/kochenderfer_wheeler_wray_2022_algorithms_for_decision_making.pdf` (12.1 MB) | `docs/theory/kochenderfer_decision_making_excerpts.md` (15 anchors) | `tools/qa_kg/fixtures/source_claims_dm.json` (1 SourceWork, 15 SourceClaims) | bridge §7 |
| `kochenderfer_wheeler_2026_algorithms_for_optimization_2e` | (Kochenderfer 2026) *Algorithms for Optimization*, 2nd ed., MIT Press, CC-BY-NC-ND, 631 pp | `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_optimization_2e.pdf` (18.9 MB) | `docs/theory/kochenderfer_optimization_excerpts.md` (15 anchors; canonical anchor for both editions) | `tools/qa_kg/fixtures/source_claims_optimization.json` (2 SourceWorks, 15 SourceClaims, 1 supersedes edge) | bridge §8 |
| `kochenderfer_wheeler_2019_algorithms_for_optimization_1e` | (Kochenderfer 2019) *Algorithms for Optimization*, 1st ed., MIT Press, CC-BY-NC-ND, 520 pp | `Documents/kochenderfer_corpus/kochenderfer_wheeler_2019_algorithms_for_optimization_1e.pdf` (8.3 MB) | (anchored at 2nd ed; 1e is registered as SourceWork witness only, supersedes by 2e) | (in `source_claims_optimization.json` as second SourceWork) | bridge §8 |

## Sharp-claim certs derived from the trilogy

| Cert | Source row | Algorithm-database link |
|---|---|---|
| [263] `qa_failure_density_enumeration_cert_v1` | bridge §3 row 1 (Validation §7.1) | (no direct entry — utility provider, surfaces in `entries/iterative_policy_evaluation/` and `entries/value_iteration/` as evidence) |
| [264] `qa_runtime_odd_monitor_cert_v1` | bridge §5 ODD-monitor row (Validation §12.1) | (no direct entry — ODD monitoring is meta to algorithm execution; surfaces in cross-references) |
| [265] `qa_counterfactual_descent_cert_v1` | bridge §5 counterfactual-explanation row (Validation §11.5) | (no direct entry — counterfactual descent is meta to algorithm execution; surfaces in cross-references) |

## Coverage status

The trilogy ingestion covers:
- **15 + 15 + 15 = 45 verbatim algorithm anchors** across the three books.
- **~30 established mappings** in `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §1-§8 (bridge rows tagged `established`).
- **3 sharp-claim cert families** ([263][264][265]) derived from the bridge.
- **9 open candidate cert claims** flagged in bridge Standing Rule #2.

The first 7 algorithm-database entries (this pass) cover a subset of the 45 anchors. Future passes can extend coverage incrementally; the catalog is additive.

## What the trilogy ingestion does NOT cover

- The Jupyter notebooks at `github.com/algorithmsbooks/algforopt-notebooks` (Julia code; not yet fetched). See `algforopt_notebooks_manifest.md` for the catalog of those notebooks.
- The broader `github.com/algorithmsbooks` org (Decision Making notebooks, Validation notebooks, ancillaries, `DecisionMakingProblems.jl`). See `algorithmsbooks_org.md` for the org-level manifest.
- Other algorithm sources (Russell+Norvig AI textbook, Sutton+Barto RL textbook, Sipser theory of computation, etc.) are out of scope for the v1 database.
