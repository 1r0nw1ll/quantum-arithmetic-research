<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database row template (catalog scaffold). Source citations + evidence links are filled in per-entry; this is the structural skeleton, not a research claim. (Kochenderfer, 2026; Dale, 2026)." -->

# `<algorithm_slug>` — Algorithm Database Entry Template

> Copy this file to `entries/<algorithm_slug>/README.md` and fill in each section. Delete sections that don't apply (`test_equivalence.py` is OPTIONAL — see "When to add a test" below).

## Source reference

- **Source**: `<book / notebook / repo>` (e.g., (Kochenderfer 2022) *Algorithms for Decision Making*, MIT Press)
- **Chapter / section**: `§<X.Y>` (e.g., §7.5)
- **Anchor in QA-MEM**: `docs/theory/kochenderfer_<book>_excerpts.md#<anchor>` (link to the verbatim anchor we ingested in QA-MEM Phase 4.x)
- **Original code location** (if from `algforopt-notebooks` or another repo): `<repo>/<notebook>/<cell>` — and a note about whether the code has been fetched yet (the repo manifest in `sources/` tracks fetched-vs-not status).

## Classical mathematical form

A 2-5 line summary of the algorithm in canonical mathematical notation. Quote the verbatim form from the QA-MEM anchor where possible (with attribution), or transcribe with an explicit `(transcribed from §X.Y)` note.

## Classical pseudocode / code

- **Pseudocode** (`classical.md`): the book's algorithm box, transcribed with attribution. Do **not** copy long Julia/Python blocks unless they are already in the QA-MEM excerpts file (which serves as the verbatim source-of-truth) or have been fetched with proper attribution from the source repo.
- **Python port** (`classical.py`, OPTIONAL): a runnable Python translation of the algorithm. Add only if (a) the algorithm is small enough that a faithful port is achievable in <50 lines, AND (b) the port is useful as a baseline for the QA equivalence test.

## QA mapping (`qa_mapping.md`)

This is the load-bearing field. Every entry must declare:

- **QA mapping status**: one of `established` / `candidate` / `open` / `rejected` (mirrors the bridge spec status field; same semantics).
- **QA counterpart**: pointer to the existing artifact (cert family, utility module, bridge row) or `none (open candidate)` / `none (rejected — Theorem NT firewall)`.
- **Evidence link**: where the empirical claim lives (cert with PASS, fixture, witness states, etc.) — only filled in if status is `established`.
- **Theorem NT boundary note**: which side of the firewall the classical algorithm lives on, which side the QA counterpart lives on, and how the boundary is crossed if at all (e.g., "classical algorithm is continuous-domain; QA counterpart is integer-only on `S_9`; the firewall is crossed at the input-tokenization boundary, not in the algorithm body").

## QA-native equivalent (`qa_native.py`, OPTIONAL)

A QA-side implementation of the algorithm's QA counterpart, where one exists. Should reuse `tools/qa_kg/orbit_failure_enumeration.py` (or whichever existing QA infrastructure) — does **not** re-implement utilities. Often this section is a thin wrapper over an existing cert or utility, with a docstring that explains the mapping.

## Equivalence test (`test_equivalence.py`, OPTIONAL — see "When to add" below)

A small Python test that exercises the classical and QA-native implementations on a canonical input and verifies the equivalence claim documented in `qa_mapping.md`.

### When to add a test

- **YES, add a test** when:
  - There's an actual equivalence claim (e.g., "QA enumeration reproduces Kochenderfer's exact ratios on `S_9`")
  - Both classical and QA-native implementations are available as runnable code
  - The equivalence is testable on small canonical inputs (no MCMC, no external data)
- **NO, skip the test** when:
  - The mapping is conceptual (e.g., "classical alpha-vector pruning corresponds to QA orbit-class equivalence pruning") — not a literal output equivalence
  - The classical algorithm is firewall-rejected as causal QA dynamics — there's no equivalence to test on the QA-discrete side
  - The QA counterpart is `open` (no implementation exists yet)

**Anti-pattern**: do **not** create fake equivalence tests for entries that only have conceptual mappings. A test that reduces to "both implementations return strings" is noise, not evidence.

## Cross-references

- Bridge spec row: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §`<N.M>`
- Cert family (if `established`): `[<NNN>] <cert_slug>` at `qa_alphageometry_ptolemy/<cert_slug>_cert_v1/`
- Utility (if reused): `tools/qa_kg/<utility_module>.py`
- Related entries: list other `algorithm_database/entries/<slug>/` rows that share concepts or evidence
