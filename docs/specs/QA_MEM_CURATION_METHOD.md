<!-- PRIMARY-SOURCE-EXEMPT: reason=process documentation for QA-MEM fixture-curation workflow; describes method, no empirical claims requiring primary-source citation. Verified end-to-end on Wildberger Phase 4.6 corpus 2026-04-28. -->
# QA-MEM Curation Method (worked Wildberger example, 2026-04-28)

**Scope**: how to author `derived_from` edges in `tools/qa_kg/fixtures/source_claims_*.json`
so a corpus's primary-source claims become load-bearing premises for the cert families
they actually ground. Verified end-to-end on the Wildberger corpus (Phase 4.6); the
process freeze below is the reusable template for the remaining 11 unlinked corpora.

## Why this exists

Phase 4.6 ingested ~100+ primary-source claims across 15 corpora into QA-MEM, but only
4 corpora (haramein, keely, levin, parker_hull) had any `derived_from` cert→claim edges.
The cert↔claim infrastructure was fully wired (extractor + per-corpus fixtures); the
gap was *fixture authorship*, not code. Diagnostic baseline `derived_from=13` reflected
that gap.

Without `derived_from` edges, the QA-MEM graph cannot answer "which cert families are
source-grounded?" or "which claims are activated by certs?" — credibility-audit
questions that motivate the diagnostic in the first place.

## The 6-step process (Wildberger-verified)

1. **Capture before-snapshot.** Run `python -m tools.qa_kg.cli build` then
   `python tools/qa_kg/diagnostics.py`. Save the snapshot header line and the
   target corpus's row from the fixture-coverage table.

2. **Triage every claim in the corpus.** For each claim in the fixture's `claims` array,
   identify candidate cert families by scope-note keyword + body text. Use the
   `qa_kg.db` cert nodes (`SELECT id, title, body FROM nodes WHERE node_type='Cert' AND
   id LIKE 'cert:fs:%' AND body LIKE '%<keyword>%'`) — cert `body` fields contain the
   `pass_desc` text, which is what the validator actually grounds in.

3. **Provisional-decision pass.** For each claim, choose
   `PROVISIONAL_APPROVE [NNN]` / `PROVISIONAL_REJECT` / `PROVISIONAL_DEFER` per the
   strict rule below. Do NOT outsource this to the human; make the call yourself.
   Output a compact summary table (`claim_id | decision | cert | one-line rationale`)
   for human review.

4. **Human approve / override.** Human reviews the table, marks any flips. Total
   review effort ~1-2 minutes per corpus when the table is well-organized.

5. **Write the APPROVE entries** into the fixture's `derived_from` array. Format:
   ```json
   {"src": "cert:fs:<family_root>", "dst": "sc:<claim_id>", "rationale": "<one sentence>"}
   ```
   `src` resolves via the cert's family-root id (verify via `SELECT id FROM nodes WHERE
   id LIKE 'cert:fs:%' AND title LIKE '%<keyword>%'`).

6. **Rebuild + verify.** Run `python -m tools.qa_kg.cli build`,
   `python tools/qa_kg/diagnostics.py`. Compare snapshot header + corpus row.
   Expected delta: `derived_from` increases by exactly the APPROVE count;
   the corpus's `own_unlinked` decreases by the APPROVE count;
   `%own_linked` rises proportionally; `fixtures_with_links` increments by 1
   (or 0, if the corpus already had any links).

## Rules

### Strict load-bearing rule (the discipline)

- **APPROVE** only when the claim supplies a source-backed premise an existing cert
  *actually depends on*. The cert's `body` (pass_desc) must reference the claim's
  content concretely — formulas, structures, or named theorems — not just touch the
  same topic.
- **REJECT** when the claim is philosophically compatible but not load-bearing.
  Examples: rhetorical assertions ("transcends Klein's Erlangen program"), reframings
  ("affine vs projective is the real division"), pictorial/closure follow-ups.
- **DEFER** when the claim is relevant but the cert dependency is not yet explicit.
  Examples: foundational ancestor papers that no current cert directly cites; sibling
  claims from the same paper as an APPROVE that don't add new premise content.

### One cert per claim (default, first-pass)

A claim usually grounds one cert. Listing 2-3 candidates during triage is fine, but
the APPROVE pass picks at most one unless the claim is *explicitly* load-bearing for
multiple certs. Two source claims feeding the same cert is acceptable when they
ground different parts of the cert body (e.g., Wildberger 3 + 4 both → [240], one for
the integer-rep model, one for quark/anti-quark triples).

### No negative edges (v0)

REJECT writes nothing. There is no inverse-edge type, no "rejected-candidate" record.
The absence of an edge is the negative signal.

## Worked example: Wildberger (2026-04-28)

- **Corpus**: 17 claims across 12 works (Wildberger 2003-2010 + Notowidigdo+Wildberger
  2019 + Rubine 2025 + Amdeberhan+Zeilberger 2025 + Mukewar 2025).
- **Triage artifact**: `tools/qa_kg/_scratch_wildberger_triage.md` — 17 cards, 8 fields
  per card, candidate certs with one-line "might ground" / "might not ground"
  reasoning. This file is scratch (leading underscore); it documents the pass but is
  not an authoritative deliverable.
- **Provisional decisions** (Claude's first pass):
  8 PROVISIONAL_APPROVE / 5 PROVISIONAL_REJECT / 4 PROVISIONAL_DEFER.
- **Human override**: 1 flip (claim 7 DEFER → APPROVE [241]). Final tally:
  9 APPROVE / 5 REJECT / 3 DEFER.
- **Outcome**:
  ```
  BEFORE: derived_from=13  fixtures_with_links=4/15  wildberger 0/17 = 0% own_linked
  AFTER:  derived_from=22  fixtures_with_links=5/15  wildberger 9/17 = 53% own_linked
  ```
  No extractor warnings. No firewall violations. Loop time ~30 minutes including
  triage authorship; ~10 minutes per corpus once the template is internalized.
- **Approved entries** (9 cert↔claim mappings across 8 certs; [240] receives two source claims):
  1. [240] Diamond sl3 ← `wildberger_2003_diamonds_abstract` (integer-rep model premise)
  2. [240] Diamond sl3 ← `wildberger_2003_diamonds_quark_triples` (quark/anti-quark triples premise)
  3. [244] Mutation Game Root Lattice ← `wildberger_mutation_game_problem`
  4. [241] Quadruple Coplanarity ← `notowidigdo_wildberger_2019_tetrahedron_abstract`
  5. [127] UHG Null ← `wildberger_2009_uhg_i_abstract`
  6. [125] Chromogeometry ← `wildberger_2008_chromogeometry_abstract`
  7. [251] G2 Mutation Game ← `wildberger_2003_g2_abstract`
  8. [141] Pell Norm ← `wildberger_2008_pell_abstract`
  9. [231] Hyper-Catalan Diagonal ← `rubine_2025_hyper_catalan_definition`

## Stale-DB caveat (discovered during the Wildberger pass)

The diagnostic reports DB state, not fixture state. If fixtures have been edited since
the last `cli build`, counts will be stale. The Wildberger pass surfaced ~92 claim
nodes and 15 work nodes that had been declared in fixtures but were not yet in
`qa_kg.db`. **Always run `python -m tools.qa_kg.cli build` before
`python tools/qa_kg/diagnostics.py`** unless you specifically want to compare against
the prior DB state.

## Next corpora

Ordered by curation effort (lowest to highest, by Will's familiarity with the source
material per memory):

1. **HeartMath / McCraty / Schumann** (item 6 of Phase 4.8 already partly authored).
2. **Briddell** (Will is co-author on the FST manuscript; intimate knowledge).
3. **Philomath / Iverson** (Will has read).
4. **Kochenderfer** (recent ingest, 2026-04-26; bridge already drafted at
   `docs/specs/QA_KOCHENDERFER_BRIDGE.md`).
5. **dm / optimization** (Kochenderfer companion volumes — defer until kochenderfer is curated).
6. **phase3** (mixed-source baseline; lowest priority).

Do not start the next corpus until this method doc is reviewed; method changes
discovered mid-stream are cheaper to capture before the second pass than after.
