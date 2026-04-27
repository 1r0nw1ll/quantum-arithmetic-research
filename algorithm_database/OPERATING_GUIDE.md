<!-- PRIMARY-SOURCE-EXEMPT: reason="Operator guide for the algorithm_database lane. Documents lane boundaries (corpus / bridge / certs / database) so future sessions do not drift across them. (Kochenderfer, 2026; Dale, 2026)." -->

# Algorithm Database — Operating Guide

Read this **before** adding any new entry, fetching any new source, or proposing changes to existing entries. This guide locks in the lane boundaries that were established across the 2026-04-26/27 work blocks. Drift across these boundaries is the failure mode this guide prevents.

## The four lanes

```
QA-MEM corpus      = primary-source evidence (PDFs, fixtures, qa_kg.db)
Kochenderfer bridge = QA vocabulary / mapping layer (docs/specs/QA_KOCHENDERFER_BRIDGE.md)
Cert ecosystem     = empirical sharp claims with PASS/FAIL validators (qa_alphageometry_ptolemy/<cert>_v1/)
algorithm_database = queryable algorithm catalog (this directory)
```

Each lane has its own discipline. **The database lane is the smallest and the most disciplined**: it does not produce new evidence, it indexes existing evidence into a queryable catalog.

## What the database lane IS

- A **front-door / query layer** over the other three lanes.
- A catalog of algorithm rows: each row maps a classical algorithm to its source location, its QA mapping status, and a pointer to evidence (cert / utility / bridge row).
- A way for a future reader to answer: *"Given this algorithm from this textbook, what's the QA conversion?"*
- An additive structure: anyone can copy `TEMPLATE/`, fill in fields, and add a row.

## What the database lane is NOT

- **NOT** a primary-source corpus. SourceWorks live in `qa_kg.db` via QA-MEM Phase 4.x. Cloned upstream repos in `external_sources/` are gitignored references, not corpus.
- **NOT** a research-claim doc. New empirical claims belong in the cert lane (`qa_alphageometry_ptolemy/<cert>_v1/`), not in `entries/<algo>/qa_mapping.md`.
- **NOT** a vocabulary-mapping spec. Vocabulary alignments belong in `docs/specs/QA_KOCHENDERFER_BRIDGE.md` rows. The database row points at the bridge row, doesn't duplicate it.
- **NOT** a place to ingest notebook code as algorithm bodies unless the algorithm is genuinely best-anchored at the notebook (rare — book pseudocode is usually canonical).

## Add-an-algorithm checklist

For each new entry, in order:

1. **Confirm the algorithm has a non-trivial QA mapping status.** If the status would be "unclear" or "needs research," stop — that's a bridge-spec / cert-lane question, not a database-lane question. Add it to the bridge as a candidate row first; come back here once the status is `established` / `candidate` / `open` / `rejected`.
2. **Pick a slug** in lowercase snake_case (e.g., `policy_iteration`).
3. **Create `entries/<slug>/`** by copying `TEMPLATE/ALGORITHM_ENTRY_TEMPLATE.md` to `entries/<slug>/README.md`.
4. **Fill in the source reference** with concrete pointers:
   - QA-MEM verbatim anchor (`docs/theory/kochenderfer_*_excerpts.md#<anchor>`) — required for every entry; it's the load-bearing attribution.
   - Source repo file + line range (e.g., `decision_making_code.jl L617-635`) — when an inventory exists in `sources/`.
5. **Fill in the QA mapping** (`status`, `QA counterpart`, `bridge spec row pointer`, `Theorem NT boundary note`).
6. **Decide on `classical.py`** (see "When to add classical.py" below).
7. **Decide on `qa_native.py` and `test_equivalence.py`** (see "When NOT to add" below).
8. **Add a row to `INDEX.md`** with status, source, evidence pointer.
9. **Run smoke tests** (see "Smoke-test commands" below).
10. **Commit** with title `feat(algorithm_database): add <slug> entry (vX.Y)`. Include a 1-line summary of status and evidence in the commit body.

## When to add `classical.py`

- **YES** when the algorithm is small enough that a faithful Python port is achievable in <50 lines AND the port is useful as a runnable baseline.
- **NO** when the algorithm requires an external solver (LP/QP/MIP), a deep-learning framework, or thousands of lines of supporting infrastructure. Cite the original code repo's location instead and note "port deferred — external solver required" in the README.

## When NOT to add `qa_native.py`

- **NEVER** add `qa_native.py` if the QA counterpart is already implemented as a cert family or utility module elsewhere (e.g., `tools/qa_kg/orbit_failure_enumeration.py` provides `orbit_family_s9`). Reference the existing artifact in `qa_mapping.md` and stop.
- **NEVER** add `qa_native.py` for entries with status `rejected` (the QA counterpart explicitly does not exist; the entry documents the firewall rejection).
- **NEVER** add `qa_native.py` for entries with status `open` (the QA counterpart is a future cert candidate; building the code without the cert reverses the discipline).
- **YES** add `qa_native.py` only when the QA counterpart is genuinely new code that doesn't fit any existing utility AND isn't load-bearing enough to deserve its own cert family.

In v1.0–v1.2, **zero entries needed `qa_native.py`** — every QA counterpart was already covered by an existing cert or utility.

## When to add `test_equivalence.py`

- **YES** when there's an actual output-level equivalence claim (e.g., "QA enumeration reproduces Kochenderfer ratios bit-exact"), AND both classical and QA-native implementations are runnable on canonical inputs, AND no existing cert already provides the equivalence evidence.
- **NO** when the mapping is conceptual — e.g., "classical alpha-vector pruning corresponds to QA orbit-class equivalence pruning." A test that reduces to "both implementations return strings" is noise, not evidence. This is the **fake-test anti-pattern**; do not do it.
- **NO** when an existing cert already provides the empirical evidence. Cert [263] proves the cosmos morphospace ratios; entry `value_iteration` references it rather than duplicating.

In v1.0–v1.2, **zero entries needed `test_equivalence.py`** — all equivalence claims either had cert evidence or were conceptual mappings.

## When a mapping becomes a cert candidate

If during entry authoring you discover a sharp falsifiable claim that doesn't yet have cert evidence, **do not author the cert in the database lane**. Instead:

1. Add the entry with `status: open` and a pointer to a future cert claim.
2. Add the candidate to `docs/specs/QA_KOCHENDERFER_BRIDGE.md` Standing Rule #2 (the open-candidates list).
3. **Stop.** The cert authoring is a separate session in the cert lane: family-ID reservation, schema/validator/fixtures, FAMILY_SWEEPS registration, gate verification, etc. Do not interleave database work with cert work in the same session — that's a guaranteed drift trigger.

## Required: Theorem NT boundary note

Every entry's `qa_mapping.md` must declare:
- Which side of the Theorem NT firewall the classical algorithm lives on (continuous-domain / discrete-domain / hybrid).
- Which side the QA counterpart lives on (always discrete on the QA-causal side).
- How the boundary is crossed if at all (input projection / output projection / no crossing required).

This is non-negotiable. The database is downstream of Theorem NT; entries without a clear boundary statement are scope-drift bait.

## Required: source pointers

Every entry must cite:
- A QA-MEM verbatim anchor in `docs/theory/kochenderfer_*_excerpts.md` — required (this is the corpus-side anchor).
- A source-code repo file + line range — required only if `sources/<repo>_inventory.md` exists for the relevant repo. If the repo isn't fetched yet, write "NOT YET FETCHED — see [`sources/<repo>_manifest.md`](sources/<repo>_manifest.md)" and move on.

## Adding a new source repo (fetching upstream)

The `external_sources/.gitignore` already lists 11 known upstream repos. To add a new one:

1. `cd algorithm_database/external_sources/ && git clone --depth 1 <repo-url>`. The clone stays local; the gitignore prevents accidental commit.
2. Author `sources/<repo>_inventory.md` listing what's in the clone (per the v1.1 / v1.2 patterns).
3. Update `sources/algorithmsbooks_org.md` (or the equivalent org-level manifest) to flip the row's status from "NOT YET FETCHED" to "FETCHED + INVENTORIED <date> (vX.Y)".
4. Update relevant entry READMEs to flip "NOT YET FETCHED" lines to specific line ranges.
5. **Do not** ingest cloned content into QA-MEM as a SourceWork. SourceWork creation is a Phase 4.x corpus decision in the corpus lane, not the database lane.

## Smoke-test commands

After any database edit:

```bash
# Linter must be clean (CLEAN — no violations found):
python3 tools/qa_axiom_linter.py --all

# Each entry's classical.py self-test must run cleanly:
for f in algorithm_database/entries/*/classical.py; do python3 "$f"; done

# Human-tract gate must still PASS (database adds no cert family;
# count should stay constant unless you also added a cert in the cert lane):
cd qa_alphageometry_ptolemy && python3 qa_meta_validator.py 2>&1 | grep -E "\[266\]|HUMAN-TRACT"
```

Expected: linter CLEAN, all 7+ self-tests run with sensible numerics, [266] Human-tract gate PASS.

## Anti-patterns (recap)

- ❌ Add a cert from the database lane.
- ❌ Ingest a notebook as a QA-MEM SourceWork without a corpus-lane decision.
- ❌ Author `qa_native.py` when an existing cert/utility already covers the QA counterpart.
- ❌ Author `test_equivalence.py` for conceptual mappings ("fake test").
- ❌ Skip the Theorem NT boundary note.
- ❌ Skip the QA-MEM verbatim anchor (the load-bearing attribution).
- ❌ Bypass family-ID reservation when a database entry surfaces a sharp claim that ought to become a new cert (the cert lane has its own discipline; respect it).

## Reading order for a future session

1. This file (`OPERATING_GUIDE.md`).
2. `INDEX.md` (front-door catalog).
3. `TEMPLATE/ALGORITHM_ENTRY_TEMPLATE.md` (per-entry structure).
4. `sources/*_inventory.md` (what's already fetched and indexed).
5. `entries/<existing>/` (worked examples).
6. `docs/specs/QA_KOCHENDERFER_BRIDGE.md` (the upstream mapping layer this database is a front-door for).

Then propose work. Do not propose work before reading 1-3 at minimum.
