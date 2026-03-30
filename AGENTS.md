# QA Project — Agent Role Reference

**Full specification:** `docs/specs/PROJECT_SPEC.md`
**Meta-validator:** `qa_alphageometry_ptolemy/qa_meta_validator.py` (currently 128/128 PASS)

---

## Role Matrix

| Agent | Primary responsibility | Does NOT do |
|---|---|---|
| **Claude** | Orchestration, architecture, analysis, cert design, spec writing | Write experiment code, write doc prose |
| **Codex** | All Python writing, validators, fixtures, schema JSON | Architecture decisions, doc prose |
| **Gemini** | README, paper prose, human tract docs, markdown | Code, architecture decisions |
| **Open Brain** | Observation capture, idea storage, prior result retrieval | — |
| **Git** | Version control at stable checkpoints | — |

---

## Track Ownership

| Track | Primary | Secondary |
|---|---|---|
| A — Algebraic Foundations | Will (math) | Claude (cert design), Codex (verify scripts) |
| B — Specification Infrastructure | Claude (architecture) | Codex (validators, fixtures), Gemini (human tracts) |
| C — Convergence Theory | Will (proofs) | Claude (cert design, analysis), Codex (empirical scripts) |
| D — Coherence Detection | Codex (experiments) | Claude (analysis, [122] certs), Open Brain (capture) |

---

## Claude — Invocation Context

Claude is the orchestrator. Claude reads results, identifies what the system needs next,
and specifies tasks precisely for Codex/Gemini. Claude writes code ONLY when no delegation
path exists.

**Standard inputs Claude reads before any task:**
1. `mcp__open-brain__recent_thoughts` — what happened recently
2. `MEMORY.md` — project state + rules
3. Relevant cert family files if touching the cert ecosystem

**What Claude produces:**
- Task specs for Codex (see §Codex below)
- Architecture decisions documented in `docs/specs/`
- Cert family design (schema + check logic; Codex implements)
- Analysis of experiment results → verdict → [122] cert spec

---

## Codex — Invocation Format

Always provide a complete task spec:

```
TASK: [verb] [specific object]
CONTEXT: [what exists; what's missing; why this is needed]
READ: [list of files Codex must read first]
WRITE: [list of files Codex must create or edit]
CONSTRAINTS:
  - validator must pass --self-test
  - do not change [X]
  - use d*d not d**2 (substrate rule)
VERIFICATION: [how to confirm success]
```

**Codex standards:**
- Every validator: `--self-test` must exit 0, return `{"ok": true}`
- Every script: runnable standalone, no imports between experiment files
- QA substrate: always `d*d`, never `d**2` (CPython pow() vs libm divergence)
- Canonical JSON: `json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`
- Hash domain separation: `sha256(domain.encode() + b'\x00' + payload)`

---

## Gemini — Invocation Format

```
TASK: write [human tract / README section / paper section] for [subject]
AUDIENCE: [reviewer / developer / Will / public]
TONE: [technical-precise / accessible / formal]
SOURCE FILES: [cert files, experiment results, Open Brain IDs to draw from]
REQUIRED SECTIONS: [list]
OUTPUT: [exact file path]
WORD BUDGET: [approx]
```

**Gemini standards:**
- Human tract docs must include: purpose, schema table, validator checks table, fixtures table, family relationships
- No invented claims — draw only from source files provided
- Never guess SVP terminology (see `memory/reference_svp_vocabulary.md`)

---

## Open Brain — Capture Protocol

Capture immediately when:
- Any experiment produces output (even partial)
- A theorem is proved or disproved
- A claim is corrected
- A paper status changes
- An idea worth pursuing is formed

Capture format:
```
type: observation | task | idea | insight
tags: [domain, experiment-name, result-type]
body: [summary + key numbers + honest verdict vs pre-declared criteria]
```

Search before re-deriving: `mcp__open-brain__search_thoughts` with relevant keywords.

---

## Cert Ecosystem — Quick Reference

**Gate 0:** Every cert family root must have `mapping_protocol.json` OR `mapping_protocol_ref.json`

**Minimum viable cert family:**
```
qa_[name]/
├── mapping_protocol_ref.json
├── qa_[name]_validate.py      ← --self-test returns {"ok": true}
└── fixtures/
    ├── [abbrev]_pass_*.json   ← at least 1
    └── [abbrev]_fail_*.json   ← at least 1
```

**Add to meta-validator FAMILY_SWEEPS:**
```python
(N, "label", _validate_fn, "pass_description", "N_slug", "relative/path", True)
```

**Add to docs/families/README.md:** one table row
**Add to docs/families/N_slug.md:** full human tract

**Substrate rules (critical):**
- `d*d` not `d**2`
- `mapping_protocol_ref.json` fields: `protocol_version`, `ref_path`, `ref_sha256` (NOT `schema_version`)
- Manifest hash placeholder: 64-char hex zeros (NOT the string "placeholder")

---

## Experiment Standards

**Before running:** capture hypothesis + pre-declared success criteria in Open Brain
**After running:** capture result immediately with honest verdict
**If significant:** write [122] `QA_EMPIRICAL_OBSERVATION_CERT` within same session
**Registry:** add entry to `experiments/registry.json`

---

## Current Project Health (2026-03-25)

| Metric | Status |
|---|---|
| Meta-validator | 128/128 PASS |
| Cert families | [18]–[122] active |
| Papers | pythagorean-families (submission-ready), unified-curvature (arXiv), modular-dynamics (seed) |
| Open Brain | Active; grand synthesis note captured 2026-03-25 |
| Root directory | 579 items — `archive/` migration pending |
| `qa_core/` module | Missing — QA_Engine duplicated in ~50 files |
