# QA Project — Full System Specification

**Version:** 1.0 | **Date:** 2026-03-25 | **Author:** Will Dale

This document is the authoritative specification for the QA research project. It defines
what the system IS, how its components relate, what each role is responsible for, and
what standards every artifact must meet. When in doubt, this document wins.

---

## 0. Mission and Non-Negotiables

**Mission:** Build and validate a unified theory of modular arithmetic dynamics (QA)
with applications across signal processing, finance, geometry, and neural networks.
The core object — the QA state transition algebra — must generate all three artifact
types: certs (invariants), experiments (projections), and papers (claims).

**Non-negotiables:**
1. Every empirical claim must have a pre-declared hypothesis and success criteria
2. Negative results are first-class — certify failures as honestly as successes
3. The meta-validator is the single source of truth for cert health (currently 128/128)
4. No cert ships without PASS + FAIL fixtures and a human tract doc
5. Open Brain captures happen at the moment of result, never batched at session end

---

## 1. System Architecture

The project has five layers. Information flows downward; feedback flows upward.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: PAPERS                                            │
│  Claims backed by certs and experiments                     │
│  papers/in-progress/ → papers/ready-for-submission/        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4: CERT ECOSYSTEM                                    │
│  Machine-checkable invariants, validated by meta-validator  │
│  qa_alphageometry_ptolemy/ + qa_*_cert_v1/ at root         │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: EMPIRICAL BRIDGE [122]                            │
│  Connects observations to cert claims                       │
│  verdict: CONSISTENT / CONTRADICTS / PARTIAL / INCONCLUSIVE│
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: EXPERIMENT RESULTS                                │
│  Open Brain captures, script outputs, JSON result files    │
│  Captured at moment of result; tagged by domain            │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1: QA CORE + EXPERIMENTS                             │
│  qa_core/ shared module + domain experiment scripts        │
│  experiments/signal/, experiments/finance/, etc.           │
└─────────────────────────────────────────────────────────────┘
```

**Information flow rule:** An experiment produces a result (Layer 1→2).
The result is captured in Open Brain (Layer 2). If significant, it becomes a [122]
cert (Layer 2→3). The cert can be cited in a paper claim (Layer 4→5).
A paper claim without a cert reference is informal; a cert without an Open Brain
observation is ungrounded. Both directions matter.

---

## 2. Role Specifications

Every task in this project belongs to exactly one role. Assigning the wrong role
wastes token budget and produces lower-quality output.

### Claude (Orchestrator / Architect)
**Responsible for:**
- Architecture decisions and system specifications (this document)
- Reading and analyzing experiment results
- Specifying the next task for Codex or Gemini
- Meta-validator maintenance and cert family architecture
- Catching logical errors in reasoning, cert design, or paper claims
- Bridging layers: turning an Open Brain observation into a cert spec

**Does NOT:**
- Write experiment scripts (Codex)
- Write README or paper prose (Gemini)
- Run long numerical experiments (delegate via spec)

**Token discipline:** Minimum output consistent with correct reasoning.
No padding. No restatements.

### Codex (Code Writer / File Editor)
**Responsible for:**
- All Python script writing and editing
- All cert validator implementation
- All fixture generation
- All schema JSON authoring
- Running and debugging scripts

**Standard:** Every script Codex writes must be runnable standalone.
Every validator must pass `--self-test` before committing.

### Gemini (Documentation / Prose)
**Responsible for:**
- README updates and paper prose drafts
- Human tract docs for cert families (docs/families/*.md)
- Markdown synthesis documents
- Session summaries if needed

**Standard:** Human tract docs must cover: purpose, schema fields,
validator checks, fixtures table, relationships to other families.

### Open Brain (Observation Layer)
**Responsible for:**
- Capturing every experiment result at moment of completion
- Storing ideas, hypotheses, synthesis notes
- Providing search across prior results before re-deriving

**Standard:** Every capture must have type, tags, domain.
Search before deriving — if a result exists in Open Brain it should
be retrieved, not recomputed.

### Git (Version Control)
**Responsible for:**
- Stable checkpoints after each cert family ships
- Never skip hooks or --no-verify
- Commit message describes the structural change, not just the files

---

## 3. Directory Structure (Canonical)

Current state is organic (579 items at root). This spec defines the target.
Migration is incremental — archive stale files, don't break live paths.

```
signal_experiments/
│
├── CLAUDE.md                    ← Claude Code instructions (project)
├── CONSTITUTION.md              ← project mission + non-negotiables (this doc condensed)
├── README.md                    ← public-facing entry point
├── AGENTS.md                    ← agent role quick-reference
├── requirements.txt
├── Makefile                     ← common commands: make validate, make test, make archive
│
├── qa_core/                     ← SHARED PYTHON MODULE (target; currently missing)
│   ├── __init__.py
│   ├── engine.py                ← QA_Engine / QASystem (single implementation)
│   ├── metrics.py               ← HI, E8 alignment, kappa, rho(O)
│   ├── orbit.py                 ← orbit classification, BFS reachability
│   └── logger.py                ← structured result logger → Open Brain format
│
├── experiments/                 ← ALL experiment scripts (target; currently at root)
│   ├── signal/
│   ├── finance/                 ← currently at ~/Desktop/qa_finance/ (private)
│   ├── eeg/
│   ├── geometry/
│   ├── forensics/               ← future: audio deepfake, financial fraud
│   └── registry.json            ← experiment registry (see §5)
│
├── certs/                       ← target home for cert families
│   ├── qa_mapping_protocol/     ← currently at root (migration needed)
│   ├── qa_mapping_protocol_ref/
│   ├── qa_alphageometry_ptolemy/ ← meta-validator + inline families
│   └── qa_*_cert_v1/            ← standalone families (currently at root)
│
├── papers/
│   ├── in-progress/
│   │   ├── pythagorean-families/
│   │   ├── unified-curvature/
│   │   └── modular-dynamics/
│   └── ready-for-submission/
│
├── data/                        ← datasets (gitignored large files)
│   ├── eeg/
│   ├── hyperspectral/
│   └── financial/
│
├── docs/
│   ├── families/                ← human tract docs for cert families (active)
│   ├── specs/                   ← THIS FILE and other spec docs
│   ├── architecture/            ← diagrams, flow charts
│   └── index.md
│
├── results/                     ← experiment outputs (gitignored large files)
│   └── [domain]/[script]/[timestamp]/
│
└── archive/                     ← stale artifacts (not deleted, just moved)
    ├── session_logs/            ← all SESSION_*.md, CLOSEOUT_*.md, HANDOFF_*.md
    ├── phase_artifacts/         ← phase1_workspace/, phase2_workspace/, phase2_data/
    ├── player4/                 ← all PLAYER4_*.txt, player4_transfer_package/
    └── zips/                    ← workspaces*.zip, grokking_qa_overlay*.zip, etc.
```

**Migration priority:**
1. `archive/` pass — move stale session logs, handoffs, zips (no code changes)
2. `qa_core/` creation — extract shared QA_Engine into importable module
3. `experiments/` reorganization — move scripts from root
4. `certs/` consolidation — long-term (breaks meta-validator paths; needs careful migration)

---

## 4. Cert Family Specification

Every cert family must satisfy all of the following before the meta-validator entry is added.

### Required files
```
qa_[name]_cert_v1/
├── mapping_protocol.json OR mapping_protocol_ref.json   ← Gate 0
├── qa_[name]_validate.py                                ← validator with --self-test
├── fixtures/
│   ├── [name]_pass_*.json                               ← at least 1 PASS
│   └── [name]_fail_*.json                               ← at least 1 FAIL
└── schemas/                                             ← optional JSON schema
```

### Required validator properties
- `--self-test` exits 0 and returns `{"ok": true}`
- Every PASS fixture is verified OK=true by validator
- Every FAIL fixture is verified with specific fail_type(s) detected
- Fail types are documented in KNOWN_FAIL_TYPES frozenset

### Required meta-validator entry
```python
(N, "QA [Name] family",
 _validate_[name]_family,
 "brief description: schema + validator + fixtures summary",
 "N_[slug]",
 "[path_relative_to_base_dir]", True),
```

### Required human tract
`docs/families/N_[slug].md` containing:
- Purpose (1 paragraph)
- Schema fields table
- Validator checks table
- Fixtures table (file, verdict, what it tests)
- Relationships to other families

### Cert versioning
- First version: `v1`. Breaking schema changes → `v2` (new family, old family deprecated, not deleted)
- `schema_version` field must encode family version: `QA_[NAME]_CERT.v1`

---

## 5. Experiment Specification

Every significant experiment must satisfy the following.

### Pre-registration (before running)
Define in an Open Brain capture or script header:
- **Hypothesis**: what QA predicts
- **Success criteria**: pre-declared, quantitative (not post-hoc)
- **Domain**: signal / finance / eeg / geometry / forensics
- **Script**: which script, which parameters

### Result capture (at moment of completion)
Open Brain capture with:
- Type: `observation`
- Tags: `[domain, experiment-name, result-type]`
- Key numbers explicitly stated
- Honest verdict against pre-declared criteria (PASS / FAIL / PARTIAL)

### Certification (within session if significant)
If result is publication-grade or contradicts/confirms a cert claim:
- Write a [122] QA_EMPIRICAL_OBSERVATION_CERT
- Link to parent cert and specific claim
- Document verdict with evidence
- Add to cert suite

### Experiment registry format
`experiments/registry.json` (target):
```json
{
  "experiments": [
    {
      "id": "audio_orbit_2026-03-25",
      "script": "experiments/signal/qa_audio_orbit_test.py",
      "domain": "signal",
      "hypothesis": "dynamical signals exhibit orbit-follow rates above chance after equalization",
      "success_criteria": "orbit_follow_rate > 0.111 for ≥5/6 dynamical signal types; white_noise ≈ 0.111",
      "result": "PASS",
      "open_brain_id": "44e4dded-...",
      "cert_id": "qa.cert.empirical.audio_orbit_consistent.v1",
      "captured_utc": "2026-03-25T13:55:59Z"
    }
  ]
}
```

---

## 6. Paper Specification

Papers move through defined stages. No stage-skipping.

```
idea                → Open Brain capture (type: idea)
    ↓
outline             → papers/in-progress/[name]/outline.md
    ↓
draft               → papers/in-progress/[name]/paper.tex
    ↓
cert-backed         → every major claim has a cert ID cited
    ↓
verify script PASS  → verify_[name].py passes (stdlib only, <1s)
    ↓
review-ready        → docs/families/README.md updated, no TODOs
    ↓
submitted           → papers/ready-for-submission/[name]/
```

**Claim certification requirement:** Any quantitative claim that is central to
a paper's contribution must be backed by either:
- A cert family (mathematical invariant, machine-checked), OR
- A [122] empirical cert (experimental result, pre-declared criteria)

Informal claims go in the Discussion section, not the Results section.

---

## 7. Information Flow (Cybernetic Loops)

The project has three feedback loops. Each must close or the system degrades.

### Loop A: Theory → Cert → Theory
```
Mathematical claim
    → cert validator (machine check)
        → PASS: claim is operative
        → FAIL: claim is wrong or needs refinement
            → update theory
```
**Health indicator:** meta-validator 100% green

### Loop B: Experiment → Observation → Cert → Paper
```
Experiment runs
    → result captured in Open Brain (immediate)
        → [122] cert written (same session if significant)
            → cert cited in paper claim
                → paper claim is evidence-backed
```
**Health indicator:** every significant Open Brain observation has a cert pointer

### Loop C: Paper → Peer Review → Cert Update
```
Reviewer challenges claim X
    → identify which cert backs X
        → PASS: cert is correct, respond with cert ID
        → reviewer finds cert flaw: update cert and validator
            → meta-validator must still pass after update
```
**Health indicator:** every paper claim traceable to a cert or [122] observation

---

## 8. Naming Conventions

| Artifact type | Convention | Example |
|---|---|---|
| Cert family dir | `qa_[name]_cert_v[N]` or `qa_[name]_v[N]` | `qa_empirical_observation_cert` |
| Cert validator | `qa_[name]_validate.py` | `qa_empirical_observation_cert_validate.py` |
| Fixture (PASS) | `[abbrev]_pass_[description].json` | `eoc_pass_audio_orbit_consistent.json` |
| Fixture (FAIL) | `[abbrev]_fail_[description].json` | `eoc_fail_empty_evidence.json` |
| Experiment script | `[domain]_[description]_[v].py` | `audio_orbit_coherence_v1.py` |
| Open Brain capture | auto-ID + descriptive title | — |
| Paper directory | `[topic]/` | `pythagorean-families/` |
| Human tract doc | `[N]_[slug].md` | `122_qa_empirical_observation_cert.md` |

---

## 9. Agent Invocation Protocol

When a task is ready for delegation:

**To Codex:**
```
TASK: [verb] [object]
CONTEXT: [what exists, what's needed]
INPUT: [files to read]
OUTPUT: [files to create/edit]
CONSTRAINT: [must pass --self-test / must not change X]
```

**To Gemini:**
```
TASK: write [doc type] for [subject]
AUDIENCE: [reviewer / developer / internal]
SOURCE: [cert files / experiment results to draw from]
STRUCTURE: [required sections]
OUTPUT: [target file path]
```

**To Open Brain (capture):**
Immediately on result. Type + tags always required. Pre-declared criteria verdict always stated.

---

## 10. Current Gap Analysis (2026-03-25)

| Gap | Severity | Fix |
|---|---|---|
| 579 items at root, 123 stale .md files | HIGH | `archive/` migration pass |
| No `qa_core/` shared module — QA_Engine reimplemented ~50× | HIGH | Extract to `qa_core/engine.py` |
| Experiment scripts unorganized, no registry | MEDIUM | `experiments/` dir + `registry.json` |
| 7+ workspace .zip archives at root | LOW | Move to `archive/zips/` |
| Cert families at root (not in `certs/`) | LOW | Long-term migration; meta-validator paths must update together |
| MEMORY.md at 233 lines (limit 200) | LOW | Pruning pass |
| No `experiments/finance/` — private repo separation | MEDIUM | Document separation decision; ref from spec |

---

## 11. Immediate Next Actions (in priority order)

1. **`archive/` pass** — move session logs, handoffs, zips; no code changes; zero risk
2. **`qa_core/engine.py`** — single QA_Engine implementation; update one experiment as proof of concept
3. **`experiments/registry.json`** — register the 3-4 experiments with clean results
4. **Batch [122] certs** — certify 10-15 Open Brain observations from finance + audio + EEG
5. **Forensic coherence spine** — [123]-[125] building on grand synthesis note
6. **MEMORY.md prune** — consolidate state snapshots to topic files

---

*This spec is a living document. Update it when architectural decisions change.
Do not let it go stale — a stale spec is worse than no spec.*
