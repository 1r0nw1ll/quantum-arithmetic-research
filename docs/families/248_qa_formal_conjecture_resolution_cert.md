# [248] QA Formal Conjecture Resolution Cert

**Family:** `qa_formal_conjecture_resolution_cert`
**Version:** `QA_FORMAL_CONJECTURE_RESOLUTION_CERT.v1`
**Status:** PASS (2 PASS + 1 FAIL fixtures, self-test ok)
**Primary source:** Ju, Gao, Jiang, Wu, Sun, Chen, Wang, Wang, Wang, He, Wu, Xiao, Liu, Dai, Dong (2026). *Automated Conjecture Resolution with Formal Verification.* arXiv:2604.03789. Submitted 2026-04-04. https://arxiv.org/abs/2604.03789
**Theory note:** `docs/theory/QA_AUTOMATED_CONJECTURE_RESOLUTION.md`

## Purpose

A QA-native record format for conjecture-resolution attempts. The paper introduces a two-stage pipeline (Rethlas informal candidate generator + Archon Lean 4 formalizer + Matlas/LeanSearch retrieval) that automatically resolves an open commutative-algebra problem with a machine-checkable proof. This cert family records analogous resolution events inside QA, with one important QA extension: typed failure modes.

The paper treats proof failure as binary (proof found or not). QA's failure algebra requires distinguishing four cases:
- `proved` — closure achieved
- `formal_gap` — backend cannot close, no QA invariant violated
- `qa_obstruction` — QA invariant (orbit, parity, modulus, Theorem NT) prevents the conjecture
- `generator_insufficient` — declared generator set does not span the statement
- `inconclusive` — budget exhausted without classification

Typed obstructions convert failed attempts into usable information rather than noise.

## Schema

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | yes | must equal `"QA_FORMAL_CONJECTURE_RESOLUTION_CERT.v1"` |
| `family` | string | yes | must equal `"qa_formal_conjecture_resolution_cert"` |
| `case` | string | yes | short label for the resolution event |
| `conjecture_id` | string | yes | stable identifier (e.g. `QA-2026-001`) |
| `source_claim` | string | yes | conjecture in natural language |
| `qa_translation` | string | yes | restatement in QA operators (orbit, diagonal, generator, T-operator) |
| `candidate_kind` | enum | yes | `orbit_invariant` \| `diagonal_class` \| `generator_chain` \| `bridge` \| `external` |
| `formal_backend` | enum | yes | `qa_symbolic` \| `qa_exhaustive` \| `lean4_stub` \| `python_proof` |
| `proof_status` | enum | yes | `proved` \| `formal_gap` \| `qa_obstruction` \| `generator_insufficient` \| `inconclusive` |
| `failure_mode` | string \| null | conditional | required non-null iff `proof_status != proved` |
| `generator_set` | array of strings | yes | non-empty; operators/axioms used |
| `nt_compliance` | boolean | yes | Theorem NT (observer-projection firewall) declared clean |
| `witness_path` | string | yes | path to fixture or symbolic witness file, or `"symbolic_inline"` |
| `verdict` | enum | yes | `CONSISTENT` \| `PARTIAL` \| `CONTRADICTS` \| `INCONCLUSIVE` (matches [122] vocabulary) |
| `open_questions` | array of strings | conditional | required non-empty iff `formal_backend == "lean4_stub"` |
| `source_attribution` | string | recommended | author + date + paper or theory-doc reference |

## Validator checks

| Gate | Description |
|---|---|
| `FCR_1` | Schema validity: required fields present, enums in allowed sets |
| `FCR_2` | `generator_set` is a non-empty list of non-empty strings |
| `FCR_3` | `failure_mode` non-null iff `proof_status != proved`; null iff `proved` |
| `FCR_4` | `nt_compliance` is boolean; if false, `failure_mode` must name the NT violation |
| `FCR_5` | `verdict` in `{CONSISTENT, PARTIAL, CONTRADICTS, INCONCLUSIVE}` |
| `FCR_6` | `witness_path` exists on disk or equals `"symbolic_inline"` |
| `FCR_7` | If `formal_backend == "lean4_stub"`, `open_questions` must be a non-empty array |

## Fixtures

| File | Kind | Notes |
|---|---|---|
| `fcr_pass_proved.json` | PASS | `b=e` diagonal `d=2b` identity in S_9, exhaustive, verdict `CONSISTENT` |
| `fcr_pass_formal_gap.json` | PASS | Ju et al. arXiv:2604.03789 commutative-algebra result as `lean4_stub`, verdict `PARTIAL`, open_questions present |
| `fcr_fail_missing_failure_label.json` | FAIL | `proof_status=formal_gap` with `failure_mode=null` — must be rejected by FCR_3 |

## Family relationships

- **Bridges to [122] `QA_EMPIRICAL_OBSERVATION_CERT`** — uses the same verdict vocabulary (`CONSISTENT` / `PARTIAL` / `CONTRADICTS` / `INCONCLUSIVE`).
- **Complements [223] `QA_EXPERIMENT_PROTOCOL.v1`** — [223] gates *experiment design* before code; [248] records *conjecture resolution* after attempt.
- **Complements [224] `QA_BENCHMARK_PROTOCOL.v1`** — [224] gates *benchmark comparison*; [248] gates *theorem closure*.
- **Architectural placeholder for future Lean 4 bridge** — `formal_backend: lean4_stub` is allowed and tested; an actual `formal/lean/` tree is not yet wired in.

## Open questions for Will

1. **Lean bridge scope.** Build an actual Lean 4 backend (compilation in meta-validator), or keep `lean4_stub` indefinitely as architectural placeholder?
2. **Protocol vs. witness family.** Should [248] be referenced by other certs (protocol pattern, like [223]/[224]), or stand on its own per resolution event (witness pattern, like [244])?
3. **Retroactive application.** Should existing cert families generate retroactive [248] records, or does [248] start fresh from 2026-04-15 forward?

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_formal_conjecture_resolution_cert_v1
python qa_formal_conjecture_resolution_cert_validate.py --self-test
```

Returns canonical JSON with `ok: true`, `pass_count: 2`, `fail_count: 1`.

## What breaks this cert

- Adding a `proof_status: formal_gap` (or any non-`proved` status) record without a `failure_mode` string → FCR_3 rejects.
- Setting `nt_compliance: false` without naming the NT violation in `failure_mode` → FCR_4 rejects.
- Declaring `formal_backend: lean4_stub` without populating `open_questions` → FCR_7 rejects.
- Verdict outside the four-element vocabulary → FCR_5 rejects.

## References

- Ju et al. (2026). arXiv:2604.03789.
- `docs/theory/QA_AUTOMATED_CONJECTURE_RESOLUTION.md` — full mapping rationale and corrections to the initial ChatGPT plan.
- Cert [122] `QA_EMPIRICAL_OBSERVATION_CERT` — verdict vocabulary precedent.
- Certs [223]/[224] — design-contract pattern.
- `docs/specs/QA_CODING_GUARDRAIL_ARCHITECTURE.md` — three-layer guardrail; this cert was authored by Claude under Layer 3 quarantine and reviewed at cert-submission.
