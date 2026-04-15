# QA Coding Guardrail Architecture

**Status:** active as of 2026-04-14
**Authors:** Will Dale + Claude + Codex
**Supersedes:** `memory/feedback_map_best_to_qa.md` (rule preserved) + ad-hoc per-write blocks

## Motivation

Claude's `.py`/`.sh` authorship was revoked 2026-04-14 after repeated violations of the "map best-performing to QA" rule (OB `ec46f601-bcc4-49df-bbd7-0485edc40471`: MNIST-Rot 22% vs Cohen-Welling 97.9% after inventing featurizer from scratch). A blanket block created a new problem: Codex became the only coder, which strained its rate limits. The three-layer system below restores Claude's coding under machine-checkable supervision that does not require Codex review per write.

## Architecture — three layers

### Layer 1 — deterministic lint on write (fast, per-file)

`tools/qa_axiom_linter.py` runs on every `.py` edit and commit. Catches:

- **Axiom violations** (A1/A2/T2/S1/S2/T1): `b**2`, float state in QA logic, float × modulus casts, zero-indexed states, continuous time variables, stochastic generators as QA data (T2-D-1..6).
- **Protocol runtime contract** (EXP-1, BENCH-1, EXP-ABLATION, BENCH-ABLATION, EXP-RUNTIME, BENCH-RUNTIME): any script with an explicit `EXPERIMENT_PROTOCOL_REF` or `BENCHMARK_PROTOCOL_REF` must define and call the protocol's declared `ablation.callable`, and `__main__` scripts must call `qa_reproducibility.log_run(...)` via a real import binding. AST-based, not regex-matched — a local `def log_run(): pass` does not satisfy the gate.

### Layer 2 — design-contract validation (structural, at cert-submission)

Cert families [223] `QA_EXPERIMENT_PROTOCOL.v1` and [224] `QA_BENCHMARK_PROTOCOL.v1` enforce study design *before* code review becomes relevant. Each is a 9-gate JSON-schema validator:

- **Source mapping cross-reference**: `source_mapping.primary_source` string must literally appear in the file pointed to by `source_mapping.theory_doc`. Kills the "anchor: 'Some 2020 paper'" garbage-string loophole.
- **Null design 4-tuple**: `generating_process`, `held_fixed`, `permuted`, `independence_argument`. Prose "independence argument" alone is not sufficient — the structural held-fixed/permuted split forces naming what's independent.
- **SOTA baseline**: numeric threshold OR explicit `null_result_acceptable=true` with reason. Mirrors the MNIST-Rot failure class.
- **Ablation contract**: callable name + destroyed QA structure + expected direction.
- **Reproducibility manifest**: seed, `data_sha256` (hex or `pending`/`synthetic_only` sentinel), package versions, results ledger path.
- Plus domain gates (pre-registration, decision rules, observer projection, calibration provenance, framework inheritance, metrics, real-data status).

### Layer 3 — quarantine and batched Codex review (deep, async)

`llm_qa_wrapper/cert_gate_hook.py` lets Claude `.py` writes land on disk, but records each write as a `QA_CLAUDE_PYTHON_QUARANTINE.v1` packet under `llm_qa_wrapper/quarantine/pending/`. Each packet carries a pre-write snapshot when available. The hook also appends `enforcement_markers: ["CLAUDE_PYTHON_WRITE_QUARANTINED"]` to the ledger record so weekly audit tooling can find Python-write events without parsing packet JSON. `git commit` fails with `CODEX_REVIEW_PENDING` while any packet is pending, even if the collab marker is present.

Codex runs `tools/qa_codex_quarantine_review.py {list,approve,reject}` at cert-submission time. One review session per cert family, not one per Claude write. This amortizes the review load and catches the class of failure schemas cannot prove away: ML-training-data bleed (softmax in T-operator position, gradient descent where orbit classification belongs), subtle continuous-time loops, suspicious "novel" constructions that don't appear in any cited primary source.

Approved packets move to `quarantine/approved/`; rejected ones to `quarantine/rejected/`. On reject, the review tool restores the original file bytes from the pre-write snapshot, or removes the file if the quarantined write created it. If no snapshot is available, the reject decision is marked advisory and requires manual cleanup. Every decision appends a record to `llm_qa_wrapper/quarantine/codex_reviews.jsonl` with `review_sha256` for auditability.

This is a soft development gate, not a sandbox rollback system. Rejecting a packet restores the Python file, but it does not undo downstream artifacts produced before review, such as result ledgers, plots, cache files, or generated JSON. The commit gate prevents publishing those artifacts unnoticed; local cleanup remains part of the review decision when rejected code was executed.

## What each layer catches

| Failure class | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| `b**2` instead of `b*b` | ✓ | — | — |
| float state surviving observer boundary | ✓ | — | — |
| Random graph generator as QA data | ✓ (T2-D) | — | ✓ |
| Invented SOTA featurizer (MNIST-Rot class) | — | ✓ (SOTA threshold + source cross-ref) | ✓ |
| Circular null design (Bateson×ATT&CK class) | — | ✓ (null 4-tuple) | ✓ |
| Ablation declared but never called | ✓ (EXP-ABLATION) | — | — |
| `log_run` called from fake local stub | ✓ (import-binding check) | — | — |
| Softmax/grad-descent smuggled into QA dynamics | — | — | ✓ |
| Overclaim in commit message / theory doc | — | — | ✓ (via results ledger cross-check) |

## Exemption mechanism

Layers 1 and 3 accept HTML-comment exemption markers in the first 500 bytes of the file:

```
<!-- PRIMARY-SOURCE-EXEMPT: reason="..." approver="Will" ts="ISO8601" -->
<!-- CODING-GUARDRAIL-EXEMPT: reason="..." approver="Will" ts="ISO8601" -->
```

Every exemption is logged. Weekly review tallies exemption frequency — if usage trends up, the gate is becoming a rubber stamp and needs tightening.

## Deferred / open items

1. **Reachability proof for `run_ablation()` calls.** Current linter accepts unreachable calls like `if False: run_ablation()`. Low priority; not observed in practice.
2. **Sibling-only protocol migration.** Explicit `EXPERIMENT_PROTOCOL_REF` / `BENCHMARK_PROTOCOL_REF` scripts get the strict runtime obligations; sibling-only `experiment_protocol.json` scripts are grandfathered. New sibling-only scripts created after 2026-04-14 should either be converted to explicit refs or monitored in the weekly review.
3. **Pending-`data_sha256` age scan.** Any results-ledger entry where `data_sha256 == "pending"` and age > 7 days should be flagged by the weekly review pipeline.
4. **Parallel-Claude circumvention.** Spawning a Claude instance outside `signal_experiments/` to bypass the hooks is a self-circumvention vector. See `llm_qa_wrapper/tests/test_self_circumvention.py` for the defensive stance. The correct parallelism pattern is a worktree inside the repo with file locks via the qa-collab bus, per `CLAUDE.md` §"Multi-Session Parallelism Protocol".
5. **Generated-artifact cleanup after reject.** Layer 3 restores the Python source file on reject, but cannot infer every downstream artifact created by executing rejected code. Codex review notes should name any generated outputs that need manual removal or separate validation.

## Rollout state

- Layer 1: **live**, regression-tested (`tools/tests/test_qa_axiom_linter_protocols.py`: 9/9 pass).
- Layer 2: **live**, [223]/[224] in meta-validator registry, 9 gates each.
- Layer 3: **live**, `llm_qa_wrapper/tests/test_cert_gate_hook.py`: 21/21 pass including write-lands quarantine, grep-able ledger markers, commit blocking, reject rollback, and read-only/VCS Python-path noise regression tests.

The `CLAUDE_PYTHON_WRITE_FORBIDDEN` hard block is intentionally removed from the active policy. Static audit now treats reintroduction of that marker as a regression; the active enforcement markers are `CLAUDE_PYTHON_WRITE_QUARANTINED` for Python writes and `CODEX_REVIEW_PENDING` for commit blocking.

## References

- `docs/specs/PRIMARY_SOURCE_GATE_HOOK_SPEC.md` — related primary-source enforcement for theory docs.
- `memory/feedback_map_best_to_qa.md` — the rule this architecture operationalizes.
- `EXPERIMENT_AXIOMS_BLOCK.md` — Parts A/B/C axioms that [223]/[224] harvest.
- `llm_qa_wrapper/ARCHITECTURE.md` — ledger-chain and hook-invocation substrate.
- OB `ec46f601-bcc4-49df-bbd7-0485edc40471` (2026-04-14) — triggering incident and privilege-revocation directive.
- OB session trail 2026-04-14: privilege revocation → primary-source gate implementation → design-contract hardening → quarantine/review loop.
