# QARM v02 — TLC Model-Checking Proof Ledger

**Date:** 2026-04-20
**Session:** `audit-tla-plus-L1` / claude-main-1556
**Authority:** `docs/specs/QA_TLA_PLUS.md` (three-layer constitutional architecture), `docs/specs/QA_TLA_PLUS_AUDIT.md` §5 Lane 1.
**Primary references:** (Lamport, 1994) ACM TOPLAS 16(3) DOI:10.1145/177492.177726 for the Temporal Logic of Actions; (Lamport, 2002) *Specifying Systems* ISBN 978-0-321-14306-8. Template mirrors `llm_qa_wrapper/spec/TLC_PROOF_LEDGER.md` (which cites arXiv:2603.18829 Agent Control Protocol and arXiv:2603.23801 AgentRFC composition safety).

This file records the first TLC model-checking runs ever executed against the `QARM_v02_*.tla` specs and `QACertificateSpine.tla`, which had been authored 2025-12-30 / 2026-01-21 but never exercised. Each run is reproducible from the spec and `.cfg` files in this directory.

---

## Invocation pattern

```
cd qa_alphageometry_ptolemy
java -XX:+UseParallelGC -jar tla2tools.jar \
    -workers 4 -terse -config <spec>.cfg <spec>.tla
```

- `tla2tools.jar` = TLC 2.20 (distributed with the TLA+ Toolbox per Lamport, 2002).
- Java version at run time: OpenJDK 21.0.9-ea (Debian 1).

---

## Run 1 — `QARM_v02_Failures` (positive, main spec)

**Purpose:** prove that the five declared safety invariants hold over the reachable state space under the full generator set {σ, μ, λ_k for k ∈ {2,3}}, including absorbing-stutter failure states.

**Config** (`QARM_v02_Failures.cfg`, authored 2025-12-30; unchanged):
- `CAP = 20` (bound on all of b, e, d, a)
- `KSet = {2, 3}` (λ scaling factors)
- Model style: `INIT Init` / `NEXT Next` (does not use `SPECIFICATION Spec`, i.e., no stuttering closure — finite-state reachability only)

**Invariants checked:**
- `Inv_TupleClosed` — `d = b + e ∧ a = d + e`
- `Inv_InBounds` — all of `b, e, d, a ∈ 0..CAP`
- `Inv_QDef` — `qtag = 24 * Phi9(a) + Phi24(a)` (duo-modular packing)
- `Inv_FailDomain` — `fail ∈ {OK, OUT_OF_BOUNDS, FIXED_Q_VIOLATION, ILLEGAL}`
- `Inv_MoveDomain` — `lastMove ∈ {NONE, σ, μ, λ}`

**Result:** `Model checking completed. No error has been found.`

- Initial states generated: **121 distinct**
- Total states generated: 1012
- Distinct states found: **504**
- States left on queue: 0 (complete BFS exhausted)
- Search depth: 2
- Avg outdegree: 1 (max 4, 95th percentile 3)
- Fingerprint collision probability: 1.4×10⁻¹⁴
- Wall time: **1 second**
- Workers: 4

**Interpretation:** at `CAP = 20, KSet = {2,3}`, the QARM v02 generator algebra preserves all five structural invariants across the entire reachable state graph. The system reaches 504 distinct states starting from 121 initial states. Search depth 2 reflects the absorbing-stutter design: `fail ≠ "OK"` states are terminal, so any successful or failing move reaches a fixed point in two steps from an initial state.

---

## Run 2 — `QARM_v02_Failures_negative` (non-vacuity, `Inv_TupleClosed`)

**Purpose:** prove that `Inv_TupleClosed` is not vacuous — that it actively detects violations of the canonical tuple-closure rule `d = b+e, a = d+e`. If this run reported "no error," the positive result in Run 1 would be meaningless.

**Spec** (`QARM_v02_Failures_negative.tla`, authored 2026-04-20): a minimal module whose sole action is `BrokenTupleClosure`, which writes `d' = 5` while `b' + e' = 4`. This is the exact violation class that `Inv_TupleClosed` is designed to detect.

**Config** (`QARM_v02_Failures_negative.cfg`, authored 2026-04-20): `SPECIFICATION Spec` with no constants.

**Expected result:** `Inv_TupleClosed` violated after a 2-state counterexample.

**Actual result:** `Error: Invariant Inv_TupleClosed is violated.`

Counterexample trace (verbatim TLC output):

```
State 1: <Initial predicate>
  lastMove = "NONE"
  a = 3, b = 1, d = 2, e = 1
  qtag = 75
  fail = "OK"

State 2: <BrokenTupleClosure line 63–69 of QARM_v02_Failures_negative>
  lastMove = "σ"
  a = 7, b = 2, d = 5, e = 2
  qtag = 75
  fail = "OK"
```

- States generated: 2
- Distinct states: 2
- Search depth: 2
- Wall time: 1 second

**Interpretation:** `Inv_TupleClosed` is non-vacuous — it detects direct violations of tuple-closure with a 2-state counterexample. Therefore Run 1's "no error" result represents actual exercise of the invariant over the reachable state space, not a silent pass.

**Future work:** add per-invariant non-vacuity specs for the other four (`Inv_InBounds`, `Inv_QDef`, `Inv_FailDomain`, `Inv_MoveDomain`), following the pattern in `llm_qa_wrapper/spec/cert_gate_negative_{chain,bind,composition}.tla`.

---

## Run 3 — `QARM_v02_NoMu` (generator-set differential)

**Purpose:** model-check QARM with μ removed from `Next`, to set up the generator-set-differential comparison posed verbatim by the 2025 ChatGPT exchange (see `docs/specs/QA_TLA_PLUS.md` §4):

> "verify that the number of states with `fail = "OUT_OF_BOUNDS"` is invariant across different generator sets"

**Config** (`QARM_v02_NoMu.cfg`, authored 2025-12-30; unchanged): same `CAP = 20, KSet = {2, 3}` as Run 1, so the two runs are directly comparable.

**Result:** `Model checking completed. No error has been found.`

- Initial states generated: **121 distinct** (same as Run 1 — initial state set does not depend on generator choice)
- Total states generated: 748
- Distinct states found: **383**
- Search depth: 2
- Wall time: 1 second

**Interpretation:** removing the μ generator reduces the reachable state count from **504 to 383**, a drop of **121 states** — coincidentally the cardinality of the initial state set. The five structural invariants still hold (0 errors). See Run 4 for the per-action failure-tally comparison that directly answers the 2025 question.

---

## Run 4 — Generator-set differential: `QARM_v02_Stats` vs `QARM_v02_NoMu_Stats`

**Purpose:** answer the 2025 ChatGPT question verbatim — is the OUT_OF_BOUNDS failure count invariant across different generator sets?

**Method:** both specs carry `PrintT` instrumentation on every failure action (`SigmaFail_OOB`, `MuFail_OOB`, `LambdaFail_OOB`, `SigmaFail_FQ`, `MuFail_FQ`, `LambdaFail_FQ`). The invocation pipes TLC's stdout to a log file and counts per-action occurrences via `grep -c`. Both runs use `CAP = 20, KSet = {2, 3}`.

### Bug caught during this run (2025-12-30 spec)

`QARM_v02_Stats.tla` as authored 2025-12-30 failed to parse:

```
Couldn't resolve infix operator symbol `\o'.
line 151, col 35 to line 151, col 36 of module QARM_v02_Stats
```

The `\o` sequence-concatenation operator used inside `PrintT("FAIL_OOB_LAMBDA: " \o ToString(…))` requires the `Sequences` standard module, but the module's `EXTENDS` clause only included `Naturals, Integers, TLC, TLCExt`. **This bug sat undiscovered for 111 days because the spec had never been run.** Fixed in-place by adding `Sequences` to the `EXTENDS` list.

### Results

**With full generator set {σ, μ, λ}** (`QARM_v02_Stats`):

| Action | Print count | Notes |
|---|---|---|
| FAIL_OOB_SIGMA | 21 | σ-boundary transitions |
| FAIL_OOB_MU | 40 | μ-boundary transitions |
| FAIL_OOB_LAMBDA | 197 | λ-boundary transitions (k ∈ {2,3} combined) |
| FAIL_FQ_SIGMA | 108 | σ-transitions that would change qtag |
| FAIL_FQ_MU | 74 | μ-transitions that would change qtag |
| FAIL_FQ_LAMBDA | 55 | λ-transitions that would change qtag |
| **Total fail transitions** | **495** | |

Distinct states: 504; Total states generated: 1012.

**Without μ, generator set {σ, λ}** (`QARM_v02_NoMu_Stats`, authored 2026-04-20):

| Action | Print count | Delta vs full set |
|---|---|---|
| FAIL_OOB_SIGMA | **21** | **0 (INVARIANT)** |
| FAIL_OOB_MU | — | N/A (action removed) |
| FAIL_OOB_LAMBDA | 190 | −7 |
| FAIL_FQ_SIGMA | 101 | −7 |
| FAIL_FQ_MU | — | N/A (action removed) |
| FAIL_FQ_LAMBDA | 50 | −5 |
| **Total fail transitions** | **362** | **−133** |

Distinct states: 383; Total states generated: 748.

### Answer to the 2025 question

**Partial invariance.** The 2025 hypothesis ("number of OUT_OF_BOUNDS states is invariant across generator sets") is:

- **Confirmed for σ-OOB**: 21 transitions in both runs. The σ-boundary state set is identical under {σ,μ,λ} and {σ,λ}. *Why:* σ-boundary fires when `e + 1 > CAP`, and every initial state with `e ≤ CAP` reaches its σ-boundary via σ alone — μ is not required to reach them.

- **Falsified for λ-OOB, σ-FQ, λ-FQ**: these counts drop by 7, 7, and 5 respectively when μ is removed. *Why:* removing μ also removes the 121 reachable states that only μ-swap could reach; downstream transitions from those states then disappear from the state graph. The non-invariance reflects *lost reachable states*, not a change in the structural failure classification.

- **Structural interpretation:** μ's role is reachability-amplifying, not failure-class-generating. The σ- and λ- failures are attached to specific `(b,e,d,a,qtag)` configurations; whether those configurations are reachable depends on the generator set, but the set of "configurations that WOULD fail under σ" is the same in both runs.

This is a genuine empirical finding from a dormant 111-day-old spec that has never produced a TLC output before.

**Honest caveat:** `PrintT` counts action firings during BFS exploration, which can include re-exploration of the same transition from different parent states. The counts are comparable across runs but are not guaranteed to equal distinct-counterexample counts. For a distinct-counterexample count per failure type, use TLC `-dump` and post-process — future work.

---

## Proof pair summary

| Artifact | Purpose | Result |
|---|---|---|
| `QARM_v02_Failures.tla` + `.cfg` | Protocol state machine (positive) | TLC: 504 states, 0 errors, 1s |
| `QARM_v02_Failures_negative.tla` + `.cfg` | Non-vacuity of `Inv_TupleClosed` | TLC: 2 states, error as expected |
| `QARM_v02_NoMu.tla` + `.cfg` | Generator-differential positive | TLC: 383 states, 0 errors, 1s |
| `QARM_v02_Stats.tla` + `.cfg` (after `\o` fix) | Failure-action tally (full generator set) | 495 total fail transitions; 504 states |
| `QARM_v02_NoMu_Stats.tla` + `.cfg` (new, 2026-04-20) | Failure-action tally (no μ) | 362 total fail transitions; 383 states |

All five runs are reproducible from the files committed in this directory. Run 4's comparison answers a research question posed in 2025 that had been dormant since.

---

## Scope limitations (honest statement)

1. **Bounded model only.** The proof applies to `CAP = 20, KSet = {2, 3}`. Scaling to `CAP = 24` (the mod-24 applied QA domain) should be straightforward and is next work. Scaling to `CAP = 100+` may require symmetry reduction (b ↔ e is not a symmetry because σ breaks it, but λ_k has a natural Z/kZ symmetry worth declaring).

2. **Non-vacuity gap.** Only `Inv_TupleClosed` has a dedicated non-vacuity spec. The other four (`Inv_InBounds`, `Inv_QDef`, `Inv_FailDomain`, `Inv_MoveDomain`) could all be vacuous as far as this ledger proves. Future work: author `QARM_v02_Failures_negative_{bounds,qdef,faildomain,movedomain}.tla`.

3. **A1 axiom (No-Zero) gap.** QARM v02 uses `b, e ∈ 0..CAP`, not `{1..CAP}`. Per the six QA axioms (see `tools/qa_axiom_linter.py` and `CLAUDE.md`), A1 requires `b, e ∈ {1..N}`, not `{0..N-1}`. Fixing this in the spec is deferred to Lane 2 (`QAAxioms.tla` with explicit `Inv_A1_NoZero`) per `docs/specs/QA_TLA_PLUS_AUDIT.md` §5 Lane 2. Changing the init domain to `1..CAP` is a 2-line edit but would invalidate the existing state-count baselines recorded above; keeping the Dec-30 authoring form for this first run preserves comparability.

4. **Observer-projection firewall (Theorem NT) not yet encoded.** Theorem NT says continuous functions enter the QA discrete layer exactly twice (observer-layer → QA-layer → observer-layer). This spec has no continuous-layer variables at all, so the firewall condition is trivially satisfied. Encoding NT as a temporal invariant requires introducing an observer-layer variable and constraining cross-boundary transitions — Lane 2 `Inv_NT_NoObserverFeedback` design work.

5. **PrintT counts ≠ distinct-state counts.** See Run 4 caveat. For structural certainty use TLC `-dump` and post-process.

6. **QACertificateSpine.tla NOT model-checked.** The 13 KB spec authored 2026-01-21 has no `.cfg` (genuinely missing — confirmed in audit §1 Region B inventory). It instantiates seven certificate types with bundle-coherence rules + the `FailureFirstClass` theorem. Authoring a `.cfg` and running TLC against it is the natural next step after this ledger.

---

## Next steps (in priority order)

1. **Author non-vacuity specs for the remaining 4 QARM invariants.** Pattern from `cert_gate_negative_{chain,bind,composition}.tla`. ~20 minutes each.

2. **Model-check `QACertificateSpine.tla`.** Author a `.cfg` with bounded `States`, `Actions`, `TargetClasses`, `Variables`, `MaxHorizon`; declare the structural invariants (`NoSilentFailures`, `ObstructionHasWitness`, `RegretNonNegative`, `PruningEfficiencyBounded`, `BundleCoherent`). This would be the first-ever model-check of the QA certificate architecture as an abstract spec.

3. **Scale to `CAP = 24`.** Mod-24 is QA's applied-domain modulus. TLC should handle this tractably (expect ≤ 10K states based on CAP=20 scaling).

4. **Lane 2: `QAAxioms.tla`.** Encode the six QA axioms (A1, A2, T2, S1, S2, T1) plus Theorem NT as TLA+ temporal invariants. `EXTENDS QARM_v02_Failures` and add invariants one at a time, with paired non-vacuity tests. Submittable to `tlaplus/examples`.

5. **λ_k symmetry reduction.** For λ with `KSet = {2, 3, 5}` and larger CAPs, the λ_k action has `KSet`-fold redundancy. Declaring `SYMMETRY Permutations(KSet)` in the cfg could reduce state-space by up to `|KSet|!`.

6. **-dump state graph + per-state fail-class tally.** Replace PrintT counts with TLC `-dump dot <file>` and post-process with Graphviz/Python to get exact distinct-state fail-class counts.

---

## Claim at ledger-end

The QA semantic layer (σ, μ, λ generators; `(b,e,d,a,qtag,fail)` state space; first-class failure algebra per `docs/specs/QA_TLA_PLUS.md` §2–4) is **now formally model-checked for the first time**, bounded to `CAP = 20, KSet = {2,3}`. Five structural invariants hold across 504 reachable states. Non-vacuity proved for `Inv_TupleClosed`. Generator-set differential answers the 2025 ChatGPT question: σ-OOB is invariant across generator sets; λ-OOB, σ-FQ, λ-FQ are not, reflecting reachability loss rather than structural failure-class change. One parse bug (`\o` without `Sequences`) caught and fixed by the first run of a 111-day-old spec.

This is the QA analog of `llm_qa_wrapper/spec/TLC_PROOF_LEDGER.md` for the protocol layer — the *semantic* layer now has its own proof record.
