# Quantum Arithmetic — Axiom System as TLA+ Temporal Invariants

**Submission target:** [github.com/tlaplus/examples](https://github.com/tlaplus/examples) (`specifications/QuantumArithmetic/`)
**Status:** Standalone package ready for review. PR not yet opened.
**Provenance:** Extracted from `signal_experiments` monorepo (Will Dale, 2026).
Audit trail: `qa_alphageometry_ptolemy/QARM_PROOF_LEDGER.md` in the source
repo records 20 TLC runs across the development cycle.

---

## What this package is

A self-contained TLA+ specification of the **Quantum Arithmetic (QA)**
generator algebra, with its six compliance axioms plus Theorem NT
(Observer Projection Firewall) encoded as TLA+ temporal invariants over
the reachable state graph. Each runtime-checkable invariant has a paired
non-vacuity spec that produces a minimal counterexample, mirroring the
proof-pair discipline common in the Paxos / Raft / distributed-snapshot
exemplars elsewhere in this corpus.

The package is **self-contained**: no external dependencies beyond the
TLA+ standard modules (`Naturals`, `Integers`, `TLC`, `Sequences`,
`FiniteSets`). Reproducing every result requires only `tla2tools.jar`.

---

## Why this might be interesting to TLA+ users

1. **First-class failure algebra.** Failures (`OUT_OF_BOUNDS`,
   `FIXED_Q_VIOLATION`, `ILLEGAL`) are modelled as absorbing-stutter
   state variables, not silent transition blocks. This matches the 2025
   design correspondence of the system's authors (Dale et al.) and gives
   TLC a concrete way to measure generator-set differentials on a
   failure-class-by-failure-class basis (see `QARM_PROOF_LEDGER.md`
   Run 4 in the source repo for a non-trivial example of this producing
   a research finding).

2. **Observer-projection firewall as temporal invariant.** Theorem NT
   of the QA axiom block asserts that the continuous–discrete
   boundary is crossed exactly twice per trace (input at `Init`, output
   at a dedicated `Project` action). This is encoded here with two
   observer-layer variables (`obs_float`, `obs_cross_count`) and two
   distinct invariants:
   - `Inv_T2_FirewallRespected` (spatial firewall — observer state
     immutable during discrete evolution)
   - `Inv_NT_NoObserverFeedback` (temporal bound — at most two
     boundary crossings).
   To our knowledge this is the first TLA+ encoding of a
   continuous-discrete firewall as a runtime-checkable property.

3. **Axiom hygiene.** Six of the seven invariants (A1, A2, S2, T1, T2, NT)
   have dedicated non-vacuity counterexample specs, so the positive
   result is demonstrably non-vacuous on each axiom. S1 (no `^2`
   operator on state variables) is structural and documented as such.

4. **Duo-modular packing.** The `qtag = 24 * phi9(a) + phi24(a)` packing
   puts `qtag` in `[0, 239]` and preserves both the mod-9 and mod-24
   invariants simultaneously — an efficient hash for TLC's fingerprint
   set and a research object in its own right.

---

## Background — one-paragraph QA

**Quantum Arithmetic** is a modular-arithmetic framework over pairs
`(b, e)` with derived coordinates `d = b + e` and `a = b + 2e`. Dynamics
live in the discrete integer layer; continuous functions enter only as
**observer projections** at the input and output boundaries. The system
uses three generator actions:

- **σ** (sigma): `e → e + 1`
- **μ** (mu):    swap `b ↔ e`
- **λ_k** (lambda): scale `(b, e) → (k·b, k·e)` for `k ∈ KSet`

Each generator is paired with explicit failure actions (`OUT_OF_BOUNDS`
when successor exceeds `CAP`, `FIXED_Q_VIOLATION` when successor breaks
duo-modular invariance). The failure states are absorbing — once in a
fail state, the tuple is frozen.

The six compliance axioms are:

| Axiom | Rule | Runtime check? |
|---|---|---|
| A1 | No-Zero: `b, e ∈ {1..CAP}` | Yes (`Inv_A1_NoZero`) |
| A2 | Derived Coords: `d = b+e`, `a = b+2e` | Yes (`Inv_A2_DerivedCoords`) |
| S1 | No `^2` operator; use `b*b` | Structural (module text) |
| S2 | Integer state: `b, e, d, a ∈ Nat` | Yes (`Inv_S2_IntegerState`) |
| T1 | Integer path time: discrete generator alphabet only | Yes (`Inv_T1_IntegerPathTime`) |
| T2 | Firewall: observer outputs don't feed back as QA inputs | Yes (`Inv_T2_FirewallRespected`) |

Plus **Theorem NT** (Observer Projection Firewall): boundary crossed
exactly twice per trace, encoded as `Inv_NT_NoObserverFeedback`.

---

## Module inventory

| File | Role |
|---|---|
| `QARM_v02_Failures_A1.tla` | Base generator algebra. σ/μ/λ_k actions + paired failure actions. A1-compliant Init (`b, e ∈ 1..CAP`). |
| `QARM_v02_Failures_A1.cfg` | Bounded model (`CAP = 20`, `KSet = {2, 3}`). |
| `QAAxioms.tla` | EXTENDS the base. Adds observer-layer variables (`obs_float`, `obs_cross_count`) + `Project` action + the seven axiom invariants. |
| `QAAxioms.cfg` | CAP=20 positive check, all 7 invariants active. |
| `QAAxioms_cap24.cfg` | CAP=24 applied-domain scale check (mod-24). |
| `QAAxioms_negative_A1.tla` | Non-vacuity: writes `b' = 0`, violates `Inv_A1_NoZero`. |
| `QAAxioms_negative_A2.tla` | Non-vacuity: writes `d' ≠ b' + e'`, violates `Inv_A2_DerivedCoords`. |
| `QAAxioms_negative_S2.tla` | Non-vacuity: writes `b' = "ghost"` (non-Nat), violates `Inv_S2_IntegerState`. |
| `QAAxioms_negative_T1.tla` | Non-vacuity: writes `lastMove' = "t_continuous"` outside alphabet. |
| `QAAxioms_negative_T2.tla` | Non-vacuity: writes `obs_float' = 42` during discrete step. |
| `QAAxioms_negative_NT.tla` | Non-vacuity: increments `obs_cross_count` past 2. |
| `QAAxioms_negative_*.cfg` | Paired configs (one per negative spec). |
| `NOTICE.md` | Attribution notice (mirrors source repo `LICENSE_NOTICE.md`). |

8 TLA+ modules + 9 configs + 1 README + 1 NOTICE = 19 files total.

---

## Reproduction

From this directory, with `tla2tools.jar` on the path (or at a known
location referenced by `$TLA2TOOLS`):

```bash
# Positive check: all 7 invariants hold over the reachable state graph.
# Expected: 90 initial states / 470 distinct / depth 3 / 0 errors / ~1 s.
java -XX:+UseParallelGC -jar $TLA2TOOLS -workers 4 -terse \
    -config QAAxioms.cfg QAAxioms.tla

# Applied-domain scale: same spec, CAP = 24 (mod-24 QA).
# Expected: 132 initial states / 686 distinct / 0 errors / ~5 s.
java -XX:+UseParallelGC -jar $TLA2TOOLS -workers 4 -terse \
    -config QAAxioms_cap24.cfg QAAxioms.tla

# Base spec (QARM generator algebra, without axiom layer).
# Expected: 90 initial states / 374 distinct / 0 errors / ~1 s.
java -XX:+UseParallelGC -jar $TLA2TOOLS -workers 4 -terse \
    -config QARM_v02_Failures_A1.cfg QARM_v02_Failures_A1.tla

# Six non-vacuity checks — each produces a 2-state counterexample
# for its target invariant.
for axiom in A1 A2 S2 T1 T2 NT; do
  echo "=== Non-vacuity: Inv_${axiom} ==="
  java -XX:+UseParallelGC -jar $TLA2TOOLS -workers 4 -terse \
      -config QAAxioms_negative_${axiom}.cfg QAAxioms_negative_${axiom}.tla
done
```

---

## Expected results

| Spec / Config | States | Depth | Outcome |
|---|---|---|---|
| `QARM_v02_Failures_A1` @ CAP=20 | 90 init / 374 distinct | 2 | 0 errors (5 base invariants hold) |
| `QAAxioms` @ CAP=20 | 90 init / 470 distinct | 3 | 0 errors (all 7 axiom invariants hold) |
| `QAAxioms` @ CAP=24 | 132 init / 686 distinct | 3 | 0 errors |
| `QAAxioms_negative_A1` | 2 | 2 | `Inv_A1_NoZero` violated (b'=0 counterexample) |
| `QAAxioms_negative_A2` | 2 | 2 | `Inv_A2_DerivedCoords` violated (d'=99 ≠ b'+e') |
| `QAAxioms_negative_S2` | 2 | 2 | `Inv_S2_IntegerState` fails to evaluate (b'="ghost" — string ∉ Nat) |
| `QAAxioms_negative_T1` | 2 | 2 | `Inv_T1_IntegerPathTime` violated (lastMove' out of alphabet) |
| `QAAxioms_negative_T2` | 2 | 2 | `Inv_T2_FirewallRespected` violated (obs_float' modified by QA step) |
| `QAAxioms_negative_NT` | 2 | 2 | `Inv_NT_NoObserverFeedback` violated (3rd boundary crossing attempted) |

All counterexamples are minimal (Init + 1 step). The wall-time budget on
a 4-core commodity workstation (OpenJDK 21 + TLC 2.20) is under 30 s for
the full package including the scale test.

**Note on the S2 counterexample.** TLC reports
`Error: Evaluating invariant Inv_S2_IntegerState failed` (rather than the
more familiar "Invariant X is violated") because the successor-state
assignment `b' = "ghost"` produces a state on which the invariant's
`b \in Nat` predicate is undefined — TLC cannot decide whether a string
is a natural number. This is the expected outcome for a type-domain
violation: the invariant fails, but via an evaluation error rather than
a boolean violation. Either way, the negative spec demonstrates S2 is
non-vacuous.

---

## Design notes

### Why `EXTENDS` rather than a single monolithic module

`QAAxioms.tla` `EXTENDS QARM_v02_Failures_A1` so that the generator
algebra and the axiom invariants are separable concerns. This matches
the upstream convention in `specifications/Paxos/` (where
`Paxos.tla` composes `Consensus.tla`) and lets the base be reused by
other extension modules (e.g., alternative axiom systems, or
domain-specific refinements at larger `CAP`).

### The `Project` action and the observer-layer variables

The base spec has no notion of a continuous/observer layer. `QAAxioms.tla`
introduces two variables:

- `obs_float ∈ 0..(3·CAP)` — observer scalar. Set to `0` at `Init`
  (symbolic "input seed") and updated exactly once by the `Project`
  action to the current value of the QA coord `a`.
- `obs_cross_count ∈ {1, 2}` — number of boundary crossings so far.
  `1` at `Init` (input crossing); `2` after `Project` fires (output
  crossing). Reaching `3` would indicate observer-layer feedback into
  the discrete layer, which Theorem NT forbids.

The extended `Next_ext` has three disjuncts:

1. `QA_firewalled`: a base-spec `Next` move PLUS `UNCHANGED <<obs_float,
   obs_cross_count>>`. Only fires when `obs_cross_count = 1`.
2. `Project`: the output boundary crossing.
3. `PostProjectStutter`: absorbing after `Project`.

### Why `Inv_S1_NoSquareOperator` is `b * b >= 0`

TLA+ has no native `^2` or `**` operator on state variables in the
relevant modules; S1's "no libm-ULP-drift-prone squaring" rule is
**syntactic** over module text, not a state predicate over reachable
values. To keep all seven axioms as named invariants in a single
consistent stack, `Inv_S1_NoSquareOperator` is defined as the trivially
true state predicate `b * b >= 0`. This locks in the `b*b` convention at
the module level (any future author introducing `b^2` would diff against
this predicate) and satisfies TLC's preference for state invariants to
reference at least one state variable (otherwise TLC warns about
constant-level formulas). The S1 enforcement at the *text* level is
handled by an external linter in the source repo.

---

## Findings during development (brief)

During the 2026-04-20 authoring cycle, running TLC against the dormant
QARM specs surfaced real bugs that had sat for 111 days because the
specs had never been parsed:

- `QARM_v02_Stats.tla` used the `\o` sequence-concatenation operator
  without `Sequences` in its `EXTENDS` clause — caught by TLC's parser
  on first attempt.
- The companion `QACertificateSpine.tla` module (not included in this
  submission package — scope-limited) had a function-type syntax error
  at `evidence: Variables -> STRING` (needs brackets:
  `evidence: [Variables -> STRING]`) and used an undeclared `NULL`
  sentinel. Both caught by TLC on first parse.

The TLA+ model-checker is more effective at flushing out stale specs
than any static analysis, even when the specs never reach a "real" TLC
run. This is one of the reasons we're submitting.

---

## References

Primary TLA+ references:

- Lamport, L. (1994). *The Temporal Logic of Actions.* ACM Transactions
  on Programming Languages and Systems 16(3), 872–923. DOI:
  [10.1145/177492.177726](https://doi.org/10.1145/177492.177726).
- Lamport, L. (2002). *Specifying Systems: The TLA+ Language and Tools
  for Hardware and Software Engineers.* Addison-Wesley.
  ISBN 978-0-321-14306-8.
- Cousineau, D., Doligez, D., Lamport, L., Merz, S., Ricketts, D., &
  Vanzetto, H. (2012). *TLA+ Proofs.* FM 2012 (LNCS 7436), 147–154.

QA source repo (for provenance and additional context — the full
`QARM_PROOF_LEDGER.md` documents every TLC run, including ones not
reproduced here):

- `signal_experiments` monorepo (Will Dale, 2026).
  Relevant artifacts:
  - `qa_alphageometry_ptolemy/QARM_PROOF_LEDGER.md` — 20-run ledger.
  - `qa_alphageometry_ptolemy/QARM_v02_Failures.tla` — the pre-A1
    variant (documents the provenance of the A1 correction).
  - `docs/specs/QA_TLA_PLUS.md` — three-layer constitutional
    architecture (TLA+ = constitution, Rust/Python = engine,
    learning = exploration).

---

## What's not here (and why)

- **`QACertificateSpine.tla`** — the certificate-architecture spec
  included in the source repo is out of scope for this submission. It
  currently has a falsified `FailureFirstClass` theorem under TLC
  (caught during the same authoring cycle — see source-repo ledger
  Runs 15–16 for the two counterexamples). Resolving that is a design
  decision pending in the source repo; we'd rather submit a clean
  package than ship a known-broken theorem.
- **Lean 4 layer.** The source repo also hosts a Lean 4 proof of
  ledger-data invariants (`llm_qa_wrapper/spec/LedgerInvariants.lean`)
  that complements the cert-gate protocol TLA+ spec. That's its own
  separate artifact and not part of this submission.
- **CAP > 24 scale.** State-space growth is 1.46× from CAP=20 to
  CAP=24. Scaling to CAP=48 or beyond is future work and not shipped
  here.
