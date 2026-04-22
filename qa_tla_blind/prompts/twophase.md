# TwoPhase — Blind Reproduction Prompt (Mode B)

## Source

Target spec: `tlaplus/Examples/specifications/transaction_commit/TwoPhase.tla`
(hidden from reproducing session).

Chosen over the sibling `specifications/TwoPhase/TwoPhase.tla` because that
sibling is a different protocol (Lamport's two-phase *handshake*, a trivial
Producer/Consumer alternation with only three variables `p`, `c`, `x`) and
does not carry the distributed-commit structure this benchmark is meant to
probe. The transaction-commit version is the canonical Gray–Lamport
Two-Phase Commit spec, with the full RM/TM actor set and a non-trivial
safety property, which is what the Mode B Wildberger object model was
written to be tested against. The sibling README confirms the naming split.

Companion module referenced by the target spec: `TCommit.tla` (the
specification the TwoPhase spec *refines*). The reproducing session is not
shown either module. A model-checking harness `TwoPhase.cfg` exists and
exhaustively searches the protocol for a small, fixed number of resource
managers (order of magnitude: single digits); the harness pins one
`INVARIANT` on the module itself.

## Problem statement (English only)

Two-Phase Commit (2PC) is the classical distributed-atomicity protocol for
coordinating a single transaction across N independent resource managers
(RMs) and one transaction manager (TM). Each RM starts in a working state
and may locally decide to prepare (tentatively agree to commit) or to
unilaterally abort. The TM observes which RMs have announced themselves
prepared; once every RM is prepared, the TM may decide to commit the
transaction and broadcast that decision, after which each prepared RM may
adopt the committed decision. Alternatively, while the TM is still in its
initial state it may spontaneously abort the transaction and broadcast
that decision instead; any RM may then adopt the aborted decision. The
core safety claim is that the RMs never arrive at globally inconsistent
outcomes: no RM commits while another has aborted, and no RM aborts while
another has committed.

## Actors and state variables (names only)

The spec declares a single constant (the set of resource managers) and
four state variables:

- `rmState` — for each RM, its local state (one of working, prepared,
  committed, aborted).
- `tmState` — the TM's state (one of init, committed, aborted).
- `tmPrepared` — the set of RMs from which the TM has received a Prepared
  announcement.
- `msgs` — the pool of in-flight messages, modeled as a monotone set that
  messages are added to but never removed from (receipt-without-consumption
  semantics).

That is four state variables total, plus one constant parameter.

## Process structure (English)

- N resource managers, indexed over the constant RM set; each carries a
  local RM-state that is one of working, prepared, committed, or aborted.
- One transaction manager with a single TM-state that is one of init,
  committed, or aborted.
- A single shared message pool. Messages are one of three kinds: a
  Prepared announcement sent from a specific RM to the TM, a Commit
  broadcast from the TM, or an Abort broadcast from the TM. Messages
  accumulate; nothing is consumed. An action that in an implementation
  would be "enabled on receipt of message M" is here enabled by the
  presence of M in the pool, which means the same message may be acted
  on more than once without effect.
- Transitions are labeled actions. The standard TPC action set the
  reproducer should expect to recover:
  - `RMPrepare` — an RM moves from working to prepared and posts a
    Prepared announcement for itself.
  - `RMChooseToAbort` — an RM unilaterally moves from working to aborted
    without posting a message (the spec simplifies by folding the would-be
    unilateral abort into a TM-spontaneous abort path).
  - `TMRcvPrepared` — the TM, while still in init, observes a Prepared
    announcement and adds the sender to the set of prepared RMs.
  - `TMCommit` — the TM, while in init and with every RM already in its
    prepared set, transitions to committed and broadcasts a Commit.
  - `TMAbort` — the TM, while in init, transitions to aborted and
    broadcasts an Abort (this is the spec's only abort-origination path).
  - `RMRcvCommitMsg` — any RM, on the presence of a Commit broadcast,
    transitions to committed.
  - `RMRcvAbortMsg` — any RM, on the presence of an Abort broadcast,
    transitions to aborted.
- The next-state relation is the disjunction of the two TM self-steps
  (`TMCommit`, `TMAbort`) with an existential over RM of the five
  RM-parameterised steps above.

## Named properties to reproduce

The target module carries exactly two named predicates the reproducer
should recover; one is asserted as an invariant via the module's
model-checking harness, the other is asserted as an invariant via the
refinement theorem connecting this module to the companion specification
it implements. Together they are the benchmark's two scoring targets:

- `TPTypeOK` — the type-correctness invariant. Intuitively: each of the
  four state variables ranges over its declared universe — each RM's
  local state is one of the four named strings; the TM's state is one of
  its three named strings; the set of prepared-RMs-seen-by-TM is a
  subset of RM; the message pool is a subset of the well-typed message
  space (Prepared-with-sender, Commit, Abort).

- `TCConsistent` — the cross-RM safety invariant inherited from the
  companion specification the protocol refines. Intuitively: no pair of
  RMs ends up holding mutually contradictory terminal decisions. The
  reproducer must derive what "two RMs disagree" means as a predicate
  over `rmState`; the exact disagreement-forbidding form is deliberately
  not leaked here. Both the quantifier structure over the RM set and the
  specific pairings of terminal local states that count as "disagreement"
  should fall out from the informal safety statement in the problem-
  statement section and from first-principles atomicity reasoning.

No liveness property is specified. The module is a pure-safety spec; the
source header explicitly notes that "what must happen" is deferred to
a liveness spec not given here.

## What is explicitly withheld

- The body of `TPInit` (the initial predicate).
- The body of `TPNext` (the next-state relation) and the bodies of the
  seven named actions it disjoins.
- The body of `TCConsistent`. The reproducer must derive the
  disagreement-forbidding condition from the English safety statement
  above.
- The exact shape of the Message set (which message kinds exist, how
  they carry their sender, whether RM-ids are tagged on broadcasts).
  The reproducer is told only that three kinds exist (Prepared, Commit,
  Abort), that Prepared carries a sender, and that the pool is
  monotone-accumulating.
- The refinement theorem and the substitution under which TwoPhase
  implements the companion spec.
- Any TLC counterexamples or reachable-state counts.
- The exact value pinned for the RM constant in the model-checking
  harness. The constant is parametric in the spec; the MC harness pins
  it to a small fixed cardinality (single digits) for bounded checking.
  The reproducer should treat N as parametric when encoding and note
  whether their encoding survives arbitrary N.

## Notes for the reproducing session

- This is Mode B: you are given a pre-specified QA-native object model
  (Wildberger primitives) to use as your encoding alphabet. That file is
  `qa_tla_blind/mode_b/wildberger_object_model.md`. Read it before
  encoding. It fixes the primitives (points, lines, quadrances in three
  colors, spreads, TQF, cross-ratio, spread polynomials, SL(3) hexagonal
  rings, 4D diagonal rule) and the transforms (translations, reflections,
  spread-polynomial rotations, projective maps, mutation moves) so that
  your Contribution score measures encoding fit, not discoverability.

- QA axioms still apply. A1 (states live in `{1,...,N}`, never zero). A2
  (derived coordinates `d = b+e`, `a = b+2e` are derived, never assigned
  independently). T1 (path-time is the integer path length `k`; no
  continuous time). T2 / Theorem NT (observer-projection firewall:
  continuous functions are observer projections only, never causal
  inputs to QA logic). S1 (`b*b`, never `b**2`). S2 (integer or Fraction
  state only; no numpy-float state).

- The protocol has a natural bipartite / tri-valued structure: each RM's
  terminal decision sits in `{commit, abort, undecided}` with the
  intermediate `{working, prepared}` feeding into it, and the TM's
  state sits in a parallel but smaller three-valued set. Consider whether
  the global system state is naturally a point in a lattice product
  (one factor per RM, one for the TM, one for the message pool), whether
  the RM-local state is naturally a point in a small `P²(Z)` or
  `Caps(N, N)` shape, and whether the specific disagreement pattern the
  safety invariant forbids has a **chromogeometric / null-cone**
  interpretation in one of the three Wildberger metrics.

- Safety is "no disagreement." What does that mean geometrically? That
  is the central Mode B question. The object model's red / green / blue
  quadrances give three parallel ways to express "the decisions of two
  RMs are equal / opposite / complementary"; the reproducer should make
  a concrete choice, justify which chromogeometric view encodes the
  forbidden pattern most naturally, and write the invariant as a null-
  cone condition (or a comparably principled geometric condition) in
  that view. Nominal renaming — calling `rmState` a "point" without
  load-bearing use of the primitives — scores at the `ornamental-
  overlay` level under Mode B.

- The spec is parametric in `N`, the number of RMs. The model-checking
  harness pins `N` to a small fixed single-digit cardinality for
  exhaustive search; the exact pinned value is not revealed here in
  order not to leak the bounded state space. Your encoding should
  survive arbitrary `N`, and you should flag explicitly any place where
  your encoding would not.
