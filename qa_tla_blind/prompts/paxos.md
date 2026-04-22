# Paxos — Blind Reproduction Prompt (Mode B)

## Source

Target spec: `tlaplus/Examples/specifications/Paxos/Paxos.tla` (hidden from
reproducing session). This is the canonical single-decree Paxos by Leslie
Lamport (1998/2001), written as a "high-level" message-passing algorithm
without explicit leader or learner processes. It refines a companion
abstract spec `Voting.tla` (also hidden), which itself refines a
top-level consensus spec `Consensus.tla` (also hidden). The refinement
stack is three layers: Consensus ← Voting ← Paxos.

A model-checking harness `MCPaxos.tla` / `MCPaxos.cfg` exists and
exhaustively searches the protocol at a small fixed number of acceptors
and values (order of magnitude: single digits, with a small finite ballot
cutoff). The harness pins four conjunctive-slice invariants on the module
itself and pins one refinement property connecting this spec to the
`Voting` spec it implements.

## Problem statement (English only)

Single-decree Paxos is the canonical distributed-consensus protocol for
getting a set of acceptor processes to agree on exactly one value in the
face of message delays and arbitrary process crashes. A set of proposers
(leaders, kept implicit in this spec) drives a sequence of numbered
ballots; each ballot has two phases. In phase 1 a leader solicits
promises from a quorum of acceptors and learns which values, if any,
earlier ballots have already voted for; in phase 2 the leader picks a
value consistent with what it has learned and asks a quorum to vote for
it. A value is *chosen* when some quorum has all voted for it in the
same ballot. The safety goal is that at most one value is ever chosen,
even though many ballots may run concurrently and any acceptor may crash
at any time. The algorithm won the Turing Award in 2013 and is the
backbone of most production consensus systems (Chubby, Spanner, etcd's
Raft is a simplified cousin).

## Actors and state variables (names only)

The spec declares three constant parameters (the set of values, the set
of acceptors, the set of quorums) and four state variables:

- `maxBal` — for each acceptor, the highest ballot number it has promised
  to participate in (or a distinguished below-all-ballots sentinel if it
  has not yet promised anything).
- `maxVBal` — for each acceptor, the ballot number of the last value it
  voted for (or the sentinel if it has not voted).
- `maxVal` — for each acceptor, the actual value it voted for at
  `maxVBal`, paired with `maxVBal` to describe that acceptor's last vote
  (or a distinguished non-value sentinel if it has not voted).
- `msgs` — the pool of all messages ever sent. Monotone: messages
  accumulate and are never removed; an acceptor or leader acts "on
  receipt" by reading the pool, which means the same message can be
  re-acted on without effect.

That is four state variables total, plus three constant parameters
(values, acceptors, quorums), plus a derived message-type universe with
four message kinds (phase 1a, phase 1b, phase 2a, phase 2b).

## Process structure (English)

- N acceptors, indexed over the constant acceptor set; each carries the
  three per-acceptor pieces of local state above (`maxBal[a]`,
  `maxVBal[a]`, `maxVal[a]`).
- Leaders are external / implicit — the spec does not model leader-local
  state; any action that in an implementation would be "leader L does X"
  is here an existentially-quantified step that any ballot could take,
  constrained by the message pool.
- Ballots are drawn from a totally ordered infinite set (the natural
  numbers, in the spec), with a sentinel value strictly below all
  ballots standing in for "no ballot yet." Ballot numbers are the
  *only* way proposals are distinguished; in a real deployment each
  ballot is uniquely tied to a proposer by a numbering convention not
  modeled here.
- The standard Paxos action set the reproducer should expect to recover:
  - `Phase1a(b)` — some leader begins ballot `b` by posting a `"1a"`
    message announcing `b`. No acceptor or proposer state changes.
  - `Phase1b(a)` — acceptor `a`, on the presence of a `"1a"` message
    with ballot `b` strictly greater than its current `maxBal[a]`,
    bumps `maxBal[a] := b` and posts a `"1b"` promise message
    carrying its last-vote pair `<maxVBal[a], maxVal[a]>`.
  - `Phase2a(b, v)` — some leader, having seen `"1b"` promises from a
    full quorum for ballot `b` *and* not having already issued a `"2a"`
    for this `b`, picks a value `v` that is consistent with the quorum's
    last-vote evidence (free choice if the quorum reports no prior
    votes; otherwise the value from the highest-ballot prior vote in the
    quorum) and posts a `"2a"` proposal.
  - `Phase2b(a)` — acceptor `a`, on the presence of a `"2a"` with
    ballot `b` at least `maxBal[a]`, sets `maxBal[a] := b`,
    `maxVBal[a] := b`, `maxVal[a] := v`, and posts a `"2b"` vote
    message.
- A value is *chosen* (learner-side; not itself a step of this spec)
  when some quorum of acceptors has all posted `"2b"` messages for the
  same `<b, v>` pair.
- The next-state relation is the disjunction: over ballots `b`, either
  `Phase1a(b)` or `Phase2a(b, v)` for some value `v`; and over
  acceptors `a`, either `Phase1b(a)` or `Phase2b(a)`.

## Named properties to reproduce

The target spec's model-checking harness pins the following as scoring
targets. The reproducer must recover these (or clearly-equivalent
Mode-B-encoded analogues) as named predicates:

- `TypeOK` — the type-correctness invariant. Intuitively: the three
  per-acceptor maps are functions from acceptor to (ballot ∪ {sentinel})
  and (value ∪ {non-value}) as appropriate, and the message pool is a
  subset of the well-typed message universe.
- The **refinement property** connecting this spec to the abstract
  `Voting` spec — stated as the theorem `Spec => V!Spec` under a
  refinement mapping that projects the Paxos state onto `<votes,
  maxBal>` where `votes[a]` is the set of `<ballot, value>` pairs
  witnessed by `"2b"` messages from `a`. The reproducer should recover
  this as "Paxos implements an abstract ballot-voting consensus spec
  under the stated projection." The body of the refinement mapping
  and the proof obligation (the inductive invariant) are deliberately
  not leaked here.
- Three conjunctive slices of an inductive invariant `Inv` (named in
  the harness as `Inv2`, `Inv3`, `Inv4` — the parts beyond `TypeOK`).
  They together encode: each acceptor's `<maxVBal, maxVal>` really does
  correspond to one of its `"2b"` votes; every `"1b"` message's
  last-vote field really does match the sender's vote-history; every
  `"2a"` message is backed by a quorum that "shows safe" the value for
  that ballot, and no two `"2a"` messages for the same ballot carry
  different values. The reproducer does **not** need to recover these
  word-for-word; they are the scaffolding of the safety proof and the
  scorer will accept a correctly-encoded analogue that carries the
  same informational content under the Mode B primitives.

A liveness property (`MCLiveness` in the harness: eventually *some*
value is chosen, under fairness of a specific ballot and its
acceptors) is also named but is optional for the reproducer; Paxos
liveness is outside the core safety proof and the Mode B scorer will
not penalize its omission.

That is one type invariant, one refinement theorem, three inductive-
invariant slices, plus the optional liveness — call it **five
safety-scoped named properties** for Mode B scoring (Types +
refinement + three Inv-slices), with liveness as a sixth optional.

## What is explicitly withheld

- The body of `Init` and the body of `Next`.
- The bodies of the four action predicates (`Phase1a`, `Phase1b`,
  `Phase2a`, `Phase2b`). Action *names* are public-domain Paxos and
  are given above; the guards and updates inside each are not.
- The body of the refinement mapping (the definition of `votes` in
  terms of `"2b"` messages) and the body of `V!Spec` that this one
  refines. The reproducer must reconstruct what "Paxos implements
  abstract voting" means from the informal structure above.
- The body of the inductive invariant `Inv` and the bodies of its
  conjunctive slices. The reproducer must derive the three slices'
  informational content from the English summary in the previous
  section.
- The exact quorum definition pinned in the model-checking harness.
  The spec-level `Quorum` is parametric and only constrained by an
  assumption that every quorum is a subset of the acceptor set and
  that any two quorums intersect non-trivially. The harness pins a
  specific enumerated family of quorums for bounded checking; the
  exact family and the specific cardinality condition (e.g. "size
  strictly greater than half of N," "size at least some threshold,"
  etc.) are not revealed. The reproducer must derive what "quorum"
  means from the intersection assumption plus the informal "majority"
  hint in the problem statement.
- The exact ballot-number structure beyond "totally ordered infinite
  set with a below-all sentinel." Whether the concrete choice is the
  naturals, some initial segment thereof, or a synthetic order is not
  revealed. The reproducer should encode ballots as an abstract
  totally-ordered infinite carrier and note whether their chosen
  Mode-B primitive (spread-polynomial monoid? projective coordinate
  line? synthetic integer strict order?) actually supplies that
  carrier.
- The small finite values the harness pins for N (acceptors), V
  (values), and B (ballot cutoff). Order of magnitude only: small
  single-digit cardinalities for all three, with the ballot cutoff
  bounding how many phase-1 rounds the checker explores.
- All TLC counterexamples, reachable-state counts, and state-depth
  numbers.

## Notes for the reproducing session

- **This is Mode B.** Read `qa_tla_blind/mode_b/wildberger_object_model.md`
  before encoding. That file fixes the primitive alphabet the scorer
  measures encoding fit against (points, lines, quadrances in three
  colors, spreads, TQF, cross-ratio, spread polynomials, SL(3)
  hexagonal rings, the 4D diagonal rule, and the declared transforms:
  translations, reflections, spread-polynomial rotations, projective
  maps, mutation moves). Nominal renaming — calling `maxBal` a "point"
  without load-bearing use of Wildberger primitives — scores at the
  `ornamental-overlay` level.

- **Four structural axes the scorer will examine separately.** Be
  explicit about each in your attempt and state which Wildberger
  primitive you are using to discharge it:

  1. **Order / ballot structure.** Ballots form a totally ordered
     infinite carrier with a below-all sentinel. How do you encode this
     in Wildberger primitives? Candidate encodings: spread-polynomial
     monoid composition (degree as proxy for ballot rank); a
     projective-line coordinate with a chosen point-at-infinity; a
     strict synthetic sequence reached by repeated application of a
     designated mutation-game move. Which carries a genuine total order
     *and* supplies the "strictly greater than" guard needed by
     `Phase1b` without smuggling in a continuous metric?

  2. **Quorum intersection.** The only substantive assumption on
     `Quorum` is that any two quorums share at least one acceptor.
     How do you encode this geometrically? Candidate encodings:
     chromogeometric intersection (two lines in one of the three
     metrics share a point); set-lattice meet of subset-indexed
     elements; TQF collinearity over quorum-member points forcing a
     shared coplanar representative. The chosen encoding must make
     the intersection lemma *derivable* from your primitives, not
     re-asserted as a fresh axiom.

  3. **Message monotonicity.** The message pool `msgs` grows and never
     shrinks; this is the same gap the TwoPhase Mode B attempt had to
     face. Does the Wildberger bundle supply a monotone-accumulating
     primitive, or is this a genuine repeat gap that Mode B as
     currently specified cannot encode without ad-hoc extension? Be
     honest: if you have to import set-union from outside the
     primitive list, flag it and score yourself accordingly.

  4. **Guarded receipt / conditional update.** Each of `Phase1b` and
     `Phase2b` does a monotonic upward update of `maxBal[a]` guarded by
     a strict (or non-strict) comparison with an incoming ballot.
     Mutation-game moves are a natural candidate for "guarded transform
     on the acceptor's local point"; projective maps with a fixed
     ordering are another. Make a concrete choice and justify why the
     guard is expressible in the primitives (not just the update).

- **The central safety claim is consensus**, and it is strictly
  stronger than the TwoPhase "no disagreement" pairwise form.
  Consensus says *any two chosen values are equal*, full stop. The
  classical proof flows through the **quorum-intersection lemma**
  (any two quorums share at least one acceptor, and that shared
  acceptor's monotone `maxBal` forces agreement across ballots). The
  Mode B question is: what does this lemma look like geometrically
  in your chosen primitive view? A chromogeometric null-cone
  condition? A cross-ratio invariance across a shared projective
  point? A mutation-game move that cannot separate two objects once
  they meet? Make this explicit; it is the heart of Paxos and is
  where encoding fit is most load-bearing.

- **QA axioms A1/A2/T1/T2/NT/S1/S2 still apply.** In particular:
  ballot numbers almost certainly need an A1 shift (no ballot zero as
  a valid ballot state — use `{1,…,N}` with the sentinel encoded as
  a reserved symbol outside the ballot set, not as `0`). The
  `<maxVBal, maxVal>` pair, if encoded as a derived coordinate from
  an acceptor's vote-history point, must obey A2 (`d = b + e`,
  `a = b + 2e` style — derived, never independently assigned). The
  message pool is a discrete accumulating object and should not carry
  any continuous time or floating-point weighting (T1, S2). Any real-
  valued quantity (timeout intervals, failure probabilities) is an
  observer projection and cannot be a causal input to the protocol
  logic (T2, Theorem NT).

- **The spec is parametric in N (acceptors), V (values), and in the
  ballot cutoff B.** The model-checking harness pins all three to
  small single-digit cardinalities for exhaustive search; the exact
  values are not revealed here so as not to leak the bounded state
  space. Your encoding should survive arbitrary N and V and a
  countable ballot carrier. Flag explicitly any place where your
  encoding relies on a fixed N (e.g. a specific quorum enumeration
  that does not generalize) or a bounded ballot range.
