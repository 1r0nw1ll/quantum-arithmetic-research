# Bakery — Blind Reproduction Prompt

## Source
Target spec: `tlaplus/Examples/specifications/Bakery-Boulangerie/Bakery.tla` (hidden from reproducing session).

Companion model-checking harness: `MCBakery.tla` + `MCBakery.cfg` (also hidden).
A sibling variant (`Boulanger.tla`) exists in the same directory and is out of scope for
this reproduction.

## Problem statement (English only)
The Bakery algorithm is Leslie Lamport's 1974 solution to Dijkstra's concurrent-programming
problem: N asynchronous processes repeatedly need exclusive access to a shared critical
section, and the algorithm must coordinate them using only read/write shared variables
(no atomic test-and-set, no hardware locks). Each process about to contend takes a
"ticket number" that is strictly greater than every ticket it currently sees held by
others, then waits until every other contender either has no ticket or has a
lexicographically larger ticket. The ticket-number assignment is non-atomic — while a
process is choosing its number, another process may read a stale value — so the
algorithm additionally maintains a "choosing" flag per process to guard the read.
The algorithm's load-bearing safety property is **mutual exclusion**: at no reachable
state may two distinct processes simultaneously be in their critical section. The usual
liveness properties (deadlock-freedom, starvation-freedom) are also expected to hold
under fair scheduling.

## State variables (names only)
Global (per-process functions indexed by process id):

- `num` — the ticket number currently held by each process (0 means "no ticket").
- `flag` — a per-process Boolean flag indicating the process is currently in the act of
  choosing its ticket.

Per-process local scratch (one copy per process):

- `unchecked` — the set of other processes this process still needs to inspect in the
  current waiting loop.
- `max` — a running maximum observed while scanning other processes' tickets.
- `nxt` — the process id currently being inspected inside the waiting loop.
- `pc` — a per-process program-counter label naming which step of the process body the
  process is at.

## Process structure (English)
There are N identical processes (ids `1..N`). Each process executes a non-terminating
outer loop alternating **non-critical section → entry protocol → critical section →
exit protocol → (back to non-critical)**. The entry protocol is broken into several
sub-steps:

1. raise a "I am choosing" flag and prepare to scan peers,
2. scan every other process's ticket and remember the running maximum,
3. commit a ticket strictly greater than that maximum,
4. lower the choosing flag and prepare the wait-phase,
5. a wait loop that, for each peer in turn, waits until that peer is either
   not-choosing + ticket-free, or holds a lexicographically larger ticket than us,
6. enter the critical section,
7. exit by releasing the ticket (setting it back to the "no-ticket" sentinel).

The reproducing session must decide how many distinguishable control locations this
pattern requires (Lamport's encoding uses a small finite set of labels including
`ncs`, an entry phase, a waiting phase, `cs`, and an exit phase; the exact label
set is part of what you reproduce, not something handed to you).

## Named properties to reproduce
The reproducing session must write a QA-native form for each of these. For each, a
short English hint is given; the predicate body is withheld.

Safety invariants (declared in the model-check config as `INVARIANTS`):

- `MutualExclusion` — no two distinct processes are simultaneously in the critical
  section. This is the primary theorem the algorithm exists to establish.
- `TypeOK` — every state variable stays in its declared domain (ticket numbers are
  naturals, flags are Boolean, the program counter takes only legal labels, per-process
  scratch has the right shape). A type-correctness well-formedness invariant; it is
  not itself the mutual-exclusion claim, but `MutualExclusion` is proved via an
  inductive invariant of which `TypeOK` is one conjunct.
- `Inv` — the full inductive invariant = `TypeOK` **and** a deeper per-process
  predicate (call it the "ticket-ordering witness") that captures, for each process
  currently waiting or in CS, the precise reason every other process cannot sneak
  past it into CS first. The reproducer is responsible for deriving the form of this
  witness; it is the load-bearing inductive strengthening of mutual exclusion.

Liveness properties (declared in the spec but commented out in the Bakery model config
due to an upstream TLC issue; still part of the intended reproduction target):

- `DeadlockFree` — whenever some process is trying to enter CS, eventually some process
  enters CS. ("The system as a whole always makes progress.")
- `StarvationFree` — every individual process that starts trying to enter CS eventually
  enters CS. ("No process is permanently overtaken.")

Fairness assumption backing the liveness claims:

- Weak fairness on each process's non-`ncs` steps (a process that has left the
  non-critical section and has an enabled step will eventually take it). The
  reproducer must state the fairness assumption explicitly; both liveness properties
  depend on it.

## What is explicitly withheld
- The `Init` predicate body (the exact initial values for each variable).
- The `Next`-state action body: the detailed per-label transitions (how each labeled
  step updates `num`, `flag`, `unchecked`, `max`, `nxt`, `pc`).
- The exact predicate bodies for `MutualExclusion`, `TypeOK`, and the inductive
  strengthening inside `Inv`. Names only.
- The fairness formula's exact temporal form (you are told "weak fairness per
  non-`ncs` step" in prose; writing it out is on you).
- The PlusCal / TLA+ label alphabet beyond the prose hints above.
- The full TLAPS proof obligation structure used in the ground truth.
- Any TLC model-checker counterexamples, state-space sizes, or bug traces.

The reproducer should know two structural facts because they are central to the
problem and not leakage of the proof:

- **Tie-breaking:** two processes can legally pick the same ticket number (the
  choosing phase is non-atomic). Ties are therefore resolved by a secondary key;
  the natural choice is lexicographic comparison of `<<ticket, process-id>>`, and
  the algorithm only works if this secondary key is a strict total order on process
  ids. Deciding whether and how to use process id as the tie-breaker is part of the
  reproduction.
- **Ticket unboundedness:** the ticket numbers are drawn from the naturals and can
  grow without bound over an infinite execution. For model-checking, a separate
  harness constrains them to a bounded range; for the algorithmic spec itself they
  are unbounded naturals.

## Notes for the reproducing session
- Parameter count: the ground-truth model config fixes `N = 2` and bounds ticket
  numbers by `MaxNat = 2` for TLC exhaustive search. The *spec* itself is parametric
  in `N` (any natural); the bound is a model-checking artifact, not a property of the
  algorithm. Your QA-native reproduction should state the algorithm parametrically and
  treat concrete `N` as a model instance.
- The spec is parametric and infinite-state in principle (unbounded tickets); TLC
  validated the bounded instance `(N=2, MaxNat=2)` exhaustively (~655k distinct
  states). The TLAPS proof of mutual exclusion is parametric in `N`.
- QA axiom considerations when encoding:
  - **A1 (no-zero):** process ids are `1..N`, not `0..N-1`. Ticket numbers naturally
    include a "no ticket" sentinel; in the ground truth that sentinel is `0`, but
    QA encoding must decide how to represent "absent ticket" without introducing a
    `0` element into the QA state alphabet — a separate absent-flag, or ticket drawn
    from `{1..}` with a dedicated "no-ticket" tag, are both candidates.
  - **A2 (derived coords):** if you introduce a `(b, e)` pair to encode per-process
    state, `d = b + e` and `a = b + 2e` must be derived, never independently assigned.
  - **T1 (path time):** "time" in this algorithm is integer step count; there are no
    continuous clocks. Your transition relation should step by integer path length.
  - **T2 / NT (observer firewall):** nothing in this problem is continuous; there is
    no observer projection to worry about inside the QA dynamics. If you introduce
    any continuous metric (e.g. for analysis or visualization) it lives strictly
    outside the QA transition relation.
  - **S1 / S2 (no `**2`, no float state):** tickets, ids, and control-location encodings
    must all be `int` (or `Fraction` if you need exact rationals for some derived
    quantity). No `float`, no `numpy.random`, no `** 2`.
- Mutual exclusion is the load-bearing safety invariant; everything else (`TypeOK`,
  the inductive strengthening) exists to support a proof of it. The deliverable is
  not just "a system that happens to achieve mutual exclusion" — it is **a
  predicate named `MutualExclusion`, an inductive strengthening of it, and a
  machine-checkable argument that the strengthening is preserved by every
  transition**. That triple (safety predicate + inductive strengthening +
  preservation witness) is what must survive into the QA-native form.
- The Boulanger variant in the same directory is a stronger, more intricate
  relative (it replaces the non-atomic ticket read with a different mechanism); do
  not consult it.
