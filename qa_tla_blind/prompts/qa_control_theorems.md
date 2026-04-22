# QA Control Theorems — Blind Reproduction Prompt (POSITIVE CONTROL)

## Source (hidden from reproducing session)

The ground-truth artifacts are withheld. They are, for audit purposes only:

- `QA_CONTROL_THEOREMS.md`
- `Formalizing tuple drift in quantum-native learning/paper1_qa_control.tex`
- `qa_alphageometry_ptolemy/QARM_v02_Stats.tla` (machine-validated by TLC)
- `qa_alphageometry_ptolemy/ALL_INVARIANTS_VALIDATED.md`

The reproducer (a fresh Claude session with no access to the files above) must
derive the answers from the problem definition below.

## Problem domain

Fix a positive integer `N`. Define the square lattice of **capability tuples**:

```
Caps(N, N) := { (b, e) ∈ ℤ_{>0}² : 1 ≤ b ≤ N and 1 ≤ e ≤ N }
```

So `|Caps(N, N)| = N²`, indexing is **1-based** (states `{1, …, N}` — no zero
element), and `b`, `e` are the primitive integer coordinates of the QA tuple
`(b, e, d, a)` with derived coords `d = b + e` and `a = b + 2e` (the derived
coords are not required for this benchmark but must be respected if invoked —
they are not independent variables).

The object of study is the **directed transition graph**
`G_Σ = (Caps(N,N), E_Σ)` whose edges are the legal applications of generators
in `Σ`, with each generator treated as a partial function on `Caps(N,N)`.

## Generators

Four partial functions are defined on `Caps(N,N)`. An application is **legal**
iff the image lies inside `Caps(N,N)` and any additional per-generator legality
condition is satisfied; otherwise it is a classified **failure** (see below).

- **σ (growth / increment):**
  `σ(b, e) = (b, e + 1)`.
  Legal iff the image is in `Caps(N, N)`.

- **μ (coordinate swap):**
  `μ(b, e) = (e, b)`.
  Legal iff the image is in `Caps(N, N)`.

- **λ₂ (scale-up by 2):**
  `λ₂(b, e) = (2b, 2e)`.
  Legal iff the image is in `Caps(N, N)`.

- **ν (halve):**
  `ν(b, e) = (b/2, e/2)`.
  Legal iff both `b` and `e` are even **and** the image is in `Caps(N, N)`.

The full generator set of interest is `Σ = {σ, μ, λ₂, ν}`; sub-problems also
consider subsets such as `{σ, λ₂}`, `{σ, μ, λ₂}`, and so on.

## Failure modes

Every illegal application is assigned exactly one classification:

- **OUT_OF_BOUNDS** — the image `(b', e')` fails at least one of
  `1 ≤ b' ≤ N`, `1 ≤ e' ≤ N`.
- **PARITY** — a prerequisite parity condition on the inputs is not met
  (e.g., a halving generator applied when at least one coordinate is odd).

A given generator may produce only a subset of these failure modes; the
reproducer must decide which applies to which and justify.

## Questions to answer (derive closed forms in `N`)

All answers should be closed forms in `N` (or piecewise-closed, e.g. using
`⌊·⌋` for parity). "By BFS", "by enumeration", or "depends on N, run the
computer" are not acceptable answers — if the reproduction cannot get past
enumeration, note that and stop.

1. **SCC structure under full `Σ = {σ, μ, λ₂, ν}`** (and also the sub-case
   `Σ' = {σ, μ, λ₂}`, which is simpler and the paper's primary statement):
   - How many strongly connected components does `G_Σ'` have?
   - What is the maximum SCC size?
   - Classify the SCCs structurally (i.e. describe which tuples form which
     kind of SCC and why).

2. **Edge counts per generator.** For each `g ∈ {σ, μ, λ₂, ν}`, give a closed
   form for the number of legal directed edges `g` contributes to `G_Σ` on
   `Caps(N, N)`.

3. **Failure counts per generator, by failure mode.** For each `g` and each
   failure mode `f ∈ {OUT_OF_BOUNDS, PARITY}`, give a closed form for the
   number of states in `Caps(N, N)` at which `g` fails with mode `f`.
   (If a particular combination never fires, say so and justify.)

4. **Monotonicity under generator expansion.** Let `Σ₁ ⊆ Σ₂` be two generator
   sets over the same `Caps(N, N)`. How does `#SCC(G_{Σ₂})` compare to
   `#SCC(G_{Σ₁})`? State the direction and prove it. Does the same direction
   hold for `max|SCC|`?

5. **Structural question.** Which generators in `{σ, μ, λ₂, ν}` can
   participate in a directed cycle within `Caps(N, N)`, and which cannot?
   Justify each case using properties of the generator (not by computer
   search). Use this to explain the SCC structure in Q1.

## What is explicitly withheld from this prompt

- Closed-form values of `#SCC`, `max|SCC|`, any edge count, or any failure
  count.
- The direction of the monotonicity in Q4.
- Any suggested proof strategy (no potential-function hint, no orbit-structure
  hint, no per-generator cycle-participation hint).
- The names of any lemma or theorem from the source paper.
- Any predicate body, `Init`, or `Next` action from the TLA+ specification.

If the reproducer wants a hint of any of the above, they must derive it.

## Context hint (allowed)

This is QA's own machine-validated control-theoretic result (Jan 2026).
Closed forms are expected — QA framing, applied correctly, should produce
generator-relative edge/failure/SCC counts over this discrete generator
algebra, plus a monotonicity statement under generator expansion. An answer of
the form "depends on enumeration" indicates the reproduction has failed to
find the intended structure, not that the structure is absent.

## Reproducer instructions

Produce a single markdown artifact at `attempts/qa_control_theorems.md`
containing:

1. **QA state encoding.** State the `(b, e)` assignment and any derived-coord
   commitments. Confirm No-Zero (states `{1, …, N}`), the `d = b + e`,
   `a = b + 2e` convention, and that the boundary of `Caps(N, N)` is a hard
   constraint, not a soft penalty.
2. **Observer projection.** Identify what (if anything) crosses the Theorem
   NT firewall here. The problem is already fully discrete — name what the
   observer layer would be, or state explicitly that this problem lives
   entirely inside the QA discrete layer and no firewall crossing is required.
3. **Generator analysis.** For each of σ, μ, λ₂, ν, state whether it is a
   bijection on `Caps(N, N)`, whether it is invertible within `Σ`, and
   whether it can participate in a directed cycle. Justify.
4. **Closed-form derivations.** Answer Q1–Q5 above with closed forms and
   proofs. Do not enumerate. If you must case-split on parity of `N`, do so
   explicitly.
5. **QA-native classification.** If the SCCs or the reachable-state
   stratification admit a Cosmos / Satellite / Singularity reading (24-cycle
   / 8-cycle / fixed point), state it; if they do not, say so and explain
   why this problem doesn't load that structure.
6. **Honest-failure box.** If any of Q1–Q5 cannot be closed-form-answered,
   say which and why. A partial, honest answer beats a confident wrong one.

Use QA-native vocabulary where it applies: orbits, derived coords, No-Zero,
Theorem NT observer-projection firewall, integer path-time (T1), Cosmos /
Satellite / Singularity. Where it doesn't apply, say so — ornamental overlay
is a benchmark-failure tag.
