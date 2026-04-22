# PROOF SKETCH (Lane C) — oddN full-Σ SCC decomposition

**Author:** Lane C formal verifier (main-session fallback; see FORMAL_STATUS.json).
**Scope:** truth of the candidate claim. NOT a novelty judgment.
**Companion artifact:** `verify_extension.py` — exhaustive SCC enumeration.

## Setup (recap)

`Caps(N, N) = { (b, e) : 1 ≤ b, e ≤ N }`, 1-based. Generators:

- `σ(b, e) = (b, e+1)`     legal iff `e ≤ N−1`
- `μ(b, e) = (e, b)`       always in-bounds
- `λ₂(b, e) = (2b, 2e)`    legal iff `b, e ≤ ⌊N/2⌋`
- `ν(b, e) = (b/2, e/2)`   legal iff `b, e` both even

`Σ = {σ, μ, λ₂, ν}`. Let `G_Σ(N)` be the directed transition graph on `Caps(N, N)`.

## Claim

- `N = 1`: `#SCC = 1`, `max|SCC| = 1`. (Trivial.)
- `N ≥ 2` **even**: `#SCC(G_Σ(N)) = 1`, `max|SCC| = N²`.
- `N ≥ 3` **odd**: `#SCC(G_Σ(N)) = N + 1` with components
  1. one *inner* SCC equal to `Caps(N−1, N−1)` as a vertex set (size `(N−1)²`)
  2. `N − 1` *border* 2-cycles `{(N, k), (k, N)}` for `k ∈ {1, …, N−1}`
  3. one *singleton* SCC `{(N, N)}`

## Strategy: dual induction on N by parity

Prove by mutual induction: `Even(N) ⇒ Odd(N+1) ⇒ Even(N+2)`, with base cases `N=1, 2`.

The induction works because the odd-`N` decomposition isolates
`Caps(N−1, N−1)` as the inner SCC, and `N−1` is even — so the odd-`N` case
reduces to the even-`(N−1)` case applied to a proper subgraph.

## Base cases

- **N = 1.** `Caps(1,1) = {(1,1)}`. σ needs `e ≤ 0`: fails. μ maps `(1,1) → (1,1)`: self-loop. λ₂ needs `b ≤ 0`: fails. ν needs `b` even: fails. One node, one trivial SCC of size 1. ✓
- **N = 2.** Nodes: `(1,1), (1,2), (2,1), (2,2)`. Check explicitly:
  - σ-edges: `(1,1)→(1,2)`, `(2,1)→(2,2)`.
  - μ-edges: `(1,1)↔(1,1)`, `(1,2)↔(2,1)`, `(2,2)↔(2,2)`.
  - λ₂-edges: `(1,1) → (2,2)` (since `⌊2/2⌋ = 1`).
  - ν-edges: `(2,2) → (1,1)`.
  - From `(1,1)`: reaches `(1,2)` (σ), `(2,2)` (λ₂), `(2,1)` (μ∘σ).
    Return paths: `(2,2) → (1,1)` (ν); `(2,1) → (2,2) → (1,1)` (σ then ν);
    `(1,2) → (2,1) → (2,2) → (1,1)` (μ then σ then ν).
  - All four nodes in one SCC. ✓ `max|SCC| = 4 = 2²`.

## Even-N inductive step (`N ≥ 4`, `N` even)

**Goal:** `G_Σ(N)` has exactly one SCC covering all `N²` nodes.

**Lemma E1 (ν/λ₂ round-trip).** For every `(b, e)` with `b, e ≤ N/2`:
`λ₂(b, e) = (2b, 2e)` is legal, and `ν(2b, 2e) = (b, e)` is legal (since
`2b, 2e` are even). So `(b, e) ↔ (2b, 2e)` in `G_Σ(N)`, always.

**Lemma E2 (σ forward within each row).** Within each fixed `b`, σ gives a
forward path `(b, 1) → (b, 2) → … → (b, N)`. The σ-reverse direction does
NOT exist directly — but within a strongly-connected component that contains
(b, N), μ-swaps plus ν/λ₂ recover connectivity. We show this below.

**Lemma E3 (anchor).** `(1, 1)` and `(N, N)` are in the same SCC for even N.
Proof: repeatedly λ₂ from `(1,1)`: `(1,1) → (2,2) → (4,4) → … → (2^k, 2^k)`
where `2^k ≤ N`. The *largest* such is `2^{⌊log₂ N⌋}`. From there, climb σ
in `e` to `(2^k, N)` (if `2^k ≠ N`), then μ-swap to `(N, 2^k)`, then σ up
(no — e=2^k is not N; σ legal). Wait — this argument is cleaner reversed:

**Cleaner anchor (E3').** For even N, every state `(b, e)` is in the SCC
containing `(1, 1)`. Proof sketch:

1. *Forward reachability from (1,1):* σ lets us set e to any value in
   `{1, …, N}`. μ lets us swap b and e. λ₂ doubles. By composing, we can
   reach every state: `(1,1) →σ* (1, e) →μ (e, 1) →σ* (e, b) →μ (b, e)`.
   This uses only σ and μ; every target `(b, e)` is reachable from `(1,1)`.
2. *Reverse reachability back to (1,1):* from any `(b, e)`, we need a path
   to `(1, 1)`. Use ν when possible; otherwise we need to reduce one coord
   without using ν.

   Consider `(b, e)` with either coord odd. Without ν, we cannot halve.
   BUT: with both σ and μ we can **increase** coords and swap them, then
   arrange both to be even. Specifically, if `b` is odd and `< N`, we have
   no generator that increases `b`. Hmm.

   Actually this is where the even-N condition matters. The claim is
   that the ENTIRE graph is one SCC — but for reverse reachability we need
   to show every state has a path back to (1,1). Let me rebuild this.

3. *Alternative reverse argument.* For even N, consider the path-trace:

   From `(b, e)`, at least one of the following holds:
     - if `b` odd and `e` odd: σ to make e even (if `e < N`), giving
       `(b, e+1)` with `e+1` even. Else if `e = N`, μ-swap to `(N, b)` =
       `(even, odd)`, and apply ν-path once b has been made even via λ₂.

   This gets tangled. The cleaner fact: even-N allows "diagonal descent via
   (1, 1)". We need to show that the strongly-connected component containing
   (1, 1) has every node.

Rather than fully derive (E3') here, observe that the enumeration in
`verify_extension.py` confirms `#SCC = 1` for N = 2, 4, 6, 8, 10, 16, 20,
30, 64. This is an **empirical verification** of the even-N case; the
structural proof is nontrivial and is deferred to PROOF_SKETCH v2. See
"Completeness gap" below.

## Odd-N inductive step (`N ≥ 3`, `N` odd)

**Step 1: The border pairs are 2-cycles.**

Fix `k ∈ {1, …, N−1}`. Consider `(N, k)`:

- σ: `(N, k+1)` if `k < N−1`, else `(N, N)` if `k = N−1`. Legal.
- μ: `(N, k) → (k, N)`. Legal.
- λ₂: requires `N ≤ ⌊N/2⌋`. For N odd ≥ 3, `⌊N/2⌋ = (N−1)/2 < N`. Fails.
- ν: requires N even. N is odd. Fails.

So out-edges from `(N, k)`: σ to `(N, k+1)` and μ to `(k, N)`.

Consider `(k, N)`:

- σ: requires `e = N ≤ N−1`. Fails.
- μ: `(k, N) → (N, k)`. Legal.
- λ₂: requires `N ≤ ⌊N/2⌋`. Fails (odd N).
- ν: requires N even. Fails.

So out-edges from `(k, N)`: μ to `(N, k)` only.

Therefore `{(N, k), (k, N)}` with two μ-edges is **mutually reachable** and
has no other outgoing edges to other 2-cycles via μ or ν or λ₂. The σ-edge
`(N, k) → (N, k+1)` leaves the pair — but does `(N, k+1)` have a path back
to `(N, k)`? `(N, k+1)` only reaches `(N, k+2)` and `(k+1, N)`. Neither has
any generator legal that could reduce either coord back to `(N, k)`: σ only
increases e, μ swaps, λ₂ is illegal at b=N for odd N, ν is illegal at b=N
for odd N. So no path back. **The pair is a maximal SCC of size 2.** ✓

**Step 2: The corner `(N, N)` is a singleton.**

At `(N, N)`:

- σ: `e = N`, fails.
- μ: `(N, N) → (N, N)`. Self-loop.
- λ₂: `b = N > ⌊N/2⌋` for odd N ≥ 3. Fails.
- ν: `b = N` odd. Fails.

No outgoing edges except the μ self-loop. So `(N, N)` is a sink, size-1 SCC. ✓

*Predecessors of `(N, N)`:* `(N, N−1)` via σ, and the μ self-loop. The σ
predecessor does not contradict the singleton status — `(N, N−1)` is in a
different SCC (it's in the `{(N, N−1), (N−1, N)}` 2-cycle), and its σ-edge
to `(N, N)` is just an inter-SCC edge.

**Step 3: The inner set is `Caps(N−1, N−1)` and is one SCC.**

Claim: the induced subgraph of `G_Σ(N)` on `Caps(N−1, N−1)` is **exactly**
the graph `G_Σ(N−1)` with parameter `N−1` (for `Σ = {σ, μ, λ₂, ν}`).

Proof: for each generator, check that it stays within `Caps(N−1, N−1)` under
the same legality conditions:

- σ: legal on `Caps(N−1, N−1)` iff `e ≤ N−2`, which is the `σ`-legality
  condition for parameter `N−1`. And `(b, e+1)` with `e+1 ≤ N−1` stays in
  `Caps(N−1, N−1)`. ✓
- μ: always legal, maps into `Caps(N−1, N−1)` since both coords stay ≤ N−1. ✓
- λ₂: legal iff `b, e ≤ ⌊N/2⌋`. For odd N, `⌊N/2⌋ = (N−1)/2 = ⌊(N−1)/2⌋`
  (since N−1 is even). Same condition as parameter `N−1`. The image
  `(2b, 2e)` has `2b ≤ N−1`, `2e ≤ N−1`, stays in `Caps(N−1, N−1)`. ✓
- ν: legal iff both coords even. Image `(b/2, e/2)` has both ≤ (N−1)/2 ≤
  N−1. ✓

So the induced subgraph is isomorphic to `G_Σ(N−1)` with `N−1` even. By
the even-N claim (empirically verified, and inductively assumed for N−1 <
N), `G_Σ(N−1)` has one SCC of size `(N−1)²`. Therefore the inner set is
one SCC in `G_Σ(N)`. ✓

**Step 4: The inner SCC is disjoint from the border pairs and the corner.**

The border pairs and corner lie outside `Caps(N−1, N−1)` (they all have
some coord equal to N). So they are disjoint from the inner SCC as node
sets. ✓

**Step 5: No inner-border strongly-connecting edges exist.**

Consider an edge from inner `(b, e)` to border or corner. σ can take
`(b, N−1) → (b, N)` — exits inner. μ on inner stays inner (both coords
≤ N−1). λ₂ on inner stays inner (image has both ≤ N−1). ν on inner stays
inner. So ONLY σ exits inner, and only to the `(b, N)` side of a border
pair.

From that border node `(b, N)`, we showed in Step 1 that only μ is legal,
going to `(N, b)`, which has σ-paths forward and μ back to `(b, N)` — no
way to return to `(b, e)` inner. So the σ-exit is a one-way edge. It does
not merge the inner SCC with any border pair. ✓

Inbound: σ-predecessor of `(b, N−1)` is `(b, N−2)` (inner). No border or
corner node sends an edge into inner — we verified their outgoing edges
are all within border or corner. ✓

**Step 6: Assembly.**

Counting: `1 (inner) + (N−1) (border pairs) + 1 (corner) = N + 1` SCCs.
Sizes: `(N−1)² + 2(N−1) + 1 = N²`. ✓

This completes the odd-N case, **conditional on the even-N claim for `N−1`**.

## Completeness gap

The even-N case's structural proof (every state reachable to and from every
other) is sketched but not rigorously completed in §Even-N. The current
status is:

- **Empirical:** enumeration passes for N ∈ {2, 4, 6, 8, 10, 16, 20, 30, 64}.
- **Partial structural argument:** the ν/λ₂ round-trip (E1) handles the
  even-coord subgrid. The σ/μ forward reachability (E2, E3' step 1) handles
  every state being reachable from (1, 1). The REVERSE direction (every
  state reaches (1, 1)) requires a cleaner odd-coord argument.

A completed v2 proof would:

1. Partition `Caps(N, N)` by parity class of each coord: (even, even),
   (even, odd), (odd, even), (odd, odd).
2. Show `(1, 1)` reaches any (odd, odd) state via σ only; swaps to (even, odd)
   via μ composition; etc. (Forward direction.)
3. Show every (even, even) state returns to (1, 1) via ν-halving repeatedly
   until both coords are 1.
4. Show every (odd, odd) state can reach an (even, even) state via σ (climbing
   e to an even value) + μ (swap) + σ (climbing the now-odd e to even again).
   For N even, N is a valid σ-target for e, and intermediate even values exist.
5. Conclude every state reaches (1, 1), closing the SCC.

This argument hinges on *N being even so that σ can reach N*, which is the
even-coord value (1,1) needs as a μ-partner in the path `(b, e) →σ (b, N) →μ
(N, b) → … → (b', 1) →σ (b', …)`. When N is odd, σ cannot reach N from e=1
of the relevant parity, breaking the reachability chain — exactly what
forces the border 2-cycles and singleton to be separate SCCs.

## Status

- **Enumerative verification:** `verify_extension.py` passes for 17 N values
  up to N=65, covering the claim over a span of 2 orders of magnitude.
  All edge counts match `|σ|=N(N−1)`, `|μ|=N²`, `|λ₂|=|ν|=⌊N/2⌋²`; failure
  counts match `σ_OOB=N`, `λ₂_OOB=ν_PARITY=N²−⌊N/2⌋²`.
- **Structural proof:** odd-N case complete, conditional on even-N. Even-N
  case has E1 (ν/λ₂ round-trip) and E3' part 1 (forward from (1,1)) but
  the reverse direction is sketched not closed.
- **Net status:** `proof_sketch_provided` with a flagged gap in the even-N
  reverse direction. The odd-N case — the more interesting and more novel-
  looking part — is structurally complete modulo the recursive even-N call.

## What would close the gap

A ~1-page even-N reverse-reachability lemma, case-split on coord parity,
exploiting that σ can reach e=N and μ can swap b↔e when N is even. This is
elementary but tedious; deferred to v2.
