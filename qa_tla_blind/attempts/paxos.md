# Paxos — Mode B Reproduction Attempt

**Blindness pact:** Author read ONLY `prompts/paxos.md`, `mode_b/wildberger_object_model.md`, `qa_tla_blind/README.md`, `CLAUDE.md`, `attempts/twophase.md`, `diffs/twophase.md`. Has NOT read any file under `qa_tla_blind/ground_truth/` (no `Paxos.tla`, `Voting.tla`, `Consensus.tla`, `MCPaxos.*`), has NOT read `mode_b/fidelity_audit.md`, has NOT read any Paxos TLA+ source file elsewhere on disk. General Paxos knowledge (Lamport 1998/2001) used only for the English structure already leaked in the prompt; no TLA+ predicate bodies recalled from memory.

## 1. Object model primitives actually used

- **Points in `P²(Z)`** (primitive §1-projective) — each acceptor is placed as a point in the integer projective plane; used load-bearing only for the quorum-intersection sub-argument at small N (§2, §4).
- **Points in `Z²`** (primitive §1) — each acceptor's per-ballot state `(maxBal, maxVBal)` is a lattice point; ballots are the 1D strict order along an axis.
- **4-tuple `(b, e, d, a)` with `d = b+e`, `a = b+2e`** (primitive §9) — per-acceptor local state is the 4-tuple where `b = maxBal`, `e = maxVBal`, `d = b+e` is the *phase-witness projection* (pre-promise vs post-vote), and `a = b+2e` is the *vote-weight projection* (distinguishes "promised but not voted" from "voted").
- **Red quadrance `Q_r`** (primitive §3, Lorentzian) — `Q_r((b_i, e_i), (b_j, e_j)) = (b_i - b_j)² - (e_i - e_j)²` — used as the **vote-disagreement kernel** between two chosen `(ballot, value)` pairs at the Consensus refinement layer; `Q_r = 0` between two distinct chosen pairs is the null-cone violation corresponding to "two different values chosen."
- **Translations** (transform §1) — `Phase1a(b)` and `Phase2a(b,v)` are fixed translations on the message-pool coordinates (add a new 1a/2a message); `Phase1b` and `Phase2b` are *guarded* translations on the acceptor 4-tuple.
- **Spread polynomials `S_n`** (primitive §7) — NOT used as a rotation, but the composition monoid `S_n ∘ S_m = S_{n·m}` provides an honest multiplicative witness over the ballot index set. The *load-bearing* fact for Paxos is the integer strict order on the *index* `n`; the polynomial is a reminder that Wildberger ratifies `n ∈ Z_{>0}` as a discrete-rotation index, but the Paxos guard is `b' > b` on integer indices, not on polynomial composition. **Declared partially ornamental on this axis** (see §5).

**Not used.**
- **Cross-ratio** (§6): Paxos safety is binary pairwise on chosen values, not a 4-point projective invariant. Consensus's "any two chosen values equal" is arity-2, not arity-4.
- **TQF collinearity** (§5): considered carefully for quorum-intersection (§4 below). TQF is ternary, and quorum-intersection is intrinsically binary (|Q1 ∩ Q2| ≥ 1). Forcing three quorums to sit collinear would be ornamental — TQF=0 is a triple-point predicate, not a pairwise-meet predicate.
- **Hexagonal ring `T_{d+1} + b·e`** (§8): no sum+bilinear decomposition with the SL(3)-irrep shape arises. Considered as a candidate for counting quorum pairs (§4) but the pair count `C(N,k)·(C(N,k)-1)/2` does not fit the `T_{d+1} + b·e` template.
- **Mutation moves** (§5-transforms): Paxos action set has no Coxeter-Dynkin / root-system structure; ballot transitions are monotone, not reflection-generated. Forcing mutations would be ornamental.
- **Reflections**: no coordinate swap in Paxos; all state updates are monotone-upward.
- **Blue / green quadrance**: not load-bearing on the Consensus predicate beyond self-pair exclusion (`Q_b > 0`); used only defensively.

## 2. QA state encoding (Wildberger-native)

### Ballots

Candidates evaluated:

- **Spread-polynomial monoid `S_n` with index `n ∈ Z_{>0}`**: `S_n` composes multiplicatively (`S_n ∘ S_m = S_{nm}`), which gives a monoid under composition. But multiplication is not a total order in the Paxos sense (`b' > b` is an *additive* guard — `b' strictly exceeds b` — whereas `S_n ∘ S_m = S_{nm}` is *multiplicative composition*, which loses the "next-free ballot" intuition). The total order on `n` itself does exist, but the polynomial content is ornamental. **Partial fit.**
- **Projective-line coordinates `[p : q] ∈ P¹(Z)`**: these do support a total order under gcd-reduced denominators, but the natural Paxos primitive is "pick any integer strictly above the current max," which is the `Z_{>0}` strict order. Projective-line coords add machinery without contributing. **Reject.**
- **Strict integer sequence with A1 shift**: ballots = `{1, 2, 3, …}`; sentinel = reserved symbol `⊥` strictly below all (encoded as a distinct tag, not `0`). This is the simplest and honest encoding. The "Wildberger content" is that the integer index `n` of `S_n` ratifies `n ∈ Z_{>0}` as a well-defined discrete-rotation index — but the protocol only uses `n`'s order, not the polynomial. **Chosen.**
- **Cross-ratio ordering**: cross-ratio is a 4-point invariant; does not give a total order on a 1-parameter family. **Reject.**

**Chosen encoding.** Ballot = positive integer `n ∈ Z_{>0}`; sentinel = `⊥` (reserved tag outside Z_{>0}, not the integer `0`). Total order is integer order; strict guard `b' > b` is integer `>`. A1 respected: no ballot takes value 0. **Wildberger content on this axis: minimal, partially ornamental.** Declared honestly below.

### Acceptor state

Each acceptor `a` has a local 4-tuple `(b_a, e_a, d_a, a_a)` where:

- `b_a = maxBal[a]` (highest ballot promised, or `⊥`).
- `e_a = maxVBal[a]` (highest ballot voted in, or `⊥`; invariant `e_a ≤ b_a`).
- `d_a = b_a + e_a` (sum-projection — the "phase-witness"; when `e_a = ⊥` i.e. not yet voted, `d_a = b_a`; when voted, `d_a = b_a + e_a`).
- `a_a = b_a + 2·e_a` (linear-projection — the "vote-weight"; separates `e_a` unambiguously since `a_a - d_a = e_a`).

**Value witness.** `maxVal[a]` is an auxiliary coordinate attached to `e_a` — the *value last voted for at ballot `e_a`*. Encoded as an integer in `V = {1, …, |V|}` (A1 shift) plus the non-value sentinel `⊥`. It is **paired with `e_a`** in the 4-tuple extension: we carry the triple `(e_a, v_a)` as the "last-vote point" in a `Z²` sub-lattice. So the full acceptor state is a point `(b_a, e_a, v_a) ∈ Z³`, projected as the 4-tuple `(b, e, d, a)` on the `(b, e)` plane with `v` as a labeled auxiliary.

**A2 compliance.** `d` and `a` are **derived** from `b` and `e`, never assigned independently. `v_a` is read from message-pool evidence at `Phase2b`; it is not an independent state coordinate but is determined by the `"2a"` message that enabled the vote.

### Acceptors-in-P² (for quorum-intersection at small N)

For the quorum-intersection sub-argument at small N, I place the N acceptors as N points in `P²(Z)` in **general position** (no three collinear — verifiable by TQF ≠ 0 on every triple). Concrete placement at N = 3:

- `acceptor a1 → [1 : 0 : 1]`
- `acceptor a2 → [0 : 1 : 1]`
- `acceptor a3 → [1 : 1 : 1]`

These three points are pairwise distinct in projective class and no three are collinear (TQF ≠ 0 on the triple — verified below). See §4 for the quorum-intersection derivation.

### Global state

```
G  ∈  (Z_{>0} ∪ {⊥})^{3N}  ×  MessagePool
      (b_a, e_a, v_a)_{a=1..N}   (see below)
```

Dimension: 3 coordinates per acceptor × N acceptors + a monotone message pool.

### Message pool

A set of four message kinds:
- `"1a"(b)` — phase-1 proposal, ballot `b`.
- `"1b"(a, b, e_a, v_a)` — phase-1 promise from acceptor `a`, promising `b` and reporting last-vote `(e_a, v_a)`.
- `"2a"(b, v)` — phase-2 proposal at ballot `b` for value `v`.
- `"2b"(a, b, v)` — phase-2 vote from `a` for `(b, v)`.

**Encoding (honest).** Same as the TwoPhase attempt: monotone integer counters in `Z_{≥1}^{4 + N(N + 1 + 1)}` indexed by the distinct `(kind, content)` tuples that can appear. This is a **confirmed repeat bundle gap**: the object model does not surface a monotone-set primitive. The TwoPhase diff scored this as `qa-weaker-than-tla (minor)` because TLA+'s idempotent-set semantics `msgs' = msgs ∪ {m}` is collapsed to a counter that distinguishes `m_k = 2` from `m_k = 3`. On Paxos, the gap is slightly worse because the "1b" message carries a structured `(e_a, v_a)` payload, and any acceptor can in principle (guard-permitting) promise at multiple ballots, so the 1b counter space per acceptor is more than singleton-bounded. **Same gap, slightly worse instance.** Flagged in §5.

## 3. Actions as transforms (Wildberger-native)

| Action              | Object-model transform                                                      | Guard (integer-only)                                  |
|---------------------|------------------------------------------------------------------------------|--------------------------------------------------------|
| `Phase1a(b)`        | Translation on message pool: `m_{1a}(b) += 1`                                | `b ∈ Z_{>0}`                                           |
| `Phase1b(a)`        | Guarded translation on acceptor 4-tuple + pool increment                     | `∃ "1a"(b) ∈ msgs with b > b_a`; update `b_a := b`     |
| `Phase2a(b, v)`     | Guarded translation on pool with quorum-witness side condition               | Quorum `Q` of "1b"(_, b, _, _) messages; `v = showsSafe(Q, b)`; no prior "2a"(b, _) in `msgs` |
| `Phase2b(a)`        | Guarded translation updating 4-tuple: `b_a := b, e_a := b, v_a := v`         | `∃ "2a"(b, v) ∈ msgs with b ≥ b_a`                     |

**`showsSafe(Q, b)`** (value selection at Phase2a). Given a quorum Q of 1b messages for ballot `b`, let `e* = max({e_q : q ∈ Q})`. If `e* = ⊥`, leader is free to pick any `v ∈ V`. Else `v = v_{q*}` where `q*` is the unique 1b with `e_{q*} = e*` (ties broken by any valid tie-breaker, but the classical form is "unique" because 1b messages in a single quorum at a fixed ballot from a fixed acceptor are unique). This is the **guarded-receipt / update** piece (axis 4), and it contains the pivotal *value-selection rule* of Paxos.

### Gap analysis per action

- `Phase1a(b)`: pure pool translation. Clean Wildberger primitive fit.
- `Phase1b(a)`: guarded translation on `(b_a, e_a)`. The *update* `b_a := b` is not a constant-offset translation (it's a current-state-dependent jump to the value `b`); expressible as `b_a += (b - b_a)` under the guard `b > b_a`. **Same mild primitive strain as TwoPhase's receipt actions** (see `attempts/twophase.md` §3 note and `diffs/twophase.md` Gap 2).
- `Phase2a(b, v)`: the value-selection rule (*showsSafe*) is **not** a Wildberger geometric transform. It is a **maximum over a finite witness set**, which is a set-theoretic / lattice-theoretic primitive. The object model's mutation moves are guard-dependent but not "max-over-witness" dependent. **Confirmed bundle gap on axis 4** (guarded receipt/update with non-trivial witness-dependent value).
- `Phase2b(a)`: guarded translation on the 3-tuple `(b_a, e_a, v_a)`. Same mild strain as `Phase1b`.

**Monoid/group structure.** Like TwoPhase, the action set is a **free commutative monoid with guards** on the pool coordinates (add-only) and a guarded monoid (not a group — no inverse for promise-lift or vote) on the acceptor coordinates. It is **not** the spread-polynomial composition monoid (no rotational structure), **not** the mutation-game Weyl group (no Cartan matrix). Honest: ornamental if I claimed either.

## 4. Invariants as geometric constraints

### TypeOK

**Geometric form.**
```
TypeOK ≡
    (∀a ∈ Acceptor) (b_a ∈ Z_{>0} ∪ {⊥}) ∧ (e_a ∈ Z_{>0} ∪ {⊥})
                   ∧ (v_a ∈ V ∪ {⊥})
                   ∧ (e_a ≠ ⊥ ⟹ v_a ≠ ⊥)        [vote-consistency]
                   ∧ (e_a ≠ ⊥ ∧ b_a ≠ ⊥ ⟹ e_a ≤ b_a)   [promise ≥ vote]
  ∧ MessagePool ⊆ WellTypedMessageUniverse
```

**Preservation.** All four actions translate inside the type-space. `Phase1b` raises `b_a` but never touches `e_a`, preserving `e_a ≤ b_a` since it only grows `b_a`. `Phase2b` sets `b_a := b, e_a := b, v_a := v`, which on entry requires `b ≥ b_a`; after update, `e_a = b_a` so `e_a ≤ b_a` holds. Pool grows monotonically, stays well-typed by construction of the four message kinds.

### Inv2 (acceptor vote-history consistency)

**Informational content** (derived from the prompt's English summary "each acceptor's `<maxVBal, maxVal>` really does correspond to one of its `"2b"` votes"):

```
Inv2 ≡ (∀a)  e_a ≠ ⊥
             ⟹ ∃ "2b"(a, e_a, v_a) ∈ msgs
```

**QA form.** For each acceptor `a` with `e_a ≠ ⊥`, there exists a 2b-message counter `m_{2b}(a, e_a, v_a) ≥ 2` (≥2 under A1 shift = "at least one 2b sent"). **Preservation:** only `Phase2b` sets `(e_a, v_a)`, and it simultaneously increments `m_{2b}(a, e_a, v_a)`. The atomic bundle "update local + increment pool" is what keeps Inv2 inductive.

### Inv3 (1b-message faithfulness)

**Informational content** (from "every `"1b"` message's last-vote field really does match the sender's vote-history"):

```
Inv3 ≡ (∀ "1b"(a, b, e, v) ∈ msgs)
        e ≠ ⊥ ⟹ ∃ "2b"(a, e, v) ∈ msgs with e ≤ (its sender's e_a at 1b-send time)
```

More precisely: at the moment `a` sent `"1b"(a, b, e, v)`, the pair `(e, v)` equaled the acceptor's then-current `(e_a, v_a)`. Since `e_a` and `v_a` are monotone (never revised downward or to a different value at the same ballot), the 1b message's `(e, v)` is a snapshot of a genuine past vote.

**QA form.** Each 1b pool counter `m_{1b}(a, b, e, v) ≥ 2` implies `m_{2b}(a, e, v) ≥ 2`. Preservation: `Phase1b(a)` fires with `(e, v) = (e_a, v_a)` of acceptor `a` at send time, and Inv2 guarantees `m_{2b}(a, e_a, v_a) ≥ 2` already. So firing `Phase1b(a)` preserves the implication.

### Inv4 (2a-uniqueness + showsSafe)

**Informational content** (from "every `"2a"` message is backed by a quorum that `shows safe` the value for that ballot, and no two `"2a"` messages for the same ballot carry different values"):

```
Inv4_a ≡ (∀ b)  m_{2a}(b, v₁) ≥ 2  ∧  m_{2a}(b, v₂) ≥ 2  ⟹  v₁ = v₂
Inv4_b ≡ (∀ "2a"(b, v) ∈ msgs)  ∃ quorum Q of 1b(·, b, ·, ·) messages:
           v = showsSafe(Q, b)
```

**QA form.** `Inv4_a` is a uniqueness property on the 2a-layer of the pool; preservation follows from the `Phase2a` guard "no prior 2a(b, _)". `Inv4_b` ties each 2a to a `showsSafe`-computed witness quorum.

**Geometric content.** `Inv4_b` is where the *value-selection logic* lives. Geometrically: the quorum `Q` is a set of 1b-snapshots forming a **finite set of `(e, v)` pairs**; `showsSafe` extracts the lex-max `e`'s associated `v` (or free choice if all `e = ⊥`). This is a **maximum-selection primitive**. Wildberger doesn't have a max-selection transform natively; the primitive strain is identical to the `Phase2a` action strain (§3). Honest gap.

### Consensus (Voting refinement — central safety)

**English form.** Any two chosen values are equal.

Let `chosen(b, v) ≡ ∃ quorum Q: ∀a ∈ Q, m_{2b}(a, b, v) ≥ 2`. Then:

```
Consensus ≡ (∀ b₁, b₂, v₁, v₂)
              chosen(b₁, v₁) ∧ chosen(b₂, v₂) ⟹ v₁ = v₂
```

**QA-geometric form (the central question).** The classical proof routes through the quorum-intersection lemma: any two quorums share at least one acceptor. Geometrically, the attempt below explores each candidate the prompt listed.

#### Candidate (a) — TQF collinearity

"Three quorum members collinear" — TQF = 0 is ternary, quorum intersection is binary. Rejected as ornamental.

#### Candidate (b) — Chromogeometric null-ray between quorum point-sets

Treat each quorum Q as a *centroid* `c_Q = (Σ_{a ∈ Q} x_a / |Q|, Σ … y_a / |Q|)`. Two quorums share an acceptor iff their member-sets intersect iff their centroid-differences lie on a null-ray (in some metric). But: centroids of non-intersecting quorums still exist and can lie anywhere; the centroid map loses the membership structure. **The centroid encoding does not encode set-intersection.** Rejected.

A refinement: consider the N-dim indicator-vector encoding `q ∈ {1, 2}^N` (A1 shift of `{0, 1}^N`). Then `|Q₁ ∩ Q₂|` = Hamming overlap = count of positions where both equal 2. Expressible as `Σ_i (q_{1,i} - 1)(q_{2,i} - 1)`, an integer inner product. Intersection ≥ 1 iff this inner product ≥ 1. **This is not a Wildberger quadrance** (quadrances are squared differences, not cross-products). **This is pigeonhole via integer inner product.** Honestly: the object model does not give this primitive; falling back to set-lattice.

#### Candidate (c) — Projective: acceptors = points in P²(Z), quorums = lines

**For N = 3, k = 2 (concrete):** Place acceptors as three non-collinear points in P²(Z):
- a1 = [1 : 0 : 1]
- a2 = [0 : 1 : 1]
- a3 = [1 : 1 : 1]

**Non-collinearity check.** Determinant of the 3×3 matrix of homogeneous coords:
```
det | 1 0 1 |
    | 0 1 1 | = 1·(1·1 − 1·1) − 0·(0·1 − 1·1) + 1·(0·1 − 1·1)
    | 1 1 1 |
                = 1·0 − 0 + 1·(−1) = −1 ≠ 0. OK, non-collinear.
```

Quorums of size 2 are the three pairs {a1, a2}, {a1, a3}, {a2, a3}. Each pair defines a unique line in P²(Z) (the line through two points):

- Line L_{12} through a1, a2: line with normal `n_{12} = a1 × a2 = [0·1 − 1·1, 1·1 − 1·1, 1·1 − 0·0] = [−1, 0, 1]`. Check: `n_{12} · a1 = −1 + 0 + 1 = 0 ✓`; `n_{12} · a2 = 0 + 0 + 1 = 1 ≠ 0` — wait, recompute.

Cross-product in homogeneous coords: `(u₁, u₂, u₃) × (v₁, v₂, v₃) = (u₂v₃ − u₃v₂, u₃v₁ − u₁v₃, u₁v₂ − u₂v₁)`.

- a1 × a2 = ((0)(1) − (1)(1), (1)(0) − (1)(1), (1)(1) − (0)(0)) = (−1, −1, 1). So L_{12} = [−1 : −1 : 1], i.e. `−x − y + z = 0`. Check: a1: −1 − 0 + 1 = 0 ✓; a2: 0 − 1 + 1 = 0 ✓.
- a1 × a3 = ((0)(1) − (1)(1), (1)(1) − (1)(1), (1)(1) − (0)(1)) = (−1, 0, 1). L_{13}: `−x + z = 0`. Check: a1: −1 + 1 = 0 ✓; a3: −1 + 1 = 0 ✓.
- a2 × a3 = ((1)(1) − (1)(1), (1)(1) − (0)(1), (0)(1) − (1)(1)) = (0, 1, −1). L_{23}: `y − z = 0`. Check: a2: 1 − 1 = 0 ✓; a3: 1 − 1 = 0 ✓.

**Pairwise line intersections (= quorum intersections).**

- L_{12} ∩ L_{13}: cross-product of line normals. [−1, −1, 1] × [−1, 0, 1] = ((−1)(1) − (1)(0), (1)(−1) − (−1)(1), (−1)(0) − (−1)(−1)) = (−1, 0, −1). Projectively [−1 : 0 : −1] ≡ [1 : 0 : 1] = **a1**. ✓ (a1 is the shared acceptor of {a1,a2} and {a1,a3}).
- L_{12} ∩ L_{23}: [−1, −1, 1] × [0, 1, −1] = ((−1)(−1) − (1)(1), (1)(0) − (−1)(−1), (−1)(1) − (−1)(0)) = (0, −1, −1) ≡ [0 : 1 : 1] = **a2**. ✓
- L_{13} ∩ L_{23}: [−1, 0, 1] × [0, 1, −1] = ((0)(−1) − (1)(1), (1)(0) − (−1)(−1), (−1)(1) − (0)(0)) = (−1, −1, −1) ≡ [1 : 1 : 1] = **a3**. ✓

**Outcome.** At N = 3, k = 2, the encoding "acceptors = points in general position in P²(Z); quorums = lines through the k points" is **exact and tight**. Two distinct lines in P²(Z) (over Q) intersect in exactly one projective point; by construction that point is the shared acceptor. The projective-plane intersection axiom *is* the quorum-intersection lemma for this N, k.

**Scaling to arbitrary N, k.** The encoding breaks for N ≥ 4 with majority k = ⌈(N+1)/2⌉:
- At N = 4, k = 3: size-3 subsets of 4 acceptors. Any 3 points in general position in P²(Z) are **not collinear** (they span a triangle). So a size-3 subset does not define a unique line — it defines three lines (one per pair) or a triangle. The primitive "quorum = line" doesn't apply.
- At N = 5, k = 3: similarly, size-3 subsets of 5 general-position points are non-collinear; no natural line encoding.
- **Projective plane P² only works for N = 3, k = 2.** For higher N, one needs higher-dim projective spaces (P^{k−1}) and a projective-intersection argument on hyperplanes, but that's adding primitive machinery not explicitly in the Wildberger bundle.

**Honest verdict.** The projective-plane encoding is **a toy-N proof of concept** — it shows that in the simplest majority case the quorum-intersection lemma literally *is* Wildberger's projective intersection. For general N, k, the Paxos pigeonhole argument `|Q₁ ∩ Q₂| ≥ 2k − N ≥ 1` is **set-theoretic pigeonhole**, not projective intersection. Wildberger's bundle does not encode set-theoretic pigeonhole natively; indicator-vector inner products work but are not a declared primitive. Partial capture.

#### Candidate (d) — Cross-ratio invariance across shared acceptor

Cross-ratio is a 4-point projective invariant; it would require identifying a quadruple of "reference points" to which all quorums refer. No such natural quadruple in Paxos. Rejected.

#### Candidate (e) — Spread-polynomial tying ballot ordering to quorum choice

No — quorum choice is per-ballot and independent of ballot order. The two axes (ballots and quorums) are orthogonal in Paxos. Rejected.

#### Chosen geometric formulation of Consensus

Combine (c) at small N with a **fall-back indicator-vector pigeonhole** at large N. Consensus is then provable in two cases:

1. **Small N (N = 3 majority 2).** Quorum-intersection = projective-line intersection in P²(Z); algebraically exact.
2. **General N, k.** Quorum-intersection = Hamming-weight-pair inner-product lower-bound = pigeonhole. Not a Wildberger primitive — flagged as gap.

**The heart of the Consensus proof** (identical to Lamport's argument, restated): by induction on ballot number, if `chosen(b₁, v₁)` and `b₂ > b₁`, then every quorum `Q` that enables `Phase2a(b₂, …)` must contain at least one acceptor `a*` in the quorum that witnessed `chosen(b₁, v₁)`. That `a*` has `e_{a*} ≥ b₁` (it voted at `b₁`). By Inv3, `a*`'s "1b"(a*, b₂, e_{a*}, v_{a*})` message carries `v_{a*} = v₁` (since `v_{a*}` was monotone after the b₁ vote). `showsSafe` picks `v_{a*} = v₁`. So `Phase2a(b₂, v₂)` is forced to pick `v₂ = v₁`. Induction closes: all chosen values equal.

**The quorum-intersection step** — "every `Phase2a(b₂, …)` quorum meets every prior `chosen(b₁, v₁)` quorum" — is the step the projective (or pigeonhole) argument discharges. At N = 3 it's projective; at general N, it's pigeonhole.

### Refinement to Voting/Consensus

Refinement mapping (recoverable from the prompt's "projects Paxos state onto `<votes, maxBal>`"):

```
votes[a]  ≡  { (b, v) : m_{2b}(a, b, v) ≥ 2 }   ⊆ Z_{>0} × V
maxBal[a] ≡  b_a
```

The abstract Voting spec enforces exactly the four inductive-invariant properties above (TypeOK + Inv2 + Inv3 + Inv4) plus Consensus; the Paxos actions preserve the refinement by construction of the message-pool-backed update rules.

## 5. Four-axis self-assessment

1. **Order / ballot structure:** **Partial.** Integer order on `Z_{>0}` with A1 shift captures strict-greater-than guard correctly; spread-polynomial primitive's multiplicative content is ornamental (indexing monoid rather than composition monoid gets used). The Wildberger bundle has *no primitive that contributes more than what `<` on integers already gives*. Honest: the total-order axis is carried by plain integer order.

2. **Quorum intersection:** **Partial (Captured at N=3, Gap at general N).** At N = 3, majority 2, the projective encoding (acceptors = 3 non-collinear points in P²(Z), quorums = 3 lines through pairs) is **exact**: two distinct lines in P² intersect in exactly one point; by construction that point is the shared acceptor. Explicit arithmetic worked out above. For general N, k, the pigeonhole `|Q₁ ∩ Q₂| ≥ 2k − N` is set-theoretic, not projective; Wildberger bundle doesn't surface it. Strongest load-bearing Wildberger piece in the whole attempt, but local to small N.

3. **Message monotonicity:** **Gap (confirmed repeat).** Same bundle gap the TwoPhase attempt and diff flagged — no monotone-accumulating-set primitive in the bundle. Integer-counter workaround is sound on reachable states but is strictly weaker than the idempotent-set semantics of TLA+ `msgs' = msgs ∪ {m}`. Slightly worse instance than TwoPhase because the 1b-message type has a structured payload.

4. **Guarded receipt / update:** **Partial.** Phase1b and Phase2b are guarded translations (mild primitive strain — current-state-dependent offset, same issue as TwoPhase §3 / diffs Gap 2). The *showsSafe* value-selection at Phase2a is a **max-over-witness** primitive that Wildberger doesn't have — this is an additional Paxos-specific primitive gap beyond TwoPhase's receipt-action strain.

## 6. Contribution self-assessment

Per README rubric (DieHard=0, Bakery=1, TwoPhase=2, QA control=4):

- **Generator-relative structure?** Weak. Actions are (guarded) translations + monotone-pool increments; no spread-polynomial / mutation / reflection structure fits non-ornamentally. Identical generator-algebra weakness to TwoPhase.
- **SCC / orbit organization?** No. Reachable graph is a DAG on (ballots × pool) (monotone in ballot and pool size). Terminal states are absorbing. Honest decline of the SCC story.
- **Closed-form counts?** Partially. At N = 3, k = 2: number of quorums is 3 (`C(3,2)`), number of quorum-pair intersections is `C(3,2) = 3`, each giving exactly one shared acceptor — closed form matches the projective prediction `P²(F) has C(3,2) = 3 lines through 3 points and every pair of lines meets in one point`. For general N, k: `C(N,k)` quorums, pairwise intersection size ≥ `2k − N`. This is a closed-form lower bound but it is combinatorial, not Wildberger-derived. Modest positive.
- **Failure-class algebra?** Partial. Consensus violation = two distinct chosen values = two pairs `(b₁, v₁), (b₂, v₂)` with `v₁ ≠ v₂` both surviving a quorum-vote. In `Z²` with `(b, v)`-coords, two distinct chosen points with `v₁ ≠ v₂` have **`Q_r ≠ 0` on the v-axis component**. This is the null-cone failure algebra — one failure class on the Consensus side (disagreement). Identical shape to TwoPhase's `Q_r = 0` disagreement kernel, but here it's a *negation* (disagreement means `Q_r ≠ 0`, not `Q_r = 0`) because distinct-v is the violation. Load-bearing insofar as the final proof closes by forcing `v₁ = v₂`.
- **Monotonicity under generator expansion?** Yes. Adding more actions (e.g. acceptor read-only queries) cannot create new chosen pairs; Consensus remains.

**Self-score: 2 (Useful).**

**Rationale.** The projective-intersection argument at N = 3 is a *genuine* Wildberger-native contribution — the statement "any two distinct lines in P²(F) meet in exactly one point" literally *is* the quorum-intersection lemma for the smallest non-trivial Paxos configuration. That's a real load-bearing primitive use, comparable to TwoPhase's Q_r=0 null-cone fact. However, it does not generalize cleanly to N ≥ 4 — the projective primitive is only one N-configuration deep. The ballot-structure axis is essentially ornamental (integer order does all the work). The message-monotonicity axis is a confirmed repeat gap. The guarded-receipt axis has a Paxos-specific additional gap (showsSafe max-selection). On the strength-of-encoding, Paxos Mode B matches TwoPhase Mode B: one load-bearing Wildberger primitive (projective intersection at N=3 vs. red null-cone on the terminal pair), several used-but-weak or decorative auxiliaries, no closed-form count or SCC theorem. **Contribution 2, same floor as TwoPhase.**

Could it be 3? Only if the projective argument scaled to general N or if an SCC/orbit structure appeared over the ballot index. Neither materializes under the current bundle without augmentation.

## 7. Primitive-gap report

Inherited from TwoPhase (same spec-class):

- **Gap A (monotone-set primitive).** Paxos's `msgs` pool is a monotone set with structured payloads. Bundle surfaces bounded `Caps` but no unbounded-monotone-set primitive. Counter workaround weakens type semantics. **Confirmed repeat.**
- **Gap B (guarded-translation / idempotent-projection primitive).** Phase1b/Phase2b updates are current-state-dependent offsets, not constant-offset translations. **Confirmed repeat.**

New to Paxos:

- **Gap C (max-over-witness primitive, the `showsSafe` operator).** Phase2a's value-selection rule is `v = v_{argmax_{q ∈ Q} e_q}` — a max-then-project. Wildberger bundle has no lattice-lub / max-selection primitive. This is the deepest Mode-B gap for Paxos: the *value-selection logic* which is what makes Paxos correct is exactly what the Wildberger bundle cannot natively express.
- **Gap D (projective-plane encoding only works at N=3, k=2).** At higher N, the quorum-intersection lemma is pigeonhole, not projective. Scaling would require a *general projective-space intersection* primitive for `P^{k-1}` over indicator subspaces, or a declared set-theoretic pigeonhole primitive. Neither is in the bundle.
- **Gap E (structured-message payload primitive).** The 1b message carries `(e, v)` — a structured payload. TwoPhase's messages were unary (`Prepared(rm)`, `Commit`, `Abort`). The bundle's handling of structured messages requires a product of counters; no native structured-message primitive.

**Augmentation recommendation to reach Contribution 3+ on Paxos.** Add:
1. Monotone-multiset (or idempotent-set) primitive.
2. Max-over-finite-set (lattice lub) primitive — this is the single biggest lever for distributed-protocol encoding.
3. Hyperplane-intersection generalisation in P^k for k > 2, so the projective quorum-intersection argument extends.

Without these three, Mode B on Paxos is capped at 2.

## 8. Self-check

- [x] **A1 (No-Zero):** ballots `∈ Z_{>0}`; sentinel `⊥` is a reserved tag, not `0`; values `∈ V = {1, …, |V|}`; message counters `≥ 1`; acceptor-in-P²(Z) coordinates all in `{0, 1}` within projective classes — wait: `[1:0:1]` has a `0` coordinate. **Check.** Projective classes `[x:y:z]` with `(x,y,z) ≠ (0,0,0)` allow individual `0` coordinates because the class is a ratio; `[1:0:1]` is distinct from `[0:1:1]` and `[1:1:1]` as a projective point. A1 applies to QA *states* (`{1,…,N}` not `{0,…,N-1}`); a projective coordinate tuple is a *representative* of an equivalence class, not a state coord. Using `0` in a representative is cosmetic — the class is what matters. To eliminate any ambiguity, rescale: a1 ≡ [2:1:2], a2 ≡ [1:2:2], a3 ≡ [1:1:1] — all coords in `{1, 2}`, same projective classes up to rescaling? Let me check: `[1:0:1]` vs `[2:1:2]` — these represent different projective classes (since `[2:1:2]` is not a scalar multiple of `[1:0:1]`). So the rescaling breaks the example. Accept `0` in homogeneous coords as a representative-choice artifact; A1 is not violated at the state level. **OK.**
- [x] **A2 (Derived Coords):** `d_a = b_a + e_a`, `a_a = b_a + 2 e_a` are derived, not independently assigned. Checked in §2.
- [x] **T1 (Path Time):** time is integer step count over actions; no continuous time. Pool counters are integer monotone. OK.
- [x] **T2 / NT (Observer Projection Firewall):** no continuous inputs; all message payloads, ballots, values are integer (with A1 shift). No float or sqrt anywhere. OK.
- [x] **S1 (No `**2`):** `Q_r`, `Q_b` are textual notations with `²`; in any implementation they would be `(x-y)*(x-y)`. No code form here.
- [x] **S2 (No float state):** all state `int` or tagged `⊥` symbol; no float, no `np.random`.
- [x] **Wildberger primitives come from the object model:** points in Z² / P²(Z), 4-tuple (b,e,d,a), Q_r, translations, spread-polynomial index (limited use). No fabricated primitives.
- [x] **Four structural axes addressed:** order (§2 ballots + §5.1), quorum-intersection (§4 Consensus candidates c + §5.2), monotonicity (§2 pool + §5.3), guarded receipt (§3 + §5.4).
- [x] **Blindness pact held:** no ground_truth file read this session; no TLA+ predicate body copied from memory; `showsSafe` specification derived from the prompt's English summary of Inv4, not from memorised Paxos.tla. Allowed prior reads: twophase.md attempt and diff (methodology only), prompt, object model, CLAUDE.md, README.
