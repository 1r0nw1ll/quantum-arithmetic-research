# Paxos — Mode B Scored Diff

## Ground truth summary

- **VARIABLES** (`Paxos.tla:34-38`): `maxBal, maxVBal, maxVal, msgs`.
- **Message universe** (`Paxos.tla:28-32`):
  ```
  [type : {"1a"}, bal : Ballot]
  \cup [type : {"1b"}, acc : Acceptor, bal : Ballot, mbal : Ballot \cup {-1}, mval : Value \cup {None}]
  \cup [type : {"2a"}, bal : Ballot, val : Value]
  \cup [type : {"2b"}, acc : Acceptor, bal : Ballot, val : Value]
  ```
- **QuorumAssumption** (`Paxos.tla:13-14`):
  ```
  /\ \A Q \in Quorum : Q \subseteq Acceptor
  /\ \A Q1, Q2 \in Quorum : Q1 \cap Q2 # {}
  ```
  Identical in `Voting.tla:16-17`. Only the pairwise non-empty intersection is assumed; majority is not built in.
- **TypeOK** (`Paxos.tla:67-70`):
  ```
  /\ maxBal  \in [Acceptor -> Ballot \cup {-1}]
  /\ maxVBal \in [Acceptor -> Ballot \cup {-1}]
  /\ maxVal  \in [Acceptor -> Value \cup {None}]
  /\ msgs    \subseteq Message
  ```
- **Init** (`Paxos.tla:73-76`): `maxBal = [a |-> -1]`, `maxVBal = [a |-> -1]`, `maxVal = [a |-> None]`, `msgs = {}`.
- **Action bodies** (one-line summaries; `Send(m) == msgs' = msgs \cup {m}` at L82):
  - `Phase1a(b)` (L91-92): `Send([type |-> "1a", bal |-> b])`; `UNCHANGED <<maxBal, maxVBal, maxVal>>`.
  - `Phase1b(a)` (L100-106): `\E m \in msgs : m.type="1a" /\ m.bal > maxBal[a] /\ maxBal' = [EXCEPT ![a] = m.bal] /\ Send("1b", a, m.bal, maxVBal[a], maxVal[a])`; `UNCHANGED <<maxVBal, maxVal>>`.
  - `Phase2a(b,v)` (L128-141): guards `~ \E m : m.type="2a" /\ m.bal=b` and `\E Q \in Quorum : (all a \in Q have 1b at b) /\ (Q1bv = {} \/ \E m \in Q1bv : m.mval = v /\ \A mm \in Q1bv : m.mbal \geq mm.mbal)`; then `Send("2a", b, v)`; `UNCHANGED <<maxBal, maxVBal, maxVal>>`.
  - `Phase2b(a)` (L152-158): `\E m \in msgs : m.type="2a" /\ m.bal \geq maxBal[a] /\ maxBal' = [EXCEPT ![a] = m.bal] /\ maxVBal' = [EXCEPT ![a] = m.bal] /\ maxVal' = [EXCEPT ![a] = m.val] /\ Send("2b", a, m.bal, m.val)`.
- **Refinement** (`Paxos.tla:185-197`):
  ```
  votes == [a \in Acceptor |-> {<<m.bal, m.val>> : m \in {mm \in msgs : mm.type="2b" /\ mm.acc=a}}]
  V == INSTANCE Voting
  THEOREM Spec => V!Spec
  ```
- **Inductive invariant `Inv`** (`Paxos.tla:203-216`):
  ```
  Inv == /\ TypeOK
         /\ \A a : IF maxVBal[a] = -1 THEN maxVal[a] = None
                                       ELSE <<maxVBal[a], maxVal[a]>> \in votes[a]       (* Inv2 *)
         /\ \A m \in msgs :
               /\ m.type="1b" => /\ maxBal[m.acc] \geq m.bal
                                 /\ m.mbal \geq 0 => <<m.mbal, m.mval>> \in votes[m.acc]  (* Inv3 *)
               /\ m.type="2a" => /\ \E Q : V!ShowsSafeAt(Q, m.bal, m.val)
                                 /\ \A mm : mm.type="2a" /\ mm.bal=m.bal => mm.val=m.val  (* Inv4 *)
         /\ V!Inv
  ```
  `V!ShowsSafeAt(Q, b, v)` (`Voting.tla:118-122`):
  ```
  /\ \A a \in Q : maxBal[a] \geq b
  /\ \E c \in -1..(b-1) :
       /\ (c # -1) => \E a \in Q : VotedFor(a, c, v)
       /\ \A d \in (c+1)..(b-1), a \in Q : DidNotVoteAt(a, d)
  ```
- **`V!Inv` = safety target on Voting side** (`Voting.tla:176`): `TypeOK /\ VotesSafe /\ OneValuePerBallot`.
- **Top safety (Consensus layer)** (`Voting.tla:190` and `Consensus.tla:36-37`):
  ```
  Voting: THEOREM Spec => C!Spec           (C == INSTANCE Consensus)
  Consensus.Inv: TypeOK /\ Cardinality(chosen) \leq 1
  ```
  I.e. "at most one value is ever chosen." Paxos.Spec ⇒ Voting.Spec ⇒ Consensus.Spec transitively discharges this.

## Per-invariant scoring (Recovery)

### TypeOK

- **TLA+** (`Paxos.tla:67-70`): functions `Acceptor → Ballot ∪ {-1}`, `Acceptor → Value ∪ {None}`, `msgs ⊆ Message`.
- **Attempt** (`attempts/paxos.md:103-109`): each acceptor `(b_a, e_a, v_a) ∈ (Z_{>0} ∪ {⊥})² × (V ∪ {⊥})`, with derived `d_a = b_a+e_a`, `a_a = b_a+2e_a`, plus two extra conjuncts `(e_a≠⊥ ⟹ v_a≠⊥)` and `(e_a≠⊥ ∧ b_a≠⊥ ⟹ e_a ≤ b_a)`; `MessagePool ⊆ WellTypedMessageUniverse`.
- **Verdict:** `strengthened` with a benign type-side weakening on messages.
- **Notes:** The two extra conjuncts are genuine strengthenings — TLA+'s `TypeOK` does NOT require `maxVBal[a] = -1 ⇔ maxVal[a] = None` (that is folded into Inv2 in the ground truth, see L204-206) nor `maxVBal[a] ≤ maxBal[a]` (which in ground truth follows from Phase2b's `m.bal ≥ maxBal[a]` guard plus monotonicity, established inductively as part of `V!Inv`/OneValuePerBallot, not declared as type). Pulling them into TypeOK is a reasonable Mode B normalisation but does strictly rule out states that bare TLA+ `TypeOK` alone permits. Same counter-encoding weakening as TwoPhase on the pool side (`m_{kind}(payload) ∈ Z_{≥1}` vs idempotent set).

### Inv2 — acceptor vote-history consistency

- **TLA+** (`Paxos.tla:204-206`):
  ```
  \A a \in Acceptor : IF maxVBal[a] = -1 THEN maxVal[a] = None
                                          ELSE <<maxVBal[a], maxVal[a]>> \in votes[a]
  ```
- **Attempt** (`attempts/paxos.md:117-120`):
  ```
  (∀a) e_a ≠ ⊥ ⟹ ∃ "2b"(a, e_a, v_a) ∈ msgs
  ```
- **Verdict:** `reproduced` (informational content equivalent; the `⊥`-branch handled implicitly by the attempt's `e_a=⊥ ⟹ v_a=⊥` type conjunct).

### Inv3 — 1b-message faithfulness

- **TLA+** (`Paxos.tla:207-210`):
  ```
  (m.type = "1b") => /\ maxBal[m.acc] \geq m.bal
                     /\ (m.mbal \geq 0) => <<m.mbal, m.mval>> \in votes[m.acc]
  ```
- **Attempt** (`attempts/paxos.md:128-131`):
  ```
  (∀ "1b"(a, b, e, v) ∈ msgs) e ≠ ⊥ ⟹ ∃ "2b"(a, e, v) ∈ msgs
  ```
  with prose at L133 that `(e, v)` is a snapshot of the sender's then-current `(e_a, v_a)`.
- **Verdict:** `weakened`. The attempt captures the `m.mbal ≥ 0 ⇒ votes` conjunct (the second conjunct) but **misses the first conjunct `maxBal[m.acc] \geq m.bal`** — that once an acceptor has emitted a `"1b"` at ballot `b`, its `maxBal` cannot decrease below `b`. This is the monotonicity-of-promise witness and is a non-trivial strengthening in the ground truth. Marked as missed-sub-conjunct.

### Inv4 — 2a uniqueness + ShowsSafe

- **TLA+** (`Paxos.tla:211-215`):
  ```
  (m.type = "2a") => /\ \E Q \in Quorum : V!ShowsSafeAt(Q, m.bal, m.val)
                     /\ \A mm \in msgs : mm.type="2a" /\ mm.bal=m.bal => mm.val=m.val
  ```
- **Attempt** (`attempts/paxos.md:142-145`):
  ```
  Inv4_a: m_{2a}(b, v1) ≥ 2 ∧ m_{2a}(b, v2) ≥ 2 ⟹ v1 = v2
  Inv4_b: (∀"2a"(b,v) ∈ msgs) ∃ quorum Q of 1b(·,b,·,·) : v = showsSafe(Q, b)
  ```
  with `showsSafe` at L86 = `argmax_e`-then-value.
- **Verdict:** `reproduced` on the uniqueness conjunct; `weakened` on the ShowsSafe conjunct. The attempt's `showsSafe` collapses the ground truth's existential over a *witness prefix* `\E c \in -1..(b-1)` (which additionally requires `\A d \in (c+1)..(b-1), a \in Q : DidNotVoteAt(a, d)` — the "nobody after c voted in this quorum" clause) into a pure `argmax e_q`. For the *chosen value* part, `argmax` coincides with the ground truth because the max-witnessed ballot is by construction the largest `c` for which any `a \in Q` voted — but the attempt does not capture the "and no-one in Q voted between c and b" clause, which is load-bearing in the ground-truth safety proof (it is what makes `ShowsSafeAt ⇒ SafeAt`, `Voting.tla:124-127`). Minor weakening on the DidNotVoteAt side; correct on the argmax side.

### `V!Inv` (VotesSafe + OneValuePerBallot)

- **TLA+** (`Voting.tla:100-107`):
  ```
  VotesSafe      == \A a, b, v : VotedFor(a,b,v) ⇒ SafeAt(b,v)
  OneValuePerBallot == \A a1,a2,b,v1,v2 : VotedFor(a1,b,v1) ∧ VotedFor(a2,b,v2) ⇒ v1=v2
  ```
- **Attempt:** not named separately; the attempt rolls the OneValuePerBallot content into Inv4_a (2a-uniqueness) and the VotesSafe content into the inductive narrative at §4 Consensus (L229: "showsSafe picks v_{a*} = v_1. So Phase2a(b2, v2) is forced to pick v2 = v1"). This is an encoding of the same safety chain but not a named predicate.
- **Verdict:** `missed` as a named predicate; `reproduced` as informational content inside the Consensus argument. Counted as 0.5.

### Consensus (refinement to Voting/Consensus safety)

- **TLA+**: `Paxos.Spec ⇒ V!Spec` (`Paxos.tla:197`), `V!Spec ⇒ C!Spec` (`Voting.tla:190`), with `C!Inv` = `Cardinality(chosen) \leq 1` (`Consensus.tla:36-37`). I.e. at most one value ever chosen.
- **Attempt** (`attempts/paxos.md:158-161`, formal, and `attempts/paxos.md:229`, informal):
  ```
  Consensus ≡ ∀b1,b2,v1,v2 : chosen(b1,v1) ∧ chosen(b2,v2) ⟹ v1=v2
  ```
  where `chosen(b,v) ≡ ∃Q : ∀a ∈ Q, m_{2b}(a,b,v) ≥ 2`. Proof via quorum-intersection lemma, with the lemma dischargeable projectively at N=3 (see Axis 2) and by pigeonhole at N ≥ 4.
- **Verdict:** `reproduced` (informational content equivalent to `Cardinality(chosen) ≤ 1`), with an explicit `proof-gap` at N ≥ 4 — the projective-intersection primitive only covers N = 3, k = 2; general-N closure falls back on set-theoretic pigeonhole which is not a Wildberger primitive.

### Recovery tally

- 5 named scoring targets (TypeOK + refinement/Consensus + Inv2 + Inv3 + Inv4) per the prompt at L135-138.
- **TypeOK** strengthened; **Inv2** reproduced; **Inv3** weakened (missing first conjunct); **Inv4** reproduced on uniqueness, weakened on ShowsSafe; **Consensus refinement** reproduced-with-proof-gap at N ≥ 4.
- Aggregate: **5/5 reproduced or partially reproduced; 0 missed; 0 wrong; 2 weakenings (Inv3 half, Inv4 ShowsSafe); 1 strengthening (TypeOK); 1 proof-gap (Consensus at N ≥ 4)**.

## Structural-axis assessment (Mode B Paxos-specific)

### Axis 1: Order / ballot structure

- **Attempt** (`attempts/paxos.md:28-33`): ballot = positive integer in `Z_{>0}`, sentinel `⊥` as reserved tag; strict guard `b' > b` is integer `<`. Spread-polynomial index `S_n` named but declared partial-ornamental at L12 and L33.
- **Ground-truth** (`Paxos.tla:16`): `Ballot == Nat`; strict guard `m.bal > maxBal[a]` (L102) and non-strict `m.bal \geq maxBal[a]` (L153) — both integer `<`/`≤`.
- **Load-bearing test:** does any Wildberger primitive do work beyond what integer `<` already does? The attempt correctly answers *no*: the spread-polynomial composition monoid `S_n ∘ S_m = S_{nm}` is multiplicative, whereas the Paxos guard is the additive order on `Z_{>0}`. Integer `<` carries the axis unaided.
- **Verdict:** **captured via integer order; Wildberger piece ornamental.** The ballot axis lands on plain `Z_{>0}` — a correct encoding, but the Wildberger primitives (`S_n`, projective-line coordinates, mutation moves) all considered and honestly declined. Fires `ornamental-overlay` on the spread-polynomial label specifically.

### Axis 2: Quorum intersection

- **Attempt** (`attempts/paxos.md:48-56, 174-212`): acceptors at N = 3 placed as points in `P²(Z)`, specifically `a1=[1:0:1], a2=[0:1:1], a3=[1:1:1]`; quorums = pairs of acceptors = lines through pairs.

**Independent verification of the N = 3 projective arithmetic:**

- **Non-collinearity.** `det([[1,0,1],[0,1,1],[1,1,1]]) = 1·(1−1) − 0·(0−1) + 1·(0−1) = 0 − 0 − 1 = −1 ≠ 0.` General-position confirmed.
- **Line through each pair, via homogeneous cross product:**
  - `L_{12} = a1 × a2 = (0·1 − 1·1,  1·0 − 1·1,  1·1 − 0·0) = (−1, −1, 1)`. Incidence: a1: `1·(−1)+0·(−1)+1·(1)=0 ✓`; a2: `0·(−1)+1·(−1)+1·(1)=0 ✓`.
  - `L_{13} = a1 × a3 = (0·1 − 1·1,  1·1 − 1·1,  1·1 − 0·1) = (−1, 0, 1)`. Incidence: a1: `−1+0+1=0 ✓`; a3: `−1+0+1=0 ✓`.
  - `L_{23} = a2 × a3 = (1·1 − 1·1,  1·1 − 0·1,  0·1 − 1·1) = (0, 1, −1)`. Incidence: a2: `0+1−1=0 ✓`; a3: `0+1−1=0 ✓`.
- **Line intersections = shared acceptor:**
  - `L_{12} ∩ L_{13} = (−1,−1,1) × (−1,0,1) = (−1·1 − 1·0, 1·(−1) − (−1)·1, (−1)·0 − (−1)·(−1)) = (−1, 0, −1) ≡ [1:0:1] = a1. ✓` (a1 is in both `{a1,a2}` and `{a1,a3}`.)
  - `L_{12} ∩ L_{23} = (−1,−1,1) × (0,1,−1) = ((−1)(−1) − 1·1, 1·0 − (−1)(−1), (−1)·1 − (−1)·0) = (0, −1, −1) ≡ [0:1:1] = a2. ✓`
  - `L_{13} ∩ L_{23} = (−1,0,1) × (0,1,−1) = (0·(−1) − 1·1, 1·0 − (−1)(−1), (−1)·1 − 0·0) = (−1, −1, −1) ≡ [1:1:1] = a3. ✓`

The attempt's arithmetic is correct. At N = 3, majority = 2: the three majority quorums `{a1,a2}, {a1,a3}, {a2,a3}` are in bijection with the three lines `L_{12}, L_{13}, L_{23}`, and every pairwise line intersection in P² *is* the shared acceptor. **The projective-plane intersection axiom literally is the quorum-intersection lemma for this N.**

- **Scale check (N ≥ 4).** At N = 4, majority = 3: size-3 subsets of 4 general-position points in P²(Z) are NOT in general collinear — three non-collinear points span a triangle, not a line. So "quorum = line through 3 points" is not well-defined; any three non-collinear points determine zero lines (strictly: no single line passes through all three). One could move to P³ (hyperplanes through size-3 subsets) but the Wildberger object model specifies P² (primitive §1-projective, `[x:y:z]` triples). The attempt correctly flags this as a genuine bundle gap at L207-212. Confirmed by independent check: raising the dimension to encode size-k quorums as P^{k-1} hyperplanes is machinery not in the object-model primitive list.
- **Pigeonhole fallback.** At general N, k: `|Q1 ∩ Q2| ≥ 2k − N`. For majority quorums (`k = ⌈(N+1)/2⌉`), `2k − N ≥ 1`. This is set-theoretic pigeonhole via indicator-vector inner product, not a Wildberger primitive.
- **Verdict:** **captured at N = 3, k = 2 (genuinely load-bearing: projective intersection = quorum intersection, arithmetically exact); gap at N ≥ 4 (projective primitive does not generalise within the bundle).** This is the single strongest Wildberger use in the attempt.

### Axis 3: Message monotonicity

- **Attempt** (`attempts/paxos.md:74-75, 250`): flagged as confirmed repeat gap from TwoPhase; monotone integer counters `Z_{≥1}^{...}` as workaround; notes that the `"1b"` payload `(e_a, v_a)` makes the instance worse than TwoPhase's unary messages.
- **Ground-truth check.** Is `msgs` monotone? Every action that modifies `msgs` does so via `Send(m) == msgs' = msgs \cup {m}` (`Paxos.tla:82`). All four actions `Phase1a`, `Phase1b`, `Phase2a`, `Phase2b` invoke `Send` to add a message; none remove. **Confirmed monotone.**
- **Wildberger primitive check.** The object model surfaces `Caps(N, N)` (bounded), `Z²`, `P²(Z)`, quadrances, spreads, TQF, cross-ratio, spread polynomials, SL(3) rings, 4D diagonals, translations, reflections, rotations, projective maps, mutation moves. **None is an unbounded monotone accumulating set.** Closest candidate, `Caps`, is bounded — wrong shape. A creative reproducer cannot synthesise monotone-set from this alphabet without augmenting it with a new primitive.
- **Verdict:** **gap (confirmed repeat from TwoPhase).** `qa-weaker-than-tla (minor)` on the pool type. Not reproducer oversight — real bundle gap.

### Axis 4: Guarded receipt / update

- **Attempt** (`attempts/paxos.md:84-86, 92, 149, 252`): Phase1b/Phase2b are guarded translations (same "current-state-dependent offset" strain as TwoPhase Gap 2). Phase2a's `showsSafe` value-selection is the Paxos-specific additional gap: `v = v_{argmax_{q ∈ Q} e_q}` — a max-over-finite-set primitive.
- **Ground-truth check.** Phase2a (`Paxos.tla:128-141`): the value `v` chosen must satisfy `\E m \in Q1bv : m.mval = v /\ \A mm \in Q1bv : m.mbal \geq mm.mbal` — exactly "v is the value reported by the 1b message with the largest m.mbal in the quorum." This IS max-over-finite-set followed by projection-to-value. The attempt's `showsSafe(Q, b) = v_{argmax_{q ∈ Q} e_q}` matches this (up to the DidNotVoteAt side-clause mentioned in Inv4 scoring above).
- **Wildberger primitive check.** Translations (§1-transforms) give add-a-constant. Reflections (§2-transforms) give sign-flip. Rotations via spread polynomials (§3-transforms) give rational rotations. Projective maps (§4-transforms) give linear-fractional. Mutation moves (§5-transforms) send `α_i → -α_i + Σ |⟨α_i, α_j⟩| α_j` — coefficient-bounded root-system reflections. **None is lattice-lub / max-over-finite-set.** A max-selector is a set-lattice primitive (`∨` in a distributive lattice); the Wildberger bundle does not surface one. Creative attempts to synthesise max via iterated pairwise-reflection do not terminate to a closed form.
- **Verdict:** **gap (new, Paxos-specific).** This is the single deepest Mode-B gap for Paxos — the value-selection logic that makes Paxos *correct* is exactly what the bundle cannot natively express.

## Contribution score (0-4)

Per-primitive load-bearing assessment (attempt's five invoked primitives):

1. **Points in `Z²` + 4-tuple `(b, e, d, a)` (object model §1, §9).** **USED BUT WEAK.** Acceptor local state placed in the 4-tuple; `d` and `a` derived but don't enter any invariant — TypeOK, Inv2, Inv3, Inv4 all reference `b` and `e` (= `maxBal, maxVBal`) only. Same weakness shape as TwoPhase's 4-tuple.

2. **Points in `P²(Z)` + quorum lines (object model §1-projective, §2).** **LOAD-BEARING AT N = 3; GAP AT N ≥ 4.** Verified arithmetically above: at N = 3, k = 2, the identity `line ∩ line = shared point` in P²(Z) is *exactly* the quorum-intersection lemma; arithmetic is tight and non-decorative. For N ≥ 4 the primitive does not lift and the attempt correctly falls back to pigeonhole (a non-Wildberger argument). **Partial load-bearing — the strongest Wildberger use in the attempt, but narrow in scope (one N-configuration).**

3. **Red quadrance `Q_r` (object model §3-Lorentzian).** **USED BUT WEAK (declared but not closed).** The attempt at L260 says two distinct chosen points with `v_1 ≠ v_2` have `Q_r ≠ 0` on the v-axis — a "null-cone violation" framing for Consensus failure. But this is a negation of TwoPhase's shape (in TwoPhase, `Q_r = 0` flags disagreement; in Paxos, `Q_r ≠ 0` is the supposed flag), and crucially the attempt never closes the argument via `Q_r` — the Consensus proof flow at L229 uses quorum-intersection + Inv3 + showsSafe, not `Q_r`. The `Q_r` label is attached but does no algebraic work. **Declarative, not load-bearing.**

4. **Translations + guarded translations (object model transforms §1).** **USED BUT WEAK (same as TwoPhase Gap 2).** Phase1a is a pure pool translation. Phase1b/Phase2b are "set-to-fixed-point" updates, which the attempt honestly flags at L91 as "current-state-dependent offset, not constant-offset translation" — same strain as TwoPhase.

5. **Spread polynomials `S_n` (object model §7).** **ORNAMENTAL** (by the attempt's own declaration at L12 and L28). Monoid composition is multiplicative; Paxos uses additive strict order. No discrimination added.

**Closed-form check.** The prompt-hinted test: does the encoding produce closed-form counts? 
- At N = 3: yes, `C(3,2) = 3` quorums, `C(3,2) = 3` pairwise intersections, all size exactly 1 — matched by the projective prediction (three lines in P², three pairwise meets, each a single projective point). **Closed form at this one case.**
- At N ≥ 4: no — pigeonhole gives a *lower-bound* `|Q1 ∩ Q2| ≥ 2k − N`, which is combinatorial, not Wildberger-derived. No closed-form count of reachable ballot sequences, no closed-form count of quorum families of a given size, no closed-form failure algebra beyond the negation-of-TwoPhase `Q_r ≠ 0` label.
- **No SCC / orbit structure**: attempt correctly declines this at L259 (monotone DAG on ballots × pool).

**Summary:** of 5 primitives invoked, **1 partially load-bearing (P² projective intersection, narrow at N=3 only), 3 used-but-weak (points+4-tuple, Q_r-as-label, translations), 1 ornamental (spread polynomials)**. One new load-bearing *geometric* insight (the N=3 projective correspondence) vs. TwoPhase's one load-bearing *algebraic* fact (`Q_r((3,1),(1,3)) = 0`). Parity in load-bearing density, with Paxos's piece being narrower in scale (one N only) but richer in structural content (projective-intersection is a *geometric theorem* of the ambient space, not a point-specific arithmetic coincidence).

**Final Contribution: 2 (Useful).**

**Rationale.** The N=3 projective-plane correspondence between acceptors and points, quorums and lines, and quorum-intersection and line-meet-in-a-point is a *genuine* load-bearing Wildberger use — arithmetically tight, directly discharges the quorum-intersection lemma for the smallest non-trivial Paxos config, and is verified independently above. But it does not lift to N ≥ 4 (pigeonhole is not Wildberger), Axis 1 is carried by plain integer `<` (spread polynomials ornamental), Axis 3 is a confirmed repeat gap (no monotone-set primitive), and Axis 4 exposes a *new* Paxos-specific gap (no max-over-finite-set primitive — the heart of Paxos's value-selection rule). Same floor as TwoPhase.

Could it be 3? Only if the projective argument scaled to general N (it doesn't — requires P^{k-1} generalisation not in the bundle) or if a closed-form count / SCC theorem emerged (it doesn't — the reachable graph is a monotone DAG). Not 3.

## Failure taxonomy tags

- **`ornamental-overlay` (partial, per-primitive).** Per the Mode B partial-credit rule: fires on spread polynomials (Axis 1, declared decorative) and on `Q_r`-as-Consensus-label (Axis 4-side, attached but not used in the proof). Does NOT fire on P² projective intersection (at N=3, genuinely load-bearing) nor on 4-tuple/translations (used-but-weak, not fully decorative). 2/5 primitives fire ornamental, 1/5 load-bearing, 2/5 weak.
- **`proof-gap`** — fires on Consensus refinement: the projective argument discharges quorum intersection at N = 3, k = 2, but the bundle does not extend the primitive to N ≥ 4. The attempt honestly flags this at L207-212 and falls back on set-theoretic pigeonhole outside the bundle. This is a proof-gap at the Wildberger-encoding level (the ground-truth Paxos safety proof closes uniformly in N via the quorum axiom; the attempt's projective closure only handles N=3).
- **`qa-weaker-than-tla` (minor)** — on TypeOK-message-pool (counter encoding admits unbounded counts where TLA+ idempotent-set caps at one); on Inv3 (first conjunct `maxBal[m.acc] ≥ m.bal` missed); on Inv4-ShowsSafe (DidNotVoteAt side-clause missed). Each is minor individually.
- **NOT `no-mapping-exists`** — defensible mapping was produced.
- **NOT `wrong-observer-projection`** — protocol fully discrete.
- **NOT `orbit-mismatch`** — attempt correctly declines to claim Cosmos/Satellite/Singularity structure.
- **NOT `invariant-inexpressible`** — all five named properties expressed in QA-native form.
- **NOT `qa-stronger-than-tla`** — the two TypeOK strengthenings are benign normalisations, not independent constraints.

## Primitive-gap assessment

Attempt's five gaps (`attempts/paxos.md:272-281`):

- **Gap A — monotone-set primitive (repeat from TwoPhase).** **REAL.** Verified above: no combination of `Caps(N,N)`, `Z²`, `P²(Z)`, quadrances, spreads, cross-ratios, TQF, spread polynomials, SL(3) rings, 4D diagonals, translations/reflections/rotations/projective maps/mutations surfaces unbounded-monotone-accumulating-set. Counter workaround is a weakening, not an encoding. Real gap — no creative reproducer can fill this without augmentation.
- **Gap B — guarded translation / idempotent-projection primitive (repeat from TwoPhase).** **REAL.** Mutation moves are coefficient-bounded root-system reflections (integer-Cartan-coefficient dependent); they carry Weyl-group structure Paxos does not have. No primitive in the list models "set variable to a value determined by current state and an incoming message." Same verdict as TwoPhase: real gap; forcing mutation-fit would score ornamental.
- **Gap C — max-over-finite-set lattice-lub (new, Paxos-specific).** **REAL.** The argmax step in Phase2a / `showsSafe` is a lattice-join in (`Ballot ∪ {-1}`, ≤) restricted to a finite witness set. The object model does not surface `∨`/`max`; all its transforms are motion-preserving (translate, reflect, rotate, project) or root-system-reflection, none of which is lattice-lub. Synthesising max via repeated pairwise-reflection does not close to a finite primitive expression. Real gap; the single biggest gap for distributed-protocol encoding under this bundle.
- **Gap D — hyperplane-intersection in P^{k-1} beyond P² (new, scale-specific).** **REAL at the object-model level.** The object model specifies P²(Z) triples (§1) and projective maps (transforms §4). It does NOT specify arbitrary P^{k-1} or hyperplane-intersection in higher projective spaces. A creative reproducer could in principle extend to P^{k-1} within Wildberger's broader UHG corpus (which does handle higher-dim projective geometry), but under the *pre-specified bundle* this is augmentation. Real bundle gap.
- **Gap E — structured-message payload primitive (new, encoding-specific).** **PARTIAL (workaround-expressible).** The `"1b"(a, b, e, v)` 4-tuple payload is expressible as a product of counter axes (one axis per distinct `(a, b, e, v)` combination). Cumbersome but not strictly outside the bundle — `Caps` products give this shape. Less of a "real gap" and more of an "encoding inefficiency." Marking as partial.

**Augmentation recommendation to reach Contribution 3+ on Paxos.**
1. Monotone-multiset primitive (also fixes TwoPhase Gap 1).
2. Max-over-finite-set / lattice-lub primitive (the single biggest lever for distributed-protocol encoding).
3. Hyperplane-intersection in P^{k-1} for k > 2 (lifts the N=3 projective correspondence to general N).

Without #1 and #2 in particular, Mode B on any protocol with message-pool + argmax-receive shape (Paxos, Raft, Multi-Paxos, Fast Paxos, Zab, ...) is capped at 2.

## Blindness check

Verbatim identifiers from Paxos.tla / Voting.tla / Consensus.tla, scanned against attempt:

- **`maxBal, maxVBal, maxVal, msgs`** — appear in the attempt (e.g., L40-42, L75, L100-106). **These are prompt-leaked** (`prompts/paxos.md:42-53`). **Not a blindness break** — the prompt instructs the reproducer to use exactly these names.
- **`Quorum, Ballot, Value, Acceptor`** — appear in attempt (L104, L111, L158). **Prompt-leaked** at `prompts/paxos.md:38-40, 106-108`. Not a break.
- **Action names** `Phase1a, Phase1b, Phase2a, Phase2b` — appear in attempt (L79-84, L91, L92, L149). **Prompt-leaked** at `prompts/paxos.md:75-90`. Not a break.
- **`ShowsSafeAt`** — the attempt uses `showsSafe(Q, b)` (L86, L144, L149, L229), which is a cosmetic re-case of the ground-truth `V!ShowsSafeAt(Q, b, v)`. The *prompt* uses the English phrase "shows safe" at `prompts/paxos.md:83, 123` ("value that is consistent with the quorum's last-vote evidence"; "backed by a quorum that 'shows safe' the value"). **Derived from prompt English, not copied from `Voting.tla:118-122`.** Pact held; load-bearing check because ShowsSafeAt is the one predicate whose geometric content is load-bearing.
- **`votes[a]`, `VotedFor`, `DidNotVoteAt`, `CannotVoteAt`, `SafeAt`, `NoneOtherChoosableAt`, `chosen`, `OneValuePerBallot`, `VotesSafe`, `ChosenAt`** — **do NOT appear in the attempt.** The attempt uses `chosen(b, v)` at L155 as a predicate with its own definition via `m_{2b}` counters, not as a reference to `Voting!chosen`. This is the most load-bearing blindness check: the attempt reconstructs the Voting-layer abstractions from the prompt's English summary without pulling their names.
- **Inv2 / Inv3 / Inv4 predicate bodies** — the attempt's Inv2 at L118-120 (`e_a ≠ ⊥ ⟹ ∃ "2b"(a,e_a,v_a) ∈ msgs`) does NOT copy the ground-truth's `IF maxVBal[a]=-1 THEN maxVal[a]=None ELSE <<maxVBal[a], maxVal[a]>> ∈ votes[a]` (L204-206). Inv3 at L128-131 captures the second conjunct but misses the first — if the attempt had seen `Paxos.tla:207-210` it would have seen both conjuncts and pulled them both. Inv4 at L142-145 correctly captures the uniqueness but flattens the ShowsSafeAt existential-prefix to a pure argmax — same signature of derivation-from-English, not copy-from-TLA+. **All three consistent with the declared blindness pact.**
- **Refinement framing** — the attempt at L234-240 writes `votes[a] ≡ { (b, v) : m_{2b}(a, b, v) ≥ 2 }` and `maxBal[a] ≡ b_a`. This matches the ground-truth refinement map at `Paxos.tla:185-187` ("votes[a] is the set of <b, v> from 2b messages from a"). But the prompt explicitly leaks this at `prompts/paxos.md:109-114` ("under a refinement mapping that projects the Paxos state onto <votes, maxBal> where votes[a] is the set of <ballot, value> pairs witnessed by '2b' messages from a"). **Prompt-leaked**, not a blindness break.
- **N, V, B concrete values** — attempt keeps N, V, B parametric as instructed by the prompt (L261-264). No leak.
- **MCPaxos harness content** — not referenced in the attempt.
- **ballot sentinel `-1`** — attempt uses `⊥` (tagged symbol) and explicitly argues at L33 against using `0` for A1 compliance. Does NOT copy the ground-truth choice of `-1` (which would have been a viable A1-compliant alternative but is the specific TLA+ choice). Consistent with blindness — the prompt at `prompts/paxos.md:42-49` only says "distinguished below-all-ballots sentinel" without committing to `-1`.

**Blindness pact: HELD.** The strongest load-bearing check is the Inv-bodies reconstruction: the attempt's Inv3 misses the first conjunct and Inv4 weakens the ShowsSafeAt clause — both are exactly the derivation-from-English signature, not copy-from-TLA+. A reader with ground-truth access would have caught both omissions trivially.

## Overall verdict

- **Recovery:** 5/5 named properties reproduced at informational-content level (TypeOK strengthened; Inv2 reproduced; Inv3 weakened by missing first conjunct; Inv4 uniqueness reproduced, ShowsSafeAt weakened by missing DidNotVoteAt side-clause; Consensus refinement reproduced with N ≥ 4 proof-gap). 2 minor weakenings, 1 benign strengthening, 1 scale-specific proof-gap, 0 missed, 0 wrong.
- **Contribution: 2 (Useful).** One genuinely load-bearing Wildberger primitive use (P²(Z) projective intersection at N=3, arithmetically verified: `L_{12} ∩ L_{13} = a1`, `L_{12} ∩ L_{23} = a2`, `L_{13} ∩ L_{23} = a3`, with quorums `{a1,a2}, {a1,a3}, {a2,a3}` in exact bijection with the three lines `L_{12}, L_{13}, L_{23}`). But narrow in scale (N=3 only), Axis 1 is carried by plain integer `<` (spread polynomials ornamental), Axis 3 is a confirmed repeat gap, Axis 4 exposes a new Paxos-specific gap (no lattice-lub primitive). No closed-form count, no SCC theorem, no Contribution-3 lift.
- **Per-axis:** [1: captured via integer order, Wildberger piece ornamental | 2: captured at N=3 (load-bearing, verified), gap at N≥4 | 3: gap (repeat) | 4: gap (new)]
- **Dominant tags:** `ornamental-overlay` (partial, 2/5 primitives: spread polynomials + `Q_r`-as-label), `proof-gap` (Consensus at N ≥ 4), `qa-weaker-than-tla` (minor, on TypeOK-pool + Inv3 first conjunct + Inv4 DidNotVoteAt side-clause).
- **Blindness held:** YES. Inv2/3/4 bodies reconstructed from prompt English (with the signature weakenings a derivation-from-English would produce); no ground-truth predicate body copied. Identifiers present in the attempt are prompt-leaked.
- **Threshold finding.** **Mode B Paxos caps at Contribution 2, same floor as TwoPhase.** Two consecutive distributed protocols at the same Contribution-2 ceiling despite Paxos providing a strictly stronger load-bearing geometric insight (projective intersection at N=3 is a genuine theorem of the ambient space, whereas TwoPhase's `Q_r((3,1),(1,3)) = 0` is a point-specific arithmetic coincidence) is **strong evidence that the Wildberger-only bundle is structurally insufficient for distributed-protocol dynamics**. The two Mode-B-blocking gaps — (A) monotone-accumulating-set primitive and (C) lattice-lub / max-over-finite-set primitive — are both present in any protocol with message-pool + argmax-receive shape. Per the README threshold rule ("stop early if two consecutive specs show ornamental-overlay on axes QA should dominate"), the honest read is: **pause Mode B distributed-protocol scale-out and design a bundle augmentation (lattice-lub + monotone-multiset + P^{k-1} intersection) before the next distributed-protocol Mode B attempt.**
- **Headline:** Paxos under Mode B reaches Contribution 2 via a load-bearing P²(Z) projective-intersection correspondence at N=3 (quorums = lines, three arithmetic-verified meets `L_{ij} ∩ L_{ik} = a_i` discharge quorum-intersection exactly for the smallest non-trivial Paxos config) — same Contribution floor as TwoPhase, confirming the Wildberger-only bundle caps at 2 on distributed-protocol dynamics; the three blocking gaps (monotone-set, lattice-lub for argmax-receive, hyperplane-intersection beyond P²) are consistent across both specs and motivate a bundle augmentation before further distributed-protocol Mode B scale-out.
