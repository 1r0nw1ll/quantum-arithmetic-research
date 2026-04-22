# Paxos — Mode B Scored Diff v2 (Augmented A + C + D)

## Relation to v1 + v2 augmentation goal

- v1 Paxos: **Contribution 2** via a P²(Z) line-meet correspondence that was
  tight at N=3, k=2 but scale-gapped at N≥4; msgs-monotonicity and
  showsSafe-argmax were confirmed bundle gaps.
- v2 claim: **Contribution 3** via three augmented primitives (A monotone
  multiset, C lattice-lub, D projective subspace lattice in `P^{N-1}(Z)`).
- My job: verify the delta is real, focusing on whether D does load-bearing
  work beyond renaming TLA+'s quorum axiom, whether A/C land on Paxos as
  they did on TwoPhase v2, and whether the closed-form counts survive
  small-N enumeration.

## Ground truth recap

Paxos's only substantive quorum assumption is pairwise non-empty
intersection (`Paxos.tla:13-14`, `Voting.tla:16-17`):

```
\A Q1, Q2 \in Quorum : Q1 \cap Q2 # {}
```

Actions (`Paxos.tla:82-158`) all invoke `Send(m) == msgs' = msgs \cup {m}` —
monotone add-only set semantics. Phase2a's heart (L131-139) is
`\E m \in Q1bv : m.mval = v /\ \A mm \in Q1bv : m.mbal \geq mm.mbal` —
textbook argmax-then-project over a finite witness set. `V!ShowsSafeAt`
(`Voting.tla:118-122`) additionally carries the `DidNotVoteAt` side-clause
across the `(c+1)..(b-1)` gap. Refinement to Voting (`Paxos.tla:185-197`),
Voting to Consensus (`Voting.tla:190`), Consensus safety = `Cardinality(chosen) ≤ 1`.

Ground truth details already extensively recorded in v1 diff
(`qa_tla_blind/diffs/paxos.md`); this diff focuses on what v2 changes.

## Per-invariant scoring (Recovery)

### TypeOK

v1 carried a `qa-weaker-than-tla (minor)` on the msgs counter encoding
(unbounded counters vs idempotent TLA+ set). v2 re-encodes msgs as a
Primitive-A monotone multiset with a per-x reachability bound
`msgs(x) ∈ {1, 2}` under A1 shift. Unlike TwoPhase v2, the per-tag self-
disabling argument on Paxos is per-distinct-tuple, not per-actor-per-kind:
Phase1b's strict-greater guard `b > b_a` plus the subsequent `b_a := b`
means acceptor `a` never re-emits the exact same `1b(a,b,e,v)` tuple; same
logic for `2a(b,v)` (uniqueness enforced by Phase2a's `¬∃ 2a(b,_)` guard)
and `2b(a,b,v)` (Phase2b sets `e_a := b`, precluding a second vote at
exactly `b`). **Reproduced on reachable states** (same pattern as TwoPhase
v2's closure of the msgs-type weakening).

### Inv2 / Inv3 / Inv4

- **Inv2** (acceptor vote history): v2 uses
  `e_a ≠ ⊥ ⟹ member(msgs, 2b(a, e_a, v_a))`. Phase2b atomically bundles
  the state update with `add(msgs, 2b(...))`. **Reproduced** (same content
  as v1).
- **Inv3** (1b-faithfulness — v1's weak spot): v1 dropped the first
  conjunct `maxBal[m.acc] \geq m.bal`. v2 explicitly restores both
  conjuncts (§5, L258-264): the first follows from Phase1b setting
  `b_a := b` exactly when sending `1b(a, b, _, _)` plus A's monotone
  `add`-only dynamics ensuring `b_a` never decreases. **Reproduced
  (v1 weakening closed).**
- **Inv4** (2a-uniqueness + showsSafe — v1's other weak spot): v2
  explicitly includes the DidNotVote side-clause via A.member on the
  bounded ballot interval `(e_q, b)` (§5, L266-282). **Reproduced
  (v1 weakening closed).**

### Consensus (safety)

v2 stitches D (QIL) + A (vote monotonicity + membership witnesses) + C
(argmax-receive) into an induction-on-ballot-number argument that closes
`v₁ = v₂` for any two chosen pairs. The per-primitive accounting at §4
L216-224 shows each step backed by a named primitive with no pigeonhole
fallback outside the bundle. The argument is a faithful QA-native
translation of Lamport's induction. **Reproduced with no ≥4 proof-gap**
(v1 had the pigeonhole fallback at N≥4; v2 closes it via D).

## Primitive load-bearing assessment

### Primitive A (monotone msgs) — LOAD-BEARING

Does Paxos's `msgs` equal `SUBSET Message`? Yes — `Paxos.tla:70`
declares `msgs \subseteq Message` and every mutation is
`msgs' = msgs \cup {m}` (L82). Set semantics, not multiset. The {1,2}
encoding is faithful under A1 shift exactly as TwoPhase v2.

Do 4 actions invoke A? Phase1a (L91-92) sends 1a; Phase1b (L100-106) sends
1b; Phase2a (L128-141) sends 2a; Phase2b (L152-158) sends 2b. **All 4 Send
actions invoke A.add.** Additionally Phase1b's guard tests `1a(b) ∈ msgs`
(L101), Phase2a's guard tests `1b(...)` quorum existence (L131), Phase2b's
guard tests `2a(b,v) ∈ msgs` (L152) — **3 actions additionally invoke
A.member.** Total: 4/4 actions on A.add, 3/4 on A.member.

The TypeOK reachability bound (v1's `qa-weaker-than-tla` source) is closed
by A's self-disabling logic above. Closes a v1 weakening AND supplies the
algebra.

**Verdict: LOAD-BEARING.**

### Primitive C (lub / argmax) — LOAD-BEARING

Does TLA+'s Phase2a compute a max-over-promise-ballots argmax? Yes
verbatim (`Paxos.tla:137-139`):

```
\E m \in Q1bv : /\ m.mval = v
                /\ \A mm \in Q1bv : m.mbal \geq mm.mbal
```

This IS "v is the value reported by the 1b with the largest `m.mbal` in
the quorum" — i.e., `v = v_{argmax m.mbal}`. v2's `e* = lub({e_q : q ∈ Q})`
with `v = v_{q*}` where `q* = argmax e_q` is a direct transcription (§3 row
Phase2a, §5 Inv4_b).

v1 flagged this as the single deepest Mode-B Paxos gap ("max-over-witness;
Wildberger has no lattice-lub"). v2 closes it with C. The argmax is the
central proof-carrying step of the induction (§4 closing paragraph:
"argmax picks v₁"). Non-trivial, non-rename use — v1's bundle literally
could not express it; v2's can, and uses it exactly once where needed.

**Verdict: LOAD-BEARING.** Crucially, this is the primitive that TwoPhase
v2's C failed to earn — TwoPhase had no argmax to host, so C there was a
rename; Paxos has a genuine argmax, so C here is algebraic.

### Primitive D (projective subspace lattice) — LOAD-BEARING (central check)

**Independent verification of v2's arithmetic:**

- **N=3, q=2.** Q₁={1,2}, Q₂={1,3}. span(Q₁) = {[a:b:0]}, span(Q₂) =
  {[a:0:c]}, intersection = {[a:0:0]} = {e₁}, dim 0. |Q₁∩Q₂|−1 = 1−1 = 0.
  Grassmann floor: 2+2−3−1 = 0. ✓
- **N=5, q=3.** Q₁={1,2,3}, Q₂={3,4,5}. span(Q₁) = {[a:b:c:0:0]},
  span(Q₂) = {[0:0:c:d:e]}, intersection = {[0:0:c:0:0]} = {e₃}, dim 0.
  |Q₁∩Q₂|−1 = 1−1 = 0. Grassmann floor: 3+3−5−1 = 0. ✓ tight.
- **N=7, q=4.** Q₁={1,2,3,4}, Q₂={4,5,6,7}. span(Q₁) supports on {1..4},
  span(Q₂) on {4..7}, intersection supported on {4} only. dim 0.
  |Q₁∩Q₂|−1 = 0. Grassmann floor: 4+4−7−1 = 0. ✓ tight.
- **Majority closure.** q ≥ ⌊N/2⌋+1 gives 2q ≥ N+2 (N even) or N+1
  (N odd); q₁+q₂−N ≥ 2 (even) or ≥1 (odd). Either way ≥1 ⇒ non-empty. ✓

**Does v2 SHOW the dimension-formula arithmetic, or merely cite Grassmann?**
v2 §4 derivation steps 1-6 walk through: (1) coordinate-subspace form,
(2) intersection-of-coordinate-subspaces = coordinate-subspace-of-index-
intersection (pure set arithmetic on indices, not field machinery),
(3) dimension = |Q₁∩Q₂|−1, (4) Grassmann inequality, (5) combination,
(6) majority arithmetic. Each step is elementary and worked. This is
Grassmann-as-derivation, not Grassmann-as-name-drop.

**Does D produce something TLA+'s quorum axiom does not?** Yes.
TLA+ axiomatizes `Q1 ∩ Q2 # {}` — an assumption. D provides a
**constructive derivation** of non-empty intersection from the dimension
formula plus the standard-basis embedding, and gives the closed-form bound
`|Q₁ ∩ Q₂| ≥ q₁ + q₂ − N` that TLA+ never states. TLA+ silently requires
the user to trust that the Quorum constant satisfies the axiom; D shows
why any majority family does (without being told). That IS genuine added
structure.

**Is D just pigeonhole in projective disguise?** The *inequality* is
pigeonhole. The *derivation mechanism* is projective-dimension
arithmetic. v2 is honest about this at object-model §Primitive-D L428-430:
"the bound is elementary combinatorics... what makes it QA-native is that
D DERIVES it from the dimension formula." That honesty is correct —
D contributes the derivation, not the bound.

**Verdict: LOAD-BEARING.** The quorum-intersection step closes for
arbitrary N via D, with no pigeonhole fallback and no enumeration.

## Closed-form counts (Contribution-3 marker)

### Closed form 1: `|Q₁ ∩ Q₂| ≥ q₁ + q₂ − N`

Correctness: verified small-N (above). Tight at maximally-disjoint
configuration.

Novelty: the bound is elementary combinatorics; D's derivation from the
dimension formula is the QA-native content. Not a "new combinatorial
identity," but a **derivation mechanism** that replaces the axiom with a
lemma. **Correct; load-bearingly derived from D.**

### Closed form 2: `C(N, q) · q · C(N − q, q − 1)`

- **N=3, q=2**: 3·2·C(1,1) = 3·2·1 = 6 ordered (3 unordered). Matches v1's
  enumeration (`L₁₂, L₁₃, L₂₃`, 3 lines, each pairwise-meet). ✓
- **N=5, q=3**: 10·3·C(2,2) = 10·3·1 = 30 ordered. ✓
- **N=7, q=4**: 35·4·C(3,3) = 35·4·1 = 140 ordered. Enumeration-consistent
  (maximally-disjoint pairs have |Q∩Q'|=1).

Derivation logic (v2 §6 L302-306): pick Q₁ `C(N,q)` ways; pick shared
element in Q₁ `q` ways; pick remaining q−1 elements of Q₂ from N−q
available `C(N−q, q−1)` ways. This is the standard elementary count for
ordered pairs with intersection exactly 1. **Correct; derivation is
elementary not Wildberger-specific, but uses A + D's span structure to
frame it.**

### Closed form 3: monotonicity under generator expansion

v2 asserts A's free-commutative-monotone-monoid + D's N-scaling give
reachable-state monotonicity. For Paxos, the substantive content is that
expanding (Value, Ballot, Acceptor) adds coordinates to A's base set X and
new basis directions to D's ambient `P^{N-1}(Z)` without removing any
prior trajectory. This IS a real theorem; it is also fairly automatic
once A is in place (monotone monoids are monotone under base-set
inclusion by construction). On the Contribution-3 marker this reads
similarly to TwoPhase v2's monotonicity claim — correct but "tautologous
by monoid construction."

**Correct; derivation is mechanically automatic.**

### Closed-form summary

- **3/3 correct** (vs TwoPhase v2's 2/3 where one was flatly wrong).
- **2/3 load-bearingly derived from augmented primitives:**
  (1) D-derivation of intersection bound, (2) elementary but framed via
  A+D span structure. Closed form 3 is automatic once A is in place
  (this is typical of A-based monotonicity claims — it is the *point*
  of choosing a monotone monoid).
- **Flagship count is correct this time.** No analog of TwoPhase v2's
  false `2^{N+1}`.

## Consensus safety closure

Does v2 stitch D's QIL to a full consensus-safety argument, or stop at
the lemma?

v2 §4 "QA-native consensus safety (closing the argument)" (L203-225):

1. `chosen(b₁,v₁), chosen(b₂,v₂)` with `b₁ ≤ b₂`.
2. Phase2a(b', v') with `b' > b₁` fires under a 1b quorum Q'; D-QIL gives
   a* ∈ Q₁ ∩ Q' (Q₁ = witness to chosen(b₁,v₁)).
3. a*'s `e_{a*} ≥ b₁` (voted at b₁, A.monotone on e_a).
4. Inv3 faithfulness (A.member on 2b) gives v_{a*} = v₁.
5. Phase2a's argmax (C.lub) picks v_{q*} where q* is the top-e acceptor in
   Q'. For any q ∈ Q' with e_q > e_{a*}, the ballot e_q was itself
   previously chosen with the same value, by induction hypothesis.
6. Therefore v' = v₁, induction closes.

Every step is backed by a named primitive (D, A, C) or the ballot
induction (T1). The induction step in (5) is the one subtle move — v2
invokes "induction hypothesis on ballot order" to handle the case where
some q' has higher e than a*. This is the classical Paxos argument and is
invoked correctly (q'·s e > b₁ means q' voted at a ballot c with
`b₁ ≤ c < b'`; by the induction hypothesis any chosen value at c equals
v₁, and C.argmax's tie among the top returns a value matching the highest
witnessed vote — which is v₁). The stitching is explicit, not
hand-waved.

**Consensus proof closes, contribution-3 bar on safety-closure met.**

## Four-axis re-assessment

| Axis | v1 | v2 |
|---|---|---|
| 1. Order / ballot structure | Partial (integer `<`; spread-poly ornamental) | Captured (integer `<` + C.lub; spread polys dropped honestly) |
| 2. Quorum intersection | Captured at N=3 only (P² lines) | **Captured arbitrary N** via D + std-basis placement (dimension-formula closed form) |
| 3. Message monotonicity | Gap (repeat from TwoPhase) | **Captured** via A (as TwoPhase v2) |
| 4. Guarded receipt / argmax | Gap (new to Paxos) | **Captured on argmax side** via C; guard-side projection strain (B') deferred |

Three axes fully captured, one mostly captured. v1's Paxos-specific
showstopper (Axis 4 argmax) is closed via C; v1's scale-gap (Axis 2 at
N≥4) is closed via D.

## Contribution score (0-4)

- v1 Paxos (no aug): 2.
- TwoPhase v2 (A+C): 2 — A worked, C was a rename on TwoPhase, closed-form
  count was wrong.
- Paxos v2 (A+C+D): claimed 3.

Contribution-3 markers (README §Specific markers):

| Marker | Present? | Evidence |
|---|---|---|
| Generator-relative structure | Weak | Actions are (guarded) translations + A.add + C.lub; no σ/μ/λ₂/ν algebra |
| SCC / orbit organization | No | Monotone DAG (A + integer ballot); structurally unavailable |
| Closed-form counts | **Yes** | 3/3 correct, derivation D-driven on (1) |
| Failure-class algebra | Yes | Q_r null-cone on v-axis closes safety (was declarative in v1) |
| Monotonicity under generator expansion | Yes (but automatic) | A-monoid + D-scaling; mechanical |

Three of five markers positively captured (closed-forms, failure-class,
monotonicity), with the closed-form marker finally carrying a
*D-driven arithmetic derivation* (unlike TwoPhase v2's false flagship).
All three augmented primitives are non-decoratively load-bearing. v1's two
weakenings (Inv3 first conjunct, Inv4 DidNotVote) are both closed. The
Consensus proof closes in-bundle with no pigeonhole fallback at N≥4.

Did D do what TwoPhase v2's C failed to do (non-decorative closed-form
derivation)? **Yes.** TwoPhase v2 had A load-bearing and C as rename; v2's
flagship count was wrong. Paxos v2 has A+C+D all load-bearing; flagship
count (intersection bound) is D-derived correctly; proof closes. This is
the augmentation-pathway working as designed.

Could it be 4? Would require generator-algebra depth (σ/μ/λ₂/ν shape) or
an orbit/SCC decomposition, neither of which Paxos's monotone-DAG
reachability supports. v2 honestly declines (§8 rationale, L383-385).

**Final Contribution: 3 (Strong).**

**Justification:** Three augmented primitives (A monotone multiset,
C lattice-lub/argmax, D projective subspace lattice in `P^{N-1}(Z)`) all
earn load-bearing use, each closing a specific v1 gap: A closes the msgs-
counter weakening on Recovery and covers 4/4 Send actions + 3/4 member
tests; C closes the argmax-receive gap by directly transcribing Phase2a's
max-over-promise-ballots rule; D replaces v1's N=3-only P² line-meet with
a dimension-formula derivation that gives `|Q₁ ∩ Q₂| ≥ q₁ + q₂ − N` at
arbitrary N via standard-basis placement in `P^{N-1}(Z)`. Three closed-
form counts, all correct (contrast with TwoPhase v2's false `2^{N+1}`),
with the intersection bound load-bearingly D-derived. Consensus safety
closes in-bundle via the full induction stitch (D → A → C → induction
hypothesis → v' = v₁), no pigeonhole fallback. v1's Inv3-first-conjunct
and Inv4-DidNotVote weakenings both recovered via A.member.

## Failure taxonomy tags

- **`ornamental-overlay`**: does NOT fire on D (load-bearing; verified
  derivation), A (4/4 Send actions, closes TypeOK weakening), or C (closes
  v1's argmax-receive gap, non-trivial use). Does NOT fire on Q_r either
  (v1 had it declaratively; v2 lifts it to closing the Consensus equality
  witness).
- **NOT `proof-gap`** — Consensus closes at arbitrary N in-bundle.
- **NOT `wrong`** — flagship closed form (intersection bound) verified
  correct at N=3, 5, 7 plus majority arithmetic.
- **NOT `qa-weaker-than-tla`** on any scoring target — all v1 weakenings
  closed.
- **NOT `qa-stronger-than-tla`** — encoding is tight.
- **NOT `no-mapping-exists`**, **NOT `invariant-inexpressible`**,
  **NOT `orbit-mismatch`**.

Taxonomically clean, matching the Contribution-3 bar.

## Blindness check

v2 declares (L3-12) reads confined to prompt, object model, README,
CLAUDE, v1 attempt/diff, TwoPhase v2 attempt. NOT read: any
`ground_truth/*`, fidelity_audit, aug-check docs.

- Inv2/Inv3/Inv4 bodies: v2's §5 reformulations use A.member over msgs
  multiset; do NOT copy TLA+'s `IF maxVBal[a]=-1 THEN maxVal[a]=None ELSE
  <<maxVBal[a], maxVal[a]>> ∈ votes[a]` (`Paxos.tla:204-206`), nor the
  verbatim ShowsSafeAt existential-over-c structure (`Voting.tla:118-122`).
  v2's DidNotVote re-derivation "no q in Q has voted at any ballot
  strictly between e_q and b" is first-principles reasoning from the
  promise semantics — arrives at the same content as ground truth via a
  different phrasing.
- `showsSafe` coinage (lowercase) carried from v1; prompt uses the English
  phrase "shows safe" (`prompts/paxos.md:83`) — not a leak.
- `votes`, `VotedFor`, `DidNotVoteAt`, `CannotVoteAt`, `SafeAt`,
  `OneValuePerBallot`, `VotesSafe`, `ChosenAt` — do NOT appear in v2.
  `chosen(b,v)` is defined via A.member on 2b at §4 L123, not referenced
  as `Voting!chosen`.
- Identifiers `maxBal, maxVBal, maxVal, msgs, Quorum, Ballot, Value,
  Acceptor, Phase1a, Phase1b, Phase2a, Phase2b` all prompt-leaked
  (`prompts/paxos.md:42-90, 106-108`), not blindness breaks.
- Sentinel choice: v2 uses `⊥` (tagged symbol, A1 shift) instead of TLA+'s
  `-1`; does not copy the specific `-1` constant.

**Blindness pact: HELD.**

## Delta assessment

- v1 (2) → v2 (**3**). Delta = +1.
- **Where did D help?** Axis 2 scale-gap. v1's P² correspondence was
  local to N=3; v2's `P^{N-1}(Z)` + standard-basis placement lifts QIL to
  a closed-form dimension identity at arbitrary N. This is the single
  largest delta.
- **Where did A help?** Axis 3 (TwoPhase-repeat gap); TypeOK-msgs
  weakening closed via {1,2}-bounded multiset encoding. Also A.member
  closes Inv3-first-conjunct and Inv4-DidNotVote weakenings.
- **Where did C help?** Axis 4 argmax side. Phase2a's `argmax m.mbal` is
  directly C.argmax; v1 had no primitive for this and flagged it as the
  deepest Paxos-specific gap. Crucially, unlike TwoPhase v2 (where C was a
  rename because TwoPhase has no actual argmax), Paxos hosts a genuine
  argmax so C earns its use.
- **Where did the bundle top out?** Contribution-4 markers
  (generator-algebra depth, SCC/orbit organization) remain structurally
  unavailable on Paxos's monotone-DAG reachability. Guard-side
  idempotent-projection strain (B') still present on Phase1b/Phase2b but
  not scoring-critical.
- **Implication for pathway**: The augmentation pathway *works* when the
  spec has a primitive-demand the augmentation targets. Paxos v2 is a
  proof of concept: D specifically addresses Paxos's quorum-intersection
  scale-gap and lands. TwoPhase v2 was a negative case for C because
  TwoPhase has no argmax; Paxos is a positive case for C because it does.
  Targeting matters — augmentation isn't universal.
- **Clifford pivot choice**: the safety-ceiling hypothesis
  (bundle caps at 2 on distributed protocols) is **refuted on Paxos** —
  the bundle + A+C+D reaches 3. Clifford pivot is now optional, not
  forced. Recommendation: continue iterating augmentation on other
  distributed protocols before committing to a GA pivot.

## Overall verdict

- **Recovery: 5/5 reproduced.** (TypeOK reproduced, v1 weakening closed;
  Inv2/Inv3/Inv4 reproduced, both v1 weakenings closed; Consensus safety
  closes at arbitrary N in-bundle.)
- **Contribution: 3 (Strong).** Three closed-form markers present,
  closed-form counts correct and D-derived, three augmented primitives
  load-bearing.
- **Per-primitive: A [LOAD-BEARING], C [LOAD-BEARING], D [LOAD-BEARING].**
- **Closed forms: 3/3 correct, 2/3 load-bearingly derived** (intersection
  bound D-derived; pair count D-framed but combinatorially elementary;
  generator-expansion monotonicity automatic under A).
- **Four-axis: [captured | captured arbitrary N | captured | captured
  argmax side, guard-side B' deferred].**
- **Delta from v1: +1.**
- **Augmentation-v2 verdict: SUCCESSFUL.** D does what v2 claimed: lifts
  the projective-intersection argument from N=3 to arbitrary N via a
  dimension-formula closed form, not enumeration, not pigeonhole outside
  the bundle.
- **Blindness held: yes.**
- **Safety-ceiling hypothesis: REFUTED** (v2 reaches 3 on Paxos). The
  bundle + {A, C, D} is not capped at 2 on distributed protocols; the
  previous two-spec cap was a primitive-demand gap, not a structural
  ceiling of the framework. Clifford pivot is now optional.
- **Headline:** Paxos Mode B v2 under Augmentation v2 reaches Contribution
  3: Primitive D's standard-basis placement in `P^{N-1}(Z)` lifts v1's
  N=3-only P² line-meet to a dimension-formula closed-form derivation of
  `|Q₁ ∩ Q₂| ≥ q₁ + q₂ − N` at arbitrary N, Primitive A covers all 4
  Send-actions and closes v1's TypeOK + Inv3 + Inv4 weakenings, Primitive
  C directly transcribes Phase2a's max-over-promise-ballots argmax
  (v1's deepest gap), Consensus safety closes in-bundle via the full
  D→A→C induction stitch with no pigeonhole fallback, and three closed-
  form counts are all correct (contrast TwoPhase v2's false flagship) —
  refuting the two-spec safety-ceiling hypothesis and making the Clifford
  pivot optional.
