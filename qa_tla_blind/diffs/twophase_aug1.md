# TwoPhase — Mode B Scored Diff v2 (Augmented)

## Relation to v1

- v1 Contribution: **2** (Q_r null-cone load-bearing on safety; dynamics-axis gaps on msgs-as-counter and guarded translations).
- v2 claim: **Contribution 3** via Primitives A (monotone multiset) + C (lattice-lub) load-bearingly, three closed-form counts.
- My job: verify the delta is real.

## Ground truth recap

Line refs on `TwoPhase.tla` / `TCommit.tla` already recorded in `diffs/twophase.md`
(v1 diff). Key re-usable fact: `msgs \subseteq Message` at `TwoPhase.tla:60`. Each
`Send`-style action is `msgs' = msgs \cup {m}` (L92, L101, L110): monotone
add-only **set**, not a multiset — TLA+ is idempotent (a duplicate add is a no-op).

## Per-invariant scoring (Recovery)

### TPTypeOK

- v1 carried a `qa-weaker-than-tla (minor)` tag: `msgs` encoded as unbounded
  integer counters `(m_{P,i}, m_C, m_A) ∈ Z_{≥1}^{N+2}`.
- v2 re-encodes `msgs` as a Primitive-A monotone multiset `M : X → ℤ_{≥1}`
  with the reachability-side bound `msgs(x) ∈ {1, 2}` asserted as clause (†)
  at `attempts/twophase_aug1.md:290`.
- **Key verification: is `msgs` a SET or a MULTISET in TLA+?** `TwoPhase.tla:60`
  declares `msgs \subseteq Message` and every mutation is `msgs' = msgs \cup {m}`.
  Union-with-a-singleton of a set is idempotent: a duplicate add is a no-op. So
  TLA+'s `msgs` is a **SET**, each message either present or absent — no
  multiplicity beyond 1. Under A1 shift (0 = absent → 1 = absent; 1 = present
  → 2 = present), TLA+'s set semantics lives in `{1, 2}^X`. v2's clause (†)
  `msgs(x) ∈ {1, 2}` is **exactly** the multiset-over-set encoding with the
  A1 shift.
- v2's self-disabling argument for (†) is correct by inspection: each `add`
  is paired with a state transition that takes the acting actor out of its
  guard (RMPrepare takes RM from working → prepared; TMCommit/TMAbort take
  TM from init → committed/aborted). Each `add(msgs, x)` fires ≤ once per
  trace.
- **Verdict: reproduced** (v1's minor `qa-weaker-than-tla` tag closed on
  reachable states via Primitive A + reachability-side bound).

### TCConsistent

- v1 encoding carried unchanged: `Q_r((3,1),(1,3)) = 0 ∧ Q_b((3,1),(1,3)) = 8 > 0`
  uniquely flags committed-aborted disagreement on the 4-point Terminal lattice.
  All 6 pair enumerations already verified in v1 diff.
- Augmentation does not touch safety (by design — v2 explicitly says A+C are
  dynamics-side, not safety-side).
- **Verdict: reproduced** (carried from v1).

## Augmentation primitive load-bearing assessment

### Primitive A (monotone multiset) — LOAD-BEARING

**Action use-count check.** Per v2 §3 summary table (L259-268):

| Action | A.add | A.member |
|---|---|---|
| RMPrepare(i) | ✓ | — |
| RMChooseToAbort(i) | — | — |
| TMRcvPrepared(i) | — | ✓ |
| TMCommit | ✓ | — |
| TMAbort | ✓ | — |
| RMRcvCommitMsg(i) | — | ✓ |
| RMRcvAbortMsg(i) | — | ✓ |

3 actions use `add`, 4 actions use `member` (one, TMRcvPrepared, overlaps). Per
v2's own count: 5 of 7 actions invoke A. This matches the TLA+ ground truth:
`TwoPhase.tla:80` (TMRcvPrepared tests `Prepared(rm) ∈ msgs`), L92, L101, L110
(the three Sends), L126, L134 (the two RcvMsg guards). 5/7 confirmed by
direct count against the spec.

**Algebraic work check.**
- v1's "unbounded counters" had no native primitive — v2's monotone-multiset
  monoid (free commutative, `add`-only, product lattice) IS the algebra; `add`
  and `member` are the primitive operations. The monoid's laws
  (commutativity, associativity, idempotence-free-but-with-2-saturation on
  reachable traces) reproduce TLA+'s set semantics tightly.
- The reachability-side bound (†) is genuinely new algebraic content: v1's
  encoding admitted counters of arbitrary size; v2 proves `msgs(x) ∈ {1, 2}`
  via paired self-disabling transitions.
- This closes v1's `qa-weaker-than-tla` tag on TPTypeOK. Real delta.

**Verdict on A: LOAD-BEARING.** Non-trivially active on 5 of 7 actions, closes
a v1 weakening, supplies the algebra for message-pool invariants.

### Primitive C (lattice-lub / argmax) — USED-WEAK

**TMCommit guard compression check.** v1 had `∀ i ∈ RM : χ_i = 2` (N-fold
conjunction over A1-shifted characteristic vector, `attempts/twophase.md:75`).
v2 has `lub_⊆(tmPrepared) = RM` (`attempts/twophase_aug1.md:168`).

- Are these semantically equivalent? `lub_⊆(tmPrepared) = RM` on the subset
  lattice: the least upper bound of tmPrepared under `⊆` among subsets ≤ RM
  is tmPrepared itself (idempotent). Setting it equal to RM says
  `tmPrepared = RM`, which is equivalent to `∀ i ∈ RM : i ∈ tmPrepared`,
  which is equivalent to v1's `∀ i : χ_i = 2`.
- **This is a rename, not a compression.** `tmPrepared = RM` IS the TLA+
  ground-truth form literally (`TwoPhase.tla:90`). The lattice-lub framing
  is syntactic sugar — it replaces an N-fold conjunction with a single
  equality, but the equality itself is already a TLA+ primitive. No
  algebraic work is done beyond the rename.

**Height-map monotonicity check.** v2 defines `h(init)=1, h(committed)=h(aborted)=2`.
The monotonicity theorem says `h(TM_state)` is non-decreasing along traces.

- This is genuine, but **trivial on a 2-element totally ordered image**.
  Monotonicity on `{1, 2}` with a single irreversible transition `1 → 2` is
  a one-line observation: the transition exists, its inverse does not.
  Lub/lattice vocabulary adds nothing beyond "TM can't leave its terminal
  state" — which follows directly from TM action guards requiring
  `tmState = "init"` (L89, L99).
- v2 itself flags this at L152: "Primitive C on the TM trajectory is a
  monotone-height invariant, not a terminal-discriminator." Honest but
  weak.

**Argmax/showsSafe applicability.** v2 correctly notes C's argmax component
(Paxos's load-bearing use) doesn't fire on TwoPhase (L66-71). No argmax is
needed by the TwoPhase invariants.

**Verdict on C: USED-WEAK.** The two claims (N-fold → lub-equality, height-
map monotonicity) are syntactic reframings of already-obvious TLA+ facts.
Non-ornamental (v2 does attach lattice vocabulary to genuine structural
features) but does no algebraic work — C does not drive any
contribution marker. Borderline between USED-WEAK and DECORATIVE; I land on
USED-WEAK because v2 is honest about the weakness and uses the vocabulary
correctly.

## Closed-form counts (Contribution-3 marker)

### Closed form 1: reachable terminal configurations = `2^{N+1}` — **WRONG**

v2's derivation (L392-440):
- Branch TM=committed: RM final ∈ {prepared, committed} → 2^N.
- Branch TM=aborted: RM final ∈ {prepared, aborted} → 2^N.
- Total: 2 · 2^N = 2^{N+1}.

**Manual enumeration at N=1.** Reachable (tmState, rmState[r1]) pairs (ignoring
tmPrepared and msgs for the moment, checking whether v2's 2^{N+1}=4 count can
even fit in the RM × TM state space):

| (tmState, rmState) | reachable? | path |
|---|---|---|
| (init, working) | ✓ | initial |
| (init, prepared) | ✓ | RMPrepare |
| (init, aborted) | ✓ | RMChooseToAbort |
| (init, committed) | ✗ | committed needs Commit∈msgs → TMCommit → tmState=committed |
| (committed, working) | ✗ | TMCommit needs tmPrepared=RM, all RMs must be prepared |
| (committed, prepared) | ✓ | RMPrepare, TMRcvPrepared, TMCommit; RM stays prepared |
| (committed, aborted) | ✗ | no Abort msg on this branch; aborted needs RMRcvAbortMsg or RMChooseToAbort-while-working, but working is gone by time TMCommit fires |
| (committed, committed) | ✓ | ...RMRcvCommitMsg |
| (aborted, working) | ✓ | TMAbort; RM stays working |
| (aborted, prepared) | ✓ | RMPrepare, TMAbort; RM doesn't receive Abort |
| (aborted, aborted) | ✓ | TMAbort then RMRcvAbortMsg, or RMChooseToAbort then TMAbort |
| (aborted, committed) | ✗ | no Commit msg |

**9 reachable (tmState, rmState) pairs at N=1.** Not `2^{N+1} = 4`.

v2's error: on TM=aborted branch, RM final state is NOT restricted to
{prepared, aborted}. A working RM after TMAbort fires can STAY WORKING
(RMRcvAbortMsg is optional; RMPrepare is still enabled but optional too).
v2 silently drops "working" from the set of final rmState values on the
aborted branch, missing a third choice per RM. Correct count on aborted
branch: `3^N` (each RM in {working, prepared, aborted}).

Even more strictly: if "terminal" means "no enabled action changes state"
(strict dead-end), then on each branch every RM must reach the TM's
terminal state: TM=committed → all RMs committed (1 state); TM=aborted →
all RMs aborted (1 state). Total strict terminals: **2**, not 2^{N+1}.

If "reachable configurations" (full state valuation) is meant, at N=6 TLC
reports 50816 reachable states (`TwoPhase.tla:173-174`). `2^{6+1} = 128`.
Off by a factor of ~400.

Under no natural interpretation does `2^{N+1}` match the TwoPhase state
space. **Closed form 1 is WRONG.**

### Closed form 2: `|support(msgs_terminal)| ≤ N + 1` — **CORRECT but WEAK**

- TM=committed branch: support includes {Prepared(i) : i ∈ RM} ∪ {Commit}
  = N+1 tags. Equality holds (all RMs must prepare to enable TMCommit).
- TM=aborted branch: support includes {Abort} ∪ (subset of {Prepared(i)}).
  At most N+1 tags.

Bound holds. Is it load-bearingly derived from A? Partially — the bound
follows from (a) the support-is-subset observation (A.support view is a
derived coord) and (b) the TM-branch mutual exclusion (C.height-map).
But the bound itself is also derivable directly from the TLA+ spec by
counting possible senders + TM's single broadcast — no novel algebraic
content. A gives it cleaner vocabulary; it does not create it.

**Verdict: correct but elementary.** Load-bearing on A only in the weak sense
that A defines "support" as a primitive.

### Closed form 3: trajectory length `N + 1 ≤ k ≤ 3N + 1` — **CORRECT**

Verified by direct path-counting:
- Minimum on aborted branch: TMAbort (1) + N × RMRcvAbortMsg (if we stipulate
  all RMs end aborted) = N+1.
- Maximum on committed branch: N × RMPrepare + N × TMRcvPrepared + TMCommit +
  N × RMRcvCommitMsg = 3N + 1.

Both bounds match. This bound is genuinely closed-form in N, and the
derivation uses A's monotonicity (each `add` fires at most once) plus
action self-disabling. Load-bearingly uses A.

**Verdict: correct and A-load-bearingly derived.**

### Closed-form summary

- **1/3 closed forms correct-and-A-or-C-load-bearingly derived (form 3).**
- **1/3 correct but derivable without augmentation (form 2).**
- **1/3 WRONG under every natural interpretation (form 1).**

Not sufficient to carry a Contribution-3 marker on the "closed-form counts"
axis. The flagship count v2 presents (form 1, the "2^{N+1} configurations")
is wrong, and a Contribution-3 marker requires at least one genuine closed
form. Form 3 (T1 bound) is the only genuinely load-bearing closed form,
and T1 path-length bounds alone are not Contribution-3 content — they
are Contribution-2 content (v1's Bakery calibration at 1 includes path-
time counts).

## Contribution score (0-4)

Contribution-3 rubric markers (README §Specific markers, L160-170):

1. **Generator-relative structure** (σ/μ/λ₂/ν-style named operators): **NO**.
   v2 remains a free commutative monoid + translations + idempotent
   projections + lattice-joins. Same as v1; no new generator algebra.
2. **SCC / orbit organization**: **NO** (inapplicable — monotone DAG). v2
   honestly declares this at L536-538.
3. **Closed-form counts**: **1/3 correct-and-load-bearing** (T1 bound). Form
   1 is wrong, form 2 is elementary. Form 3 alone is T1-bound content —
   Contribution-2 calibration already includes T1 counts.
4. **Failure-class algebra**: **NO**. Same single class as v1 (Q_r null-cone).
   v2 considered but did not produce a multi-class algebra.
5. **Monotonicity-under-generator-expansion**: **WEAK**. v2 asserts
   `INV_TM_monotone` and `INV_tmPrepared_monotone`. On 2-element `{1,2}` the
   height monotonicity is the observation "no inverse transition" — not a
   generator-expansion theorem. `tmPrepared ⊆ tmPrepared'` is the TLA+
   guard's literal content. Neither is the generator-expansion shape of
   QA control theorems (adding Σ can only merge, not split SCCs).

Zero strong markers; one weak marker on closed-forms (form 3); one weak
marker on monotonicity. That is **Contribution 2 content**, the same
floor as v1.

**Final score: 2 (Useful).**

**Justification (one line):** Primitive A is genuinely load-bearing on 5/7
actions and closes v1's msgs-type weakening, but Primitive C is used-weak
(rename + trivial 2-element monotonicity), and the Contribution-3 evidence
v2 stakes its claim on — the closed-form count `2^{N+1}` — is wrong under
every natural interpretation (enumerated above for N=1, yielding 9 reachable
(tmState, rmState) pairs, not 4); the other two closed forms are either
elementary (support bound) or Contribution-2-calibrated (T1 bound).

## Failure taxonomy tags

- **`ornamental-overlay` (partial, per-primitive):** fires on Primitive C
  (used-weak on TwoPhase: TMCommit guard is a rename, height monotonicity
  is trivial on 2 elements). Does NOT fire on Primitive A (load-bearing).
- **`wrong`:** fires on closed-form 1 (`2^{N+1}` reachable terminal
  configurations). The count is wrong at N=1 (actual: 9 reachable
  (tmState, rmState) pairs, or 2 strict terminals, or 50816 full states at
  N=6; `2^{N+1}` matches none of these).
- **`proof-gap`:** minor — v2 does not notice the "working" final-state
  option on the TM=aborted branch, which is the immediate cause of the
  2^{N+1} error.
- **NOT `qa-weaker-than-tla`** on TPTypeOK (A closes v1's tag).
- **NOT `invariant-inexpressible`**, **NOT `orbit-mismatch`**,
  **NOT `no-mapping-exists`**.
- **NOT `qa-stronger-than-tla`** on reachability (v2 doesn't assert the
  closed form as an invariant the protocol must satisfy — it asserts it as
  a structural count, which is simply wrong rather than an over-strict
  constraint).

## Blindness check

v2's blindness pact (L5-11): read `prompts/twophase.md`,
`mode_b/wildberger_object_model.md`, `README.md`, `CLAUDE.md`,
`attempts/twophase.md` (v1), `diffs/twophase.md` (v1 diff). NOT read:
`ground_truth/*`, `fidelity_audit.md`, `augmentation_v1_check.md`,
Paxos files.

- **TCConsistent predicate body** not copied from `TCommit.tla:54-60`.
  v2 carries v1's geometric reformulation unchanged.
- **TPTypeOK body** not copied from `TwoPhase.tla:53-60`. v2 reformulates
  the message-pool clause as multiset-with-bound.
- **Action guard bodies** — v2 reuses v1's action guards as
  translations-plus-primitive-A. No direct TLA+ copy.
- Identifiers (`rmState, tmState, tmPrepared, msgs`; action names
  `RMPrepare, TMCommit`, etc.) are prompt-leaked (`prompts/twophase.md:42-53,
  69-85`), not a break.
- Reading v1 attempt/diff is permitted per Mode B v2 protocol (v1 is not
  ground truth).

**Blindness pact: HELD.**

## Delta assessment

**Did augmentation produce a genuine Contribution delta (2 → 3)?** **NO.**

**Per-primitive delta breakdown:**
- **Primitive A** delivered the advertised load-bearing use: 5/7 actions,
  closes v1's `qa-weaker-than-tla` on TPTypeOK. Genuine positive delta in
  *encoding faithfulness* (Recovery side tightened), and Primitive A itself
  earned LOAD-BEARING.
- **Primitive C** delivered two weak uses: rename of TMCommit guard, trivial
  monotonicity on a 2-element totally ordered set. No genuine delta.
- **Carryover from v1**: Q_r null-cone safety (load-bearing, unchanged).

**Where did the Contribution-3 push fail?** On the closed-form-counts marker.
v2 needed at least one genuinely new closed form that v1 could not produce.
It advertised three. One is wrong (`2^{N+1}`), one is elementary (support
bound), one is T1-bound content (already calibrated at Contribution-2 on
Bakery). The `2^{N+1}` error is specifically load-bearing: it is the
count v2 frames as its primary Contribution-3 evidence (L438-441), and it
doesn't survive enumeration at N=1.

**Structural reason TwoPhase may cap at 2 under augmentation.** v2 itself
observes (L588-595) that TwoPhase's protocol content is "1-bit-of-safety
over a monotone-accumulation DAG" and speculates the spec may cap at 3
regardless of augmentation. My verdict: it caps at **2**, because the
specific closed-form count v2 produces to earn the 3 is false, and no
other Contribution-3 marker is available (no SCCs, no multi-class failure
algebra, no generator algebra, no non-trivial monotonicity-under-generator-
expansion).

**Key implication (per scorer prompt).** The Wildberger bundle + Augmentation
v1 is *not* rescued for distributed protocols on the Contribution axis. On
the **Recovery** axis, Primitive A cleanly closes the msgs-as-counter
weakening — that is a real bundle-augmentation win. But the Contribution
ceiling on TwoPhase stays at 2. The safety-ceiling hypothesis (bundle caps
at 2 without deeper GA) gets **more support**, not less. The augmentation
improved encoding *faithfulness* (Recovery-side refinement) without
improving *structural insight* (Contribution-side lift).

## Overall verdict

- **Recovery: 2/2 reproduced** (TPTypeOK reproduced with v1 weakening closed
  by Primitive A; TCConsistent reproduced-as-v1).
- **Contribution v1: 2 → v2: 2** (no delta). The closed-form count that was
  meant to carry the 3 is wrong under every natural interpretation; the
  other two closed forms are elementary or already-Contribution-2 material.
- **Load-bearing:** A = **LOAD-BEARING**, C = **USED-WEAK**.
- **Closed forms: 2/3 correct, 1/3 A-load-bearingly derived** (T1 bound
  only; support bound correct but elementary; reachable-terminal count
  wrong).
- **Dominant tags:** `wrong` (on closed-form 1), `ornamental-overlay`
  (partial, on Primitive C), `proof-gap` (minor, missed "working" final
  state on TM=aborted branch).
- **Blindness held:** yes.
- **Headline:** TwoPhase Mode B v2 under Augmentation v1 stays at
  Contribution 2: Primitive A is genuinely load-bearing on 5/7 actions and
  closes v1's msgs-type weakening on Recovery, but Primitive C is used-weak
  (TMCommit guard compression is a rename not an algebraic gain; 2-element
  height monotonicity is trivial), and the flagship closed-form count
  `2^{N+1}` is wrong at N=1 (actual 9 reachable `(tmState, rmState)` pairs,
  or 2 strict terminals) — v2 silently drops the "working" final-state
  option on the TM=aborted branch; the safety-ceiling hypothesis (Wildberger
  bundle caps at 2 on distributed protocols without deeper GA) gets more
  support, the augmentation improved encoding faithfulness (Recovery-side)
  without lifting Contribution.
