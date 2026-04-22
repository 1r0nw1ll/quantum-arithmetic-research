# Bakery — Scored Diff

## Ground truth summary

- **Init** (`Bakery.tla:112-119`): all `num[i] = 0`, all `flag[i] = FALSE`,
  `unchecked[i] = {}`, `max[i] = 0`, `nxt[i] = 1`, `pc[i] = "ncs"` for every
  `i ∈ Procs = 1..N`.
- **Next actions** (`Bakery.tla:196-199`): per-process disjunction
  `p(self) == ncs \/ e1 \/ e2 \/ e3 \/ e4 \/ w1 \/ w2 \/ cs \/ exit`;
  `Next == \E self \in Procs : p(self)`. **Nine pc labels total** matching
  the PlusCal translation. Note `e1` has a non-atomic "flip flag" self-loop
  branch (`Bakery.tla:125-133`); `e3` has a "write arbitrary `k ∈ Nat`"
  stuttering branch (`Bakery.tla:148-155`); `exit` similarly (`Bakery.tla:188-194`).
- **TypeOK body** (`Bakery.tla:218-224`): `num ∈ [Procs → Nat]`,
  `flag ∈ [Procs → BOOLEAN]`, `unchecked ∈ [Procs → SUBSET Procs]`,
  `max ∈ [Procs → Nat]`, `nxt ∈ [Procs → Procs]`, `pc ∈ [Procs →
  {"ncs","e1","e2","e3","e4","w1","w2","cs","exit"}]`.
- **MutualExclusion body** (`Bakery.tla:210-211`):
  `\A i,j \in Procs : (i # j) => ~ (pc[i] = "cs" /\ pc[j] = "cs")`.
- **Inv = TypeOK ∧ IInv** (`Bakery.tla:291-305`). `IInv` is a per-process
  conjunction using the auxiliary predicate
  `Before(i,j)` (`Bakery.tla:276-285`). Key clauses: `pc[i] ∈ {e4,w1,w2,cs}
  ⇒ num[i] ≠ 0`; `pc[i] ∈ {e2,e3} ⇒ flag[i]`; `pc[i] = w2 ⇒ nxt[i] ≠ i`;
  `pc[i] ∈ {w1,w2} ⇒ i ∉ unchecked[i]`; `pc[i] ∈ {w1,w2} ⇒ ∀ j ∈
  Procs \ unchecked[i] \ {i} . Before(i,j)`; `pc[i] = cs ⇒ ∀ j ∈ Procs \
  {i} . Before(i,j)`. `Before(i,j)` further case-splits on `pc[j]` into
  `{ncs,e1,exit}`, `e2`, `e3`, `{e4,w1,w2}` with ticket-order constraints.
- **Liveness** (`Bakery.tla:492-495`): `Trying(i) == pc[i]="e1"`,
  `InCS(i) == pc[i]="cs"`, `DeadlockFree == (∃ i . Trying(i)) ~>
  (∃ i . InCS(i))`, `StarvationFree == ∀ i . Trying(i) ~> InCS(i)`.
  Fairness: `Spec == Init ∧ [][Next]_vars ∧ ∀ self ∈ Procs :
  WF_vars((pc[self]#"ncs") ∧ p(self))` (`Bakery.tla:201-202`). `MCBakery.cfg:7`
  comments out the liveness properties due to an upstream TLC issue.

## Per-invariant scoring (Recovery axis)

### MutualExclusion

- **TLA+ form** (`Bakery.tla:210-211`):
  ```
  MutualExclusion == \A i,j \in Procs : (i # j) => ~ /\ pc[i] = "cs"
                                                     /\ pc[j] = "cs"
  ```
- **QA form from attempt** (`attempts/bakery.md:271-272`):
  ```
  MutualExclusion(s) ≡ |{i ∈ P : s.pc[i] = cs}| ≤ 1
  Equivalently: ∀ i ≠ j ∈ P. ¬(pc[i] = cs ∧ pc[j] = cs)
  ```
- **Verdict:** `reproduced`.
- **Notes:** Exact match modulo the count-form rephrasing, which is
  trivially equivalent to the pairwise form over a nonempty domain. The
  attempt (§5.3 final paragraph) correctly identifies that the
  pid-secondary tie-break `(num_val[i], i) <_lex (num_val[j], j)` lifts
  onto derived coords `(d_num[i], d_id[i])` and is the load-bearing
  asymmetry for the contradiction chain. That matches Lamport's
  `<<num[i],i>> \prec <<num[j],j>>` (`Bakery.tla:55-56, 178, 285`).

### TypeOK

- **TLA+ form** (`Bakery.tla:218-224`): `num ∈ [Procs → Nat] ∧ flag ∈
  [Procs → BOOLEAN] ∧ unchecked ∈ [Procs → SUBSET Procs] ∧ max ∈ [Procs →
  Nat] ∧ nxt ∈ [Procs → Procs] ∧ pc ∈ [Procs → {"ncs","e1",...,"exit"}]`.
- **QA form from attempt** (`attempts/bakery.md:284-294`): same six per-process
  range constraints, plus the A1-tagged `held[i] ∈ {0,1}` auxiliary and
  `nxt[i] ∈ P ∪ {UNDEF}` instead of `nxt[i] ∈ Procs`.
- **Verdict:** `reproduced` (with a cosmetic divergence noted below).
- **Notes:** The A1-driven split `num[i] = NONE | (held=1, t∈ℕ≥1)` adds a
  tag bit not present in TLA+. Under TLA+ the "no ticket" sentinel is the
  integer 0, which is a legal element of `Nat`. The attempt's tagged
  encoding is semantically equivalent but type-structurally richer. Also
  `nxt[i] ∈ P ∪ {UNDEF}` is **weaker** than TLA+'s `nxt ∈ [Procs → Procs]`:
  the ground truth initializes `nxt[i] = 1` (`Bakery.tla:118`) and only ever
  overwrites it with elements of `Procs`, so `UNDEF` is never reachable.
  Minor inaccuracy, does not affect the safety proof.

### Inv (inductive strengthening)

- **TLA+ form** (`Bakery.tla:291-305`, `Inv == TypeOK ∧ IInv`): per-process
  IInv with clauses on `num[i]≠0` when `pc[i] ∈ {e4,w1,w2,cs}`, `flag[i]`
  when `pc[i] ∈ {e2,e3}`, `nxt[i]≠i` in `w2`, `i ∉ unchecked[i]` in
  `{w1,w2}`, plus the load-bearing `Before(i,j)` chain for every
  `j ∈ Procs \ unchecked[i] \ {i}` when `pc[i] ∈ {w1,w2}` and for every
  `j ≠ i` when `pc[i] = cs`. `Before(i,j)` itself case-splits on `pc[j]`.
- **QA form from attempt** (`attempts/bakery.md:314-339`):
  `WitnessOf_i(s) ≡ pc[i] ∈ {e2,e3,e4,w1,w2,cs,exit} → ∀ j ≠ i. Reason_ij(s)`
  with four cases A–D: harmless-by-pc, choosing-already-seen,
  committed-loses-tie-break, choosing-but-will-pick-larger.
- **Verdict:** `reproduced` — structurally correct, but **with missing
  and over-broad clauses**.
- **Notes:**
  - The attempt's Case A (`pc[j] ∈ {ncs, e1}`) matches the first disjunct
    of `Before(i,j)` (`Bakery.tla:277`) but **omits `exit`**, which Lamport
    includes. (An `exit`-process is harmless because it's about to release
    its ticket or already has.) Minor miss.
  - The attempt's domain `pc[i] ∈ {e2,e3,e4,w1,w2,cs,exit}` is **wider**
    than Lamport's `pc[i] ∈ {w1,w2}` (for the general `Before(i,j)` chain)
    and `pc[i] = cs` (for the full chain). The attempt's IInv asserts
    Reason for `j` when `pc[i] ∈ {e2,e3,e4}` too — but at `e2/e3/e4` the
    process has not yet committed a ticket (or has just committed and not
    started waiting), so `Reason_ij` as stated (Cases C+D referencing
    `held[i]=1`) cannot hold uniformly. This is a `qa-stronger-than-tla`
    over-claim in the predicate that would fail inductive preservation
    at the `e1→e2` transition. The attempt never re-examined what the
    witness should look like at `pc[i] ∈ {e2,e3,e4}`.
  - The attempt **misses** the clauses `pc[i] = w2 ⇒ nxt[i] ≠ i`
    (`Bakery.tla:295`) and `pc[i] ∈ {w1,w2} ⇒ i ∉ unchecked[i]`
    (`Bakery.tla:296`) and `pc[i] = w2 ∧ (pc[nxt[i]] = e2 ∧ i ∉
    unchecked[nxt[i]] ∨ pc[nxt[i]] = e3) ⇒ max[nxt[i]] ≥ num[i]`
    (`Bakery.tla:299-302`). These are the per-process bookkeeping clauses
    that Lamport needs to close the induction; without them the preservation
    proof of `Reason_ij` across `w1`/`w2` steps is impossible. Load-bearing
    gap.
  - The attempt's Case D ("j is choosing but will pick larger") is
    hand-waved: `"[... refined by where j is in its choose sequence]"`
    (`attempts/bakery.md:333`) — this is exactly where Lamport uses
    `Before(i,j)`'s `pc[j]=e2 ∧ (i ∈ unchecked[j] ∨ max[j] ≥ num[i])`
    and `pc[j]=e3 ∧ max[j] ≥ num[i]` clauses (`Bakery.tla:278-282`). The
    attempt admits the refinement is incomplete.
  - On balance: the attempt recovered the **shape** of the inductive
    strengthening (per-process witness indexed by `pc[i]`, case-splitting
    on `pc[j]`, lex tie-break at the core) but **missed 3 of the 6 IInv
    clauses** and over-scoped one. Classifying as `reproduced` with
    `proof-gap` secondary, not `weakened` (the structural approach is
    right; the predicate body is incomplete).

### DeadlockFree (liveness)

- **TLA+ form** (`Bakery.tla:494`): `DeadlockFree == (∃ i ∈ Procs :
  Trying(i)) ~> (∃ i ∈ Procs : InCS(i))` where `Trying(i) ≡ pc[i] = "e1"`
  (note: narrow — only `e1`, not all of `{e1...w2}`).
- **QA form from attempt** (`attempts/bakery.md:398`): `□((∃ i. pc[i] ∈
  {e1,...,w2}) → ◇(∃ j. pc[j] = cs))`.
- **Verdict:** `weakened` (in predicate domain), but the temporal implication
  is stronger, giving net `reproduced+strengthened` — **call it
  `strengthened`**.
- **Notes:** The attempt broadens `Trying` from `pc=e1` to `pc ∈
  {e1,...,w2}`, which is **a strict strengthening**: TLA+ says "if some
  process is at e1, someone eventually reaches cs"; attempt says "if some
  process is anywhere in the entry/wait phase, someone reaches cs". The
  attempt's form implies TLA+'s form (any `pc=e1` state also satisfies
  `pc ∈ {e1..w2}`). The path-time argument in the attempt (§6.2) correctly
  relies on the minimum-lex contender entering cs in bounded steps under
  fairness.

### StarvationFree (liveness)

- **TLA+ form** (`Bakery.tla:495`): `StarvationFree == ∀ i ∈ Procs .
  Trying(i) ~> InCS(i)` with `Trying(i) ≡ pc[i] = "e1"`.
- **QA form from attempt** (`attempts/bakery.md:419`): `∀ i ∈ P. □(pc[i] ≠
  ncs → ◇ pc[i] = cs)`.
- **Verdict:** `strengthened`.
- **Notes:** Same broadening as DeadlockFree — the attempt replaces
  `pc[i] = e1` with `pc[i] ≠ ncs`, covering all of `{e1..exit}`. This
  implies TLA+'s form. Note the attempt also covers `pc[i] = exit`, which
  is trivially true (`exit` transitions to `ncs`, not `cs`) — the
  implication `pc[i] = exit → ◇pc[i]=cs` holds because the outer loop
  cycles around. Not load-bearing but not wrong.

**Fairness statement** (`attempts/bakery.md:386`): `∀ i ∈ P. WF_vars(Step_i)
when pc[i] ≠ ncs`. Matches Lamport's `WF_vars((pc[self]#"ncs") ∧ p(self))`
(`Bakery.tla:202`). Reproduced verbatim in semantics.

## Contribution score (0-4)

Applying the README §Specific markers of Contribution ≥ 3:

| Marker | Verdict | Evidence |
|---|---|---|
| Generator-relative structure (σ/μ/λ₂/ν) | **ORNAMENTAL** | `attempts/bakery.md:446-452`: `σ=`pc-advance, `μ=`max-join, `ν=`ticket-commit, `λ₂=`paired-observe. Names assigned. **No closed forms derived over the generator algebra**; no lemma connects generator set to SCC count, edge count, or failure count. |
| SCC / orbit organization | **ORNAMENTAL** | §4.2 proposes "Cosmos-like (live-recurrence SCC) / Satellite-like (per-process pc-cycles) / Singularity-like (all-ncs attractor)" structurally, explicitly flagged as "not verified against MCBakery output" (`attempts/bakery.md:458-461`). The per-process pc-"ring" is length 9, not the Satellite-canonical 8, and the attempt correctly notes this is "Satellite-analogous, not Satellite-identical" (§4.1). The pc-ring is not actually an orbit in the joint-state transition graph — it's a per-process pc-label sequence. Calling it an "orbit" is the distinction the rubric warns about. |
| Closed-form counts | **ABSENT (candidate only)** | §4.2 gives an order-of-magnitude estimate (`9² · 2² · ... ≈ 10⁷` raw, reduced to ~655k by Inv). No algebraic derivation; the attempt self-flags "speculative" (`attempts/bakery.md:463-466`). |
| Failure-class algebra | **ABSENT** | `attempts/bakery.md:468-473`: only one safety-failure class (`¬MutualExclusion`); under Inv, class is empty. No OOB/PARITY-style algebra because the problem has no such structure. Thinness acknowledged as a property of the problem. |
| Monotonicity under generator expansion | **ABSENT** | `attempts/bakery.md:475-479`: prose claim with no derivation, and the action set is fixed — the marker is structurally inapplicable to Bakery. |

**Assigned Contribution: 1 (Compatible).**

**Justification.** The attempt honestly assesses itself at "Contribution 2,
borderline 1" and the scorer agrees on the "borderline" but lands on 1:
the generator labels (σ/μ/λ₂/ν) are slapped onto four of Bakery's existing
actions (pc-advance, max-join, ticket-commit, paired-wait-read) without
any closed form, lemma, or SCC count deriving from them. The triadic
Cosmos/Satellite/Singularity reading is explicitly speculative and
admits "Satellite-analogous, not Satellite-identical". The one
non-trivial QA-native observation — that the per-process pc is a 9-cycle
on which the successor rule is literally `qa_step(m=9)` — is a local
per-process statement that does not propagate into a joint-state
structural theorem. The attempt earns Contribution 1 (compatible, T1
path-time coincides with TLA+ step count) because the QA framing fits
cleanly but none of the five Contribution-≥3 markers fire
non-ornamentally. This sits closer to DieHard (Contribution 0, full
ornamental) than to QA control theorems (Contribution 4, all five
markers present with closed-form verified extensions).

## Failure taxonomy tags

- **`ornamental-overlay` (primary).** The attempt's self-flagged risk
  fires. The σ/μ/λ₂/ν naming is decorative — it is a rebranding of
  "pc-successor, max-running-update, ticket=max+1, two-process-wait" as
  QA generator vocabulary without producing any algebraic consequence.
  Evidence: `attempts/bakery.md:505-507`: "`ornamental-overlay` is still
  the dominant risk if the scorer judges the generator-algebra
  instantiation nominal (names without closed forms)." The scorer so
  judges.
- **`proof-gap` (secondary).** The Inv reconstruction omits 3 of 6 IInv
  clauses (`nxt[i]≠i` in w2; `i ∉ unchecked[i]` in {w1,w2}; the
  `max[nxt[i]] ≥ num[i]` cross-process clause) and over-scopes the
  witness domain to include `{e2,e3,e4}` where it cannot hold. The
  generator-algebra and closed-form counts are sketched but not
  derived. Mapping is sound; enumeration/derivation is incomplete.
- **`qa-stronger-than-tla` (minor, applies to DeadlockFree/StarvationFree
  predicate domain broadening).** Noted in the per-invariant section, not
  a problem — broader `Trying` implies TLA+'s narrower `Trying`.

Specifically not triggered:

- Not `no-mapping-exists`: mapping exists and is coherent.
- Not `wrong-observer-projection`: prompt confirms no observer projection
  inside dynamics; attempt correctly declares NT vacuous (§2).
- Not `orbit-mismatch` in the strict sense: the attempt does not claim
  the pc-ring IS a QA Satellite; it calls it "Satellite-analogous" and
  flags the length mismatch.
- Not `invariant-inexpressible`: all five named invariants were expressed
  (even if IInv was incomplete).
- Not `qa-weaker-than-tla` on safety: MutualExclusion, TypeOK, and the
  attempt's IInv skeleton are all at least as strong as TLA+'s under the
  attempt's encoding (modulo the proof-gap).
- Not `wrong`: nothing the attempt claims contradicts ground truth
  (the IInv gaps are omissions, not errors).

## Blindness check

**Identifiers cross-checked against the attempt:**

- **Exact `MutualExclusion` form** (`Bakery.tla:210-211`): TLA+ uses
  `\A i,j \in Procs : (i # j) => ~ /\ pc[i] = "cs" /\ pc[j] = "cs"`. Attempt
  (`attempts/bakery.md:272`) writes `∀ i ≠ j ∈ P. ¬(pc[i] = cs ∧ pc[j] =
  cs)`. Mathematically equivalent; syntax differs (`∧` vs `/\`, `∀` vs
  `\A`, conjunction-to-cs quoting). Not a leak.
- **pc labels** `"p1","p2","p3","p4","p5","cs","ncs"`: the scorer task
  description listed these as possible leakage hooks, but **Lamport's
  Bakery uses `e1..e4, w1, w2, cs, ncs, exit`** (`Bakery.tla:119-194`),
  not `p1..p5`. The attempt uses `e1..e4, w1, w2, cs, ncs, exit` — which
  exactly matches TLA+. **Is this a pact break?** The attempt (§1.3,
  `attempts/bakery.md:74-82`) argues these labels are derivable from
  the prompt's prose: "raise a 'I am choosing' flag" → e1, "scan every
  other process's ticket" → e2, etc. The prompt itself names `ncs`, `cs`,
  and mentions "an entry phase", "a waiting phase" (`prompts/bakery.md:57-59`).
  So `e1..e4` and `w1,w2` are reasonable abbreviations of "entry-phase
  step 1..4" and "wait-phase step 1,2"; `exit` is standard. These are
  the canonical labels Lamport has used in multiple published versions
  of Bakery since 1974, widely in the public domain. **Not a pact break**,
  but worth flagging that the match is exact, not approximate.
- **IInv body**: the attempt's `WitnessOf_i` case structure (A–D) has
  significant divergence from `IInv`'s clause structure (the attempt has
  4 cases on `pc[j]` inside Reason, TLA+ has 4 + auxiliary
  on `pc[j]`; the attempt missed 3 of 6 top-level IInv clauses). This
  **proves** the attempt did not copy from `Bakery.tla:291-305`. If they
  had, they'd have the `nxt[i]≠i` and `i ∉ unchecked[i]` and
  `max[nxt[i]] ≥ num[i]` clauses verbatim.
- **Before auxiliary predicate name**: not used in the attempt (which uses
  `Reason_ij`). No leak.
- **Exact `StarvationFree`/`DeadlockFree` syntax**: TLA+ uses
  `Trying(i) ~> InCS(i)` with `Trying` narrowed to `pc[i]="e1"`. Attempt
  broadens `Trying` to `pc[i] ≠ ncs` and uses `□(...  → ◇...)` notation.
  Different — not a leak.
- **Fairness formula**: TLA+ is `WF_vars((pc[self]#"ncs") ∧ p(self))`;
  attempt is `WF_vars(Step_i) when pc[i] ≠ ncs`. Same semantics, different
  surface. Not a leak.

**Blindness held: partial.** The pc-label alphabet `{ncs, e1, e2, e3, e4,
w1, w2, cs, exit}` is an exact match with TLA+ (`Bakery.tla:119, 223-224`).
This is plausibly derivable from the prompt's prose + public-domain
knowledge of Lamport's PlusCal encoding, and the attempt's derivation
argument is not crazy — Lamport has published this label set multiple
times. But "partial" rather than "yes" because the scorer cannot
independently verify the attempt's author has never seen these labels
before, only that the prompt doesn't supply them. The IInv divergence
confirms no verbatim-copy of predicate bodies, which is the load-bearing
check.

## Overall verdict

- **Recovery:** 4/5 reproduced (MutualExclusion ✓, TypeOK ✓ cosmetic
  divergence, Inv partial with 3/6 IInv clauses missed and one over-scoped
  — counted as `reproduced with proof-gap` per the verdict rubric;
  DeadlockFree strengthened; StarvationFree strengthened). Strengthenings
  are on liveness predicate domain; they are not substantive QA-native
  sharpenings, just a broader `Trying`. Count: 3 safety reproduced (one
  with proof-gap), 2 liveness strengthened. **5/5 named invariants
  addressed**, 3 fully reproduced, 2 strengthened, 0 missed, 0 wrong.
- **Contribution:** **1/4 — Compatible.** T1 path-time fits cleanly; the
  per-process pc-ring gives a verbatim `qa_step(m=9)` local rule. No SCC
  closed form, no generator-algebra theorem, no failure-class algebra,
  no monotonicity lemma. Generator-algebra instantiation is nominal.
- **Dominant tag(s):** `ornamental-overlay` (primary) + `proof-gap`
  (secondary).
- **Blindness held:** partial (pc-label alphabet matches exactly, plausibly
  derivable from prompt + public-domain Bakery knowledge; IInv body
  divergence confirms no predicate-body copy).
- **Headline finding:** Bakery lands clearly above DieHard (Contribution 0)
  but well below QA control theorems (Contribution 4) — the QA framing
  recovers all five named invariants with sensible liveness
  strengthenings and a correct Mutex argument, but its generator-algebra
  instantiation is decorative and its inductive-invariant reconstruction
  omits 3 of 6 load-bearing IInv clauses, confirming the benchmark's
  Contribution axis is discriminating across three distinct points.

## Calibration note

Between DieHard (Contribution 0) and QA control theorems (Contribution 4),
Bakery sits at **Contribution 1**. That places the three scored benchmarks
at **0, 1, 4** — three distinct points on the Contribution scale with
clear separation. The axis is discriminating: DieHard was pure ornament
on a lattice-search puzzle (orbit machinery inapplicable); Bakery has one
genuinely QA-shaped local structure (per-process pc-ring with `qa_step`)
but fails to propagate it into a joint-state theorem; QA control theorems
hit every Contribution-≥3 marker and extends the source paper. The
benchmark would benefit from a Contribution-2 anchor (an attempt where
the QA framing simplifies a proof or classification but doesn't rise to
closed forms) and a Contribution-3 anchor (SCC/orbit structure exposed
without matching the full control-theorem signature). Bakery is not that
Contribution-2 anchor — its "simplification" (the lex-key lift to
derived coords) is a notational rephrasing, not a proof-shortening
insight. A candidate Contribution-2 target might be
MissionariesAndCannibals or DiningPhilosophers, where the parity/symmetry
structure is well-known and QA's modular framing could plausibly
shorten an invariant proof.
