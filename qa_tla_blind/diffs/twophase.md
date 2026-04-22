# TwoPhase — Mode B Scored Diff

## Ground truth summary

- **VARIABLES** (`TwoPhase.tla:21-26`): `rmState, tmState, tmPrepared, msgs`.
- **Message set** (`TwoPhase.tla:43-51`): `[type:{"Prepared"}, rm:RM] ∪ [type:{"Commit","Abort"}]`.
- **TPTypeOK** (`TwoPhase.tla:53-60`):
  ```
  /\ rmState    \in [RM -> {"working", "prepared", "committed", "aborted"}]
  /\ tmState    \in {"init", "committed", "aborted"}
  /\ tmPrepared \subseteq RM
  /\ msgs       \subseteq Message
  ```
- **TPInit** (`TwoPhase.tla:62-69`): `rmState = [rm |-> "working"]`, `tmState = "init"`, `tmPrepared = {}`, `msgs = {}`.
- **Actions** (one-line effects):
  - `TMRcvPrepared(rm)` (L75-82): guard `tmState="init" ∧ Prepared(rm)∈msgs`; `tmPrepared' = tmPrepared ∪ {rm}`.
  - `TMCommit` (L84-93): guard `tmState="init" ∧ tmPrepared=RM`; `tmState'="committed" ∧ msgs' += Commit`.
  - `TMAbort` (L95-102): guard `tmState="init"`; `tmState'="aborted" ∧ msgs' += Abort`.
  - `RMPrepare(rm)` (L104-111): guard `rmState[rm]="working"`; `rmState'[rm]:="prepared" ∧ msgs' += Prepared(rm)`.
  - `RMChooseToAbort(rm)` (L113-120): guard `rmState[rm]="working"`; `rmState'[rm]:="aborted"` (no msg).
  - `RMRcvCommitMsg(rm)` (L122-128): guard `Commit∈msgs`; `rmState'[rm]:="committed"`.
  - `RMRcvAbortMsg(rm)` (L130-136): guard `Abort∈msgs`; `rmState'[rm]:="aborted"`.
- **TPSpec** (`TwoPhase.tla:144`): `TPInit ∧ [][TPNext]_vars`.
- **Refinement theorem** (`TwoPhase.tla:163-170`): `TC == INSTANCE TCommit`; `THEOREM TPSpec => TC!TCSpec`.
- **TCConsistent** (`TCommit.tla:54-60`):
  ```
  \A rm1, rm2 \in RM : ~ /\ rmState[rm1] = "aborted"
                         /\ rmState[rm2] = "committed"
  ```
- **MC harness** (`MCTwoPhase.tla` absent — MC file not present in repo; only `TCommit.cfg` found, which pins `CONSTANT RM = {r1, r2, r3}` and `INVARIANTS TCTypeOK TCConsistent` at `TCommit.cfg:1-2`). Header comment in `TwoPhase.tla:172-175` notes the spec was TLC-checked at 6 RMs = 50816 reachable states.

## Per-invariant scoring (Recovery)

### TPTypeOK

- **TLA+ form** (`TwoPhase.tla:53-60`):
  ```
  /\ rmState    \in [RM -> {"working", "prepared", "committed", "aborted"}]
  /\ tmState    \in {"init", "committed", "aborted"}
  /\ tmPrepared \subseteq RM
  /\ msgs       \subseteq Message
  ```
- **QA form from attempt** (`attempts/twophase.md:94-101`):
  ```
  TPTypeOK ≡ (∀i∈RM)(b_i,e_i) ∈ {(1,1),(2,1),(3,1),(1,3)}
         ∧ (TM_b,TM_e) ∈ {(1,1),(3,1),(1,3)}
         ∧ (∀i∈RM) χ_i ∈ {1,2}
         ∧ (∀i∈RM) m_{P,i} ≥ 1 ∧ m_C ≥ 1 ∧ m_A ≥ 1
  ```
- **Verdict:** `reproduced` with cosmetic A1 shift + one genuine weakening.
- **Notes:**
  - The 4-point RM lattice `{(1,1),(2,1),(3,1),(1,3)}` is in bijection with `{working, prepared, committed, aborted}` via the table at `attempts/twophase.md:30-34`. TM mapping (3 points) matches the TM's 3-value set.
  - `tmPrepared \subseteq RM` is encoded as a characteristic vector in `{1,2}^N` (A1 shift); semantically equivalent.
  - `msgs \subseteq Message` is **weakened** by the attempt: the attempt replaces the set of messages with monotone integer counters `(m_P_i, m_C, m_A)`. Under TLA+, `msgs` is an idempotent *set* — the same Prepared(rm) message added twice is still one element. Under the attempt's counter encoding, incrementing `m_C` twice produces `m_C = 3` vs `m_C = 2`, which are distinguishable states where TLA+ conflates them. Because `TMCommit` is guarded by `tmState = "init"` and fires `tmState' = "committed"`, it can fire at most once per trace, so `m_C` is in fact ≤ 2 on reachable traces — the divergence is benign for safety. But `RMPrepare(rm)` can in principle fire and be re-enabled (no, it also transitions RM state so self-disables), so `m_{P,i}` is also capped at 2. In practice the encoding is equivalent on reachable states; as a type predicate it is strictly weaker than `\subseteq Message` because it admits unbounded counters.
  - Recovery verdict: **reproduced** (the reachable-state equivalence holds under the safety proof) with a benign type-side weakening the attempt does not flag.

### TCConsistent

- **TLA+ form** (`TCommit.tla:54-60`):
  ```
  \A rm1, rm2 \in RM : ~ /\ rmState[rm1] = "aborted"
                         /\ rmState[rm2] = "committed"
  ```
- **QA form from attempt** (`attempts/twophase.md:119-126`):
  ```
  TCConsistent ≡
    ¬ (∃ i, j ∈ RM, i ≠ j :
         (b_i,e_i) ∈ Terminal ∧ (b_j,e_j) ∈ Terminal
         ∧ (b_i,e_i) ≠ (b_j,e_j)
         ∧ Q_r((b_i,e_i),(b_j,e_j)) = 0
         ∧ Q_b((b_i,e_i),(b_j,e_j)) > 0)
  ```
  where `Terminal = {(3,1),(1,3)}`.
- **Verification check (independently computed):**

  For the four candidate RM states, computed `Q_r` and `Q_b` for all C(4,2) = 6 ordered pairs:

  | pair | `Q_r` | `Q_b` | `Q_g` | `Q_b² = Q_r²+Q_g²`? |
  |---|---|---|---|---|
  | working (1,1) – prepared (2,1) | 1 | 1 | 0 | ✓ |
  | working (1,1) – committed (3,1) | 4 | 4 | 0 | ✓ |
  | working (1,1) – aborted (1,3) | **−4** | 4 | 0 | ✓ |
  | prepared (2,1) – committed (3,1) | 1 | 1 | 0 | ✓ |
  | prepared (2,1) – aborted (1,3) | −3 | 5 | −4 | ✓ |
  | **committed (3,1) – aborted (1,3)** | **0** | **8** | **−8** | ✓ |

  Among non-self pairs, **only** committed-aborted satisfies `Q_r = 0 ∧ Q_b > 0`. Self-pairs have `Q_r = Q_b = 0`, which is correctly excluded by the `Q_b > 0` clause. The attempt's claim `Q_r((3,1),(1,3)) = (3-1)² - (1-3)² = 4 - 4 = 0` is arithmetic-correct; the attempt's claim that `Q_b((3,1),(1,3)) = 8` (used to exclude self-pairs) is also correct.

  Note: `Q_r(working, aborted) = -4 ≠ 0`, so working+aborted (a legal TLA+ state where one RM is still working while another has aborted) is **NOT** flagged as a disagreement. Good — TLA+ doesn't flag it either (aborted-vs-working is not committed-vs-aborted). The predicate cleanly picks out the forbidden pair and only the forbidden pair.

  **Completeness check:** TLA+'s TCConsistent forbids the pair (rm1=aborted, rm2=committed) under `∀ rm1, rm2`. The attempt's predicate is symmetric (`i ≠ j` gives both orderings) and restricts to Terminal; this is logically equivalent to forbidding any pair `{committed, aborted}` anywhere in `rmState`. Match ✓.

  **No qa-stronger/weaker mismatch.** The encoding is tight: it forbids exactly the states TLA+ forbids.

- **Verdict:** `reproduced`.
- **Notes:** The chromogeometric framing is load-bearing on the invariant side — the null-cone fact `Q_r((3,1),(1,3)) = 0 ∧ Q_b > 0` is specifically what makes the predicate uniquely identify the committed-aborted disagreement pair among the four terminal-lattice points. The attempt's `Terminal = {(3,1),(1,3)}` membership clause is what restricts scope; the null-cone clause picks out disagreement within Terminal². Without the membership clause, `Q_r(working, aborted) = -4` is negative (safely non-zero) but `Q_r(working, prepared) = 1` and `Q_r(working, committed) = 4` are also non-zero — so in fact *no non-Terminal × Terminal pair has `Q_r = 0`*, meaning the `Terminal ∋ p_i, p_j` membership restriction is structurally redundant on this specific lattice choice but is prudently included.

## Contribution score (0-4)

Per-primitive load-bearing assessment:

1. **Points in Z² + 4-tuple (b,e,d,a) (object model §1, §9).**
   **USED BUT WEAK.** The attempt places RM states at integer lattice points and lists `(d, a)` derived values in its table. The 4-tuple is not load-bearing on the invariant: `d` and `a` do not enter TCConsistent or TPTypeOK. The attempt claims `d = b+e` separates "pre-decision vs post-decision" (working: d=2, prepared: d=3, committed: d=4, aborted: d=4) — but since committed and aborted *share* d=4, the `d`-projection does NOT separate them; only `(b,e)` does. The 4-tuple embedding is cosmetic; the real structure is in Z² alone. Rating: **USED BUT WEAK**.

2. **Red quadrance Q_r (object model §3).**
   **LOAD-BEARING.** The fact `Q_r((3,1),(1,3)) = 0` is the key algebraic step that makes the null-cone predicate equivalent to "no committed-vs-aborted pair". Verified by exhaustive enumeration (above): of 6 distinct-pair configurations on the 4-point lattice, exactly one satisfies `Q_r = 0 ∧ Q_b > 0`, and it is the disagreement pair. If the lattice points were chosen differently (e.g., committed at (2,2), aborted at (1,1)), `Q_r = 1 ≠ 0` and the null-cone form would not pick out the disagreement. The specific geometry — committed at (3,1), aborted at (1,3) — is what the chromogeometric view is doing. Rating: **LOAD-BEARING**.

3. **Blue quadrance Q_b (object model §3).**
   **USED BUT WEAK.** The `Q_b > 0` clause excludes self-pairs `(committed, committed)` and `(aborted, aborted)` where `Q_r` is trivially 0 by coincidence of lattice point. This is a real discriminating condition — without it, the predicate would (vacuously) forbid any RM from being in a terminal state at all. However, the same exclusion is achievable with the simpler clause `(b_i, e_i) ≠ (b_j, e_j)` (which the attempt *also* includes, at `attempts/twophase.md:123`). So the `Q_b > 0` clause is slightly redundant with the distinctness clause. Rating: **USED BUT WEAK**.

4. **Translations (object model transform §1).**
   **USED BUT WEAK / partially DECORATIVE.** The attempt attaches "translation" labels to the seven actions. The load-bearing property of translations is *quadrance preservation*, which is invoked to argue that legal traces preserve the null-cone structure. But the *terminal actions* `RMRcvCommitMsg` and `RMRcvAbortMsg` — the ones that actually create the disagreement risk — are NOT pure translations: they are "set to fixed point" (the attempt honestly flags this as "a translation whose offset depends on the current point" at `attempts/twophase.md:80`, which is **not a translation** in the object model's sense). So the closure-under-translation argument only applies to the monotone-pool actions; the load-bearing safety argument for TCConsistent actually hinges on a different fact (the TM's init-state linearity, see L128-134) that is not a translation/quadrance-preservation argument at all. Rating: **USED BUT WEAK**.

5. **Chromogeometric Pythagorean identity Q_b² = Q_r² + Q_g² (object model §"Three-metric").**
   **DECORATIVE.** Named in `attempts/twophase.md:12` and referenced at §5 but never *used* in a proof step. The identity holds automatically for any integer-pair chromogeometric triple (verified above for all 6 pairs) and contributes no discrimination to TCConsistent — `Q_r = 0` alone suffices; `Q_b, Q_g` values at the disagreement point are incidental. The attempt's note that the identity "gives the conserved tri-metric budget" is prose; no algebraic consequence is drawn. Rating: **DECORATIVE**.

6. **(Implicit 6th, claimed not used): cross-ratio, spread polynomials, TQF, mutations, reflections, projective maps, hexagonal ring.**
   Correctly judged inapplicable in `attempts/twophase.md:14-21`. Honest non-use, not ornamental.

**Summary:** of 5 primitives actually invoked, **1 load-bearing (Q_r), 3 used-but-weak (points+4-tuple, Q_b, translations), 1 decorative (Pythagorean identity)**.

**Key load-bearing test.** Per the scorer prompt: does `Q_r((3,1),(1,3)) = 0` uniquely pick out committed-vs-aborted among all 6 distinct pairs of the 4 RM states? **YES** (verified above). But does "any other encoding would also work" — e.g., a plain predicate `(rmState[i], rmState[j]) = ("committed","aborted")` — produce the same safety condition? **Also yes**, trivially. So the null-cone encoding is *correct* but not *uniquely illuminating*: it does not produce a closed-form disagreement residue, a generalizing identity, or a structural decomposition that the plain TLA+ form lacks. The chromogeometric view adds geometric vocabulary without producing new proof power on this spec.

**Additional check: does the encoding generalize for N RMs to a sum-of-pairwise-Q_r predicate?** No. The attempt's TCConsistent is `∀ pair : ¬(null-cone)`, not `(Σ Q_r over pairs) ∈ S`. A sum-based predicate would be Contribution-3 material; the pairwise ∀ form does not lift the geometry to the N-body structure.

**Final Contribution: 2 (Useful).**

**Rationale (one sentence).** The `Q_r((3,1),(1,3)) = 0` null-cone fact is genuinely load-bearing — it's the specific algebraic fact the encoding rests on, and the exhaustive-pair verification confirms it uniquely identifies the forbidden disagreement — but four of the five other invoked primitives are weak or decorative, the generator algebra is a free commutative monoid (not a spread/mutation structure), there is no closed-form count or SCC theorem beyond O(N²) pair enumeration, and a plain state-machine form would also be equivalent; this sits above Bakery (Contribution 1, nominal σ/μ/λ₂/ν labels) but below a Contribution-3 attempt that would derive a sum-residue identity or an SCC theorem over the `N` RMs.

**Calibration:** DieHard = 0 (decorative), Bakery = 1 (compatible, nominal generator labels), **TwoPhase (Mode B) = 2 (useful, one load-bearing primitive)**, QA control theorems = 4 (decisive, all 5 Contribution-≥3 markers + verified extension). Gap from 2 → 4 is populated by (a) closed-form count over pairs, (b) SCC/orbit theorem, (c) novel derivation beyond the spec.

## Primitive-gap assessment

The attempt flagged three gaps at `attempts/twophase.md:152-162` (plus Gap 4 which is a non-gap):

- **Gap 1: monotone-counter primitive for `msgs` pool.**
  **REAL.** The Wildberger bundle as presented surfaces `Caps(N, N)` (bounded rectangle) and `P²(Z)` (projective classes) but does not give an unbounded monotone-counter primitive. The attempt's workaround — `Z_{≥1}` with A1 shift — is sound but exposes a genuine bundle gap. Distributed protocols with receive-without-consume semantics are a natural use-case and the primitive is missing. Real gap.

- **Gap 2: guarded-translation monoid class.**
  **REAL but partially coverable.** Pure constant-offset translations (object model §1) do not cover "set to fixed point" (the receipt actions). The attempt proposes a "conditional translation" = current-state-dependent-offset, which is not in the bundle's transform list. The bundle's **mutation moves** (§5, transforms) are guard-dependent in the Coxeter-Dynkin sense (mutation depends on which root is chosen), so mutation-as-primitive *might* cover guarded state-sets. But mutations introduce Weyl-group structure the 2PC action set does not have — forcing the fit would be ornamental. The honest reading: guarded translations are genuinely outside the bundle; mutations don't cover them cleanly. Real gap.

- **Gap 3: message pool as integer counter.**
  **Duplicate of Gap 1.** The attempt lists this separately but it's the same primitive gap. Counting once.

- **Gap 4: "no bivector needed"** (attempt L160-162). Correct non-gap; the bundle is sufficient on the invariant side.

**Summary:** two genuine bundle gaps (Gap 1 = monotone counter; Gap 2 = guarded-translation/idempotent-projection). These are the structural reason the attempt tops out at Contribution 2: Wildberger's rich generator structures (spread polynomials, mutations, reflections) are for *geometric-motion* dynamics; 2PC is a *monotone-accumulation* protocol, so those structures idle.

**Could mutations-as-transforms cover guarded translations?** The attempt says no — mutations preserve root-system structure (Cartan integers, Dynkin graph), and 2PC's action set has neither. Forcing the fit would score `ornamental-overlay` on the generator side. The attempt's honest rejection at L86 is correct.

## Failure taxonomy tags

- **`ornamental-overlay` (partial, 4/5 primitives).** The Pythagorean identity is name-dropped without load-bearing use (fully decorative); points+4-tuple, Q_b, and translations are used-but-weak (mild ornamentality). Per the Mode B rubric: `ornamental-overlay` fires partially when some primitives are decorative while others are load-bearing. The attempt earns a partial-ornament tag on 4 primitives but not on Q_r.
- **`proof-gap` (minor).** The "monotone-counter weakening of `msgs`" in TPTypeOK is not flagged by the attempt as a type-side weakening; the attempt claims reproduction without noting the benign gap on the type predicate. Minor.
- **NOT `no-mapping-exists`** — a defensible mapping exists and was produced.
- **NOT `wrong-observer-projection`** — protocol is fully discrete; no continuous input to misproject.
- **NOT `orbit-mismatch`** — the attempt honestly declines to claim Cosmos/Satellite/Singularity structure (monotone DAG, no SCCs).
- **NOT `invariant-inexpressible`** — both invariants expressed in QA-native form.
- **NOT `qa-stronger-than-tla` / `qa-weaker-than-tla` on TCConsistent** — tight mapping, verified by enumeration.
- **`qa-weaker-than-tla` on TPTypeOK (minor)** — the counter-encoding is weaker than `msgs \subseteq Message` as a type bound (unbounded counters admissible where TLA+ idempotent-set semantics caps at one copy). Benign on reachable states but technically a type weakening.

## Blindness check

Searched attempt for verbatim TLA+ identifiers:

- `rmState`, `tmState`, `tmPrepared`, `msgs` — **appear in the attempt** (e.g., `attempts/twophase.md:54` mentions `tmPrepared`, L55 mentions `msgs`, L13 mentions lattice vector notation using `(b_i, e_i)` for `rmState`). These are **also used verbatim in the prompt** (`prompts/twophase.md:42-53`) as the names the reproducer was told to use. **Not a blindness break** — the prompt leaked the identifier names, so reusing them is not independent recovery. The scorer prompt listed these as blindness-leak candidates, but they are in the prompt itself. Pact held on this axis.
- Action names: `TPInit`, `TPNext`, `TMRcvPrepared`, `TMCommit`, `TMAbort`, `RMPrepare`, `RMChooseToAbort`, `RMRcvCommitMsg`, `RMRcvAbortMsg` — the prompt (`prompts/twophase.md:69-85`) gives the attempt these exact English names as "the standard TPC action set the reproducer should expect to recover". The attempt uses the same names (e.g., `attempts/twophase.md:72-78`). **Prompt-supplied**, not TLA+-leaked. Pact held.
- `TPSpec`, `TC!TCSpec`, `INSTANCE TCommit` refinement framing — **does not appear in the attempt**. The refinement is mentioned only in the prompt's framing, not in TLA+ syntax.
- **TCConsistent predicate body** (`\A rm1, rm2 \in RM : ~(rmState[rm1] = "aborted" /\ rmState[rm2] = "committed")`) — **does not appear verbatim in the attempt**. The attempt's TCConsistent is derived as `¬∃i,j : Terminal membership ∧ Q_r=0 ∧ Q_b>0`, which is a geometric reformulation, not a copy. This is the load-bearing blindness check: if the attempt had seen `TCommit.tla:54-60`, the clean move would be to rephrase it as `∀i,j : ¬(…=aborted ∧ …=committed)`; instead the attempt produces a chromogeometric reformulation by geometric reasoning from the English safety statement. **Pact held** on the predicate body.
- Terminal pc-label alphabet `{working, prepared, committed, aborted}` + TM alphabet `{init, committed, aborted}` — supplied by the prompt (L44-46, L61). Not a leak.
- MC constant value (`RM = {r1, r2, r3}` or `6` for the header-cited count) — does **not** appear in the attempt; the attempt keeps N parametric, consistent with the prompt's guidance.

**Blindness pact: HELD.** The one-nearest-leak hook (TCConsistent body) is cleanly not copied; the identifiers that DO appear are prompt-supplied.

## Overall verdict

- **Recovery: 2/2 invariants reproduced** (TPTypeOK with a benign type-side counter-weakening; TCConsistent tight).
- **Contribution: 2 (Useful).** One primitive (Q_r) is load-bearing; the null-cone fact `Q_r((3,1),(1,3)) = 0 ∧ Q_b > 0` uniquely discriminates the disagreement pair among 6 candidate pairs (verified by enumeration). Four other primitives are used-but-weak or decorative. No closed-form count, no SCC structure, no novel derivation beyond the spec.
- **Dominant tags:** `ornamental-overlay` (partial, 4/5 primitives decorative or weak), `qa-weaker-than-tla` (minor, on TPTypeOK type bound), `proof-gap` (minor, unflagged type weakening).
- **Blindness held: yes.** Identifiers that appear verbatim are prompt-supplied; TCConsistent body is reformulated geometrically, not copied.
- **Mode B-specific finding:** Mode B **beat** the nearest Mode A baseline (Bakery, Contribution 1) by one point. The improvement is attributable to the pre-specified object model: the reproducer went straight to chromogeometry without having to discover the framing. The `Q_r = 0` null-cone fact is a non-decorative QA encoding insight that Mode A would plausibly not have found from the English prompt alone. However, Mode B did not lift the attempt to Contribution 3+ — the spec's monotone-accumulation dynamics structurally don't fit Wildberger's geometric-motion primitives (rotations, mutations, reflections all idle here), which is the Mode-B-specific primitive-gap finding. The object model's metric/null-cone primitives do real work; its generator primitives do not.
- **Headline:** TwoPhase under Mode B reaches Contribution 2 via a load-bearing red-quadrance null-cone encoding of TCConsistent (`Q_r((3,1),(1,3)) = 0`, uniquely picking out committed-vs-aborted among 6 candidate pairs) — beating Bakery's Mode-A Contribution 1 by using the pre-specified Wildberger primitives non-decoratively on the invariant side, while honestly flagging that the bundle's geometric-motion generators idle on 2PC's monotone-accumulation dynamics and exposing two real primitive gaps (monotone-counter + guarded-translation) that cap Mode B at 2 without bundle augmentation.
