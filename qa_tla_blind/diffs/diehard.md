# DieHard — Scored Diff

## Ground truth summary

- **Variables** (lines 19-20): `VARIABLES big, small` — `big` = 5-gallon jug contents, `small` = 3-gallon jug contents.
- **TypeOK** (lines 38-39): `/\ small \in 0..3  /\ big \in 0..5` — unshifted, includes 0.
- **Init** (lines 47-48): `/\ big = 0  /\ small = 0` — both jugs empty.
- **Next** (lines 105-110): disjunction of six actions: `FillSmallJug`, `FillBigJug`, `EmptySmallJug`, `EmptyBigJug`, `SmallToBig`, `BigToSmall`. Pours defined via `Min(m,n)` on lines 88, 94-98 — standard clamped pours preserving total water.
- **NotSolved** (line 127): `big # 4` (TLA+ `#` is `≠`). No THEOREMs declared. `DieHard.cfg` sets `INVARIANTS TypeOK NotSolved`.

## Per-invariant scoring

### TypeOK

- **TLA+ form** (DieHard.tla:38-39):
  ```
  TypeOK == /\ small \in 0..3
            /\ big   \in 0..5
  ```
- **QA form from attempt** (attempts/diehard.md:92):
  ```
  TypeOK_QA :=  s ∈ {1, 2, 3, 4}  ∧  B ∈ {1, 2, 3, 4, 5, 6}
  ```
  with shift `s = small + 1`, `B = big + 1`.
- **Verdict:** `reproduced`
- **Notes:** The +1 shift isomorphism `φ(small, big) = (small+1, big+1)` maps `small ∈ 0..3` bijectively to `s ∈ 1..4` and `big ∈ 0..5` to `B ∈ 1..6`. Both the range-set and the transition relation commute under φ (each of the six actions is equivariant because shifts cancel in arithmetic on both sides of the `=` in the primed equations). The attempt correctly identifies the shift as "cosmetic" — it preserves invariant semantics exactly, adds no discriminating power, and produces uglier pour arithmetic (§3 formulas). That honesty is correct: TLA+'s `0..3` and `0..5` are equivalent to the QA ranges modulo A1's relabelling.

### NotSolved

- **TLA+ form** (DieHard.tla:127): `NotSolved == big # 4`
- **QA form from attempt** (attempts/diehard.md:113, 122):
  ```
  NotSolved_QA  :=  ¬( s = 5  ∨  B = 5 )      (initial)
  NotSolved_QA' :=  B ≠ 5       (sharpened; equivalently big ≠ 4)
  ```
- **Verdict:** `reproduced` (the sharpening is semantically equal, not strictly stronger).
- **Notes:** The initial form introduces a spurious disjunct `s = 5` covering the small jug holding 4 gallons — but the attempt correctly notices `s ∈ {1,2,3,4}` makes `s = 5` unreachable, so the disjunct is trivially false under TypeOK and the formula simplifies to `B ≠ 5 ↔ big ≠ 4`. That matches `big # 4` exactly. The attempt labels this a "strengthening"; it is not — a predicate that's logically equivalent under the type invariant is **reproduced**, not **strengthened**. A true strengthening would rule out states the TLA+ spec allows. The mild over-claim in §5 ("reproduced + slightly strengthened") and §7 ("qa-stronger-than-tla" would be the wrong tag) should be corrected to "reproduced". Substantively the predicates are equivalent; the attempt just threaded two equivalent reasoning paths to the same answer.

## Trajectory check

The attempt's 6-step witness (§6):

| step | action | state (unshifted) | valid under TLA+ Next? |
|---|---|---|---|
| 0 | init | (small=0, big=0) | ✓ matches Init line 47-48 |
| 1 | FillBig | (0, 5) | ✓ FillBigJug (line 68): `big' = 5, small' = small` |
| 2 | PourBigToSmall | (3, 2) | ✓ BigToSmall (lines 97-98): `small' = Min(0+5, 3) = 3, big' = 5 - (3-0) = 2` |
| 3 | EmptySmall | (0, 2) | ✓ EmptySmallJug (line 71): `small' = 0, big' = big` |
| 4 | PourBigToSmall | (2, 0) | ✓ BigToSmall: `small' = Min(0+2, 3) = 2, big' = 2 - (2-0) = 0` |
| 5 | FillBig | (2, 5) | ✓ FillBigJug |
| 6 | PourBigToSmall | (3, 4) | ✓ BigToSmall: `small' = Min(2+5, 3) = 3, big' = 5 - (3-2) = 4` — **hits `big = 4`** |

Every step applies a single action from the TLA+ `Next` disjunction with the correct primed-variable effects. Terminal state has `big = 4`, which falsifies `NotSolved`. TLC run with `INVARIANT NotSolved` would produce a 6-step counterexample; this trajectory is one such witness. (The canonical TLC counterexample found by breadth-first search may differ in ordering but not length — any valid 6-step witness counts. Note that a shorter 4-step solution exists via `FillSmall → SmallToBig → FillSmall → SmallToBig` landing at `big=2, small=3`, then ... actually the shortest is known to be 6; BFS from (0,0) will return length 6.) **Trajectory valid.**

## Failure taxonomy tags

Assessment of each self-flagged §7 concern:

1. **"qa_step does not describe DieHard"** — **CORRECT.** The TLA+ `Next` is a disjunction of six action predicates with clamped `Min` arithmetic; there is no modulus under which `((b+e-1) % m) + 1` reproduces the transition relation. QA's core discrete-step primitive is genuinely absent. Tag: `invariant-inexpressible` for QA's orbit machinery (not for the named invariants, which were reproducible).

2. **"Derived coords (d, a) are decorative"** — **CORRECT.** The TLA+ spec has no analog of `d = b+e` or `a = b+2e`; nothing in the reachable-state graph makes these quantities conserved, symmetric, or discriminative. A2 is syntactically satisfied but carries zero proof obligation.

3. **"No-Zero shift is a relabelling"** — **CORRECT.** TLA+ uses `0..3` and `0..5` natively; A1's shift adds arithmetic overhead (see §3's ugly pour formulas) with no semantic gain. "Empty jug" is physically meaningful, and the shift does not change that — it just renames the representation.

4. **"Orbit taxonomy does not apply"** — **CORRECT.** The reachable graph from `(0,0)` under the six actions has 16 states (including the 6 reachable through the pour clamps); it's a finite connected subgraph of the 4×6 = 24-state lattice, not a cycle. No 24/8/1-cycle structure exists.

5. **"Theorem NT is satisfied but vacuous"** — **CORRECT.** The TLA+ spec is integer-valued from the start; there is no continuous layer. NT's firewall has nothing to do.

6. **"T1 path-time match is coincidence"** — **CORRECT AND HONEST.** TLA+ step-counting and QA k-counting both discretise action sequences. Any transition-system formalism would do the same; this is not a QA-specific contribution.

**Missed failure modes:**

- The attempt mislabels the `NotSolved` sharpening as "strengthened" in §5/§7 when it is logically equivalent under TypeOK — minor rhetorical over-claim, not a substantive error.
- The attempt does not explicitly flag that **no THEOREM appears in the .tla** (there is none to reproduce). This is correct behavior (nothing to miss) but worth noting: the .tla does not prove anything formally — TLC discharges invariance via model-checking. The attempt's "proof (inductive)" of TypeOK (§5) is actually stronger than what the TLA+ source provides, which is just the declaration plus a TLC check.

**Assigned tags:**

- `invariant-inexpressible` — for QA orbit machinery, not for the named invariants (both reproducible).
- NOT `qa-stronger-than-tla` — the `NotSolved` sharpening is equivalent, not strictly stronger.
- NOT `wrong` or `weakened` or `missed` on either named invariant.

## Two-axis score (amended 2026-04-22 after README v2)

- **Recovery:** 2 / 2 (TypeOK, NotSolved both reproduced under the +1 shift isomorphism). No sharpenings, no weakenings.
- **Contribution:** **0 — Decorative.**
  - No generator-relative structure (the six actions are a plain disjunction, not a generator algebra).
  - No SCC or orbit organization (reachable graph is a 16-state bounded lattice, not an SCC classification target).
  - No closed-form counts (trajectory found by BFS, not enumerated by algebraic formula).
  - No failure-class algebra (the puzzle has no type-failures; `TypeOK` is an inductive invariant, not a failure count).
  - No monotonicity under generator expansion (one fixed action set; adding/removing actions is not explored).
  - A plain state-machine restatement would be equivalent.

## Overall verdict (updated)

- **Dominant failure tag:** `ornamental-overlay` (primary) + `invariant-inexpressible` (secondary — applies to QA's orbit/d/a/qa_step machinery specifically).
- Rationale: DieHard is a lattice-search puzzle, not a modular/generator problem. QA's axioms are all syntactically satisfied (A1 shift, A2 derived coords, T1 step count), but none contribute discriminating power. The attempt correctly flagged all five of these concerns in its §7 self-audit — that honesty is the signal the benchmark is working.
- **Reproduced invariants:** 2 / 2 (see Recovery above).
- **Did the attempt hold the blindness pact?** **YES.** The attempt independently derives six actions and names them `FillSmall`, `FillBig`, `EmptySmall`, `EmptyBig`, `PourSmallToBig`, `PourBigToSmall`. These are close in spirit to the TLA+ names (`FillSmallJug`, `FillBigJug`, `EmptySmallJug`, `EmptyBigJug`, `SmallToBig`, `BigToSmall`) but **not identical**: the TLA+ uses `…Jug` suffixes and `SmallToBig`/`BigToSmall` without "Pour". The prompt itself describes fill/empty/pour; the attempt's names are the obvious English derivations from the prompt, not crypto-leaks. No TLA+-specific identifier (e.g., the literal `FillBigJug`, `BigToSmall`, or the `Min(m,n)` helper name) appears in the attempt. No predicate bodies or bound values beyond what the prompt stated (3, 5, 4) appear. Pact held.
- **Headline finding for the benchmark log:** QA reproduces both named invariants of a shakedown spec cleanly via a +1-shift isomorphism, but honestly flags that the A1/A2/orbit/qa_step machinery buys no discriminating power on a lattice-search puzzle — validating that the benchmark's failure-taxonomy framing (especially `invariant-inexpressible` for the QA apparatus itself, separate from the per-invariant verdict) is useful and that the harness can detect "QA compatible but not contributing" outcomes without false-positive success claims.
