# DieHard — QA-Native Reproduction Attempt

**Blindness pact:** Author has read ONLY `qa_tla_blind/prompts/diehard.md`. Has NOT
read `DieHard.tla`, `DieHard.cfg`, `DieHardest.tla`, or any associated PDF.

## 1. QA state encoding

Physical state is a 2-tuple `(small, big)` where `small` is the current integer
gallons in the 3-gallon jug and `big` the current integer gallons in the 5-gallon
jug. Physical bounds: `small ∈ {0,1,2,3}`, `big ∈ {0,1,2,3,4,5}`.

**No-Zero shift (A1).** QA forbids state 0. Apply the canonical `+1` shift:

```
s := small + 1   ∈ {1, 2, 3, 4}        (4 values)
B := big   + 1   ∈ {1, 2, 3, 4, 5, 6}   (6 values)
```

This is a pure relabelling; it preserves the reachability graph and the
transition relation under an isomorphism `φ(small, big) = (small+1, big+1)`. It
adds no semantic content — it satisfies the axiom but does not discriminate.

**(b, e) assignment.** The cleanest QA mapping treats each jug as an independent
QA state pair:

```
Small jug:   b_s = s,  e_s = 3  (capacity+1 cap as pinned energy)
Big   jug:   b_B = B,  e_B = 5
```

Then the derived coords `d = b + e`, `a = b + 2e` exist mechanically (A2) but
carry no physical meaning in this domain — see §7. This is the honest price
of enforcing A2 on a domain that is not natively a QA orbit.

**Combined state** is the 4-tuple `(s, e_s=3, B, e_B=5)` with dynamic variables
`(s, B)`.

## 2. Observer projection (Theorem NT)

The continuous quantity is physical water volume (a real number of gallons).
The observer projection is **bucketed integer gallons** — the read layer rounds
the continuous volume to the nearest integer. This crosses the firewall exactly
once at the input boundary (continuous → integer) per NT. All subsequent QA
dynamics operate on integers; no output of the QA layer is re-read as a
continuous quantity. **NT is satisfied but vacuous** — the puzzle is stated
in integers to begin with.

## 3. QA dynamics (discrete path-time)

Path-time `k ∈ ℕ` is the action step count (T1 satisfied).

**Six discrete actions**, each a deterministic transition on `(s, B)`:

| Action | `(s, B) →` | Pre-condition |
|---|---|---|
| FillSmall | `(4, B)` | `s < 4` |
| FillBig   | `(s, 6)` | `B < 6` |
| EmptySmall | `(1, B)` | `s > 1` |
| EmptyBig   | `(s, 1)` | `B > 1` |
| PourSmallToBig | `(1 + max(0, (s-1) + (B-1) - 5), 1 + min(5, (s-1) + (B-1)))` if pour fully fits else stop-at-fill | — |
| PourBigToSmall | `(1 + min(3, (s-1) + (B-1)), 1 + max(0, (s-1) + (B-1) - 3))` | — |

(Pours are clamped: stop when source is empty or destination full. The shifted
arithmetic is ugly; this is an artifact of the +1 shift with no structural gain.
Computed on unshifted `(small, big)` the pour rules are the usual
`(max(0, small+big−5), min(5, small+big))` etc.)

**Note on `qa_step`.** The canonical QA step `((b+e-1) % m) + 1` does **not**
describe this system. There is no single modulus `m` under which both jugs
evolve; the dynamics are a bounded-lattice with clamped transitions, not a
cyclic orbit. This is a structural mismatch between QA's native machinery and
the puzzle's native machinery — flagged in §7.

## 4. Orbit classification

The reachable-state set from `(s=1, B=1)` (shifted (0,0)) under the six actions
is a finite transition graph. By exhaustive enumeration (the state space is
≤ 24 physical states, 4 × 6 shifted states), every state with `0 ≤ small ≤ 3,
0 ≤ big ≤ 5` is trivially bounded; reachable subset is a connected subgraph
of this lattice. Cosmos / Satellite / Singularity orbit taxonomy does **not**
apply — the puzzle has no 24-cycle, 8-cycle, or 1-cycle orbit structure. The
reachable graph is a bounded lattice with branching, not a cycle.

## 5. Invariant restatement (QA-native forms)

### `TypeOK` (type/range)

Given in the prompt as "each jug's contents stays within capacity, non-negative".

**QA form (shifted, A1-compliant):**
```
TypeOK_QA :=  s ∈ {1, 2, 3, 4}  ∧  B ∈ {1, 2, 3, 4, 5, 6}
```

**Proof (inductive).** Base: `(1, 1)` satisfies it. Inductive: each of the
six actions maps the allowed rectangle to itself — FillSmall/EmptySmall fix B
and move s to the extremes of its allowed range; FillBig/EmptyBig symmetric;
pours preserve the total water content (in unshifted units) and clamp to
capacities, so the image stays inside the rectangle. Six sub-lemmas, one per
action. No orbit structure needed; this is a plain bounded-lattice invariant.

**Status:** reproduced. The QA form is equivalent to the natural TLA+ form
modulo the +1 shift.

### `NotSolved` (puzzle-unsolved invariant)

Given in the prompt as "a state that violates this is a witness to the solution".
The goal is to reach a state with exactly 4 gallons in one of the jugs.

**QA form (shifted):**
```
NotSolved_QA :=  ¬( s = 5  ∨  B = 5 )
```

(In shifted units, `small = 4 ↔ s = 5` and `big = 4 ↔ B = 5`.)

Hmm — but `s ∈ {1,2,3,4}`, so `s = 5` is impossible. So `NotSolved_QA` reduces to
`¬(B = 5)` since small jug capacity is 3, the small jug can never hold 4
gallons. **Sharpening:** the target is reachable only via the big jug.

```
NotSolved_QA' :=  B ≠ 5       (equivalently, big ≠ 4)
```

**Model-check expectation:** this invariant FAILS. The falsifying trajectory is
the 6-step solution — which I derive below by QA-native search, *not* by
reading ground truth.

**Status:** reproduced, with a minor sharpening (small jug can't hold 4).

## 6. Liveness / counterexample trajectory

By breadth-first search in the reachable graph, starting from shifted `(1,1)`
and applying the six actions, a witness trajectory of length `k = 6`:

| step | action | state (unshifted) |
|---|---|---|
| 0 | (init) | (0, 0) |
| 1 | FillBig | (0, 5) |
| 2 | PourBigToSmall | (3, 2) |
| 3 | EmptySmall | (0, 2) |
| 4 | PourBigToSmall | (2, 0) |
| 5 | FillBig | (2, 5) |
| 6 | PourBigToSmall | (3, 4) ← big = 4 |

So `NotSolved_QA'` fails at `k = 6`. QA path-time k = 6 (T1).

## 7. Honest commentary (failure modes to flag in diff)

This is the shakedown spec. I want to flag structural issues even where the
reproduction succeeds, because they matter for harder specs downstream.

1. **`qa_step` does not describe DieHard.** There is no single modulus `m`
   under which `((b+e-1) % m) + 1` gives the transition function. The puzzle's
   dynamics are a 6-action bounded lattice, not a cyclic orbit. QA's core
   discrete-dynamics primitive is absent here. *Candidate tag:
   `invariant-inexpressible` for orbit-structure invariants (there are none
   to reproduce, but QA brings no machinery to bear either).*

2. **Derived coords (d, a) are decorative.** Under the chosen (b, e) mapping,
   `d = b+e` and `a = b+2e` over `(s, 3)` or `(B, 5)` have no physical
   meaning — they are not conserved quantities, not symmetries, not anything
   the puzzle cares about. A2 is satisfied syntactically; it contributes no
   proof obligation and no discriminating power.

3. **No-Zero shift is a relabelling.** The +1 shift adds arithmetic ugliness
   (see the shifted pour formulas in §3) with zero semantic content. A1 is
   satisfied but offers no leverage. The underlying problem is that
   "empty jug" is a physically meaningful state that QA's A1 cannot honour
   without either a shift or a sentinel, and neither buys anything here.

4. **Orbit taxonomy does not apply.** Cosmos (24-cycle) / Satellite (8-cycle) /
   Singularity (1-cycle) structure is absent. The reachable graph is a
   finite-branching DAG-with-cycles on a bounded lattice. Calling any subset
   of it an "orbit" would be a renaming, not a classification.

5. **Theorem NT is satisfied but vacuous.** The firewall crossing
   (continuous volume → integer gallons) happens once at problem setup; no
   QA output feeds back as a continuous input. NT holds, but adds no
   discriminating power because the puzzle is natively integer-valued.

6. **Where QA *did* help (honest):** The discreteness axiom (T1, k ∈ ℕ as
   path length) matches the TLA+ `Next`-step semantics cleanly — both count
   discrete actions. This is a coincidence, not a QA contribution.

### Summary for the diff scorer

- `TypeOK`: **reproduced** (equivalent under +1 shift isomorphism).
- `NotSolved`: **reproduced + slightly strengthened** (QA version sharpens to
  `big ≠ 4` by noting small jug capacity < 4).
- **Overall failure tag:** `invariant-inexpressible` — not for the specific
  invariants given (both were reproducible), but for the QA orbit machinery
  which has no expressible form on this problem. QA here is *compatible*
  with TLA+ (both handle discrete-state finite systems), but QA brings no
  additional invariants that TLA+ missed.

This outcome is expected: DieHard is a lattice-search puzzle, not a
modular-arithmetic problem. The real test is whether QA gains traction on
Bakery (mutex via tickets — a mod-pair problem), Paxos (quorum invariants —
potentially orbit-structured), or specs with natural modular structure.

## 8. Self-check

- [x] A1: No-Zero enforced via `+1` shift (flagged as cosmetic).
- [x] A2: Derived coords stated (flagged as decorative).
- [x] T1: Path-time `k` discrete, action-step count.
- [x] T2: Firewall crossing at continuous→integer input only; no QA→continuous→QA loop.
- [x] S1: No `**2` in any of the transition functions above.
- [x] S2: All state variables are integers (not floats, not arrays of floats).
- [x] Blindness pact held: did not read `DieHard.tla`, `.cfg`, or PDF.
