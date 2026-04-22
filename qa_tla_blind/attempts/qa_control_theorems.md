# QA Control Theorems — Blind Reproduction Attempt (POSITIVE CONTROL)

**Blindness pact:** Author has read ONLY `prompts/qa_control_theorems.md` +
`qa_tla_blind/README.md` + `CLAUDE.md`. Has NOT read any file in the
forbidden-reads list (`QA_CONTROL_THEOREMS.md`, `paper1_qa_control.*`,
`QARM_v02_*.tla/.cfg`, `ALL_INVARIANTS_VALIDATED.md`, anything under
`Formalizing tuple drift in quantum-native learning/`).

## 0. QA state encoding and firewall

**State**: `(b, e) ∈ Caps(N, N) = {1, …, N} × {1, …, N}`. Indexing is
**1-based** (A1 No-Zero). Derived coords `d = b + e`, `a = b + 2e` are fixed
by convention (A2) and *not* independent variables; they are raw integers
(not mod-reduced — per MEMORY hard rule: elements use raw `d=b+e`, mod
reduction is a T-operator only). They do not drive the generator algebra
here, but they must be respected if invoked.

**Caps(N,N) boundary** is a **hard constraint**, not a penalty: an
application is legal only if the image stays inside Caps(N,N). There is no
cost / soft-penalty formulation.

**Observer projection (Theorem NT):** This problem lives **entirely inside
the QA discrete layer**. No continuous quantity crosses the firewall. There
is no float → int cast. The generators σ, μ, λ₂, ν are integer partial
functions; edge counts, failure counts, and SCC structure are combinatorial
statements about a finite directed graph on Z² ∩ [1,N]². The NT firewall is
not crossed; it is vacuously respected.

**Path-time (T1):** Edges count as integer path-time steps (k = 1 per
generator application). Legal-edge counts and failure counts below are
static (per-state count of out-edges by label), but any reachability /
SCC argument uses integer path-time.

## 1. Restated problem

Four partial functions on Caps(N,N):

- σ(b,e) = (b, e+1), legal iff e ≤ N-1.
- μ(b,e) = (e, b), always legal (image in Caps).
- λ₂(b,e) = (2b, 2e), legal iff b ≤ ⌊N/2⌋ and e ≤ ⌊N/2⌋.
- ν(b,e) = (b/2, e/2), legal iff b and e both even (image then automatically
  in Caps since b,e ≥ 2 forces b/2, e/2 ≥ 1 and b/2, e/2 ≤ N/2 ≤ N).

Questions:

1. SCC structure (#SCC, max|SCC|, classification) under full
   Σ = {σ, μ, λ₂, ν} (and sub-case Σ' = {σ, μ, λ₂}).
2. Closed-form edge counts |g-edges| for each g ∈ Σ.
3. Closed-form failure counts per (g, failure-mode) pair.
4. Monotonicity of #SCC and max|SCC| under Σ₁ ⊆ Σ₂.
5. Which generators can participate in a directed cycle in Caps(N,N), why.

## 2. Structural analysis per generator

### σ (growth on e)
- **Not a bijection** on Caps(N,N): injective on its legal domain
  (b,e) with e ≤ N-1, but not surjective (no state maps to e=1).
- **Strictly increases e by 1.** Hence σ alone admits no cycle (e is a
  monovariant).
- **Cycle participation in Σ:** YES, via ν compensation. Example on
  N ≥ 2: (1,1) →σ (1,2) →μ (2,1) →σ (2,2) →ν (1,1). So σ sits on a
  directed 4-cycle.
- **Cycle participation in Σ' = {σ,μ,λ₂}:** NO — no generator in Σ'
  decreases e, so e is non-decreasing along any path, and any σ-application
  strictly increases it. Closing a cycle with σ in it requires a strict
  decrease somewhere, which Σ' lacks.

### μ (coord swap)
- **Bijection** on Caps(N,N). Involution: μ ∘ μ = id.
- **Cycle participation:** Always — (b,e) ↔ (e,b) is a 2-cycle (and μ(b,b) =
  (b,b) is a fixed-point self-loop).
- μ has **no failure** mode: image of Caps(N,N) under swap is Caps(N,N).

### λ₂ (scale-up by 2)
- **Not a bijection.** Not surjective (image has only even coords; every
  value in the image is even).
- **Strictly multiplies b,e by 2.** Monovariant: max(b,e) strictly
  increases (unless state is (0,0), excluded by A1). λ₂ alone admits no
  cycle.
- **Cycle participation in Σ:** YES, paired with ν: (b,e) ↔λ₂/ν↔ (2b, 2e)
  whenever legal. Concretely (2,2) →λ₂ (4,4) →ν (2,2) for N ≥ 4.
- **Cycle participation in Σ':** NO (same reason as σ — no decrease).

### ν (halve)
- **Not a bijection.** Not defined when either coordinate is odd; image is
  all of Caps(⌊N/2⌋, ⌊N/2⌋) (since ν(2b, 2e) = (b,e) for any (b,e)
  with b,e ≤ ⌊N/2⌋).
- **Strictly halves** both coordinates. Alone: max strictly decreases, no
  cycle.
- **Cycle participation in Σ:** YES (pair with λ₂, as above).
- **Cycle participation in Σ':** ν ∉ Σ'.

### Summary

| gen | bijection? | cycles in Σ? | cycles in Σ'? |
|---|---|---|---|
| σ  | no | yes (via ν)     | no |
| μ  | yes (involution) | yes (trivially, 2-cycle) | yes |
| λ₂ | no | yes (via ν)     | no |
| ν  | no | yes (via λ₂)    | n/a |

## 3. Legal edge counts (closed forms)

Let **k = ⌊N/2⌋**.

- **|σ-edges| = N · (N − 1).**
  Legal iff e ≤ N−1. Count = #{b} × #{e ≤ N−1} = N(N−1).

- **|μ-edges| = N².**
  Always legal (including self-loops at (b,b)). Count = |Caps(N,N)| = N².

- **|λ₂-edges| = k² = ⌊N/2⌋².**
  Legal iff b ≤ ⌊N/2⌋ AND e ≤ ⌊N/2⌋. Count = k × k = k².

- **|ν-edges| = k² = ⌊N/2⌋².**
  Legal iff b and e both even. Evens in {1,…,N} are {2,4,…,2k}; that's k
  values each. Count = k · k = k². (Image is auto-in-Caps once parity
  passes.)

*Observation:* |λ₂| = |ν|. This is not coincidence — ν∘λ₂ = id on Caps(k,k),
so λ₂ and ν are inverse partial functions on the "doubled" sub-lattice, and
their legal-edge sets are in bijection via (b,e) ↔ (2b,2e).

## 4. Failure counts (closed forms)

Each generator is evaluated at every state in Caps(N,N) (N² trials). Outcome
is legal, OUT_OF_BOUNDS, or PARITY. Per generator, rows sum to N².

### σ
- PARITY: **0** (σ has no parity precondition).
- OUT_OF_BOUNDS: fails iff e+1 > N, i.e., e = N. States (b, N) for b = 1..N.
  Count = **N**.

### μ
- PARITY: **0**.
- OUT_OF_BOUNDS: **0** (swap never exits Caps(N,N)).
- μ is total; **μ never fails.**

### λ₂
- PARITY: **0** (λ₂ has no parity precondition).
- OUT_OF_BOUNDS: fails iff 2b > N OR 2e > N, equivalently b > k OR e > k
  (where k = ⌊N/2⌋). Count = N² − k² = **N² − ⌊N/2⌋²**.

### ν
- PARITY: fails iff b odd OR e odd. Count of (b,e) with at least one odd
  coord = N² − (#both even) = N² − k² = **N² − ⌊N/2⌋²**.
- OUT_OF_BOUNDS: classified after PARITY. Among parity-legal states (b,e
  both even, hence b,e ≥ 2), image (b/2, e/2) has b/2 ≥ 1 and
  b/2 ≤ k ≤ N. So image is always in Caps. Count = **0**.

### Consistency check

Each row of (legal + OOB + PARITY) must sum to N²:

| gen | legal | OOB | PARITY | sum |
|---|---|---|---|---|
| σ  | N(N−1) | N     | 0     | N² ✓ |
| μ  | N²     | 0     | 0     | N² ✓ |
| λ₂ | k²     | N²−k² | 0     | N² ✓ |
| ν  | k²     | 0     | N²−k² | N² ✓ |

## 5. SCC structure

Write **k = ⌊N/2⌋** as before.

### 5a. Under Σ' = {σ, μ, λ₂} (paper's primary statement)

**Claim.** #SCC(G_{Σ'}) = **N(N+1)/2**;
max|SCC|(G_{Σ'}) = **2** for N ≥ 2, **1** for N = 1.

**Structural classification.**
- For each b ∈ {1,…,N}: the singleton **{(b,b)}** is an SCC (μ is a
  self-loop there; σ and λ₂ only lead out, not back).
  → N singleton SCCs.
- For each unordered pair {b,e} with b ≠ e, b,e ∈ {1,…,N}: the 2-element
  set **{(b,e), (e,b)}** is an SCC via the μ-2-cycle.
  → C(N,2) = N(N−1)/2 two-element SCCs.

**Proof sketch.** Let φ(b,e) = b + e. Under Σ': σ adds 1, λ₂ adds (b+e) > 0,
μ adds 0. So φ is non-decreasing along any Σ'-path, with equality iff only μ
is used. A directed cycle must return φ to its starting value, hence can use
only μ. μ-only cycles are exactly the 2-cycles {(b,e),(e,b)} (plus fixed
points (b,b)). Count: N + N(N−1)/2 = N(N+1)/2.

### 5b. Under full Σ = {σ, μ, λ₂, ν}

The answer splits on parity of N.

**Case N = 1.** Caps = {(1,1)}. All of σ, λ₂, ν fail; μ is a self-loop.
#SCC = 1, max|SCC| = 1.

**Case N ≥ 2 even.** **#SCC = 1**, **max|SCC| = N²**. All of Caps(N,N) is
one SCC.

**Case N ≥ 3 odd.** **#SCC = N + 1**, **max|SCC| = (N − 1)²**.

  The N+1 SCCs are:

  1. **One "inner" big SCC of size (N−1)²:** all states (b,e) with
     b ≤ N−1 AND e ≤ N−1.
  2. **N−1 "border 2-cycle" SCCs of size 2 each:** {(N,k), (k,N)} for
     k = 1, …, N−1.
  3. **One "border singleton" SCC of size 1:** {(N,N)}, with only a μ
     self-loop (σ, λ₂, ν all fail there when N is odd).

**Proof sketch.**

*Even N.* From (1,1), a sequence of σ's and μ's reaches any (b,e) ∈
Caps(N,N): σ^{e−1}(1,1) = (1,e); μ(1,e) = (e,1); σ^{b−1}(e,1) = (e,b);
μ(e,b) = (b,e) (all steps legal since intermediate coords ≤ N). Conversely,
from any (b,e) we reach (1,1): if a coord is odd and less than N, advance
it to an even number via σ (using μ to expose the right coordinate); when
a coord equals N and N is even it is already even; then apply ν (halves)
and repeat. Since N is even, there is no "trapped odd N" value, so the
procedure always terminates at (1,1). Thus every state is in the same SCC
as (1,1).

*Odd N.* The key obstruction is that when b = N (N odd), N cannot be
reduced: ν needs b even, λ₂ overshoots (2N > N), σ only increments e, μ
only swaps N into position 2 where the same obstructions recur. Hence the
**border set** B = {(b,e) : b = N or e = N} is forward-closed (no edge
leaves B). Combined with: within B, σ on the non-N coord strictly
increments it, μ is an involution between (N,k) and (k,N), and λ₂/ν never
fire on B — the induced graph on B has exactly the SCCs listed above
(each (N,k) ↔ (k,N) 2-cycle is closed because no other generator moves
inside B across different k's, and (N,N) has only its μ self-loop).

For the inner set I = Caps(N−1, N−1) ⊂ Caps(N,N): edges leaving I go
either (i) to I itself, or (ii) via σ from (b, N−1) to (b, N) ∈ B. Since B
is forward-closed and disjoint from I, edges of type (ii) cannot sit on a
cycle with an I-endpoint. So the SCC decomposition of I in G_Σ equals the
SCC decomposition of I treated as its own Caps(N−1, N−1) with generators
Σ. Because N−1 is even, the even-N argument applies to Caps(N−1, N−1), and
I is a single SCC of size (N−1)².

Summing: N+1 SCCs with sizes (N−1)², 2, 2, …, 2 (×(N−1)), 1.

### 5c. QA-native classification (Cosmos / Satellite / Singularity)

The reachability stratification does admit a loose QA reading:

- **Inner big SCC** = "Cosmos-like": large orbit, algebraically rich,
  genuinely strongly connected.
- **Border 2-cycles** {(N,k),(k,N)} = "Satellite-like": small cyclic
  orbits, closed subsystems that cannot reach the main body.
- **Border singleton** {(N,N)} (N odd) = "Singularity": fixed under the
  only surviving generator (μ self-loop), no escape.

This is a structural analogy, not an mod-9 / mod-24 orbit identity — the
present problem is over Caps(N,N), not over a mod-m cycle, so cycle-length
sizes (24, 8, 1) are not matched numerically. The Cosmos/Satellite/Singularity
trichotomy as *qualitative shape* fits; as numerical orbit lengths it does
not. Flagged as analogical rather than load-bearing.

## 6. Monotonicity under generator expansion

**Claim (direction):**
- **#SCC is non-increasing** as Σ grows: Σ₁ ⊆ Σ₂ ⇒ #SCC(G_{Σ₁}) ≥
  #SCC(G_{Σ₂}).
- **max|SCC| is non-decreasing** as Σ grows: Σ₁ ⊆ Σ₂ ⇒ max|SCC|(G_{Σ₁}) ≤
  max|SCC|(G_{Σ₂}).

**Proof.** E_{Σ₁} ⊆ E_{Σ₂} (same vertex set, every Σ₁-edge is a Σ₂-edge).
So any directed path in G_{Σ₁} is a directed path in G_{Σ₂}. Hence
strong-connectedness in G_{Σ₁} implies strong-connectedness in G_{Σ₂}:
each SCC of G_{Σ₁} sits inside some SCC of G_{Σ₂}. The SCCs of G_{Σ₁} form
a **refinement** of the SCCs of G_{Σ₂}. A refinement has ≥ as many blocks
and every block is contained in a (possibly larger) block of the coarser
partition. Direction follows. □

**Witness of strictness.** Between Σ' and Σ = Σ' ∪ {ν}:
- N=3: #SCC(Σ') = 6, #SCC(Σ) = 4. Strictly decreases.
- N=4: #SCC(Σ') = 10, #SCC(Σ) = 1. Strictly decreases.
- max|SCC|: N=4 goes 2 → 16. Strictly increases.

So both monotonicities are strict on at least one chain Σ' ⊂ Σ.

## 7. Proof sketches (collected)

**(Q2 edge counts.)** Direct domain counting: σ needs e ≤ N−1 → N(N−1); μ
is total → N²; λ₂ needs b,e ≤ ⌊N/2⌋ → ⌊N/2⌋²; ν needs b,e even → ⌊N/2⌋².
The equality |λ₂| = |ν| follows from ν ∘ λ₂ = id_{Caps(⌊N/2⌋, ⌊N/2⌋)}.

**(Q3 failure counts.)** Failure mode assignment is forced by the generator
definitions: σ and λ₂ have only a bounds precondition, hence only OOB
failures; ν has a parity precondition (which dominates — once satisfied,
image is automatically in Caps), hence only PARITY failures; μ is total.
Counts follow by complementing the legal-edge count inside N².

**(Q1 SCC / Q4 monotonicity / Q5 cycle participation.)** The load-bearing
observation is that σ, λ₂, ν each admit a strict monovariant on their own
(σ: e; λ₂: max(b,e); ν: max(b,e) in the other direction), while μ alone is
an involution. Under Σ' = {σ,μ,λ₂}, no generator can undo σ or λ₂, so the
sum-potential φ = b+e is non-decreasing and the only cycles are μ-cycles
→ N(N+1)/2 SCCs of max size 2. Adding ν to Σ' provides the missing
decrease, which collapses SCCs wholesale:

  - If N is even, growing-then-halving reaches (1,1) from anywhere and
    back, so one giant SCC.
  - If N is odd, the value N itself is un-halveable (ν needs even, λ₂
    overshoots, σ only grows e); so any state carrying coord = N is
    absorbed into a forward-closed border B of 2N−1 states. Inside B the
    surviving dynamics is (σ,μ) only (λ₂/ν both trivially fail on N),
    producing N SCCs by the same Σ' argument restricted to B; the inner
    set I behaves like Caps(N−1, N−1) (N−1 even) under full Σ, yielding
    one big SCC.

Monotonicity is pure graph theory: adding edges refines reverse-reachability
coarsening, so SCCs merge.

## 8. QA-native framing applied (or honestly absent)

- **A1 (No-Zero):** **Load-bearing.** Caps(N,N) is {1,…,N}², not
  {0,…,N−1}². The (N,N)-singleton SCC under odd N hinges on N being the
  actual max coordinate value, with no 0-boundary escape — if we used
  {0,…,N−1}², the generator legality conditions shift (ν would need
  divisibility-by-2 checks against a different range) and the count
  structure breaks. Also prevents (0,0) from pseudo-fixed under λ₂/ν.

- **A2 (Derived coords d=b+e, a=b+2e):** **Decorative here.** The
  generators and the counting question are all stated in (b,e); d and a
  do not enter the generator legality or the SCC proof. They would enter
  if the problem asked about reachability restricted to a = constant, or
  about orbit-lifting to a 4-tuple representation, but as stated they are
  just downstream labels. The **sum-potential b+e** I used coincides with
  d — so d = b+e *does* appear as a proof device, just under its raw
  name. Honest assessment: A2 provides the natural monovariant; I would
  call A2 **light-load-bearing** because b+e is precisely d and the
  proof uses it.

- **T1 (Path-time / integer steps):** **Load-bearing.** Each generator
  application is exactly one integer step; edge counts, failure counts,
  and reachability are all stated in units of these steps. No continuous
  time enters. SCC structure is a question about integer-step directed
  reachability.

- **T2 / NT (Observer-projection firewall):** **Vacuously satisfied /
  decorative.** The problem is entirely in the discrete layer. There is
  no continuous input that gets cast to int; no observer projection is
  required. Invoking NT here is honest *restraint* rather than a
  load-bearing application. It would become load-bearing only if the
  problem was extended with a continuous-signal observer feeding
  (b,e) — which it is not.

- **S1 (no `**2`):** **Load-bearing in code, decorative in prose.** All
  count formulas here (N², k², (N−1)²) are mathematical objects, not code;
  the hard rule forbids `b**2` in Python implementations for ULP-drift
  reasons. In a TLA+/Python validator realizing this, one must write `b*b`.
  Prose math is unaffected. Honest: S1 is not contributing to this
  derivation; it constrains any implementation downstream.

- **S2 (no float state):** **Load-bearing by exclusion.** All of b, e, N
  are integers. ν(b,e) = (b/2, e/2) is integer division *conditioned on
  parity* — this is exactly what A1/S2 demands (no float ε, no rounding).
  If we allowed floats, ν would be always-legal and everything collapses.

**Load-bearing axioms for this problem:** A1, A2 (as the b+e potential),
T1, S2. Vacuous: NT (no firewall crossing). Implementation-only: S1.

## 9. Self-check

- [x] A1, A2, T1, T2/NT, S1, S2 each named and classified (load-bearing
      vs. decorative vs. vacuous).
- [x] All closed forms are functions of N only (no hidden parameters; k =
      ⌊N/2⌋ is a definable abbreviation).
- [x] Each closed form is justified by a structural argument — generator
      monovariants (σ: e ↑; λ₂: max ↑; ν: max ↓), forward-closure of the
      border for odd N, graph-refinement for monotonicity — not by
      enumeration.
- [x] N-parity case split stated explicitly (even N: one giant SCC; odd
      N ≥ 3: N+1 SCCs; N = 1 edge case stated).
- [x] Blindness pact held: only `prompts/qa_control_theorems.md`,
      `qa_tla_blind/README.md`, and `CLAUDE.md` were read. No file in
      the forbidden-reads list was opened.

## Summary table

Let **k = ⌊N/2⌋**.

| Quantity | Closed form |
|---|---|
| \|σ-edges\|  | N(N−1) |
| \|μ-edges\|  | N² |
| \|λ₂-edges\| | k² |
| \|ν-edges\|  | k² |
| σ OOB fails  | N |
| σ PARITY fails | 0 |
| μ OOB fails  | 0 |
| μ PARITY fails | 0 |
| λ₂ OOB fails | N² − k² |
| λ₂ PARITY fails | 0 |
| ν OOB fails  | 0 |
| ν PARITY fails | N² − k² |
| #SCC(G_{Σ'})  (Σ' = {σ,μ,λ₂}) | N(N+1)/2 |
| max\|SCC\|(G_{Σ'}) | 2 (N ≥ 2); 1 (N = 1) |
| #SCC(G_Σ), N even ≥ 2 | 1 |
| #SCC(G_Σ), N odd ≥ 3 | N + 1 |
| #SCC(G_Σ), N = 1 | 1 |
| max\|SCC\|(G_Σ), N even ≥ 2 | N² |
| max\|SCC\|(G_Σ), N odd ≥ 3 | (N − 1)² |
| max\|SCC\|(G_Σ), N = 1 | 1 |
| #SCC monotonicity (Σ₁ ⊆ Σ₂) | non-increasing |
| max\|SCC\| monotonicity (Σ₁ ⊆ Σ₂) | non-decreasing |
