# NOVELTY AUDIT (Lane B) — oddN full-Σ SCC decomposition

**Auditor:** general-purpose subagent, fresh context, 2026-04-22.
**Scope:** source-novelty only. NOT a truth judgment.
**Forbidden-reads pact:** HELD. I did not open `attempts/qa_control_theorems.md`,
`diffs/qa_control_theorems.md`, `EVIDENCE.md`, `PROOF_SKETCH.md`,
`verify_extension.py`, `fixtures/`, or `FORMAL_STATUS.json`.

**Authority files consulted (read in full):**

- `/home/player2/signal_experiments/Formalizing tuple drift in quantum-native learning/paper1_qa_control.tex` — 307 lines, full body read: abstract, §1 Introduction, §2 Definitions (state space, generator algebra, failure algebra, reachability), §3 Topological Structure (Lemma 3.1 SCC Monotonicity, Theorem 3.2 μ-Pairing, Theorem 3.3 Edge/Failure Counts), §4 Results (Table 1 Topology, Table 2 Failures, §4.3 Reachability), §5 Discussion, §6 Conclusion, Appendix A.
- `/home/player2/signal_experiments/Formalizing tuple drift in quantum-native learning/files/paper1_qa_control.tex` — 307 lines; `diff` against the primary copy shows byte-identical content, no additional material.
- `/home/player2/signal_experiments/QA_CONTROL_THEOREMS.md` — 51 lines, all three theorems (Lemma SCC Monotonicity, Theorem μ-Pairing, Theorem Edge/Failure Counts) read verbatim; matches the paper exactly.
- `/home/player2/signal_experiments/qa_alphageometry_ptolemy/QARM_v02_Stats.tla` — 181 lines; read all actions (SigmaSucc/Fail_OOB/Fail_FQ, MuSucc/Fail_OOB/Fail_FQ, LambdaSucc/Fail_OOB/Fail_FQ) and all invariants.
- `/home/player2/signal_experiments/qa_alphageometry_ptolemy/QARM_v02_Stats.cfg` — 11 lines; CAP=20, KSet={2,3}, three invariants listed.
- `/home/player2/signal_experiments/qa_alphageometry_ptolemy/ALL_INVARIANTS_VALIDATED.md` — 329 lines; scope is certificate-adapter invariants (fixed_q_mode serialization, generator-set closure, path continuity, packet-delta validation). No SCC / reachability / parity-split content.

## Method

I first read the CLAIM to fix the exact propositions (parts A, B, C). Then I
read `paper1_qa_control.tex` end-to-end and cross-checked the theorem extracts
in `QA_CONTROL_THEOREMS.md` against the paper (identical verbatim). I ran
`grep -niE 'parity|odd|even|border|singleton|inner|cosmos|satellite|singularity|N\+1|N-1|decomposition'`
over the paper to catch any stray remarks about parity-split decomposition
outside the named theorems; the only hits are (a) PARITY *failure type* from
ν requiring even coordinates, and (b) the μ-Pairing decomposition into diagonal
singletons + off-diagonal 2-cycles for the restricted-Σ case. Neither is a
parity-of-N SCC split on full Σ. I read the TLA+ module in full and ran
`grep -niE 'scc|component|connect|reach|orbit'` over it — zero hits. The TLA
spec models the action system (σ, μ, λ with KSet parameterising the scale
factor; no ν at all) and checks three state-level invariants; it does not
enumerate SCCs or assert any component-count property. `QARM_v02_Stats.cfg`
pins CAP=20 (even) and KSet={2,3}; TLC is not run for any odd CAP. The
`ALL_INVARIANTS_VALIDATED.md` report is scoped to certificate-adapter
serialization/closure invariants, not SCC structure.

## Part (A) — Even N #SCC and max|SCC|

- **Exact statement in sources?** No general-even-N theorem. The claim that
  `G_Σ` on `Caps(N,N)` has `#SCC = 1` and `max|SCC| = N²` is stated in the
  paper **only as two tabulated data points** for even N = 30 and N = 50,
  not as a theorem quantified over all even N. See:
  - `paper1_qa_control.tex` §4.1 lines 184: "465 components collapse to 1 for
    Caps(30,30), and 1275 collapse to 1 for Caps(50,50)."
  - §4 Table 1 (lines 196-202): rows for `Caps(30,30)` and `Caps(50,50)` under
    `{σ, μ, λ₂, ν}` give `#SCC = 1`, `Max SCC = 900` and `2500` respectively.
  - Abstract (line 29): "adding a contraction generator ν to the base set
    {σ, μ, λ₂} collapses 465 disconnected components to a single
    fully-connected component in Caps(30,30)."
  No all-even-N theorem is stated.

- **Implicitly equivalent?** Not by a one-step derivation I can find in the
  sources. Lemma 3.1 (SCC Monotonicity) gives `#SCC(G_{Σ∪ν}) ≤ #SCC(G_Σ) =
  (N²+N)/2`, which upper-bounds the count but does not force it to 1. The
  paper's narrative ("induces full connectivity") and Theorem 3.3's edge
  counts do not, on their own, prove `#SCC = 1` for all even N without an
  auxiliary reachability argument that is not supplied in the sources. The
  two even-N tabulated cases support the pattern inductively, not deductively.

- **Evidence of deliberate omission?** No explicit carve-out. The paper's
  Theorem 3.2 restriction is about Σ *without* ν (`no other generator creates
  reverse paths between distinct off-diagonal states`, line 142), which is
  silent on what happens when ν *is* included.

- **Verdict:** `partially_present_as_instances` — stated for even N ∈ {30, 50}
  as empirical table entries; not stated as a theorem over all even N.

## Part (B) — Odd N #SCC and max|SCC|

- **Exact statement in sources?** No. I find **no odd-N data point, no
  odd-N example, and no odd-N theorem** anywhere in the sources. The
  paper's entire Results section uses N ∈ {30, 50}, both even. The TLA+
  config pins CAP = 20, also even. `ALL_INVARIANTS_VALIDATED.md` does not
  touch SCC counts at all. The formulas `N + 1` and `(N - 1)²` do not
  appear anywhere in any of the consulted files.

- **Implicitly equivalent?** No. Lemma 3.1 (monotonicity) combined with
  Theorem 3.2 gives `#SCC(G_{full Σ}) ≤ (N²+N)/2`, far weaker than the
  asserted `N + 1`. There is no hinted relationship in the paper between
  ν-legality (which requires both coordinates even) and the parity of the
  cap N itself. Deriving `#SCC = N+1` and `max|SCC| = (N-1)²` from the
  stated theorems requires an additional argument about why the border row
  `b = N` and column `e = N` become isolated when N is odd — that argument
  is not in the sources.

- **Evidence of deliberate omission?** No explicit exclusion. Theorem 3.2's
  conditioning clause (line 142) — "Σ containing μ ... where no other
  generator creates reverse paths between distinct off-diagonal states" —
  is a hypothesis on the *non-ν* generator family, so it simply does not
  address what happens once ν is added. The paper's silence on odd N looks
  like scope-narrowing rather than deliberate exclusion of a known phenomenon.

- **Verdict:** `not_found`.

## Part (C) — Odd N structural decomposition (inner + border + singleton)

- **Exact statement in sources?** No. The specific three-part partition
  (one `(N-1)²` inner SCC on `Caps(N-1, N-1)`, `N-1` μ-swap 2-cycles
  `{(N,k), (k,N)}`, one singleton `{(N,N)}`) does not appear in any
  consulted file. The paper does describe μ-orbits as 2-cycles `{(b,e),
  (e,b)}` plus diagonal singletons (Theorem 3.2, lines 142-145), but that
  is a decomposition of **the entire** `Caps(N,N)` under restricted Σ, not
  a border-only decomposition under full Σ.

- **Implicitly equivalent?** No. The paper's μ-orbit structure and the
  claim's structural decomposition share the vocabulary "2-cycle" and
  "singleton," which could mislead a casual reader into thinking they are
  the same result. They are not: Theorem 3.2 gives `(N²-N)/2` off-diagonal
  2-cycles across the full lattice (restricted Σ); the claim gives exactly
  `N-1` border 2-cycles confined to the last row/column (full Σ, odd N).
  The Cosmos/Satellite/Singularity identification in the claim's "named
  interpretation" is not in the paper; the paper does not mention QA orbit
  taxonomy at all.

- **Verdict:** `not_found`.

## Related prior art in the sources

The closest prior-art anchor is **Theorem 3.2 (μ-Pairing)**, paper
lines 140-146, quoted verbatim:

> Let $G$ be the directed transition graph on $\mathrm{Caps}(N,N)$ generated
> by $\Sigma$ containing $\mu$ (coordinate swap), **where no other generator
> creates reverse paths between distinct off-diagonal states**. Then the
> SCCs are exactly the $\mu$-orbits: $N$ diagonal singletons $\{(b,b)\}$ and
> $(N^2-N)/2$ off-diagonal pairs $\{(b,e), (e,b)\}$ for $b \neq e$. Thus:
> $\#\mathrm{SCC}(G) = (N^2+N)/2$, $\max|\mathrm{SCC}| = 2$.

The hypothesis (boldface added) is the explicit restriction that matters
here: the theorem applies when Σ = `{σ, μ, λ₂}` (so σ and λ₂ strictly
increase the `V(b,e) = e` potential, per the proof on line 149, and cannot
form cycles). **Adding ν does violate this hypothesis** — ν strictly
decreases the potential `V(b,e) = e` when legal (e ↦ e/2), so ν plus σ can
form directed cycles, breaking the "no reverse paths" condition. The
paper's Table 1 (lines 196-202) observes that this is exactly where the
SCC count collapses; but it does not state what the new SCC structure is
for odd N.

Additionally:

- **Lemma 3.1 (SCC Monotonicity)**, lines 129-138: gives an inequality
  `#SCC(G_{Σ∪ν}) ≤ #SCC(G_{σ,μ,λ₂})`, a strict upper bound but not the
  parity-split counts.
- **Theorem 3.3 (Edge and Failure Counts)**, lines 159-169: closed forms
  for edge and failure counts under any Σ ⊆ {σ, μ, λ₂, ν}. Crucially, the
  ν-edge count `⌊N/2⌋²` already differs by parity of N (for even N this
  is `(N/2)²`; for odd N it is `((N-1)/2)²`), so the machinery to express
  parity splits is present, but the paper's §4 applies this only to
  N = 30 and N = 50.
- **TLA+ `QARM_v02_Stats.tla`:** models σ, μ, and a λ with scale factor
  `k ∈ KSet` on `Caps(CAP)`. The module does **not** include any ν /
  contraction action, has no SCC or reachability invariant, and the `.cfg`
  fixes `CAP = 20` (even). No parity-split TLC tabulation exists.

## Aggregate verdict

`partially_present` — with a precise split:

- **Part (A) even-N:** present as two tabulated instances (N = 30, N = 50)
  in Table 1 and the abstract; not stated as a theorem over all even N.
- **Part (B) odd-N counts:** not found, not implied by a one-step derivation
  from any stated theorem.
- **Part (C) odd-N structural decomposition:** not found, not implied.

No part of the odd-N claim, quantitative or structural, appears in the
authoritative sources. The even-N claim is stated instance-wise, which is
weaker than a theorem but still an explicit appearance.

## Confidence

**High.** The source corpus is small and finite (≈ 1100 total lines across
five files), I read all of it, and I grepped for every relevant term. The
paper quantifies N only through two even examples; the TLA model has
CAP = 20 and no ν action; the certificate-invariants report is out of scope
for SCC structure. Odd N genuinely does not appear.

## What would change the verdict

- An additional source document I was not pointed to (e.g., a companion
  appendix, a QARM variant `.tla` with CAP parameterised and ν included,
  or a prior cert family that enumerates SCCs by parity) containing the
  odd-N counts would move the verdict toward `already_present` or
  `implicitly_equivalent`.
- A one-step derivation I missed from Theorems 3.2 + 3.3 + Lemma 3.1 to
  the odd-N decomposition — but I see no obvious one; the border-isolation
  argument for odd N depends on `⌊N/2⌋ = (N-1)/2` interacting with the
  diagonal fixed-point `(N, N)` in a way that is not articulated in the
  sources and requires its own proof.
- Confirmation that the paper's "full connectivity" phrasing in §4.1 was
  intended to assert the even-N theorem universally (would promote (A) to
  `already_present`), ideally via a stated universal claim I overlooked.
