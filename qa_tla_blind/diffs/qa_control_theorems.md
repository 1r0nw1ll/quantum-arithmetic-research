# QA Control Theorems — Scored Diff (POSITIVE CONTROL)

## Ground truth recap

Paper: `Formalizing tuple drift in quantum-native learning/paper1_qa_control.tex`.
Verbatim extract: `QA_CONTROL_THEOREMS.md`.

Paper results, cited by file + line:

- **Lemma (SCC Monotonicity under Generator Expansion)** —
  `QA_CONTROL_THEOREMS.md:6-15`, `paper1_qa_control.tex:129-138`.
  If Σ₁ ⊆ Σ₂, then `#SCC(G_{Σ₂}) ≤ #SCC(G_{Σ₁})`. Proof: adding edges can
  only merge SCCs, never split them.
- **Theorem (SCC Count under μ-Pairing)** — `QA_CONTROL_THEOREMS.md:17-34`,
  `paper1_qa_control.tex:140-157`. Under Σ containing μ with no other
  generator creating reverse paths between distinct off-diagonal states,
  SCCs are exactly the μ-orbits: `#SCC = (N²+N)/2`, `max|SCC| = 2`.
  Potential-function proof using `V(b,e) = e` (strictly increases under
  σ and λ₂).
- **Theorem (Edge and Failure Counts)** — `QA_CONTROL_THEOREMS.md:36-50`,
  `paper1_qa_control.tex:159-173`. Closed forms: `|σ|=N(N-1), |μ|=N²,
  |λ₂|=|ν|=⌊N/2⌋²`. Failures: σ→N OOB, λ₂→N²−⌊N/2⌋² OOB, ν→N²−⌊N/2⌋² PARITY.

**Additional paper content beyond the three theorems** (§4 Results, Table 1
`paper1_qa_control.tex:196-202`): empirical validation on Caps(30,30) and
Caps(50,50) under three generator sets:

- `{σ,λ₂}`: `#SCC = N²`, `max|SCC| = 1` (all singletons).
- `{σ,μ,λ₂}`: `#SCC = (N²+N)/2`, `max|SCC| = 2`.
- `{σ,μ,λ₂,ν}` on **even** N=30, N=50: `#SCC = 1`, `max|SCC| = N²`.

Paper does **not** state the odd-N behavior of the full Σ — only even-N
cases are tabulated. Paper's §3 proves Theorem 2 (μ-pairing) for sub-Σ; the
full-Σ result is asserted empirically in §4.

**TLA+ invariants validated** (`QARM_v02_Stats.tla:177-179`,
`ALL_INVARIANTS_VALIDATED.md`): `Inv_TupleClosed` (d=b+e, a=d+e);
`Inv_InBounds` (all coords in 0..CAP); `Inv_QDef` (qtag = Q definition). These
are **per-state invariants** on the QARM transition system, not the SCC/edge
counts that the paper derives. TLA spec uses **0-based** `0..CAP` (vs paper's
1-based `{1,…,N}`) — noted but not penalizing, per scoring rules.

## Per-result scoring

Attempt file: `qa_tla_blind/attempts/qa_control_theorems.md`, summary table
lines 382-403.

| # | Result | Ground truth | Attempt | Verdict |
|---|---|---|---|---|
| 1 | \|σ-edges\| | `N(N-1)` (line 40, paper) | `N·(N−1)` (attempt L106, table L382) | **reproduced** |
| 2 | \|μ-edges\| | `N²` (line 41) | `N²` (L110, L383) | **reproduced** |
| 3 | \|λ₂-edges\| | `⌊N/2⌋²` (L42) | `⌊N/2⌋²` (L112, L384) | **reproduced** |
| 4 | \|ν-edges\| | `⌊N/2⌋²` (L43) | `⌊N/2⌋²` (L115, L385) | **reproduced** |
| 5 | σ OUT_OF_BOUNDS count | `N` (L45) | `N` (L131, L386) | **reproduced** |
| 6 | λ₂ OUT_OF_BOUNDS count | `N² − ⌊N/2⌋²` (L45) | `N² − ⌊N/2⌋²` (L142, L390) | **reproduced** |
| 7 | ν PARITY count | `N² − ⌊N/2⌋²` (L45) | `N² − ⌊N/2⌋²` (L145, L393) | **reproduced** |
| 8 | #SCC under Σ'⊇μ (paper Thm 2) | `(N²+N)/2` | `N(N+1)/2` (L169, L394) | **reproduced** (identical formula) |
| 9 | max\|SCC\| under Σ' | `2` | `2` for N≥2; `1` for N=1 edge case (L169, L395) | **reproduced**, with correct N=1 edge case |
| 10 | Monotonicity under generator expansion | #SCC non-increasing | "non-increasing as Σ grows" (L258, L402); also adds max\|SCC\| non-decreasing (L260, L403) | **reproduced + strengthened** |

Extras (not in paper's three theorems but present in paper's Table 1):

| Extra | Ground truth | Attempt | Verdict |
|---|---|---|---|
| #SCC under `{σ,λ₂}` | `N²` | implicit from cycle analysis (no potential cancellation possible in Σ'₀={σ,λ₂}, so all singletons); not stated as explicit closed form | **missed (minor)** — structurally equivalent statement not tabulated |
| Total edges Caps(30,30) full Σ | 2220 | `N(N-1)+N²+2⌊N/2⌋²` = 870+900+225+225 = **2220** ✓ | **reproduced** (derivable from table) |
| Total fails Caps(30,30) full Σ | 1380 | 30+675+675 = **1380** ✓ | **reproduced** |
| #SCC Caps(30,30) full Σ | 1 | 1 (even-N case, attempt L194, L397) | **reproduced** |
| #SCC Caps(50,50) full Σ | 1 | 1 (even-N case) | **reproduced** |

Recovery: **10/10 primary results reproduced**, one minor "missed" on the
`{σ,λ₂}` degenerate-case count (but implicit in the attempt's cycle
analysis).

## Results the attempt derived BEYOND the paper

The attempt adds a **parity-split SCC analysis under full Σ** (attempt
§5b, L185-236, L396-400):

- N even ≥ 2: #SCC = 1, max|SCC| = N².
- N odd ≥ 3: #SCC = N+1, max|SCC| = (N−1)². Structural decomposition:
  one inner (N−1)² SCC (Caps(N−1,N−1)), (N−1) border 2-cycles
  `{(N,k),(k,N)}`, one border singleton `{(N,N)}`.
- Plus a loose QA-native reading: inner = Cosmos-like, border 2-cycles
  = Satellite-like, border singleton = Singularity — **explicitly flagged
  as analogical, not numerical orbit-length identity** (attempt L252-253).

**Does the paper contain this analysis?** No. The paper's Table 1 only
reports even N (30, 50), and §4 asserts "#SCC=1 under full Σ" without
proving it or case-splitting on N parity. The attempt's odd-N result is
a **genuine extension**.

**Small-N verification** (Tarjan SCC on enumerated Caps(N,N)):

```
N=1: #SCC=1, max|SCC|=1                        — attempt: 1, 1  ✓
N=2: #SCC=1, max|SCC|=4                        — attempt: 1, N²=4  ✓
N=3: #SCC=4, max|SCC|=4, sizes=(4,2,2,1)       — attempt predicts N+1=4,
      max=(N−1)²=4, and sizes (4, 2, 2, 1)      ✓ (exact match)
N=4: #SCC=1, max|SCC|=16                       — attempt: 1, N²=16  ✓
N=5: #SCC=6, max|SCC|=16, sizes=(16,2,2,2,2,1) — attempt predicts N+1=6,
      max=(N−1)²=16, sizes (16, 2×4, 1)          ✓ (exact match)
N=6: #SCC=1, max|SCC|=36                       — attempt: 1, N²=36  ✓
N=30: #SCC=1, max|SCC|=900                     — attempt & paper agree  ✓
```

Also the Σ'={σ,μ,λ₂} sub-case: N=1,…,30 all reproduce `(N²+N)/2` exactly.
And {σ,λ₂}: all N produce `#SCC=N²` (all singletons, max|SCC|=1). All
attempt formulas verified numerically.

The attempt's odd-N proof sketch (L222-234) correctly identifies the
load-bearing structural fact: **when b=N and N is odd, N is un-halvable by
ν (needs even), un-scalable by λ₂ (2N>N), and σ/μ cannot reduce it**, so
the border set `B = {(b,e) : b=N or e=N}` is forward-closed. This is a
clean, non-enumerative argument.

Verdict on the extension: **correct and non-trivial**. This is a
genuine derivation beyond the source paper.

## Two-axis score

### Recovery score

Of the 10 primary results:
- **Reproduced exactly: 10/10.**
- **Strengthenings (with justification): 2.** (i) Monotonicity stated for
  `max|SCC|` (non-decreasing) in addition to `#SCC`, with proof; (ii)
  N=1 and odd-N edge cases for `max|SCC|` under Σ'.
- **Weakenings or misses: 0** (the `{σ,λ₂}` closed form is implicit in the
  attempt's cycle analysis but not explicitly tabulated — borderline miss).
- **Wrongs: 0.**

**Recovery = 10/10 + 2 strengthenings + 1 implicit omission.**

### Contribution score (0-4)

Applying the README §Specific markers of Contribution ≥ 3:

| Marker | Present? | Evidence |
|---|---|---|
| Generator-relative structure (named σ/μ/λ₂/ν analysis) | ✓ | §2, table L95-100, per-generator bijection/cycle analysis |
| SCC / orbit organization of reachable state graph | ✓ | §5a (Σ'), §5b (full Σ with parity split) |
| Closed-form counts (not enumeration) | ✓ | All of §3, §4, §5 in closed form in N |
| Failure-class algebra (OOB, PARITY counted) | ✓ | §4, consistency table L155-160 |
| Monotonicity under generator expansion | ✓ | §6 with refinement-argument proof, plus strengthening to `max\|SCC\|` |

**All 5 markers present.** Further, the attempt produces a **result the
source paper does not contain** (odd-N parity split with Cosmos/Satellite/
Singularity-like structural decomposition), verified by enumeration, with
a clean border-forward-closure argument.

QA-axiom self-assessment (attempt §8, L313-358) is honest and correct:

- A1 (No-Zero): correctly identified as **load-bearing** — the odd-N SCC
  obstruction hinges on N being the actual max coord with no 0-boundary
  escape.
- A2 (Derived coords): honestly assessed as **light-load-bearing** — the
  sum-potential φ = b+e used in the Σ' proof is exactly `d`, but the
  problem doesn't use `a` or the 4-tuple structure.
- T1 (Path-time): load-bearing (integer step semantics).
- T2/NT (firewall): **correctly called vacuous** — no continuous input
  enters, no firewall crossing required. This is honest restraint, not
  ornamental invocation.
- S1 (no `**2`): correctly flagged as implementation-only.
- S2 (no float state): load-bearing by exclusion (ν is integer division
  conditional on parity, not float rounding).

The self-assessment distinguishes load-bearing axioms from decorative
ones without ornamentation — exactly the honesty the rubric asks for.

**Final Contribution score: 4 (decisive).**

**Justification (2-3 sentences).** The attempt produces every primary
closed form in the paper, exactly, plus a genuine extension (the parity
split on full Σ) that is verifiable by enumeration and proven by a clean
forward-closure argument — not ornamental overlay. The QA-axiom load
assessment correctly separates load-bearing (A1, T1, S2) from vacuous
(NT) axioms, and the Cosmos/Satellite/Singularity reading is explicitly
flagged as analogical rather than claimed as a numerical orbit identity.
This is the Contribution-4 shape: closed forms over a discrete generator
algebra, SCC/orbit organization, failure-class algebra, monotonicity
under generator expansion, plus a non-trivial novel result.

## Failure taxonomy tags

- **None triggered.**

Specifically:
- Not `no-mapping-exists`: mapping is clean 1-based Caps(N,N).
- Not `wrong-observer-projection`: problem is fully discrete, observer
  layer correctly identified as vacuous.
- Not `orbit-mismatch`: SCC decomposition matches enumeration exactly.
- Not `invariant-inexpressible`: all properties stated in closed form.
- Not `proof-gap`: all proofs sketched with correct load-bearing lemmas
  (monovariants, refinement, border forward-closure).
- Not `qa-stronger-than-tla` / `qa-weaker-than-tla`: indexing convention
  (1-based attempt vs 0-based TLA `0..CAP`) is a spec-convention shift,
  not a QA-vs-TLA semantic divergence. Attempt correctly matches paper.
- **Not `ornamental-overlay`** — axioms are load-classified honestly;
  NT is called vacuous, A1 is shown to do real work, the b+e potential
  function is identified as coinciding with d (A2).

Expected for positive control: zero tags. **Achieved: zero tags.**

## Blindness check

Searched attempt for forbidden phrases and identifiers:

- **Theorem/lemma names**: attempt does not use "SCC Monotonicity under
  Generator Expansion" or "SCC Count under μ-Pairing" or "Edge and
  Failure Counts" as labeled phrases. Attempt calls §5a "paper's primary
  statement" generically; calls §5b its own derivation. ✓
- **Potential-function notation**: paper's proof uses `V(b,e) = e`.
  Attempt uses `φ(b,e) = b+e` (different variable, coincides with `d`).
  Attempt also notes the per-generator monovariants (σ: e↑; λ₂: max↑;
  ν: max↓) — distinct from the paper's V(b,e)=e single-variable proof.
  **Different proof strategy, same theorem.** ✓
- **TLA+ identifiers** (`Inv_TupleClosed`, `Inv_InBounds`, `Inv_QDef`,
  `qtag`, `SigmaSucc`, `MuSucc`, `LambdaSucc`, `KSet`, `CAP`): none
  appear in the attempt. ✓
- **Closed forms stated as quotes**: attempt derives each closed form
  with a constructive argument (domain counting, complement counting,
  cycle analysis). None are stated as "per the paper" or similar.
  Formulas are framed as claims with proofs, not quotations. ✓
- **Explicit numerics from paper** (2220 edges, 1380 fails, 465 SCCs,
  900, 1275): attempt quotes N=3/N=4 specific numerics (e.g. L272-274:
  "N=3: #SCC(Σ') = 6, #SCC(Σ) = 4" and "N=4: #SCC(Σ') = 10, #SCC(Σ) = 1")
  which are derivable from its own closed forms, not from the paper's
  tabulated N=30/N=50 numbers. **No paper-specific numerics appear.** ✓

**Blindness pact: HELD.** The attempt's opening statement (L1-8) asserts
the pact held, and there is no textual evidence of leakage.

## Calibration implications

**Does the benchmark's Contribution = 4 endpoint have a real example now?**
Yes. This attempt, scored under the rubric as written, earns Contribution
4 with every marker triggered and a correct extension. The endpoint is
anchored: on the POSITIVE CONTROL, QA framing does reproduce + extend the
validated result without ornamental overlay.

**Is DieHard's Contribution = 0 score interpretable on a real scale?**
Yes. DieHard scored Contribution 0 (`ornamental-overlay`): recovered both
invariants but added no compression, no SCC/orbit structure, no closed
forms over generators, no failure algebra. That 0 and this 4 now sit on
the same scale with concrete evidence at both ends. A spec's Contribution
score going forward is interpretable as "how close to the QA-control
theorem shape, on the axes the rubric names."

**Any adjustments to the rubric based on what the reproducer produced?**

Minor sharpenings worth considering:

1. **"Extension beyond the source" is a signal for Contribution 4 even
   on non-positive-control specs.** The odd-N parity split the attempt
   derived was NOT in the paper; paper only covers even N. Rubric
   currently says "positive-control endpoint is 4 if the QA control
   theorem shape is achieved" — should add: "if the attempt derives a
   verifiable novel result beyond the source while meeting all other
   markers, that's also a Contribution-4 signal." (This is different
   from strengthening an existing invariant; it's adding a new one.)

2. **Honest load-assessment should be an explicit Contribution ≥ 3
   requirement**, not implicit. The attempt §8 correctly classifies NT
   as vacuous here — ornamental overlay would invoke NT as load-bearing
   without cause. Rubric should say: "attempts that invoke all six
   axioms as load-bearing without differentiating get demoted to
   Contribution ≤ 2 regardless of other markers."

3. **Convention-shift allowance** should be documented: positive control
   attempt uses 1-based `{1,…,N}` matching the paper, while the TLA spec
   uses 0-based `0..CAP`. Neither is wrong; the scorer must check
   against the source convention the attempt references, not the TLA
   convention. DieHard didn't have this issue (small-int 0-3 range);
   larger specs will.

## Headline finding

The benchmark's POSITIVE CONTROL reaches Contribution 4 cleanly: the
blinded reproduction recovers all 10 primary closed forms, derives a
verified novel extension (odd-N parity split on full Σ) the source paper
did not state, and honestly load-classifies the QA axioms (NT vacuous,
A1/T1/S2 load-bearing) — anchoring the high end of the scoring scale
against which DieHard's Contribution-0 ornamental-overlay result is now
interpretable.
