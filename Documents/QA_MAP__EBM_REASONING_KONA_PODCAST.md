# QA Mapping: “Reasoning Energy-Based Models” (Eve Bodnia / Logical Intelligence) → QA

This is a **human-tract** mapping of the video you provided into **QA’s canonical language**:
**state manifold, generators, invariants, failure taxonomy, reachability, determinism/invariant_diff**.

Canonical QA definitions are **verbatim** from:
`Formalizing tuple drift in quantum-native learning/files/files(1)/qa_canonical.md`.

Machine-tract mapping object (passes `QA_MAPPING_PROTOCOL.v1` gates):
`Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`.

---

## 0) Video claims (as provided)

The episode frames “Kona” as a **token-free reasoning** model that:

- reasons via an **energy landscape** (low energy = consistent / correct),
- avoids LLM-style “guessing next token,”
- enforces constraints for **planning / robotics / safety-critical systems**,
- optionally uses **LLM as an interface** (language ≠ core intelligence),
- can be **verifier-gated** (Lean mentioned in the excerpt),
- exhibits **phase-transition-like** scaling regimes (EBM-dominant vs LLM-dominant).

This mapping treats those as **semantic claims**, not as verified performance assertions.

---

## 0.1) QA Mapping Protocol view: M = (S, G, I, F, R, C)

QA’s mapping intake contract (`QA_MAPPING_PROTOCOL.v1`) requires a concrete object:

- `S` (**state manifold**): what a “candidate solution/state” is.
- `G` (**generators**): legal moves / transitions.
- `I` (**invariants**): hard constraints and derived equalities that must hold.
- `F` (**failure taxonomy**): typed, witness-producing failures.
- `R` (**reachability**): the induced directed graph relation and component semantics.
- `C` (**determinism contract**): unique successor or typed FAIL, plus invariant_diff.

For this video’s architecture claim (“LLM UI + EBM core + verifier”), the corresponding machine-tract mapping object is:
`Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`.

---

## 1) QA canonical anchors (verbatim)

### 1.1 State manifold (QA primitives + derived packet)

```text
(b, e) ∈ ℤ₊²
```

```text
d = b + e
a = e + d = b + 2e
```

**Critical Constraint:** `d` and `a` are **always derived**. They are **never** independent degrees of freedom.

### 1.2 Generator algebra (QA “moves”)

```text
σ(b, e) = (b, e+1)
```

```text
μ(b, e) = (e, b)
```

```text
λ₂(b, e) = (2b, 2e)
```

```text
ν(b, e) = (b/2, e/2)   if b,e both even
         = FAIL        otherwise
```

```text
Σ_full = {σ, μ, λ₂, ν}
```

### 1.3 Failure taxonomy (typed, deterministic)

```text
Fail(s, g) = (move, fail_type, ΔI)
```

```text
fail_type ∈ {OUT_OF_BOUNDS, PARITY, PHASE_VIOLATION, INVARIANT, REDUCTION}
```

**Critical Property:** Failures are **deterministic and reproducible**, not stochastic.

### 1.4 Reachability (QA “reasoning” primitive)

```text
return_in_k(s → R*, k, Σ) :=
    ∃ (g₁, ..., g_T) ∈ Σᵀ, T ≤ k :
        g_T ∘ ⋯ ∘ g₁(s) ∈ R*
```

This is the canonical definition of “solve/planning” in QA: **reach a target class** under a generator set and invariants.

---

## 2) Core translation: EBM “energy landscape” = QA potential over a discrete reachability graph

### 2.1 EBM inference → QA reachability under a potential

EBM story (as described): map a problem into an energy landscape and “go downhill” to a valid basin.

QA translation:

- **State manifold**: the structured configuration space (in QA canonical: `Caps(N,N)` over `(b,e)`; in Sudoku/robotics: variable assignments / plans).
- **Generators**: the legal transitions (QA canonical: `σ, μ, λ₂, ν`).
- **Energy**: a deterministic scalar functional `E_task(s)` used for navigation, defined as an **exact** penalty sum over violated constraints/invariants.
- **Reasoning**: find a path `s0 → … → s*` such that `E_task(s*)` is minimal (or under a threshold) **and** invariants hold.

This is exactly “energy landscape navigation” but stated in QA primitives:

> **Energy-guided reachability search** over the directed transition graph induced by `Σ`.

### 2.2 “Evaluate many solutions at once”

QA translation: “many solutions at once” means the solver’s policy can **score** many reachable candidates (states) using `E_task` before committing to a move.

This is not a different semantics of reachability; it is an **optimization policy** over the same reachability relation.

---

## 2.3) Where “energy” lives in QA (and where it must NOT live)

To keep the “no hallucinations” claim meaningful, QA forces a separation:

- **Energy is advisory**: it is a heuristic/potential `E_task(s)` used to prioritize moves or candidates.
- **Invariants are authoritative**: hard constraints must be enforced as **gates**, not “encouraged” by energy shaping.

In other words, in QA you may use energy to *navigate*, but you must use invariants/verifiers to *accept*.

---

## 3) “LLMs hallucinate; constraints matter” → QA: invariants, determinism, failure completeness

The video’s “hallucination” critique maps cleanly to QA’s notion of invalidity:

### 3.1 Hallucination = invariant violation (or missing witness)

In QA terms, an output is “hallucinatory” if **any** of the following hold:

1) **Invariant breach**: a hard constraint fails (e.g., phase constraint `q`, legality constraints, derived-coordinate constraints).
2) **Undeclared nondeterminism**: same input can yield different outputs without an explicit stochastic contract.
3) **Missing invariant_diff**: the system cannot explain “what changed” via a deterministic `ΔI` / witness.

This is why QA requires:

- typed failures (`Fail(s,g)=(move, fail_type, ΔI)`),
- deterministic reproducibility,
- explicit reachability witnesses (paths).

### 3.2 Constraint enforcement = choosing where constraints live

QA clarifies *where constraints must live* to make “no hallucinations” meaningful:

- **Hard constraints** live as **gates** on transitions (reject illegal states), not just as “soft penalties.”
- **Soft constraints** may shape energy, but cannot replace gates in safety-critical contexts.

This directly matches the podcast’s “constraints matter” thesis.

---

## 4) Hybrid architecture (EBM + LLM) → QA: codec boundary + interface-loss accounting

Video claim: “Language is interface; EBM is core; attach LLM as UI if desired.”

QA translation: treat the LLM as a **codec generator** that compiles language into formal constraints/objectives.

Minimum QA requirements for that boundary:

- **Explicit constraint emission**: language → constraints (not prose-as-authority).
- **Interface-loss witness**: quantify ambiguity/underspecification/contradiction introduced at the codec boundary.
- **Non-authoritative outputs**: LLM outputs become proposals until accepted by invariant gates / verifiers.

This is how QA makes “LLM as UI” non-hand-wavy: it becomes a certificate boundary with its own failure algebra.

---

## 5) “Formal verification (Lean)” → QA: external verifier as a generator + witness-producing gate

When the excerpt says “attach to external verifier like Lean,” QA treats this as:

- a new generator in the generator set (decision procedure / verifier call),
- with deterministic outputs and **witnesses** on rejection (failed theorem, type error, unsat core, etc.).

In the QA spine, this is the clean way to cash out the “correctness” claim:

> correctness = **verifier-accepted**; otherwise return `VERIFIER_REJECTED` with obstruction witness.

---

## 5.1) “Sudoku as benchmark” → QA: clean invariant manifold, clean failure witness

Sudoku is a good demo precisely because it is the **QA ideal case**:

- state manifold: 9×9 grid assignments,
- invariants: row/col/box uniqueness,
- energy: total violation count (or penalty sum),
- failure: any wrong digit is an explicit invariant breach.

QA note: Sudoku’s “hallucination” is *definitionally measurable* (a violated constraint), which is why it is often used to illustrate the gulf between:

- **prose plausibility** (LLM),
- **constraint satisfaction** (EBM/solver),
- **verifier acceptance** (external checker).

---

## 6) “Scaling theory; phase transitions” → QA: SCC monotonicity + generator injection

The episode describes regime shifts (EBM dominates vs LLM dominates).

QA has a direct, already-proved phase-transition analogue:

- **SCC Monotonicity**: adding generators can only merge SCCs, never split them.
- **Connectivity transition**: on `Caps(30,30)`, expanding `{σ,μ,λ₂}` by adding `ν` yields `#SCC = 1` (full connectivity).

QA interpretation of “phase transition”:

> A qualitative capability jump occurs when generator expansion crosses a barrier and changes reachability classes (often visible as SCC merges).

This is the precise QA version of “new reasoning behavior appears once capacity/structure crosses a threshold.”

---

## 6.1) EBM-vs-LLM “regimes” → QA: which generators dominate the path cost

The episode’s “EBM-dominant vs LLM-dominant” scaling story maps to QA as:

- When the target task is **constraint-dominant** (hard invariants, tight feasibility), the shortest witness path is dominated by **constraint-preserving generators** + verifier checks.
- When the task is **language-bandwidth-dominant** (long text parsing, summarization), the path is dominated by **codec generators** (language parsing/translation) and their interface-loss.

QA makes this explicit: “dominance” is about **generator cost composition** along successful reachability witnesses.

---

## 7) What QA would require to accept the strongest podcast claims

### 7.1 “Almost can’t hallucinate”

QA acceptance criteria (minimum):

- determinism contract (unique successor or typed FAIL),
- hard invariant gates (not just penalty shaping),
- failure-complete taxonomy (every refusal is typed + witnessed),
- verifier bridge for safety-critical outputs (Lean/SMT/custom),
- run bundling / chain-of-custody (digests linking codec → trace → verifier).

### 7.2 “Token-free reasoning”

QA reframing:

- “token-free” is not the essential property;
- the essential property is **reasoning over nonlinguistic state manifolds** and **certifying reachability** under invariants.

Language can exist strictly as a codec boundary.

---

## 7.3) “Latency space / latency variables” → QA: additional state coordinates must be explicit

The excerpt attributes EBM reasoning feasibility to “latency space/variables.”

QA translation:

- If you add “latent/working-memory” variables to support planning, they are simply **new coordinates** in the state manifold.
- QA requires them to be explicit in `S`, and to be governed by invariants (what can change, what must be preserved, what counts as drift).

This prevents a common failure mode in hybrid systems:

> “Correctness lives in an untracked latent variable” (uncertifiable).

In QA, any latent variable that affects decisions must be:

- named,
- constrained,
- diffed (`ΔI` / invariant_diff),
- and included in determinism contracts.

---

## 8) Machine-tract deliverable

A concrete `QA_MAPPING_PROTOCOL.v1` mapping object for the Kona/EBM story is provided at:

`Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`

Validate it with:

```bash
python qa_mapping_protocol/validator.py Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json
```

---

## 9) Repo pointers (what this mapping connects to)

- Canonical QA definitions (must-use, immutable): `Formalizing tuple drift in quantum-native learning/files/files(1)/qa_canonical.md`
- Mapping protocol contract + validator: `qa_mapping_protocol/`
- Certificate spine (exact arithmetic, canonical hashing, failure-complete validation): `qa_alphageometry_ptolemy/`
  - Core primitives: `qa_alphageometry_ptolemy/qa_cert_core.py`
  - Meta-validator + Gate 0 mapping intake: `qa_alphageometry_ptolemy/qa_meta_validator.py`
  - Generator injection (capability as reachability): `qa_alphageometry_ptolemy/qa_generator_injection_certificate.py`
