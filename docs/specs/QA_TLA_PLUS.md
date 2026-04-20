# QA ↔ TLA+: Constitutional Layer and Formal Specification

**Rehydrated:** 2026-04-20 from ChatGPT exports that had never been migrated to the repo.
**Source provenance:** A-RAG msg_ids `4fc6fb687d69e30a`, `d366c74451763e20`, `da4315a42e09c28a`, `6f09b30907bd3143`, `6ec9d69144769f3a`, `e138120bd1fe88e7`. Original exchange dated circa 2025-03-03 (per `qa_tla_summary` doc metadata).
**Primary sources:** (Lamport, 1994) ACM TOPLAS 16(3) DOI:10.1145/177492.177726 on the Temporal Logic of Actions; (Lamport, 2002) *Specifying Systems* ISBN 978-0-321-14306-8 for the TLA+ language and TLC model checker; (Cousineau et al., 2012) LNCS 7436 for TLAPS. Cert-gate protocol cross-references: arXiv:2603.18829 (Agent Control Protocol), arXiv:2603.23801 (AgentRFC composition safety).
**Audit anchor:** `docs/specs/QA_TLA_PLUS_AUDIT.md` (2026-04-20) — state audit that surfaced this migration gap.

---

## 1. Purpose of this document

A 2025 ChatGPT exchange with Will produced the **architectural framing that the existing `QARM_v02_*.tla` specs on disk are a direct implementation of**. The connection was never recorded anywhere in the repo, so future sessions had no way to reach the strategic reasoning behind:

- Why `qtag = 24 * Phi9(a) + Phi24(a)` (duo-modular packing) in `qa_alphageometry_ptolemy/QARM_v02_Failures.tla:41`.
- Why `fail ∈ {OK, OUT_OF_BOUNDS, FIXED_Q_VIOLATION, ILLEGAL}` is a first-class state variable rather than a transition guard in the same file, line 55.
- Why failures are encoded as absorbing-stutter states preserving the tuple rather than as silent transition refusals (line 216–217: `\/ /\ fail # "OK" /\ UNCHANGED <<...>>`).

Those design choices were deliberate and flow from the three-layer constitutional architecture documented below. Recovering this context prevents future sessions from re-deriving it or, worse, breaking it by "simplifying" the fail-state machinery.

---

## 2. The three-layer architecture (from the ChatGPT "FP for QA" exchange)

> **Source: msg `6ec9d69144769f3a` (assistant), conv "FP for QA", verbatim excerpt.**

> **Executive recommendation (commit-level)**
>
> **Yes: use TLA⁺. But only as the *constitutional layer*, not as your programming language.**
>
> If you try to *implement* QA in TLA⁺, you will stall. If you use TLA⁺ to *define what is allowed*, QA will harden instantly.
>
> ---
>
> **The correct role of TLA+ in QA**
>
> TLA⁺ should answer exactly **three** questions for QA:
>
> 1. What is a legal QA state?
> 2. Which moves are legal from that state?
> 3. Which things can *never* happen (impossibility theorems)?
>
> That's it. Everything else — performance, learning, visualization — belongs elsewhere.
>
> ---
>
> **Recommended QA architecture (lock this in):**
>
> **🔐 Layer 1 — TLA⁺ = QA Constitution.** Authoritative, slow, exact, unambiguous. Defines: state variables `(b,e,d,a, q, phase24, phase9, …)`; generator actions `σ, μ, λ, ν`; invariants (canonical packet, non-reduction, fixed-q); failure classes (as unreachable states); reachability claims (SCC isolation, collapse, return-in-k). Used for: proving impossibility, proving phase transitions, freezing definitions permanently. Nothing experimental goes here.
>
> **⚙️ Layer 2 — Rust (FP-style) = QA Engine.** Fast, executable, testable. All moves are pure functions. All moves return `Result<State, FailType>`. `FailType` is an enum that mirrors the TLA⁺ failure algebra. Property tests mirror TLA⁺ invariants 1:1. Rust is not the authority. Rust is the *implementation under oath*.
>
> **🧠 Layer 3 — RML / learning.** Allowed to explore, guess, optimize — but **never** to redefine truth. Learners only choose *among legal moves*. They never create moves. They never override invariants. They can fail, but failures are classified by Layer 2. This cleanly solves Epistemia: learning ≠ truth.
>
> **The declared rule: Any change to QA legality must be made in TLA⁺ first.** This single rule prevents silent theory drift.

**Status of this architecture today (2026-04-20):**

- **Layer 1:** PARTIALLY REALIZED. `QARM_v02_Failures.tla` + `QARM_v02_NoMu.tla` + `QARM_v02_Stats.tla` + `QACertificateSpine.tla` exist under `qa_alphageometry_ptolemy/`, but have never been TLC-model-checked in the 2026 era (no `.cfg`, no proof-ledger entry, no TTrace). See `QA_TLA_PLUS_AUDIT.md` §1 Region B.
- **Layer 2:** NOT IN RUST. The QA engine is implemented in Python across per-domain scripts (`QA_Engine`/`QASystem` re-implemented per file with variations — see `CLAUDE.md` "Working with this codebase"). The cert-gate wrapper layer (`llm_qa_wrapper/`) IS hardened following an analogous discipline and has been model-checked (81,629 states clean, Lean-verified — see `QA_TLA_PLUS_AUDIT.md` §1 Region A).
- **Layer 3:** PARTIALLY REALIZED. RML sketches in `qa_lab/AI_ARCHITECTURE_INTEGRATION.md`; actual learning code isn't bound to a TLA+ legality contract.
- **Declared rule:** NOT ENFORCED. Nothing prevents QA-legality changes from being made in Python first. The audit (§5 Lane 2) proposes `QAAxioms.tla` as a first step toward enforcement at the axiom layer.

---

## 3. Why TLA+ matches QA (from msg `6f09b30907bd3143`)

> **Source: msg `6f09b30907bd3143` (assistant), conv "FP for QA", verbatim excerpt, structural edits minor.**

> Short answer: **yes — TLA⁺ is *extremely* well-suited to QA**, but **not in the way most people initially think**. It's not a replacement for a functional language; it's a **different layer of rigor** that fits QA almost uncannily well.
>
> **What TLA+ actually is (and isn't):**
>
> - ❌ Not an implementation language
> - ❌ Not a functional language
> - ✅ A formal state-transition logic
> - ✅ A reachability + invariant + failure-mode language
>
> **Why TLA⁺ matches QA shockingly well:**
>
> **1. QA states = TLA states.** Your QA state is `⟨ tuple, invariants, phase tags, lineage ⟩`. In TLA⁺: `VARIABLES tuple, invariants, phase, lineage`. No mutation. Just states.
>
> **2. QA moves = TLA actions.** Your generators σ, μ, λ, ν become TLA⁺ actions:
>
> ```tla
> σ ≜ /\ legality_conditions
>      /\ tuple' = σ(tuple)
>      /\ invariants' = invariants
> ```
>
> Every move explicitly states when it is legal, what changes, and what *must not* change. This directly enforces your non-reduction axiom and canonical invariants.
>
> **3. QA invariants = TLA INVARIANT clauses.**
>
> ```tla
> Invariant ≜
>   /\ C = 2*e*d
>   /\ G^2 = C^2 + F^2
>   /\ phase24 = a % 24
> ```
>
> And the model checker will try *every possible move sequence* to break it. This is anti-Epistemia by construction.
>
> **4. Failure modes become *first-class objects*.** In QA, failures are not bugs — they are OUT_OF_BOUNDS, PARITY, NON_REDUCTION, FIXED_Q_VIOLATION. In TLA⁺, illegal transitions simply do not exist; or, if you want them observable, you encode them explicitly as states. This maps perfectly to your principle: "Failure modes form a finite, classifiable algebra." TLA⁺ was designed for exactly that.
>
> **Where TLA⁺ is better than FP for QA:**
>
> > **QA is fundamentally a control / reachability theory.** That's TLA⁺'s native domain.
>
> TLA⁺ excels at: reachability (can state t be reached from s?), connected components (are these states in the same SCC?), phase transitions (what minimal generator set causes connectivity collapse?), "return-in-k-steps" bounded reachability, proving **impossibility results** (this move can *never* succeed).
>
> Functional languages *implement* these ideas. TLA⁺ **proves** them.
>
> **Where TLA⁺ is not enough:**
>
> | Task | TLA⁺ | FP |
> |------|------|-----|
> | Formal move legality | ✅ | ✅ |
> | Invariant proofs | ✅ | ⚠️ |
> | Reachability / SCC reasoning | ✅ | ⚠️ |
> | Executable QA engine | ❌ | ✅ |
> | Performance / scaling | ❌ | ✅ |
> | Learning algorithms | ❌ | ⚠️ |
> | Numeric exploration | ❌ | ✅ |
>
> So TLA⁺ should **never** be your runtime QA engine.
>
> **The architecture that actually fits QA:**
>
> TLA⁺ is one of the few tools that lets you formally prove "this can never happen." Given how often you say things like "this component is isolated," "this failure is fundamental," "this generator cannot close the defect" — you are already thinking in temporal logic. You just hadn't named it yet.

---

## 4. Duo-modular q + first-class failures (from the constitutional-layer thought)

> **Source: msg `e138120bd1fe88e7` (thought-level), conv "TLA⁺ as constitutional layer for QA", verbatim excerpt.**

> ChatGPT's advice to canonize `q := (φ₉(a), φ₂₄(a))` as duo-modular is exactly right. This locks in your strongest invariance results and prevents retroactive ambiguity about what "fixed-q" meant in your theorems.
>
> The packing strategy (`qtag = 24*φ₉ + φ₂₄`) is particularly elegant — it gives you a single Nat in range [0,239] that TLC can hash efficiently while preserving both symmetries. This is better than a record for model checking performance.
>
> **Critical next step: making failures first-class**
>
> ChatGPT's suggestion to add explicit failure states is the highest-ROI move. Your current spec blocks illegal transitions (they vanish), but your actual QA theory says failures are algebraic objects with invariant properties. The spec should reflect this.
>
> The upgrade would add:
>
> ```tla
> VARIABLES
>   b, e, d, a,
>   qtag,
>   fail  \* {"OK", "OUT_OF_BOUNDS", "PARITY", "NON_REDUCTION", "FIXED_Q_VIOLATION"}
> ```
>
> Then actions like:
>
> ```tla
> SigmaFail_OUT_OF_BOUNDS ==
>   /\ e + 1 > CAP
>   /\ fail = "OK"
>   /\ fail' = "OUT_OF_BOUNDS"
>   /\ UNCHANGED <<b, e, d, a, qtag>>
> ```
>
> This makes your "absolute invariance of OUT_OF_BOUNDS failure counts" directly model-checkable: you can ask TLC to verify that the number of states with `fail = "OUT_OF_BOUNDS"` is invariant across different generator sets.
>
> **Recommendation: TLA+ v0.2 with explicit failure encoding.** The constitution must come before the implementation. Once you have failures as first-class TLA+ objects, the Rust mirror becomes mechanical: `enum FailType` maps 1:1 to the TLA+ fail states, and property tests directly check TLA+ invariants. This prevents the common trap where implementation details (like how Rust represents failures) accidentally become definitional.

**Direct repo trace:** This exact design lives today at `qa_alphageometry_ptolemy/QARM_v02_Failures.tla`. Compare:

- Duo-modular packing: line 41 — `QDef(bv, ev, dv, av) == 24 * Phi9(av) + Phi24(av)`.
- First-class fail variable: lines 22–26 — `VARIABLES b, e, d, a, qtag, fail, lastMove`.
- Fail state domain: line 55 — `fail \in {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}`.
- Absorbing-stutter on failure: lines 216–217 — `\/ /\ fail # "OK" /\ UNCHANGED <<b,e,d,a,qtag,fail,lastMove>>`.
- Three generator actions σ / μ / λ_k: lines 83–207 with paired `Succ` / `Fail_OUT_OF_BOUNDS` / `Fail_FIXED_Q` actions.

The `QARM_v02_NoMu.tla` variant (μ removed from Next) and `QARM_v02_Stats.tla` variant (with `PrintT` instrumentation) are designed to answer the exact empirical question posed in §3 verbatim: *"what minimal generator set causes connectivity collapse?"* — and the ChatGPT thought above: *"verify that the number of states with `fail = "OUT_OF_BOUNDS"` is invariant across different generator sets."* **This experiment has never been run. It is the single-highest-leverage TLC run available in the repo right now.**

---

## 5. The `qa_tla_summary` structured document (2025-03-03)

> **Source: msg `4fc6fb687d69e30a` (user, full doc pasted) and `d366c74451763e20` (assistant, document artifact). Title: "Formal Specification and Verification of Quantum Arithmetic Using TLA+", dated 2025-03-03. Verbatim.**

> **Abstract.** This study explores the application of **TLA+ (Temporal Logic of Actions)** to **Quantum Arithmetic (QA)** by defining mathematical structures, transformations, and invariants for formal verification. A TLA+ specification was developed to model the **(b, e, d, a)** number system, ensuring correctness of state transitions. The **TLC model checker** was employed to validate critical QA properties, including rational point calculations on an ellipse. The findings contribute to the formalization of QA, enabling potential applications in quantum computing, cryptography, and mathematical theorem verification.
>
> **Key Findings:**
> - Developed TLA+ specifications for (b, e, d, a) transformations.
> - Implemented state transitions and invariant verification in TLA+.
> - Identified and corrected syntactic and logical errors during execution.
> - Ensured monotonicity properties (e.g., non-decreasing sums).
> - Verified rational point calculations on quantum ellipses.
>
> **Mathematical formulations:**
>
> Invariant property for result consistency: `result = operand1 + operand2`
> State transition relations: `operand1' = operand1 + 1`, `operand2' = operand2 + 2`, `result' = operand1' + operand2'`
> Correctness condition for rational points: `4 × d × e = Number of Rational Points on Ellipse`
>
> **The following TLA+ specification defines the Quantum Arithmetic model (initial v1 sketch):**
>
> ```tla
> --------------------------- MODULE QAformalization ---------------------------
> EXTENDS Naturals
>
> VARIABLES result, operand1, operand2
>
> Init ==
>     /\ result = 0
>     /\ operand1 = 0
>     /\ operand2 = 0
>
> Next ==
>     /\ operand1' = operand1 + 1
>     /\ operand2' = operand2 + 2
>     /\ result' = operand1 + 1 + operand2 + 2
>
> Spec == Init /\ [][Next]_<<result, operand1, operand2>>
>
> INV ==
>     /\ operand1 >= 0
>     /\ operand2 >= 0
>     /\ result' >= result
>     /\ result' = operand1 + 1 + operand2 + 2
>
> THEOREM QA_Correctness == Spec => []INV
> =============================================================================
> ```
>
> **Limitations & Refinements (from the 2025-03-03 document):** Scalability issues (scope limited to basic transformations); state explosion risk (large-scale models require optimizations in TLC checking); integration gaps (alignment with Lean / Coq is future work).
>
> **Future research directions declared in the original doc:**
> 1. Extending QA to higher-dimensional arithmetic (modular forms, matrix operations).
> 2. Linking TLA+ models to cryptographic protocols for post-quantum security validation.
> 3. Implementing automated theorem discovery using AI-assisted logic checking.
> 4. Optimizing model checking algorithms for handling larger arithmetic frameworks.
> 5. Connecting QA to quantum circuit design using TLA+ for formal quantum gate verification.

**Note on the 2025-03-03 `QAformalization` sketch vs the 2025-12-30 `QARM_v02_Failures` spec on disk:** the QAformalization sketch is a toy module (operands 1 & 2 incrementing) that does not yet encode the (b,e,d,a) tuple algebra. `QARM_v02_Failures.tla` is the v0.2 version advocated in §4 — it supersedes `QAformalization` and should be treated as canonical. The `QAformalization` module is historical.

---

## 6. Earlier framing reference (from msg `da4315a42e09c28a`)

> **Source: msg `da4315a42e09c28a` (user), conv "TLA+ for Quantum Arithmetic", verbatim (the first introductory exchange before the three-layer architecture crystallized).**

> **Applying TLA+ to Quantum Arithmetic**
>
> TLA+ is a formal specification language developed by Leslie Lamport for modeling and verifying concurrent and distributed systems. It uses simple mathematics to describe system behaviors and properties, allowing for precise modeling and verification.
>
> While TLA+ is traditionally used in computer science, its mathematical foundation makes it adaptable to other domains, including Quantum Arithmetic (QA). Here's how TLA+ can be applied to QA:
>
> 1. **Modeling Quantum Arithmetic Structures**: QA involves unique numerical frameworks, such as the (b, e, d, a) decomposition and Fibonacci Box structures. These can be represented in TLA+ by defining appropriate data structures and state variables. For instance, a Fibonacci Box could be modeled as a record with fields corresponding to its components.
>
> 2. **Specifying Transformations and Operations**: QA operations, such as transformations within the Fibonacci Box or calculations of Quantum Ellipse parameters, can be specified as actions in TLA+. These actions define how the system transitions from one state to another, capturing the essence of QA operations.
>
> 3. **Verifying Properties and Invariants**: TLA+ allows the specification of invariants — properties that must hold true in all reachable states. In the context of QA, invariants could include relationships like `b × e = in-circle radius` or `d × a = ex-circle radius of G`. The TLA+ model checker, TLC, can then be used to verify these invariants, ensuring the correctness of QA operations.
>
> **Benefits of Using TLA+ for QA:** Precision (rigorous mathematical framework), Verification (with TLC), Abstraction (modeling QA concepts at a high level without implementation details).
>
> **Getting Started:** Familiarize yourself with TLA+ syntax and semantics (see the TLA+ Home Page for resources and tutorials); start by modeling basic QA structures and operations in TLA+; use the TLA+ Toolbox and the TLC model checker to validate your QA models.

---

## 7. Integration with current repo state (2026-04-20)

### What exists on disk today

| ChatGPT recommendation (2025) | Repo artifact (2026-04-20) | Status |
|---|---|---|
| TLA+ v0.2 with `fail` first-class + duo-modular qtag | `qa_alphageometry_ptolemy/QARM_v02_Failures.tla` | Implemented Dec 30, never TLC-checked |
| Model-check generator-set differential | `QARM_v02_NoMu.tla` + `QARM_v02_Stats.tla` | Scaffolded Dec 30, never run |
| Cert architecture + failure-first-class | `QACertificateSpine.tla` | Authored Jan 21, never TLC-checked |
| Cert-gate protocol (ACP/AgentRFC) — separate infra layer | `llm_qa_wrapper/spec/cert_gate*.tla` + Lean 4 | **Model-checked Apr 19 (81,629 states clean); Lean-verified; production-grade** |
| Rust engine (Layer 2) | — | NOT in Rust; QA engine is Python per-domain |
| Learning as Layer 3 (bound to Layer 1 contract) | `qa_lab/AI_ARCHITECTURE_INTEGRATION.md` + per-experiment code | Sketched; contract not enforced |
| Declared rule: "Any change to QA legality must be made in TLA⁺ first" | `tools/qa_axiom_linter.py` (text-level axiom enforcement) | Linter is strong but operates on source text, not the formal spec. Proposed `QAAxioms.tla` (audit §5 Lane 2) would close this. |

### Forward pointers

- **Run TLC against the existing QARM specs.** Audit §5 Lane 1 details the first step (author `QARM_v02_Failures.cfg`, author one negative-test spec, run, record in `QARM_PROOF_LEDGER.md`). The generator-set differential experiment (`QARM_v02_NoMu` vs `QARM_v02_Failures` failure-count comparison) is a ~5-minute TLC run and directly answers the question posed in §4 verbatim above.
- **Lift the 6 QA axioms into TLA+.** Audit §5 Lane 2 proposes `QAAxioms.tla` EXTENDS `QARM_v02_Failures` with `Inv_A1_NoZero`, `Inv_A2_DerivedCoords`, `Inv_T1_IntegerPathTime`, and (harder) `Inv_NT_NoObserverFeedback`. This realizes the declared rule of §2: changes to QA legality must pass through TLA+ first.
- **The wrapper-layer discipline is the template.** `llm_qa_wrapper/spec/TLC_PROOF_LEDGER.md` documents the Phase 2a/2b pair-proof protocol (positive spec + per-invariant negative test + Lean plumbing). Apply the same template to the QARM stack.

---

## 8. Canonical position statement

For the record, harvested from this prior work:

1. **TLA+ is QA's constitutional layer, not its implementation language.** The TLA+ spec is the authority; the code is guilty until it agrees.
2. **Failures are first-class algebraic objects in QA, not silent transition blocks.** Encode them as observable state variables with their own domain.
3. **QA is a control / reachability theory over arithmetic state space.** That is TLA+'s native domain — the fit is structural, not metaphorical.
4. **Cross-layer rule:** any change to QA legality must be made in TLA+ first. Rust / Python / learning code mirrors the TLA+ spec; never the other way round.

These four statements predate the current cert ecosystem and axiom linter. They are consistent with VISION.md §1's "first-class failure algebra" differentiator and with Theorem NT's observer-projection firewall. The claim of this document is that they should be treated as **standing architectural invariants** until Will explicitly supersedes them.

---

## References

- Lamport, L. (1994). *The Temporal Logic of Actions.* ACM TOPLAS 16(3), 872–923. DOI: 10.1145/177492.177726.
- Lamport, L. (2002). *Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers.* Addison-Wesley. ISBN 978-0-321-14306-8.
- Cousineau, D., Doligez, D., Lamport, L., Merz, S., Ricketts, D., & Vanzetto, H. (2012). *TLA+ Proofs.* In *FM 2012: Formal Methods* (LNCS 7436), 147–154.
- de Moura, L., & Ullrich, S. (2021). *The Lean 4 Theorem Prover and Programming Language.* In *CADE 28* (LNCS 12699), 625–635.
- Liu et al. *Agent Control Protocol.* arXiv:2603.18829.
- *AgentRFC: Composition Safety for Agent Systems.* arXiv:2603.23801.
- TLA+ examples corpus: github.com/tlaplus/examples (Paxos, Raft, distributed snapshots — Lamport, 2001).
- Repo cross-refs: `docs/specs/QA_TLA_PLUS_AUDIT.md`, `docs/specs/VISION.md` §1, `llm_qa_wrapper/spec/TLC_PROOF_LEDGER.md`, `qa_alphageometry_ptolemy/QARM_v02_Failures.tla`.
