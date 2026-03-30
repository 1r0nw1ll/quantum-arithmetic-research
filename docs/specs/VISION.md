# QA Project — Strategic Vision & Thread Audit

**Version:** 1.0 | **2026-03-25** | See `CONSTITUTION.md` for the 1-page distillation.

---

## 1. The Thesis

QA is a formal specification language for reality-constrained computation.
Formally: a state-transition algebra over Z[φ]/mZ[φ] (the ring of integers of Q(√5) modulo m)
whose orbit structure partitions every state space into cosmos (24-cycle), satellite (8-cycle),
and singularity (fixed point).

Its defining advantage over prior formal specification systems is **first-class failure algebra**:
QA specifies not only what is reachable under lawful generators but what is structurally
unreachable and why — the geometry of impossibility, machine-checked via the cert ecosystem.

This places QA in a specific lineage and fills a specific gap:

| System | Specifies |
|---|---|
| Hoare logic | Pre/post-conditions on program states |
| TLA+ / Lamport | Behaviors of distributed systems over time |
| Lean / Coq | Types as propositions; proofs as programs |
| **QA** | **Reachable vs unreachable structure in modular dynamical systems + failure algebra** |

---

## 2. The Four Research Tracks

Everything in this project belongs to one of four tracks, or it does not belong.

### Track A — Algebraic Foundations
**Question:** What is the mathematical structure of Z[φ]/mZ[φ]?

- Q(√5) algebraic structure: f(b,e) = b² + be − e² = N(b + eφ), norm in Q(√5)
- T = F² = linear rep of ×φ² in Z[φ]; orbit classification by v₃(f)
- Inert primes p → orbits of size p²−1; spectrum theorem Im(f)={r: v_p(r)≠1}
- Pythagorean families = orbits of F on (Z/9Z)²; Barning-Berggren = parabolic flow
- GF(9) remark: 3 inert → Z[φ]/3Z[φ] ≅ GF(9); five families = φ-orbits in GF(9)²
- **NEW (2026-03-27): Rank-Quantized Modular Fibonacci Recurrence theorem** — `qa_resonance_theorem.md`
  - OFR(k,m) = Fib_hits(π_k)/m exactly for odd m; π_k[j] = π_1[k·j mod m] (subsampling law)
  - Fib_hits(π₁,m) = 1+[m≡1 mod 4] for odd m≥7; exception m=5 → 0 (maximally anti-Fibonacci)
  - m=9 (QA modulus): OFR(k=1) = 2/9 exactly ✓

**Papers:**
- `pythagorean-families/paper.tex` — **ARXIV-READY** (11pp, verify script 10/10 PASS)
- `modular-dynamics/outline.md` — proofs done (inert/split/ramified cases), needs prose

### Track B — Specification Infrastructure
**Question:** How do we make QA claims machine-checkable and independently verifiable?

- QA_CORE_SPEC.v1 [107]: the kernel (state_space, generators, invariants, failure_algebra, gates [0..5])
- Cert ecosystem [18]–[122]: 128 families, meta-validator enforces Gate 0–5 protocol
- QA-ORBIT benchmark: 20,394 tasks (invariant_pred, orbit_class, reachability, shortest_witness)
- [122] Empirical Observation Cert: bridge from Open Brain observations to cert ecosystem
- AlphaGeometry theorem prover: QA guidance for symbolic geometric search

**Papers:**
- `qa_synthetic_data/PAPER_DRAFT.md` — **POLISHED** (NeurIPS D&B track target)
- AlphaGeometry paper (23pp, arXiv-ready, Week 4 complete)

### Track C — Convergence Theory
**Question:** Does QA orbit structure govern gradient descent convergence?

- Finite-Orbit Descent Theorem: L_{t+L} = ρ(O)·L_t (exact identity for scalar quadratic)
- ρ(O) = ∏(1−κ_t)² orbit contraction factor; κ_min > 0 ⟺ ρ(O) < 1
- Empirical: r(mean_κ, final_loss) = −0.845 (gain-robust, 5 seeds, lr sweep)
- Łojasiewicz orbit descent (B1+B2a+B3): φ_{t+L} ≤ φ_t − (1−α)·C(O) [intrinsic, H-crit eliminated]
- Three architecture classes with structural gain: GNN [98], Attention [99], Gradient L2 [101]

**Papers:**
- `unified-curvature/paper.tex` — **ARXIV-READY** (14pp, all theorems proved)
- `agency-decomposition/paper.tex` — complete v3.1 (9pp, awaiting venue)

### Track D — Coherence Detection
**Question:** Do QA orbits observably manifest coherence structure in physical/biological/financial domains?

- **Audio:** ~~OFR as orbit coherence detector~~ — **CLOSED 2026-03-27. Reclassified Track A.** See `qa_resonance_theorem.md`. OFR is a combinatorial aliasing effect, not a dynamical-systems signal. The Track A result is real (see below).
- **EEG:** HI 2.0 achieves +6.7pp over HI 1.0 (68.9% acc) on balanced CHB-MIT chb01; F1=0.735 (13D features) on phase 2 pipeline
- **Finance:** Crypto singularity strategy p=0.0025 (BTC mod-9); 6/6 assets positive SR; modulus = market sophistication parameter. Curvature → vol FAILS cross-asset (finance script 26, correctly documented)
- **Seismic:** Infrastructure validated; awaits labeled IRIS dataset (HI_max 29.8pp separation on synthetic)

**Papers:**
- `phase2_paper_draft_REVISED_HONEST.md` — draft (~10pp), needs CHB-MIT full dataset
- Finance paper — in progress (crypto narrative complete, structural argument needed)

---

## 3. Full Thread Audit

Every work area classified against the four tracks.

| Work Area | Track | Status | Action |
|---|---|---|---|
| Q(√5) algebraic structure | A | Active | Core — protect |
| Pythagorean families paper | A | **ARXIV-READY** | Ship this week |
| Modular dynamics paper | A | Proofs done, outline exists | Write prose now |
| Cert ecosystem [18]–[122] | B | **128/128 PASS** | Maintain |
| QA_CORE_SPEC.v1 [107] | B | Shipped | Foundation — never deprecate |
| QA-ORBIT benchmark | B | Polished | Submit NeurIPS D&B |
| AlphaGeometry theorem prover | B/D | Week 4 complete | arXiv |
| [122] Empirical Observation Cert | B | Shipped 2026-03-25 | Use it — batch certify results |
| Unified Curvature paper | C | **ARXIV-READY** | Ship this week |
| Empirical κ validation (r=−0.845) | C | Done | Already in paper |
| Łojasiewicz proofs (B1+B2+B3) | C | Done | Already in paper |
| Agency Decomposition paper | C | Complete v3.1 | Select venue |
| QALM 2.0 architecture | C (speculative) | Design done, 30–40h unvalidated | **DEFER** — ship Track C papers first |
| Audio OFR → combinatorial aliasing theorem | **A** | **CLOSED 2026-03-27** | Reclassified Track A; see `qa_resonance_theorem.md` |
| EEG seizure detection HI 2.0 | D | Draft exists | Scale to full CHB-MIT (21 patients) |
| Finance crypto singularity | D | p=0.0025 validated | Write paper — structural argument first |
| Finance curvature → vol | D | **CONTRADICTS** pre-declared criteria | Certified as [122]; done |
| Seismic classification | D | Infrastructure ready | Acquire labeled IRIS data |
| Locality Dominance paper | B/D | Complete 14pp | Remote sensing venue |
| PAC-Bayes Phase 1 | C (subsumed) | Subsumed by Unified Curvature | **ARCHIVE** |
| **Quartz/Piezoelectric** | ❌ NONE | Post-hoc QA labeling, unvalidated physics | **ARCHIVE** |
| **ARC-AGI clone** | ❌ NONE | External benchmark, zero QA integration | **ARCHIVE** |
| **GraphRAG / knowledge graph** | ❌ TOOL | Generic retrieval, not a research contribution | **ARCHIVE** |
| **node_modules / package.json** | ❌ TOOL | JS debris | **ARCHIVE** |
| **supabase/** | ❌ TOOL | Tool experiment | **ARCHIVE** |
| **Session logs (123 .md files)** | ❌ | Operational noise from prior sessions | **ARCHIVE** |
| **Workspaces*.zip (7 copies)** | ❌ | Operational noise | **ARCHIVE** |
| **Player4 artifacts** | ❌ | Multi-player coordination artifacts | **ARCHIVE** |
| **Phase1/Phase2 workspaces** | ❌ | Superseded phase artifacts | **ARCHIVE** |
| BSD conjecture certs [82–85] | B (infra demo) | Elliptic curve arithmetic using cert infrastructure; no Z[φ]/orbit connection — proves framework is domain-general | Maintain, do not feature in papers |
| Fairness/Safety certs [90–92] | B (infra demo) | Demographic parity + equalized odds + prompt injection using exact rational substrate — infrastructure generality demo, not QA orbit research | Maintain, do not feature in papers |

---

## 4. Publication Roadmap

Sequenced by dependency and readiness. Ship in order.

### Immediate (this week)
1. **Pythagorean Families** → arXiv + IJNT submission
   - 11pp, all claims proved, verify script 10/10 PASS
   - Author: Will Dale
2. **Unified Curvature** → arXiv
   - 14pp, B1+B2+B3 proved, κ empirical validation r=−0.845
   - Author: Will Dale

### Short-term (1 month)
3. **QA-ORBIT Benchmark** → NeurIPS 2026 D&B track
   - 20,394 tasks, algebraic generalization probe, capacity sweep
4. **AlphaGeometry** → arXiv
   - 23pp, Week 4 complete, 100% correctness preservation

### Medium-term (2–3 months)
5. **Modular Dynamics** → arXiv
   - Inert-prime orbit theorem, orbit size formula p=3..31, GF(p²) structure
6. **Finance: Crypto Singularity** → arXiv / JFE
   - Write structural argument (WHY orbit degeneracy = momentum extreme) first
   - Crypto 4-asset validated, modulus selection frozen
7. **Phase 2 EEG** → signal processing venue (IEEE TNSRE or similar)
   - Expand CHB-MIT to full 21-patient suite; rerun with HI 2.0

### Long-term (3–6 months)
8. **Agency Decomposition** → venue selection (NeurIPS workshop or journal)
9. **Locality Dominance** → remote sensing venue (TGRS or GRSL)
10. **Forensic Coherence Spine** [123]–[125] → when audio + EEG results replicate

---

## 5. What Does NOT Belong

The following work areas have no demonstrated connection to the four tracks.
They are archived, not deleted — retrievable if a connection is later identified.

| Area | Reason | Location after archive |
|---|---|---|
| Quartz/Piezoelectric | Post-hoc QA labeling of physics; no validated contribution | `archive/tool_experiments/quartz/` |
| ARC-AGI | External benchmark clone, no QA solver | `archive/benchmarks/arc_agi/` |
| GraphRAG | Generic knowledge graph tool; not a research contribution | `archive/tool_experiments/graphrag/` |
| node_modules / JS | Wrong language ecosystem for this project | `archive/tool_experiments/js/` |
| Supabase | Tool experiment, no research output | `archive/tool_experiments/supabase/` |
| Session logs (123 files) | Operational artifacts from prior sessions | `archive/session_logs/` |
| Workspaces*.zip (7) | Redundant archives | `archive/zips/` |
| Player4 artifacts | Multi-AI coordination artifacts from prior phase | `archive/handoffs/` |
| Phase 1/2 workspace dirs | Superseded by current structure | `archive/phase_artifacts/` |
| PAC-Bayes Phase 1 | Fully subsumed by Unified Curvature paper | `archive/phase_artifacts/` |

**Classified as Track B infrastructure demos (maintained, not featured):**
- BSD conjecture cert families [82–85]: elliptic curve arithmetic — proves cert framework is domain-general.
- Fairness/Safety certs [90–92]: demographic parity + prompt injection — same rationale.

---

## 6. Immediate Actions

| Priority | Action | Owner | Effort |
|---|---|---|---|
| 1 | Ship Pythagorean Families to arXiv | Will | 1 hour |
| 2 | Ship Unified Curvature to arXiv | Will | 1 hour |
| 3 | Audio autocorrelation baseline experiment | Codex | 2 hours |
| 4 | Batch-certify 10–15 Open Brain results as [122] certs | Claude+Codex | 3 hours |
| 5 | Write structural argument for finance paper (WHY singularity = momentum extreme) | Claude | 1 session |
| 6 | Scale EEG to full CHB-MIT 21-patient suite | Codex | 1 session |
| 7 | Forensic coherence spine [123]–[125] design | Claude | 1 session |
| 8 | qa_core/ shared module (extract QA_Engine) | Codex | 1 session |
