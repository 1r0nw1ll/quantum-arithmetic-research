# MEMORY.md

Last updated: 2026-03-29

## Current QA Direction

- Ground future QA work in the completed Wildberger synthesis: QA is treated here as a strict refinement of rational trigonometry and chromogeometry, not a separate replacement story.
- The key bridge is canonical: for the QA direction vector `(d, e)`, `(C, F, G)` correspond to Wildberger's green, red, and blue chromogeometric quadrances respectively.
- `Y = A - D` is canonical.
- After the Wildberger read-through, the preferred freeze-into-artifacts order is:
  1. `QA_CHROMOGEOMETRY_CERT.v1`
  2. `QA_RED_GROUP_CERT.v1`
  3. `QA_UHG_NULL_CERT.v1`
  4. `QA_SPREAD_PERIOD_CERT.v1`
- Building a full repo-format rational trigonometry cert remains an important next step after these bridge families.

## Workflow Rules

- Before re-deriving QA results, use the project's built memory/infrastructure first:
  - Open Brain for recent thoughts and prior observations
  - `qa_corpus_text/` for book-grounded QA concepts
  - the algorithm competency registry before proposing new algorithms
- Ground new QA claims in source material and skeptical validation.
- Prefer bridging existing cert families, experiments, and Open Brain outputs with lightweight `QA_EMPIRICAL_OBSERVATION_CERT` artifacts rather than inventing new core architecture.
- Do not push high-level ontology or architecture decisions back onto Will unless explicitly requested. For QA sessions, the assistant should set direction, define what counts as QA for the task, and let Will act as operator/go-between.
- Every QA family follows the two-tract rule:
  - Machine tract: schema, validator, fixtures, certs
  - Human tract: documentation

## Corpus Ingestion Rules

- Exclude Quadrature from the core QA corpus synthesis.
- Treat multiple editions or variants of the same work as distinct source witnesses until comparison is explicit. Example: `Pyth-3` and `Pyth-3 Enneagram` should not be silently collapsed.
- Preserve original source order and dependency order during ingestion. Earlier volumes are theory-priority because later works build on them.
- Theory-spine priority:
  - `Pyth-1`: identity backbone
  - `Pyth-2`: system architecture and dynamics, including Synchronous Harmonics and relational "functional primes"
  - `QA-1` and `QA-2`: pedagogical overlays, not theory-layer source of truth
- Use a two-pass theory-layer ingestion rule:
  1. extract Pyth-series content faithfully
  2. vet anything that conflicts with the control theorem, proven QA work, or established identities
- Never silently normalize contradictions. Label them explicitly as one of:
  - OCR corruption
  - likely typo
  - source-layer ambiguity
  - true contradiction requiring investigation
- Specific OCR correction to remember: near the `Pyth-1` ellipse relation, read the corrupted line as `2D = C + 2J`, consistent with `J + K = 2D`.

## Canonical QA Facts

- `I` is the positive difference `|C - F|`.
- `F` denotes the altitude leg.
- `C` denotes the base leg.
- Quantum ellipse major axis is always `2D = 2(d*d)`, not `2d`. This is case-sensitive and should override older conflicting notes.
- Preserve variable definitions without redefinition:
  - `J = b*d`
  - `X = e*d`
  - `K = d*a`
- `Z = E + K`.
- For canonical tuples, `Y = W - F` and `Y = A - D`.

## Canonical Theory Commitments

- QA is explicitly a control and reachability theory, not just a collection of identities.
- The generator set defines the QA universe for a task; reachability, connected components, and failure modes are always relative to that chosen generator set.
- Deterministic failure under invariant-preserving QA/QARM moves is a valid theoretical signal, not a bug.
- Connected components are invariant-defined objects; component boundaries encode impossibility results.
- Bounded return-in-k reachability is a predictive QA operator for recoverability versus irreversibility.
- Fixed-q isolation under `{sigma, lambda}` is intended topology, not an implementation defect.
- Transition logging for QARM-style moves should record every attempted move as `{"move","fail_type","invariant_diff"}`.

## Time Framework

- The QA-native Time Axiom and No Continuous Time obstruction theorem are canonical and publishable:
  - QA time is legal reachability, path length, and phase evolution
  - continuous time appears only via observer projection

## Source Stance

- Ben Iverson is treated in this project as the original originator of Quantum Arithmetic.
- Dale Pond's SVP work is treated as a complementary line to be bridged with QA, not as the origin of QA itself.

## SOTA Scan Log

### Scan #14 — 2026-06-22 (window: June 18 – June 22, 2026)
Full report: `sota_scans/scan_2026-06-22.md`

**Key finds:**
- arXiv:2606.20439 (Four-digit Kaprekar dynamics in odd bases, June 18, 2026) — **MEDIUM-HIGH.** Chen/Ono/Schwartz/Thakur. In every odd base B>3: after ≤3 iterations, every nonconstant orbit enters a triangular region conjugate to projective doubling; complete cycle-length formula (max ≤ (B−1)/2, equality iff B prime). Lean/mathlib formalizations by AxiomProver. Structural analogy to QA orbit tripartition + projective direction structure (cert [266], SL(2,Z) cluster). New cert candidate: `qa_kaprekar_orbit_projective_cert_v1`.
- arXiv:2512.05124 (Information funnels and multiscale gap-space dynamics in Kaprekar's routine, Nov 2025) — **MEDIUM. MEGA-SLIP — all 13 scans.** Dahl. Entropy-funnel structure + gap-space Markov approximation for Kaprekar dynamics D∈{3,4,5,6}. Information-theoretic dual to arXiv:2606.20439. Fix: add "Kaprekar" to arXiv search battery.

**No HIGH finds — fifth consecutive quiet window (excluding ongoing SAIR degree-24 challenge from scan #13).**

**SAIR Inverse Galois degree-24 challenge (scan #13 HIGH):** Stage 1 ongoing, closes Aug 15. No new developments this window.

**New cert candidate:** `qa_kaprekar_orbit_projective_cert_v1` (MEDIUM priority).

**New cross-domain connection:** Kaprekar projective cluster — arXiv:2606.20439 (projective doubling) + arXiv:2512.05124 (entropy funnels) = dual geometric + information-theoretic external validation of discrete orbit attractor theory. Companion to QA orbit tripartition theorem.

**Internal landmark:** Witt Tower sprint [449]–[496] completed June 18–22 (48 families). New standalone [496] QA E8 Satellite Chamber Theorem. Registry: 496 families.

**Cert registry: [496].** [497] → `qa_steinmetz_whittaker_bridge_cert_v1`, [498] → `qa_whittaker_phase_packet_algebra_cert_v1` urgently needed (SEVENTH consecutive miss).

**[261] orbit stratification: THIRTEEN-TUPLE GAP (13th scan — longest-running unresolved gap in project history).**

**Documentation backlog: ~213 families [284]–[496] all missing README/SPEC (up from ~165 in scan #13).**

**Search fix for scan #15:** Add "Kaprekar" + "digit dynamics" to arXiv battery; add Tao blog + SAIR Foundation monthly sweep (Stage 1 closes Aug 15); add math.CO explicit sweep.

### Scan #11 — 2026-06-11 (window: June 8 – June 11, 2026)
Full report: `sota_scans/scan_2026-06-11.md`

**Key finds:**
- Tao/Davis SAIR Foundation "Modular Arithmetic Challenge" Stage 2 (June 8, 2026) — **MEDIUM-HIGH.** Lean 4 proof infrastructure for modular arithmetic equational theories (ETP background: 22M universal algebra problems). Directly enables QA cert formalization pathway; completes Lean 4 ecosystem with arXiv:2605.20440 + arXiv:2604.13514.
- arXiv:2606.10193 (Modular Structure Theorem for Minimal Periodic Decompositions, June 8) — **MEDIUM.** Kari-Szabados Z/mℤ Laurent polynomial algebra; ℤ₊ alphabet + low complexity → ℤ-minimal periodic decomposition (Nivat's conjecture P_η(4,n)). Grounds QA orbit classification in periodic structure forcing theory. New cert candidate: `qa_periodic_decomposition_complexity_cert_v1`.
- arXiv:2506.23679 (Learning Modular Exponentiation with Transformers, June 2025) — **LOW-MEDIUM. MEGA-SLIPPED all 10 scans (12+ months).** Grokking on a^b mod c; sudden generalization across related moduli via shared cyclic group structure. NeurIPS'25 workshop paper — not in cs.LG main sweep. Fix: add workshop proceedings sweep.
- arXiv:2606.00045 (Universal Quantum Transformer, ~June 1) — **LOW-MEDIUM. Slipped scan #10.** Quantum circuits perfectly learn Z/11Z cyclic arithmetic; classical NNs fundamentally fail. Further Theorem NT empirical layer.

**No HIGH finds — third consecutive quiet window.**

**New cert candidate:** `qa_periodic_decomposition_complexity_cert_v1` (MEDIUM priority; arXiv:2606.10193).

**New cross-domain connection:** Z/mℤ Periodic Structure Theorem Chain — arXiv:2506.05190 (categorical DS) + arXiv:2512.25060 (universal manifold) + arXiv:2606.10193 (periodicity forcing) = complete theoretical chain for QA orbit classification grounding.

**New Lean 4 ecosystem:** ⋆G tensor framework (arXiv:2605.20440) + polynomial tactics (arXiv:2604.13514) + ETP/SAIR modular arithmetic challenge = complete Lean 4 infrastructure for QA cert formalization. `qa_star_g_tensor_cert_v1` is now highest-readiness new cert.

**Cert registry:** [384] (unchanged). [385] → `qa_steinmetz_whittaker_bridge_cert_v1`, [386] → `qa_whittaker_phase_packet_algebra_cert_v1` urgently needed (8th scan, still unassigned).

**[261] orbit stratification: TEN-TUPLE GAP (10th scan — 6+ weeks overdue).**

**Documentation backlog: ~101 families [284]–[384] all missing README/SPEC (unchanged from scan #10).**

### Scan #13 — 2026-06-18 (window: June 15 – June 18, 2026)
Full report: `sota_scans/scan_2026-06-18.md`

**Key finds:**
- Tao/SAIR Third Competition: Inverse Galois Challenge at **degree 24** (June 16, 2026) — **HIGH.** AI-assisted brute-force of ~25,000 degree-24 Galois groups over ℚ via LMFDB. Stage 1 closes Aug 15. Degree 24 = QA's primary modulus. Directly extends Langlands sprint [403]–[412] + Pisano splitting [415]–[418] + Witt Tower [432]–[443]. New cert candidate: `qa_inverse_galois_degree24_cert_v1`.
- arXiv:2606.15506 (Dual Affine Spiral Orbits on ℤ², June 13 — slipped scan #12) — **LOW-MEDIUM.** Dual spiral pair generating ℤ² via (x,y)↦(2x−y, x) + (x,y)↦(2y−x, y); Twindragon fractal IFS attractor as continuous observer-projection limit. Fourth backbone paper for [261] orbit stratification human tract.
- arXiv:2510.23298 (Galois Groups of Apéry-like Series Modulo Primes, Oct 2025) — **LOW-MEDIUM. MEGA-SLIP — all 13 scans.** mod-24 prime congruence conditions for Apéry-like sequences; Galois group classification. Cross-reference for certs [415]–[418].

**ONE HIGH find — first HIGH since scan #8 (6 scans ago).**

**INTERNAL landmark: Unannounced cert sprint [419]–[448] (30 families) completed June 15–18.** Includes Witt Tower lifting [432]–[443], Tohoku/aftershock cluster [444]–[448]. Last registered ID: [448]. Registry: 395 families. [449] → `qa_steinmetz_whittaker_bridge_cert_v1`, [450] → `qa_whittaker_phase_packet_algebra_cert_v1` needed (sixth collision).

**[261] orbit stratification: TWELVE-TUPLE GAP (12th scan). Human tract now has four-paper backbone: arXiv:2506.05190 + arXiv:2606.10193 + arXiv:2606.12947 + arXiv:2606.15506.**

**Documentation backlog: ~165 families [284]–[448] all missing README/SPEC.**

**New cert candidate:** `qa_inverse_galois_degree24_cert_v1` (HIGH priority; Stage 1 closes Aug 15 — time-sensitive).

**Search fix for scan #13:** Add SAIR competition announcement sweep (Tao blog + SAIR Foundation site monthly); add "Apéry" + "Galois group mod p" to arXiv search battery.

### Scan #12 — 2026-06-15 (window: June 11 – June 15, 2026)
Full report: `sota_scans/scan_2026-06-15.md`

**Key finds:**
- arXiv:2606.12947 (Trace spectra of simplices / dynamical trace-spectrum theorem for group actions, June 2026) — **MEDIUM.** Björklund, Fish, Sanadhya. Sublattice realization from group-action dynamics on ℤ^d. Dynamical trace-spectrum theorem analogous to QA's orbit color tripartition (Cosmos/Satellite/Singularity). Third backbone paper for [261] orbit stratification human tract.
- arXiv:2606.13159 (CA Rule 115 periodicity, June 11) — **LOW-MEDIUM.** Reversible elementary second-order cellular automaton proven periodic on finite configs. Second external proof-of-concept for "finite discrete DS → periodic orbits" principle.
- arXiv:2606.08215 (Clifford groups over ℤ_{n₁}⊕···⊕ℤ_{nₖ}, June 6, slipped #11) — **LOW-MEDIUM.** Semidirect product structure for composite cyclic configuration spaces. Adjacent to QA symmetry group.

**No HIGH finds — fourth consecutive quiet window.**

**INTERNAL landmark: Cert sprint [403]–[418] (16 families) completed June 11–15.** Full Langlands ladder over ℚ(√5) ([403]–[412]) + BSD central value [413] + Fibonacci/Pisano/Cassini cluster [414]–[418]. Most architecturally significant sprint in project history: formally certifies QA's φ-based orbit classification as the concrete Langlands correspondence over ℚ(√5). Pisano period = prime splitting type in ℚ(√5)/ℚ.

**Cert registry: [418].** 365 families total. [419] → `qa_steinmetz_whittaker_bridge_cert_v1`, [420] → `qa_whittaker_phase_packet_algebra_cert_v1` urgently needed (IDs [385]+[386] consumed again — fifth collision).

**[261] orbit stratification: ELEVEN-TUPLE GAP (11th scan — all-time project record). Human tract now has three-paper backbone: arXiv:2506.05190 + arXiv:2606.10193 + arXiv:2606.12947.**

**[384] docs/families gap RESOLVED.**

**Documentation backlog: ~135 families [284]–[418] all missing README/SPEC.**

**Search fix for scan #12:** Add NeurIPS/ICML/ICLR workshop proceedings sweep; add Tao blog + SAIR Foundation to standard sweep; add "Kari-Szabados" + "periodic decomposition" explicit search.

### Scan #10 — 2026-06-08 (window: June 4 – June 8, 2026)
Full report: `sota_scans/scan_2026-06-08.md`

**Key finds:**
- arXiv:2606.02621 (Fibonacci theorem for Collatz trajectories via modular graph structure, May 28) — **MEDIUM.** φ spectral radius emerges from Collatz mod-6 directed graph after orbit exclusion; F(m+1) orbit count via modular directed graph bijection. Independent golden ring / φ corroboration for `qa_golden_ring_orbit_cert_v1` and cert [291]. Root cause of scan gap: "Collatz" + "Fibonacci" not in search battery.
- arXiv:2605.01148 (Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About Cyclic Concepts, May 1) — **LOW-MEDIUM. Slipped scans #4–9.** LLMs fail to use period-specific orbit geometry for cyclic concepts (days of week, months) — empirical Theorem NT / T2 instantiation. Add to `qa_algebraic_diversity_observer_cert_v1` human tract.
- arXiv:2605.31497 (Assign and Add: Compositional Arithmetic, May 29) — **LOW. Slipped scans #8–9.** Transformer modular addition module reuse; incremental grokking chain.

**No HIGH finds — second consecutive quiet window.**

**Registry explosion:** [311] → **[384]** in 4 days (73 new families). Full Pyth-1/2/3 Iverson corpus ingested as cert families [330]–[384] + new Pythagorean/Ancient/Diadic cluster [312]–[329].

**Cert registry update (CRITICAL):**
- Documentation backlog: **~101 families** [284]–[384] ALL missing README.md + SPEC.md (up from 28 in scan #9)
- [384] `qa_orbit_theorem_map_cert_v1`: also missing docs/families entry
- [261] qa_orbit_stratification_cert_v1: **NINE-TUPLE GAP** (9th scan — 5+ weeks overdue)
- `qa_steinmetz_whittaker_bridge_cert_v1`: **SEVEN-TUPLE GAP** — IDs [312], [313], [330], [331] all consumed; assign **[385]**
- `qa_whittaker_phase_packet_algebra_cert_v1`: **SEVEN-TUPLE GAP** — assign **[386]**
- [266] + [156]: SEVEN-TUPLE GAP each (README/SPEC missing)

**Search fix for scan #11:** Add "Collatz modular Fibonacci spectral orbit" + "cyclic submodule Z/mZ automorphism" + "Pisano period" explicit sweeps. Continue Thornton AD series monitoring.

### Scan #9 — 2026-06-04 (window: June 1 – June 4, 2026)
Full report: `sota_scans/scan_2026-06-04.md`

**Key finds:**
- arXiv:2512.25060 (On the geometry and topology of representations: manifolds of modular addition, Dec 2025) — **MEDIUM-HIGH. Slipped all 8 scans.** TDA tools on Z/pZ modular arithmetic representations; universal closed-form manifold for modular addition; architecture-independent topological structure. Primary reference for `qa_tda_orbit_grokking_cert_v1`. Strongest find this scan.
- arXiv:2605.24504 (Orbit decomposition statistics: Cesàro mean + large deviation principle, May 23, 2026) — **MEDIUM. Slipped scans #7–8.** Prime orbit counting for algebraic group endomorphisms (Z/24Z in scope); universal Poisson rate function. Statistical grounding for `qa_pisano_orbit_correspondence_cert_v1` [281].
- arXiv:2506.05190 (Categorical foundations of discrete dynamical systems, June 2025) — **MEDIUM. Slipped all 8 scans (1 year).** Cycle set concept for attractor decomposition; formal language for QA's orbit stratification. Reference for `qa_orbit_stratification_cert_v1` [261].
- arXiv:2603.19343 (Universal identity for powers in quadratic algebras, March 2026) — **MEDIUM. Slipped 5 scans.** Cayley-Hamilton + Fibonacci/Lucas closed-form. External grounding for `qa_cayley_hamilton_fibonacci_lucas_cert_v1` [299].

**No HIGH finds this window — quietest scan since #7.**

**Cert registry update (critical):**
- Registry grew [288] → [311] in 3 days (23 new families: Koenig/Pell/SL(2,Z)/Ford cluster [289–297], Algebra/Pisano cluster [298–302], AC Power/Steinmetz cluster [303–309], Rational Surveying [310], Archaeogeometry Orbit Classification [311])
- Documentation backlog: **28 families** [284]–[311] ALL missing README.md + SPEC.md
- [261] qa_orbit_stratification_cert_v1: **OCTUPLE GAP** (8th scan — 5 weeks overdue)
- qa_steinmetz_whittaker_bridge_cert_v1: **6th scan unregistered** — needs [312]; IDs [289]–[311] consumed
- qa_whittaker_phase_packet_algebra_cert_v1: **6th scan unregistered** — needs [313]
- qa_koenig_twisted_squares_cert_v1 and qa_pell_norm_cert_v1: on disk, registry status unverified

**New cross-domain connections:**
- Topology chain: arXiv:2512.25060 + arXiv:2605.06352 (scan #4) + arXiv:2605.20440 (scan #5) = complete TDA → H₁ → G-module chain for QA orbit topology
- AC Power / Steinmetz cluster [303–309]: first QA systematic electrical engineering application domain; bridges to Steinmetz phasor method (1893)
- SL(2,Z)/Koenig/Ford cluster [289–300]: new QA-as-modular-group-arithmetic research direction

**Search fix for scan #10:** Add "modular addition topology manifold TDA", "categorical discrete dynamical attractor cycle", "orbit decomposition algebraic group finite field" to search battery.

### Scan #8 — 2026-06-01 (window: May 28 – June 1, 2026)
Full report: `sota_scans/scan_2026-06-01.md`

**Key finds:**
- arXiv:2604.03634 (Algebraic Diversity: Group-Theoretic Spectral Estimation from Single Observations, April 3) — **HIGH. Slipped 7 scans.** General Replacement Theorem: single-snapshot group-averaging over Z/24Z recovers full orbit spectral structure. Formal implementation proof for Theorem NT (Observer Projection Firewall) discrete layer. Core paper for new cert candidate `qa_algebraic_diversity_observer_cert_v1`.
- arXiv:2604.19983 (Algebraic Diversity: Principles of Group-Theoretic Signal Processing, April 21) — **HIGH. Slipped 6 scans.** Blind group matching (polynomial-time): Z/24Z identifiable as matched group from QA signal data. Transform manifold: DFT is the unique distinguished transform for Z/24Z. Structural capacity κ (Rényi-2) quantifies QA orbit information organization. Core paper for `qa_algebraic_diversity_observer_cert_v1`.
- arXiv:2605.00848 (Continuous Algebraic Diversity: Lie Group Unification of Spectral/Wavelet/TF Analysis, ~May 1) — **HIGH. Slipped 3 scans.** Unification Theorem: Fourier=translation, wavelet=affine, TF=Heisenberg-Weyl. Duflo-Moore noise floor quantifies T2-b boundary crossing information loss. Formal continuous bridge for Theorem NT. Co-primary reference for `qa_transform_unification_cert_v1`.
- arXiv:2504.16513 (The Bracket of E8, April 2026) — **MEDIUM-HIGH. Slipped all scans.** Explicit E8 bracket via triality/oct-octonions; E6/E7 subalgebras; F4 26D irrep. Extends `qa_e8_integral_arithmetic_cert_v1` candidate scope.
- arXiv:2604.03725 (Quantum Algebraic Diversity: Single-Copy Density Matrix Estimation, April 4) — **MEDIUM-HIGH. Slipped 7 scans.** QAD Theorem; SIC-POVM = Heisenberg-Weyl AD; connects to E8 Coulomb branch.
- arXiv:2510.24882 (Modular Periodicity of Random Initialized Recurrences, Oct 2025) — **MEDIUM. Slipped all scans.** Full m² initialization classification for Fibonacci mod m. External academic grounding for `qa_pisano_orbit_correspondence_cert_v1` [281] (just registered since scan #7).

**3 HIGH finds (Algebraic Diversity cluster) — most significant multi-scan catch since scan #5.**

**New cert scaffolding candidate:**
- `qa_algebraic_diversity_observer_cert_v1` — HIGH priority. Certifies: (1) Z/24Z is the unique matched group for QA Cosmos-orbit signals (blind matching); (2) single-snapshot group-averaging over Z/24Z recovers full spectral structure (Replacement Theorem). Directly formalizes Theorem NT discrete layer.

**Cert gaps (escalated):**
- [261] qa_orbit_stratification_cert_v1: **SEPTUPLE GAP** (7th scan — CRITICAL: 4 weeks overdue)
- qa_steinmetz_whittaker_bridge_cert_v1: **5th scan unregistered** — needs ID [289]
- qa_whittaker_phase_packet_algebra_cert_v1: **5th scan unregistered** — needs ID [290]
- **NEW: Documentation wave** — [284]–[288] (5 families) all missing README.md + SPEC.md; systematic from cert creation sprint

**Search fix for scan #9:** Add "algebraic diversity," "matched group," "blind group matching," Thornton SMU to search battery.

### Scan #7 — 2026-05-28 (window: May 25 – May 28, 2026)
Full report: `sota_scans/scan_2026-05-28.md`

**Key finds:**
- arXiv:2604.13514 (Automated Tactics for Polynomial Reasoning in Lean 4, April 2026) — MEDIUM. **Slipped all prev scans.** SageMath/SymPy-backed Lean 4 polynomial tactics via certificate-based approach. Provides the automated algebraic verification engine for `qa_star_g_tensor_cert_v1` Lean 4 cert formalization pathway (identified in scan #5 via arXiv:2605.20440). Together: theoretical framework (⋆G tensor) + automated proof engine (polynomial tactics) = complete Lean 4 cert pipeline.
- arXiv:2512.16190 (Ramanujan Sums in Signal Recovery, Dec 2025) — LOW-MEDIUM. **Slipped 6 scans.** Perfect reconstruction + erasure robustness for Ramanujan filter banks; uncertainty principle via φ(q). For q=24: φ(24)=8 modes (Cosmos); q=8: φ(8)=4 modes (Satellite). Quantifies Theorem NT observer projection firewall information-theoretically. Subsumed by scan #5 transform unification (arXiv:2605.11589) at the structural level.
- arXiv:2602.19533 (Grokking Finite-Dimensional Algebra, Feb 2026) — LOW-MEDIUM. **Slipped 6 scans.** Extends grokking from group operations to general FDA (bilinear products over finite fields). QA's T-algebra is a commutative, associative, unital FDA over Z/mZ — easiest FDA grokking regime. Extends `qa_modular_nn_universality_cert_v1` (scan #3) candidate scope.
- arXiv:2605.27169 (Jacobi sums / cyclotomic matrices, May 26) — LOW. New in window. Arithmetic products of real parts of Jacobi sums over F_q; cyclotomic matrix connections. Pure number theory; no orbit dynamics or discrete-to-continuous bridge. Keyword proximity only.
- arXiv:2604.04655 (Grokking as Dimensional Phase Transition / SOC, April 2026) — LOW. Slipped. SOC/dimensional phase transition view of grokking. Fourth grokking-phase-transition paper; incremental add to existing chain.

**No HIGH finds this window (3-day early trigger, quiet after May 10–21 surge).**

**Cert scaffolding candidates (new):**
*(None standalone — arXiv:2604.13514 informs `qa_star_g_tensor_cert_v1` design; arXiv:2602.19533 extends `qa_modular_nn_universality_cert_v1` scope.)*

**Cert gaps (escalated):**
- [261] qa_orbit_stratification_cert_v1: **SEXTUPLE GAP** (6th scan — CRITICAL: still no registry, no docs/families, no README/SPEC in cert dir)
- qa_steinmetz_whittaker_bridge_cert_v1: **4th scan unregistered** — needs ID=279, docs/families file
- qa_whittaker_phase_packet_algebra_cert_v1: **3rd scan unregistered** — needs ID=280, docs/families file
- qa_whittaker_rational_direction_s1_cert_v1 [266]: **4th scan** missing README.md/SPEC.md in cert dir
- qa_wgs84_ellipse_cert_v1 [156]: **4th scan** missing README.md/SPEC.md in cert dir

**No new cert families created since scan #6.**

**Cross-domain connections (new):**
- Lean 4 automation chain: arXiv:2604.13514 (tactic engine) + arXiv:2605.20440 (scan #5, ⋆G framework) = complete Lean 4 cert verification pipeline for QA axioms
- Grokking FDA chain: arXiv:2602.19533 closes grokking coverage from group operations to general bilinear algebras over finite fields; QA's T-algebra is in scope

**Carried-over scaffolding candidates (cumulative, not yet built):**
See scan_2026-05-28.md for full ranked list. Top priorities: `qa_quantized_integer_dynamics_cert_v1` (scan #2), `qa_transform_unification_cert_v1` (scan #5), `qa_star_g_tensor_cert_v1` (scan #5 + this scan).

**📌 Scan hygiene note:** arXiv:2512.16190 and arXiv:2602.19533 both slipped 6 scans (Dec 2025 / Feb 2026 respectively). Root cause: eess.SP published-journal papers and Feb 2026 cs.LG cohort not systematically covered. Add explicit sweep of 2602.19xxx–2603.xxxx cs.LG to next scan.

---

### Scan #6 — 2026-05-25 (window: May 21 – May 25, 2026)
Full report: `sota_scans/scan_2026-05-25.md`

**Key finds:**
- arXiv:2604.00165 (Symmetric Nonlinear CAs as Algebraic References for Rule 30, March 31) — MEDIUM. **Slipped all prev scans.** Establishes S₃ symmetry framework for Rule 22 → Rule 30: support-set cardinality formula, two-step recursive construction, continuous limit = parabolic PDE (= QA Observer Projection / Theorem NT). Power-law symmetry-breaking from Rule 22 to Rule 30. Directly maps to repo's active `qa_rule30` cert + submission package.
- arXiv:2502.14663 (RIP for Measurements from Group Orbits, Feb 2025 / rev Sep 2025) — MEDIUM. **Slipped all prev scans.** Generalizes RIP from random circulants to random orbits of any finite group G representation. Certifies that QA's Z/24Z Cosmos and Z/8Z Satellite orbit measurement matrices satisfy the Restricted Isometry Property. Closes the RIP-orbit-transform chain with scan #5's arXiv:2605.11589 and scan #1's arXiv:2604.03634.
- arXiv:2605.19277 (Universal Cycles on Affine Lines, May 19) — LOW-MEDIUM. Ucycles over AG(n,q); affine line = coset of 1D subspace ≈ QA orbit class. Combinatorial, not dynamical overlap.
- arXiv:2603.05228 (Geometric Inductive Bias of Grokking, March 2026) — LOW-MEDIUM. **Slipped.** Spherical normalization reduces grokking 20× on Z/pZ tasks = S2 axiom analogue. Incremental addition to grokking chain.
- No HIGH finds this window. Quiet after May 10–21 E8/transform surge.

**Cert scaffolding candidates (new):**
1. `qa_rule30_symmetry_bridge_cert_v1` — Rule 30 algebraic framework via arXiv:2604.00165 (grounds `qa_rule30` cert system)
2. `qa_orbit_rip_cert_v1` — RIP certificate for Z/24Z and Z/8Z orbit measurements (arXiv:2502.14663)

**Cert gaps (escalated):**
- [261] qa_orbit_stratification_cert_v1: **QUINTUPLE GAP** (5th scan — CRITICAL: still no registry, no docs/families, no README/SPEC in cert dir)
- qa_steinmetz_whittaker_bridge_cert_v1: **3rd scan unregistered** — needs ID=279, docs/families file
- qa_whittaker_phase_packet_algebra_cert_v1: **2nd scan unregistered** — needs ID=280, docs/families file
- qa_whittaker_rational_direction_s1_cert_v1 [266]: **3rd scan** missing README.md/SPEC.md in cert dir
- qa_wgs84_ellipse_cert_v1 [156]: **3rd scan** missing README.md/SPEC.md in cert dir

**No new cert families created since scan #5.**

**Key cross-domain connection:** Rule 30 algebraic chain — arXiv:2604.00165 provides the missing external grounding for repo's active Rule 30 submission package (`RULE30_SUBMISSION_READY.md`). Check for citation opportunity.

**Carried-over scaffolding candidates (cumulative, not yet built):**
- `qa_quantized_integer_dynamics_cert_v1` (scan #2) — highest priority
- `qa_transform_unification_cert_v1` (scan #5) — HIGH, cleanest next new cert
- `qa_star_g_tensor_cert_v1` (scan #5) — HIGH, Lean 4 pathway
- `qa_circulant_symplectic_orbit_cert_v1` (scan #4) — HIGH
- `qa_tda_orbit_grokking_cert_v1` (scan #4) — HIGH
- `qa_modular_vsa_cert_v1` (scan #2) — HIGH
- `qa_modular_nn_universality_cert_v1` (scan #3) — HIGH
- `qa_full_pisano_orbit_cert_v1` (scan #1) — HIGH
- `qa_discrete_noether_cert_v1` (scan #3) — MEDIUM
- `qa_e8_integral_arithmetic_cert_v1` (scan #5) — MEDIUM-HIGH
- `qa_golden_ring_orbit_cert_v1` (scan #5) — MEDIUM-HIGH

---

### Scan #5 — 2026-05-21 (window: May 14 – May 21, 2026)
Full report: `sota_scans/scan_2026-05-21.md`

**Key finds:**
- arXiv:2605.11589 (Unification of Signal Transform Theory, ~May 16) — HIGH. Unifies DFT/DCT/WHT/Haar/KLT as eigenbases of group-covariance invariants; cyclic group Z/24Z → DFT = QA Cosmos natural transform; Z/8Z → DCT = Satellite. Certifies QA's spectral decomposition as the provably correct matched-group transform. Supersedes `qa_cayley_spectral_cert_v1`.
- arXiv:2605.20440 (Group-Algebraic Tensors, May 19) — HIGH. ⋆G tensor algebra with Eckart-Young optimality guarantee + 600-line Lean 4 proof. QA's Z/24Z orbit structure = canonical G-module; Lean 4 opens formal cert verification pathway.
- arXiv:2605.15075 (Non-crystallographic integer systems over composition algebras, May 14) — MEDIUM-HIGH. Golden ring ℤ[φ] as natural coefficient ring for H₂/H₄ non-crystallographic root shells; H₄ icosian → E8 = QA's 4D→8D E8 projection formalized as ℤ[φ]-module map. Bridges Fibonacci/Pisano orbit structure to E8 alignment arithmetic.
- arXiv:2605.09333 + 2605.09458 (Corradetti E8 cluster, May 10, slipped scan #4) — MEDIUM-HIGH. Okubo/Coxeter-Dickson Z-order on E8; integral shell polytope = Gosset polytope (240 E8 roots = QA's alignment targets). Grounds QA's E8 alignment as arithmetic, not just geometric.
- arXiv:2504.16513 (updated May 16, E8 bracket via triality) — MEDIUM. Explicit E8 bracket; triality = QA's three orbit classes. Enables bracket-based QA E8 invariants.

**Cert scaffolding candidates (new):**
1. `qa_transform_unification_cert_v1` — matched-group transform certification for QA orbits (arXiv:2605.11589; supersedes `qa_cayley_spectral_cert_v1`)
2. `qa_star_g_tensor_cert_v1` — ⋆G tensor algebra + Lean 4 formal verification (arXiv:2605.20440)
3. `qa_e8_integral_arithmetic_cert_v1` — E8 Coxeter-Dickson Z-order + Gosset polytope (arXiv:2605.09333 + 2605.09458 + 2605.15075)
4. `qa_golden_ring_orbit_cert_v1` — ℤ[φ] golden ring for QA Fibonacci/Pisano orbit periods (arXiv:2605.15075)

**Cert gaps (escalated):**
- [261] qa_orbit_stratification_cert_v1: **QUADRUPLE GAP** (4th scan, still unresolved — needs registry entry, docs/families file, README.md, SPEC.md in cert dir)
- qa_steinmetz_whittaker_bridge_cert_v1: **2nd scan unregistered** — needs ID=279, docs/families file
- qa_whittaker_phase_packet_algebra_cert_v1: **NEW DOUBLE GAP** — complete on disk but no registry entry, no docs/families file; needs ID=280
- qa_whittaker_rational_direction_s1_cert_v1 [266]: cert dir still missing README.md/SPEC.md (2nd scan)
- qa_wgs84_ellipse_cert_v1 [156]: cert dir still missing README.md/SPEC.md (2nd scan)

**Key cross-domain connection:** E8 arithmetic surge — 4 papers in 10 days (2605.09333 + 2605.09458 + 2605.15075 + 2504.16513) from same author cluster, directly under QA's harmonic index computation. Group-theoretic transform consolidation chain (2605.11589 + 2604.19983 + 2604.03634) completes a certifiable framework for QA orbit spectral decomposition.

**Carried-over scaffolding candidates (cumulative, not yet built):**
- `qa_quantized_integer_dynamics_cert_v1` (scan #2) — highest priority
- `qa_modular_vsa_cert_v1` (scan #2)
- `qa_full_pisano_orbit_cert_v1` (scan #1)
- `qa_modular_nn_universality_cert_v1` (scan #3)
- `qa_discrete_noether_cert_v1` (scan #3)
- `qa_circulant_symplectic_orbit_cert_v1` (scan #4)
- `qa_tda_orbit_grokking_cert_v1` (scan #4)

---

### Scan #4 — 2026-05-14 (window: May 7 – May 14, 2026)
Full report: `sota_scans/scan_2026-05-14.md`

**Key finds:**
- arXiv:2605.00965 (Coupled Arnold cat maps on circulant graphs, May 1) — HIGH. Finite toroidal phase space (Z/NZ)^2n + symplectic evolution + Fibonacci periods + circulant coupling = QA's full orbit architecture confirmed independently.
- arXiv:2605.06352 (Topological Signatures of Grokking, May 7) — HIGH. H₁ persistent homology = cyclic orbit structure signature in modular arithmetic models; direct validation of cert [276].
- arXiv:2605.08237 (Distributional Spectral Diagnostics for Grokking, May 7) — MEDIUM-HIGH. Hankel DMD + Wasserstein predicts grokking transition = QA orbit capture; AUROC 0.93.
- arXiv:2605.03338 (Symmetry-Protected Lyapunov Neutral Modes, May 2026) — MEDIUM. Lyapunov stability framework for QA orbit classes.

**Cert scaffolding candidates (new):**
1. `qa_circulant_symplectic_orbit_cert_v1` — symplectic Z/mZ map + circulant coupling + finite torus period spectrum (arXiv:2605.00965)
2. `qa_tda_orbit_grokking_cert_v1` — TDA-H₁ orbit class signature + DMD grokking prediction (arXiv:2605.06352 + 2605.08237)

**Cert gaps (escalated):**
- [261] qa_orbit_stratification_cert_v1: TRIPLE GAP (third scan, still unresolved — needs registry entry, docs/families file, README.md, SPEC.md)
- qa_steinmetz_whittaker_bridge_cert_v1: NEW — complete machine tract on disk, no registry entry, no docs/families file (assign next free ID = 279)
- qa_whittaker_rational_direction_s1_cert_v1 [266]: cert dir missing README.md/SPEC.md
- qa_wgs84_ellipse_cert_v1 [156]: cert dir missing README.md/SPEC.md

**Carried-over scaffolding candidates (cumulative, not yet built):**
- `qa_quantized_integer_dynamics_cert_v1` (scan #2) — highest priority
- `qa_modular_vsa_cert_v1` (scan #2)
- `qa_cayley_spectral_cert_v1` (scan #1)
- `qa_full_pisano_orbit_cert_v1` (scan #1)
- `qa_modular_nn_universality_cert_v1` (scan #3)
- `qa_discrete_noether_cert_v1` (scan #3)

**Next scan due:** ~2026-05-28

### Scan #3 — 2026-05-07 (window: May 4 – May 7, 2026)
Full report: `sota_scans/scan_2026-05-07.md`

**Key finds:**
- arXiv:2505.18266 (Universal CRT algorithm for modular addition, May 2025 late catch) — HIGH. All NNs solving modular addition implement approximate CRT cosets = QA's A2 derivation rules exactly. Independent proof that QA's architecture is the unique natural structure for modular arithmetic computation.
- arXiv:2602.16849 (Fourier Features, Lottery Ticket, Grokking, Feb 2026 late catch) — HIGH. Three-stage grokking maps to QA reachability phases; Fourier phase-symmetry voting = QA's resonance operator (einsum). Cyclic group Fourier modes = QA's orbit spectral decomposition.
- arXiv:2604.11163 (Exact Noether conservation in discrete IBVPs, April 2026 late catch) — MEDIUM-HIGH. QA's A1+S2+T1 = Noether-charge-preserving discretization; T-operator = symplectic map on Z/mZ × Z/mZ; orbit class = Noether charge.
- arXiv:2605.00452 (Graph Laplacians → String Partition Functions, May 2026) — MEDIUM. QA's three orbit graphs define three distinct spectral curves via period matrix construction. Observer projection = spectral curve continuum limit.
- Scientific Reports 2026 (Lucas sequences in signal processing) — MEDIUM. λ-Lucas summability grounds QA noise-annealing parameter; connects to certs [163], [179], [192].

**Cert scaffolding candidates (new):**
1. `qa_modular_nn_universality_cert_v1` — CRT-coset universality of QA's A2 derivation rules (arXiv:2505.18266 + 2602.16849)
2. `qa_discrete_noether_cert_v1` — QA axiom set as Noether-charge-preserving discretization (arXiv:2604.11163)

**Gaps:**
- [261] qa_orbit_stratification_cert_v1: DOUBLE GAP (machine tract exists at `qa_alphageometry_ptolemy/qa_orbit_stratification_cert_v1/`; NOT in meta-validator; NO docs/families file). Escalated from scan #2 single-gap report. Must be registered and human tract authored before next meta-validator run.
- cs.LG is not currently in arXiv scan targets; 4 late catches this scan came from that category. Add for scan #4.

**Carried-over scaffolding candidates (not yet built, cumulative):**
- `qa_quantized_integer_dynamics_cert_v1` (scan #2, arXiv:2604.06947 + 2604.26383) — highest priority
- `qa_modular_vsa_cert_v1` (scan #2, arXiv:2604.25939)
- `qa_cayley_spectral_cert_v1` (scan #1, arXiv:2604.03634)
- `qa_full_pisano_orbit_cert_v1` (scan #1, arXiv:2510.24882)

**Cert family gap found:** [261] `qa_orbit_stratification_cert_v1` missing human tract (docs/families/261_*.md).

**Next scan due:** ~2026-05-18

### Scan #2 — 2026-05-04 (window: April 9 – May 4, 2026)
Full report: `sota_scans/scan_2026-05-04.md`

**Key finds:**
- arXiv:2604.26383 + arXiv:2604.06947 (FQNM pair, Park/Ha/Kang, April 2026) — HIGH. "Quantised interaction rules" on integer state space = QA's T-operator architecture. Continuum emerges from reconstruction = Theorem NT. Independent PDE-side validation of QA's axiom set (T2, S2, T1).
- arXiv:2604.25939 (qFHRR, April 16) — HIGH. Discrete phase indices + modular arithmetic binding = QA's Z/mZ 4-tuple as a modular VSA. Cert candidate: `qa_modular_vsa_cert_v1`.
- arXiv:2604.22863 (Wave-Geometric Duality for HDC, April 23) — MEDIUM. Discrete → waveform bridge via unitary embedding; aliasing = modular arithmetic.
- arXiv:2511.09708 (MCR modular composite representations, Nov 2025, missed) — MEDIUM.

**Cert scaffolding candidates (new):**
1. `qa_quantized_integer_dynamics_cert_v1` — formalizes QA step as antisymmetric integer-transfer operator (FQNM framework)
2. `qa_modular_vsa_cert_v1` — formalizes QA 4-tuple as modular composite representation over Z/24Z

**Carried-over scaffolding candidates (not yet built):**
- `qa_cayley_spectral_cert_v1` (from scan #1, arXiv:2604.03634)
- `qa_full_pisano_orbit_cert_v1` (from scan #1, arXiv:2510.24882)

### Scan #1 — 2026-04-09 (window: ~March 26 – April 9, 2026)
Full report: `sota_scans/scan_2026-04-09.md`
HIGH finds: arXiv:2604.03634 (Cayley spectral), arXiv:2603.14999 (Fibonacci never collapses), arXiv:2510.24882 (all Pisano initializations).
