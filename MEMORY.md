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
