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

**Cert family gap found:** [261] `qa_orbit_stratification_cert_v1` missing human tract (docs/families/261_*.md).

**Next scan due:** ~2026-05-18

### Scan #1 — 2026-04-09 (window: ~March 26 – April 9, 2026)
Full report: `sota_scans/scan_2026-04-09.md`
HIGH finds: arXiv:2604.03634 (Cayley spectral), arXiv:2603.14999 (Fibonacci never collapses), arXiv:2510.24882 (all Pisano initializations).
