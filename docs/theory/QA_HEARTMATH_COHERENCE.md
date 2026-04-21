# QA HeartMath Coherence Firewall — Theory Note

**Status:** design note companion to cert family [259] `qa_heartmath_coherence_cert_v1`.
**Scope:** frames the Theorem NT (observer-projection firewall) constraint for HeartMath-sourced cardiac-rhythm data. Does not derive new mathematical content; the cert and validator are the load-bearing artifacts.

## Primary sources

- Oschman, J. L., & Oschman, N. H. (2015). The Heart as a Bi-Directional Scalar Field Antenna. *Journal of Vortex Science and Technology* 2:121. DOI: 10.4172/2090-8369.1000121.
- Danielson, K., Jeffers, K., Kaiser, L., McKinley, L., Kuhn, T., & Voorhies, G. (2014). Sustained Hospital-based Wellness Program. *Global Advances in Health and Medicine* 3(Suppl 1):BPA05. DOI: 10.7453/gahmj.2014.BPA05.
- Edwards, S. D. (2018). Call for papers: Special section on HeartMath for Psychology. *Journal of Psychology in Africa* 28(5):432–433. DOI: 10.1080/14330237.2018.1528007.
- Tomasino, D. (1997). New Technology Provides Scientific Evidence of Water's Capacity to Store and Amplify Weak Electromagnetic and Subtle Energy Fields. *Institute of HeartMath Publication No. 97-002*.

All four PDFs are on disk under `Documents/heartmath_corpus/`. Verbatim page-numbered excerpts are in `docs/theory/heartmath_phase4_8_excerpts.md`; 18 claim IDs are registered in `tools/qa_kg/fixtures/source_claims_heartmath.json`.

## The firewall claim

Cardiac-rhythm coherence as reported in the HeartMath literature — ECG frequency-spectrum coherence, HRV LF/HF ratio, brain–heart cross-coherence — is a **continuous observer-projection output**. The measurement is real. The measurement is not, however, the underlying dynamics.

Per Theorem NT (observer-projection firewall; see `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md` + `QA_AXIOMS_BLOCK.md`), continuous functions enter QA only through declared observer projections, and the observer/QA boundary is crossed at most twice per trace — input projection (continuous signal → discrete state) and output projection (discrete state → continuous reading). No interior re-projection through a continuous intermediate is permitted. A float value drawn from the output side that re-enters QA dynamics as a causal input is a T2-b violation (see `memory/feedback_theorem_nt_observer_projection.md` and `memory/feedback_raw_vs_mod_derived.md`).

Cert [259] enforces this discipline on HeartMath-data-bearing fixtures: the QA state variable must remain an integer tuple (`b`, `e`, `d`, `a`), the orbit-class label must be discrete (`Singularity`, `Satellite`, `Cosmos`), and continuous coherence measurements live inside `observer_projections[].reading` blocks, never on the QA state path.

## The rhythm-class correspondence

The three orbit classes correspond to three qualitatively distinct cardiac-rhythm modes reported in the HeartMath primary-source literature:

- **Singularity** — resting or still rhythm, no cyclical entrainment, no broadcasting. Fixed point of the dynamics.
- **Satellite** — entrained 8-cycle rhythm, cross-coherent with a coupled system (heart–brain, heart–ECG-frequency-spectrum alignment). This is the class anchored by Oschman 2015's reporting of the Rein/McCraty result: "Rein's resonance hypothesis was confirmed in two studies at the Institute of HeartMath in collaboration with Rollin McCraty. These studies demonstrated (a) coherence in the electrocardiographic (ECG) frequency spectra of individuals whose attention was focused in the heart area while generating positive emotions such as love, care or appreciation and (b) a correlation between the ECG coherent patterns and those also occurring in the brain and other parts of the body." (Oschman 2015, p.2)
- **Cosmos** — 24-cycle broadcasting rhythm, master-oscillator regime with coherent frequency radiation system-wide. Anchored by Oschman 2015's master-oscillator quote (p.2): "the heart acts as a master electrical oscillator capable of radiating coherent frequencies which promote the health and vitality of the entire human system."

This is a classification correspondence, not a numerical equivalence. The cert certifies that any HeartMath-derived rhythm-state trace *can* be expressed cleanly in these three labels and that continuous measurements stay on the observer side of the firewall. The cert does not certify that any particular coherence-ratio value numerically equals a particular HI value — that would require the numerical data McCraty's peer-reviewed Global Coherence Initiative papers carry, which is Phase 4.8 item 6 corpus expansion and not yet on disk.

## The bidirectional-antenna / Theorem NT symmetry

Oschman 2015 appeals to antenna theory in a structural footnote (p.3, fn3): "From basic antenna theory we know that an antenna of a particular geometry will function for both transmission and reception of electromagnetic waves and, presumably, of potential waves."

This reciprocity — same structure for input and output — is exactly the Theorem NT requirement that the two boundary crossings (input projection and output projection) have identical structural form. The cert enforces this via `HMC_BOUNDARY_CROSSINGS`: each rhythm state has at most two projections, one `input` and one `output`, and if both are present they are distinct directions on the same projection structure.

Primary sources citing this directly in the cert witnesses: `oschman_2015_bidirectional_antenna`.

## What this cert does not claim

1. **No HI ↔ coherence-ratio numerical mapping.** The 2026-03-25 OB research-agenda entry flagged this as a target mapping. The 4 Scholar-page results acquired in the Phase 4.8 kickoff do not carry the numerical primary-source data needed to ground the mapping. McCraty's peer-reviewed GCI papers and Radin's IONS presentiment/RDNG publications are the next corpus expansion (Phase 4.8 item 6).
2. **No claim that HeartMath measurement methodology is "correct" or "incorrect."** The cert enforces structural form of the mapping, not truth of the underlying empirical results.
3. **No Schumann-resonance ↔ mod-9/24 orbit-harmonics claim.** Also flagged in the 2026-03-25 agenda, also deferred — specific integer frequency-ratio claims require primary-source numerical data beyond the 4 papers in scope here.
4. **No HRV-measurement replacement.** QA does not do HRV measurement. The cert describes what the coherence phenomenon structurally reduces to when the observer projection is declared properly; it does not propose an alternative measurement apparatus.

## Relation to existing certs

- Cert [257] `qa_integer_state_pipeline_cert_v1` certifies the two-boundary-crossing invariant for GLM-5's token pipeline (arXiv:2602.15763). Cert [259] is the same structural claim rebased on cardiac-rhythm data instead of LLM tokens.
- Cert [122] `qa_empirical_observation_cert` is the generic bridge from experimental observations to the cert ecosystem. Cert [259] is a domain-specialized cert with hardcoded primary-source witness IDs for the HeartMath corpus.
- Cert [218] `qa_haramein_scaling_diagonal_cert_v1` is the structural template: primary-source PDF + excerpts MD + source_claims fixture + validator + 1 PASS + 1 FAIL.
- Cert [197] `qa_see_longitudinal_transverse_cert_v1` covers generator/observer mode duality for See 1917. Oschman 2015's scalar-antenna framing bridges to See 1917; a future cert could formalize that bridge — not this cert.

## Phase 4.8 item 5 grounding

`docs/specs/QA_MEM_PHASE_4_8_HANDOFF.md` §2 item 5 defines this cert's scope:

> **Cert family candidate** `qa_heartmath_coherence_cert_v1` — grounds in Danielson 2014 + extracted HRV-coherence quotes + the OB 2026-03-25 research-agenda Thread 3 mapping (HI ↔ cardiac coherence ratio; brain–heart cross-coherence ↔ QA Markovian coupling; Schumann ↔ mod-9/24 orbit harmonics).

The cert covers the firewall discipline for all three mapping goals without claiming the numerical equivalences of any of them. Those await Phase 4.8 item 6 primary-source acquisition.

## Source

Will Dale + Claude 2026-04-20.
