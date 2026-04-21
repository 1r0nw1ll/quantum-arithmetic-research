<!-- PRIMARY-SOURCE-EXEMPT: reason='Phase 4.8 item 6 acquisition requirements spec; primary sources are what this doc targets for future acquisition, not yet on disk' -->

# QA-MEM Phase 4.8 Item 6 — McCraty / Radin Corpus Acquisition Requirements

**Status:** scope spec — NOT yet executed. Written 2026-04-20 after cert [259] `qa_heartmath_coherence_cert_v1` landed (commit `c387f84`).
**Authority:** `docs/specs/QA_MEM_PHASE_4_8_HANDOFF.md` §2 item 6; OB 2026-03-25 research-agenda Thread 3.
**Goal:** acquire the primary-source numerical data required to build `qa_heartmath_mapping_cert_v1` — the follow-up to [259] that makes numerical claims about HRV coherence, heart-brain cross-coherence, and Schumann harmonic ratios instead of only structural claims.

## Why this doc exists

[259] certifies the observer-projection firewall discipline for HeartMath data but deliberately does not claim any numerical equivalence. The 2026-03-25 OB agenda lists three candidate numerical mappings:

1. HRV coherence ratio ↔ QA Harmonic Index
2. Brain–heart cross-coherence ↔ QA Markovian coupling matrix
3. Schumann resonance (7.83 Hz fundamental, 14.3 / 20.8 / 27.3 / 33.8 Hz harmonics) ↔ mod-9 / mod-24 orbit-harmonic integer ratios

Each of these requires specific primary-source *numerical data* to ground a cert claim. The 4 PDFs acquired in the Phase 4.8 kickoff (Tomasino/Danielson/Oschman/Edwards) are short editorial or review pieces — they do not carry the numerical time-series, correlation tables, or spectral ratio data needed.

Until McCraty's peer-reviewed HeartMath Institute papers and Radin's IONS publications land on disk, any numerical mapping cert would be inventing numbers from secondary/marketing sources, which violates `memory/feedback_map_best_to_qa.md` (find what works → extract generator → certify the mapping) and `memory/feedback_primary_sources_vs_consensus.md`.

This spec pre-commits the scope of what to acquire, so the eventual acquisition session doesn't drift into "grab everything HeartMath-adjacent."

## Acquisition targets

### A. McCraty — HeartMath Institute research library

**Landing page:** `https://www.heartmath.org/research/research-library/`
**Primary author of interest:** Rollin McCraty (Director of Research, HeartMath Institute)

Priority papers (peer-reviewed, numerical-data-bearing):

| # | Paper | Where it goes |
|---|---|---|
| 1 | McCraty, Atkinson, Tomasino, Bradley (2009). *The Coherent Heart: Heart–Brain Interaction, Psychophysiological Coherence and the Emergence of a System Wide Order*. Integral Review, 2, 10–115. | Already cited in Edwards 2018 references. Long-form; carries HRV time-series figures and coherence ratio definitions. **Highest priority.** |
| 2 | McCraty (2017). *New frontiers in heart rate variability and social coherence research*. Frontiers in Public Health 5:267. DOI: 10.3389/fpubh.2017.00267. | Peer-reviewed, open-access via Frontiers. Group-synchronization experimental data. |
| 3 | McCraty & Zayas (2014). *Cardiac coherence, self-regulation, autonomic stability, and psychosocial well-being*. Frontiers in Psychology 5:1090. DOI: 10.3389/fpsyg.2014.01090. | Peer-reviewed, open-access. Canonical HRV coherence ratio definitions. |
| 4 | Alabdulgader, McCraty, Atkinson, Dobyns, Vainoras, Ragulskis, Stolc (2018). *Long-term study of heart rate variability responses to changes in the solar and geomagnetic environment*. Scientific Reports 8(1):2663. DOI: 10.1038/s41598-018-20932-x. | Peer-reviewed Nature group. Geomagnetic ↔ HRV coupling. Relevant to Schumann-harmonic mapping track. |
| 5 | McCraty (2016). *Science of the Heart Volume 2. Exploring the Role of the Heart in Human Performance*. HeartMath Institute. DOI: 10.13140/RG.2.1.3873.5128. | Institutional monograph, book-length. Canonical reference for all HeartMath metric definitions. Treat as a reference manual, not a claim source. |

### B. Radin — IONS presentiment / RDNG / global consciousness

**Landing pages:** `noetic.org/publications/` (IONS publications portal) + `deanradin.org/publications/` (personal page) + arXiv where duplicated.
**Primary author of interest:** Dean Radin (Chief Scientist, Institute of Noetic Sciences)

Priority papers:

| # | Paper | Where it goes |
|---|---|---|
| 6 | Radin, D. (2004). *Event-related electroencephalographic correlations between isolated human subjects*. Journal of Alternative and Complementary Medicine 10(2). | Pair-EEG coupling under isolation. Directly probes brain–brain cross-coherence analogue at distance. |
| 7 | Mossbridge, Tressoldi, Utts (2012). *Predictive physiological anticipation preceding seemingly unpredictable stimuli: a meta-analysis*. Frontiers in Psychology 3:390. DOI: 10.3389/fpsyg.2012.00390. | Meta-analysis of 26 presentiment studies. Statistical-structure data; effect-size tables. |
| 8 | Nelson, R. (2015). *The Global Consciousness Project: subtle interconnections and correlations with global events*. EXPLORE 11(2). | GCP 20-year RNG dataset correlations. Relevant to population-scale coherence track. |
| 9 | Bem, D. J. (2011). *Feeling the future*. Journal of Personality and Social Psychology 100(3). DOI: 10.1037/a0021524. | Original 9-experiment presentiment paper. Included for completeness; contested methodology documented in own literature. |

Acquisition posture on Radin: the set is broader than McCraty and skews more contested. Prioritize 6 + 7 + 8 as the peer-reviewed numerical-data-bearing subset. Bem 9 is a reference-completeness capture, not a primary claim source.

### C. Schumann resonance specific

**Primary source:** Schumann 1952 original German paper + Williams 1992 *Science* review.

| # | Paper | Where it goes |
|---|---|---|
| 10 | Schumann, W. O. (1952). *Über die strahlungslosen Eigenschwingungen einer leitenden Kugel...*. Zeitschrift für Naturforschung 7a:149–154. | Original derivation. Gives the fundamental 7.83 Hz and the 14.1/20.3/26.4/32.4/38.4 harmonic series (note: Schumann's own values, not the slightly-different NOAA-observed 14.3/20.8/27.3/33.8). |
| 11 | Williams, E. R. (1992). *The Schumann resonance: A global tropical thermometer*. Science 256(5060):1184–1187. DOI: 10.1126/science.256.5060.1184. | Modern review with measured harmonic values. |

The integer-ratio claim (mod-9/24 orbit-harmonic mapping) stands or falls on which observed harmonic series is taken as canonical — Schumann's theoretical values or the NOAA-observed. Cert authoring will need to declare which.

## Per-target acquisition criteria

For each paper on the list:

1. **Retrievable PDF on disk** under `Documents/heartmath_corpus/` (McCraty/Schumann) or a new `Documents/radin_ions/` directory (Radin). Apply the same Documents/ ingress-allowlist discipline as Wildberger/Haramein carve-outs.
2. **Excerpts MD snapshot** with page-numbered verbatim quotes at `docs/theory/<topic>_excerpts.md`. Per target paper, at minimum:
    - 1 quote defining the numerical metric (coherence ratio definition, cross-coherence formula, harmonic ratio numerical value)
    - 1 quote stating the empirical result (correlation coefficient, sample size, p-value, or measured harmonic frequency)
    - 1 quote stating the methodology (sampling rate for HRV papers, measurement apparatus for Schumann, statistical test for Radin)
3. **Claim fixture** in `tools/qa_kg/fixtures/source_claims_<topic>.json` with the verbatim quotes as SourceClaim nodes, domain = `psychophysiology` (HeartMath/Radin) or `physics` (Schumann).
4. **CORPUS_INDEX.md row** per paper with topic tags + status.

## What the eventual cert needs from the data

To support `qa_heartmath_mapping_cert_v1` (numerical — the follow-up to [259] structural), the acquired data must carry:

### For HI ↔ HRV coherence ratio mapping

- A **formal definition** of the HRV coherence ratio as used in the HeartMath peer-reviewed literature (not the marketing copy). Target: McCraty+Zayas 2014 §Methods, or McCraty 2017 §Methods.
- A **numerical table or figure** showing coherence-ratio values across at least two of the three qualitatively distinct rhythm states ([259]'s Singularity/Satellite/Cosmos analogues).
- A **measurement protocol** (sampling rate, window size, spectral estimator) that can be recomputed from the paper — a validator requires recomputability.

Without all three, the cert cannot ground a formal HI ↔ coherence-ratio claim; it can at best assert qualitative correspondence, which [259] already does.

### For brain–heart cross-coherence ↔ Markovian coupling

- At least one paper with a **cross-spectral coherence matrix** between two signals (ECG + EEG, ECG + ECG under social synchrony, etc.).
- An explicit **coupling coefficient** (integer or rational form preferred; a real-valued coherence estimate is second-best).
- Event markers / timestamps so a discrete-time structural analogue can be defined.

McCraty 2017 (social coherence, Frontiers Public Health) is the primary target.

### For Schumann ↔ mod-9 / mod-24 orbit-harmonic ratios

- Williams 1992 *Science* harmonic values.
- Compute `14.3 / 7.83`, `20.8 / 7.83`, etc., check whether each ratio's continued-fraction or rational approximation lands at a mod-9 / mod-24 canonical ratio.
- The cert claim would be: the first N Schumann harmonics sit on a specific QA diagonal (TBD which — probably `fixed-d` analogous to [218] Haramein) with bounded relative error.

This is the cleanest of the three numerical mappings because the data is already numerical and already public-domain. Could be the **first** numerical HeartMath-family cert if McCraty data acquisition runs into paywalls.

## Acquisition sequence (recommended)

1. **Schumann first (C)** — simplest, most public, unblocks a numerical cert independent of HeartMath paywalls. Williams 1992 is AAAS Science — likely accessible via institutional access or Sci-Hub-equivalent legal routes. Schumann 1952 is frequently reproduced in Schumann-resonance review papers; acquisition via Zeitschrift für Naturforschung archives.
2. **McCraty open-access (#2, #3, #4)** — Frontiers and Nature/Scientific Reports are CC-BY open-access. `curl` should work against direct DOI resolvers. No Cloudflare issues expected. Alabdulgader+McCraty 2018 (Scientific Reports) is the highest-numerical-density target.
3. **McCraty institutional monograph (#5)** — HeartMath Institute ResearchGate link or direct PDF.
4. **McCraty long-form (#1)** — *Integral Review* is open-access but small press; direct download.
5. **Radin peer-reviewed (#6, #7, #8)** — Frontiers in Psychology CC-BY, EXPLORE via Elsevier (likely paywalled), JACM (Journal of Alternative and Complementary Medicine) mixed-access. Retrieve what is freely available; defer paywalled items with a CORPUS_INDEX row noting the blocker.
6. **Bem 2011 (#9)** — APA PsycNET paywall likely; defer with row.

Per the ingress-allowlist discipline: write PDFs via cwd-scoped `curl -o <basename>.pdf` from inside `Documents/<subdir>/` (no `Documents/` substring in the Bash command body), since `Documents/heartmath_corpus/` and `Documents/radin_ions/` are not yet in `DOCUMENTS_PDF_INGRESS_PREFIXES`. See `memory/feedback_cert_gate_bridge_health.md`. Extending the allowlist is a WRAPPER_SELF_MODIFICATION edit blocked on Codex bridge proper health (Phase 4.8 item 4).

## What this spec does NOT commit to

- **No cert authoring is scoped in this spec.** `qa_heartmath_mapping_cert_v1` structure comes *after* data lands, not before.
- **No ingest-pipeline automation is scoped.** Manual pypdf / fitz text extraction, manual verbatim quote selection, manual claim-fixture authoring — same pattern as [259] / Phase 4.6 Haramein/Wildberger. Automation is a future cost-amortization question, not a blocker.
- **No timeline.** Acquisition is the unblock; the mapping cert work follows naturally once ≥1 of the three numerical tracks has ≥1 paper with ≥1 verbatim-quotable numerical result on disk.

## Related specs

- `docs/specs/QA_MEM_PHASE_4_8_HANDOFF.md` — parent handoff doc, §2 item 6 is the root of this spec.
- `docs/specs/QA_MEM_SCOPE.md` — QA-MEM architecture scope; any new fixture / domain changes respect the schemas defined there.
- `docs/theory/QA_HEARTMATH_COHERENCE.md` — cert [259] design note, explicitly names the numerical mapping as deferred.
- `memory/feedback_map_best_to_qa.md` — the discipline this spec serves: find what works in the McCraty/Radin primary literature first, then map through QA, rather than inventing QA-flavored HRV metrics.
- `memory/feedback_cert_scaffold_gotchas.md` — scaffold-time gate pitfalls hit during [259] scaffolding; reuse the DOI-citation pattern in scope_note for any mapping_protocol_ref.json in the eventual cert.

## References

1. McCraty, R., et al. (2009). *The Coherent Heart*. Integral Review 2:10–115.
2. McCraty, R. (2017). *New frontiers in heart rate variability and social coherence research*. Frontiers in Public Health 5:267. DOI: 10.3389/fpubh.2017.00267.
3. McCraty, R., & Zayas, M. A. (2014). *Cardiac coherence, self-regulation, autonomic stability, and psychosocial well-being*. Frontiers in Psychology 5:1090. DOI: 10.3389/fpsyg.2014.01090.
4. Alabdulgader, A., et al. (2018). *Long-term study of heart rate variability responses to changes in the solar and geomagnetic environment*. Scientific Reports 8(1):2663. DOI: 10.1038/s41598-018-20932-x.
5. McCraty, R. (2016). *Science of the Heart Volume 2*. HeartMath Institute. DOI: 10.13140/RG.2.1.3873.5128.
6. Radin, D. (2004). *Event-related electroencephalographic correlations between isolated human subjects*. JACM 10(2).
7. Mossbridge, J., Tressoldi, P., Utts, J. (2012). *Predictive physiological anticipation preceding seemingly unpredictable stimuli*. Frontiers in Psychology 3:390. DOI: 10.3389/fpsyg.2012.00390.
8. Nelson, R. (2015). *The Global Consciousness Project*. EXPLORE 11(2).
9. Bem, D. J. (2011). *Feeling the future*. JPSP 100(3). DOI: 10.1037/a0021524.
10. Schumann, W. O. (1952). *Über die strahlungslosen Eigenschwingungen einer leitenden Kugel*. Zeitschrift für Naturforschung 7a:149–154.
11. Williams, E. R. (1992). *The Schumann resonance: A global tropical thermometer*. Science 256(5060):1184–1187. DOI: 10.1126/science.256.5060.1184.
