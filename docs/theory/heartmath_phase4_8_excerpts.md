<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.8 HeartMath corpus kickoff. Placeholder for verbatim excerpt anchors. Full quote extraction + domain assignment deferred to Phase 4.8 body. Retrieval notes + source URLs are authoritative; per-PDF anchor sections are stubs awaiting hand-curated quotes. Do not treat stub quotes as primary record. -->

# HeartMath Corpus — Phase 4.8 Primary-Source Excerpts (KICKOFF — STUB)

**Corpus root:** `Documents/heartmath_corpus/`
**Scope:** QA-MEM Phase 4.8 kickoff — 4 SourceWork nodes registered in `tools/qa_kg/fixtures/source_claims_heartmath.json`. Claims extraction (verbatim anchors) pending.
**Domain:** `""` (unclassified) on all Phase-4.8-kickoff nodes. Candidate extension: `psychophysiology` (HRV, cardiac coherence, brain–heart coupling) — decision deferred until `tools/qa_kg/domain_taxonomy.json` is updated and [254] R10 re-validated.
**Origin search:** Google Scholar `scholar.google.com/scholar?start=130&q=heartmath&hl=en&as_sdt=4007` (results 131–140 of q=heartmath). 4 of 10 results retrievable (PDF on disk); 6 are citation-only / paywalled / foreign-journal (see `tools/qa_kg/CORPUS_INDEX.md` HeartMath section for the full roster).

**QA-research grounding:** this corpus was prioritized because an OB research-agenda entry dated 2026-03-25 explicitly flagged HeartMath/McCraty/Radin as one of three unfilled mapping threads (alongside Langlands and modern signal analysis). Full mapping goals: HRV coherence ratio ↔ QA harmonic index HI; brain–heart cross-spectral coherence ↔ QA Markovian coupling matrix; Schumann resonance (7.83 Hz, 14.3 Hz, …) ↔ mod-9 / mod-24 orbit harmonics.

**Firewall note:** HeartMath-adjacent literature blends observer-projection (continuous HRV power-spectral density, LF/HF ratios, biofeedback game outputs) with primary structural claims (cardiac rhythm as ordered vs disordered, entrainment as cross-coherence). Per the project's Theorem NT observer-projection rule, QA-mapping entries MUST enter through discrete orbit/rhythm classification, not through direct float-valued coherence-ratio inputs. The Phase 4.5 feedback-rule `feedback_primary_sources_vs_consensus.md` applies: read the primary HeartMath / Oschman / Tomasino text before substituting mainstream-consensus priors on "fringe" flags.

---

## Tomasino (1997) — Water as EM/subtle-energy storage

**Publication:** Institute for HeartMath Publication No. 97, 1997
**Authors:** Dana Tomasino
**On-disk:** `Documents/heartmath_corpus/tomasino_1997_water_em_storage.pdf` (190KB, 4pp; retrieved directly from othernetworks.org)

### #tomasino-1997-abstract (STUB — pending hand-extraction)

> [PLACEHOLDER — QUOTE PENDING PHASE 4.8 BODY] "Research has shown that water is a liquid crystal with a pliable lattice matrix that is capable of adopting many structural forms. The structure of water gives it an infinite capacity to store …" (Scholar snippet only; verify against PDF before promoting to primary claim.)

---

## Danielson et al. (2014) — Sustained Hospital-based Wellness Program

**Publication:** Global Advances in Health and Medicine 3(Suppl 1):BPA05, 2014
**Authors:** Kimberly Danielson, Kay Jeffers, Linda Kaiser, et al.
**On-disk:** `Documents/heartmath_corpus/danielson_2014_hospital_wellness.pdf` (425KB; retrieved via PMC3923282 PDF fetch under browser credentials)

### #danielson-2014-abstract (STUB — pending hand-extraction)

> [PLACEHOLDER — QUOTE PENDING PHASE 4.8 BODY] Council of America extended HeartMath to every population health …; HeartMath mastery by more than one thousand employees allowed the organization to introduce HeartMath's [practices at scale]. (Scholar snippet only; verify against PDF before promoting.)

---

## Oschman & Oschman (2015) — Heart as Bi-Directional Scalar Field Antenna

**Publication:** Journal of Vortex Science and Technology 2(2), 2015
**Authors:** James L. Oschman, Nora H. Oschman
**On-disk:** `Documents/heartmath_corpus/oschman_2015_heart_bidirectional_scalar_antenna.pdf` (510KB; retrieved via web.archive.org snapshot 20210316115729 of sparkleoflife.com.au — original host suspended at ingest time)

### #oschman-2015-abstract (STUB — pending hand-extraction)

> [PLACEHOLDER — QUOTE PENDING PHASE 4.8 BODY] "Rein's resonance hypothesis was confirmed in two studies at the Institute of HeartMath … these results strengthened the HeartMath Institute's theory that the heart [as antenna]." (Scholar snippet only; verify against PDF before promoting.)

**Mapping note:** this paper links HeartMath's coherence research to Whittaker-style longitudinal / scalar decomposition literature — adjacent to the See 1917 longitudinal-transverse duality already certified in [197] (`qa_see_longitudinal_transverse_cert_v1`). A Phase 4.8 claim extraction should check for a structural map between Oschman's "bi-directional scalar antenna" and the generator/observer mode duality in [197].

---

## Edwards (2018) — Call for Papers: HeartMath for Psychology

**Publication:** Journal of Psychology in Africa 28(5), 2018 (Taylor & Francis)
**DOI:** 10.1080/14330237.2018.1528007
**Authors:** S. Edwards (Editorial)
**On-disk:** `Documents/heartmath_corpus/edwards_2018_cfp_heartmath_psychology.pdf` (499KB; retrieved via tandfonline.com browser-context fetch — curl returned Cloudflare challenge page)

### #edwards-2018-cfp (STUB — pending hand-extraction)

> [PLACEHOLDER — QUOTE PENDING PHASE 4.8 BODY] "…evaluation of psychological behaviour using HeartMath approaches, methods and techniques … from the HeartMath website at https://www.heartmath.org/research/research-library." (Scholar snippet only; verify against PDF before promoting.)

---

## Retrieval failures (NOT on disk — Scholar results 131-140)

The following entries from the same Scholar page did not yield a retrievable PDF during Phase 4.8 kickoff. Each row lists the hardest-blocker; a Phase 4.8 body pass may revisit via ILL, archive.org, or direct author contact.

| Scholar idx | Title | Authors / Year | Blocker |
|---|---|---|---|
| 1 | *EmWave Desktop©* | Institute of HeartMath, 2012 | [CITATION] — product citation, no paper exists |
| 2 | *Evaluation of HeartMath meditation explorations with a longer breath cycle* | Edwards, David, Hermann et al. 2023 (*Dialogo*) | EBSCO paywall; no open version located |
| 3 | *The Biologic Effects of the Interventions of Heart-Generated Coherence and Focused Intention on Distilled Water Using Plant Growth as an Objective …* | J. Simmons, 2010 | [CITATION] — thesis / unpublished; no PDF indexed |
| 4 | *The HeartMath Solution* | Childre, Martin, Beech, 1999 | Published book (Harper); no PDF (CiNii stub only) |
| 8 | *Effects of Psychoeducation using the HeartMath™ system on stress and wellness for Korean caregivers of children with disabilities* | 명화숙, 명한나, 제인 2014 (Korean journal of child psychology) | Foreign-journal paywall (kyobobook.co.kr); no open version |
| 9 | *The effectiveness of paced breathing versus game-biofeedback on heart rate variability* | Böckeler, Cornforth, Drummond et al. 2020 (IEEE EMBC) | IEEE Xplore paywall |

---

## Phase 4.8 follow-up actions (tracked)

1. **Excerpt extraction** — replace STUB sections above with verbatim p-numbered quotes pulled from the four on-disk PDFs (pypdf or `pdftotext` first-page metadata for locators).
2. **Claims array** — populate `tools/qa_kg/fixtures/source_claims_heartmath.json` `claims` array with 3–5 quotes per work, following the Phase 4.6 Haramein schema.
3. **Domain taxonomy extension** — add `psychophysiology` to `tools/qa_kg/domain_taxonomy.json`, re-validate [254] R10, retro-stamp every HeartMath SourceWork + SourceClaim from `""` to `psychophysiology`.
4. **Ingress-whitelist extension** — add `Documents/heartmath_corpus/` to `DOCUMENTS_PDF_INGRESS_PREFIXES` in `llm_qa_wrapper/cert_gate_hook.py` **and** the case statement in `.claude/hooks/pretool_guard.sh`. Requires routing through Codex quarantine review (WRAPPER_SELF_MODIFICATION) — deferred while Codex bridge is dead.
5. **Cert family candidate** — `qa_heartmath_coherence_cert_v1` grounding the HRV-coherence-ratio ↔ QA HI formal map identified in the 2026-03-25 OB research-agenda entry.
6. **Corpus expansion** — the Scholar page at `start=130` is ONE slice. A fuller HeartMath primary-source survey should pull McCraty's peer-reviewed HRV + Global Coherence Initiative papers (heartmath.org/research/research-library/) and Radin's IONS publications on presentiment / RDNG. Both are explicitly called out in the 2026-03-25 agenda.
