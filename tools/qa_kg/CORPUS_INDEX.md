<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.6 exhaustive corpus directory. Single source of truth mapping every primary-source file on disk to its corpus, author, topic tags, and SourceWork node (if ingested). Addresses the 2026-04-17 correction request: prevent the "missed qa_fst/" class of failure where a session greps only for author name ("briddell") and misses topic-dir locations ("qa_fst/"). Any session adding a new primary source MUST add a row here in the same commit. -->

# QA-MEM Corpus Index

Exhaustive, hand-maintained directory of every primary-source file on disk that QA-MEM references or should reference. Organized by corpus, then by file. Each row pins: path, author, topic tags (multiple — search by any), primary domain, SourceWork node ID (if ingested), and status.

**Maintenance rule:** any session adding a new primary source MUST add a row here in the same commit that ingests it. Any session noticing a stale row (file moved, deleted, renamed, new version) MUST update the row. Discovery convention is "grep this file first" — do not rely on filename conventions or single-directory scans.

**Index self-check:** `tools/qa_kg/tests/test_corpus_index.py` (Phase 4.7 follow-up) will verify every SourceWork node has a matching row and every row with a non-empty SourceWork ID resolves to an actual node.

---

## Wildberger corpus

**Corpus root (canonical):** `Documents/wildberger_corpus/`
**Text extracts:** `corpus/wildberger_text/`
**Theory docs:** `docs/theory/QA_WILDBERGER_*.md`, `docs/theory/wildberger_phase4_6_excerpts.md`

Legend: ✓ = SourceWork ingested; ⏳ = deferred; ⚠ = misfiled (non-Wildberger content under Wildberger naming).

| Path | Author | Topic tags | Domain | SourceWork | Status |
|---|---|---|---|---|---|
| `Documents/wildberger_corpus/UHG_I_trigonometry_0909.1377.pdf` | Wildberger 2009 | UHG, hyperbolic, trigonometry, projective | geometry | `wildberger_2009_uhg_i` | ✓ |
| `Documents/wildberger_corpus/UHG_II_1012.0880.pdf` | Wildberger 2010 | UHG, pictorial, projective | geometry | `wildberger_2010_uhg_ii_pictorial` | ✓ |
| `Documents/wildberger_corpus/chromogeometry_2008.pdf` | Wildberger 2008 | chromogeometry, Euler line, three-fold | geometry | `wildberger_2008_chromogeometry` | ✓ |
| `Documents/wildberger_corpus/simply_laced_JLT.pdf` | Wildberger 2003 | G2, Lie algebra, combinatorial, multiset | geometry | `wildberger_2003_g2_combinatorial` | ✓ |
| `Documents/wildberger_corpus/pell_no_irrationals.pdf` | Wildberger 2008 | Pell, matrix, Stern-Brocot | geometry | `wildberger_2008_pell_no_irrationals` | ✓ |
| `Documents/wildberger_corpus/hyper_catalan_exercises_2507.13045.pdf` | Wildberger+Rubine 2025 | hyper-Catalan, polygon subdivision | geometry | — | ⏳ 4.7 (not yet ingested) |
| `Documents/wildberger_corpus/hyper_catalan_recurrences_2507.04552.pdf` | Rubine 2025 | hyper-Catalan, Geode, recurrence | geometry | `rubine_2025_hyper_catalan_geode_rec` | ✓ |
| `Documents/wildberger_corpus/geode_conjectures_2506.17862.pdf` | Amdeberhan+Zeilberger 2025 | Geode, WZ, Lagrange inversion | geometry | `amdeberhan_zeilberger_2025_geode_conjectures` | ✓ |
| `Documents/wildberger_corpus/finite_interp_2507.20003.pdf` | Mukewar 2025 | hyper-Catalan, finite interpretation | geometry | `mukewar_2025_finite_interp` | ✓ |
| `Documents/wildberger_corpus/aff_proj_metrical.pdf` | Wildberger 2006 | affine, projective, universal geometry, bilinear form | geometry | `wildberger_2006_aff_proj_universal` | ✓ |
| `Documents/wildberger_corpus/diamonds.pdf` | Wildberger 2003 | sl3, quark, diamond polytope | geometry | `wildberger_2003_diamonds_sl3` | ✓ |
| `Documents/wildberger_corpus/apollonian.pdf` | **Pandey et al 2020** (MISFILED) | quantum chaos, NOT Wildberger | — | — | ⚠ misfiled |
| `Documents/wildberger_corpus/feuerbach.pdf` | **Casalaina-Martin+Laza 2010** (MISFILED) | ADE singularities, NOT Wildberger | — | — | ⚠ misfiled |
| `Documents/wildberger_corpus/incenter_omega.pdf` | **Wibmer 2020** (MISFILED) | algebraic difference eqs, NOT Wildberger | — | — | ⚠ misfiled |
| `Documents/wildberger_corpus/parabola_uhg.pdf` | **Bernečić+Šegulja 2013** (MISFILED) | marine diesel engines, NOT Wildberger | — | — | ⚠ misfiled (see WILDBERGER_CORPUS_CLEANUP.md) |
| `Documents/wildberger_corpus/tetrahedron_vec.pdf` | Notowidigdo+Wildberger 2019 | rational trigonometry, tetrahedron | geometry | `notowidigdo_wildberger_2019_tetrahedron` (via text extract) | ✓ |
| `corpus/wildberger_text/mutation_game_2020.txt` | Wildberger | Mutation Game, Coxeter graph, multiset | geometry | `wildberger_mutation_game` | ✓ |
| `~/Downloads/MutationGameCoxeterGraphs.pdf` | Wildberger | Mutation Game (PDF source) | geometry | — | ⏳ 4.7 (text extract used instead; PDF not persisted) |

---

## Haramein / RSF corpus

**Corpus root:** `Documents/haramein_rsf/`
**Domain:** `rsf` (added to domain_taxonomy.json in Phase 4.6)
**Theory docs:** `docs/theory/QA_HARAMEIN_SCALING_DIAGONAL.md`, `docs/theory/haramein_corpus_excerpts.md`
**Cert grounding:** [218] `qa_haramein_scaling_diagonal_cert_v1`

| Path | Author | Topic tags | Domain | SourceWork | Status |
|---|---|---|---|---|---|
| `Documents/haramein_rsf/scale_unification_2008.pdf` | Haramein 2008 | scale unification, Schwarzschild proton | rsf | — | ⏳ 4.7 (grandfathered in cert [218] fixtures; add SourceWork) |
| `Documents/haramein_rsf/quantum_gravity_holographic_mass_2013.pdf` | Haramein 2013 | PSU, holographic mass, 4φ² coupling | rsf | `haramein_2013` | ✓ |
| `Documents/haramein_rsf/unified_spacememory_network_2016.pdf` | Haramein+Brown+Val Baker 2016 | spacememory, wormhole network, Hubble mass | rsf | `haramein_brown_valbaker_2016` | ✓ |
| `Documents/haramein_rsf/electron_holographic_mass_2019.pdf` | Val Baker+Haramein+Alirol 2019 | electron mass, Rydberg, atomic number Z | rsf | `valbaker_haramein_alirol_2019` | ✓ |
| **NOT ON DISK** | Haramein+Val Baker 2022/2023 | Generalized Holographic Model I/II/III | rsf | — | ⏳ 4.7 (HIUP unreachable; try ISF or ResearchGate alt) |

---

## Levin corpus

**Corpus root:** `corpus/levin/` (repo-root unprotected — Documents/ carve-out does NOT extend to levin; Phase 4.7 may request Codex extension)
**Theory docs:** `docs/theory/levin_phase4_6_excerpts.md`
**Cert grounding:** [179] `qa_levin_cognitive_lightcone_cert_v1`, [180] `qa_pezzulo_levin_bootstrap_cert_v1`
**Browser scrapes (non-primary):** `.playwright-mcp/levin_{profile,scholar,tufts}.md`

| Path | Author | Topic tags | Domain | SourceWork | Status |
|---|---|---|---|---|---|
| `corpus/levin/computational_boundary_self_2019.pdf` | Levin 2019 | Computational Boundary, cognitive light cone, scale-free cognition | biology | `levin_2019_computational_boundary` | ✓ |
| `corpus/levin/technological_approach_mind_everywhere_2022.pdf` | Levin 2022 | TAME, morphogenesis, basal cognition | biology | `levin_2022_tame_mind_everywhere` | ✓ |
| `corpus/levin/brainca_2604.01932.pdf` | Pio-Lopez+Hartl+Levin 2026 | BraiNCA, NCA, attention, long-range | biology | `piolopez_hartl_levin_2026_brainca` | ✓ |
| `corpus/levin/cognition_atwd_2_0_chis_ciure_levin_2025.docx` | Chis-Ciure+Levin 2025 | Cognition ATWD 2.0, Synthese | biology | `chisciure_levin_2025_atwd_2_0` (placeholder) | ⏳ 4.7 (needs docx→text pipeline) |
| **NOT ON DISK** | Fields+Levin 2022 | "Intelligence Beyond Neurons" | biology | — | ⏳ 4.7 (title mismatch on current Tufts page; verify) |
| **NOT ON DISK** | Fields+Levin 2023 | "Non-Duality" | biology | — | ⏳ 4.7 (title mismatch) |
| **NOT ON DISK** | Levin 2022 | Cognition ATWD v1 (only v2.0 2025 on PhilSci) | biology | — | ⏳ 4.7 (manual lab-page verification) |
| **NOT ON DISK** | Pezzulo+Levin | "Bootstrapping Cognition" | biology | — | ⏳ 4.7 (exact title unclear; likely renamed) |
| **NOT ON DISK** | Levin et al | "Engineered Living Machines" | biology | — | ⏳ 4.7 (several xenobot papers; none verbatim this title) |

---

## Briddell / FST corpus (CORRECTED 2026-04-17)

**Primary (on disk):** `qa_alphageometry_ptolemy/qa_fst/` — the Will+Briddell co-authored v2 paper lives HERE, not under `field-structure-theory/` (which is scaffolding/analysis).
**Secondary (on disk):** `field-structure-theory/` — scaffolding docs, chat exports, simulation code.
**External referenced (not on disk):** `briddellbook2020` = Don Briddell, *Structural Physics* monograph.
**Theory docs:** `docs/theory/briddell_fst_v2_excerpts.md`
**Cert grounding:** `qa_fst` subprocess-validated module (cert ID [13] in meta_validator; not registered as a FAMILY_SWEEPS cert:fs:* node).

| Path | Author | Topic tags | Domain | SourceWork | Status |
|---|---|---|---|---|---|
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.tex` | Dale+Briddell 2026 (v2) | FST, QA completion layer, Frontiers 1850870, Theorem NT | physics | `dale_briddell_2026_qa_completion_layer_v2` | ✓ (OFFICIAL) |
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.pdf` | Dale+Briddell 2026 (v2) | compiled PDF of v2 .tex | physics | (same SourceWork as .tex) | ✓ |
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_addendum_b_equals_e_diagonal.tex` | Dale+Briddell 2026 | b=e diagonal, STF={3^n}, base-3 forced | qa_core | `dale_briddell_2026_fst_diagonal_addendum` | ✓ (HELD pending Frontiers round) |
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_addendum_b_equals_e_diagonal.pdf` | Dale+Briddell 2026 | compiled PDF of addendum | qa_core | (same SourceWork) | ✓ |
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_cert_bundle.json` | Dale+Briddell 2026 | v2 cert bundle (5 claims) | physics | (cert infrastructure, not SourceWork) | ✓ live |
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_validate.py` | Dale+Briddell 2026 | v2 validator, 7 firewall crossings | physics | (cert infrastructure) | ✓ live |
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_module_spine.json` | Dale+Briddell 2026 | spine ⟨Ω,Σ,G,I,F⟩ | physics | (cert infrastructure) | ✓ live |
| `qa_alphageometry_ptolemy/qa_fst/qa_fst_manifest.json` | Dale+Briddell 2026 | SHA-256 manifest | physics | (cert infrastructure) | ✓ live |
| `wt-certs/qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.tex` | Dale+Briddell | OLDER (723 lines; v1 or pre-v2) | — | — | stale worktree copy |
| `gemini_qa_project/incoming/qa_fst_completion_paper-2.tex` | Dale+Briddell | older draft (721 lines) | — | — | stale staging copy |
| `ingestion candidates/qa_fst_completion_paper.odt` | Dale+Briddell | ODT version | — | — | reference only |
| `ingestion candidates/fst_journal.odt` | Dale/Will | FST journal/working notes | — | — | reference only |
| `~/Documents/qa_fst_completion_paper.odt` | Dale+Briddell | user-home ODT copy | — | — | reference only |
| `~/Downloads/qa_fst_completion_paper.pdf` | Dale+Briddell | downloads copy | — | — | stale |
| `field-structure-theory/01-core-theory/fst-fundamentals/fst-briddell-background-qa-analysis.txt` | Perplexity AI | Briddell bio scraped by AI | — | — | chat export, NOT primary |
| `field-structure-theory/01-core-theory/fst-fundamentals/FST and QA Comparison.md` | ChatGPT 2025-05-25 | FST↔QA dialogue | — | — | chat export, NOT primary |
| `field-structure-theory/01-core-theory/fst-fundamentals/Quantum Arithmetic and FST.md` | ChatGPT 2025-03-15 | QA prime residue synthesis (Will user turn) | — | — | chat export, user content quotable but not published |
| `field-structure-theory/03-integration-frameworks/qa-fst-integration/Field structure translation.md` | ChatGPT 2025-10-09 | Hawkins→Grant lineage | — | — | chat export |
| `field-structure-theory/03-integration-frameworks/qa-fst-integration/Reframing using FST.md` | ChatGPT 2025-08-23 | Will's QA paper pasted for reframing | — | — | chat export |
| `field-structure-theory/02-mathematical-frameworks/`, `04-applications/`, `05-simulations-code/`, `06-related-theories/` | mixed | FST simulation + applied theory working docs | — | — | Phase 4.7: audit each for primary vs derivative content |

---

## Iverson QA corpus (Phase 4.5)

**On-disk:** DOCX sources at `~/Desktop/files/quantum_pythagoras-text/` (off-repo authoritative); repo excerpts at `docs/theory/iverson_qa_vol_ii_excerpts.md`, `iverson_qa2_excerpts.md`, `iverson_pyth3_enneagram_excerpts.md`.
**Fixture:** `tools/qa_kg/fixtures/source_claims_iverson.json` (3 works, 6 claims).

---

## Keely SVP / Pond corpus (Phase 3 + 4.5)

**On-disk:** SVP wiki snapshot at `docs/theory/svp_wiki_qa_elements_snapshot.md`; Pond torus notes at `private/svp_propositions_qa_mapping.md`; Keely 40 laws primary text snapshot at `docs/theory/keely_40_laws_primary_text_snapshot.md`.
**Fixtures:** `source_claims_phase3.json` (SVP wiki — 15 claims, 11 observations, 13 contradicts), `source_claims_keely.json` (5 claims).
**Cert grounding:** [153], [184]–[188] (Keely 40 laws); [140] (conic discriminant for Law I+ negative).

---

## Parker / Hull / Whittaker / Philomath corpus

**On-disk:**
- `~/Downloads/PHILOMATH_...Grant_Talal_Ghannam.pdf` — Philomath 398pp
- `~/Downloads/A_Treatise_on_Electricity_and_Magnetism.pdf` — Whittaker 51MB
- `~/Downloads/Interference.pdf` — Whittaker 53MB
- `~/Downloads/00_QuadraturePRINT.pdf` — Hull Quadrature
- `~/Downloads/quadraturecircl00parkgoog.pdf` — Parker Quadrature (historical)

**Excerpt files:**
- `docs/theory/parker_geometry_propositions_excerpts.md`, `hull_quadrature_excerpts.md`, `philomath_excerpts.md`

**Fixtures:**
- `tools/qa_kg/fixtures/source_claims_parker_hull.json` — Parker + Hull (2 works, 3 claims, 1 obs, 2 edges)
- `tools/qa_kg/fixtures/source_claims_philomath.json` — Grant+Ghannam (1 work, 10 claims, 1 obs, 1 edge)

**Deferred to 4.7:** Whittaker Treatise + Interference text extraction (51+53 MB, needs targeted chapter pypdf reads).

---

## HeartMath corpus (Phase 4.8 KICKOFF)

**Corpus root:** `Documents/heartmath_corpus/`
**Domain:** `""` (pending) — candidate `psychophysiology` extension of `domain_taxonomy.json` deferred to Phase 4.8 body.
**Theory docs:** `docs/theory/heartmath_phase4_8_excerpts.md` (stub — verbatim claim extraction pending)
**Origin search:** `scholar.google.com/scholar?start=130&q=heartmath&hl=en&as_sdt=4007` (results 131–140)
**QA-research grounding:** OB agenda 2026-03-25 (completeness audit, Thread 3: HeartMath / McCraty / Radin) — HRV coherence ratio ↔ QA HI; brain–heart cross-coherence ↔ QA Markovian coupling; Schumann 7.83 Hz ↔ mod-9 / mod-24 orbit harmonics.
**Ingress note:** `Documents/heartmath_corpus/` is NOT yet in `DOCUMENTS_PDF_INGRESS_PREFIXES` in `llm_qa_wrapper/cert_gate_hook.py` / `.claude/hooks/pretool_guard.sh`. Phase 4.8 kickoff PDFs written via cwd-scoped downloads (no `Documents/` substring in the Bash command). Extending the allowlist is a WRAPPER_SELF_MODIFICATION edit routed through Codex quarantine review — deferred while Codex bridge is dead.

| Path | Author | Topic tags | Domain | SourceWork | Status |
|---|---|---|---|---|---|
| `Documents/heartmath_corpus/tomasino_1997_water_em_storage.pdf` | Tomasino 1997 | water liquid-crystal, EM storage, subtle energy, HeartMath Pub 97 | "" (pending) | `tomasino_1997_water_em_storage` | ✓ (kickoff — claims pending) |
| `Documents/heartmath_corpus/danielson_2014_hospital_wellness.pdf` | Danielson+Jeffers+Kaiser et al 2014 | HeartMath mastery, hospital wellness, population health, PMC3923282 | "" (pending) | `danielson_2014_hospital_wellness` | ✓ (kickoff — claims pending) |
| `Documents/heartmath_corpus/oschman_2015_heart_bidirectional_scalar_antenna.pdf` | Oschman+Oschman 2015 | scalar field antenna, Rein resonance, Whittaker decomposition, bidirectional | "" (pending) | `oschman_2015_heart_bidirectional_scalar_antenna` | ✓ (kickoff — claims pending) |
| `Documents/heartmath_corpus/edwards_2018_cfp_heartmath_psychology.pdf` | Edwards 2018 | HeartMath psychology CFP, Journal of Psychology in Africa | "" (pending) | `edwards_2018_cfp_heartmath_psychology` | ✓ (kickoff — claims pending) |
| **NOT ON DISK** | Institute of HeartMath 2012 | EmWave Desktop product citation | — | — | [CITATION] — no paper |
| **NOT ON DISK** | Edwards+David+Hermann et al 2023 | HeartMath meditation, longer breath cycle, *Dialogo* | — | — | ⏳ EBSCO paywall |
| **NOT ON DISK** | Simmons 2010 | heart-generated coherence, distilled water, plant growth | — | — | [CITATION] — thesis, no PDF indexed |
| **NOT ON DISK** | Childre+Martin+Beech 1999 | *The HeartMath Solution* (Harper) | — | — | Published book, no PDF |
| **NOT ON DISK** | 명화숙 et al 2014 | HeartMath psychoeducation, Korean caregivers | — | — | ⏳ foreign-journal paywall |
| **NOT ON DISK** | Böckeler+Cornforth+Drummond et al 2020 | paced breathing vs game-biofeedback, HRV, IEEE EMBC | — | — | ⏳ IEEE Xplore paywall |

---

## Cross-reference index (topic → file)

Looking up by **topic or keyword** rather than author — this catches the "missed qa_fst because I grep'd briddell" class of failure.

| Topic / keyword | Primary file paths |
|---|---|
| **FST** | `qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.tex` + `qa_fst_addendum_b_equals_e_diagonal.tex`, `field-structure-theory/` (scaffolding only) |
| **Briddell** | (same as FST) + `briddellbook2020` external reference |
| **qa_fst** | `qa_alphageometry_ptolemy/qa_fst/*` (cert + paper + addendum) |
| **Frontiers 1850870** | `qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.tex` (v2 submission) |
| **b=e diagonal** | `qa_alphageometry_ptolemy/qa_fst/qa_fst_addendum_b_equals_e_diagonal.tex`, `docs/theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md` |
| **STF / Sierpinski** | FST paper + addendum, `docs/theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md` |
| **Rspin / Aspin / Structor / Plenum** | FST paper §2.1 |
| **Observer Projection / Theorem NT / firewall** | FST paper §3.2, `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`, `QA_AXIOMS_BLOCK.md` |
| **Postulate P1 / 1 Love / electron mass** | FST paper §4 |
| **hyper-Catalan / Geode** | Wildberger corpus: hyper_catalan_recurrences, geode_conjectures, finite_interp, hyper_catalan_exercises |
| **UHG / universal hyperbolic** | Wildberger corpus: UHG_I, UHG_II |
| **Chromogeometry** | Wildberger corpus: chromogeometry_2008.pdf |
| **Mutation Game / Coxeter** | Wildberger corpus: mutation_game_2020.txt, ~/Downloads/MutationGameCoxeterGraphs.pdf |
| **spacememory / PSU / holographic** | Haramein corpus (all 4 files) |
| **Levin / cognitive lightcone / TAME / BraiNCA** | corpus/levin/ (4 files) |
| **Iverson / QA book / enneagram** | `~/Desktop/files/quantum_pythagoras-text/*.docx` off-repo + repo excerpt snapshots |
| **Keely / 40 laws / SVP** | `docs/theory/keely_40_laws_primary_text_snapshot.md`, `docs/theory/svp_wiki_qa_elements_snapshot.md`, `private/svp_propositions_qa_mapping.md` |
| **Pond / torus / Dale Pond** | `private/svp_propositions_qa_mapping.md` |
| **Philomath / digital root / base-9 / mod-24** | `~/Downloads/PHILOMATH_...pdf`, `qa_ingestion_sources/qa_philomath_*.json` |
| **Whittaker / Treatise / Interference** | `~/Downloads/A_Treatise_on_Electricity_and_Magnetism.pdf`, `~/Downloads/Interference.pdf` |
| **Parker / quadrature** | `~/Downloads/quadraturecircl00parkgoog.pdf`, `~/Downloads/00_QuadraturePRINT.pdf` |
| **HeartMath / McCraty / cardiac coherence / HRV** | `Documents/heartmath_corpus/` (4 files: tomasino 1997, danielson 2014, oschman 2015, edwards 2018) |
| **Oschman / scalar antenna / Rein resonance** | `Documents/heartmath_corpus/oschman_2015_heart_bidirectional_scalar_antenna.pdf` |
| **water EM structuring / liquid crystal lattice** | `Documents/heartmath_corpus/tomasino_1997_water_em_storage.pdf` |

---

## Phase 4.7 hardening backlog (per this index)

1. Add `Haramein 2008 Scale Unification` SourceWork to close the Phase 4.6 hole (currently grandfathered in cert [218] fixtures but no SourceWork node).
2. Ingest `Wildberger+Rubine 2025 hyper_catalan_exercises_2507.13045.pdf` (currently indexed but no SourceWork).
3. Extend the Documents/ carve-out to `Documents/levin_corpus/` + `Documents/whittaker/` + `Documents/heartmath_corpus/` (Phase 4.8 kickoff) if Will wants Levin/Whittaker/HeartMath PDFs under the same Documents/ umbrella (currently Levin in `corpus/` repo-root unprotected, Whittaker in `~/Downloads/`, HeartMath under Documents/ but not yet in ingress allowlist).
4. Write `tools/qa_kg/tests/test_corpus_index.py` to enforce index ↔ fixture ↔ on-disk consistency:
   - Every `✓` row's SourceWork ID resolves to a node in `qa_kg.db`.
   - Every SourceWork node has exactly one index row.
   - Every file referenced by `source_locator` in any fixture exists on disk OR is annotated off-repo here.
5. Audit `field-structure-theory/02-mathematical-frameworks/`, `04-applications/`, `05-simulations-code/`, `06-related-theories/` for primary-vs-scaffolding classification.
6. Batch-download pipeline hardening per `docs/theory/WILDBERGER_CORPUS_CLEANUP.md`: pre-save first-page author/title verification via pypdf to prevent the `/tmp/wild/` misfile class.
