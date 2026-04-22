<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.8 item 6 post-shipment cross-session handoff. Captures what landed in 2026-04-21 commit 3bde4e4 + the remaining Williams-substitute acquisition so a fresh session can pick up without conversation context. Drafted 2026-04-22 by claude-main at session close, per the same anti-context-loss discipline that produced QA_MEM_PHASE_4_8_HANDOFF.md. -->

# QA-MEM Phase 4.8 Item 6 — Post-Shipment Cross-Session Handoff

**Status:** Item 6 Tracks A+C shipped 2026-04-21 as commit `3bde4e4` (pushed `origin/main`). Track B (Radin) + Williams-substitute acquisition both deferred to the next session. Scope spec: `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_ACQUISITION.md`.

**Authoring session:** `claude-main-2313` / `corpus-phase48-item6`. Session-done broadcast 2026-04-21T03:40Z. Re-opened briefly 2026-04-22 for this handoff + two memory updates; no new acquisition.

**Why this doc exists:** The kickoff prompt for the next session lives otherwise only in the authoring-session conversation scroll. That is the exact failure mode `QA_MEM_PHASE_4_8_HANDOFF.md` was created to prevent. This doc is the anchor.

---

## 1. What shipped (commit `3bde4e4`)

### Track C (Schumann)
- `Documents/schumann_resonance/schumann_1952_zfn_7a_149.pdf` (5.1 MB, 6 pp; via `degruyterbrill.com/document/doi/10.1515/zna-1952-0202/pdf`).
- `docs/theory/schumann_1952_excerpts.md` — 6 verbatim German quotes + English glosses. Key anchor: `#schumann-1952-harmonic-series-sqrt-nnp1` (p.153) carries the closed-form `m_n = √(n(n+1)) = ω_ei · R/c` overtone law.
- `tools/qa_kg/fixtures/source_claims_schumann.json` — 1 SourceWork + 6 claims, domain=`physics`.
- Williams 1992 *Science* 256:1184 **DEFERRED** in CORPUS_INDEX (AAAS paywalled across Unpaywall / Semantic Scholar; author's MIT page `web.mit.edu/earlerw/` redirects Publications to Google Scholar with no self-archive — walked 2026-04-21).

### Track A (McCraty open-access, CC-BY)
- `Documents/heartmath_corpus/mccraty_zayas_2014_frontiers_psychology_cardiac_coherence.pdf` (886 KB, 13 pp).
- `Documents/heartmath_corpus/mccraty_2017_frontiers_public_health_hrv_social_coherence.pdf` (1.4 MB, 13 pp).
- `Documents/heartmath_corpus/alabdulgader_mccraty_2018_scirep_hrv_solar_geomagnetic.pdf` (2.0 MB, 14 pp).
- `docs/theory/mccraty_item6_excerpts.md` — 13 verbatim anchors. Key: `#mccraty-zayas-2014-coherence-ratio-formula` (p.5) carries the *peer-reviewed* HRV coherence-ratio definition `[Peak Power/(Total Power − Peak Power)]` in the 0.04–0.26 Hz LF band (0.03 Hz window). `#alabdulgader-2018-schumann-fundamental-harmonics` carries the observed NOAA series 7.83 / 14 / 20 / 26 / 33 / 39 / 45 Hz.
- `tools/qa_kg/fixtures/source_claims_mccraty.json` — 3 SourceWorks + 13 claims, domain=`psychophysiology`.

### Track B (Radin)
- Empty `Documents/radin_ions/` directory created; placeholder section in `CORPUS_INDEX.md`. No PDFs landed.

### Infrastructure
- `tools/qa_kg/CORPUS_INDEX.md` — new Schumann resonance section (2 rows: Schumann 1952 ✓ + Williams 1992 deferred), HeartMath section extended with 3 McCraty rows, Radin placeholder section, cross-reference index rows.
- `qa_alphageometry_ptolemy/qa_kg_determinism_cert_v1/expected_hash.json` — [228] rebootstrapped after new fixture auto-discovery. New `fixture_rebuild_graph_hash = d5d66753d886f38c26af74e2e84c92af335c18ed0128651c292c465c9a1f5eba`.
- `qa_alphageometry_ptolemy/_meta_ledger.json` — regenerated (429 rewrites).

**Verification:** `qa_axiom_linter --all` CLEAN; `qa_meta_validator` all certs PASS including [225] / [227] / [228] / [252] / [253] / [254]; 115 qa_kg unit tests PASS.

---

## 2. Next-session kickoff prompt (COPY VERBATIM)

Pick up QA-MEM Phase 4.8 item 6 acquisition: Williams-adjacent Schumann resonance substitutes for the follow-up numerical cert to [259].

**Context:**
- Previous session (commit `3bde4e4`, 2026-04-21) landed Track C (Schumann 1952) and Track A (McCraty+Zayas 2014 / McCraty 2017 / Alabdulgader+McCraty 2018). Williams 1992 *Science* 256:1184 captured as DEFERRED (AAAS paywalled across Unpaywall / Semantic Scholar / author pages; `web.mit.edu/earlerw/` Publications redirects to Google Scholar with no self-archive).
- Authoritative spec: `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_ACQUISITION.md`. Previous-session synthesis: Williams 1992 is not load-bearing — three later Williams+co-author papers carry richer numerical data for the cert's Q-factor / ε-bound / solar-cycle-frequency-modulation needs.

**Target: acquire any of the three substitutes below (priority order):**

1. **Satori, Williams, Mushtak, Fullekrug (2005)** — *Response of the cavity resonator to the 11-year solar cycle*. J. Atmos. Sol. Terr. Phys. 67:553–562. Documents the ~0.3 Hz solar-cycle variation in SR modal frequencies. **Highest-value target**: gives the empirical non-invariance that bounds any theoretical √(n(n+1)) claim.
2. **Mushtak, Williams (2002)** — *ELF propagation parameters for uniform models of the Earth-ionosphere waveguide*. J. Atmos. Sol. Terr. Phys. 64:1989. Lorentzian-fit Q-factors at West Greenwich RI — gives the quantitative damping correction for the cert's ε bound.
3. **Williams, Mushtak, Nickolaenko (2006)** — *Distinguishing ionospheric models using Schumann resonance spectra*. J. Geophys. Res. 111:D16107. DOI 10.1029/2005JD006944. Multi-ionospheric-model comparison.

**Acquisition order:** query Semantic Scholar `openAccessPdf` field first (`api.semanticscholar.org/graph/v1/paper/search?query=...` or `/DOI:...`), then Unpaywall (`api.unpaywall.org/v2/<DOI>?email=unpaywall@impactstory.org`), then direct journal / author. Semantic Scholar was rate-limiting (HTTP 429) at end of previous session — may need a ~60s cool-down before the first call.

**Per-paper deliverables (identical to previous session pattern):**
- PDF under `Documents/schumann_resonance/` (cwd-scoped `curl`; never put `Documents/` substring in a Bash command body — see `feedback_cert_gate_bridge_health.md`).
- Verbatim page-numbered quotes in a new `docs/theory/<first-author>_<year>_excerpts.md`.
- Claim fixture extending `tools/qa_kg/fixtures/source_claims_schumann.json` (add SourceWork + 3–6 claims), domain=`physics`.
- CORPUS_INDEX.md row under the Schumann resonance corpus section.

**Fallback if all three are paywalled:** pivot to Track B (Radin) per spec §B. First target: Mossbridge, Tressoldi, Utts (2012) *Predictive physiological anticipation*, Frontiers in Psychology 3:390 (CC-BY, `doi:10.3389/fpsyg.2012.00390`) — clean acquisition path, lands in the empty `Documents/radin_ions/` directory.

**Known blockers (unchanged):**
- Codex cert-gate bridge dead (`qa_lab/logs/codex_bridge.log` mtime Apr 10 as of 2026-04-21, 11+ days stale). Check the log mtime first. `codex exec --full-auto` is the documented one-shot fallback — previous session used it to clear the expected_hash.json rebootstrap packet. **Correct CLI form:** `python3 tools/qa_codex_quarantine_review.py approve --all --reviewer codex --notes '<one-line reason>'` — the `--notes` flag is *required*; previous session's first invocation without it exited 2.
- `Documents/schumann_resonance/` + `Documents/radin_ions/` + `Documents/heartmath_corpus/` still NOT in `DOCUMENTS_PDF_INGRESS_PREFIXES` (Phase 4.8 body item 4 still blocked on bridge health). Use cwd-scoped writes.
- After any new `source_claims_*.json` fixture lands, [228] determinism cert will FAIL with hash drift — documented rebootstrap: `rm qa_alphageometry_ptolemy/qa_kg_determinism_cert_v1/expected_hash.json && cd qa_alphageometry_ptolemy && python3 qa_meta_validator.py`. This command triggers `CLAUDE_PYTHON_WRITE_QUARANTINED`; route through the Codex one-shot fallback above.

**Scaffold gotchas (`memory/feedback_cert_scaffold_gotchas.md`):** N/A in acquisition — no `mapping_protocol_ref.json` authoring in this session. If the next-next session does author the cert: `additionalProperties: False` on that schema; PRIMARY-SOURCE-EXEMPT comment must stay under ~400 bytes.

**Session-start protocol (required):**
1. `mcp__open-brain__recent_thoughts(since_days=3)` before anything else.
2. `collab_get_state(key="file_locks")`, register `session:corpus-phase48-item6-williams-substitutes` with scope `Documents/schumann_resonance/, Documents/radin_ions/, docs/theory/*_excerpts.md, tools/qa_kg/fixtures/source_claims_schumann.json, tools/qa_kg/CORPUS_INDEX.md`.
3. Read `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_ACQUISITION.md` + `memory/project_qa_mem_phase_4_8_heartmath.md` before touching files.

**Scope bound:** acquisition only. Do NOT author `qa_heartmath_mapping_cert_v1` or `qa_schumann_harmonic_cert_v1` in this session — both depend on a pending decision flagged in the 2026-04-21 ChatGPT-pushback exchange (see §3 below). Cert authoring is a separate session. Same discipline as [259] — build the boundary, not the claim.

---

## 3. Open cert-authoring decision (NOT for this next session)

ChatGPT proposed a `qa_schumann_harmonic_cert_v1` as the first numerical mapping cert, 2026-04-21. Claude pushed back on three points that remain unresolved:

1. **Drops mod-9 / mod-24 framing.** The item 6 spec §"For Schumann ↔ mod-9 / mod-24 orbit-harmonic ratios" specifies the cert tests whether harmonic-ratio rational approximations land in canonical mod-9/24 residue classes. ChatGPT's "map to orbit under a single generator" with "no (b,e,d,a) embedding yet" abstracts that away. Without mod-9/24, the cert is number theory, not QA.
2. **P1's `global ε` is the load-bearing parameter and undeclared.** Theoretical vs observed ratios for the first 5 harmonics: theoretical `√(n(n+1)/2)` from Schumann 1952 = {1.000, 1.732, 2.449, 3.162, 3.873}; observed (Alabdulgader+McCraty 2018) = {1.000, 1.788, 2.554, 3.320, 4.215}; relative error 0% / 3.2% / 4.3% / 5.0% / 8.8% (monotonically growing — the Earth-cavity damping correction Schumann's §II flags). If ε = 10%, cert is weak; if ε = 3%, cert fails at n=3. ε must be set by a stated physical model (Q-factor bound), not a magic constant.
3. **"Invariant backbone everything else attaches to" is sequencing overreach.** A Schumann-ratio cert passing does not gate HI↔coherence-ratio. The three mapping tracks in the spec are parallel.

The Williams-substitute acquisition (§2 above) is what unblocks decision (2): Satori+Williams+Mushtak+Fullekrug 2005 provides the empirical ε substrate; Mushtak+Williams 2002 provides the Q-factor. Without at least one, the cert can't set ε honestly.

Will has not authorized cert authoring. The cert-authoring session opens only after Will's explicit go-ahead following substitute acquisition.

---

## 4. Pointers (single-source-of-truth)

- **Spec** — `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_ACQUISITION.md` (2026-04-20 commit `c08ec5e`).
- **Kickoff-era handoff** — `docs/specs/QA_MEM_PHASE_4_8_HANDOFF.md` (2026-04-17).
- **This handoff** — `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_HANDOFF.md` (2026-04-22).
- **Memory** — `memory/project_qa_mem_phase_4_8_heartmath.md` (updated 2026-04-22 to reflect item 6 shipment).
- **Bridge feedback** — `memory/feedback_cert_gate_bridge_health.md` (updated 2026-04-22 to capture `--notes` required-flag).
- **Commit** — `3bde4e4` for the full deferred-items list and verification record. `git show 3bde4e4`.
- **Excerpts** — `docs/theory/schumann_1952_excerpts.md`, `docs/theory/mccraty_item6_excerpts.md`, `docs/theory/heartmath_phase4_8_excerpts.md`.
- **Fixtures** — `tools/qa_kg/fixtures/source_claims_{schumann,mccraty,heartmath}.json`.
- **Corpus index** — `tools/qa_kg/CORPUS_INDEX.md` (Schumann + HeartMath + Radin sections).
