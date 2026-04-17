<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.6 corpus-hygiene observation record. Documents the misfile anomaly discovered during Wildberger primary-source extraction. No quoted material — pure hygiene audit record. -->

# Wildberger Corpus — Phase 4.6 Cleanup Audit

**Audit date:** 2026-04-17
**Session:** qa-mem-phase-4-6-full-corpus

## Finding: 4 of 8 `/tmp/wild/*.txt` extracts were misfiled (not Wildberger)

During Phase 4.6 Wildberger SourceClaim extraction, 4 of the 8 `.txt` files that had been extracted from `/tmp/wild/*.pdf` in an earlier session turned out to be misnamed — the filenames suggested Wildberger papers but the content came from unrelated authors. This is a hygiene anomaly to record, not a data corruption (the text itself is correct for what the files actually are; only the naming convention broke).

| Filename in `/tmp/wild/` (now persisted to `corpus/wildberger_text/`) | Actual paper | Author(s) | Year |
|---|---|---|---|
| `apollonian.txt` | *Adiabatic eigenstate deformations as a sensitive probe for quantum chaos* | Pandey, Claeys, Campbell, Polkovnikov, Sels | 2020 |
| `feuerbach.txt` | *Simultaneous Semi-Stable Reduction for Curves with ADE Singularities* | Casalaina-Martin, Laza | 2010 |
| `incenter_omega.txt` | *On the dimension of systems of algebraic difference equations* | Wibmer | 2020 |
| `parabola_uhg.txt` | *Heat transfer influence analysis on combustion pressure in two-stroke slow-speed marine diesel engines* | Bernečić, Šegulja | 2013 |

The first three (quantum chaos, ADE singularities, algebraic difference equations) are in broadly-adjacent mathematical territory and may have landed in the `/tmp/wild/` queue from a Wildberger-neighbour topic search. The fourth (marine diesel engines) is a full-on misfile and should be removed from any Wildberger corpus claims.

## Files that ARE genuine Wildberger text extracts

| Filename | Actual paper | Notes |
|---|---|---|
| `aff_proj_metrical.txt` | *Affine and Projective Universal Geometry* (arXiv:math/0612499) | Wildberger 2006 — keep |
| `diamonds.txt` | *Quarks, diamonds, and representations of sl₃* | Wildberger 2003 — keep |
| `mutation_game_2020.txt` | *The Mutation Game, Coxeter Graphs, and Partially Ordered Multisets* | Wildberger — keep |
| `tetrahedron_vec.txt` | *Generalised vector products and metrical trigonometry of a tetrahedron* (arXiv:1909.08814) | Notowidigdo & Wildberger 2019 — keep |

## Action taken

- No file deletions (corpus PDFs in `Documents/wildberger_corpus/` are the authoritative store; `/tmp/wild/` text extracts are derivative artefacts).
- Phase 4.6 `source_claims_phase4_6_wildberger.json` fixture uses **only the 4 genuine Wildberger text extracts** as verbatim-quote sources, plus direct PDF reads of the 12 canonical Wildberger PDFs in `Documents/wildberger_corpus/` via pypdf.
- The 4 misfiled `.txt` files remain on disk under `corpus/wildberger_text/` in case their (non-Wildberger) content becomes useful to other domains — but they are **not referenced** by any SourceWork or SourceClaim in the QA-MEM graph.
- Future Wildberger ingestion sessions should verify first-page author/title lines against the expected filename before committing extracts to a Wildberger-tagged location.

## Upstream cause (hypothesis)

Phase 4.5 or an earlier session ran a batched PDF download with filename-from-URL heuristics. The download URLs for the 4 misfiled papers likely contained substrings that matched a Wildberger keyword list (e.g., "Apollonian" → `apollonian.txt`, "Feuerbach" → `feuerbach.txt`) but the actual PDFs from the resolved URLs were different papers. No pre-save author-line verification step existed in that pipeline.

A Phase 4.7 hardening item: when batch-downloading primary sources, extract the first-page author+title line via pypdf before finalizing the filename. Reject (or flag for manual rename) any download whose first-page author doesn't match the expected author.
