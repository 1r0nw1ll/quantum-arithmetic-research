arXiv Package — QA Raman Spectroscopy (Quantum Arithmetic)
=========================================================

Contents
--------
- manuscript_arxiv.tex          (uses PDF include for Figure 0)
- manuscript_arxiv.bbl          (bibliography content; \input in TeX)
- SI_arxiv.tex                  (supplementary information; no citations)
- references.bib                (included for completeness, not required if using .bbl)
- figures/
  - figure0_qageometry.pdf
  - qa_benchmark_report.png
  - cm24_hist.png
  - fm24_hist.png

Notes
-----
- Figure 0 is included as a PDF to avoid TikZ requirements on arXiv.
- Bibliography is provided via manuscript_arxiv.bbl; arXiv can compile without running bibtex.
- All figure paths are relative to figures/.

Build locally
-------------
cd arxiv
pdflatex -interaction=nonstopmode manuscript_arxiv.tex
pdflatex -interaction=nonstopmode manuscript_arxiv.tex

(Optional) SI
pdflatex -interaction=nonstopmode SI_arxiv.tex
pdflatex -interaction=nonstopmode SI_arxiv.tex
