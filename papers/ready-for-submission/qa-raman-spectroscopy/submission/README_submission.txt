Nature Communications Submission Folder — QA Raman Spectroscopy
================================================================

Contents
--------
- manuscript_final.tex / manuscript_final.pdf
- SI_final.tex / SI_final.pdf
- references.bib
- cover_letter_naturecomms.txt
- figures/ (Figure 0 TikZ/PDF, benchmark/residue images)
- artifacts/ (CSV/JSON used in SI and reproducibility)

Compile locally (pdfLaTeX)
--------------------------
cd submission

# Manuscript
pdflatex -interaction=nonstopmode manuscript_final.tex
bibtex manuscript_final
pdflatex -interaction=nonstopmode manuscript_final.tex
pdflatex -interaction=nonstopmode manuscript_final.tex

# SI
pdflatex -interaction=nonstopmode SI_final.tex
bibtex SI_final
pdflatex -interaction=nonstopmode SI_final.tex
pdflatex -interaction=nonstopmode SI_final.tex

Notes
-----
- Figure 0 is embedded via native TikZ within the manuscript.
- All figure paths are relative to the submission/ root (figures/, artifacts/).
- The k-NN sweep figure is intentionally not included in this package (commented in the TeX) to avoid missing-graphic warnings.
