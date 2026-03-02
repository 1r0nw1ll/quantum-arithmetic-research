# arXiv Submission Checklist

## Pre-Submission Verification

Use this checklist before submitting any paper to arXiv.

## 1. Document Completeness

### Required Files
- [ ] Main manuscript (.tex file)
- [ ] Bibliography (.bib file or embedded)
- [ ] All figures (PDF, EPS, or PNG format)
- [ ] Supplementary Information (if applicable)
- [ ] README or 00README.XXX file (optional but recommended)

### File Format Requirements
- [ ] All figures are in acceptable formats (PDF, EPS, PNG, JPEG)
- [ ] No proprietary formats (no .doc, .docx)
- [ ] Figure file sizes are reasonable (<10MB each)
- [ ] Total submission size < 50MB (strict limit)
- [ ] No absolute file paths in \includegraphics commands

## 2. LaTeX Compilation

### Compilation Tests
- [ ] Clean compile with pdflatex (no errors)
- [ ] Bibliography compiles correctly with bibtex
- [ ] All cross-references resolve correctly
- [ ] No missing citations (? marks)
- [ ] No missing figures (empty boxes)
- [ ] No overfull/underfull hbox warnings (critical ones)

### Commands to Test
```bash
# Clean compilation test
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex

# Check for errors
grep -i "error\|warning" manuscript.log
```

## 3. arXiv-Specific Requirements

### Package Compatibility
- [ ] No incompatible packages (check arXiv compatibility list)
- [ ] No custom .sty files unless absolutely necessary (include if needed)
- [ ] No \usepackage{hyperref} issues (arXiv adds this automatically)
- [ ] No PGF/TikZ version conflicts

### Common Incompatibilities to Avoid
- Avoid: `\pdfoutput=1` (arXiv sets this)
- Avoid: absolute paths
- Avoid: `\input{|"command"}` (shell escapes disabled)
- Check: font encoding issues with non-ASCII characters

## 4. Content Verification

### Abstract
- [ ] Abstract length < 1920 characters (arXiv limit)
- [ ] No LaTeX math in abstract (use Unicode or spell out)
- [ ] No citations in abstract
- [ ] No figures/tables in abstract

### Figures
- [ ] All figures referenced in text
- [ ] All figures have captions
- [ ] Figure resolution adequate (>300 DPI for photos)
- [ ] Vector graphics used where appropriate (plots, diagrams)
- [ ] Figure file names have no spaces or special characters

### References
- [ ] All citations have entries in bibliography
- [ ] All bibliography entries are cited (no unused entries)
- [ ] URLs in bibliography are properly formatted
- [ ] DOIs included where available
- [ ] No broken hyperlinks

## 5. Metadata Preparation

### Required Metadata
- [ ] Title (concise, descriptive)
- [ ] Author names and affiliations
- [ ] Primary category selected
- [ ] Cross-list categories identified (if applicable)
- [ ] Comments field prepared (optional)
- [ ] MSC/PACS/JEL codes (if applicable)

### Category Selection Guide

**For QA Papers:**
- Primary: `cs.LG` (Machine Learning) - for ML applications
- Primary: `math.NT` (Number Theory) - for pure math
- Cross-list: `math.NT`, `physics.comp-ph`, `stat.ML`, `cs.DS`

**Common Categories:**
- `cs.LG` - Machine Learning
- `stat.ML` - Machine Learning (Statistics)
- `math.NT` - Number Theory
- `math.DS` - Dynamical Systems
- `physics.comp-ph` - Computational Physics
- `cs.DS` - Data Structures and Algorithms

## 6. Submission Package Creation

### Package Contents
- [ ] All .tex files
- [ ] All .bib files
- [ ] All figures
- [ ] Any custom .sty or .cls files
- [ ] README file (optional)

### Create ZIP Package
```bash
cd papers/ready-for-submission/qa-raman-spectroscopy
zip -r arxiv_submission.zip manuscript.tex SI.tex references.bib figures/*.pdf
```

### Package Verification
- [ ] Unzip in clean directory
- [ ] Test compilation from scratch
- [ ] Verify all files present
- [ ] Check total package size < 50MB

## 7. Final Checks

### Quality Assurance
- [ ] Spell-check entire document
- [ ] Grammar check (if possible)
- [ ] Verify all author names spelled correctly
- [ ] Verify all affiliations correct
- [ ] Check acknowledgments section
- [ ] Verify funding information (if applicable)

### Ethics and Reproducibility
- [ ] Data availability statement (if applicable)
- [ ] Code availability (GitHub link in footnote)
- [ ] Conflicts of interest disclosed
- [ ] Proper attribution of prior work

## 8. arXiv Account Setup

### Before First Submission
- [ ] arXiv account created
- [ ] Email verified
- [ ] Endorsement obtained (if required for category)
- [ ] ORCID linked (recommended)

## 9. Submission Process

### During Submission
1. [ ] Upload files or tarball
2. [ ] Select primary classification
3. [ ] Add cross-list categories
4. [ ] Enter title
5. [ ] Enter authors
6. [ ] Paste abstract (verify < 1920 chars)
7. [ ] Add comments (optional)
8. [ ] Add DOI of journal version (if applicable)
9. [ ] Add report number (if applicable)
10. [ ] Review and submit

### After Submission
- [ ] Note submission identifier (arXiv:YYMM.NNNNN)
- [ ] Verify scheduled announcement date
- [ ] Check for processing errors (email notification)
- [ ] Review proof when available
- [ ] Correct any issues before announcement

## 10. Post-Announcement

### After Paper Goes Live
- [ ] Verify PDF renders correctly
- [ ] Check all links work
- [ ] Verify metadata displays correctly
- [ ] Download and save official arXiv PDF
- [ ] Share arXiv link (Twitter, email lists, etc.)
- [ ] Update CV with arXiv citation

## Paper-Specific Checklists

### QA Raman Spectroscopy
- [ ] Files in `papers/ready-for-submission/qa-raman-spectroscopy/arxiv/`
- [ ] Abstract verified < 1920 chars
- [ ] Primary: cs.LG, Cross-list: math.NT, physics.comp-ph
- [ ] 6 figures included and properly referenced
- [ ] GitHub link in acknowledgments/footnote

### Phase 1 PAC-Bayes (after completion)
- [ ] Primary: cs.LG or stat.ML
- [ ] Cross-list: math.NT, math.PR (Probability)
- [ ] 3 figures included
- [ ] Theorem environments compile correctly

### Phase 2 Signal Classification (after completion)
- [ ] Primary: cs.LG or eess.SP (Signal Processing)
- [ ] Cross-list: stat.ML
- [ ] 7 figures included
- [ ] All dataset descriptions accurate

## Common arXiv Errors and Fixes

| Error | Fix |
|-------|-----|
| "Unknown graphics extension" | Convert to PDF, EPS, or PNG |
| "File not found" | Check relative paths, no absolute paths |
| "Package hyperref Warning" | arXiv loads hyperref automatically, remove from preamble |
| "TeX capacity exceeded" | Simplify TikZ diagrams or pre-compile as PDF |
| "Font not found" | Use standard LaTeX fonts or embed custom fonts |
| "Bibliography errors" | Ensure .bib file included and properly formatted |

## Useful Links

- arXiv submission guidelines: https://arxiv.org/help/submit
- arXiv category taxonomy: https://arxiv.org/category_taxonomy
- arXiv TeX/LaTeX guide: https://arxiv.org/help/submit_tex
- arXiv help pages: https://arxiv.org/help

---

**Last Updated**: December 2025
**Version**: 1.0
