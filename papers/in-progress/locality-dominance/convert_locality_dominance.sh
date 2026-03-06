#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_MD="${ROOT_DIR}/locality_dominance_paper.md"
OUT_TEX="${ROOT_DIR}/locality_dominance_paper_pandoc.tex"

if [[ -x /tmp/pandoc-3.9/bin/pandoc ]]; then
  PANDOC=/tmp/pandoc-3.9/bin/pandoc
elif command -v pandoc >/dev/null 2>&1; then
  PANDOC="$(command -v pandoc)"
else
  echo "pandoc not found (expected either /tmp/pandoc-3.9/bin/pandoc or PATH)." >&2
  exit 1
fi

"${PANDOC}" "${SRC_MD}" \
  -o "${OUT_TEX}" \
  -s \
  -V documentclass=IEEEtran \
  -V classoption=journal

# Cleanup pass for IEEEtran compatibility and pdflatex-safe symbols.
perl -0pi -e '
  s/\\author\{Will Dale \\and Pond Science Institute\}/\\author{Will Dale}/g;
  s/\\begin\{center\}\\rule\{0\.5\\linewidth\}\{0\.5pt\}\\end\{center\}\n\n//g;
  s/\\section\{Abstract\}\\label\{abstract\}/\\begin{abstract}/g;
  s/\\section\{Introduction\}\\label\{introduction\}/\\end{abstract}\n\n\\section{Introduction}/g;
' "${OUT_TEX}"

perl -i -pe '
  s/×/\\(\\times\\)/g;
  s/Δ/\\(\\Delta\\)/g;
  s/κ/\\(\\kappa\\)/g;
  s/≤/\\(\\leq\\)/g;
  s/≥/\\(\\geq\\)/g;
  s/≈/\\(\\approx\\)/g;
  s/∈/\\(\\in\\)/g;
  s/Σ/\\(\\sum\\)/g;
  s/−/-/g;
  s/²/^2/g;
  s/μ\\_c/\\(\\mu_c\\)/g;
  s/ε\\_i/\\(\\varepsilon_i\\)/g;
  s/\(1 - \\\(\\sum\\\) p\^2\)/(1 - \\(\\sum p^2\\))/g;
' "${OUT_TEX}"

echo "Wrote ${OUT_TEX}"
