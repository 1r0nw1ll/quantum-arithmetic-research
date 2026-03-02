#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FIG_SCRIPT="${ROOT_DIR}/generate_paper_figures_locality.py"
TEX_BASENAME="locality_dominance_paper"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH." >&2
  exit 1
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found in PATH." >&2
  exit 1
fi

if ! command -v bibtex >/dev/null 2>&1; then
  echo "bibtex not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${FIG_SCRIPT}" ]]; then
  echo "Missing figure generator script: ${FIG_SCRIPT}" >&2
  exit 1
fi

echo "[1/2] Regenerating figures into: ${PAPER_DIR}"
python3 "${FIG_SCRIPT}" --outdir "${PAPER_DIR}"

echo "[2/2] Rebuilding PDF (pdflatex + bibtex)"
cd "${PAPER_DIR}"
pdflatex -interaction=nonstopmode -halt-on-error "${TEX_BASENAME}.tex" >/tmp/locality_pdflatex_1.log
bibtex "${TEX_BASENAME}" >/tmp/locality_bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error "${TEX_BASENAME}.tex" >/tmp/locality_pdflatex_2.log
pdflatex -interaction=nonstopmode -halt-on-error "${TEX_BASENAME}.tex" >/tmp/locality_pdflatex_3.log

echo "Wrote: ${PAPER_DIR}/${TEX_BASENAME}.pdf"
