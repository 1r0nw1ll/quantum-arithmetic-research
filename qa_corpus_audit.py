#!/usr/bin/env python3
"""
qa_corpus_audit.py — QA Corpus Triage Pass

Inventories all PDF and DOCX files in the QA/Pythagoras corpus.
For each file:
  - PDF: detect text layer presence, estimate text density, classify OCR difficulty
  - DOCX: extract paragraph count, char count, classify extractability
  - Recommend extraction strategy per file

Output: qa_corpus_audit_report.json + printed summary table.

Usage:
  python qa_corpus_audit.py
  python qa_corpus_audit.py --corpus-dir /path/to/corpus
  python qa_corpus_audit.py --sample-pages 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────

CORPUS_ROOT = Path("/home/player2/Desktop/files/quantum_pythagoras-text/quantum_pythagoras-text")

# Text density thresholds (chars per page)
TEXT_DENSE_THRESH   = 500   # pages with >500 chars → real text layer
TEXT_SPARSE_THRESH  = 50    # pages with 50–500 chars → partial / degraded text
# OCR difficulty ratings
OCR_EASY   = "easy"    # DOCX available OR PDF has dense text
OCR_MEDIUM = "medium"  # PDF has sparse/partial text
OCR_HARD   = "hard"    # PDF is image-only (no text layer)
OCR_SKIP   = "skip"    # Cover pages, tiny files, non-content

# Book series classification — extracted from path/filename patterns
SERIES_PATTERNS = {
    "QA-1":        r"QA-1|QA.Book1|QA.1.All",
    "QA-2":        r"QA-2|QA.Book2|QA.2.All|QA.2_ALL",
    "QA-3":        r"QA-3|QA.Book3|XQA-3",
    "QA-4":        r"QA.4|Books.3..4|Vol.II",
    "QA-Workbook": r"Workbook",
    "Quadrature":  r"Quadrature",
    "Pyth-1":      r"Pyth.Vol.1|Pyth.1.All|Pythagoras.vol.1",
    "Pyth-2":      r"Pyth.vol.2|Pyth.2.ALL|Pythagoras.vol.2",
    "Pyth-3":      r"Pyth.vol.3|Pyth.3.All|Pythagoras.vol.3|Pyth.3.Ennea",
    "Cover":       r"cover|Cover|COVER",
    "Preface":     r"Preface|preface",
}


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class FileAudit:
    rel_path: str
    file_type: str          # "pdf" | "docx" | "rtf" | "other"
    size_kb: float
    series: str             # book series label
    page_count: int = 0
    sampled_pages: int = 0
    total_chars: int = 0
    mean_chars_per_page: float = 0.0
    dense_pages: int = 0    # pages with >TEXT_DENSE_THRESH chars
    sparse_pages: int = 0   # pages with TEXT_SPARSE_THRESH–TEXT_DENSE_THRESH chars
    empty_pages: int = 0    # pages with <TEXT_SPARSE_THRESH chars
    has_text_layer: bool = False
    text_layer_quality: str = ""  # "good" | "partial" | "none"
    docx_paragraphs: int = 0
    docx_chars: int = 0
    docx_extractable: bool = False
    ocr_difficulty: str = ""
    recommended_strategy: str = ""
    notes: str = ""
    errors: list = field(default_factory=list)


# ── series detection ──────────────────────────────────────────────────────────

def detect_series(path: Path) -> str:
    name = path.name + " " + str(path.parent)
    for series, pattern in SERIES_PATTERNS.items():
        if re.search(pattern, name, re.IGNORECASE):
            return series
    return "unknown"


# ── PDF audit ─────────────────────────────────────────────────────────────────

def audit_pdf(path: Path, sample_pages: int = 5) -> dict:
    """Returns dict of pdf-specific metrics."""
    result = {
        "page_count": 0,
        "sampled_pages": 0,
        "total_chars": 0,
        "mean_chars_per_page": 0.0,
        "dense_pages": 0,
        "sparse_pages": 0,
        "empty_pages": 0,
        "has_text_layer": False,
        "text_layer_quality": "none",
        "errors": [],
    }
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        result["page_count"] = len(doc)

        # Sample: first N, middle N, last N pages
        n = len(doc)
        if n == 0:
            doc.close()
            return result

        indices = set()
        step = max(1, n // sample_pages)
        for i in range(0, n, step):
            indices.add(i)
        if n > 1:
            indices.add(n - 1)
        indices = sorted(indices)[:sample_pages * 3]  # cap at 3x sample to avoid huge docs
        result["sampled_pages"] = len(indices)

        page_chars = []
        for i in indices:
            page = doc[i]
            text = page.get_text("text")
            chars = len(text.strip())
            page_chars.append(chars)
            if chars >= TEXT_DENSE_THRESH:
                result["dense_pages"] += 1
            elif chars >= TEXT_SPARSE_THRESH:
                result["sparse_pages"] += 1
            else:
                result["empty_pages"] += 1

        doc.close()

        total = sum(page_chars)
        result["total_chars"] = total
        result["mean_chars_per_page"] = total / len(page_chars) if page_chars else 0.0
        result["has_text_layer"] = result["dense_pages"] > 0 or result["sparse_pages"] > 0

        # Quality classification
        dense_ratio = result["dense_pages"] / len(page_chars) if page_chars else 0
        if dense_ratio >= 0.5:
            result["text_layer_quality"] = "good"
        elif result["has_text_layer"]:
            result["text_layer_quality"] = "partial"
        else:
            result["text_layer_quality"] = "none"

    except Exception as e:
        result["errors"].append(f"pdf_audit_error: {e}")

    return result


# ── DOCX audit ────────────────────────────────────────────────────────────────

def audit_docx(path: Path) -> dict:
    result = {
        "docx_paragraphs": 0,
        "docx_chars": 0,
        "docx_extractable": False,
        "errors": [],
    }
    try:
        import docx as python_docx
        doc = python_docx.Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        result["docx_paragraphs"] = len(paras)
        result["docx_chars"] = sum(len(p) for p in paras)
        result["docx_extractable"] = result["docx_chars"] > 100
    except Exception as e:
        result["errors"].append(f"docx_audit_error: {e}")
    return result


# ── strategy recommendation ───────────────────────────────────────────────────

def recommend_strategy(audit: FileAudit) -> tuple[str, str]:
    """Returns (ocr_difficulty, recommended_strategy)."""

    series = audit.series

    # Skip covers and tiny files
    if series == "Cover" or audit.size_kb < 10:
        return OCR_SKIP, "skip — cover/tiny"

    if audit.file_type == "docx":
        if audit.docx_extractable:
            return OCR_EASY, "extract_docx — python-docx, no OCR needed"
        else:
            return OCR_MEDIUM, "docx_corrupt_or_empty — try RTF or paired PDF"

    if audit.file_type == "pdf":
        # Check if paired DOCX exists
        paired_docx = audit.rel_path.replace(".pdf", ".docx")
        has_paired = False  # will be filled in post-processing

        quality = audit.text_layer_quality
        if quality == "good":
            return OCR_EASY, "extract_pdf_text — pymupdf text layer sufficient"
        elif quality == "partial":
            return OCR_MEDIUM, "extract_pdf_partial — pymupdf + flag low-density pages for OCR"
        else:
            return OCR_HARD, "ocr_required — image-only PDF, use tesseract or cloud OCR"

    return OCR_MEDIUM, "manual_review — unknown file type"


# ── main audit ────────────────────────────────────────────────────────────────

def run_audit(corpus_dir: Path, sample_pages: int = 5) -> list[FileAudit]:
    audits: list[FileAudit] = []

    # Collect all relevant files (skip macOS resource forks starting with ._)
    pdf_files  = [p for p in corpus_dir.rglob("*.pdf")  if not p.name.startswith("._")]
    docx_files = [p for p in corpus_dir.rglob("*.docx") if not p.name.startswith("._")]
    rtf_files  = [p for p in corpus_dir.rglob("*.rtf")  if not p.name.startswith("._")]

    all_files = [(p, "pdf") for p in pdf_files] + \
                [(p, "docx") for p in docx_files] + \
                [(p, "rtf") for p in rtf_files]
    all_files.sort(key=lambda x: str(x[0]))

    print(f"Found {len(pdf_files)} PDFs, {len(docx_files)} DOCXs, {len(rtf_files)} RTFs")
    print(f"Auditing {len(all_files)} files...\n")

    for path, ftype in all_files:
        rel = str(path.relative_to(corpus_dir))
        size_kb = path.stat().st_size / 1024
        series = detect_series(path)

        audit = FileAudit(
            rel_path=rel,
            file_type=ftype,
            size_kb=round(size_kb, 1),
            series=series,
        )

        if ftype == "pdf":
            metrics = audit_pdf(path, sample_pages)
            audit.page_count = metrics["page_count"]
            audit.sampled_pages = metrics["sampled_pages"]
            audit.total_chars = metrics["total_chars"]
            audit.mean_chars_per_page = round(metrics["mean_chars_per_page"], 1)
            audit.dense_pages = metrics["dense_pages"]
            audit.sparse_pages = metrics["sparse_pages"]
            audit.empty_pages = metrics["empty_pages"]
            audit.has_text_layer = metrics["has_text_layer"]
            audit.text_layer_quality = metrics["text_layer_quality"]
            audit.errors.extend(metrics["errors"])

        elif ftype == "docx":
            metrics = audit_docx(path)
            audit.docx_paragraphs = metrics["docx_paragraphs"]
            audit.docx_chars = metrics["docx_chars"]
            audit.docx_extractable = metrics["docx_extractable"]
            audit.errors.extend(metrics["errors"])

        elif ftype == "rtf":
            audit.notes = "RTF — use python-rtf or pandoc for extraction"

        diff, strategy = recommend_strategy(audit)
        audit.ocr_difficulty = diff
        audit.recommended_strategy = strategy

        status = "✓" if diff in (OCR_EASY, OCR_SKIP) else ("~" if diff == OCR_MEDIUM else "✗")
        chars_info = ""
        if ftype == "pdf":
            chars_info = f"  {audit.page_count}pp, {audit.text_layer_quality} text"
        elif ftype == "docx":
            chars_info = f"  {audit.docx_chars:,} chars"
        print(f"  {status} [{series:12s}] {ftype.upper()} {path.name[:50]}{chars_info}")

        audits.append(audit)

    return audits


# ── summary tables ────────────────────────────────────────────────────────────

def print_summary(audits: list[FileAudit]) -> None:
    print("\n" + "="*70)
    print("CORPUS AUDIT SUMMARY")
    print("="*70)

    by_diff: dict[str, list[FileAudit]] = {OCR_EASY: [], OCR_MEDIUM: [], OCR_HARD: [], OCR_SKIP: []}
    for a in audits:
        by_diff.setdefault(a.ocr_difficulty, []).append(a)

    print(f"\n  EASY   (extract directly):  {len(by_diff[OCR_EASY])} files")
    print(f"  MEDIUM (partial/needs fix):  {len(by_diff[OCR_MEDIUM])} files")
    print(f"  HARD   (OCR required):       {len(by_diff[OCR_HARD])} files")
    print(f"  SKIP   (covers/tiny):        {len(by_diff[OCR_SKIP])} files")

    # Primary books (not covers, not duplicates in unsorted/)
    primary = [a for a in audits if "unsorted" not in a.rel_path.lower()
               and a.ocr_difficulty != OCR_SKIP]

    print(f"\n{'─'*70}")
    print("PRIMARY BOOKS (excl. covers/unsorted):")
    print(f"{'─'*70}")
    print(f"  {'SERIES':<15} {'TYPE':<6} {'STRATEGY':<50} FILE")

    for a in sorted(primary, key=lambda x: (x.series, x.file_type)):
        fname = Path(a.rel_path).name[:40]
        print(f"  {a.series:<15} {a.file_type.upper():<6} {a.recommended_strategy[:48]:<50} {fname}")

    print(f"\n{'─'*70}")
    print("HARD OCR FILES (need tesseract/cloud):")
    print(f"{'─'*70}")
    for a in by_diff[OCR_HARD]:
        print(f"  [{a.series}] {a.rel_path}")

    print(f"\n{'─'*70}")
    print("RECOMMENDED EXTRACTION ORDER:")
    print(f"{'─'*70}")
    order = [
        "1. Extract all DOCX files with python-docx (immediate, lossless)",
        "2. Extract good-text PDFs with pymupdf (fast, no OCR needed)",
        "3. Partial-text PDFs: extract + flag low-density pages",
        "4. Image-only PDFs: run tesseract (slow — do last or use cloud OCR)",
        "5. RTF: convert with pandoc → markdown",
    ]
    for line in order:
        print(f"  {line}")


# ── output ────────────────────────────────────────────────────────────────────

def write_report(audits: list[FileAudit], out_path: Path) -> None:
    data = {
        "total_files": len(audits),
        "by_difficulty": {
            "easy":   sum(1 for a in audits if a.ocr_difficulty == OCR_EASY),
            "medium": sum(1 for a in audits if a.ocr_difficulty == OCR_MEDIUM),
            "hard":   sum(1 for a in audits if a.ocr_difficulty == OCR_HARD),
            "skip":   sum(1 for a in audits if a.ocr_difficulty == OCR_SKIP),
        },
        "files": [asdict(a) for a in audits],
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nReport written → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="QA Corpus Triage Audit")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_ROOT,
                        help="Root directory of QA/Pythagoras corpus")
    parser.add_argument("--sample-pages", type=int, default=5,
                        help="Number of pages to sample per PDF (default: 5)")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/player2/signal_experiments/qa_corpus_audit_report.json"),
                        help="Output JSON report path")
    args = parser.parse_args(argv)

    if not args.corpus_dir.exists():
        print(f"ERROR: corpus directory not found: {args.corpus_dir}")
        sys.exit(1)

    print(f"QA Corpus Audit — {args.corpus_dir}")
    print(f"Sample pages per PDF: {args.sample_pages}\n")

    audits = run_audit(args.corpus_dir, args.sample_pages)
    print_summary(audits)
    write_report(audits, args.out)


if __name__ == "__main__":
    main()
