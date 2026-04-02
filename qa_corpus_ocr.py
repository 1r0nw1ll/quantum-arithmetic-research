#!/usr/bin/env python3
"""
qa_corpus_ocr.py — QA Corpus OCR Pipeline

Processes image-only PDFs from the OCR backlog.
Uses fitz (pymupdf) to render pages → PIL images → easyocr (CPU mode).

Priority order (from audit):
  1. QA-3   — no DOCX equivalent, unique content
  2. Pyth-1 — foundational Pythagoras vol 1
  3. Pyth-2 — foundational Pythagoras vol 2
  4. QA-Workbook
  5. 2nd editions (QA-Book1-2ed, QA-Book2-2ed, QA-Book3-2ed)

Usage:
  python qa_corpus_ocr.py --target qa3         # OCR QA-3 only
  python qa_corpus_ocr.py --target all         # all backlog files
  python qa_corpus_ocr.py --target qa3 --dpi 200   # faster, lower quality
  python qa_corpus_ocr.py --list               # show backlog
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

CORPUS_ROOT = Path("/home/player2/Desktop/files/quantum_pythagoras-text/quantum_pythagoras-text")
OUT_DIR     = Path("/home/player2/signal_experiments/qa_corpus_text")

# ── OCR backlog — priority order ──────────────────────────────────────────────

OCR_BACKLOG = [
    {
        "target_id": "qa3",
        "series": "QA-3",
        "rel_path": "QA-3-PRINT/QA-3_PDFs/XQA-3-All-Pages.pdf",
        "priority": 1,
        "note": "No DOCX equivalent — unique content",
    },
    {
        "target_id": "pyth1",
        "series": "Pyth-1",
        "rel_path": "Pyth-Vol-1-B1300-PRINT/Pyth-1-All-Pages.pdf",
        "priority": 2,
        "note": "Pythagoras vol 1 — empty DOCX",
    },
    {
        "target_id": "pyth2",
        "series": "Pyth-2",
        "rel_path": "Pyth-vol-2-B0023497-PRINT/00-Pyth-2-ALL-PAGES.pdf",
        "priority": 3,
        "note": "Pythagoras vol 2 — empty DOCX",
    },
    {
        "target_id": "workbook",
        "series": "QA-Workbook",
        "rel_path": "QA-Workbook.pdf",
        "priority": 4,
        "note": "No DOCX equivalent",
    },
    {
        "target_id": "qa1_2ed",
        "series": "QA-1",
        "rel_path": "QA-Book1-2ed-P020307-PRINT/01-QA-Book1-ALL-PAGES.pdf",
        "priority": 5,
        "note": "2nd edition — may have corrections vs DOCX",
    },
    {
        "target_id": "qa2_2ed",
        "series": "QA-2",
        "rel_path": "QA-Book2-2ed-P030307/00-ALL-PAGES.pdf",
        "priority": 6,
        "note": "2nd edition",
    },
    {
        "target_id": "qa3_2ed",
        "series": "QA-3",
        "rel_path": "QA-Book3-2ed-P040307/00-QA-P040307-ALL-PAGES.pdf",
        "priority": 7,
        "note": "2nd edition of QA-3",
    },
    {
        "target_id": "pyth3_small",
        "series": "Pyth-3",
        "rel_path": "Pyth-vol-3-B0023597-PRINT/Pyth-3-All-Pages-Smaller.pdf",
        "priority": 8,
        "note": "Pyth-3 all-pages — near-empty DOCX version",
    },
]


# ── OCR engine ────────────────────────────────────────────────────────────────

def check_deps() -> list[str]:
    missing = []
    try:
        import fitz
        # Verify OCR capability (pymupdf >= 1.24 has built-in tesseract OCR)
        if not hasattr(fitz.Page, 'get_textpage_ocr'):
            missing.append("pymupdf >= 1.24 (pip install --upgrade pymupdf --break-system-packages)")
    except ImportError:
        missing.append("pymupdf (pip install pymupdf --break-system-packages)")
    return missing


def ocr_pdf(pdf_path: Path, dpi: int = 200, lang: str = "eng",
            page_limit: Optional[int] = None, verbose: bool = True) -> str:
    """OCR a PDF page-by-page using pymupdf built-in OCR. Returns full text.

    pymupdf >= 1.24 bundles tesseract OCR — no system install required.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    n = len(doc)
    pages_to_do = list(range(n)) if page_limit is None else list(range(min(n, page_limit)))
    texts = []

    if verbose:
        print(f"  OCR: {pdf_path.name} ({n} pages, dpi={dpi})")

    for i in pages_to_do:
        t0 = time.time()
        page = doc[i]

        # Built-in tesseract OCR via pymupdf
        tp = page.get_textpage_ocr(language=lang, dpi=dpi, full=False)
        text = page.get_text(textpage=tp).strip()
        elapsed = time.time() - t0

        texts.append(f"\n<!-- page {i+1} -->\n{text}")

        if verbose and (i % 5 == 0 or i == pages_to_do[-1]):
            chars = len(text)
            print(f"    page {i+1:4d}/{n} — {chars:5d} chars — {elapsed:.1f}s")

    doc.close()
    return "\n".join(texts)


def run_ocr_target(entry: dict, out_dir: Path, dpi: int = 300,
                   page_limit: Optional[int] = None) -> dict:
    """OCR one backlog entry, write output markdown."""
    pdf_path = CORPUS_ROOT / entry["rel_path"]
    if not pdf_path.exists():
        return {"ok": False, "error": f"not found: {pdf_path}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{entry['series'].lower().replace('-', '_')}__ocr__{entry['target_id']}.md"
    out_path = out_dir / slug

    # Skip if already done
    if out_path.exists():
        existing_chars = len(out_path.read_text(encoding="utf-8"))
        if existing_chars > 500:
            print(f"  ✓ already extracted: {slug} ({existing_chars:,} chars)")
            return {"ok": True, "skipped": True, "path": str(out_path)}

    t_start = time.time()
    try:
        text = ocr_pdf(pdf_path, dpi=dpi, page_limit=page_limit)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    elapsed = time.time() - t_start
    char_count = len(text)

    header = f"""---
source: {entry['rel_path']}
series: {entry['series']}
method: tesseract_ocr
dpi: {dpi}
chars: {char_count}
ocr_time_sec: {elapsed:.0f}
extracted: {time.strftime('%Y-%m-%d')}
note: {entry['note']}
---

# {entry['series']} — OCR Extract ({entry['target_id']})

"""
    out_path.write_text(header + text, encoding="utf-8")

    print(f"  ✓ {entry['series']:12s} → {slug} ({char_count:,} chars, {elapsed:.0f}s)")
    return {"ok": True, "path": str(out_path), "chars": char_count, "elapsed": elapsed}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="QA Corpus OCR Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", choices=[e["target_id"] for e in OCR_BACKLOG] + ["all"],
                       help="Which file(s) to OCR")
    group.add_argument("--list", action="store_true", help="List backlog with priority")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Render DPI (200=fast/lower quality, 300=default, 400=high)")
    parser.add_argument("--page-limit", type=int, default=None,
                        help="Only OCR first N pages (for testing)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if args.list:
        print(f"\nOCR Backlog ({len(OCR_BACKLOG)} files):\n")
        for e in sorted(OCR_BACKLOG, key=lambda x: x["priority"]):
            pdf = CORPUS_ROOT / e["rel_path"]
            exists = "✓" if pdf.exists() else "✗"
            print(f"  {e['priority']}. [{e['target_id']:10s}] {exists} {e['series']:12s} — {e['note']}")
            print(f"               {e['rel_path']}")
        return

    # Check deps before starting
    missing = check_deps()
    if missing:
        print("ERROR: Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    if args.target == "all":
        targets = sorted(OCR_BACKLOG, key=lambda x: x["priority"])
    else:
        targets = [e for e in OCR_BACKLOG if e["target_id"] == args.target]

    print(f"QA Corpus OCR — {len(targets)} target(s), dpi={args.dpi}")
    if args.page_limit:
        print(f"  (page limit: {args.page_limit} pages per file — test mode)")
    print()

    results = []
    for entry in targets:
        result = run_ocr_target(entry, args.out_dir, args.dpi, args.page_limit)
        result["target_id"] = entry["target_id"]
        results.append(result)

    # Summary
    ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDone: {ok}/{len(results)} succeeded")

    # Write result log
    log_path = args.out_dir / "ocr_run_log.json"
    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text())
        except Exception:
            pass
    existing.extend(results)
    log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
