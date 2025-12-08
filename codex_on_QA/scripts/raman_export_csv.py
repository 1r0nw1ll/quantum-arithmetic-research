#!/usr/bin/env python3
"""
Export Raman spectra under qa_lab/qa_data/raman/* into a simple QA-ready CSV:

Schema: id,b,e,label

- id: relative path to the spectrum file
- b: peak amplitude (max intensity, robustly normalized)
- e: spacing between the top two peaks (|w2 - w1|)
- label: integer material class index (alphabetical order of subdirs)

Notes:
- Parses both "Processed" and "Raw" .txt spectra (RRUFF-like headers).
- Skips files with fewer than 10 valid (wavenumber,intensity) pairs.
"""
from __future__ import annotations

import os
import csv
import math
from typing import List, Tuple, Optional

ROOT = os.path.join('qa_lab', 'qa_data', 'raman')
OUT_CSV = os.path.join('codex_on_QA', 'out', 'raman_qa.csv')


def parse_spectrum(path: str) -> Optional[List[Tuple[float, float]]]:
    data: List[Tuple[float, float]] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                # Expect lines like: "wavenumber, intensity"
                # Some files might use whitespace; be permissive
                parts = [p.strip() for p in s.replace('\t', ',').split(',') if p.strip()]
                if len(parts) != 2:
                    continue
                try:
                    x = float(parts[0]); y = float(parts[1])
                except ValueError:
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    data.append((x, y))
    except Exception:
        return None
    if len(data) < 10:
        return None
    return data


def peak_features(data: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    # b: robust peak height (max intensity scaled by median absolute deviation)
    # e: spacing between top two peak positions (in x)
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    n = len(ys)
    # Robust scaling for b
    sorted_y = sorted(ys)
    med = sorted_y[n // 2]
    mad = sorted([abs(y - med) for y in ys])[n // 2] or 1.0
    ymax = max(ys)
    b = (ymax - med) / (mad if mad != 0 else 1.0)
    # e: spacing between top-2 peaks in x by top intensities
    # Find indices of top-2 intensities with some separation
    idx_sorted = sorted(range(n), key=lambda i: ys[i], reverse=True)
    i1 = idx_sorted[0]
    w1 = xs[i1]
    w2 = None
    # pick next with distinct x (> 1 unit apart)
    for i in idx_sorted[1:]:
        if abs(xs[i] - w1) > 1.0:
            w2 = xs[i]
            break
    if w2 is None:
        return None
    e = abs(w2 - w1)
    return b, e


def main() -> int:
    classes = [d for d in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT, d))]
    label_map = {name: idx for idx, name in enumerate(classes)}
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_rows = 0
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'b', 'e', 'label'])
        for cname in classes:
            cdir = os.path.join(ROOT, cname)
            for fn in os.listdir(cdir):
                if not fn.lower().endswith('.txt'):
                    continue
                path = os.path.join(cdir, fn)
                data = parse_spectrum(path)
                if not data:
                    continue
                feats = peak_features(data)
                if not feats:
                    continue
                b, e = feats
                w.writerow([os.path.relpath(path), f'{b:.6f}', f'{e:.6f}', label_map[cname]])
                out_rows += 1
    print(f'Wrote {OUT_CSV} with {out_rows} rows across {len(classes)} classes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

