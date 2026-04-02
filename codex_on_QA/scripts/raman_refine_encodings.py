#!/usr/bin/env python3
"""
Refine Raman encodings and export QA-ready CSVs with improved (b,e):

Encodings produced:
  - peaks3: (b,e) = (nu2-nu1, nu3-nu2) from top-3 peaks
  - centroid: b = centroid of top-20% intensity region; e = mean spacing between peak x's in that region
  - pca2: (b,e) = (PC1, PC2) from PCA on resampled spectra (global 1024-point grid)

Outputs:
  codex_on_QA/out/raman_qa_peaks3.csv
  codex_on_QA/out/raman_qa_centroid.csv
  codex_on_QA/out/raman_qa_pca2.csv

Then you can bench with qa_csv_bench.py and update master summaries.
"""
from __future__ import annotations

import os
import csv
import math
from typing import List, Tuple, Optional, Dict

import numpy as np

ROOT = os.path.join('qa_lab', 'qa_data', 'raman')
OUTDIR = os.path.join('codex_on_QA', 'out')


def parse_spectrum(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    xs: List[float] = []
    ys: List[float] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = [p.strip() for p in s.replace('\t', ',').split(',') if p.strip()]
                if len(parts) != 2:
                    continue
                try:
                    x = float(parts[0]); y = float(parts[1])
                except ValueError:
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x); ys.append(y)
    except Exception:
        return None
    if len(xs) < 10:
        return None
    X = np.asarray(xs); Y = np.asarray(ys)
    # ensure sorted by x
    order = np.argsort(X)
    return X[order], Y[order]


def detect_peaks(x: np.ndarray, y: np.ndarray) -> List[int]:
    # Simple local maxima
    idx = []
    n = len(y)
    if n < 3:
        return idx
    for i in range(1, n-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            idx.append(i)
    # Sort by height descending
    idx.sort(key=lambda i: y[i], reverse=True)
    return idx


def enc_peaks3(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    idx = detect_peaks(x, y)
    if len(idx) < 3:
        return None
    p = idx[:3]
    p.sort(key=lambda i: x[i])
    w1, w2, w3 = x[p[0]], x[p[1]], x[p[2]]
    b = abs(w2 - w1)
    e = abs(w3 - w2)
    return b, e


def enc_centroid(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    # Top-20% intensity region
    q = np.quantile(y, 0.8)
    mask = y >= q
    if mask.sum() < 3:
        return None
    xs = x[mask]; ys = y[mask]
    b = float((xs * ys).sum() / ys.sum())
    # peak spacings within region
    idx = detect_peaks(xs, ys)
    if len(idx) >= 2:
        w = np.sort(xs[idx])
        diffs = np.diff(w)
        e = float(np.mean(np.abs(diffs))) if len(diffs) > 0 else float(np.std(xs))
    else:
        e = float(np.std(xs))
    return b, e


def collect_paths() -> Tuple[List[str], Dict[str, int]]:
    classes = [d for d in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT, d))]
    cls2id = {c: i for i, c in enumerate(classes)}
    paths = []
    for c in classes:
        for fn in os.listdir(os.path.join(ROOT, c)):
            if fn.lower().endswith('.txt'):
                paths.append(os.path.join(ROOT, c, fn))
    return paths, cls2id


def write_csv(rows: List[Tuple[str, float, float, int]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','b','e','label'])
        for r in rows:
            w.writerow([r[0], f'{r[1]:.6f}', f'{r[2]:.6f}', r[3]])
    print('Wrote', out_path, 'rows:', len(rows))


def pca2_encode() -> None:
    # Build global grid and PCA
    paths, cls2id = collect_paths()
    samples = []
    metas = []
    min_x = float('inf'); max_x = float('-inf')
    parsed: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for p in paths:
        sp = parse_spectrum(p)
        if not sp:
            continue
        x,y = sp
        parsed.append((p,x,y))
        if x[0] < min_x: min_x = float(x[0])
        if x[-1] > max_x: max_x = float(x[-1])
    if not parsed:
        return
    grid = np.linspace(min_x, max_x, 1024)
    for p,x,y in parsed:
        yi = np.interp(grid, x, y)
        # Standardize per spectrum to reduce scale effect
        mu = yi.mean(); sd = yi.std() or 1.0
        yi = (yi - mu) / sd
        samples.append(yi.astype(np.float32))
        cls = os.path.basename(os.path.dirname(p))
        metas.append((p, cls2id.get(cls, 0)))
    X = np.vstack(samples)
    # Fit PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    rows = []
    for (p, lab), z in zip(metas, Z):
        b, e = float(z[0]), float(z[1])
        rows.append((p, b, e, lab))
    write_csv(rows, os.path.join(OUTDIR, 'raman_qa_pca2.csv'))


def peaks3_encode() -> None:
    paths, cls2id = collect_paths()
    rows: List[Tuple[str,float,float,int]] = []
    for p in paths:
        sp = parse_spectrum(p)
        if not sp:
            continue
        x,y = sp
        enc = enc_peaks3(x,y)
        if not enc:
            continue
        b,e = enc
        cls = os.path.basename(os.path.dirname(p))
        rows.append((p, b, e, cls2id.get(cls, 0)))
    write_csv(rows, os.path.join(OUTDIR, 'raman_qa_peaks3.csv'))


def centroid_encode() -> None:
    paths, cls2id = collect_paths()
    rows: List[Tuple[str,float,float,int]] = []
    for p in paths:
        sp = parse_spectrum(p)
        if not sp:
            continue
        x,y = sp
        enc = enc_centroid(x,y)
        if not enc:
            continue
        b,e = enc
        cls = os.path.basename(os.path.dirname(p))
        rows.append((p, b, e, cls2id.get(cls, 0)))
    write_csv(rows, os.path.join(OUTDIR, 'raman_qa_centroid.csv'))


def main() -> int:
    peaks3_encode()
    centroid_encode()
    # PCA can be heavier; keep last to avoid partial writes if interrupted
    pca2_encode()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

