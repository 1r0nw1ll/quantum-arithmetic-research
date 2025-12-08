#!/usr/bin/env python3
"""
Cluster-aware Raman band windows and QA encodings.

Pipeline:
  1) Parse spectra under qa_lab/qa_data/raman/*
  2) Preprocess (baseline MA, clamp, normalize, common grid)
  3) Compute coarse band integrals across whole grid
  4) KMeans cluster spectra in coarse-integral space (k clusters)
  5) For each cluster, derive dynamic windows (W_L,W_F,W_S) from its mean spectrum
  6) Recompute per-spectrum (b,e) using cluster-specific windows for:
       - bandratio (log I_L / (I_L+I_S), log I_S / (I_L+I_S))
       - FO intensity ratio (b = normalized nu_f in W_F, e = log(I_s/I_f))
       - FO frequency shift (b = normalized nu_f, e = (nu_s - nu_f) / (range))
  7) Write CSVs and bench via qa_csv_bench.py

Outputs (CSV + JSON under codex_on_QA/out):
  - raman_qa_bandratio_bcwin_cluster.csv + _csv_bench.json
  - raman_qa_fundovt_bcwin_cluster.csv + _csv_bench.json
  - raman_qa_fundovt_shift_bcwin_cluster.csv + _csv_bench.json
"""
from __future__ import annotations

import os
import csv
import math
from typing import List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

OUT = os.path.join('codex_on_QA', 'out')
SRC = os.path.join('qa_lab', 'qa_data', 'raman')


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


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
    order = np.argsort(X)
    return X[order], Y[order]


def smooth_mavg(y: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(y, kernel, mode='same')


def baseline_ma(y: np.ndarray, frac: float = 0.01) -> np.ndarray:
    k = max(5, int(frac * len(y)))
    if k % 2 == 0:
        k += 1
    return smooth_mavg(y, k)


def dynamic_windows(grid: np.ndarray, Ycorr_mean: np.ndarray) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
    g = grid
    ym = smooth_mavg(Ycorr_mean, max(5, len(g)//200))
    def find_min(a: float, b: float) -> Optional[float]:
        m = (g >= a) & (g <= b)
        if m.sum() < 5:
            return None
        idx = np.argmin(ym[m])
        return float(g[m][idx])
    t1 = find_min(550.0, 1100.0) or 800.0
    t2 = find_min(2400.0, 3200.0) or 2700.0
    lo = float(g[0]); hi = float(g[-1])
    t1 = max(lo + 50.0, min(t1, hi - 200.0))
    t2 = max(t1 + 200.0, min(t2, hi - 50.0))
    return (lo, t1), (t1, t2), (t2, hi)


def band_integral(nu: np.ndarray, S: np.ndarray, a: float, b: float) -> float:
    m = (nu >= a) & (nu <= b)
    if m.sum() < 2:
        return 0.0
    val = float(np.trapz(S[m], nu[m]))
    if not math.isfinite(val):
        return 0.0
    return max(val, 0.0)


def bench_csv(csv_path: str) -> None:
    # Use existing single-seg bench
    from codex_on_QA.scripts.qa_csv_bench import load_csv, bench_all
    import json
    X, y, _ = load_csv(csv_path)
    res = bench_all(X, y, modes=('raw','qa21','qa27','qa83'))
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_json = os.path.join(OUT, f'{stem}_csv_bench.json')
    with open(out_json, 'w') as f:
        json.dump({'csv': csv_path, 'n': int(len(y)), 'modes': res}, f, indent=2)
    print('Wrote', out_json)


def main() -> int:
    ensure_dir(OUT)
    classes = [d for d in sorted(os.listdir(SRC)) if os.path.isdir(os.path.join(SRC, d))]
    paths: List[str] = []
    for c in classes:
        for fn in os.listdir(os.path.join(SRC, c)):
            if fn.lower().endswith('.txt'):
                paths.append(os.path.join(SRC, c, fn))
    if not paths:
        print('No spectra found')
        return 0
    # Parse and prep
    specs: List[Tuple[str, np.ndarray, np.ndarray]] = []
    nu_min = float('inf'); nu_max = float('-inf')
    for p in paths:
        pr = parse_spectrum(p)
        if not pr:
            continue
        x, y = pr
        specs.append((p, x, y))
        nu_min = min(nu_min, float(x[0])); nu_max = max(nu_max, float(x[-1]))
    if not specs:
        print('No valid spectra parsed')
        return 0
    grid = np.linspace(nu_min, nu_max, 2048)
    corrected: List[Tuple[str, np.ndarray, int]] = []
    for p, x, y in specs:
        yi = np.interp(grid, x, y)
        yc = yi - baseline_ma(yi, frac=0.01)
        yc[yc < 0] = 0.0
        sc = float(yc.sum()) or 1.0
        yc = yc / sc
        corrected.append((p, yc, classes.index(os.path.basename(os.path.dirname(p)))))
    # Coarse integrals for clustering
    bins = 12
    edges = np.linspace(grid[0], grid[-1], bins+1)
    feats = []
    for _, yc, _ in corrected:
        vec = []
        for i in range(bins):
            vec.append(band_integral(grid, yc, edges[i], edges[i+1]))
        feats.append(vec)
    feats = np.asarray(feats, dtype=float)
    # Normalize per sample to emphasize relative band profiles
    row_sums = feats.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    feats = feats / row_sums
    # Cluster
    k = min(5, max(3, len(set([lab for _,_,lab in corrected])) // 3))
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_ids = km.fit_predict(feats)
    # Derive windows per cluster
    cluster_windows: List[Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]] = []
    for cid in range(k):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            # fallback to global mean
            Ymean = np.mean(np.stack([yc for _, yc, _ in corrected], axis=0), axis=0)
        else:
            Ymean = np.mean(np.stack([corrected[i][1] for i in idx], axis=0), axis=0)
        cluster_windows.append(dynamic_windows(grid, Ymean))
    # Compute encodings per spectrum with cluster windows
    band_rows: List[str] = []
    fo_rows: List[str] = []
    fishift_rows: List[str] = []
    for i, (p, yc, lab) in enumerate(corrected):
        cid = int(cluster_ids[i])
        W_L, W_F, W_S = cluster_windows[cid]
        # bandratio
        IL = band_integral(grid, yc, *W_L)
        IS = band_integral(grid, yc, *W_S)
        denom = IL + IS + 1e-9
        b_band = math.log((IL + 1e-9) / denom)
        e_band = math.log((IS + 1e-9) / denom)
        band_rows.append(f"{p},{b_band:.6f},{e_band:.6f},{lab}\n")
        # fundamental in fingerprint, overtone in stretch
        x = grid; y = yc
        mF = (x >= W_F[0]) & (x <= W_F[1])
        mS = (x >= W_S[0]) & (x <= W_S[1])
        if mF.sum() == 0 or mS.sum() == 0:
            b_fo = 0.0; e_fo = 0.0; e_shift = 0.0
        else:
            idx_f = int(np.argmax(y[mF])); idx_s = int(np.argmax(y[mS]))
            xF = x[mF]; yF = y[mF]
            xS = x[mS]; yS = y[mS]
            nu_f = float(xF[idx_f]); I_f = float(yF[idx_f])
            nu_s = float(xS[idx_s]); I_s = float(yS[idx_s])
            b_fo = (nu_f - grid[0]) / (grid[-1] - grid[0] + 1e-9)
            e_fo = math.log((I_s + 1e-9) / (I_f + 1e-9))
            e_shift = (nu_s - nu_f) / (grid[-1] - grid[0] + 1e-9)
        fo_rows.append(f"{p},{b_fo:.6f},{e_fo:.6f},{lab}\n")
        fishift_rows.append(f"{p},{b_fo:.6f},{e_shift:.6f},{lab}\n")
    # Write CSVs and bench
    band_csv = os.path.join(OUT, 'raman_qa_bandratio_bcwin_cluster.csv')
    with open(band_csv, 'w') as f:
        f.write('id,b,e,label\n')
        for row in band_rows:
            f.write(row)
    print('Wrote', band_csv)
    bench_csv(band_csv)

    fo_csv = os.path.join(OUT, 'raman_qa_fundovt_bcwin_cluster.csv')
    with open(fo_csv, 'w') as f:
        f.write('id,b,e,label\n')
        for row in fo_rows:
            f.write(row)
    print('Wrote', fo_csv)
    bench_csv(fo_csv)

    shift_csv = os.path.join(OUT, 'raman_qa_fundovt_shift_bcwin_cluster.csv')
    with open(shift_csv, 'w') as f:
        f.write('id,b,e,label\n')
        for row in fishift_rows:
            f.write(row)
    print('Wrote', shift_csv)
    bench_csv(shift_csv)

    print('Cluster-aware Raman windows completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

