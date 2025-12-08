#!/usr/bin/env python3
"""
Export multi-segment Raman QA coordinates: per spectrum, split the fingerprint
band into three equal-energy subbands and compute local (b,e) per subband.

Output CSV schema: id,b1,e1,b2,e2,b3,e3,label

- Preprocessing: baseline correction (moving-average), clamp negatives,
  area normalize, resample to a common grid, dynamic windows.
- Local (b,e) per subband (centroid + log width) is QA-friendly and stable.

Run:
  PYTHONPATH=. python codex_on_QA/scripts/raman_multiseg_export.py
  PYTHONPATH=. python codex_on_QA/scripts/qa_csv_bench_multiseg.py \
    --csv codex_on_QA/out/raman_qa_fingerprint_multiseg.csv
"""
from __future__ import annotations

import os
import csv
import math
from typing import List, Tuple, Optional

import numpy as np

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


def baseline_ma(y: np.ndarray, frac: float = 0.05) -> np.ndarray:
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


def subbands_equal_energy(x: np.ndarray, y: np.ndarray, w1: float, w2: float) -> List[Tuple[float, float]]:
    m = (x >= w1) & (x <= w2)
    if m.sum() < 5:
        return [(w1, (w1+w2)/2), ((w1+w2)/2, w2)]
    xs = x[m]; ys = y[m]
    c = np.cumsum(ys)
    total = float(c[-1]) or 1.0
    t13 = total/3.0; t23 = 2*total/3.0
    i13 = int(np.searchsorted(c, t13))
    i23 = int(np.searchsorted(c, t23))
    i13 = min(max(i13, 1), len(xs)-2)
    i23 = min(max(i23, i13+1), len(xs)-1)
    b1 = (xs[0], xs[i13])
    b2 = (xs[i13], xs[i23])
    b3 = (xs[i23], xs[-1])
    return [b1, b2, b3]


def centroid_width(x: np.ndarray, y: np.ndarray, w1: float, w2: float) -> Tuple[float, float]:
    m = (x >= w1) & (x <= w2)
    xs = x[m]; ys = y[m]
    if m.sum() < 3 or float(ys.sum()) <= 0:
        mu = 0.5*(w1+w2); sigma = (w2-w1)/6.0
    else:
        w = ys / float(ys.sum())
        mu = float(np.sum(w * xs))
        var = float(np.sum(w * (xs - mu)**2))
        sigma = math.sqrt(max(var, 1e-12))
    # Normalize b to [0,1] within subband; e as log width
    b = (mu - w1) / (w2 - w1 + 1e-9)
    e = math.log(sigma + 1e-9)
    return b, e


def main() -> int:
    ensure_dir(OUT)
    classes = [d for d in sorted(os.listdir(SRC)) if os.path.isdir(os.path.join(SRC, d))]
    if not classes:
        print('No classes found under', SRC)
        return 0
    paths: List[str] = []
    for c in classes:
        for fn in os.listdir(os.path.join(SRC, c)):
            if fn.lower().endswith('.txt'):
                paths.append(os.path.join(SRC, c, fn))
    if not paths:
        print('No Raman spectra found in', SRC)
        return 0
    # Parse and find global grid bounds
    parsed: List[Tuple[str, np.ndarray, np.ndarray]] = []
    nu_min = float('inf'); nu_max = float('-inf')
    for p in paths:
        sp = parse_spectrum(p)
        if not sp:
            continue
        x, y = sp
        parsed.append((p, x, y))
        if x[0] < nu_min: nu_min = float(x[0])
        if x[-1] > nu_max: nu_max = float(x[-1])
    if not parsed:
        print('No valid spectra parsed')
        return 0
    grid = np.linspace(nu_min, nu_max, 2048)
    # Preprocess each spectrum to corrected, area-normalized on grid
    corrected: List[Tuple[str, np.ndarray]] = []
    for p, x, y in parsed:
        yi = np.interp(grid, x, y)
        base = baseline_ma(yi, frac=0.05)
        yc = yi - base
        yc[yc < 0] = 0.0
        sc = float(yc.sum()) or 1.0
        yc = yc / sc
        corrected.append((p, yc))
    # Dynamic windows from mean corrected spectrum
    Ymean = np.mean(np.stack([yc for _, yc in corrected], axis=0), axis=0)
    W_L, W_F, W_S = dynamic_windows(grid, Ymean)
    # Build multiseg (b,e) for each spectrum
    out_rows: List[Tuple[str,float,float,float,float,float,float,int]] = []
    for p, yc in corrected:
        cdir = os.path.basename(os.path.dirname(p))
        label = classes.index(cdir)
        # Find equal-energy subbands inside fingerprint
        subs = subbands_equal_energy(grid, yc, *W_F)
        vals: List[Tuple[float,float]] = []
        for (a,b) in subs:
            b_loc, e_loc = centroid_width(grid, yc, a, b)
            vals.append((b_loc, e_loc))
        # Pad to 3 segments if needed
        while len(vals) < 3:
            vals.append((0.5, math.log(1e-3)))
        b1,e1 = vals[0]
        b2,e2 = vals[1]
        b3,e3 = vals[2]
        out_rows.append((p, b1,e1,b2,e2,b3,e3, label))
    out_csv = os.path.join(OUT, 'raman_qa_fingerprint_multiseg.csv')
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','b1','e1','b2','e2','b3','e3','label'])
        for r in out_rows:
            w.writerow([r[0], f'{r[1]:.6f}', f'{r[2]:.6f}', f'{r[3]:.6f}', f'{r[4]:.6f}', f'{r[5]:.6f}', f'{r[6]:.6f}', r[7]])
    print('Wrote', out_csv, 'rows:', len(out_rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

