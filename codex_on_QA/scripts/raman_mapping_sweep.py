#!/usr/bin/env python3
"""
Raman mapping sweep: generate QA-friendly (b,e) encodings and benchmark.

Encodings implemented per spectrum S(nu):
  - bandratio:  b = log(I_L / (I_L+I_S)), e = log(I_S / (I_L+I_S)),
                with band integrals over lattice (W_L) and stretch (W_S)
  - fundovt:    b = normalized nu_f (dominant in fingerprint W_F),
                e = log(I_s / I_f) (dominant intensities in W_S vs W_F)
  - fingerprint: b = normalized centroid in W_F, e = log(std in W_F)

Outputs CSVs under codex_on_QA/out:
  - raman_qa_bandratio.csv
  - raman_qa_fundovt.csv
  - raman_qa_fingerprint.csv

Also runs the CSV bench to produce *_csv_bench.json for each.
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


def collect_paths() -> Tuple[List[str], List[str]]:
    classes = [d for d in sorted(os.listdir(SRC)) if os.path.isdir(os.path.join(SRC, d))]
    paths: List[str] = []
    for c in classes:
        for fn in os.listdir(os.path.join(SRC, c)):
            if fn.lower().endswith('.txt'):
                paths.append(os.path.join(SRC, c, fn))
    return classes, paths


def band_integral(nu: np.ndarray, S: np.ndarray, w1: float, w2: float) -> float:
    mask = (nu >= w1) & (nu <= w2)
    if mask.sum() < 2:
        return 0.0
    # numpy.trapz is fine; SciPy not required. Add small floor for stability.
    val = float(np.trapz(S[mask], nu[mask]))
    if not math.isfinite(val):
        return 0.0
    return max(val, 0.0)


def detect_peaks(x: np.ndarray, y: np.ndarray) -> List[int]:
    idx = []
    n = len(y)
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            idx.append(i)
    idx.sort(key=lambda i: y[i], reverse=True)
    return idx


def write_csv(rows: List[Tuple[str, float, float, int]], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'b', 'e', 'label'])
        for r in rows:
            w.writerow([r[0], f'{r[1]:.6f}', f'{r[2]:.6f}', r[3]])
    print('Wrote', out_path, 'rows:', len(rows))


def run_bench(csv_path: str) -> None:
    # Import bench function and write JSON beside the CSV
    from codex_on_QA.scripts.qa_csv_bench import load_csv, bench_all
    import json
    X, y, _ids = load_csv(csv_path)
    results = bench_all(X, y, modes=('raw', 'qa21', 'qa27', 'qa83'))
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_json = os.path.join(OUT, f'{stem}_csv_bench.json')
    with open(out_json, 'w') as f:
        json.dump({'csv': csv_path, 'n': int(len(y)), 'modes': results}, f, indent=2)
    print('Wrote', out_json)


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
    # Find minima near expected boundaries for fingerprint (~800) and stretch (~2700)
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
    # Ensure ordering and margins
    t1 = max(lo + 50.0, min(t1, hi - 200.0))
    t2 = max(t1 + 200.0, min(t2, hi - 50.0))
    W_L = (lo, t1)
    W_F = (t1, t2)
    W_S = (t2, hi)
    return W_L, W_F, W_S


def peak_prominences(y: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    # Simple prominence: height minus max(nearest left/right minima)
    n = len(y)
    prom = np.zeros_like(idxs, dtype=float)
    for k, i in enumerate(idxs):
        i = int(i)
        # walk left to next rise
        j = i
        left_min = y[i]
        while j > 0 and y[j-1] <= y[j]:
            j -= 1
            if y[j] < left_min:
                left_min = y[j]
        # walk right to next rise
        j = i
        right_min = y[i]
        while j < n-1 and y[j+1] <= y[j]:
            j += 1
            if y[j] < right_min:
                right_min = y[j]
        ref = max(left_min, right_min)
        prom[k] = max(0.0, y[i] - ref)
    return prom


def find_peak_by_prominence(x: np.ndarray, y: np.ndarray, mask: np.ndarray, min_ratio: float = 0.02, smooth_win: int = 7) -> int:
    # Returns absolute index in x of the most prominent peak in masked window
    xm = x[mask]; ym = y[mask]
    if len(xm) < 3:
        # fallback: argmax
        idx_rel = int(np.argmax(ym)) if len(ym) else 0
        return int(np.where(mask)[0][0] + idx_rel) if len(ym) else 0
    # light smoothing
    if smooth_win > 1:
        kernel = np.ones(smooth_win, dtype=float) / smooth_win
        ysm = np.convolve(ym, kernel, mode='same')
    else:
        ysm = ym
    # local maxima
    locs = np.where((ysm[1:-1] > ysm[:-2]) & (ysm[1:-1] > ysm[2:]))[0] + 1
    if len(locs) == 0:
        idx_rel = int(np.argmax(ysm))
        return int(np.where(mask)[0][0] + idx_rel)
    prom = peak_prominences(ysm, locs)
    thresh = min_ratio * (float(ysm.max()) if ysm.size else 1.0)
    valid = np.where(prom >= thresh)[0]
    if len(valid) == 0:
        best_rel = int(locs[int(np.argmax(prom))])
    else:
        best_rel = int(locs[valid[int(np.argmax(prom[valid]))]])
    return int(np.where(mask)[0][0] + best_rel)


def main() -> int:
    classes, paths = collect_paths()
    if not paths:
        print('No Raman spectra found in', SRC)
        return 0
    # Dataset-wide nu bounds and uniform grid
    parsed_raw: List[Tuple[str, np.ndarray, np.ndarray]] = []
    nu_min = float('inf'); nu_max = float('-inf')
    for p in paths:
        sp = parse_spectrum(p)
        if not sp:
            continue
        x, y = sp
        parsed_raw.append((p, x, y))
        if x[0] < nu_min: nu_min = float(x[0])
        if x[-1] > nu_max: nu_max = float(x[-1])
    if not parsed_raw:
        print('No valid Raman spectra parsed')
        return 0
    grid = np.linspace(nu_min, nu_max, 2048)
    Ycorr_list: List[Tuple[str, np.ndarray]] = []
    # Build baseline-corrected, area-normalized spectra on a common grid
    for p, x, y in parsed_raw:
        yi = np.interp(grid, x, y)
        base = baseline_ma(yi, frac=0.05)
        yc = yi - base
        yc[yc < 0] = 0.0
        sc = yc.sum() or 1.0
        yc = yc / sc
        Ycorr_list.append((p, yc))
    # Compute dynamic windows from mean corrected spectrum
    Ymean = np.mean(np.stack([yc for _, yc in Ycorr_list], axis=0), axis=0)
    W_L, W_F, W_S = dynamic_windows(grid, Ymean)
    eps = 1e-9

    # A) bandratio
    rows_band: List[Tuple[str, float, float, int]] = []
    for p, yc in Ycorr_list:
        I_L = band_integral(grid, yc, *W_L)
        I_S = band_integral(grid, yc, *W_S)
        denom = I_L + I_S + eps
        b = math.log((I_L + eps) / denom)
        e = math.log((I_S + eps) / denom)
        lab = os.path.basename(os.path.dirname(p))
        rows_band.append((p, b, e, classes.index(lab)))
    band_csv = os.path.join(OUT, 'raman_qa_bandratio_bcwin.csv')
    write_csv(rows_band, band_csv)
    run_bench(band_csv)

    # B) fundamental-overtone
    rows_fo: List[Tuple[str, float, float, int]] = []
    for p, yc in Ycorr_list:
        x = grid; y = yc
        mF = (x >= W_F[0]) & (x <= W_F[1])
        mS = (x >= W_S[0]) & (x <= W_S[1])
        if mF.sum() == 0 or mS.sum() == 0:
            continue
        i_f_abs = find_peak_by_prominence(x, y, mF, min_ratio=0.02, smooth_win=7)
        i_s_abs = find_peak_by_prominence(x, y, mS, min_ratio=0.02, smooth_win=7)
        nu_f = float(x[i_f_abs]); I_f = float(y[i_f_abs])
        nu_s = float(x[i_s_abs]); I_s = float(y[i_s_abs])
        b = (nu_f - nu_min) / (nu_max - nu_min + eps)
        e = math.log((I_s + eps) / (I_f + eps))
        lab = os.path.basename(os.path.dirname(p))
        rows_fo.append((p, b, e, classes.index(lab)))
    fo_csv = os.path.join(OUT, 'raman_qa_fundovt_bcwin_prom.csv')
    write_csv(rows_fo, fo_csv)
    run_bench(fo_csv)

    # C) fingerprint centroid+sharpness
    rows_fp: List[Tuple[str, float, float, int]] = []
    for p, yc in Ycorr_list:
        x = grid; y = yc
        mF = (x >= W_F[0]) & (x <= W_F[1])
        if mF.sum() < 3:
            continue
        xF = x[mF]; yF = y[mF]
        total = float(yF.sum()) + eps
        w = yF / total
        mu = float(np.sum(w * xF))
        var = float(np.sum(w * (xF - mu) ** 2))
        sigma = math.sqrt(max(var, 0.0))
        b = (mu - W_F[0]) / (W_F[1] - W_F[0] + eps)
        e = math.log(sigma + eps)
        lab = os.path.basename(os.path.dirname(p))
        rows_fp.append((p, b, e, classes.index(lab)))
    fp_csv = os.path.join(OUT, 'raman_qa_fingerprint_bcwin.csv')
    write_csv(rows_fp, fp_csv)
    run_bench(fp_csv)

    # D) fundamental–overtone (frequency shift)
    rows_fo_shift: List[Tuple[str, float, float, int]] = []
    for p, yc in Ycorr_list:
        x = grid; y = yc
        mF = (x >= W_F[0]) & (x <= W_F[1])
        mS = (x >= W_S[0]) & (x <= W_S[1])
        if mF.sum() == 0 or mS.sum() == 0:
            continue
        i_f_abs = find_peak_by_prominence(x, y, mF, min_ratio=0.02, smooth_win=7)
        i_s_abs = find_peak_by_prominence(x, y, mS, min_ratio=0.02, smooth_win=7)
        nu_f = float(x[i_f_abs])
        nu_s = float(x[i_s_abs])
        b = (nu_f - nu_min) / (nu_max - nu_min + eps)
        e = (nu_s - nu_f) / (nu_max - nu_min + eps)
        lab = os.path.basename(os.path.dirname(p))
        rows_fo_shift.append((p, b, e, classes.index(lab)))
    fo_shift_csv = os.path.join(OUT, 'raman_qa_fundovt_shift_bcwin_prom.csv')
    write_csv(rows_fo_shift, fo_shift_csv)
    run_bench(fo_shift_csv)

    # E) fingerprint PCA2
    # Build sub-grid on fingerprint window for consistent PCA
    subN = 512
    fp_grid = np.linspace(W_F[0], W_F[1], subN)
    mats = []
    metas = []
    for p, yc in Ycorr_list:
        x = grid
        # take only fingerprint region, interpolate corrected spectrum
        yi = np.interp(fp_grid, x, yc)
        # z-score within fingerprint to remove scale
        mu = float(yi.mean()); sd = float(yi.std() or 1.0)
        yi = (yi - mu) / sd
        mats.append(yi.astype(np.float32))
        lab = os.path.basename(os.path.dirname(p))
        metas.append((p, classes.index(lab)))
    if mats:
        from sklearn.decomposition import PCA
        Xfp = np.vstack(mats)
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(Xfp)
        rows_fp_pca: List[Tuple[str, float, float, int]] = []
        for (p, lab), z in zip(metas, Z):
            b, e = float(z[0]), float(z[1])
            rows_fp_pca.append((p, b, e, lab))
        fp_pca_csv = os.path.join(OUT, 'raman_qa_fingerprint_pca2_bcwin.csv')
        write_csv(rows_fp_pca, fp_pca_csv)
        run_bench(fp_pca_csv)

    print('Raman mapping sweep completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
