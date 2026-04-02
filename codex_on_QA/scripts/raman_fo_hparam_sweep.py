#!/usr/bin/env python3
"""
Hyperparameter sweep for Raman fundamental–overtone (FO) bcwin mapping.

Grid:
  - baseline window frac: [0.005, 0.01, 0.02]
  - smoothing: [off, on] (small moving average on corrected spectra)
  - boundary slack (cm^-1): [5.0, 10.0, 15.0]

Score:
  score = LogRegAcc + 0.2 * KMeansARI   (evaluated on QA-21 features)

Outputs:
  - codex_on_QA/out/raman_fo_v2_default.json  (best params + bench summary)
  - codex_on_QA/out/raman_qa_fundovt_bcwin_v2.csv
  - codex_on_QA/out/raman_qa_fundovt_bcwin_v2_csv_bench.json
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split

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


def bench_qa21(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # KMeans ARI @ k = #classes; LogReg accuracy
    k = len(np.unique(y))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    lab = km.fit_predict(X)
    ari = float(adjusted_rand_score(y, lab))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    return ari, acc


def build_qa21_features_from_be(BE: np.ndarray) -> np.ndarray:
    # BE: (n, 2) array of (b,e)
    from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector
    feats: List[np.ndarray] = []
    for b, e in BE:
        v, _ = qa_feature_vector(float(b), float(e), mode='qa21')
        feats.append(v)
    return np.vstack(feats)


def main() -> int:
    ensure_dir(OUT)
    classes = [d for d in sorted(os.listdir(SRC)) if os.path.isdir(os.path.join(SRC, d))]
    if not classes:
        print('No classes under', SRC)
        return 0
    paths: List[str] = []
    for c in classes:
        for fn in os.listdir(os.path.join(SRC, c)):
            if fn.lower().endswith('.txt'):
                paths.append(os.path.join(SRC, c, fn))
    # Parse + bounds
    specs: List[Tuple[str, np.ndarray, np.ndarray]] = []
    nu_min = float('inf'); nu_max = float('-inf')
    for p in paths:
        pr = parse_spectrum(p)
        if not pr:
            continue
        x, y = pr
        specs.append((p, x, y))
        nu_min = min(nu_min, float(x[0]))
        nu_max = max(nu_max, float(x[-1]))
    if not specs:
        print('No valid spectra parsed')
        return 0
    grid = np.linspace(nu_min, nu_max, 2048)

    # Hyperparam grid
    frac_list = [0.005, 0.01, 0.02]
    smooth_on = [False, True]
    slack_list = [5.0, 10.0, 15.0]

    best = {'score': -1.0}

    for frac in frac_list:
        # preprocess with baseline frac
        corrected: List[Tuple[str, np.ndarray]] = []
        labels: List[int] = []
        for p, x, y in specs:
            yi = np.interp(grid, x, y)
            base = baseline_ma(yi, frac=frac)
            yc = yi - base
            yc[yc < 0] = 0.0
            sc = float(yc.sum()) or 1.0
            yc = yc / sc
            corrected.append((p, yc))
            labels.append(classes.index(os.path.basename(os.path.dirname(p))))
        Ymean = np.mean(np.stack([yc for _, yc in corrected], axis=0), axis=0)
        W_L, W_F, W_S = dynamic_windows(grid, Ymean)
        for do_smooth in smooth_on:
            corr2: List[np.ndarray] = []
            for _, yc in corrected:
                if do_smooth:
                    yc2 = smooth_mavg(yc, 7)
                else:
                    yc2 = yc
                corr2.append(yc2)
            for slack in slack_list:
                # adjust windows
                W_F_adj = (W_F[0]-slack, W_F[1]+slack)
                W_S_adj = (W_S[0]-slack, W_S[1]+slack)
                # compute FO features (b: normalized nu_f; e: log(I_s/I_f))
                BE: List[Tuple[float,float]] = []
                for yc in corr2:
                    x = grid; y = yc
                    mF = (x >= W_F_adj[0]) & (x <= W_F_adj[1])
                    mS = (x >= W_S_adj[0]) & (x <= W_S_adj[1])
                    if mF.sum() == 0 or mS.sum() == 0:
                        BE.append((0.0, 0.0))
                        continue
                    idx_f = int(np.argmax(y[mF])); idx_s = int(np.argmax(y[mS]))
                    xF = x[mF]; yF = y[mF]
                    xS = x[mS]; yS = y[mS]
                    nu_f = float(xF[idx_f]); I_f = float(yF[idx_f])
                    nu_s = float(xS[idx_s]); I_s = float(yS[idx_s])
                    b = (nu_f - nu_min) / (nu_max - nu_min + 1e-9)
                    e = math.log((I_s + 1e-9) / (I_f + 1e-9))
                    BE.append((b, e))
                BE_arr = np.asarray(BE, dtype=float)
                X_qa21 = build_qa21_features_from_be(BE_arr)
                y = np.asarray(labels, dtype=int)
                ari, acc = bench_qa21(X_qa21, y)
                score = acc + 0.2 * ari
                if score > best['score']:
                    best = {
                        'score': score,
                        'baseline_frac': frac,
                        'smoothing': do_smooth,
                        'slack': slack,
                        'ari': ari,
                        'logreg_acc': acc,
                    }

    # Build final CSV with best params and bench all modes via qa_csv_bench.py
    ensure_dir(OUT)
    with open(os.path.join(OUT, 'raman_fo_v2_default.json'), 'w') as f:
        json.dump(best, f, indent=2)
    print('Selected FO v2 params:', best)

    # Rebuild features under best params and write CSV
    frac = best['baseline_frac']; do_smooth = best['smoothing']; slack = best['slack']
    corrected: List[Tuple[str, np.ndarray, int]] = []
    for p, x, y in specs:
        yi = np.interp(grid, x, y)
        base = baseline_ma(yi, frac=frac)
        yc = yi - base
        yc[yc < 0] = 0.0
        sc = float(yc.sum()) or 1.0
        yc = yc / sc
        if do_smooth:
            yc = smooth_mavg(yc, 7)
        corrected.append((p, yc, classes.index(os.path.basename(os.path.dirname(p)))))
    Ymean = np.mean(np.stack([yc for _, yc, _ in corrected], axis=0), axis=0)
    W_L, W_F, W_S = dynamic_windows(grid, Ymean)
    W_F_adj = (W_F[0]-slack, W_F[1]+slack)
    W_S_adj = (W_S[0]-slack, W_S[1]+slack)
    out_csv = os.path.join(OUT, 'raman_qa_fundovt_bcwin_v2.csv')
    with open(out_csv, 'w') as f:
        f.write('id,b,e,label\n')
        for p, yc, lab in corrected:
            x = grid; y = yc
            mF = (x >= W_F_adj[0]) & (x <= W_F_adj[1])
            mS = (x >= W_S_adj[0]) & (x <= W_S_adj[1])
            if mF.sum() == 0 or mS.sum() == 0:
                b = 0.0; e = 0.0
            else:
                idx_f = int(np.argmax(y[mF])); idx_s = int(np.argmax(y[mS]))
                xF = x[mF]; yF = y[mF]
                xS = x[mS]; yS = y[mS]
                nu_f = float(xF[idx_f]); I_f = float(yF[idx_f])
                nu_s = float(xS[idx_s]); I_s = float(yS[idx_s])
                b = (nu_f - nu_min) / (nu_max - nu_min + 1e-9)
                e = math.log((I_s + 1e-9) / (I_f + 1e-9))
            f.write(f"{p},{b:.6f},{e:.6f},{lab}\n")
    print('Wrote', out_csv)

    # Bench all modes via existing CSV bench
    from codex_on_QA.scripts.qa_csv_bench import load_csv, bench_all
    X, y, _ = load_csv(out_csv)
    res = bench_all(X, y, modes=('raw','qa21','qa27','qa83'))
    out_json = os.path.join(OUT, 'raman_qa_fundovt_bcwin_v2_csv_bench.json')
    with open(out_json, 'w') as f:
        json.dump({'csv': out_csv, 'n': int(len(y)), 'modes': res, 'best_params': best}, f, indent=2)
    print('Wrote', out_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

