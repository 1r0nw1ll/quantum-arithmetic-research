#!/usr/bin/env python3
"""
Generate two deliberately QA-hostile synthetic datasets and benchmark them
with the existing QA CSV pipeline to expose limits of single-tuple QA encodings.

Datasets:
  1) three_factor: label depends on 3 independent latent factors via XOR; (b,e)
     only encode 2 of them, so any QA mapping on (b,e) cannot recover the third.
  2) symmetry_break: label depends on x-translation; (b,e) are translation-
     invariant w.r.t. x (constructed from y only), so QA is blind by design.

Outputs:
  - codex_on_QA/out/adversarial_three_factor.csv + _csv_bench.json
  - codex_on_QA/out/adversarial_symbreak.csv    + _csv_bench.json
"""
from __future__ import annotations

import os
import csv
import json
from typing import Tuple

import numpy as np

OUT = os.path.join('codex_on_QA', 'out')
os.makedirs(OUT, exist_ok=True)


def bench_csv(csv_path: str) -> None:
    from codex_on_QA.scripts.qa_csv_bench import load_csv, bench_all
    X, y, _ = load_csv(csv_path)
    res = bench_all(X, y, modes=('raw','qa21','qa27','qa83'))
    out_json = os.path.splitext(csv_path)[0] + '_csv_bench.json'
    with open(out_json, 'w') as f:
        json.dump({'csv': csv_path, 'n': int(len(y)), 'modes': res}, f, indent=2)
    print('Wrote', out_json)


def gen_three_factor(n: int = 4000, seed: int = 0) -> Tuple[str, int]:
    rng = np.random.RandomState(seed)
    z1 = rng.randn(n)
    z2 = rng.randn(n)
    z3 = rng.randn(n)
    # label = XOR(signs of z1, z2, z3), map {-1,+1} -> {0,1}
    s1 = (z1 > 0).astype(int)
    s2 = (z2 > 0).astype(int)
    s3 = (z3 > 0).astype(int)
    y = (s1 ^ s2) ^ s3
    # (b,e) encode only z1 and z2 (nonlinear squashing)
    b = np.tanh(z1)
    e = np.tanh(z2)
    path = os.path.join(OUT, 'adversarial_three_factor.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','b','e','label'])
        for i in range(n):
            w.writerow([f'tf_{i}', f'{b[i]:.6f}', f'{e[i]:.6f}', int(y[i])])
    print('Wrote', path)
    return path, n


def gen_three_factor_multi(n: int = 4000, seed: int = 0) -> Tuple[str, int]:
    # Two QA tuples per sample: (u,v) and (w, 0.0)
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, size=n)
    v = rng.uniform(-1, 1, size=n)
    w = rng.uniform(-1, 1, size=n)
    def sign(x):
        return np.where(x >= 0, 1, -1)
    s_u = sign(u); s_v = sign(v); s_w = sign(w)
    pos = ((s_u == 1).astype(int) + (s_v == 1).astype(int) + (s_w == 1).astype(int))
    y = (pos % 2).astype(int)
    path = os.path.join(OUT, 'adversarial_three_factor_multi.csv')
    with open(path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['id','b1','e1','b2','e2','label'])
        for i in range(n):
            wtr.writerow([f'tfm_{i}', f'{u[i]:.6f}', f'{v[i]:.6f}', f'{w[i]:.6f}', f'{0.0:.6f}', int(y[i])])
    print('Wrote', path)
    return path, n


def gen_symbreak(n: int = 4000, seed: int = 1) -> Tuple[str, int]:
    rng = np.random.RandomState(seed)
    # Two classes differ in x-mean; y identical.
    n2 = n // 2
    x1 = rng.randn(n2) - 2.0  # class 0 shifted left
    x2 = rng.randn(n - n2) + 2.0  # class 1 shifted right
    y1 = rng.randn(n2)
    y2 = rng.randn(n - n2)
    # (b,e) are translation-invariant w.r.t x: constructed from y only
    b = np.concatenate([np.abs(y1), np.abs(y2)], axis=0)
    e = np.concatenate([y1**2, y2**2], axis=0)
    labels = np.concatenate([np.zeros(n2, dtype=int), np.ones(n - n2, dtype=int)], axis=0)
    path = os.path.join(OUT, 'adversarial_symbreak.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','b','e','label'])
        for i in range(n):
            w.writerow([f'sb_{i}', f'{b[i]:.6f}', f'{e[i]:.6f}', int(labels[i])])
    print('Wrote', path)
    return path, n


def main() -> int:
    p1, _ = gen_three_factor()
    bench_csv(p1)
    p1m, _ = gen_three_factor_multi()
    # use autotuple bench for multi‑pair CSV
    from codex_on_QA.scripts.qa_csv_bench_autotuple import OUTDIR as _OUT
    os.environ['PYTHONPATH'] = '.'
    import subprocess, sys
    subprocess.run([sys.executable, 'codex_on_QA/scripts/qa_csv_bench_autotuple.py', '--csv', p1m], check=True)
    p2, _ = gen_symbreak()
    bench_csv(p2)
    print('Adversarial datasets generated and benchmarked.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
