#!/usr/bin/env python3
"""
Extract (b,e) QA tuples from small image patches and emit a CSV suitable for
qa_csv_bench.py and graph construction.

Directory layout (expected):
  <data_dir>/class0/*.png
  <data_dir>/class1/*.png
  ...

b,e per patch are computed from a simple 2‑band split of the 2D FFT energy:
  - b = low‑frequency energy (within radius <= lowcut fraction)
  - e = high‑frequency energy (outside that radius)

Usage (example):
  PYTHONPATH=. python codex_on_QA/scripts/image_patch_csv.py \
    --data data/images_small \
    --out  codex_on_QA/out/image_patches.csv \
    --patch-size 16 --stride 8 --topk 25 --lowcut 0.25

CSV schema: id,b,e,label
  id    = "<relative_path>:<y>:<x>" for the patch origin
  label = integer class id assigned by folder enumeration order
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.image as mpimg


def list_images(root: Path) -> List[Tuple[Path, int]]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    classes = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            classes.append(entry)
    pairs: List[Tuple[Path, int]] = []
    for cid, cls in enumerate(classes):
        for p in sorted(cls.rglob('*')):
            if p.is_file() and p.suffix.lower() in exts:
                pairs.append((p, cid))
    return pairs


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img.astype(float)
    elif img.ndim == 3:
        # average RGB if present
        g = img[..., :3].mean(axis=2).astype(float)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    # normalize to [0,1]
    if g.max() > 1.0:
        g = g / 255.0
    return g


def patch_fft_b_e(patch: np.ndarray, lowcut: float) -> Tuple[float, float]:
    # remove DC bias
    Z = patch - float(patch.mean())
    F = np.fft.fft2(Z)
    S = np.abs(np.fft.fftshift(F)) ** 2  # power spectrum, centered
    h, w = S.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    ry = (yy - cy) / max(cy, 1)
    rx = (xx - cx) / max(cx, 1)
    r = np.sqrt(ry * ry + rx * rx)
    low_mask = (r <= lowcut)
    b = float(S[low_mask].sum())
    e = float(S[~low_mask].sum())
    return b, e


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract (b,e) from image patches into a CSV")
    ap.add_argument('--data', required=True, help='Root directory with class subfolders')
    ap.add_argument('--out', required=True, help='Output CSV path')
    ap.add_argument('--patch-size', type=int, default=16)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--topk', type=int, default=25, help='Keep top‑K energetic patches per image (by total energy b+e)')
    ap.add_argument('--lowcut', type=float, default=0.25, help='Low‑frequency cutoff (fraction of Nyquist radius)')
    args = ap.parse_args()

    root = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = list_images(root)
    if not images:
        raise SystemExit(f"No images found under {root}")

    rows: List[Tuple[str, float, float, int]] = []
    ps = args.patch_size; st = args.stride

    for ipath, label in images:
        try:
            img = mpimg.imread(str(ipath))
        except Exception as e:
            print(f"[warn] cannot read {ipath}: {e}")
            continue
        g = to_gray(img)
        H, W = g.shape
        patches = []
        for y in range(0, max(H - ps + 1, 0), st):
            for x in range(0, max(W - ps + 1, 0), st):
                patch = g[y:y+ps, x:x+ps]
                if patch.shape != (ps, ps):
                    continue
                b, e = patch_fft_b_e(patch, args.lowcut)
                energy = b + e
                pid = f"{ipath.relative_to(root)}:{y}:{x}"
                patches.append((energy, pid, b, e, label))
        if not patches:
            continue
        # pick top‑K by energy
        patches.sort(key=lambda t: t[0], reverse=True)
        keep = patches[: args.topk] if args.topk > 0 else patches
        for _, pid, b, e, lab in keep:
            rows.append((pid, b, e, lab))

    # write CSV
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id','b','e','label'])
        for rid, b, e, lab in rows:
            w.writerow([rid, f"{b:.9f}", f"{e:.9f}", lab])
    print(f"Wrote {out_path} rows={len(rows)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

