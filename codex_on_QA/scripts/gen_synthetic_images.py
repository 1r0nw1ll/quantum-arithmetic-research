#!/usr/bin/env python3
"""
Generate a tiny two-class image dataset under codex_on_QA/data/images_small
for exercising the image patch QA pipeline.

Classes:
  - circles: filled disks at random positions/radii
  - squares: filled squares at random positions/sizes

Usage:
  python codex_on_QA/scripts/gen_synthetic_images.py --n 40 --size 96

This will create ~n images per class and save PNGs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def draw_circle(img: np.ndarray, cx: int, cy: int, r: int, val: float = 1.0):
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[mask] = val


def draw_square(img: np.ndarray, x0: int, y0: int, s: int, val: float = 1.0):
    H, W = img.shape
    x1 = min(W, x0 + s)
    y1 = min(H, y0 + s)
    img[y0:y1, x0:x1] = val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='codex_on_QA/data/images_small')
    ap.add_argument('--n', type=int, default=40, help='Images per class')
    ap.add_argument('--size', type=int, default=96, help='Image width/height (square)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    (root / 'circles').mkdir(parents=True, exist_ok=True)
    (root / 'squares').mkdir(parents=True, exist_ok=True)

    H = W = args.size

    # Circles
    for i in range(args.n):
        img = np.zeros((H, W), dtype=float)
        # draw 1–3 disks
        for _ in range(rng.integers(1, 4)):
            r = int(rng.uniform(8, H/4))
            cx = int(rng.uniform(r, W - r))
            cy = int(rng.uniform(r, H - r))
            draw_circle(img, cx, cy, r, val=1.0)
        plt.imsave(root / 'circles' / f'c_{i:03d}.png', img, cmap='gray', vmin=0.0, vmax=1.0)

    # Squares
    for i in range(args.n):
        img = np.zeros((H, W), dtype=float)
        # draw 1–3 squares
        for _ in range(rng.integers(1, 4)):
            s = int(rng.uniform(10, H/3))
            x0 = int(rng.uniform(0, W - s))
            y0 = int(rng.uniform(0, H - s))
            draw_square(img, x0, y0, s, val=1.0)
        plt.imsave(root / 'squares' / f's_{i:03d}.png', img, cmap='gray', vmin=0.0, vmax=1.0)

    print(f'Wrote synthetic images to {root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

