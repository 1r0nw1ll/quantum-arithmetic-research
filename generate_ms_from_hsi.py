#!/usr/bin/env python3
"""
Synthesize an 8-band multispectral (MS) cube from Houston 2013 HSI patches.

- Input:  HSI_Tr.mat with key 'Data' shaped (N, 11, 11, 144)
- Output: MS_Tr.mat with key 'Data' shaped (N, 11, 11, 8)

 Modes:
 - equal: average 144 bands into 8 equal bins (18 bands each)
 - wv2:   approximate WorldView-2 spectral bands using wavelength ranges
          (uses linear wavelength grid 364–1046 nm across 144 bands), flat mean per band
 - wv2_weighted: WV2 with bandpass weighting (triangular response within each band)

Usage:
  python generate_ms_from_hsi.py --mode wv2

This script overwrites multimodal_data/MS_Tr.mat by default.
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat


def approximate_wavelengths(bands: int) -> np.ndarray:
    # Approximate HSI wavelengths (nm) based on Houston 2013 meta
    # Start ~364 nm, end ~1046 nm, length 144
    start_nm, end_nm = 364.0, 1046.1
    return np.linspace(start_nm, end_nm, bands)


def synthesize_equal(hsi: np.ndarray) -> np.ndarray:
    N, H, W, B = hsi.shape
    bins = 8
    bands_per_bin = B // bins
    ms = np.zeros((N, H, W, bins), dtype=np.float32)
    for i in range(bins):
        s = i * bands_per_bin
        e = (i + 1) * bands_per_bin if i < bins - 1 else B
        ms[..., i] = hsi[..., s:e].mean(axis=-1, dtype=np.float64)
    return ms


def synthesize_wv2(hsi: np.ndarray) -> np.ndarray:
    """Map HSI to approximate WV2 bands via wavelength ranges (nm)."""
    N, H, W, B = hsi.shape
    wl = approximate_wavelengths(B)
    # WV2 spectral range definitions (nm)
    wv2_ranges = [
        (400, 450),   # Coastal
        (450, 510),   # Blue
        (510, 580),   # Green
        (585, 625),   # Yellow
        (630, 690),   # Red
        (705, 745),   # Red Edge
        (770, 895),   # NIR1
        (860, 1040),  # NIR2
    ]
    ms = np.zeros((N, H, W, 8), dtype=np.float32)
    for i, (lo, hi) in enumerate(wv2_ranges):
        mask = (wl >= lo) & (wl <= hi)
        idx = np.where(mask)[0]
        if idx.size == 0:
            # Fallback: pick nearest band
            nearest = np.argmin(np.abs(wl - ((lo + hi) / 2)))
            ms[..., i] = hsi[..., nearest].astype(np.float32)
        else:
            ms[..., i] = hsi[..., idx].mean(axis=-1, dtype=np.float64)
    return ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize 8-band MS from HSI patches")
    ap.add_argument("--mode", choices=["equal", "wv2", "wv2_weighted"], default="wv2",
                    help="Synthesis mode: equal bins or WV2 mapping (default: wv2)")
    ap.add_argument("--hsi-path", type=str, default="multimodal_data/HSI_Tr.mat",
                    help="Path to input HSI .mat file with key 'Data' (default: multimodal_data/HSI_Tr.mat)")
    ap.add_argument("--out-path", type=str, default="multimodal_data/MS_Tr.mat",
                    help="Output MS .mat path (default: multimodal_data/MS_Tr.mat)")
    args = ap.parse_args()

    hsi_path = Path(args.hsi_path)
    out_path = Path(args.out_path)
    if not hsi_path.exists():
        raise FileNotFoundError(f"HSI file not found: {hsi_path}")

    data = loadmat(hsi_path)
    key = next(k for k in data.keys() if not k.startswith("__"))
    hsi = data[key]
    if hsi.ndim != 4 or hsi.shape[-1] != 144:
        raise ValueError(f"Expected HSI shape (N,11,11,144), got {hsi.shape}")

    if args.mode == "equal":
        ms = synthesize_equal(hsi)
    elif args.mode == "wv2":
        ms = synthesize_wv2(hsi)
    elif args.mode == "wv2_weighted":
        # Triangular bandpass per WV2 band
        N, H, W, B = hsi.shape
        wl = approximate_wavelengths(B)
        wv2_ranges = [
            (400, 450), (450, 510), (510, 580), (585, 625),
            (630, 690), (705, 745), (770, 895), (860, 1040)
        ]
        ms = np.zeros((N, H, W, 8), dtype=np.float32)
        for i, (lo, hi) in enumerate(wv2_ranges):
            c = 0.5 * (lo + hi)
            hw = max(1e-6, 0.5 * (hi - lo))
            w = 1.0 - np.abs(wl - c) / hw
            w = np.clip(w, 0.0, 1.0)
            s = w.sum()
            if s <= 0:
                # Fallback to nearest wavelength
                nearest = np.argmin(np.abs(wl - c))
                ms[..., i] = hsi[..., nearest].astype(np.float32)
            else:
                w = w / s
                ms[..., i] = (hsi * w).sum(axis=-1, dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(out_path, {"Data": ms})
    print(f"Saved {out_path} with shape {ms.shape} using mode='{args.mode}'")


if __name__ == "__main__":
    main()
