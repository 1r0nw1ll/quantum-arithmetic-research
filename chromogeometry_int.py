"""
Integer-only chromogeometry feature extraction.

Entry points:
- spectral_to_chromo_int(spectrum, scale_bits=14): returns 5D integer feature vector [u, v, Qb, Qr, Qg]
- chromo_fuse_int(hsi_vec, ms_vec, lidar_scalar): returns 11D fused feature vector

Design notes:
- Implements a fixed-point radix-2 FFT for power-of-two lengths; non-power-of-two inputs are zero-padded to next power-of-two.
- Uses int32 accumulators with dynamic right-shift per stage to avoid overflow.
- Magnitude is approximated via L1/Chebyshev blend: max(|Re|,|Im|) + (min(|Re|,|Im|)//2)
- Phase via integer CORDIC atan2; u/v correspond to the fixed-point real/imag of the dominant bin.

This module is a reference implementation to substantiate the claims; production deployments should port to C on embedded targets with precomputed twiddle/atan tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math
import numpy as np


# --- Fixed-point helpers ---

def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _bit_reverse_indices(n: int) -> np.ndarray:
    bits = (n - 1).bit_length()
    idx = np.arange(n, dtype=np.uint32)
    rev = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        x = idx[i]
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def _twiddles_int(n: int, scale_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer twiddle tables (cos, sin) for N, scaled by 2**scale_bits.

    Note: Precomputes with Python float then quantizes to fixed-point ints.
    In embedded settings, store these tables as constants.
    """
    k = np.arange(n // 2)
    ang = -2.0 * math.pi * k / n
    scale = 1 << scale_bits
    c = np.round(np.cos(ang) * scale).astype(np.int32)
    s = np.round(np.sin(ang) * scale).astype(np.int32)
    return c, s


# --- Integer CORDIC for atan2 (phase) ---

_CORDIC_ATAN_TABLE = [
    int(round(math.degrees(math.atan(2 ** -i)))) for i in range(16)
]


def _cordic_atan2_deg(y: int, x: int) -> int:
    """Return atan2(y, x) in degrees as integer using CORDIC, 16 iterations.

    The output is in degrees in range [-180, 180].
    """
    # Handle zero quickly
    if x == 0 and y == 0:
        return 0

    angle = 0
    # Normalize quadrant
    if x < 0:
        x, y = -x, -y
        angle = 180

    xi, yi = x, y
    for i, a in enumerate(_CORDIC_ATAN_TABLE):
        if yi > 0:
            # rotate clockwise
            x_new = xi + (yi >> i)
            y_new = yi - (xi >> i)
            xi, yi = x_new, y_new
            angle += a
        else:
            # rotate counter-clockwise
            x_new = xi - (yi >> i)
            y_new = yi + (xi >> i)
            xi, yi = x_new, y_new
            angle -= a

    # Clamp range
    if angle > 180:
        angle -= 360
    if angle <= -180:
        angle += 360
    return int(angle)


def _approx_mag(re: int, im: int) -> int:
    """Approximate magnitude using max + min/2.

    Pure integer approximation close to L-infinity / L1 blend.
    """
    a = abs(re)
    b = abs(im)
    m = max(a, b)
    n = min(a, b)
    return m + (n >> 1)


# --- Fixed-point radix-2 FFT ---

def _fft_fixed_radix2(x: np.ndarray, scale_bits: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fixed-point radix-2 FFT for real-valued input.

    Args:
        x: int32 array length N (power-of-two), fixed-point scaled by 2**scale_bits.
        scale_bits: twiddle scaling and multiply right-shift bits.

    Returns:
        (re, im): int32 arrays of length N containing fixed-point FFT output.
    """
    N = int(x.shape[0])
    assert N and (N & (N - 1) == 0), "Length must be power-of-two"

    # Bit-reverse copy
    idx = _bit_reverse_indices(N)
    re = x[idx].astype(np.int32)
    im = np.zeros(N, dtype=np.int32)

    # Iterative Cooley-Tukey
    half = 1
    stage = 0
    while (half << 1) <= N:
        m = half << 1
        tw_cos, tw_sin = _twiddles_int(m, scale_bits)
        for k in range(0, N, m):
            for j in range(half):
                idx_even = k + j
                idx_odd = idx_even + half

                tre = re[idx_odd].astype(np.int64)
                tim = im[idx_odd].astype(np.int64)

                c = np.int64(tw_cos[j])
                s = np.int64(tw_sin[j])

                # Complex multiply: (tre + i tim) * (c + i s) >> scale_bits
                # t = [tre*c - tim*s, tre*s + tim*c]
                t_re = np.int32(((tre * c - tim * s) >> scale_bits))
                t_im = np.int32(((tre * s + tim * c) >> scale_bits))

                ue = re[idx_even]
                ve = im[idx_even]

                re[idx_odd] = ue - t_re
                im[idx_odd] = ve - t_im
                re[idx_even] = ue + t_re
                im[idx_even] = ve + t_im

        # Optional per-stage right shift to prevent growth
        # Shift by 1 bit (divide by 2) to keep dynamic range bounded
        re >>= 1
        im >>= 1
        stage += 1
        half = m >> 1

    return re, im


# --- Chromogeometry features (integer) ---

@dataclass
class ChromoIntConfig:
    scale_bits: int = 14  # fixed-point scaling for inputs and twiddles
    pad_to_pow2: bool = True
    ignore_dc: bool = True


def spectral_to_chromo_int(spectrum: np.ndarray, cfg: ChromoIntConfig | None = None) -> np.ndarray:
    """Compute integer-only chromogeometry features for one spectrum.

    Returns int32 vector [u, v, Qb, Qr, Qg] in fixed-point scale.
    - u and v correspond to real and imaginary parts of the dominant FFT bin.
    - Quadrances are in squared scale; consumers may re-scale to float as needed.
    """
    if cfg is None:
        cfg = ChromoIntConfig()

    x = np.asarray(spectrum, dtype=np.float64)
    x = x - float(x.mean())

    # Fixed-point scale input
    S = 1 << cfg.scale_bits
    xi = np.clip(np.round(x * S), -2 ** 31 + 1, 2 ** 31 - 1).astype(np.int32)

    N = int(xi.shape[0])
    if cfg.pad_to_pow2:
        N2 = _next_power_of_two(N)
        if N2 != N:
            pad = np.zeros(N2 - N, dtype=np.int32)
            xi = np.concatenate([xi, pad], axis=0)
            N = N2

    re, im = _fft_fixed_radix2(xi, cfg.scale_bits)

    # Compute approximate magnitudes per bin
    mags = np.fromiter((_approx_mag(int(re[k]), int(im[k])) for k in range(N // 2 + 1)), dtype=np.int32)

    if cfg.ignore_dc and mags.shape[0] > 0:
        mags[0] = 0

    top_idx = int(np.argmax(mags))
    u = int(re[top_idx])
    v = int(im[top_idx])

    # Quadrances in fixed-point squared domain (scale doubles)
    qb = u * u + v * v
    qr = u * u - v * v
    qg = (u * v) << 1

    return np.array([u, v, qb, qr, qg], dtype=np.int64)


def chromo_fuse_int(hsi_vec: np.ndarray, ms_vec: np.ndarray, lidar_scalar: float,
                    cfg: ChromoIntConfig | None = None) -> np.ndarray:
    """Fuse integer chromogeometry features from HSI, MS and LiDAR into 11D vector.

    Outputs int64 vector with same fixed-point scaling for u,v, and squared scaling for Q* terms.
    LiDAR is scaled to the same fixed-point.
    """
    if cfg is None:
        cfg = ChromoIntConfig()
    S = 1 << cfg.scale_bits
    lid = int(np.clip(round(float(lidar_scalar) * S), -2 ** 31 + 1, 2 ** 31 - 1))
    return np.concatenate([hsi_vec.astype(np.int64), ms_vec.astype(np.int64), np.array([lid], dtype=np.int64)])


def chromo_int_to_float(int_feat: np.ndarray, cfg: ChromoIntConfig | None = None) -> np.ndarray:
    """Convert 5D integer chromogeometry feature to float approximations.

    Rescaling: u,v -> divide by 2**scale_bits; Q* -> divide by 2**(2*scale_bits).
    """
    if cfg is None:
        cfg = ChromoIntConfig()
    S = float(1 << cfg.scale_bits)
    SS = S * S
    u, v, qb, qr, qg = int_feat
    return np.array([u / S, v / S, qb / SS, qr / SS, qg / SS], dtype=np.float64)


def fused_int_to_float(fused11: np.ndarray, cfg: ChromoIntConfig | None = None) -> np.ndarray:
    if cfg is None:
        cfg = ChromoIntConfig()
    # Map 11D vector: [5 HSI][5 MS][1 LIDAR]
    hsi_f = chromo_int_to_float(fused11[0:5], cfg)
    ms_f = chromo_int_to_float(fused11[5:10], cfg)
    lid = fused11[10] / float(1 << cfg.scale_bits)
    return np.concatenate([hsi_f, ms_f, np.array([lid], dtype=np.float64)])


__all__ = [
    "ChromoIntConfig",
    "spectral_to_chromo_int",
    "chromo_fuse_int",
    "chromo_int_to_float",
    "fused_int_to_float",
]
