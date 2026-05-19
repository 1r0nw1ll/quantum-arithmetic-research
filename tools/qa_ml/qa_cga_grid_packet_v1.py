"""QA/CGA grid packets for Fengbo-style 3D geometry multivectors.

Pepe Fengbo represents irregular 3D CFD geometry as fixed-resolution
G(3,0,0) voxel multivectors:

    P = mask + coordinate vector + normal-dual bivector
    V = mask + coordinate vector

This module supplies the QA packet boundary for that construction. It
quantizes observer-side coordinates/normals into exact integer packets and
records enough metadata to dequantize for relative-L2 observer metrics.

QA_COMPLIANCE = "qa_cga_grid_packet_v1 - exact int packet boundary; floats only at observer encode/decode"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class QAFengboPacket:
    """Quantized Fengbo geometry packet for one voxel/sample."""

    i: int
    j: int
    k: int
    grid_size: int
    modulus: int
    mask: int
    vector_q: tuple[int, int, int]
    bivector_q: tuple[int, int, int]
    trivector_q: int = 0


def _clip_unit(value: float) -> float:
    return float(min(1.0, max(-1.0, value)))


def quantize_unit(value: float, modulus: int) -> int:
    """Quantize a value in [-1, 1] to an integer in [0, modulus]."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    clipped = _clip_unit(value)
    return int(round((clipped + 1.0) * modulus / 2.0))


def dequantize_unit(value_q: int, modulus: int) -> float:
    """Decode an integer in [0, modulus] back to [-1, 1]."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if value_q < 0 or value_q > modulus:
        raise ValueError("quantized value outside [0, modulus]")
    return (2.0 * float(value_q) / float(modulus)) - 1.0


def normal_to_dual_bivector(normal: Iterable[float]) -> tuple[float, float, float]:
    """Return (B12, B13, B23) for B = I3*n in G(3,0,0)."""
    n1, n2, n3 = [float(v) for v in normal]
    return (n3, -n2, n1)


def dual_bivector_to_normal(bivector: Iterable[float]) -> tuple[float, float, float]:
    """Return normal (n1, n2, n3) from dual bivector components (B12,B13,B23)."""
    b12, b13, b23 = [float(v) for v in bivector]
    return (b23, -b13, b12)


def grid_index(coord: Iterable[float], grid_size: int) -> tuple[int, int, int]:
    """Map coordinate in [-1,1]^3 to Fengbo-style voxel indices."""
    if grid_size <= 1:
        raise ValueError("grid_size must be > 1")
    out = []
    for value in coord:
        clipped = _clip_unit(float(value))
        idx = int(round((clipped + 1.0) * (grid_size - 1) / 2.0))
        out.append(min(grid_size - 1, max(0, idx)))
    return (out[0], out[1], out[2])


def encode_pressure_packet(
    coord: Iterable[float],
    normal: Iterable[float],
    *,
    grid_size: int,
    modulus: int,
    mask: int = 1,
    inlet_velocity: float = 0.0,
) -> QAFengboPacket:
    """Encode Pepe's pressure geometry multivector P as a QA packet."""
    c = tuple(float(v) for v in coord)
    b = normal_to_dual_bivector(normal)
    return QAFengboPacket(
        i=grid_index(c, grid_size)[0],
        j=grid_index(c, grid_size)[1],
        k=grid_index(c, grid_size)[2],
        grid_size=grid_size,
        modulus=modulus,
        mask=int(mask),
        vector_q=tuple(quantize_unit(v, modulus) for v in c),
        bivector_q=tuple(quantize_unit(v, modulus) for v in b),
        trivector_q=quantize_unit(inlet_velocity, modulus),
    )


def encode_velocity_packet(
    coord: Iterable[float],
    *,
    grid_size: int,
    modulus: int,
    mask: int = 1,
) -> QAFengboPacket:
    """Encode Pepe's velocity geometry multivector V as a QA packet."""
    c = tuple(float(v) for v in coord)
    return QAFengboPacket(
        i=grid_index(c, grid_size)[0],
        j=grid_index(c, grid_size)[1],
        k=grid_index(c, grid_size)[2],
        grid_size=grid_size,
        modulus=modulus,
        mask=int(mask),
        vector_q=tuple(quantize_unit(v, modulus) for v in c),
        bivector_q=(0, 0, 0),
        trivector_q=0,
    )


def decode_vector(packet: QAFengboPacket) -> np.ndarray:
    return np.asarray([dequantize_unit(v, packet.modulus) for v in packet.vector_q], dtype=np.float64)


def decode_dual_normal(packet: QAFengboPacket) -> np.ndarray:
    bivector = [dequantize_unit(v, packet.modulus) for v in packet.bivector_q]
    return np.asarray(dual_bivector_to_normal(bivector), dtype=np.float64)


def decode_trivector(packet: QAFengboPacket) -> float:
    return dequantize_unit(packet.trivector_q, packet.modulus)


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float(np.linalg.norm(estimate))
    return float(np.linalg.norm(reference - estimate) / denom)
