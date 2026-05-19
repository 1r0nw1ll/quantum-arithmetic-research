"""QA-ML v3.3 CGA motor primitive (dual-quaternion form).

A CGA motor M = T·R encodes a rigid SE(3) transformation as a single
element of the even subalgebra of Cl(4,1). In the compact dual-quaternion
representation it has 8 coefficients:

    M = (r, d)

where r is a unit quaternion representing the rotation, and d is the
dual part encoding the translation via:

    d = ½ · t · r            (quaternion product, t as pure quaternion)

Composition of two motors is the dual-quaternion product:

    (r₁, d₁) · (r₂, d₂) = (r₁·r₂, r₁·d₂ + d₁·r₂)

This file provides:
  - `motor_from_se3(R, t)`         build a motor from a 3×3 rotation + 3-vector
  - `motor_to_se3(M)`              extract (R, t) from a motor
  - `motor_compose(M_a, M_b)`      dual-quaternion product
  - `motor_act(M, p)`              apply motor to a 3D point
  - `motor_inverse(M)`             motor inverse
  - `motor_quantize(M, m)`         snap motor to QA mod-m grid (per-coefficient)
  - `motor_quantize_se3(M, m)`     snap (R, t) channels via Euler/grid (parity variant)

Coefficient ordering (8 floats):
  [r_w, r_x, r_y, r_z, d_w, d_x, d_y, d_z]

Parity hypothesis: at m≥144 the QA-quantized motor compose matches the
continuous compose within sub-degree rotation error and sub-millimeter
translation error, validating discrete-fractional parity for SE(3)
composition. Validated against `experiments/qa_ml/57_pepe_qa_motor_parity.py`.

QA_COMPLIANCE = "qa_ml_motor_v3_3 — observer-projection at quantize boundary; continuous math inside compose, QA discretization at output"
"""

from __future__ import annotations

import numpy as np

QUAT_EPS = 1e-12


# ---------- quaternion helpers ----------


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → unit quaternion [w, x, y, z], stable Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(np.linalg.norm(q), QUAT_EPS)


def R_from_quat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < QUAT_EPS:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - z * w),       s * (x * z + y * w)],
        [s * (x * y + z * w),       1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),       s * (y * z + x * w),       1.0 - s * (x * x + y * y)],
    ], dtype=np.float64)


# ---------- motor (dual quaternion) ----------


def motor_from_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build motor M = (r, d) from rotation matrix R and translation t.

    d = ½ · t_q · r where t_q = [0, t_x, t_y, t_z]."""
    r = quat_from_R(R)
    t_q = np.array([0.0, t[0], t[1], t[2]], dtype=np.float64)
    d = 0.5 * quat_mul(t_q, r)
    return np.concatenate([r, d])


def motor_to_se3(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode motor → (R, t). t = 2·(d · r⁻¹) taking only the vector part."""
    r = M[:4]
    d = M[4:]
    r_conj = quat_conj(r)
    t_q = 2.0 * quat_mul(d, r_conj)
    return R_from_quat(r), t_q[1:]


def motor_compose(M_a: np.ndarray, M_b: np.ndarray) -> np.ndarray:
    """Dual-quaternion product: (r_a r_b, r_a d_b + d_a r_b)."""
    r_a, d_a = M_a[:4], M_a[4:]
    r_b, d_b = M_b[:4], M_b[4:]
    r_out = quat_mul(r_a, r_b)
    d_out = quat_mul(r_a, d_b) + quat_mul(d_a, r_b)
    return np.concatenate([r_out, d_out])


def motor_inverse(M: np.ndarray) -> np.ndarray:
    """Motor inverse: (r*, -r* · d · r*)."""
    r, d = M[:4], M[4:]
    r_conj = quat_conj(r)
    d_out = -quat_mul(quat_mul(r_conj, d), r_conj)
    return np.concatenate([r_conj, d_out])


def motor_act(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply motor to a 3D point: equivalent to R p + t."""
    R, t = motor_to_se3(M)
    return R @ np.asarray(p, dtype=np.float64) + t


# ---------- QA quantization ----------


def motor_quantize(M: np.ndarray, m: int, t_scale: float = 2.0) -> np.ndarray:
    """Snap motor coefficients to a QA mod-m grid.

    Rotation part (r): the unit-quaternion coefficients live in [-1, 1] so
    we quantize on a grid of step 2/m.
    Dual part (d): magnitudes scale with translation, so we quantize on a
    grid of step 2·t_scale/m (range [-t_scale, t_scale]).

    After quantization the r part is re-normalized to a unit quaternion so
    the resulting motor still represents a valid SE(3) element."""
    r = M[:4]
    d = M[4:]
    grid_r = 2.0 / m
    r_q = np.round(r / grid_r) * grid_r
    norm = np.linalg.norm(r_q)
    if norm < QUAT_EPS:
        r_q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        r_q = r_q / norm
    grid_d = 2.0 * t_scale / m
    d_q = np.round(d / grid_d) * grid_d
    return np.concatenate([r_q, d_q])


def motor_quantize_se3(M: np.ndarray, m: int, t_scale: float = 2.0) -> np.ndarray:
    """Alternate quantization path: decode to (R, t), Euler-snap R, grid-snap t,
    re-encode. Matches the path used in `experiments/qa_ml/29` and `30`."""
    from math import cos, pi, sin
    R, t = motor_to_se3(M)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = float(np.arctan2(-R[1, 2], R[1, 1]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    grid_a = 2 * pi / m
    rx_q = round(rx / grid_a) * grid_a
    ry_q = round(ry / grid_a) * grid_a
    rz_q = round(rz / grid_a) * grid_a
    cx, sx = cos(rx_q), sin(rx_q)
    cy_, sy_ = cos(ry_q), sin(ry_q)
    cz, sz = cos(rz_q), sin(rz_q)
    R_q = np.array([
        [cy_ * cz, sx * sy_ * cz - cx * sz, cx * sy_ * cz + sx * sz],
        [cy_ * sz, sx * sy_ * sz + cx * cz, cx * sy_ * sz - sx * cz],
        [-sy_, sx * cy_, cx * cy_],
    ], dtype=np.float64)
    grid_t = 2.0 * t_scale / m
    t_q = np.round(t / grid_t) * grid_t
    return motor_from_se3(R_q, t_q)
