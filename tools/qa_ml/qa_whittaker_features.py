"""Observer features from the QA Whittaker rational direction substrate.

The exact substrate comes from cert family [273]. This module only converts
canonical rational S2 packets into bounded observer features for ML probes.

QA_COMPLIANCE = "observer features over exact [273] packets; no physics claim"
"""

from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent.parent
S2_VALIDATOR = (
    REPO
    / "qa_alphageometry_ptolemy"
    / "qa_whittaker_rational_direction_s2_cert_v1"
    / "qa_whittaker_rational_direction_s2_cert_validate.py"
)

WHITTAKER_FEATURE_NAMES: tuple[str, ...] = (
    "w_x",
    "w_y",
    "w_z",
    "w_z2",
    "w_den_log",
    "w_dot_xy",
    "w_phase_residual",
)


def _load_s2_validator():
    spec = importlib.util.spec_from_file_location("qa_wrd_s2_for_ml", S2_VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def whittaker_unit_points(m: int = 5) -> tuple[tuple[float, float, float, int], ...]:
    """Return [273] S2 packets as observer floats plus denominator.

    `m=5` gives 676 directions, enough for a compact nearest-direction
    observer packet while keeping feature construction cheap.
    """
    module = _load_s2_validator()
    model = module.build_model(m)
    out = []
    for x_num, y_num, z_num, den in model["points"]:
        out.append((x_num / den, y_num / den, z_num / den, den))
    return tuple(out)


def nearest_whittaker_features(
    cos_phase: float,
    sin_phase: float,
    m: int = 5,
) -> tuple[float, ...]:
    """Nearest [273] S2 direction features for a unit-circle phase point.

    The target embeds the complex phase as (cos, sin, 0) on S2. Nearest
    direction maximizes dot product with that equatorial point. Returned
    features are observer floats; the exact packet source remains [273].
    """
    best = None
    best_dot = -float("inf")
    for x, y, z, den in whittaker_unit_points(m):
        dot = x * cos_phase + y * sin_phase
        if dot > best_dot:
            best_dot = dot
            best = (x, y, z, den)
    assert best is not None
    x, y, z, den = best
    phase_residual = 1.0 - best_dot
    return (
        float(x),
        float(y),
        float(z),
        float(z * z),
        float(math.log1p(den)),
        float(best_dot),
        float(phase_residual),
    )


def whittaker_feature_matrix(
    cos_phase: np.ndarray,
    sin_phase: np.ndarray,
    m: int = 5,
) -> np.ndarray:
    rows = [
        nearest_whittaker_features(float(c), float(s), m=m)
        for c, s in zip(cos_phase, sin_phase)
    ]
    return np.asarray(rows, dtype=np.float32)
