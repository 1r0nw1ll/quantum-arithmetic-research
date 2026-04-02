"""
QA Feature Map v4.0

Adds higher‑complexity, relational/local spectral/structured nonlinear
invariants on top of the existing v3 stack.

Modes
-----
 - qa21  : canonical 21 (pointwise)
 - qa27  : qa21 + 6 canonical‑expanded (pointwise)
 - qa83  : v3 full stack (pointwise)
 - qa96  : qa83 + 13 new invariants (6 neighborhood stats + 3 local spectral + 4 nonlinear composites)

New (qa96) invariants (names)
-----------------------------
Neighborhood statistics (k‑NN in QA‑27 space):
 - eps_neigh_mean, eps_neigh_var
 - FoverC_neigh_mean, FoverC_neigh_var
 - GoverC_neigh_mean, GoverC_neigh_var

Local spectral (covariance of [eps, F/C, G/C, theta, R_h, E_QA] over neighbors):
 - lambda1_local  (largest eigenvalue of local covariance)
 - lambda2_local  (second largest)
 - anisotropy_local = lambda1 / (trace + eps)

Nonlinear composites (pointwise):
 - curvature_energy = eps * E_QA
 - shape_energy     = (F/C) * (G/C)
 - eps_cos_theta    = eps * cos(theta)
 - eps_sin_theta    = eps * sin(theta)

Notes
-----
 - Neighborhoods are computed in QA‑27 space by default using sklearn NearestNeighbors.
 - For efficiency, you may cache knn indices externally and pass them here (future extension).

"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    # Optional; if unavailable, raise a clear error at call time
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

from .qa_feature_map_v3 import (
    compute_qa_invariants,
    CANONICAL_21,
    EXPANDED_6,
)


QA_MODES_V4 = ("qa21", "qa27", "qa83", "qa96", "qa96lite", "qa100")


def _compute_pointwise_arrays(b: np.ndarray, e: np.ndarray, mode: str) -> Dict[str, np.ndarray]:
    """Compute pointwise QA features using v3 invariants.

    Returns a dict[name] -> array (n,). For qa21/qa27 we include exactly
    those names; for qa83 we include the full v3 set by computing the
    full invariant dict per sample and stacking common keys.
    """
    b = np.asarray(b, dtype=float)
    e = np.asarray(e, dtype=float)
    n = b.shape[0]

    # Determine the name set we need to return for non‑qa83 modes
    if mode == "qa21":
        name_list = list(CANONICAL_21)
    elif mode == "qa27":
        name_list = list(CANONICAL_21) + list(EXPANDED_6)
    else:
        # qa83: infer names from the first sample dict
        inv0 = compute_qa_invariants(float(b[0]), float(e[0]))
        name_list = list(inv0.keys())

    out: Dict[str, np.ndarray] = {name: np.zeros(n, dtype=float) for name in name_list}
    for i in range(n):
        inv = compute_qa_invariants(float(b[i]), float(e[i]))
        # v3 returns superset; fill the subset we want
        for name in name_list:
            out[name][i] = float(inv.get(name, 0.0))
    return out


def compute_neighborhood_indices(X: np.ndarray, k: int = 16, metric: str = "euclidean") -> np.ndarray:
    """Compute k‑NN indices for each row in X.

    Returns indices array of shape (n, k). Requires scikit‑learn.
    """
    if not _HAS_SKLEARN:  # pragma: no cover
        raise RuntimeError("scikit‑learn is required for compute_neighborhood_indices; install scikit‑learn")
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    _dist, idx = nn.kneighbors(X)
    return idx


def compute_qa_features_v4(
    b: np.ndarray,
    e: np.ndarray,
    *,
    mode: str = "qa83",
    k_neigh: int = 16,
    base_mode_for_knn: str = "qa27",
) -> Dict[str, np.ndarray]:
    """Compute QA features, including the QA‑96 invariants when requested.

    Parameters
    ----------
    b, e : arrays of shape (n,)
        QA tuple components per sample.
    mode : {"qa21","qa27","qa83","qa96"}
        Feature mode to compute.
    k_neigh : int
        Number of neighbors for neighborhood/spectral invariants (qa96).
    base_mode_for_knn : {"qa27","qa83"}
        Which QA feature set to use to build the k‑NN space.

    Returns
    -------
    features : dict[str, np.ndarray]
        Mapping of feature name -> (n,) array.
    """
    if mode not in QA_MODES_V4:
        raise ValueError(f"Unsupported QA mode for v4: {mode}")

    b = np.asarray(b, dtype=float)
    e = np.asarray(e, dtype=float)
    n = b.shape[0]

    # For legacy modes, return pointwise subsets from v3
    if mode in ("qa21", "qa27", "qa83"):
        return _compute_pointwise_arrays(b, e, mode)

    # qa96 / qa96lite / qa100: start from qa83 base
    base = _compute_pointwise_arrays(b, e, "qa83")

    # If qa100 requested, add 17 Rational Trigonometry invariants.
    if mode == "qa100":
        # Core tuples
        b_arr = base.get("b", b)
        e_arr = base.get("e", e)
        d_arr = base.get("d")
        a_arr = base.get("a")
        if d_arr is None or a_arr is None:
            raise KeyError("qa100 requires 'd' and 'a' from v3 invariants")
        # Triangle legs
        C = base["C"]; F = base["F"]; G = base["G"]
        # Canonical J,K,X if missing
        J = base.get("J", b_arr * d_arr)
        K = base.get("K", d_arr * a_arr)
        Xv = base.get("X", e_arr * d_arr)

        eps_small = 1e-12
        C2 = C**2; F2 = F**2; G2 = G**2
        # Quadrance cross-terms
        base["Q_CF"] = C2 + F2
        base["Q_CG"] = C2 + G2
        base["Q_FG"] = F2 + G2
        # Spreads
        base["s_C"] = C2 / (F2 + G2 + eps_small)
        base["s_F"] = F2 / (C2 + G2 + eps_small)
        base["s_G"] = G2 / (C2 + F2 + eps_small)
        # Triple-spread
        S_sum = base["s_C"] + base["s_F"] + base["s_G"]
        S_prod = base["s_C"]*base["s_F"] + base["s_F"]*base["s_G"] + base["s_G"]*base["s_C"]
        base["S_spread_sum"] = S_sum
        base["S_spread_prod"] = S_prod
        base["S_spread_resid"] = S_sum - 2.0*S_prod
        # Cross-spreads on (J,K,X)
        J2 = J**2; K2 = K**2; X2 = Xv**2
        base["s_JK"] = (J - K)**2 / (J2 + K2 + eps_small)
        base["s_XK"] = (Xv - K)**2 / (X2 + K2 + eps_small)
        base["s_JX"] = (J - Xv)**2 / (J2 + X2 + eps_small)
        # Quadrance–spread mixed
        base["M1_QC_sF"] = C2 * base["s_F"]
        base["M2_QF_sC"] = F2 * base["s_C"]
        base["M3_QG_sC"] = G2 * base["s_C"]
        # Projective cross-ratios
        num1 = (b_arr - d_arr) * (e_arr - a_arr)
        den1 = (b_arr - a_arr) * (e_arr - d_arr)
        base["CR1_be_da_over_ba_ed"] = num1 / (den1 + eps_small)
        num2 = (b_arr - e_arr) * (d_arr - a_arr)
        den2 = (b_arr - a_arr) * (d_arr - e_arr)
        base["CR2_be_da_over_ba_de"] = num2 / (den2 + eps_small)
        return base

    # Build k‑NN in QA‑27 space (default) for neighborhood definitions
    if base_mode_for_knn not in ("qa27", "qa83"):
        raise ValueError("base_mode_for_knn must be 'qa27' or 'qa83'")

    # Minimal set for knn: eps, F_over_C, G_over_C, theta, R_h, E_QA
    required_knn = ["eps", "F_over_C", "G_over_C", "theta", "R_h", "E_QA"]
    for nm in required_knn:
        if nm not in base:
            raise KeyError(f"qa96 requires '{nm}' from v3 invariants; not found")

    X_knn = np.stack([base[nm] for nm in required_knn], axis=1)
    idx = compute_neighborhood_indices(X_knn, k=k_neigh, metric="euclidean")

    eps_arr = base["eps"]
    Fc_arr = base["F_over_C"]
    Gc_arr = base["G_over_C"]
    theta = base["theta"]
    R_h = base["R_h"]
    E_QA = base["E_QA"]

    # Prepare outputs (13 dims)
    eps_neigh_mean = np.zeros(n)
    eps_neigh_var = np.zeros(n)
    FoverC_neigh_mean = np.zeros(n)
    FoverC_neigh_var = np.zeros(n)
    GoverC_neigh_mean = np.zeros(n)
    GoverC_neigh_var = np.zeros(n)

    lambda1_local = np.zeros(n)
    lambda2_local = np.zeros(n)
    anisotropy_local = np.zeros(n)

    curvature_energy = eps_arr * E_QA
    shape_energy = Fc_arr * Gc_arr
    eps_cos_theta = eps_arr * np.cos(theta)
    eps_sin_theta = eps_arr * np.sin(theta)

    tiny = 1e-9

    for i in range(n):
        nbrs = idx[i]  # (k,)
        # Neighborhood stats
        e_n = eps_arr[nbrs]
        fc_n = Fc_arr[nbrs]
        gc_n = Gc_arr[nbrs]

        m_e = float(np.mean(e_n))
        v_e = float(np.mean((e_n - m_e) ** 2))
        m_fc = float(np.mean(fc_n))
        v_fc = float(np.mean((fc_n - m_fc) ** 2))
        m_gc = float(np.mean(gc_n))
        v_gc = float(np.mean((gc_n - m_gc) ** 2))

        eps_neigh_mean[i] = m_e
        eps_neigh_var[i] = v_e
        FoverC_neigh_mean[i] = m_fc
        FoverC_neigh_var[i] = v_fc
        GoverC_neigh_mean[i] = m_gc
        GoverC_neigh_var[i] = v_gc

        # Local spectral: covariance of [eps, F/C, G/C, theta, R_h, E_QA]
        V = np.stack([
            e_n,
            fc_n,
            gc_n,
            theta[nbrs],
            R_h[nbrs],
            E_QA[nbrs],
        ], axis=1)  # (k, 6)

        # Covariance with rows as observations
        cov = np.cov(V, rowvar=False, bias=False)
        cov = 0.5 * (cov + cov.T)
        try:
            vals = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            vals = np.linalg.eigvals(cov).real
        vals = np.sort(np.real(vals))
        if vals.size >= 1:
            lambda1 = float(vals[-1])
        else:
            lambda1 = 0.0
        if vals.size >= 2:
            lambda2 = float(vals[-2])
        else:
            lambda2 = 0.0
        tr = float(np.sum(vals))

        lambda1_local[i] = lambda1
        lambda2_local[i] = lambda2
        anisotropy_local[i] = lambda1 / (tr + tiny)

    # Attach new invariants
    base.update({
        # Neighborhood stats
        "eps_neigh_mean": eps_neigh_mean,
        "eps_neigh_var": eps_neigh_var,
        "FoverC_neigh_mean": FoverC_neigh_mean,
        "FoverC_neigh_var": FoverC_neigh_var,
        "GoverC_neigh_mean": GoverC_neigh_mean,
        "GoverC_neigh_var": GoverC_neigh_var,
        # Local spectral
        "lambda1_local": lambda1_local,
        "lambda2_local": lambda2_local,
        "anisotropy_local": anisotropy_local,
        # Nonlinear composites
        "curvature_energy": curvature_energy,
        "shape_energy": shape_energy,
        "eps_cos_theta": eps_cos_theta,
        "eps_sin_theta": eps_sin_theta,
    })

    # If only the lightweight subset is requested, filter down here
    if mode == "qa96lite":
        subset = [
            "eps_neigh_mean",
            "eps_neigh_var",
            "FoverC_neigh_mean",
            "FoverC_neigh_var",
            "GoverC_neigh_mean",
            "GoverC_neigh_var",
            "anisotropy_local",
        ]
        return {k: base[k] for k in subset}

    return base
