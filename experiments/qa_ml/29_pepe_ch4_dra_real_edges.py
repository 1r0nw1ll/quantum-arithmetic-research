"""Real-edge DRA pipeline + QA-quantized DRA variant, replacing synthetic Figs 4.16/4.17 + Tables 4.5/4.6.

QA enters as: the predicted SE(3) rotation R̂ is snapped to the QA-mod-M_QA grid
(each Euler angle quantized to nearest 2π/M_QA grid point). This is the SE(3)
analog of Ch 3's rotor-angle QA discretization. Compare 4 methods:
  RANSAC          continuous-SE(3) baseline
  ICP             continuous-SE(3) baseline
  DRA-like        continuous-SE(3) closed-form GA-spirit
  QA-DRA          DRA-like + final QA-mod-M_QA quantization of rotation

Uses the ModelNet40 airplane meshes already in corpus/modelnet40/airplane/ to
build 3D line bundles (mesh edges), then runs real alignment methods to estimate
the SE(3) motor M̂ that maps a source bundle to a transformed target.

Task is correspondence-FREE: given two sets of 3D lines (no per-line pairing),
estimate the rotation + translation between them. This is exactly Pepe DRA's
problem (PDF §4.3) — just on airplane mesh edges instead of Structured3D /
Semantic3D room scenes.

Three real methods compared (Pepe Tables 4.5/4.6 mirror):
  RANSAC   — random correspondences + SVD Procrustes fit, n_iter trials,
             pick the one with lowest residual. Classic baseline.
  ICP      — Iterative Closest Point on line midpoints. Match each source midpoint
             to nearest target midpoint, fit SE(3) via SVD, iterate.
  DRA-like — Closed-form GA-spirit alignment: centroid translation + principal-axis
             rotation (PCA on line midpoints). Single forward pass, no iteration,
             O(N) per bundle — Pepe's equivariant-layer ideology in spirit.

Outputs:
  qa_fig_4_16_modelnet_edge_alignment.png   2 example alignments, source/target/predicted
  qa_fig_4_17_modelnet_edge_alignment.png   2 more examples (different rotations)
  qa_table_4_5_modelnet_line_reg_errors.png Rotation/translation error per method
  qa_table_4_6_modelnet_dra_relative.png    DRA relative performance vs baselines

QA_COMPLIANCE = "qa_ml_pepe_ch4_dra_real_edges — real 3D edge geometry; correspondence-free SE(3) estimation"
"""

from __future__ import annotations

import json
import sys
import time
from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).parent / "ch4_qa_real_pose_replica"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AIRPLANE_DIR = REPO / "corpus" / "modelnet40" / "airplane"
CACHE = Path(__file__).parent / "cache_ch4_dra"
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 0
N_TEST_MESHES = 50         # number of test airplanes
N_EDGES_PER_MESH = 60      # subsample edges per mesh
RANSAC_ITERS = 200
PARTIAL_VIEW_OVERLAP = 0.6  # fraction of edges shared between source and target views
NOISE_SIGMA = 0.05          # per-vertex Gaussian noise in normalized units
M_QA = 72                   # QA modulus for SE(3) rotation discretization (finer than Ch 3 mod-24: 5°/grid step)


# ---------- mesh edge extraction ----------

def parse_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse OFF file → (vertices [V,3], triangles [T,3])."""
    with path.open("r") as f:
        first = f.readline().strip()
        if first == "OFF":
            counts = f.readline().split()
        elif first.startswith("OFF"):
            counts = first[3:].split()
        else:
            raise ValueError(f"{path.name}: not an OFF file")
        n_v, n_f = int(counts[0]), int(counts[1])
        verts = np.empty((n_v, 3), dtype=np.float32)
        for i in range(n_v):
            verts[i] = [float(x) for x in f.readline().split()[:3]]
        tris = np.empty((n_f, 3), dtype=np.int64)
        for i in range(n_f):
            parts = f.readline().split()
            tris[i] = [int(parts[1]), int(parts[2]), int(parts[3])]
    return verts, tris


def extract_edges(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Extract unique mesh edges as [N, 2, 3] (endpoint A, endpoint B in 3D)."""
    edge_set = set()
    for tri in tris:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            a, b = int(tri[i]), int(tri[j])
            if a > b:
                a, b = b, a
            edge_set.add((a, b))
    edges = np.array(list(edge_set), dtype=np.int64)
    # Convert vertex indices to 3D endpoints
    endpoints = np.stack([verts[edges[:, 0]], verts[edges[:, 1]]], axis=1)
    return endpoints


def lines_to_midpoint_dir(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lines [N, 2, 3] → (midpoints [N, 3], directions [N, 3] unit, lengths [N])."""
    a, b = lines[:, 0], lines[:, 1]
    mid = (a + b) / 2.0
    diff = b - a
    length = np.linalg.norm(diff, axis=-1)
    direction = diff / (length[:, None] + 1e-9)
    return mid, direction, length


def normalize_and_subsample(lines: np.ndarray, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    """Center the bundle on its centroid, scale to unit RMS, keep n_keep edges by length."""
    mid, _, length = lines_to_midpoint_dir(lines)
    centroid = mid.mean(axis=0)
    lines = lines - centroid
    rms = np.sqrt(np.mean(np.linalg.norm(lines.reshape(-1, 3), axis=-1) ** 2))
    if rms > 0:
        lines = lines / rms
    # Keep n_keep longest edges to favor stable structure
    mid2, dir2, length2 = lines_to_midpoint_dir(lines)
    if len(lines) > n_keep:
        top_idx = np.argsort(length2)[::-1][:n_keep]
        lines = lines[top_idx]
    return lines


def transform_lines(lines: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply SE(3) (R, t) to line endpoints."""
    a = lines[:, 0] @ R.T + t
    b = lines[:, 1] @ R.T + t
    return np.stack([a, b], axis=1)


# ---------- alignment methods ----------

def fit_se3_from_correspondences(src_pts: np.ndarray, tgt_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Procrustes / SVD: given matched src/tgt point sets, return (R, t) minimising ||R src + t - tgt||."""
    c_src = src_pts.mean(axis=0)
    c_tgt = tgt_pts.mean(axis=0)
    src_c = src_pts - c_src
    tgt_c = tgt_pts - c_tgt
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = c_tgt - R @ c_src
    return R.astype(np.float32), t.astype(np.float32)


def align_ransac(src_lines: np.ndarray, tgt_lines: np.ndarray, n_iters: int = RANSAC_ITERS,
                 rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC: try random 4-point correspondence sets, fit SE(3), pick lowest residual.
    Correspondences are sampled from line midpoints."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    src_mid, _, _ = lines_to_midpoint_dir(src_lines)
    tgt_mid, _, _ = lines_to_midpoint_dir(tgt_lines)
    n_src, n_tgt = src_mid.shape[0], tgt_mid.shape[0]
    best_residual = float("inf")
    best_R = np.eye(3, dtype=np.float32)
    best_t = np.zeros(3, dtype=np.float32)
    for _ in range(n_iters):
        src_idx = rng.choice(n_src, size=4, replace=False)
        tgt_idx = rng.choice(n_tgt, size=4, replace=False)
        try:
            R, t = fit_se3_from_correspondences(src_mid[src_idx], tgt_mid[tgt_idx])
        except np.linalg.LinAlgError:
            continue
        # Inlier residual: nearest-neighbor distance from transformed src to tgt
        src_xform = src_mid @ R.T + t
        d = np.linalg.norm(src_xform[:, None, :] - tgt_mid[None, :, :], axis=-1)
        nn_d = d.min(axis=1)
        residual = nn_d.mean()
        if residual < best_residual:
            best_residual = residual
            best_R, best_t = R, t
    return best_R, best_t


def align_icp(src_lines: np.ndarray, tgt_lines: np.ndarray, max_iter: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """ICP on line midpoints."""
    src_mid, _, _ = lines_to_midpoint_dir(src_lines)
    tgt_mid, _, _ = lines_to_midpoint_dir(tgt_lines)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    for _ in range(max_iter):
        src_xform = src_mid @ R.T + t
        d = np.linalg.norm(src_xform[:, None, :] - tgt_mid[None, :, :], axis=-1)
        nn_idx = d.argmin(axis=1)
        R_new, t_new = fit_se3_from_correspondences(src_mid, tgt_mid[nn_idx])
        # Check convergence
        if np.allclose(R, R_new, atol=1e-5) and np.allclose(t, t_new, atol=1e-5):
            break
        R, t = R_new, t_new
    return R.astype(np.float32), t.astype(np.float32)


def qa_quantize_rotation(R: np.ndarray, m: int = M_QA) -> np.ndarray:
    """Snap a rotation matrix to the QA mod-m grid by quantizing each Euler angle
    to the nearest 2π/m grid point. Parallel to Ch 3's qa_quantize_angle on
    rotor angles, here applied to SE(3) rotations."""
    # ZYX Euler decomposition
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = float(np.arctan2(-R[1, 2], R[1, 1]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    grid = 2 * pi / m
    rx_q = round(rx / grid) * grid
    ry_q = round(ry / grid) * grid
    rz_q = round(rz / grid) * grid
    cx, sx = cos(rx_q), sin(rx_q)
    cy, sy_ = cos(ry_q), sin(ry_q)
    cz, sz = cos(rz_q), sin(rz_q)
    return np.array([
        [cy * cz, sx * sy_ * cz - cx * sz, cx * sy_ * cz + sx * sz],
        [cy * sz, sx * sy_ * sz + cx * cz, cx * sy_ * sz - sx * cz],
        [-sy_, sx * cy, cx * cy],
    ], dtype=np.float32)


def qa_quantize_translation(t: np.ndarray, m: int = M_QA, scale: float = 2.0) -> np.ndarray:
    """Snap translation to a QA grid on [-scale, scale]^3. Grid step = 2*scale/m."""
    grid = 2.0 * scale / m
    return (np.round(t / grid) * grid).astype(np.float32)


def align_dra_like(src_lines: np.ndarray, tgt_lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DRA-spirit closed-form alignment: centroid translation + PCA principal-axis rotation.

    Pepe DRA's claim is that GA-equivariant features (cross-line moments) carry enough
    info to estimate the motor without iterative correspondence search. This is the
    correspondence-free, single-pass GA-spirit baseline: use centroids (equivariant
    under translation) and principal axes (equivariant under rotation).

    To handle the PCA orientation ambiguity correctly under noise we enumerate the
    full octahedral group (24 proper rotations = signed permutations with det=+1).
    A pure sign-flip search misses cases where PCA eigenvalue ordering is unstable
    (common for elongated bundles where two principal-axis eigenvalues are close).
    The inlier score is the *median* nearest-neighbor distance — robust to the
    non-overlapping portion of partial views."""
    from itertools import permutations, product
    src_mid, src_dir, src_len = lines_to_midpoint_dir(src_lines)
    tgt_mid, tgt_dir, tgt_len = lines_to_midpoint_dir(tgt_lines)
    # Centroid translation
    c_src = src_mid.mean(axis=0)
    c_tgt = tgt_mid.mean(axis=0)
    # PCA on centered midpoints (weighted by edge length)
    src_c = src_mid - c_src
    tgt_c = tgt_mid - c_tgt
    cov_src = (src_c * src_len[:, None]).T @ src_c / max(src_len.sum(), 1e-9)
    cov_tgt = (tgt_c * tgt_len[:, None]).T @ tgt_c / max(tgt_len.sum(), 1e-9)
    _, V_src = np.linalg.eigh(cov_src)   # columns are eigenvectors, ascending eigenvalue
    _, V_tgt = np.linalg.eigh(cov_tgt)

    best_R, best_t, best_resid = np.eye(3, dtype=np.float32), c_tgt - c_src, float("inf")
    # Octahedral search: 6 permutations × 8 sign combos → 24 proper rotations
    for perm in permutations(range(3)):
        # Permutation matrix: column i is e_{perm[i]}, so it permutes the source axes
        P = np.zeros((3, 3))
        for i, pi in enumerate(perm):
            P[pi, i] = 1.0
        for signs in product([1.0, -1.0], repeat=3):
            S = P * np.array(signs)[None, :]
            R_cand = V_tgt @ S @ V_src.T
            if np.linalg.det(R_cand) <= 0:
                continue
            t_cand = c_tgt - R_cand @ c_src
            src_xform = src_mid @ R_cand.T + t_cand
            d = np.linalg.norm(src_xform[:, None, :] - tgt_mid[None, :, :], axis=-1)
            resid = float(np.median(d.min(axis=1)))   # robust to non-overlapping points
            if resid < best_resid:
                best_resid = resid
                best_R = R_cand
                best_t = t_cand
    return best_R.astype(np.float32), best_t.astype(np.float32)


def align_qa_dra(src_lines: np.ndarray, tgt_lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QA-DRA: run DRA-like closed-form, then snap (R, t) to QA mod-M_QA grid.
    Parallel to Ch 3 where φ_ij was snapped to mod-24 after the continuous angle
    was computed. The QA grid acts as a regularizer when the input bundles are
    noisy / partial."""
    R_cont, t_cont = align_dra_like(src_lines, tgt_lines)
    R_qa = qa_quantize_rotation(R_cont, m=M_QA)
    t_qa = qa_quantize_translation(t_cont, m=M_QA, scale=2.0)
    return R_qa, t_qa


# ---------- error metrics ----------

def rotation_error_deg(R_true: np.ndarray, R_pred: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    Rd = R_true @ R_pred.T
    tr = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def translation_error(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    return float(np.linalg.norm(t_pred - t_true))


# ---------- main pipeline ----------

def build_test_set() -> list[dict]:
    """Load N_TEST_MESHES airplane meshes, extract edges, generate alignment pairs."""
    test_files = sorted((AIRPLANE_DIR / "test").glob("*.off"))[:N_TEST_MESHES]
    rng = np.random.default_rng(SEED)
    out = []
    print(f"  building {len(test_files)} alignment test pairs ...")
    for k, f in enumerate(test_files):
        try:
            verts, tris = parse_off(f)
        except Exception as e:
            print(f"    skip {f.name}: {e}")
            continue
        edges = extract_edges(verts, tris)
        if len(edges) < int(N_EDGES_PER_MESH / PARTIAL_VIEW_OVERLAP):
            continue
        # Normalize the whole edge set first
        edges = normalize_and_subsample(edges, n_keep=10**6, rng=rng)
        # Generate known SE(3) transform
        R_true = Rotation.random(random_state=rng.integers(0, 2**31 - 1)).as_matrix().astype(np.float32)
        t_true = rng.normal(scale=0.5, size=3).astype(np.float32)
        # PARTIAL VIEWS: source and target see different (overlapping) subsets of edges
        n_total = len(edges)
        n_overlap = int(N_EDGES_PER_MESH * PARTIAL_VIEW_OVERLAP)
        n_unique = N_EDGES_PER_MESH - n_overlap
        all_idx = rng.permutation(n_total)
        overlap_idx = all_idx[:n_overlap]
        src_extra = all_idx[n_overlap:n_overlap + n_unique]
        tgt_extra = all_idx[n_overlap + n_unique:n_overlap + n_unique * 2]
        src_idx = np.concatenate([overlap_idx, src_extra])
        tgt_idx = np.concatenate([overlap_idx, tgt_extra])
        rng.shuffle(src_idx); rng.shuffle(tgt_idx)
        src = edges[src_idx].copy()
        tgt_clean = edges[tgt_idx].copy()
        tgt = transform_lines(tgt_clean, R_true, t_true)
        # NOISE: per-vertex Gaussian on both source and target
        src = src + rng.normal(scale=NOISE_SIGMA, size=src.shape).astype(np.float32)
        tgt = tgt + rng.normal(scale=NOISE_SIGMA, size=tgt.shape).astype(np.float32)
        out.append({"name": f.stem, "src": src, "tgt": tgt, "R_true": R_true, "t_true": t_true})
        if (k + 1) % 10 == 0:
            print(f"    {k + 1}/{len(test_files)} pairs")
    print(f"  built {len(out)} pairs total")
    return out


def make_qa_dra_at(m: int):
    """Factory: QA-DRA at a specific mod value."""
    def fn(src, tgt):
        R_cont, t_cont = align_dra_like(src, tgt)
        return qa_quantize_rotation(R_cont, m=m), qa_quantize_translation(t_cont, m=m, scale=2.0)
    fn.__name__ = f"qa_dra_mod{m}"
    return fn


def run_alignment(test_pairs: list[dict]) -> dict:
    """Run all methods (continuous baselines + QA-DRA at multiple M_QA values)."""
    methods = {
        "RANSAC": align_ransac,
        "ICP": align_icp,
        "DRA-like": align_dra_like,
        "QA-DRA m=12": make_qa_dra_at(12),
        "QA-DRA m=24": make_qa_dra_at(24),
        "QA-DRA m=48": make_qa_dra_at(48),
        "QA-DRA m=72": make_qa_dra_at(72),
        "QA-DRA m=144": make_qa_dra_at(144),
    }
    results = {name: {"rot": [], "trans": [], "examples": []} for name in methods}
    print(f"\nRunning {len(methods)} methods × {len(test_pairs)} pairs ...")
    rng = np.random.default_rng(SEED + 1)
    for k, pair in enumerate(test_pairs):
        for mname, fn in methods.items():
            t0 = time.time()
            if mname == "RANSAC":
                R_pred, t_pred = fn(pair["src"], pair["tgt"], rng=rng)
            else:
                R_pred, t_pred = fn(pair["src"], pair["tgt"])
            results[mname]["rot"].append(rotation_error_deg(pair["R_true"], R_pred))
            results[mname]["trans"].append(translation_error(pair["t_true"], t_pred))
            if k < 4:   # store first 4 for visualization
                src_aligned = transform_lines(pair["src"], R_pred, t_pred)
                results[mname]["examples"].append({
                    "name": pair["name"], "src": pair["src"], "tgt": pair["tgt"],
                    "src_aligned": src_aligned,
                    "rot_err": results[mname]["rot"][-1],
                    "trans_err": results[mname]["trans"][-1],
                })
        if (k + 1) % 10 == 0:
            print(f"    {k + 1}/{len(test_pairs)} pairs done")
    return results


# ---------- figure renderers ----------

def save_close(fig, name: str):
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def render_lines_3d(ax, lines: np.ndarray, color: str, linewidth: float = 0.7, alpha: float = 0.7):
    """Render a 3D line bundle on a matplotlib 3D axis."""
    for line in lines:
        a, b = line[0], line[1]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=linewidth, alpha=alpha)


def fig_4_16_17(results: dict, fig_num: str, example_idx_start: int = 0):
    """2 alignment examples (rows) × 3 columns (source / target / aligned)."""
    method = "DRA-like"   # Pepe shows DRA's results in Figs 4.16/4.17
    examples = results[method]["examples"][example_idx_start:example_idx_start + 2]
    if len(examples) < 2:
        return
    fig = plt.figure(figsize=(15, 9))
    for row, ex in enumerate(examples):
        ax_src = fig.add_subplot(2, 3, row * 3 + 1, projection="3d")
        ax_tgt = fig.add_subplot(2, 3, row * 3 + 2, projection="3d")
        ax_ali = fig.add_subplot(2, 3, row * 3 + 3, projection="3d")
        render_lines_3d(ax_src, ex["src"], "green", linewidth=0.8)
        render_lines_3d(ax_tgt, ex["tgt"], "red", linewidth=0.8)
        render_lines_3d(ax_ali, ex["tgt"], "red", linewidth=0.5, alpha=0.35)
        render_lines_3d(ax_ali, ex["src_aligned"], "blue", linewidth=0.8)
        for ax in (ax_src, ax_tgt, ax_ali):
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax_src.set_title(f"source: {ex['name']}")
        ax_tgt.set_title("target (known R, t applied)")
        ax_ali.set_title(f"DRA-like aligned (red=target, blue=predicted)\nrot err = {ex['rot_err']:.2f}°  trans err = {ex['trans_err']:.3f}")
    fig.suptitle(f"Fig {fig_num} analog — ModelNet40 airplane edge alignment (real 3D lines)", y=1.00)
    fig.tight_layout()
    save_close(fig, f"qa_fig_{fig_num.replace('.', '_')}_modelnet_edge_alignment.png")


def fig_table_4_5_errors(results: dict):
    """Table 4.5 analog: rotation + translation errors per method."""
    methods = list(results.keys())
    rot_arr = np.array([results[m]["rot"] for m in methods])
    trans_arr = np.array([results[m]["trans"] for m in methods])
    rot_med = rot_arr.mean(axis=1) if False else np.median(rot_arr, axis=1)
    rot_p25 = np.percentile(rot_arr, 25, axis=1)
    rot_p75 = np.percentile(rot_arr, 75, axis=1)
    trans_med = np.median(trans_arr, axis=1)
    trans_p25 = np.percentile(trans_arr, 25, axis=1)
    trans_p75 = np.percentile(trans_arr, 75, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(len(methods))
    base_colors = ["#999999", "#cc6600", "#22aa44"]
    qa_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(methods) - 3)))
    colors = base_colors + [tuple(c) for c in qa_colors]
    axes[0].bar(x, rot_med, color=colors, edgecolor="black",
                yerr=[rot_med - rot_p25, rot_p75 - rot_med], capsize=5)
    for k, v in enumerate(rot_med):
        axes[0].text(k, v + 1, f"{v:.1f}°", ha="center", fontsize=10)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("rotation error (deg)")
    axes[0].set_title("(a) median rotation error (IQR error bars)")
    axes[0].grid(alpha=0.3, axis="y")

    colors_t = colors
    axes[1].bar(x, trans_med, color=colors_t, edgecolor="black",
                yerr=[trans_med - trans_p25, trans_p75 - trans_med], capsize=5)
    for k, v in enumerate(trans_med):
        axes[1].text(k, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("translation error (norm units)")
    axes[1].set_title("(b) median translation error (IQR error bars)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Table 4.5 analog — line-registration errors on real ModelNet40 airplane edges\n"
                 f"({len(results[methods[0]]['rot'])} test pairs)", y=1.02)
    fig.tight_layout()
    save_close(fig, "qa_table_4_5_modelnet_line_reg_errors.png")
    return {
        m: {
            "rot_median": float(rot_med[k]), "rot_p25": float(rot_p25[k]), "rot_p75": float(rot_p75[k]),
            "trans_median": float(trans_med[k]), "trans_p25": float(trans_p25[k]), "trans_p75": float(trans_p75[k]),
        }
        for k, m in enumerate(methods)
    }


def fig_table_4_6_relative(stats: dict):
    """Table 4.6 analog: DRA-like and QA-DRA-at-various-M relative performance vs RANSAC baseline (positive = improvement)."""
    methods = list(stats.keys())
    base_rot = stats["RANSAC"]["rot_median"]
    base_trans = stats["RANSAC"]["trans_median"]
    rel_rot = [base_rot - stats[m]["rot_median"] for m in methods]
    rel_trans = [base_trans - stats[m]["trans_median"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(len(methods))
    base_colors = ["#cccccc", "#cc6600", "#22aa44"]
    qa_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(methods) - 3)))
    colors = base_colors + [tuple(c) for c in qa_colors]
    bars = axes[0].bar(x, rel_rot, color=colors, edgecolor="black")
    for k, v in enumerate(rel_rot):
        axes[0].text(k, v + (1 if v >= 0 else -3), f"{v:+.1f}°", ha="center", fontsize=10)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("relative rotation improvement vs RANSAC (deg)")
    axes[0].set_title("(a) rotation: higher is better")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].grid(alpha=0.3, axis="y")

    bars = axes[1].bar(x, rel_trans, color=colors, edgecolor="black")
    for k, v in enumerate(rel_trans):
        axes[1].text(k, v + (0.005 if v >= 0 else -0.015), f"{v:+.3f}", ha="center", fontsize=10)
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("relative translation improvement vs RANSAC (units)")
    axes[1].set_title("(b) translation: higher is better")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Table 4.6 analog — relative performance vs RANSAC baseline", y=1.02)
    fig.tight_layout()
    save_close(fig, "qa_table_4_6_modelnet_dra_relative.png")


# ---------- main ----------

def main() -> int:
    print(f"=== QA Ch4 DRA real-edge pipeline ===")
    print(f"  using {AIRPLANE_DIR}")
    print(f"  N_TEST_MESHES = {N_TEST_MESHES}, N_EDGES_PER_MESH = {N_EDGES_PER_MESH}")

    test_pairs = build_test_set()
    results = run_alignment(test_pairs)

    # Print summary
    print("\nPer-method median errors:")
    for m in results:
        med_rot = np.median(results[m]["rot"])
        med_trans = np.median(results[m]["trans"])
        print(f"  {m:>10}  rot = {med_rot:6.2f}°   trans = {med_trans:.3f}")

    # Save raw results
    raw = {m: {"rot": [float(x) for x in results[m]["rot"]],
               "trans": [float(x) for x in results[m]["trans"]]}
           for m in results}
    (OUT_DIR / "dra_alignment_results.json").write_text(json.dumps(raw, indent=2))

    print("\nRendering figures:")
    fig_4_16_17(results, "4.16", example_idx_start=0)
    fig_4_16_17(results, "4.17", example_idx_start=2)
    stats = fig_table_4_5_errors(results)
    fig_table_4_6_relative(stats)

    # Remove the old synthetic versions
    for old in ["qa_fig_4_16_line_alignment.png", "qa_fig_4_17_line_alignment.png",
                "qa_table_4_5_6_line_reg.png"]:
        old_path = OUT_DIR / old
        if old_path.exists():
            old_path.unlink()
            print(f"  removed synthetic {old}")

    print(f"\nAll DRA real-edge figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
