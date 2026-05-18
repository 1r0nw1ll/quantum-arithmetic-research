"""Noise σ × overlap sweep of DRA-like vs QA-DRA on ModelNet40 airplane edges.

Tests where the QA-DRA m=72 advantage from `29_pepe_ch4_dra_real_edges.py` holds
and where it breaks. Hypothesis: the QA grid acts as a beneficial regularizer
when the underlying signal is noisy/incomplete, so the QA win should grow with
σ and shrink with overlap; at low σ + high overlap the continuous closed-form
should match or beat QA.

Sweep:
  noise σ      ∈ {0.02, 0.05, 0.10}     per-vertex Gaussian
  overlap     ∈ {0.4, 0.6, 0.8}         fraction of shared edges between views
  methods     = DRA-like, QA-DRA m ∈ {24, 48, 72, 144}

Output:
  qa_dra_sweep_rotation_heatmap.png   median rotation error per method, 3×3 σ×overlap grid
  qa_dra_sweep_advantage_map.png      (DRA-like − QA-DRA m=72), positive = QA wins
  qa_dra_sweep_results.json           full numerical results

QA_COMPLIANCE = "qa_ml_pepe_ch4_dra_noise_overlap_sweep — falsifiability map for QA-DRA m=72 win"
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

# Import the building blocks from script 29 without running its main.
_spec = importlib.util.spec_from_file_location("dra_real_edges", HERE / "29_pepe_ch4_dra_real_edges.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_off = _mod.parse_off
extract_edges = _mod.extract_edges
normalize_and_subsample = _mod.normalize_and_subsample
transform_lines = _mod.transform_lines
align_dra_like = _mod.align_dra_like
qa_quantize_rotation = _mod.qa_quantize_rotation
qa_quantize_translation = _mod.qa_quantize_translation
rotation_error_deg = _mod.rotation_error_deg
translation_error = _mod.translation_error

OUT_DIR = HERE / "ch4_qa_real_pose_replica"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AIRPLANE_DIR = REPO / "corpus" / "modelnet40" / "airplane"

SEED = 0
N_TEST_MESHES = 50
N_EDGES_PER_MESH = 60

SIGMAS = [0.02, 0.05, 0.10]
OVERLAPS = [0.4, 0.6, 0.8]
QA_MODS = [24, 48, 72, 144]


def build_test_set(noise_sigma: float, overlap: float, rng: np.random.Generator) -> list[dict]:
    """Same data construction as 29.build_test_set, parameterized over (σ, overlap)."""
    test_files = sorted((AIRPLANE_DIR / "test").glob("*.off"))[:N_TEST_MESHES]
    out = []
    for f in test_files:
        try:
            verts, tris = parse_off(f)
        except Exception:
            continue
        edges = extract_edges(verts, tris)
        if len(edges) < int(N_EDGES_PER_MESH / max(overlap, 0.1)):
            continue
        edges = normalize_and_subsample(edges, n_keep=10**6, rng=rng)
        R_true = Rotation.random(random_state=rng.integers(0, 2**31 - 1)).as_matrix().astype(np.float32)
        t_true = rng.normal(scale=0.5, size=3).astype(np.float32)
        n_total = len(edges)
        n_overlap = int(N_EDGES_PER_MESH * overlap)
        n_unique = N_EDGES_PER_MESH - n_overlap
        all_idx = rng.permutation(n_total)
        overlap_idx = all_idx[:n_overlap]
        src_extra = all_idx[n_overlap:n_overlap + n_unique]
        tgt_extra = all_idx[n_overlap + n_unique:n_overlap + n_unique * 2]
        if len(src_extra) < n_unique or len(tgt_extra) < n_unique:
            continue
        src_idx = np.concatenate([overlap_idx, src_extra])
        tgt_idx = np.concatenate([overlap_idx, tgt_extra])
        rng.shuffle(src_idx); rng.shuffle(tgt_idx)
        src = edges[src_idx].copy()
        tgt_clean = edges[tgt_idx].copy()
        tgt = transform_lines(tgt_clean, R_true, t_true)
        src = src + rng.normal(scale=noise_sigma, size=src.shape).astype(np.float32)
        tgt = tgt + rng.normal(scale=noise_sigma, size=tgt.shape).astype(np.float32)
        out.append({"src": src, "tgt": tgt, "R_true": R_true, "t_true": t_true})
    return out


def run_condition(noise_sigma: float, overlap: float) -> dict:
    """Run all methods on one (σ, overlap) cell. Returns dict {method: {rot_med, trans_med, rot_iqr, trans_iqr}}."""
    rng = np.random.default_rng(SEED + int(noise_sigma * 1000) * 17 + int(overlap * 100))
    pairs = build_test_set(noise_sigma, overlap, rng)
    methods = ["DRA-like"] + [f"QA-DRA m={m}" for m in QA_MODS]
    raw = {m: {"rot": [], "trans": []} for m in methods}
    for pair in pairs:
        R_cont, t_cont = align_dra_like(pair["src"], pair["tgt"])
        raw["DRA-like"]["rot"].append(rotation_error_deg(pair["R_true"], R_cont))
        raw["DRA-like"]["trans"].append(translation_error(pair["t_true"], t_cont))
        for m in QA_MODS:
            R_q = qa_quantize_rotation(R_cont, m=m)
            t_q = qa_quantize_translation(t_cont, m=m, scale=2.0)
            raw[f"QA-DRA m={m}"]["rot"].append(rotation_error_deg(pair["R_true"], R_q))
            raw[f"QA-DRA m={m}"]["trans"].append(translation_error(pair["t_true"], t_q))
    stats = {}
    for m, d in raw.items():
        rot_arr = np.asarray(d["rot"])
        trans_arr = np.asarray(d["trans"])
        stats[m] = {
            "n": int(len(rot_arr)),
            "rot_median": float(np.median(rot_arr)),
            "rot_p25": float(np.percentile(rot_arr, 25)),
            "rot_p75": float(np.percentile(rot_arr, 75)),
            "trans_median": float(np.median(trans_arr)),
            "trans_p25": float(np.percentile(trans_arr, 25)),
            "trans_p75": float(np.percentile(trans_arr, 75)),
        }
    return stats


def render_rotation_heatmap(results: dict):
    """One subplot per method; each shows a 3×3 σ×overlap heatmap of median rotation error."""
    methods = ["DRA-like"] + [f"QA-DRA m={m}" for m in QA_MODS]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 4.2), squeeze=False)
    axes = axes[0]
    # Collect for shared color scale
    all_vals = []
    for m in methods:
        for si, s in enumerate(SIGMAS):
            for oi, o in enumerate(OVERLAPS):
                all_vals.append(results[f"{s:.2f}_{o:.1f}"][m]["rot_median"])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    for k, m in enumerate(methods):
        grid = np.zeros((len(SIGMAS), len(OVERLAPS)))
        for si, s in enumerate(SIGMAS):
            for oi, o in enumerate(OVERLAPS):
                grid[si, oi] = results[f"{s:.2f}_{o:.1f}"][m]["rot_median"]
        im = axes[k].imshow(grid, cmap="magma_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes[k].set_xticks(range(len(OVERLAPS))); axes[k].set_xticklabels([f"{o:.0%}" for o in OVERLAPS])
        axes[k].set_yticks(range(len(SIGMAS))); axes[k].set_yticklabels([f"σ={s:.2f}" for s in SIGMAS])
        axes[k].set_title(m)
        axes[k].set_xlabel("overlap")
        if k == 0:
            axes[k].set_ylabel("noise σ")
        for si in range(len(SIGMAS)):
            for oi in range(len(OVERLAPS)):
                axes[k].text(oi, si, f"{grid[si, oi]:.1f}°", ha="center", va="center",
                             color="white" if grid[si, oi] > (vmin + vmax) / 2 else "black",
                             fontsize=10)
    fig.subplots_adjust(right=0.92)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("median rotation error (deg)")
    fig.suptitle("σ × overlap sweep — median rotation error per method (50 test pairs / cell)", y=1.02)
    fig.savefig(OUT_DIR / "qa_dra_sweep_rotation_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'qa_dra_sweep_rotation_heatmap.png'}")


def render_advantage_map(results: dict):
    """ΔRot = DRA-like − QA-DRA m=72. Positive = QA wins. Same for translation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax_idx, key in enumerate(["rot_median", "trans_median"]):
        grid = np.zeros((len(SIGMAS), len(OVERLAPS)))
        for si, s in enumerate(SIGMAS):
            for oi, o in enumerate(OVERLAPS):
                cell = results[f"{s:.2f}_{o:.1f}"]
                grid[si, oi] = cell["DRA-like"][key] - cell["QA-DRA m=72"][key]
        vmax = float(np.max(np.abs(grid)))
        im = axes[ax_idx].imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[ax_idx].set_xticks(range(len(OVERLAPS))); axes[ax_idx].set_xticklabels([f"{o:.0%}" for o in OVERLAPS])
        axes[ax_idx].set_yticks(range(len(SIGMAS))); axes[ax_idx].set_yticklabels([f"σ={s:.2f}" for s in SIGMAS])
        axes[ax_idx].set_xlabel("overlap")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("noise σ")
        fmt = "{:+.2f}°" if key == "rot_median" else "{:+.3f}"
        for si in range(len(SIGMAS)):
            for oi in range(len(OVERLAPS)):
                axes[ax_idx].text(oi, si, fmt.format(grid[si, oi]), ha="center", va="center",
                                  color="black", fontsize=10)
        axes[ax_idx].set_title(f"Δ {'rotation (deg)' if key == 'rot_median' else 'translation'} = DRA-like − QA-DRA m=72\n(red = QA wins, blue = continuous wins)")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
    fig.suptitle("QA-DRA m=72 advantage over continuous DRA-like across σ × overlap", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_dra_sweep_advantage_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'qa_dra_sweep_advantage_map.png'}")


def main() -> int:
    print(f"=== Noise × overlap sweep for QA-DRA ===")
    print(f"  σ ∈ {SIGMAS}    overlap ∈ {OVERLAPS}    M_QA ∈ {QA_MODS}")
    print(f"  {len(SIGMAS) * len(OVERLAPS)} conditions × ~50 pairs × {1 + len(QA_MODS)} methods")
    t0 = time.time()
    results = {}
    for s in SIGMAS:
        for o in OVERLAPS:
            label = f"{s:.2f}_{o:.1f}"
            print(f"\n[ σ={s:.2f}  overlap={o:.0%} ]")
            stats = run_condition(s, o)
            results[label] = stats
            for m in ["DRA-like"] + [f"QA-DRA m={m}" for m in QA_MODS]:
                print(f"  {m:>15}  rot = {stats[m]['rot_median']:5.2f}°   trans = {stats[m]['trans_median']:.3f}   (n={stats[m]['n']})")
    print(f"\nSweep finished in {time.time() - t0:.1f}s")

    (OUT_DIR / "qa_dra_sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"  wrote {OUT_DIR / 'qa_dra_sweep_results.json'}")

    print("\nRendering figures:")
    render_rotation_heatmap(results)
    render_advantage_map(results)

    # Final summary line
    print("\n=== Δ rotation (DRA-like − QA-DRA m=72), positive = QA wins ===")
    for s in SIGMAS:
        row = []
        for o in OVERLAPS:
            cell = results[f"{s:.2f}_{o:.1f}"]
            d = cell["DRA-like"]["rot_median"] - cell["QA-DRA m=72"]["rot_median"]
            row.append(f"{d:+5.2f}°")
        print(f"  σ={s:.2f}   ovl 40%={row[0]}   60%={row[1]}   80%={row[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
