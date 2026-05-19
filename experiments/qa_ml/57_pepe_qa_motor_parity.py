"""QA Motor parity benchmark — does QA-quantized motor composition match continuous SE(3)?

Tests the dual-quaternion CGA motor primitive at multiple QA moduli against
the standard continuous 4×4 matrix composition baseline. The hypothesis is
discrete-fractional parity: at sufficient grid resolution (e.g. m=144) the
QA-quantized motor composition should match continuous to sub-degree
rotation error and sub-millimeter translation error.

Task: compose a chain of N random SE(3) poses sequentially. Compare the
final pose to the continuous reference. Sweep N ∈ {1, 5, 10, 20, 50, 100}
and m ∈ {12, 24, 48, 72, 144, 288}. Drift accumulates with N; the question
is at what m the parity holds across chain lengths.

This is the GA→QA primitive validation Pepe Ch 4's CGAPoseNet relies on:
if motors compose accurately at QA mod-m, then a pose regression head
that outputs QA-quantized motors can be trained without loss of SE(3)
fidelity.

QA_COMPLIANCE = "qa_ml_pepe_qa_motor_parity — continuous CGA motor as observer reference; QA discretization at each compose step"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from tools.qa_ml.qa_motor_v3_3 import (
    motor_from_se3, motor_to_se3, motor_compose, motor_quantize, motor_quantize_se3,
)

OUT_DIR = Path(__file__).parent / "ch4_qa_motor_parity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
N_TRIALS = 30
CHAIN_LENGTHS = [1, 5, 10, 20, 50, 100]
QA_MODS = [12, 24, 48, 72, 144, 288]
T_SCALE = 1.0    # translation samples ~N(0, 0.3) so |t| < 1 with high prob


# ---------- continuous baseline ----------

def random_se3(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    R = Rotation.random(random_state=rng.integers(0, 2**31 - 1)).as_matrix()
    t = rng.normal(scale=0.3, size=3)
    return R.astype(np.float64), t.astype(np.float64)


def compose_continuous(poses: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Standard 4×4 SE(3) compose: T_total = T_1 · T_2 · ... · T_N."""
    R_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros(3, dtype=np.float64)
    for R, t in poses:
        t_acc = R_acc @ t + t_acc
        R_acc = R_acc @ R
    return R_acc, t_acc


def compose_motor_continuous(poses: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Compose via dual-quaternion motors, no quantization (parity self-check)."""
    M = motor_from_se3(np.eye(3), np.zeros(3))
    for R, t in poses:
        M = motor_compose(M, motor_from_se3(R, t))
    return M


def compose_motor_qa(
    poses: list[tuple[np.ndarray, np.ndarray]],
    m: int,
    quantizer=motor_quantize,
) -> np.ndarray:
    """Compose via dual-quaternion motors, quantize after each step."""
    M = motor_from_se3(np.eye(3), np.zeros(3))
    for R, t in poses:
        M_step = motor_from_se3(R, t)
        M = motor_compose(M, M_step)
        M = quantizer(M, m, t_scale=T_SCALE)
    return M


# ---------- error metrics ----------

def rotation_error_deg(R_ref: np.ndarray, R_pred: np.ndarray) -> float:
    Rd = R_ref @ R_pred.T
    tr = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def translation_error(t_ref: np.ndarray, t_pred: np.ndarray) -> float:
    return float(np.linalg.norm(t_pred - t_ref))


# ---------- main sweep ----------

def run_sweep() -> dict:
    """For each (N, m) cell, run N_TRIALS chains; return median errors."""
    print(f"=== QA Motor parity sweep ===")
    print(f"  N_TRIALS = {N_TRIALS}, chain lengths = {CHAIN_LENGTHS}, moduli = {QA_MODS}")
    results = {"chain_lengths": CHAIN_LENGTHS, "moduli": QA_MODS, "cells": {}}
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # First: validate the continuous-motor compose matches the 4×4 baseline (self-check)
    sample_poses = [random_se3(rng) for _ in range(20)]
    R_ref, t_ref = compose_continuous(sample_poses)
    M_motor = compose_motor_continuous(sample_poses)
    R_motor, t_motor = motor_to_se3(M_motor)
    selfcheck_rot = rotation_error_deg(R_ref, R_motor)
    selfcheck_trans = translation_error(t_ref, t_motor)
    print(f"\nSelf-check (motor-compose vs 4×4 matrix on 20-pose chain):")
    print(f"  rotation drift = {selfcheck_rot:.2e}°    translation drift = {selfcheck_trans:.2e}")
    results["selfcheck"] = {"rotation_deg": selfcheck_rot, "translation": selfcheck_trans}

    for n in CHAIN_LENGTHS:
        print(f"\n[ chain length N = {n} ]")
        for m in QA_MODS:
            rot_errs = []
            trans_errs = []
            for trial in range(N_TRIALS):
                poses = [random_se3(rng) for _ in range(n)]
                R_ref, t_ref = compose_continuous(poses)
                M_qa = compose_motor_qa(poses, m, quantizer=motor_quantize)
                R_qa, t_qa = motor_to_se3(M_qa)
                rot_errs.append(rotation_error_deg(R_ref, R_qa))
                trans_errs.append(translation_error(t_ref, t_qa))
            r_med = float(np.median(rot_errs))
            t_med = float(np.median(trans_errs))
            r_p75 = float(np.percentile(rot_errs, 75))
            t_p75 = float(np.percentile(trans_errs, 75))
            key = f"N{n}_m{m}"
            results["cells"][key] = {
                "n": n, "m": m,
                "rot_median": r_med, "rot_p75": r_p75,
                "trans_median": t_med, "trans_p75": t_p75,
            }
            print(f"  m={m:>4}   rot = {r_med:6.3f}° (p75 {r_p75:6.3f}°)   trans = {t_med:.4f} (p75 {t_p75:.4f})")
    print(f"\nSweep finished in {time.time() - t0:.1f}s")
    return results


# ---------- rendering ----------

def render_parity_heatmap(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    rot_grid = np.zeros((len(CHAIN_LENGTHS), len(QA_MODS)))
    trans_grid = np.zeros((len(CHAIN_LENGTHS), len(QA_MODS)))
    for i, n in enumerate(CHAIN_LENGTHS):
        for j, m in enumerate(QA_MODS):
            cell = results["cells"][f"N{n}_m{m}"]
            rot_grid[i, j] = cell["rot_median"]
            trans_grid[i, j] = cell["trans_median"]

    for ax, grid, title, fmt, cbar_label in [
        (axes[0], rot_grid, "median rotation drift (deg)", "{:.2f}°", "rotation drift (deg)"),
        (axes[1], trans_grid, "median translation drift (norm units)", "{:.3f}", "translation drift"),
    ]:
        im = ax.imshow(grid, cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(QA_MODS))); ax.set_xticklabels([f"m={m}" for m in QA_MODS])
        ax.set_yticks(range(len(CHAIN_LENGTHS))); ax.set_yticklabels([f"N={n}" for n in CHAIN_LENGTHS])
        ax.set_title(title)
        ax.set_xlabel("QA modulus")
        if ax is axes[0]:
            ax.set_ylabel("chain length")
        vmax = grid.max()
        for i in range(len(CHAIN_LENGTHS)):
            for j in range(len(QA_MODS)):
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                        color="white" if grid[i, j] > vmax * 0.5 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    fig.suptitle(f"QA Motor parity — drift vs continuous SE(3) compose ({N_TRIALS} trials/cell)", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "qa_motor_parity_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def render_drift_scaling(results: dict):
    """Drift vs chain length, one line per modulus. Shows per-step error budget."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(QA_MODS)))
    for j, m in enumerate(QA_MODS):
        rot_line = [results["cells"][f"N{n}_m{m}"]["rot_median"] for n in CHAIN_LENGTHS]
        trans_line = [results["cells"][f"N{n}_m{m}"]["trans_median"] for n in CHAIN_LENGTHS]
        axes[0].plot(CHAIN_LENGTHS, rot_line, "o-", color=colors[j], label=f"m={m}")
        axes[1].plot(CHAIN_LENGTHS, trans_line, "o-", color=colors[j], label=f"m={m}")
    for ax, ylabel, title in [
        (axes[0], "rotation drift (deg)", "Rotation drift vs chain length"),
        (axes[1], "translation drift (norm units)", "Translation drift vs chain length"),
    ]:
        ax.set_xlabel("number of composed poses N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"QA Motor parity scaling — per-step drift × chain length", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "qa_motor_parity_scaling.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    results = run_sweep()
    (OUT_DIR / "qa_motor_parity_results.json").write_text(json.dumps(results, indent=2))
    print(f"  wrote {OUT_DIR / 'qa_motor_parity_results.json'}")
    print("\nRendering figures:")
    render_parity_heatmap(results)
    render_drift_scaling(results)

    print("\n=== Parity verdict ===")
    print(f"Continuous-motor self-check (dual-quat vs 4×4):  rot {results['selfcheck']['rotation_deg']:.2e}°, trans {results['selfcheck']['translation']:.2e}")
    print(f"At chain length N=10:")
    for m in QA_MODS:
        c = results["cells"][f"N10_m{m}"]
        print(f"  m={m:>4}   rot = {c['rot_median']:6.3f}°   trans = {c['trans_median']:.4f}")
    print(f"\nParity threshold proposal: m ≥ {[m for m in QA_MODS if results['cells'][f'N10_m{m}']['rot_median'] < 1.0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
