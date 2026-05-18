"""QA Chapter 3 visual replica with REAL proteins.

Where 24_pepe_ch3_qa_visual_replica.py used abstract QA orbit chains,
THIS script uses real PDB structures (the proteins Pepe actually shows
in Chapter 3) and applies QA as the computational lens on top of real
biology:

  Pepe                            Real biology preserved                QA enters where
  ----                            ----------------------               ----------------
  Backbone N-Cα-C planes          Parsed from PDB ATOM records         (b, e) ← discretized (Φ, Ψ) per residue
  Rotor R_ij between planes       SO(3) rotor from real 3D coords      ϕ_ij snapped to QA mod-m grid
  Cost C_ij = 2(1−cos ϕ)          Range [0, 2] preserved              Cost from QA-quantized ϕ_ij
  Contact d_ij < 15 Å             Cα-Cα Euclidean distance in Å        gating threshold unchanged
  Secondary structure             DSSP-like Φ/Ψ heuristic              Family color overlay
  Dihedral angles Φ, Ψ, Ω         Real residue dihedrals               QA mod-m bucketed copies
  Dot product maps NCα/NCβ/NN     Real oriented-point inner products   QA-quantized direction vectors

Proteins covered (matching Pepe Ch3 figures):
  Fig 3.1: 1dmp (HIV-1 protease)
  Fig 3.2: 2hc5 (chain 2HC5A)
  Fig 3.3: 4jzk (chain 4JZK)
  Fig 3.5/3.6: 1laf vs 2lao (Lys/Arg/Orn-binding protein conformations)
  Fig 3.9/3.10: 1bvm (NMR ensemble)
  Fig 3.11: 3i40 (Insulin)
  Fig 3.12: 12as (chain 12asA)
  Fig 3.13: 1a3n (Haemoglobin)
  Fig 3.14: 3i41 (chain 3i41A)

QA_COMPLIANCE = "qa_ml_pepe_ch3_real_protein_replica — observer-side; real PDB coords; QA discretization of rotor angles; A1/A2 compliant"
"""

from __future__ import annotations

import sys
from math import cos, pi, sin
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).parent / "ch3_qa_real_protein_replica"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDB_DIR = REPO / "corpus" / "pdb_pepe_ch3"

M_QA = 24                      # QA modulus — discretization grid for rotation angles
CONTACT_ANGSTROMS = 15.0       # Pepe's 15 Å contact threshold
MAX_RESIDUES = 300             # truncate long chains for visualization (Pepe also crops)

SECSTRUCT_COLOR = {
    "H": "#cc0000",   # α-helix (red)
    "E": "#229922",   # β-sheet (green)
    "T": "#2266cc",   # turn (blue)
    "C": "#cccccc",   # coil / other (grey)
}


# ---------- PDB parsing ----------

def load_backbone(pdb_path: Path, chain_id: Optional[str] = None) -> dict:
    """Parse N-Cα-C atom coordinates per residue from a PDB file.

    Returns dict with:
      residues: list of (chain, resnum, resname)
      N, Ca, C: float64[L, 3] arrays of atom coordinates in Å
      Cb: float64[L, 3] — Cβ if present, else nan (glycine)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())  # first model only
    residues_out, N, Ca, C, Cb = [], [], [], [], []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            try:
                n_atom = res["N"].get_coord()
                ca_atom = res["CA"].get_coord()
                c_atom = res["C"].get_coord()
            except KeyError:
                continue
            cb_atom = res["CB"].get_coord() if "CB" in res else np.array([np.nan, np.nan, np.nan])
            residues_out.append((chain.id, res.id[1], res.get_resname()))
            N.append(n_atom)
            Ca.append(ca_atom)
            C.append(c_atom)
            Cb.append(cb_atom)
        if chain_id is None:
            break  # first chain only by default
    return {
        "residues": residues_out,
        "N": np.asarray(N, dtype=np.float64),
        "Ca": np.asarray(Ca, dtype=np.float64),
        "C": np.asarray(C, dtype=np.float64),
        "Cb": np.asarray(Cb, dtype=np.float64),
    }


def load_nmr_ensemble(pdb_path: Path, chain_id: Optional[str] = None, max_models: int = 20) -> list[dict]:
    """Parse all models from an NMR ensemble PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    out = []
    for model in structure.get_models():
        residues_out, N, Ca, C, Cb = [], [], [], [], []
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                try:
                    n_atom = res["N"].get_coord()
                    ca_atom = res["CA"].get_coord()
                    c_atom = res["C"].get_coord()
                except KeyError:
                    continue
                cb_atom = res["CB"].get_coord() if "CB" in res else np.array([np.nan, np.nan, np.nan])
                residues_out.append((chain.id, res.id[1], res.get_resname()))
                N.append(n_atom); Ca.append(ca_atom); C.append(c_atom); Cb.append(cb_atom)
            if chain_id is None:
                break
        out.append({
            "residues": residues_out,
            "N": np.asarray(N, dtype=np.float64),
            "Ca": np.asarray(Ca, dtype=np.float64),
            "C": np.asarray(C, dtype=np.float64),
            "Cb": np.asarray(Cb, dtype=np.float64),
        })
        if len(out) >= max_models:
            break
    return out


# ---------- backbone geometry ----------

def plane_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Unit normal to the plane through 3 points."""
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    return n / max(norm, 1e-9)


def plane_normals(N: np.ndarray, Ca: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-residue plane normals from N-Cα-C triplets."""
    L = N.shape[0]
    out = np.empty((L, 3), dtype=np.float64)
    for i in range(L):
        out[i] = plane_normal(N[i], Ca[i], C[i])
    return out


def rotor_angle(n1: np.ndarray, n2: np.ndarray) -> float:
    """Rotation angle (radians) between two plane normals. The plane-rotation
    angle is the angle between the normals."""
    c = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return float(np.arccos(c))


def dihedral_angle(p1, p2, p3, p4) -> float:
    """Dihedral angle around the p2-p3 axis (Praxeolitic formula, standard convention).
    Returns angle in radians in (-π, π]."""
    b1 = p1 - p2
    b2 = p3 - p2
    b3 = p4 - p3
    b2_unit = b2 / max(np.linalg.norm(b2), 1e-9)
    # Project b1 and b3 onto plane perpendicular to b2
    v = b1 - np.dot(b1, b2_unit) * b2_unit
    w = b3 - np.dot(b3, b2_unit) * b2_unit
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b2_unit, v), w))
    return float(np.arctan2(y, x))


def per_residue_dihedrals(bb: dict) -> dict:
    """Φ (C_{i-1}–N_i–Cα_i–C_i), Ψ (N_i–Cα_i–C_i–N_{i+1}), Ω (Cα_{i-1}–C_{i-1}–N_i–Cα_i)."""
    L = bb["N"].shape[0]
    phi = np.full(L, np.nan)
    psi = np.full(L, np.nan)
    omega = np.full(L, np.nan)
    for i in range(L):
        if i > 0:
            phi[i] = dihedral_angle(bb["C"][i - 1], bb["N"][i], bb["Ca"][i], bb["C"][i])
            omega[i] = dihedral_angle(bb["Ca"][i - 1], bb["C"][i - 1], bb["N"][i], bb["Ca"][i])
        if i < L - 1:
            psi[i] = dihedral_angle(bb["N"][i], bb["Ca"][i], bb["C"][i], bb["N"][i + 1])
    return {"phi": phi, "psi": psi, "omega": omega}


def secondary_structure_heuristic(phi: np.ndarray, psi: np.ndarray) -> list[str]:
    """Ramachandran-region classifier with standard ranges + smoothing.
    α-helix region: φ ∈ [-160, -30], ψ ∈ [-90, 30]; cluster center (-60, -45).
    β-sheet region: φ ∈ [-180, -45], ψ ∈ [60, 180].
    Anything else with negative φ → turn; positive φ → coil."""
    out = []
    for f, p in zip(phi, psi):
        if np.isnan(f) or np.isnan(p):
            out.append("C")
            continue
        f_deg = np.degrees(f)
        p_deg = np.degrees(p)
        if -160 <= f_deg <= -30 and -90 <= p_deg <= 30:
            out.append("H")
        elif -180 <= f_deg <= -45 and 60 <= p_deg <= 180:
            out.append("E")
        elif f_deg < 0:
            out.append("T")
        else:
            out.append("C")

    # Smooth: isolated H/E surrounded by C is downgraded to T; runs of 3+ H/E are kept
    smoothed = list(out)
    L = len(out)
    for i in range(L):
        if smoothed[i] in ("H", "E"):
            left = smoothed[max(0, i - 1)] if i > 0 else "C"
            right = smoothed[min(L - 1, i + 1)] if i < L - 1 else "C"
            if left != smoothed[i] and right != smoothed[i]:
                smoothed[i] = "T"
    return smoothed


# ---------- QA discretization ----------

def qa_state_from_angle_pair(phi: float, psi: float, m: int) -> tuple[int, int]:
    """Map a (Φ, Ψ) dihedral pair to a QA state (b, e) ∈ {1..m}^2.
    The mapping bins each angle into {1..m} by wrapping to [0, 2π).
    A1 compliant — never hits 0."""
    if np.isnan(phi):
        phi = 0.0
    if np.isnan(psi):
        psi = 0.0
    b = int(((phi % (2 * pi)) / (2 * pi) * m)) + 1
    e = int(((psi % (2 * pi)) / (2 * pi) * m)) + 1
    return min(max(b, 1), m), min(max(e, 1), m)


def qa_quantize_angle(angle_rad: float, m: int) -> float:
    """Snap an angle to the QA mod-m grid: nearest multiple of 2π/m."""
    grid = 2 * pi / m
    k = round((angle_rad % (2 * pi)) / grid)
    return float(k * grid)


# ---------- map computation (real biology + QA discretization) ----------

def compute_pepe_qa_maps(bb: dict, m: int, contact_a: float) -> dict:
    L = bb["Ca"].shape[0]
    # Real Cα-Cα distance
    diff = bb["Ca"][:, None, :] - bb["Ca"][None, :, :]
    dist_a = np.linalg.norm(diff, axis=2)
    contact = (dist_a < contact_a).astype(np.uint8)

    # Plane normals + rotor angles (real 3D)
    normals = plane_normals(bb["N"], bb["Ca"], bb["C"])
    phi_real = np.zeros((L, L), dtype=np.float64)
    phi_qa = np.zeros((L, L), dtype=np.float64)
    cost_real = np.zeros((L, L), dtype=np.float64)
    cost_qa = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            if not contact[i, j]:
                continue
            ang = rotor_angle(normals[i], normals[j])
            phi_real[i, j] = ang
            cost_real[i, j] = 2.0 * (1.0 - cos(ang))
            ang_qa = qa_quantize_angle(ang, m)
            phi_qa[i, j] = ang_qa
            cost_qa[i, j] = 2.0 * (1.0 - cos(ang_qa))

    return {
        "contact": contact,
        "distance_a": dist_a,
        "phi_real": phi_real,
        "phi_qa": phi_qa,
        "cost_real": cost_real,
        "cost_qa": cost_qa,
        "normals": normals,
    }


def compute_qa_angle_maps(bb: dict, dihedrals: dict, m: int, contact_a: float) -> dict:
    """Three angle maps Φ_ij, Ψ_ij, Ω_ij in QA discretization gated by contact.
    Pepe defines Φ_ij/Ψ_ij/Ω_ij as the dihedral angle of residue i in the
    context of pair (i, j), gated by d_ij < 15Å. Simplest faithful version:
    Φ_ij = QA-quantized(Φ_i) if d_ij < 15Å else 0. Similarly for Ψ, Ω."""
    L = bb["Ca"].shape[0]
    diff = bb["Ca"][:, None, :] - bb["Ca"][None, :, :]
    dist_a = np.linalg.norm(diff, axis=2)
    contact = (dist_a < contact_a)

    def quantize_with_sign(a: float) -> float:
        if np.isnan(a):
            return 0.0
        # Wrap to [-π, π] then snap to mod-m grid in that range
        a = ((a + pi) % (2 * pi)) - pi
        grid = 2 * pi / m
        return float(round(a / grid) * grid)

    phi_q = np.array([quantize_with_sign(v) for v in dihedrals["phi"]])
    psi_q = np.array([quantize_with_sign(v) for v in dihedrals["psi"]])
    omega_q = np.array([quantize_with_sign(v) for v in dihedrals["omega"]])

    phi_map = np.where(contact, phi_q[:, None], 0.0)
    psi_map = np.where(contact, psi_q[:, None], 0.0)
    omega_map = np.where(contact, omega_q[:, None], 0.0)
    return {"phi": phi_map, "psi": psi_map, "omega": omega_map}


def compute_qa_dotprod_maps(bb: dict, m: int, contact_a: float) -> dict:
    """Three QA-discretized oriented-point dot-product maps mimicking NCα/NCβ/NN."""
    L = bb["Ca"].shape[0]
    diff = bb["Ca"][:, None, :] - bb["Ca"][None, :, :]
    dist_a = np.linalg.norm(diff, axis=2)
    contact = (dist_a < contact_a)

    # Oriented point at each atom site: position + plane normal.
    # Normal direction is plane(N, Cα, C) for all three (no Cβ-plane variation here)
    # but we vary the BASE POINT and the QA-discretized direction.
    normals = plane_normals(bb["N"], bb["Ca"], bb["C"])

    def quantize_vec(v: np.ndarray, m: int) -> np.ndarray:
        """Quantize a unit vector by converting to (θ, φ) spherical and snapping
        each angle to the QA mod-m grid."""
        x, y, z = v
        theta = np.arctan2(np.sqrt(x * x + y * y), z)  # polar
        phi = np.arctan2(y, x)                          # azimuth
        theta_q = qa_quantize_angle(theta, m)
        phi_q = qa_quantize_angle(phi, m)
        return np.array([
            np.sin(theta_q) * np.cos(phi_q),
            np.sin(theta_q) * np.sin(phi_q),
            np.cos(theta_q),
        ], dtype=np.float64)

    normals_q = np.array([quantize_vec(n, m) for n in normals], dtype=np.float64)

    # Three "atom-site" oriented points: position at Cα/Cβ/N + QA-quantized normal.
    # Different bases give different "atomic" perspectives even though all share
    # the same plane normal — distinguishing feature is the BASE-POINT direction
    # (Cα→Cβ axis vs Cα→N axis) projected onto the normal direction.
    Ca = bb["Ca"]
    Cb_valid = ~np.isnan(bb["Cb"]).any(axis=1)

    def site_oriented_points(base_choice: str):
        ops = np.zeros_like(Ca)
        if base_choice == "nca":
            for i in range(L):
                axis = bb["N"][i] - bb["Ca"][i]
                axis = axis / max(np.linalg.norm(axis), 1e-9)
                ops[i] = axis
        elif base_choice == "ncb":
            for i in range(L):
                if Cb_valid[i]:
                    axis = bb["Cb"][i] - bb["Ca"][i]
                    axis = axis / max(np.linalg.norm(axis), 1e-9)
                else:
                    axis = np.zeros(3)  # glycine
                ops[i] = axis
        else:  # nn
            for i in range(L):
                axis = bb["C"][i] - bb["N"][i]
                axis = axis / max(np.linalg.norm(axis), 1e-9)
                ops[i] = axis
        # QA-quantize the axis directions
        return np.array([quantize_vec(v, m) if np.linalg.norm(v) > 0 else v for v in ops])

    maps = {}
    for name in ("nca", "ncb", "nn"):
        ops_q = site_oriented_points(name)
        mat = np.zeros((L, L), dtype=np.float64)
        for i in range(L):
            for j in range(L):
                if not contact[i, j]:
                    continue
                # Cosine of angle between site axes; preserves sign
                ni = ops_q[i]; nj = ops_q[j]
                if np.linalg.norm(ni) < 1e-6 or np.linalg.norm(nj) < 1e-6:
                    continue
                mat[i, j] = float(np.dot(ni, nj))
        maps[name] = mat
    return maps


# ---------- plotting ----------

def save_close(fig, name: str):
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def fig_3_1(bb: dict, pdb_id: str):
    """Real backbone as N-Cα-C planes, full + close-up."""
    L = bb["N"].shape[0]
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    Ca = bb["Ca"]
    ax1.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-k", linewidth=0.7, alpha=0.7)
    # Plane triangles per residue, every 8th to reduce clutter
    for i in range(0, L, max(1, L // 25)):
        tri = np.array([bb["N"][i], bb["Ca"][i], bb["C"][i], bb["N"][i]])
        ax1.plot(tri[:, 0], tri[:, 1], tri[:, 2], "-", color="C0", linewidth=1.2, alpha=0.6)
    ax1.set_xlabel("x (Å)"); ax1.set_ylabel("y (Å)"); ax1.set_zlabel("z (Å)")
    ax1.set_title(f"{pdb_id.upper()} — backbone as N-Cα-C planes (L={L} residues)")
    # Close-up of first 20 residues
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    L0 = min(20, L)
    Ca0 = bb["Ca"][:L0]
    ax2.plot(Ca0[:, 0], Ca0[:, 1], Ca0[:, 2], "-k", linewidth=1.4)
    for i in range(L0):
        tri = np.array([bb["N"][i], bb["Ca"][i], bb["C"][i], bb["N"][i]])
        ax2.plot(tri[:, 0], tri[:, 1], tri[:, 2], "-", color="C0", linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("x (Å)"); ax2.set_ylabel("y (Å)"); ax2.set_zlabel("z (Å)")
    ax2.set_title("first 20 residues — close-up")
    fig.suptitle(f"Fig 3.1 analog — {pdb_id.upper()} backbone planes (real PDB coords)", y=1.02)
    save_close(fig, f"qa_fig_3_1_{pdb_id}_backbone.png")


def fig_3_2(maps: dict, pdb_id: str, chain_id: str):
    """Contact / distance / cost maps for real protein (cost from QA-quantized ϕ)."""
    L = maps["contact"].shape[0]
    Llim = min(MAX_RESIDUES, L)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].imshow(maps["contact"][:Llim, :Llim], cmap="Greys", origin="lower")
    axes[0].set_title(f"(a) contact map (Cα–Cα < {CONTACT_ANGSTROMS}Å)")
    im1 = axes[1].imshow(maps["distance_a"][:Llim, :Llim], cmap="viridis", origin="lower", vmax=25)
    axes[1].set_title("(b) distance map (Cα–Cα, Å)")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)
    im2 = axes[2].imshow(maps["cost_qa"][:Llim, :Llim], cmap="inferno", origin="lower", vmin=0, vmax=2)
    axes[2].set_title(f"(c) QA cost map (φ snapped to mod-{M_QA} grid)")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)
    for ax in axes:
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
    fig.suptitle(f"Fig 3.2 analog — protein chain {pdb_id.upper()}{chain_id} (real biology + QA cost)", y=1.02)
    save_close(fig, f"qa_fig_3_2_{pdb_id}_three_maps.png")


def fig_3_3(maps: dict, secstruct: list[str], pdb_id: str, chain_id: str):
    """Cost map with secondary-structure overlay (real DSSP-heuristic labels)."""
    L = maps["contact"].shape[0]
    Llim = min(MAX_RESIDUES, L)
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_axes([0.06, 0.08, 0.6, 0.74])
    im = ax.imshow(maps["cost_qa"][:Llim, :Llim], cmap="inferno", origin="lower", vmin=0, vmax=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("QA cost C = 2(1−cos φ_QA)")
    ax.set_xlabel("residue j")
    ax.set_ylabel("residue i")
    # Secondary structure bar
    bar_top = fig.add_axes([0.06, 0.84, 0.6 * 0.93, 0.03])
    for k in range(Llim):
        bar_top.add_patch(plt.Rectangle((k, 0), 1, 1, color=SECSTRUCT_COLOR[secstruct[k]]))
    bar_top.set_xlim(0, Llim); bar_top.set_ylim(0, 1)
    bar_top.set_xticks([]); bar_top.set_yticks([])
    bar_top.set_title("secondary structure (red=α-helix, green=β-sheet, blue=turn, grey=coil)", fontsize=10)
    # Histogram
    ax2 = fig.add_axes([0.74, 0.13, 0.23, 0.69])
    from collections import Counter
    counts = Counter(secstruct[:Llim])
    cats = ["H", "E", "T", "C"]
    ax2.bar(cats, [counts.get(c, 0) for c in cats], color=[SECSTRUCT_COLOR[c] for c in cats])
    ax2.set_title("secondary-structure distribution")
    ax2.set_ylabel("residue count")
    fig.suptitle(f"Fig 3.3 analog — {pdb_id.upper()}{chain_id} cost map + secondary structure", y=0.97)
    out_path = OUT_DIR / f"qa_fig_3_3_{pdb_id}_cost_secstruct.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def fig_3_4(proteins: list[tuple[str, dict, dict]]):
    """Multiple proteins — same QA cost-map methodology, varied patterns."""
    n = len(proteins)
    nrow = 2
    ncol = (n + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    for ax, (pdb_id, bb, maps) in zip(axes.flat, proteins):
        L = min(MAX_RESIDUES, maps["contact"].shape[0])
        ax.imshow(maps["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
        ax.set_title(f"{pdb_id.upper()} (L={L})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(proteins):]:
        ax.axis("off")
    fig.suptitle("Fig 3.4 analog — QA cost maps across proteins (real PDB + QA-quantized φ)", y=1.005)
    save_close(fig, "qa_fig_3_4_multi_protein.png")


def fig_3_5_6(bb_a: dict, bb_b: dict, maps_a: dict, maps_b: dict, label_a: str, label_b: str):
    """Two conformations (real proteins) and cost-map difference."""
    # 3D structures (Fig 3.5)
    fig5 = plt.figure(figsize=(15, 6))
    ax1 = fig5.add_subplot(1, 2, 1, projection="3d")
    Ca = bb_a["Ca"]; ax1.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="cyan", linewidth=1.4)
    ax1.scatter(Ca[:, 0], Ca[:, 1], Ca[:, 2], color="cyan", s=10, edgecolor="black", linewidth=0.3)
    ax1.set_title(label_a)
    ax2 = fig5.add_subplot(1, 2, 2, projection="3d")
    Ca = bb_b["Ca"]; ax2.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="magenta", linewidth=1.4)
    ax2.scatter(Ca[:, 0], Ca[:, 1], Ca[:, 2], color="magenta", s=10, edgecolor="black", linewidth=0.3)
    ax2.set_title(label_b)
    for ax in (ax1, ax2):
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    fig5.suptitle("Fig 3.5 analog — two conformations (real PDB)", y=1.02)
    save_close(fig5, "qa_fig_3_5_two_conformers.png")

    # Cost map difference (Fig 3.6)
    L = min(maps_a["cost_qa"].shape[0], maps_b["cost_qa"].shape[0], MAX_RESIDUES)
    fig6, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = axes[0].imshow(maps_a["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    axes[0].set_title(f"(a) {label_a}")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)
    im1 = axes[1].imshow(maps_b["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    axes[1].set_title(f"(b) {label_b}")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)
    diff = maps_a["cost_qa"][:L, :L] - maps_b["cost_qa"][:L, :L]
    im2 = axes[2].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-2, vmax=2)
    axes[2].set_title("(c) M_A − M_B")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)
    for ax in axes:
        ax.set_xlabel("residue j"); ax.set_ylabel("residue i")
    fig6.suptitle("Fig 3.6 analog — QA cost-map difference between two real-protein conformations", y=1.02)
    save_close(fig6, "qa_fig_3_6_cost_diff.png")


def fig_3_11(bb: dict, dihedrals: dict, pdb_id: str):
    """Per-residue Φ/Ψ/Ω with QA discretization overlay."""
    L = min(30, bb["N"].shape[0])
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    Ca = bb["Ca"][:L]
    ax1.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-k", linewidth=1.6)
    ax1.scatter(Ca[:, 0], Ca[:, 1], Ca[:, 2], c=range(L), cmap="viridis", s=40, edgecolor="k")
    for k in range(0, L, 5):
        ax1.text(Ca[k, 0], Ca[k, 1], Ca[k, 2] + 0.5, f"{k}", fontsize=8)
    ax1.set_xlabel("x (Å)"); ax1.set_ylabel("y (Å)"); ax1.set_zlabel("z (Å)")
    ax1.set_title(f"{pdb_id.upper()} backbone close-up (residues 0..{L-1})")

    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(L)
    phi = dihedrals["phi"][:L]; psi = dihedrals["psi"][:L]; omega = dihedrals["omega"][:L]
    phi_q = np.array([qa_quantize_angle(v, M_QA) if not np.isnan(v) else np.nan for v in phi])
    psi_q = np.array([qa_quantize_angle(v, M_QA) if not np.isnan(v) else np.nan for v in psi])
    omega_q = np.array([qa_quantize_angle(v, M_QA) if not np.isnan(v) else np.nan for v in omega])
    ax2.plot(x, phi, "-o", label="Φ_i (real)", color="C0", alpha=0.5)
    ax2.plot(x, phi_q, "x", label="Φ_i (QA mod-24)", color="C0", markersize=6)
    ax2.plot(x, psi, "-s", label="Ψ_i (real)", color="C1", alpha=0.5)
    ax2.plot(x, psi_q, "x", label="Ψ_i (QA mod-24)", color="C1", markersize=6)
    ax2.plot(x, omega, "-^", label="Ω_i (real)", color="C2", alpha=0.5)
    ax2.set_xlabel("residue index i"); ax2.set_ylabel("angle (rad)")
    ax2.set_title(f"{pdb_id.upper()} per-residue dihedrals: real vs QA mod-{M_QA}")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.legend(fontsize=8)
    fig.suptitle(f"Fig 3.11 analog — {pdb_id.upper()} dihedral close-up", y=1.02)
    save_close(fig, f"qa_fig_3_11_{pdb_id}_dihedrals.png")


def fig_3_12(angles: dict, pdb_id: str):
    """Three angle maps Φ / Ψ / Ω on real protein, QA-discretized."""
    L = min(MAX_RESIDUES, angles["phi"].shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, mat) in zip(axes, [("Φ (QA-quantized)", angles["phi"]),
                                       ("Ψ (QA-quantized)", angles["psi"]),
                                       ("Ω (QA-quantized)", angles["omega"])]):
        im = ax.imshow(mat[:L, :L], cmap="RdBu_r", origin="lower", vmin=-pi, vmax=pi)
        ax.set_title(name); ax.set_xlabel("residue j"); ax.set_ylabel("residue i")
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Fig 3.12 analog — {pdb_id.upper()} QA-quantized dihedral angle maps", y=1.02)
    save_close(fig, f"qa_fig_3_12_{pdb_id}_angle_maps.png")


def fig_3_14(dotmaps: dict, pdb_id: str):
    """NCα / NCβ / NN QA-quantized dot product maps."""
    L = min(MAX_RESIDUES, dotmaps["nca"].shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, mat) in zip(axes, [("NCα (Cα base, QA-quantized normal)", dotmaps["nca"]),
                                       ("NCβ (Cβ base, QA-quantized normal)", dotmaps["ncb"]),
                                       ("NN (N base, QA-quantized normal)", dotmaps["nn"])]):
        vmax = max(0.05, np.abs(mat).max())
        im = ax.imshow(mat[:L, :L], cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(name); ax.set_xlabel("residue j"); ax.set_ylabel("residue i")
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Fig 3.14 analog — {pdb_id.upper()} QA-quantized oriented-point dot products", y=1.02)
    save_close(fig, f"qa_fig_3_14_{pdb_id}_dotprod_maps.png")


def fig_3_7(bb_a: dict, bb_b: dict, label_a: str, label_b: str, pdb_a: str, pdb_b: str):
    """Two 3D structures (SARS-CoV-2 spike open vs closed) — Fig 3.7 analog."""
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    Ca = bb_a["Ca"]
    ax1.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="#1f77b4", linewidth=1.0, alpha=0.85)
    ax1.set_title(f"{pdb_a.upper()} — {label_a} ({Ca.shape[0]} residues)")
    ax1.set_xlabel("x (Å)"); ax1.set_ylabel("y (Å)"); ax1.set_zlabel("z (Å)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    Ca = bb_b["Ca"]
    ax2.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="#ff7f0e", linewidth=1.0, alpha=0.85)
    ax2.set_title(f"{pdb_b.upper()} — {label_b} ({Ca.shape[0]} residues)")
    ax2.set_xlabel("x (Å)"); ax2.set_ylabel("y (Å)"); ax2.set_zlabel("z (Å)")
    fig.suptitle("Fig 3.7 analog — SARS-CoV-2 spike conformations (real PDB)", y=1.02)
    save_close(fig, f"qa_fig_3_7_{pdb_a}_{pdb_b}_two_conformers.png")


def fig_3_8(maps_a: dict, maps_b: dict, pdb_a: str, pdb_b: str, label_a: str, label_b: str, n_first: int = 500):
    """Cost maps + difference for SARS-CoV-2 conformations — Fig 3.8 analog (first 500 residues)."""
    L = min(maps_a["cost_qa"].shape[0], maps_b["cost_qa"].shape[0], n_first)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = axes[0].imshow(maps_a["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    axes[0].set_title(f"(a) {pdb_a.upper()} — {label_a}")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)
    im1 = axes[1].imshow(maps_b["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    axes[1].set_title(f"(b) {pdb_b.upper()} — {label_b}")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)
    diff = maps_a["cost_qa"][:L, :L] - maps_b["cost_qa"][:L, :L]
    im2 = axes[2].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-2, vmax=2)
    axes[2].set_title("(c) M_A − M_B")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)
    for ax in axes:
        ax.set_xlabel("residue j"); ax.set_ylabel("residue i")
    fig.suptitle(f"Fig 3.8 analog — first {L} residues of {pdb_a.upper()} vs {pdb_b.upper()}", y=1.02)
    save_close(fig, f"qa_fig_3_8_{pdb_a}_{pdb_b}_cost_diff.png")


def fig_3_9(ensemble: list[dict], pdb_id: str):
    """NMR ensemble in 3D — 3 sample models + complete overlay, Fig 3.9 analog."""
    K = len(ensemble)
    fig = plt.figure(figsize=(16, 4.6))
    sample_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx in range(3):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
        for m in ensemble:
            Ca = m["Ca"]
            ax.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="gray", linewidth=0.5, alpha=0.18)
        Ca = ensemble[idx]["Ca"]
        ax.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color=sample_colors[idx], linewidth=1.4)
        ax.set_title(f"NMR model {idx + 1}")
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    for m in ensemble:
        Ca = m["Ca"]
        ax.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", linewidth=0.6, alpha=0.4)
    ax.set_title(f"complete ensemble (K={K})")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    fig.suptitle(f"Fig 3.9 analog — {pdb_id.upper()} NMR ensemble (real PDB models)", y=1.05)
    save_close(fig, f"qa_fig_3_9_{pdb_id}_ensemble.png")


def fig_3_10(ensemble: list[dict], pdb_id: str):
    """NMR pairwise cost-map differences and average, Fig 3.10 analog."""
    K = len(ensemble)
    cost_maps = [compute_pepe_qa_maps(m, M_QA, CONTACT_ANGSTROMS)["cost_qa"] for m in ensemble]
    L = min(MAX_RESIDUES, cost_maps[0].shape[0])
    # Three sample cost maps + average pairwise |M_k - M_{k+1}|
    avg_diff = np.zeros((L, L), dtype=np.float64)
    for k in range(K - 1):
        avg_diff += np.abs(cost_maps[k][:L, :L] - cost_maps[k + 1][:L, :L])
    avg_diff /= max(K - 1, 1)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for idx in range(3):
        im = axes[idx].imshow(cost_maps[idx][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
        axes[idx].set_title(f"(a)-(c) NMR model {idx + 1} cost")
        plt.colorbar(im, ax=axes[idx], shrink=0.7)
    im3 = axes[3].imshow(avg_diff, cmap="magma", origin="lower", vmin=0, vmax=1)
    axes[3].set_title(f"(d) avg pairwise |M_k − M_{{k+1}}| (K={K})")
    plt.colorbar(im3, ax=axes[3], shrink=0.7)
    for ax in axes:
        ax.set_xlabel("residue j"); ax.set_ylabel("residue i")
    fig.suptitle(f"Fig 3.10 analog — {pdb_id.upper()} NMR pairwise cost-map difference", y=1.02)
    save_close(fig, f"qa_fig_3_10_{pdb_id}_ensemble_avg_diff.png")


def fig_3_13(bb: dict, pdb_id: str):
    """Backbone with oriented-point normals overlay — Fig 3.13 analog (hemoglobin)."""
    L = bb["N"].shape[0]
    normals = plane_normals(bb["N"], bb["Ca"], bb["C"])
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    Ca = bb["Ca"]
    ax1.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-k", linewidth=0.7, alpha=0.7)
    # Oriented-point arrows (Cα as base, normal as direction); sample every 5th
    arrow_len = 3.0
    for i in range(0, L, max(1, L // 25)):
        nrm = normals[i] * arrow_len
        ax1.quiver(Ca[i, 0], Ca[i, 1], Ca[i, 2], nrm[0], nrm[1], nrm[2],
                   color="C3", linewidth=0.8, arrow_length_ratio=0.25)
    ax1.scatter(Ca[:, 0], Ca[:, 1], Ca[:, 2], color="#222266", s=8, alpha=0.7)
    ax1.set_xlabel("x (Å)"); ax1.set_ylabel("y (Å)"); ax1.set_zlabel("z (Å)")
    ax1.set_title(f"{pdb_id.upper()} — oriented points (Cα + plane normal)")
    # Close-up of 5 residues
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    L0 = min(5, L)
    Ca0 = Ca[20:20 + L0]; normals0 = normals[20:20 + L0]
    ax2.plot(Ca0[:, 0], Ca0[:, 1], Ca0[:, 2], "-k", linewidth=2.0)
    for k in range(L0):
        nrm = normals0[k] * 2.0
        ax2.quiver(Ca0[k, 0], Ca0[k, 1], Ca0[k, 2], nrm[0], nrm[1], nrm[2],
                   color="C3", linewidth=2.0, arrow_length_ratio=0.3)
        ax2.scatter(Ca0[k, 0], Ca0[k, 1], Ca0[k, 2], color="#222266", s=80, edgecolor="k")
        ax2.text(Ca0[k, 0], Ca0[k, 1], Ca0[k, 2] + 0.4, f"Q_{{Cα,{k}}}", fontsize=8)
    ax2.set_xlabel("x (Å)"); ax2.set_ylabel("y (Å)"); ax2.set_zlabel("z (Å)")
    ax2.set_title("close-up: 5 oriented points")
    fig.suptitle(f"Fig 3.13 analog — {pdb_id.upper()} backbone as oriented points (real PDB)", y=1.02)
    save_close(fig, f"qa_fig_3_13_{pdb_id}_oriented_points.png")


def fig_3_17(maps: dict, secstruct: list[str], pdb_id: str):
    """Comprehensive orientational maps panel for one protein — Fig 3.17 analog.
    Shows contact, distance, QA cost, real cost, and SS bar in one figure."""
    L = min(MAX_RESIDUES, maps["contact"].shape[0])
    fig = plt.figure(figsize=(16, 7))
    # 4 panels in a row
    ax0 = fig.add_axes([0.04, 0.13, 0.21, 0.72])
    ax0.imshow(maps["contact"][:L, :L], cmap="Greys", origin="lower")
    ax0.set_title("contact"); ax0.set_xlabel("j"); ax0.set_ylabel("i")
    ax1 = fig.add_axes([0.28, 0.13, 0.21, 0.72])
    im1 = ax1.imshow(maps["distance_a"][:L, :L], cmap="viridis", origin="lower", vmax=25)
    ax1.set_title("distance (Å)"); ax1.set_xlabel("j")
    plt.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)
    ax2 = fig.add_axes([0.52, 0.13, 0.21, 0.72])
    im2 = ax2.imshow(maps["cost_real"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    ax2.set_title("cost M (Pepe continuous φ)"); ax2.set_xlabel("j")
    plt.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02)
    ax3 = fig.add_axes([0.76, 0.13, 0.21, 0.72])
    im3 = ax3.imshow(maps["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    ax3.set_title(f"cost M (QA mod-{M_QA} φ)"); ax3.set_xlabel("j")
    plt.colorbar(im3, ax=ax3, shrink=0.7, pad=0.02)
    # Secondary-structure bar at top
    bar_top = fig.add_axes([0.04, 0.88, 0.93, 0.025])
    for k in range(L):
        bar_top.add_patch(plt.Rectangle((k, 0), 1, 1, color=SECSTRUCT_COLOR[secstruct[k]]))
    bar_top.set_xlim(0, L); bar_top.set_ylim(0, 1)
    bar_top.set_xticks([]); bar_top.set_yticks([])
    bar_top.set_title("secondary structure (red=α-helix, green=β-sheet, blue=turn, grey=coil)", fontsize=10)
    fig.suptitle(f"Fig 3.17 analog — {pdb_id.upper()} orientational maps comparison (real biology + QA)", y=0.99)
    out_path = OUT_DIR / f"qa_fig_3_17_{pdb_id}_orientational_maps.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def fig_3_22(maps: dict, angles: dict, dotmaps: dict, secstruct: list[str], pdb_id: str):
    """Combined orientation panel: 4 maps + SS bar — Fig 3.22 analog.
    Pepe shows cost map M, three angle maps Φ/Ψ/Ω, and three dot product maps with SS overlay.
    We show cost M + φ + ψ + dot-NCα + SS bar."""
    L = min(MAX_RESIDUES, maps["contact"].shape[0])
    fig = plt.figure(figsize=(17, 6))
    panels = [
        ("cost M", maps["cost_qa"], "inferno", 0, 2),
        ("Φ map", angles["phi"], "RdBu_r", -pi, pi),
        ("Ψ map", angles["psi"], "RdBu_r", -pi, pi),
        ("NCα dot", dotmaps["nca"], "RdBu_r", -1, 1),
    ]
    for idx, (title, mat, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_axes([0.04 + idx * 0.24, 0.10, 0.21, 0.74])
        im = ax.imshow(mat[:L, :L], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_xlabel("j")
        if idx == 0:
            ax.set_ylabel("i")
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    bar_top = fig.add_axes([0.04, 0.87, 0.93, 0.025])
    for k in range(L):
        bar_top.add_patch(plt.Rectangle((k, 0), 1, 1, color=SECSTRUCT_COLOR[secstruct[k]]))
    bar_top.set_xlim(0, L); bar_top.set_ylim(0, 1)
    bar_top.set_xticks([]); bar_top.set_yticks([])
    bar_top.set_title("secondary structure (red=α, green=β, blue=turn, grey=coil)", fontsize=10)
    fig.suptitle(f"Fig 3.22 analog — {pdb_id.upper()} orientation maps + secondary structure", y=0.99)
    out_path = OUT_DIR / f"qa_fig_3_22_{pdb_id}_orientation_secstruct.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def fig_3_27_metric_bars(metrics_per_protein: dict):
    """QA-discretization error bars across proteins — Fig 3.27 analog.
    For each protein, compute the mean |cost_real - cost_qa| as a QA discretization
    error, alongside mean cost values. Bars per protein."""
    pdb_ids = list(metrics_per_protein.keys())
    real_means = [metrics_per_protein[p]["cost_real_mean"] for p in pdb_ids]
    qa_means = [metrics_per_protein[p]["cost_qa_mean"] for p in pdb_ids]
    disc_errors = [metrics_per_protein[p]["discretization_error"] for p in pdb_ids]
    contact_density = [metrics_per_protein[p]["contact_density"] for p in pdb_ids]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    x = np.arange(len(pdb_ids))
    axes[0].bar(x, real_means, color="#cc6600")
    axes[0].set_xticks(x); axes[0].set_xticklabels([p.upper() for p in pdb_ids], rotation=30)
    axes[0].set_title("mean Pepe cost ⟨M_real⟩"); axes[0].set_ylabel("cost in [0, 2]")
    axes[1].bar(x, qa_means, color="#996633")
    axes[1].set_xticks(x); axes[1].set_xticklabels([p.upper() for p in pdb_ids], rotation=30)
    axes[1].set_title(f"mean QA cost ⟨M_QA⟩ (mod-{M_QA})"); axes[1].set_ylabel("cost in [0, 2]")
    axes[2].bar(x, disc_errors, color="#444499")
    axes[2].set_xticks(x); axes[2].set_xticklabels([p.upper() for p in pdb_ids], rotation=30)
    axes[2].set_title("QA discretization error ⟨|M_real − M_QA|⟩"); axes[2].set_ylabel("error")
    axes[3].bar(x, contact_density, color="#229922")
    axes[3].set_xticks(x); axes[3].set_xticklabels([p.upper() for p in pdb_ids], rotation=30)
    axes[3].set_title("contact density (< 15 Å)"); axes[3].set_ylabel("fraction")
    fig.suptitle("Fig 3.27 analog — per-protein metric bars (real cost vs QA discretization)", y=1.04)
    save_close(fig, "qa_fig_3_27_per_protein_metric_bars.png")


def fig_3_23_24_schematic():
    """Architecture schematic for QA-CGENN-analog projector — Fig 3.23/3.24 analog.
    Shows the conceptual data flow from real backbone → QA discretization → cost
    feature → orientation feature."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    boxes = [
        (0.5, 4.0, 2.2, 1.3, "PDB backbone\n(N, Cα, C in Å)", "#dddddd"),
        (3.4, 4.0, 2.2, 1.3, "plane normals\nn_i = (N×Cα×C)/||·||", "#ffe9c2"),
        (6.3, 4.5, 2.4, 0.8, "rotor angle\nφ_ij = arccos(n_i·n_j)", "#c9e7ff"),
        (6.3, 3.4, 2.4, 0.8, "QA quantize\nφ → 2π·k/24", "#a4d6fb"),
        (9.5, 4.0, 2.2, 1.3, "QA cost\nC = 2(1−cos φ_QA)", "#ffd2c2"),
        (12.3, 4.0, 1.3, 1.3, "M_ij\nmap", "#ffb29c"),
        (0.5, 1.5, 2.2, 1.3, "PDB backbone\n(real PDB)", "#dddddd"),
        (3.4, 1.5, 2.2, 1.3, "real dihedrals\nΦ_i, Ψ_i, Ω_i", "#ffe9c2"),
        (6.3, 1.5, 2.4, 1.3, "QA quantize\neach angle → mod-24", "#a4d6fb"),
        (9.5, 1.5, 2.2, 1.3, "QA Φ/Ψ/Ω maps\nor dot products", "#ffd2c2"),
        (12.3, 1.5, 1.3, 1.3, "angle\n+ dot\nmaps", "#ffb29c"),
    ]
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=col, edgecolor="black", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.5)
    # Arrows
    arrows = [
        (2.7, 4.65, 0.7, 0), (5.6, 4.65, 0.7, 0),
        (8.7, 4.9, 0.8, -0.05), (8.7, 3.8, 0.8, 0.05),
        (11.7, 4.65, 0.6, 0),
        (2.7, 2.15, 0.7, 0), (5.6, 2.15, 0.7, 0), (8.7, 2.15, 0.7, 0), (11.7, 2.15, 0.6, 0),
    ]
    for x, y, dx, dy in arrows:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(7, 5.7, "cost-map path (Pepe Eq 3.7 ↔ QA quantization)",
            ha="center", fontsize=11, style="italic", color="#333366")
    ax.text(7, 0.7, "angle / dot product map path (Pepe §3.4 ↔ QA quantization)",
            ha="center", fontsize=11, style="italic", color="#333366")
    fig.suptitle("Fig 3.23/3.24 analog — QA discretization as Pepe's projector module", y=0.98)
    save_close(fig, "qa_fig_3_23_24_qa_projector_schematic.png")


def fig_3_15_five_maps(maps: dict, angles: dict, dotmaps: dict, pdb_id: str):
    """Five orientational features in one figure: M_α, N_α, Φ, Ψ, Ω — Fig 3.15 analog.
    Pepe Fig 3.15 shows these five orientational maps for the same protein."""
    L = min(MAX_RESIDUES, maps["contact"].shape[0])
    fig = plt.figure(figsize=(18, 4.4))
    panels = [
        ("(a) M_α (QA cost)", maps["cost_qa"], "inferno", 0.0, 2.0),
        ("(b) N_α (NCα dot)", dotmaps["nca"], "RdBu_r", -1.0, 1.0),
        ("(c) Φ map", angles["phi"], "RdBu_r", -pi, pi),
        ("(d) Ψ map", angles["psi"], "RdBu_r", -pi, pi),
        ("(e) Ω map", angles["omega"], "RdBu_r", -pi, pi),
    ]
    for idx, (title, mat, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(1, 5, idx + 1)
        im = ax.imshow(mat[:L, :L], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_xlabel("j")
        if idx == 0:
            ax.set_ylabel("i")
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.suptitle(f"Fig 3.15 analog — five orientational maps for {pdb_id.upper()} (real biology + QA-quantization)", y=1.04)
    save_close(fig, f"qa_fig_3_15_{pdb_id}_five_orientations.png")


def fig_3_16_17_ground_truth_panels(maps: dict, angles: dict, dotmaps: dict, pdb_id: str, fignum: str):
    """Ground-truth-side of Figs 3.16/3.17 (predicted half requires PSP pipeline).
    Layout: 5 ground-truth maps × 2 rows (top = ground truth, bottom = placeholder)."""
    L = min(MAX_RESIDUES, maps["contact"].shape[0])
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    panels = [
        ("M_α (QA cost)", maps["cost_qa"], "inferno", 0.0, 2.0),
        ("N_α (NCα dot)", dotmaps["nca"], "RdBu_r", -1.0, 1.0),
        ("Φ", angles["phi"], "RdBu_r", -pi, pi),
        ("Ψ", angles["psi"], "RdBu_r", -pi, pi),
        ("Ω", angles["omega"], "RdBu_r", -pi, pi),
    ]
    for idx, (title, mat, cmap, vmin, vmax) in enumerate(panels):
        # Top row: ground truth
        ax = axes[0, idx]
        im = ax.imshow(mat[:L, :L], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"GT {title}")
        plt.colorbar(im, ax=ax, shrink=0.65, pad=0.03)
        ax.set_xticks([]); ax.set_yticks([])
        # Bottom row: predicted placeholder
        ax = axes[1, idx]
        ax.text(0.5, 0.5, f"predicted {title}\n(requires PSP\npipeline run)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#999"))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("ground truth", fontsize=11)
    axes[1, 0].set_ylabel("predicted", fontsize=11)
    fig.suptitle(f"Fig {fignum} analog — {pdb_id.upper()} orientational maps: ground truth (computed) + predicted (PSP-pipeline placeholder)", y=1.01)
    save_close(fig, f"qa_fig_{fignum.replace('.', '_')}_{pdb_id}_gt_vs_pred.png")


def fig_3_19_20_21_psp_results(bb: dict, maps: dict, pdb_id: str):
    """PSP coordinate-prediction results display — Fig 3.19/3.20/3.21 analog.
    Pepe shows: original 3D model in green/red/yellow, ground truth vs predicted
    coords (top row), and distance maps with MAE/SSIM/GDT_TS (bottom row).
    Without a PSP pipeline, render the GROUND TRUTH only and label the
    prediction half as placeholder."""
    L = min(MAX_RESIDUES, bb["Ca"].shape[0])
    Ca = bb["Ca"][:L]
    dist_a = maps["distance_a"][:L, :L]
    fig = plt.figure(figsize=(15, 7))

    # Top-left: 3D ground-truth backbone
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="red", linewidth=1.4)
    ax1.set_title(f"{pdb_id.upper()} ground-truth Cα chain")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # Top-mid: GT vs predicted overlay (here just GT — predicted requires pipeline)
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="red", linewidth=1.4, label="GT (red)")
    ax2.set_title("GT (red) — predicted (blue) placeholder")
    ax2.text2D(0.05, 0.95, "predicted overlay\nrequires PSP run",
               transform=ax2.transAxes, fontsize=9,
               bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#888"))

    # Top-right: distance map (ground truth)
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(dist_a, cmap="viridis", origin="lower", vmax=25)
    ax3.set_title("ground-truth distance map D (Å)")
    ax3.set_xlabel("j"); ax3.set_ylabel("i")
    plt.colorbar(im3, ax=ax3, shrink=0.7)

    # Bottom-left: predicted distance map placeholder
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.text(0.5, 0.5, "predicted D\n(requires PSP pipeline)", ha="center", va="center",
             transform=ax4.transAxes, fontsize=11,
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#888"))
    ax4.set_xticks([]); ax4.set_yticks([])

    # Bottom-mid: QA cost map (ground truth orientation)
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(maps["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    ax5.set_title("ground-truth QA cost map M")
    ax5.set_xlabel("j"); ax5.set_ylabel("i")
    plt.colorbar(im5, ax=ax5, shrink=0.7)

    # Bottom-right: metric placeholder
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.65, f"protein: {pdb_id.upper()}\nL = {L} residues\ncontact density = {maps['contact'][:L, :L].mean():.3f}\nmean cost M = {maps['cost_qa'][:L, :L][maps['contact'][:L, :L] > 0].mean():.3f}",
             ha="center", va="center", transform=ax6.transAxes, fontsize=11)
    ax6.text(0.5, 0.20, "GDT_TS, GDT_HA, MAE, SSIM\nrequire PSP-pipeline predictions",
             ha="center", va="center", transform=ax6.transAxes, fontsize=10,
             style="italic", color="#666")
    ax6.set_xticks([]); ax6.set_yticks([])

    fig.suptitle(f"Fig 3.19/3.20/3.21 analog — {pdb_id.upper()} (ground-truth side; predictions require PSP)", y=1.00)
    save_close(fig, f"qa_fig_3_19_20_21_{pdb_id}_psp_results_template.png")


def fig_3_28_cgenn_layers(bb: dict, maps: dict, dotmaps: dict, pdb_id: str):
    """CGENN inputs/outputs visualization — Fig 3.28 analog.
    Pepe Fig 3.28 shows successive CGENN layer outputs as 3D reconstructions.
    Without a CGENN pipeline, show inputs (real PDB Cα) + the QA-quantized
    orientational feature stack the CGENN would receive."""
    L = min(MAX_RESIDUES, bb["Ca"].shape[0])
    Ca = bb["Ca"][:L]
    fig = plt.figure(figsize=(16, 8))
    # Top row: input feature stack
    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    ax1.plot(Ca[:, 0], Ca[:, 1], Ca[:, 2], "-", color="gray", linewidth=1.0)
    ax1.set_title("input: real Cα chain")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax2 = fig.add_subplot(2, 4, 2)
    im2 = ax2.imshow(maps["contact"][:L, :L], cmap="Greys", origin="lower")
    ax2.set_title("input: contact map"); ax2.set_xticks([]); ax2.set_yticks([])
    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(maps["cost_qa"][:L, :L], cmap="inferno", origin="lower", vmin=0, vmax=2)
    ax3.set_title("input: QA cost M"); ax3.set_xticks([]); ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3, shrink=0.7)
    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.imshow(dotmaps["nca"][:L, :L], cmap="RdBu_r", origin="lower", vmin=-1, vmax=1)
    ax4.set_title("input: QA dot NCα"); ax4.set_xticks([]); ax4.set_yticks([])
    plt.colorbar(im4, ax=ax4, shrink=0.7)
    # Bottom row: placeholder for layer outputs
    for k in range(4):
        ax = fig.add_subplot(2, 4, 5 + k)
        ax.text(0.5, 0.5, f"CGENN layer {k+1} output\n(requires CGENN training)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#888"))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Fig 3.28 analog — {pdb_id.upper()} CGENN inputs (real + QA) + outputs (PSP-pipeline placeholder)", y=1.00)
    save_close(fig, f"qa_fig_3_28_{pdb_id}_cgenn_layers.png")


def fig_3_18_schematic():
    """Graph Transformer (Pepe Fig 3.18) analog: schematic of how QA-discretized
    orientation features plug into a downstream model. Visualization only — no
    PSP pipeline run here."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5); ax.axis("off")
    boxes = [
        (0.3, 2.5, 1.8, 1.3, "PDB sequence\n+ MSA features", "#dddddd"),
        (2.5, 2.5, 1.7, 1.3, "PDNET 57-channel\nfeature stack", "#ffe9c2"),
        (4.6, 4.0, 2.3, 1.1, "+ QA cost map M\n(orientation sidecar)", "#c9e7ff"),
        (4.6, 1.6, 2.3, 1.1, "+ QA angle maps\nΦ/Ψ/Ω", "#c9e7ff"),
        (7.3, 2.5, 2.3, 1.3, "Graph Transformer\n3 layers × 4 heads", "#a4d6fb"),
        (9.9, 2.5, 2.0, 1.3, "3D projector\n(MVL/T-FCGP)", "#ffd2c2"),
        (12.0, 2.8, 0.8, 0.8, "Cα\ncoords", "#ffb29c"),
    ]
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=col, edgecolor="black", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.5)
    arrows = [
        (2.1, 3.15, 0.4, 0), (4.2, 3.15, 0.4, 0.6), (4.2, 3.15, 0.4, -0.9),
        (6.9, 4.55, 0.4, -1.2), (6.9, 2.15, 0.4, 1.2),
        (9.6, 3.15, 0.3, 0), (11.9, 3.15, 0.1, 0),
    ]
    for x, y, dx, dy in arrows:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(6.5, 5.2, "QA sidecars route into PDNET-style pipeline",
            ha="center", fontsize=11, style="italic", color="#333366")
    fig.suptitle("Fig 3.18 analog — QA orientation sidecar in graph-transformer PSP pipeline (schematic)", y=0.98)
    save_close(fig, "qa_fig_3_18_pipeline_schematic.png")


# ---------- main ----------

def main() -> int:
    print(f"QA Pepe Ch3 real-protein replica  M_QA={M_QA}  contact={CONTACT_ANGSTROMS}Å")

    # Load all the proteins Pepe shows
    print("\nLoading PDB structures:")
    proteins = {}
    for pdb_id in ["1dmp", "2hc5", "4jzk", "1laf", "2lao", "1bvm", "3i40", "12as", "3i41",
                   "1a3n", "6vxx", "6vyb", "1a70", "1a3a", "2gom", "1yqh", "1z0j", "2ehw"]:
        pdb_path = PDB_DIR / f"{pdb_id}.pdb"
        bb = load_backbone(pdb_path)
        print(f"  {pdb_id}: {bb['N'].shape[0]} residues, first={bb['residues'][0] if bb['residues'] else None}")
        proteins[pdb_id] = bb

    # Per-protein maps for ones we'll render
    print("\nComputing QA maps:")
    maps = {}
    dihedrals = {}
    secstructs = {}
    for pdb_id in ["1dmp", "2hc5", "4jzk", "1laf", "2lao", "12as", "3i41", "1a3n",
                   "6vxx", "6vyb", "1a70", "1a3a", "2gom", "1yqh", "1z0j", "2ehw"]:
        bb = proteins[pdb_id]
        m = compute_pepe_qa_maps(bb, M_QA, CONTACT_ANGSTROMS)
        d = per_residue_dihedrals(bb)
        ss = secondary_structure_heuristic(d["phi"], d["psi"])
        maps[pdb_id] = m
        dihedrals[pdb_id] = d
        secstructs[pdb_id] = ss
        print(f"  {pdb_id}: contact density = {m['contact'].mean():.3f}  ss = {dict(zip(*np.unique(ss, return_counts=True)))}")

    # ---- Render figures ----
    print("\nRendering figures:")
    # Fig 3.1: 1dmp HIV-1 protease
    fig_3_1(proteins["1dmp"], "1dmp")
    # Fig 3.2: 2hc5
    fig_3_2(maps["2hc5"], "2hc5", "A")
    # Fig 3.3: 4jzk with secondary-structure overlay
    fig_3_3(maps["4jzk"], secstructs["4jzk"], "4jzk", "A")
    # Fig 3.4: multi-protein survey
    fig_3_4([(pid, proteins[pid], maps[pid]) for pid in ["2hc5", "4jzk", "12as", "3i41", "1a3n", "1dmp"]])
    # Fig 3.5 + 3.6: 1laf open vs 2lao closed
    fig_3_5_6(proteins["1laf"], proteins["2lao"], maps["1laf"], maps["2lao"],
              "1LAF (open)", "2LAO (closed)")
    # Fig 3.11: 3i40 insulin dihedrals
    fig_3_11(proteins["3i40"], dihedrals.get("3i40") or per_residue_dihedrals(proteins["3i40"]), "3i40")
    # Fig 3.12: 12as angle maps
    angles_12as = compute_qa_angle_maps(proteins["12as"], per_residue_dihedrals(proteins["12as"]), M_QA, CONTACT_ANGSTROMS)
    fig_3_12(angles_12as, "12as")
    # Fig 3.14: 3i41 dot-product maps
    dot_3i41 = compute_qa_dotprod_maps(proteins["3i41"], M_QA, CONTACT_ANGSTROMS)
    fig_3_14(dot_3i41, "3i41")

    # Fig 3.7 + 3.8: SARS-CoV-2 spike open (6vyb) vs closed (6vxx). Pepe shows
    # "first 500 residues" — truncate.
    fig_3_7(proteins["6vxx"], proteins["6vyb"], "closed", "open", "6vxx", "6vyb")
    fig_3_8(maps["6vxx"], maps["6vyb"], "6vxx", "6vyb", "closed", "open", n_first=500)

    # Fig 3.9 + 3.10: NMR ensemble of 1BVM
    print("\nLoading NMR ensemble for 1BVM ...")
    ensemble_1bvm = load_nmr_ensemble(PDB_DIR / "1bvm.pdb", max_models=20)
    print(f"  loaded {len(ensemble_1bvm)} NMR models, each L={ensemble_1bvm[0]['N'].shape[0]}")
    fig_3_9(ensemble_1bvm, "1bvm")
    fig_3_10(ensemble_1bvm, "1bvm")

    # Fig 3.13: Hemoglobin (1a3n) oriented points
    fig_3_13(proteins["1a3n"], "1a3n")

    # Fig 3.17: 1a70A orientational maps comparison
    fig_3_17(maps["1a70"], secstructs["1a70"], "1a70")

    # Fig 3.22: combined orientation + secondary structure (4jzk, mixed α+β protein)
    angles_4jzk = compute_qa_angle_maps(proteins["4jzk"], dihedrals["4jzk"], M_QA, CONTACT_ANGSTROMS)
    dot_4jzk = compute_qa_dotprod_maps(proteins["4jzk"], M_QA, CONTACT_ANGSTROMS)
    fig_3_22(maps["4jzk"], angles_4jzk, dot_4jzk, secstructs["4jzk"], "4jzk")

    # Fig 3.27: per-protein metric bars (QA discretization error)
    metrics_per_protein = {}
    for pdb_id in ["1dmp", "2hc5", "4jzk", "12as", "3i41", "1a3n"]:
        m = maps[pdb_id]
        ct = m["contact"].astype(bool)
        if ct.sum() == 0:
            continue
        cost_real = m["cost_real"][ct]
        cost_qa = m["cost_qa"][ct]
        metrics_per_protein[pdb_id] = {
            "cost_real_mean": float(cost_real.mean()),
            "cost_qa_mean": float(cost_qa.mean()),
            "discretization_error": float(np.abs(cost_real - cost_qa).mean()),
            "contact_density": float(m["contact"].mean()),
        }
    fig_3_27_metric_bars(metrics_per_protein)

    # Fig 3.18 + 3.23/3.24: pipeline + projector schematics
    fig_3_18_schematic()
    fig_3_23_24_schematic()

    # Fig 3.15: five orientational features on one protein (1a70A per Pepe Fig 3.17)
    angles_1a70 = compute_qa_angle_maps(proteins["1a70"], dihedrals["1a70"], M_QA, CONTACT_ANGSTROMS)
    dot_1a70 = compute_qa_dotprod_maps(proteins["1a70"], M_QA, CONTACT_ANGSTROMS)
    fig_3_15_five_maps(maps["1a70"], angles_1a70, dot_1a70, "1a70")

    # Fig 3.16: ground-truth vs predicted orientational maps for 1a3a (predicted = placeholder)
    angles_1a3a = compute_qa_angle_maps(proteins["1a3a"], dihedrals["1a3a"], M_QA, CONTACT_ANGSTROMS)
    dot_1a3a = compute_qa_dotprod_maps(proteins["1a3a"], M_QA, CONTACT_ANGSTROMS)
    fig_3_16_17_ground_truth_panels(maps["1a3a"], angles_1a3a, dot_1a3a, "1a3a", "3.16")

    # Fig 3.17 (updated): ground-truth vs predicted orientational maps for 1a70A
    # (overwrites the earlier 4-panel version with the 5-map GT+pred layout)
    fig_3_16_17_ground_truth_panels(maps["1a70"], angles_1a70, dot_1a70, "1a70", "3.17")

    # Fig 3.19/3.20/3.21: PSP coord-prediction template for 2gomA, 1yqhA, 1z0jB
    for pdb_id in ["2gom", "1yqh", "1z0j"]:
        fig_3_19_20_21_psp_results(proteins[pdb_id], maps[pdb_id], pdb_id)

    # Fig 3.28: CGENN inputs/outputs for 2ehw
    dot_2ehw = compute_qa_dotprod_maps(proteins["2ehw"], M_QA, CONTACT_ANGSTROMS)
    fig_3_28_cgenn_layers(proteins["2ehw"], maps["2ehw"], dot_2ehw, "2ehw")

    # Fig 3.25/3.26: training/validation loss curves — explicit skip note
    print("\n  Fig 3.25/3.26 (training/validation loss curves): SKIPPED — requires running")
    print("    the PSP pipeline to generate real loss histories. Cannot be faithfully")
    print("    recreated without actual training; not fabricating placeholder curves.")

    print(f"\nAll figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
