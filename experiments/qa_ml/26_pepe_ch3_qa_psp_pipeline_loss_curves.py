"""Pepe Ch3 PSP pipeline run for Figs 3.25 / 3.26 — real training and
validation loss curves.

Task: predict the five orientational pairwise features (M_α cost map,
N_α dot-product map, Φ, Ψ, Ω angle maps) for protein residue pairs from
per-pair input features. Real training run with log-cosh loss (Pepe Eq
3.17 — `L_X = log(cosh(X_P − X_T))`).

This is a stripped-down PSP analog:
  - Pepe uses PDNET's 57-channel sequence/MSA feature stack + Graph
    Transformer + 3D projector. Reproducing that requires DEEPCOV/
    PSICOV splits, ESM/PSSM features, and several hours of GPU.
  - Here we use the 18 PDB structures already on disk (the proteins
    Pepe references in Ch3). Pairwise features = (amino-acid one-hots,
    sequence separation, distance, contact, |i−j|/L). Targets = the
    five orientational maps computed from real PDB coordinates with
    QA-mod24 quantization.
  - Train on 12 proteins, validate on 6. Per-pair MLP with 5 separate
    output heads.

Two figure outputs:
  qa_fig_3_25_training_loss_curves.png    (per-feature training loss vs epoch)
  qa_fig_3_26_validation_loss_curves.png  (per-feature validation loss vs epoch)

These are REAL loss curves from REAL training on REAL protein orientational maps.
Not Pepe's absolute numbers (different feature stack, different model, different
dataset size); same per-feature comparison Pepe makes.

QA_COMPLIANCE = "qa_ml_pepe_ch3_psp_pipeline — observer-side training; integer QA-quantized targets; A1/A2 compliant"
"""

from __future__ import annotations

import json
import sys
import time
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# Reuse map computation from script 25
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

# Import functions from script 25 via direct exec (script names start with digit)
spec_path = Path(__file__).parent / "25_pepe_ch3_qa_real_protein_replica.py"
ns: dict = {"__file__": str(spec_path), "__name__": "ch3_visual_replica_helpers"}
exec(compile(spec_path.read_text(), str(spec_path), "exec"), ns)
load_backbone = ns["load_backbone"]
per_residue_dihedrals = ns["per_residue_dihedrals"]
compute_pepe_qa_maps = ns["compute_pepe_qa_maps"]
compute_qa_angle_maps = ns["compute_qa_angle_maps"]
compute_qa_dotprod_maps = ns["compute_qa_dotprod_maps"]


OUT_DIR = Path(__file__).parent / "ch3_qa_real_protein_replica"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDB_DIR = REPO / "corpus" / "pdb_pepe_ch3"
RESULT_PATH = OUT_DIR / "psp_pipeline_loss_history.json"

M_QA = 24
CONTACT_ANGSTROMS = 15.0
SEED = 0

TRAIN_IDS = ["1dmp", "2hc5", "4jzk", "12as", "3i41", "1a3n", "1a70", "1a3a",
             "2gom", "2ehw", "1bvm", "6vxx"]
VAL_IDS = ["1laf", "2lao", "3i40", "1yqh", "1z0j", "6vyb"]

# Standard 20 amino acids
AA_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ---------- featurize ----------

def aa_one_hot(resname: str) -> np.ndarray:
    v = np.zeros(20, dtype=np.float32)
    if resname in AA_TO_IDX:
        v[AA_TO_IDX[resname]] = 1.0
    return v


def build_pair_dataset(bb_dict: dict, maps_per: dict, dihedrals_per: dict,
                       angles_per: dict, dotmaps_per: dict, pdb_ids: list[str],
                       max_pairs_per_protein: int = 4000, rng: np.random.Generator = None) -> dict:
    """Build (X, Y) pairwise feature/target tensors from a set of proteins.

    X features per pair (i, j) within contact:
      [aa_one_hot_i (20), aa_one_hot_j (20), normalized_sep, distance_a,
       normalized_pos_i, normalized_pos_j, contact (always 1 here)]
    Y targets per pair:
      [M_α cost (QA), N_α NCα-dot, Φ, Ψ, Ω]
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    X_list, Y_list = [], []
    for pid in pdb_ids:
        bb = bb_dict[pid]
        m = maps_per[pid]
        angs = angles_per[pid]
        dots = dotmaps_per[pid]
        L = bb["N"].shape[0]
        contact = m["contact"].astype(bool)
        # Pair indices i < j, in contact
        idx_i, idx_j = np.where(np.triu(contact, k=1))
        if len(idx_i) == 0:
            continue
        # Sample if too many
        if len(idx_i) > max_pairs_per_protein:
            keep = rng.choice(len(idx_i), size=max_pairs_per_protein, replace=False)
            idx_i = idx_i[keep]; idx_j = idx_j[keep]
        # AA names
        aa_i_arr = np.array([aa_one_hot(bb["residues"][i][2]) for i in idx_i])
        aa_j_arr = np.array([aa_one_hot(bb["residues"][j][2]) for j in idx_j])
        sep = np.abs(idx_i - idx_j).astype(np.float32)
        sep_norm = (sep / max(L, 1)).reshape(-1, 1)
        dist = m["distance_a"][idx_i, idx_j].astype(np.float32).reshape(-1, 1) / 30.0  # scale
        pos_i = (idx_i.astype(np.float32) / max(L, 1)).reshape(-1, 1)
        pos_j = (idx_j.astype(np.float32) / max(L, 1)).reshape(-1, 1)
        contact_feat = np.ones((len(idx_i), 1), dtype=np.float32)
        X = np.concatenate([aa_i_arr, aa_j_arr, sep_norm, dist, pos_i, pos_j, contact_feat], axis=1)
        Y = np.stack([
            m["cost_qa"][idx_i, idx_j].astype(np.float32),
            dots["nca"][idx_i, idx_j].astype(np.float32),
            angs["phi"][idx_i, idx_j].astype(np.float32),
            angs["psi"][idx_i, idx_j].astype(np.float32),
            angs["omega"][idx_i, idx_j].astype(np.float32),
        ], axis=1)
        X_list.append(X); Y_list.append(Y)
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return {"X": X.astype(np.float32), "Y": Y.astype(np.float32)}


# ---------- model ----------

class PairwiseOrientationNet(nn.Module):
    """Per-pair MLP with 5 output heads. Matches Pepe's per-pair feature
    target structure: each (i, j) pair predicts (M, N, Φ, Ψ, Ω)."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Five output heads with appropriate activations:
        #   M_α ∈ [0, 2]      — sigmoid-scaled
        #   N_α ∈ [−1, 1]     — tanh (Pepe explicitly uses tanh for N_α)
        #   Φ/Ψ/Ω ∈ [−π, π]   — tanh-scaled by π
        self.head_M = nn.Linear(hidden, 1)
        self.head_N = nn.Linear(hidden, 1)
        self.head_phi = nn.Linear(hidden, 1)
        self.head_psi = nn.Linear(hidden, 1)
        self.head_omega = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        m = 2.0 * torch.sigmoid(self.head_M(h)).squeeze(-1)
        n = torch.tanh(self.head_N(h)).squeeze(-1)
        phi = pi * torch.tanh(self.head_phi(h)).squeeze(-1)
        psi = pi * torch.tanh(self.head_psi(h)).squeeze(-1)
        omega = pi * torch.tanh(self.head_omega(h)).squeeze(-1)
        return torch.stack([m, n, phi, psi, omega], dim=1)


def logcosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pepe Eq 3.17: L_X = log(cosh(X_P − X_T)). Per-channel mean."""
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff) + 1e-12), dim=0)  # [5]


# ---------- training loop ----------

def train(epochs: int = 100, batch_size: int = 1024, lr: float = 1e-3) -> dict:
    print(f"PSP pipeline: train={TRAIN_IDS} val={VAL_IDS}")
    print("Loading + computing maps for all proteins ...")

    proteins = {}
    maps_per = {}
    dihedrals_per = {}
    angles_per = {}
    dotmaps_per = {}
    for pid in TRAIN_IDS + VAL_IDS:
        bb = load_backbone(PDB_DIR / f"{pid}.pdb")
        proteins[pid] = bb
        m = compute_pepe_qa_maps(bb, M_QA, CONTACT_ANGSTROMS)
        d = per_residue_dihedrals(bb)
        ang = compute_qa_angle_maps(bb, d, M_QA, CONTACT_ANGSTROMS)
        dot = compute_qa_dotprod_maps(bb, M_QA, CONTACT_ANGSTROMS)
        maps_per[pid] = m
        dihedrals_per[pid] = d
        angles_per[pid] = ang
        dotmaps_per[pid] = dot
    print(f"  loaded {len(proteins)} proteins")

    rng = np.random.default_rng(SEED)
    train_data = build_pair_dataset(proteins, maps_per, dihedrals_per, angles_per, dotmaps_per, TRAIN_IDS, rng=rng)
    val_data = build_pair_dataset(proteins, maps_per, dihedrals_per, angles_per, dotmaps_per, VAL_IDS, rng=rng)
    print(f"  train pairs = {train_data['X'].shape[0]:,}  feat dim = {train_data['X'].shape[1]}")
    print(f"  val pairs   = {val_data['X'].shape[0]:,}")

    torch.manual_seed(SEED)
    X_train = torch.tensor(train_data["X"], dtype=torch.float32)
    Y_train = torch.tensor(train_data["Y"], dtype=torch.float32)
    X_val = torch.tensor(val_data["X"], dtype=torch.float32)
    Y_val = torch.tensor(val_data["Y"], dtype=torch.float32)

    model = PairwiseOrientationNet(in_dim=X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = X_train.shape[0]
    feature_names = ["M_α", "N_α", "Φ", "Ψ", "Ω"]
    history = {"epoch": [], "train_per_feat": [], "val_per_feat": []}

    print(f"\nTraining for {epochs} epochs (batch_size={batch_size}, lr={lr})")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_losses = torch.zeros(5)
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train[idx]; yb = Y_train[idx]
            pred = model(xb)
            losses = logcosh_loss(pred, yb)   # [5]
            total = losses.sum()
            opt.zero_grad()
            total.backward()
            opt.step()
            epoch_losses += losses.detach()
            n_batches += 1
        train_per_feat = (epoch_losses / max(n_batches, 1)).numpy()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_losses = logcosh_loss(val_pred, Y_val).numpy()

        history["epoch"].append(epoch + 1)
        history["train_per_feat"].append(train_per_feat.tolist())
        history["val_per_feat"].append(val_losses.tolist())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            tr_str = "  ".join(f"{fn}={v:.4f}" for fn, v in zip(feature_names, train_per_feat))
            vl_str = "  ".join(f"{fn}={v:.4f}" for fn, v in zip(feature_names, val_losses))
            print(f"  ep {epoch+1:3d}  train: {tr_str}")
            print(f"             val:   {vl_str}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    return {
        "feature_names": feature_names,
        "epochs": history["epoch"],
        "train_per_feat": np.array(history["train_per_feat"]),
        "val_per_feat": np.array(history["val_per_feat"]),
        "elapsed_s": elapsed,
    }


# ---------- plotting ----------

FEAT_COLORS = ["#cc3322", "#cc8822", "#22aa44", "#3366cc", "#7733aa"]


def fig_3_25_train_loss(history: dict):
    """Fig 3.25 analog — training-loss curves per orientational feature."""
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = history["epochs"]
    feats = history["feature_names"]
    tr = history["train_per_feat"]
    for k, fn in enumerate(feats):
        ax.plot(epochs, tr[:, k], "-", color=FEAT_COLORS[k], linewidth=1.8, label=fn)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training log-cosh loss")
    ax.set_title("Fig 3.25 analog — training loss per orientational feature\n"
                 "(real PSP-style training run, log-cosh loss per Pepe Eq 3.17)")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_fig_3_25_training_loss_curves.png", dpi=130, bbox_inches="tight")
    print(f"  wrote {OUT_DIR / 'qa_fig_3_25_training_loss_curves.png'}")
    plt.close(fig)


def fig_3_26_val_loss(history: dict):
    """Fig 3.26 analog — validation-loss curves per orientational feature."""
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = history["epochs"]
    feats = history["feature_names"]
    vl = history["val_per_feat"]
    for k, fn in enumerate(feats):
        ax.plot(epochs, vl[:, k], "-", color=FEAT_COLORS[k], linewidth=1.8, label=fn)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation log-cosh loss")
    ax.set_title("Fig 3.26 analog — validation loss per orientational feature\n"
                 "(held-out proteins: " + ", ".join(p.upper() for p in VAL_IDS) + ")")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_fig_3_26_validation_loss_curves.png", dpi=130, bbox_inches="tight")
    print(f"  wrote {OUT_DIR / 'qa_fig_3_26_validation_loss_curves.png'}")
    plt.close(fig)


def main() -> int:
    print("Pepe Ch3 PSP pipeline — real training run for Figs 3.25/3.26")
    history = train(epochs=120, batch_size=1024, lr=1e-3)
    # Save history JSON for reproducibility
    RESULT_PATH.write_text(json.dumps({
        "feature_names": history["feature_names"],
        "epochs": history["epochs"],
        "train_per_feat": history["train_per_feat"].tolist(),
        "val_per_feat": history["val_per_feat"].tolist(),
        "elapsed_s": history["elapsed_s"],
        "train_ids": TRAIN_IDS,
        "val_ids": VAL_IDS,
        "config": {"M_QA": M_QA, "CONTACT_ANGSTROMS": CONTACT_ANGSTROMS, "epochs": 120, "batch_size": 1024, "lr": 1e-3},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote loss history to {RESULT_PATH}")

    # Render Figs 3.25 and 3.26
    fig_3_25_train_loss(history)
    fig_3_26_val_loss(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
