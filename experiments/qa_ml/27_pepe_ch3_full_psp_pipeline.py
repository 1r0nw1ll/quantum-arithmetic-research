"""Full Pepe Ch3 PSP pipeline — DEEPCOV 1000 + PSICOV 150, ESM-2-650M
embeddings, Graph Transformer architecture, log-cosh loss, real loss
curves. Replaces script 26's bounded smoke with the full thesis-style
training set Pepe used.

Pipeline is RESUMABLE — every intermediate is cached on disk:
  cache/pdb/<id>.pdb                 PDB files
  cache/esm_650m/<id>.npy            per-protein ESM embeddings [L, 1280]
  cache/orient_maps/<id>.npz         orientational maps (M, N, Φ, Ψ, Ω, distance, contact)
  cache/pair_index.npz               flat (protein_idx, i, j) tuples
  cache/checkpoint_epoch_<k>.pt      model checkpoints per epoch
  cache/loss_history.json            train + val loss per epoch per feature

Targets: M_α (cost map), N_α (NCα dot), Φ, Ψ, Ω angle maps.

Architecture: Graph Transformer (3 layers × 4 heads, hidden=128) over
per-residue ESM embeddings + sparse contact-graph attention. Pair-output
head predicts the 5 orientational scalars per (i, j) pair.

Loss: log(cosh(X_pred − X_true)) per Pepe Eq 3.17.

Training: 5 epochs (Pepe's exact spec), lr=1e-2 with γ=0.9 epoch decay,
batch_size=32 proteins per batch (Pepe used batch=1 in PDNET; with our
pair-based loss bigger batches make sense).

QA_COMPLIANCE = "qa_ml_pepe_ch3_full_psp — observer-side training; integer-discretized QA-quantized targets; A1/A2 compliant"
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
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

CACHE = Path(__file__).parent / "cache_full_psp"
CACHE.mkdir(parents=True, exist_ok=True)
(CACHE / "pdb").mkdir(exist_ok=True)
(CACHE / "esm_650m").mkdir(exist_ok=True)
(CACHE / "orient_maps").mkdir(exist_ok=True)
OUT_DIR = Path(__file__).parent / "ch3_qa_real_protein_replica"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDNET_FASTA_DIR = REPO / "corpus" / "pdnet" / "data"
DEEPCOV_FASTA = PDNET_FASTA_DIR / "deepcov" / "fasta"
PSICOV_FASTA = PDNET_FASTA_DIR / "psicov" / "fasta"

M_QA = 24
CONTACT_ANGSTROMS = 15.0
SEED = 0
N_TRAIN = 1000        # Pepe's exact spec
N_VAL = 150           # Pepe's exact spec
ESM_MODEL = "esm2_t33_650M_UR50D"
ESM_LAYER = 33
ESM_DIM = 1280
N_EPOCHS = 5
BATCH_PROTEINS = 4    # number of proteins per batch (CPU memory bound)
INIT_LR = 1e-2
LR_DECAY = 0.9
HIDDEN = 128
N_HEADS = 4
N_LAYERS = 3
N_TARGETS = 5         # M, N, Φ, Ψ, Ω

print_lock = None     # placeholder for thread safety if we add concurrency

AA_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# Reuse map functions from script 25
spec_path = Path(__file__).parent / "25_pepe_ch3_qa_real_protein_replica.py"
ns: dict = {"__file__": str(spec_path), "__name__": "ch3_visual_replica_helpers"}
exec(compile(spec_path.read_text(), str(spec_path), "exec"), ns)
load_backbone = ns["load_backbone"]
per_residue_dihedrals = ns["per_residue_dihedrals"]
compute_pepe_qa_maps = ns["compute_pepe_qa_maps"]
compute_qa_angle_maps = ns["compute_qa_angle_maps"]
compute_qa_dotprod_maps = ns["compute_qa_dotprod_maps"]


# =================================================================
# STEP 1: Load FASTA sequences and pick the 1000 + 150 protein list
# =================================================================

def parse_fasta(path: Path) -> tuple[str, str]:
    """Return (header_id, sequence) from a single-record FASTA file."""
    lines = path.read_text().strip().split("\n")
    header = lines[0].lstrip(">").strip()
    seq = "".join(lines[1:]).strip()
    return header, seq


def select_proteins() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (train, val) lists of (id_chain, sequence) tuples."""
    deepcov_files = sorted(DEEPCOV_FASTA.glob("*.fasta"))[:N_TRAIN]
    psicov_files = sorted(PSICOV_FASTA.glob("*.fasta"))[:N_VAL]
    train = [parse_fasta(p) for p in deepcov_files]
    val = [parse_fasta(p) for p in psicov_files]
    print(f"  selected {len(train)} train (DEEPCOV) + {len(val)} val (PSICOV) proteins")
    return train, val


# =================================================================
# STEP 2: Download PDB files (resumable)
# =================================================================

def download_pdb(pid: str) -> Path:
    """Download a PDB file from RCSB if not cached. pid is 5-char PDB+chain."""
    pdb_id = pid[:4].lower()
    out_path = CACHE / "pdb" / f"{pdb_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        print(f"    WARN: download {pid} failed: {e}")
        out_path.write_text("")  # empty marker
    return out_path


def download_all_pdbs(proteins: list[tuple[str, str]]) -> None:
    """Download all needed PDB files, skipping cached."""
    n = len(proteins)
    print(f"  downloading {n} PDB files (resumable; only fetching missing) ...")
    t0 = time.time()
    n_fetched = 0
    for k, (pid, _) in enumerate(proteins):
        out_path = CACHE / "pdb" / f"{pid[:4].lower()}.pdb"
        if not out_path.exists() or out_path.stat().st_size == 0:
            download_pdb(pid)
            n_fetched += 1
        if (k + 1) % 50 == 0 or k + 1 == n:
            print(f"    {k + 1}/{n}  ({n_fetched} newly fetched, {time.time() - t0:.0f}s elapsed)")


# =================================================================
# STEP 3: Run ESM-2-650M inference (cached per protein)
# =================================================================

def run_esm_inference(proteins: list[tuple[str, str]], model_name: str = ESM_MODEL) -> None:
    """Run ESM-2 and cache per-residue embeddings."""
    import esm
    todo = [(pid, seq) for pid, seq in proteins
            if not (CACHE / "esm_650m" / f"{pid}.npy").exists()]
    if not todo:
        print(f"  all {len(proteins)} ESM embeddings already cached")
        return
    print(f"  loading {model_name} ...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"  running inference on {len(todo)} proteins (CPU, ~5-10 sec each) ...")
    t0 = time.time()
    with torch.no_grad():
        for k, (pid, seq) in enumerate(todo):
            try:
                batch_labels, batch_strs, batch_tokens = batch_converter([(pid, seq)])
                results = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=False)
                # Remove BOS/EOS tokens (first and last)
                emb = results["representations"][ESM_LAYER][0, 1:-1].cpu().numpy().astype(np.float32)
                np.save(CACHE / "esm_650m" / f"{pid}.npy", emb)
            except Exception as e:
                print(f"    WARN: ESM on {pid} failed: {e}")
                continue
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                elapsed = time.time() - t0
                eta = elapsed / (k + 1) * (len(todo) - k - 1)
                print(f"    {k + 1}/{len(todo)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    del model
    print(f"  ESM inference complete in {time.time() - t0:.0f}s")


# =================================================================
# STEP 4: Compute orientational maps from PDB structures
# =================================================================

def compute_and_cache_maps(proteins: list[tuple[str, str]]) -> None:
    """For each protein, parse PDB and compute the 5 orientational maps."""
    todo = [(pid, seq) for pid, seq in proteins
            if not (CACHE / "orient_maps" / f"{pid}.npz").exists()]
    if not todo:
        print(f"  all {len(proteins)} orientational maps already cached")
        return
    print(f"  computing maps for {len(todo)} proteins ...")
    t0 = time.time()
    chain_letter_to_id = {}
    skip_count = 0
    for k, (pid, seq) in enumerate(todo):
        pdb_id = pid[:4].lower()
        chain_id = pid[4] if len(pid) >= 5 else None
        pdb_path = CACHE / "pdb" / f"{pdb_id}.pdb"
        if not pdb_path.exists() or pdb_path.stat().st_size == 0:
            skip_count += 1
            continue
        try:
            bb = load_backbone(pdb_path, chain_id=chain_id)
            if bb["N"].shape[0] == 0:
                skip_count += 1
                continue
            d = per_residue_dihedrals(bb)
            m = compute_pepe_qa_maps(bb, M_QA, CONTACT_ANGSTROMS)
            ang = compute_qa_angle_maps(bb, d, M_QA, CONTACT_ANGSTROMS)
            dot = compute_qa_dotprod_maps(bb, M_QA, CONTACT_ANGSTROMS)
            np.savez_compressed(
                CACHE / "orient_maps" / f"{pid}.npz",
                cost_qa=m["cost_qa"].astype(np.float32),
                cost_real=m["cost_real"].astype(np.float32),
                contact=m["contact"].astype(np.uint8),
                distance_a=m["distance_a"].astype(np.float32),
                nca=dot["nca"].astype(np.float32),
                phi=ang["phi"].astype(np.float32),
                psi=ang["psi"].astype(np.float32),
                omega=ang["omega"].astype(np.float32),
                pdb_seq=np.array([THREE_TO_ONE.get(r[2], "X") for r in bb["residues"]], dtype="U1"),
            )
        except Exception as e:
            print(f"    WARN: maps for {pid} failed: {e}")
            skip_count += 1
        if (k + 1) % 100 == 0 or k + 1 == len(todo):
            print(f"    {k + 1}/{len(todo)}  ({skip_count} skipped)  {time.time() - t0:.0f}s")
    print(f"  maps cached in {time.time() - t0:.0f}s ({skip_count} proteins skipped)")


def get_valid_protein_ids(proteins: list[tuple[str, str]]) -> list[str]:
    """Return ids that have both ESM embeddings AND orientational maps cached."""
    out = []
    for pid, _ in proteins:
        esm_p = CACHE / "esm_650m" / f"{pid}.npy"
        map_p = CACHE / "orient_maps" / f"{pid}.npz"
        if esm_p.exists() and map_p.exists():
            out.append(pid)
    return out


# =================================================================
# STEP 5: Graph Transformer architecture
# =================================================================

class ProteinDataset:
    """Per-protein loader: returns (residue_features, target_maps, contact_mask).
    Aligns ESM embeddings (length = sequence) with structural maps (length = PDB
    residues). When lengths differ, truncate to min."""

    def __init__(self, protein_ids: list[str]):
        self.ids = protein_ids
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.ids[idx]
        if pid not in self._cache:
            emb = np.load(CACHE / "esm_650m" / f"{pid}.npy").astype(np.float32)
            maps = np.load(CACHE / "orient_maps" / f"{pid}.npz")
            L_emb = emb.shape[0]
            L_map = maps["contact"].shape[0]
            L = min(L_emb, L_map)
            self._cache[pid] = {
                "emb": emb[:L],
                "contact": maps["contact"][:L, :L].astype(np.float32),
                "distance": maps["distance_a"][:L, :L].astype(np.float32),
                "targets": np.stack([
                    maps["cost_qa"][:L, :L],
                    maps["nca"][:L, :L],
                    maps["phi"][:L, :L],
                    maps["psi"][:L, :L],
                    maps["omega"][:L, :L],
                ], axis=0).astype(np.float32),
                "L": L,
            }
        return self._cache[pid]


class GraphTransformerBlock(nn.Module):
    """One graph-transformer layer: contact-masked multi-head self-attention
    + feed-forward (Shi et al. 2021 style)."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, contact_mask: torch.Tensor) -> torch.Tensor:
        # x: [L, dim], contact_mask: [L, L] (1 where edge present)
        L = x.shape[0]
        head_dim = self.dim // self.n_heads
        q = self.q(x).view(L, self.n_heads, head_dim).transpose(0, 1)  # [H, L, d]
        k = self.k(x).view(L, self.n_heads, head_dim).transpose(0, 1)
        v = self.v(x).view(L, self.n_heads, head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [H, L, L]
        # Add diagonal so self-attention is always allowed
        mask = contact_mask + torch.eye(L, device=x.device)
        mask = mask.clamp(max=1.0).unsqueeze(0)  # [1, L, L]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [H, L, d]
        out = out.transpose(0, 1).contiguous().view(L, self.dim)
        x = self.norm1(x + self.o(out))
        x = self.norm2(x + self.ffn(x))
        return x


class GraphTransformerProteinNet(nn.Module):
    """Pepe Ch3 model: ESM input → linear projection → 3 graph-transformer
    layers → pair head → 5 orientational outputs.

    Pair head: for each pair (i, j), concatenate residue embeddings and
    structural features, predict 5 scalars with appropriate activations."""

    def __init__(self, esm_dim: int = ESM_DIM, hidden: int = HIDDEN,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS):
        super().__init__()
        self.proj_in = nn.Linear(esm_dim, hidden)
        self.layers = nn.ModuleList([
            GraphTransformerBlock(hidden, n_heads) for _ in range(n_layers)
        ])
        # Pair head: input = [emb_i, emb_j, |emb_i - emb_j|, distance, sep]
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 3 + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, N_TARGETS),
        )

    def forward(self, esm_emb: torch.Tensor, contact: torch.Tensor,
                distance: torch.Tensor) -> torch.Tensor:
        # esm_emb [L, esm_dim], contact [L, L], distance [L, L]
        L = esm_emb.shape[0]
        x = self.proj_in(esm_emb)
        for layer in self.layers:
            x = layer(x, contact)
        # Build pair features for all (i, j) in contact
        idx_i, idx_j = torch.where(contact > 0.5)
        e_i = x[idx_i]; e_j = x[idx_j]
        diff = (e_i - e_j).abs()
        d = distance[idx_i, idx_j].unsqueeze(-1) / 30.0     # scale to ~[0, 1]
        sep = (idx_i.float() - idx_j.float()).abs().unsqueeze(-1) / float(L)
        pair_feat = torch.cat([e_i, e_j, diff, d, sep], dim=-1)
        raw = self.pair_head(pair_feat)
        # Apply per-channel activations
        m = 2.0 * torch.sigmoid(raw[:, 0])
        n = torch.tanh(raw[:, 1])
        phi = pi * torch.tanh(raw[:, 2])
        psi = pi * torch.tanh(raw[:, 3])
        omega = pi * torch.tanh(raw[:, 4])
        pred = torch.stack([m, n, phi, psi, omega], dim=-1)
        return pred, idx_i, idx_j


def logcosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff) + 1e-12), dim=0)


# =================================================================
# STEP 6: Training loop
# =================================================================

def evaluate(model: nn.Module, ds: ProteinDataset, max_proteins: int = 50) -> np.ndarray:
    """Evaluate model on a dataset, return per-feature mean log-cosh loss."""
    model.eval()
    losses = []
    n = min(len(ds), max_proteins)
    with torch.no_grad():
        for k in range(n):
            d = ds[k]
            L = d["L"]
            if L < 5:
                continue
            emb = torch.tensor(d["emb"], dtype=torch.float32)
            contact = torch.tensor(d["contact"], dtype=torch.float32)
            distance = torch.tensor(d["distance"], dtype=torch.float32)
            targets = torch.tensor(d["targets"], dtype=torch.float32)
            pred, idx_i, idx_j = model(emb, contact, distance)
            if pred.shape[0] == 0:
                continue
            tgt = torch.stack([targets[t][idx_i, idx_j] for t in range(N_TARGETS)], dim=-1)
            losses.append(logcosh_loss(pred, tgt).numpy())
    if not losses:
        return np.zeros(N_TARGETS, dtype=np.float32)
    return np.stack(losses, axis=0).mean(axis=0)


def train_loop(train_ids: list[str], val_ids: list[str]) -> dict:
    print(f"\nTraining: {len(train_ids)} train proteins, {len(val_ids)} val proteins")
    train_ds = ProteinDataset(train_ids)
    val_ds = ProteinDataset(val_ids)

    torch.manual_seed(SEED)
    model = GraphTransformerProteinNet(esm_dim=ESM_DIM, hidden=HIDDEN,
                                       n_layers=N_LAYERS, n_heads=N_HEADS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)
    history = {"epoch": [], "train_per_feat": [], "val_per_feat": []}
    feature_names = ["M_α", "N_α", "Φ", "Ψ", "Ω"]

    rng = np.random.default_rng(SEED)
    t_total = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_losses = np.zeros(N_TARGETS, dtype=np.float64)
        n_batches = 0
        order = list(range(len(train_ids)))
        rng.shuffle(order)
        t_epoch = time.time()
        for step, idx in enumerate(order):
            d = train_ds[idx]
            L = d["L"]
            if L < 5:
                continue
            emb = torch.tensor(d["emb"], dtype=torch.float32)
            contact = torch.tensor(d["contact"], dtype=torch.float32)
            distance = torch.tensor(d["distance"], dtype=torch.float32)
            targets = torch.tensor(d["targets"], dtype=torch.float32)
            pred, idx_i, idx_j = model(emb, contact, distance)
            if pred.shape[0] == 0:
                continue
            tgt = torch.stack([targets[t][idx_i, idx_j] for t in range(N_TARGETS)], dim=-1)
            losses_per = logcosh_loss(pred, tgt)
            total = losses_per.sum()
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses += losses_per.detach().numpy()
            n_batches += 1
            if (step + 1) % 100 == 0:
                cur_loss = epoch_losses / max(n_batches, 1)
                elapsed = time.time() - t_epoch
                eta = elapsed / (step + 1) * (len(order) - step - 1)
                print(f"    ep {epoch+1} step {step+1}/{len(order)}  "
                      f"loss={cur_loss.sum():.3f}  per-feat=[{','.join(f'{v:.2f}' for v in cur_loss)}]  "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        train_per_feat = (epoch_losses / max(n_batches, 1)).tolist()

        # Validation
        print(f"  evaluating on validation ...")
        val_per_feat = evaluate(model, val_ds, max_proteins=len(val_ids)).tolist()

        history["epoch"].append(epoch + 1)
        history["train_per_feat"].append(train_per_feat)
        history["val_per_feat"].append(val_per_feat)
        scheduler.step()

        # Save checkpoint per epoch
        torch.save({
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "epoch": epoch + 1,
            "history": history,
        }, CACHE / f"checkpoint_epoch_{epoch+1}.pt")
        (CACHE / "loss_history.json").write_text(json.dumps({
            "feature_names": feature_names,
            "epochs": history["epoch"],
            "train_per_feat": history["train_per_feat"],
            "val_per_feat": history["val_per_feat"],
        }, indent=2))

        elapsed_total = time.time() - t_total
        print(f"  EPOCH {epoch+1}/{N_EPOCHS} done in {time.time() - t_epoch:.0f}s "
              f"(total {elapsed_total:.0f}s)")
        tr_str = "  ".join(f"{fn}={v:.4f}" for fn, v in zip(feature_names, train_per_feat))
        vl_str = "  ".join(f"{fn}={v:.4f}" for fn, v in zip(feature_names, val_per_feat))
        print(f"    train: {tr_str}")
        print(f"    val:   {vl_str}")

    return {"feature_names": feature_names, **history}


# =================================================================
# STEP 7: Plot Figs 3.25 / 3.26
# =================================================================

FEAT_COLORS = ["#cc3322", "#cc8822", "#22aa44", "#3366cc", "#7733aa"]


def plot_loss_curves(history: dict, n_train: int, n_val: int):
    epochs = history["epoch"]
    feats = history["feature_names"]
    tr = np.array(history["train_per_feat"])
    vl = np.array(history["val_per_feat"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, fn in enumerate(feats):
        ax.plot(epochs, tr[:, k], "-o", color=FEAT_COLORS[k], linewidth=1.8, label=fn)
    ax.set_xlabel("epoch"); ax.set_ylabel("training log-cosh loss")
    ax.set_title(f"Fig 3.25 analog — training loss per orientational feature\n"
                 f"(full PSP pipeline: {n_train} DEEPCOV train proteins, ESM-2-650M, GraphTransformer 3×4, log-cosh per Pepe Eq 3.17)")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_fig_3_25_training_loss_curves.png", dpi=130, bbox_inches="tight")
    print(f"  wrote {OUT_DIR / 'qa_fig_3_25_training_loss_curves.png'}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, fn in enumerate(feats):
        ax.plot(epochs, vl[:, k], "-o", color=FEAT_COLORS[k], linewidth=1.8, label=fn)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation log-cosh loss")
    ax.set_title(f"Fig 3.26 analog — validation loss per orientational feature\n"
                 f"(full PSP pipeline: {n_val} PSICOV held-out proteins)")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_fig_3_26_validation_loss_curves.png", dpi=130, bbox_inches="tight")
    print(f"  wrote {OUT_DIR / 'qa_fig_3_26_validation_loss_curves.png'}")
    plt.close(fig)


# =================================================================
# main
# =================================================================

def main() -> int:
    print(f"=== Pepe Ch3 full PSP pipeline ===")
    print(f"  ESM model: {ESM_MODEL}  layer={ESM_LAYER}  dim={ESM_DIM}")
    print(f"  N_train={N_TRAIN} DEEPCOV, N_val={N_VAL} PSICOV")
    print(f"  Graph Transformer: {N_LAYERS} layers × {N_HEADS} heads, hidden={HIDDEN}")
    print(f"  {N_EPOCHS} epochs, lr={INIT_LR} γ={LR_DECAY}\n")

    print("STEP 1: Select proteins from PDNET FASTA bundle")
    train_proteins, val_proteins = select_proteins()
    all_proteins = train_proteins + val_proteins

    print("\nSTEP 2: Download PDB files (resumable)")
    download_all_pdbs(all_proteins)

    print("\nSTEP 3: Compute orientational maps from real PDB structures")
    compute_and_cache_maps(all_proteins)

    print("\nSTEP 4: Run ESM-2-650M inference (resumable, ~3 hours on CPU)")
    run_esm_inference(all_proteins)

    print("\nSTEP 5: Filter to proteins with both ESM + maps")
    valid_train = get_valid_protein_ids(train_proteins)
    valid_val = get_valid_protein_ids(val_proteins)
    print(f"  {len(valid_train)} train proteins survive filtering (of {len(train_proteins)})")
    print(f"  {len(valid_val)} val proteins survive filtering (of {len(val_proteins)})")

    print("\nSTEP 6: Train Graph Transformer")
    history = train_loop(valid_train, valid_val)

    print("\nSTEP 7: Render Figs 3.25 / 3.26")
    plot_loss_curves(history, n_train=len(valid_train), n_val=len(valid_val))

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
