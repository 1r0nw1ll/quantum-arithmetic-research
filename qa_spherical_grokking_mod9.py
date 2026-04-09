"""
QA Spherical Grokking Experiment — Mod-9 Addition
==================================================
Tests the combined prediction from cert families [198], [199], [200]:

    A spherical transformer trained on mod-9 addition should discover
    exactly 5 modes during grokking, with frequencies matching QA's
    3 Cosmos (period 24) + 1 Satellite (period 8) + 1 Singularity (period 1).

Implements:
  - Standard transformer (baseline) on mod-9 addition
  - Spherical transformer (L2-normalized, Yildirim 2603.05228) on mod-9 addition
  - Eigenvalue extraction post-grokking (Schiffman 2602.22600)
  - Fourier mode analysis (Pudelko 2510.24882)
  - Residual stream magnitude tracking

QA axiom compliance:
  - All QA state in {1,...,m}, never {0,...,m-1} (A1) — note: NN uses 0-indexed
    internally; QA interpretation layer adds 1
  - No b*b power shorthand (S1) — only b*b where needed
  - No float QA state (S2) — NN floats are observer projections (Theorem NT)
  - Continuous outputs (eigenvalues, Fourier modes) are observer projections ONLY

Author: Will Dale
"""
QA_COMPLIANCE = {
    "observer": "neural_network_weights_and_activations",
    "state_alphabet": "mod9",
    "cert_family": "[198]+[199]+[200]",
    "theorem_nt": "All NN continuous values (eigenvalues, Fourier modes, logits) "
                  "are observer projections. QA discrete structure (orbit type, "
                  "mode count) is the causal layer. The spherical constraint "
                  "architecturally enforces Theorem NT by removing magnitude DOF.",
}

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# ─── Configuration ───────────────────────────────────────────────────────────

SEED = 42
TRAIN_FRACTION = 0.5   # 50% train — standard in grokking literature
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 1
LR = 1e-3
NUM_EPOCHS = 100000
LOG_EVERY = 200
DEVICE = "cpu"

# Spherical transformer parameters (Yildirim 2603.05228)
SPHERE_EPSILON = 1e-6
SPHERE_TAU = 10.0     # Temperature for cosine similarity logits
LOGIT_CLAMP = 10.0    # Clamp logits to [-10, 10]

# Run both m=9 (QA target, composite) and m=97 (prime, control)
MODULI = [97, 9]

# Eigenvalue extraction: at these fractions of total epochs
EIGENVALUE_EXTRACT_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]

OUTPUT_DIR = Path("results_spherical_grokking_mod9")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ─── Dataset ─────────────────────────────────────────────────────────────────

def make_mod_addition_dataset(m):
    """Generate all (x, y) -> (x+y) mod m pairs as token sequences."""
    data = []
    targets = []
    for x in range(m):
        for y in range(m):
            data.append(torch.tensor([x, y], dtype=torch.long))
            targets.append((x + y) % m)
    data = torch.stack(data)
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets


def train_test_split(data, targets, train_frac, seed=42):
    """Split into train/test by fraction."""
    n = len(data)
    n_train = int(train_frac * n)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return (data[train_idx], targets[train_idx],
            data[test_idx], targets[test_idx])


# ─── Models ──────────────────────────────────────────────────────────────────

class StandardTransformer(nn.Module):
    """Standard 1-layer transformer for modular addition (baseline)."""

    def __init__(self, d_model, num_heads, num_layers, vocab_size, seq_len=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=4*d_model,
            batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x_emb = self.embedding(x)
        out = self.transformer(x_emb)
        logits = self.linear(out[:, -1])  # Take last position
        return logits

    def get_residual_norms(self, x):
        """Extract residual stream norms for Theorem NT tracking."""
        with torch.no_grad():
            x_emb = self.embedding(x)
            out = self.transformer(x_emb)
            return out.norm(dim=-1).mean().item()


class SphericalTransformer(nn.Module):
    """
    Spherical transformer (Yildirim 2603.05228).
    L2-normalizes activations after each sublayer, uses cosine similarity
    for logits. Removes magnitude as a degree of freedom.

    QA interpretation: This architecturally enforces Theorem NT by
    preventing continuous magnitude from serving as a causal input.
    The network is forced to encode information in discrete phase
    relationships on the unit hypersphere.
    """

    def __init__(self, d_model, num_heads, num_layers, vocab_size, seq_len=2,
                 epsilon=SPHERE_EPSILON, tau=SPHERE_TAU, logit_clamp=LOGIT_CLAMP):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon
        self.tau = tau
        self.logit_clamp = logit_clamp

        self.embedding = nn.Embedding(vocab_size, d_model)
        # Build transformer sublayers manually for spherical projection
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Classification head: fixed-norm prototype vectors (no bias)
        self.class_prototypes = nn.Linear(d_model, vocab_size, bias=False)

    def _sphere_project(self, x):
        """Project onto unit hypersphere: x / max(||x||, epsilon)."""
        norms = x.norm(dim=-1, keepdim=True)
        return x / torch.clamp(norms, min=self.epsilon)

    def forward(self, x):
        # Embed and project to sphere
        h = self._sphere_project(self.embedding(x))

        # Self-attention sublayer + spherical projection
        attn_out, _ = self.self_attn(h, h, h)
        h = self._sphere_project(self.norm1(h + attn_out))

        # Feedforward sublayer + spherical projection
        ff_out = self.ff(h)
        h = self._sphere_project(self.norm2(h + ff_out))

        # Take last position
        h_last = h[:, -1]  # [batch, d_model], unit norm

        # Cosine similarity logits (both h and prototypes normalized)
        proto_normed = F.normalize(self.class_prototypes.weight, dim=-1)
        logits = self.tau * (h_last @ proto_normed.T)  # scaled cosine sim
        logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        return logits

    def get_residual_norms(self, x):
        """Should always be ~1.0 for spherical model."""
        with torch.no_grad():
            h = self._sphere_project(self.embedding(x))
            attn_out, _ = self.self_attn(h, h, h)
            h = self._sphere_project(self.norm1(h + attn_out))
            ff_out = self.ff(h)
            h = self._sphere_project(self.norm2(h + ff_out))
            return h.norm(dim=-1).mean().item()


# ─── Eigenvalue / Fourier Mode Extraction ────────────────────────────────────

def extract_operator_eigenvalues(model, m):
    """
    Extract the linear operator the model has learned for addition mod m.

    For a model computing f(x,y) = (x+y) mod m, the Fourier representation
    should be diagonal in the DFT basis with eigenvalues e^{2*pi*i*k/m}
    for each active frequency k.

    We probe the model's embedding space to extract the effective operator.
    """
    model.eval()
    with torch.no_grad():
        # Get embedding vectors for each token
        if hasattr(model, 'embedding'):
            W_emb = model.embedding.weight.detach().cpu().numpy()  # [m, d_model]
        else:
            return None, None

        # Compute the model's output matrix: M[x,y] = argmax f(x,y)
        output_matrix = np.zeros((m, m), dtype=int)
        logit_matrix = np.zeros((m, m, m))
        for x in range(m):
            for y in range(m):
                inp = torch.tensor([[x, y]], dtype=torch.long)
                logits = model(inp).detach().cpu().numpy()[0]
                output_matrix[x, y] = np.argmax(logits)
                logit_matrix[x, y] = logits

        # Accuracy check
        correct = 0
        for x in range(m):
            for y in range(m):
                if output_matrix[x, y] == (x + y) % m:
                    correct += 1
        accuracy = correct / (m * m)

        # DFT analysis of the logit matrix
        # For each output class c, compute the 2D DFT of logit_matrix[:,:,c]
        # The model should be sparse in frequency domain after grokking
        dft_power = np.zeros((m, m))
        for c in range(m):
            fft2d = np.fft.fft2(logit_matrix[:, :, c])
            dft_power += np.abs(fft2d)

        # Extract dominant frequencies
        # Normalize
        dft_power_norm = dft_power / dft_power.max() if dft_power.max() > 0 else dft_power

        # Find modes above threshold
        threshold = 0.1
        dominant_modes = []
        for kx in range(m):
            for ky in range(m):
                if dft_power_norm[kx, ky] > threshold:
                    dominant_modes.append({
                        'kx': kx, 'ky': ky,
                        'power': float(dft_power_norm[kx, ky]),
                        'period_x': m / kx if kx > 0 else float('inf'),
                        'period_y': m / ky if ky > 0 else float('inf'),
                    })

        # Sort by power
        dominant_modes.sort(key=lambda d: d['power'], reverse=True)

        # Eigenvalue extraction from embedding space
        # The key insight (Schiffman): after grokking, the embedding matrix
        # contains rotation operators. We extract via SVD of the cross-correlation.
        # For addition mod m, the "shift" operator T_1[x] = (x+1) mod m
        # should be represented as a rotation.
        shift_matrix = np.zeros((m, m))
        for x in range(m):
            shift_matrix[(x + 1) % m, x] = 1.0

        # Project shift operator into embedding space
        # T_emb = W_emb^T @ shift @ W_emb (pseudoinverse)
        W_pinv = np.linalg.pinv(W_emb)
        T_emb = W_pinv @ shift_matrix @ W_emb

        eigenvalues = np.linalg.eigvals(T_emb)

        # Eigenvalues on unit circle = discrete modes
        eig_magnitudes = np.abs(eigenvalues)
        eig_phases = np.angle(eigenvalues)

        return {
            'accuracy': accuracy,
            'output_matrix': output_matrix,
            'dominant_modes': dominant_modes,
            'n_dominant_modes': len(dominant_modes),
            'eigenvalues': eigenvalues,
            'eig_magnitudes': eig_magnitudes,
            'eig_phases': eig_phases,
            'dft_power': dft_power_norm,
        }, dft_power_norm


def count_unit_circle_eigenvalues(eigenvalues, tol=0.1):
    """
    Count eigenvalues on or near the unit circle.
    After grokking, these should snap from |lambda|<1 to |lambda|=1
    (Schiffman 2602.22600).
    """
    mags = np.abs(eigenvalues)
    on_circle = np.sum(np.abs(mags - 1.0) < tol)
    return int(on_circle)


def classify_eigenvalue_periods(eigenvalues, m, tol=0.05):
    """
    Classify eigenvalues by their implied period.
    An eigenvalue lambda = e^{2*pi*i*k/p} has period p.

    QA prediction for m=9:
      - 3 modes with period dividing 24 (Cosmos)
      - 1 mode with period dividing 8 (Satellite)
      - 1 mode with period 1 (Singularity / DC component)
    """
    mags = np.abs(eigenvalues)
    phases = np.angle(eigenvalues)

    classifications = []
    for i, (mag, phase) in enumerate(zip(mags, phases)):
        if abs(mag - 1.0) > 0.2:
            classifications.append({'idx': i, 'mag': float(mag), 'phase': float(phase),
                                    'type': 'off-circle', 'period': None})
            continue

        # Determine period from phase: phase = 2*pi*k/period
        if abs(phase) < tol:
            # DC component / period 1
            classifications.append({'idx': i, 'mag': float(mag), 'phase': float(phase),
                                    'type': 'singularity', 'period': 1})
        else:
            # period = 2*pi / |phase| ... but on Z/mZ the period must divide m
            # or a multiple of m (if the mode sees the full Pisano period)
            raw_period = abs(2 * np.pi / phase)
            # Check against QA orbit periods
            best_match = None
            for p in [1, 3, 8, 9, 24]:
                # Does k/p give a phase close to what we see?
                for k in range(1, p):
                    expected_phase = 2 * np.pi * k / p
                    if abs(abs(phase) - expected_phase) < tol or abs(abs(phase) - (2*np.pi - expected_phase)) < tol:
                        if best_match is None or p < best_match:
                            best_match = p

            if best_match == 1:
                orbit_type = 'singularity'
            elif best_match in (8,):
                orbit_type = 'satellite'
            elif best_match in (3, 9, 24):
                orbit_type = 'cosmos'
            else:
                orbit_type = 'unclassified'

            classifications.append({'idx': i, 'mag': float(mag), 'phase': float(phase),
                                    'type': orbit_type, 'period': best_match,
                                    'raw_period': float(raw_period)})

    return classifications


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_model(model, train_data, train_targets, test_data, test_targets,
                model_name, m, lr=LR, num_epochs=NUM_EPOCHS, wd=0.0,
                eig_extract_epochs=None):
    """Train a model and track metrics for QA analysis."""
    if eig_extract_epochs is None:
        eig_extract_epochs = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  betas=(0.9, 0.99), eps=1e-25)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'residual_norm': [], 'epoch': [],
        'n_unit_circle_eigs': [],
    }
    eigenvalue_snapshots = {}
    grokking_epoch = None

    print(f"\n{'='*60}")
    print(f"Training: {model_name} on mod-{m} addition")
    print(f"{'='*60}")
    start = time.time()

    for epoch in range(num_epochs):
        # ── Train step ──
        model.train()
        perm = torch.randperm(len(train_data))
        x_train = train_data[perm]
        y_train = train_targets[perm]

        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        # ── Evaluate ──
        if epoch % LOG_EVERY == 0 or epoch in eig_extract_epochs:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_data)
                train_loss = loss_fn(train_logits, train_targets).item()
                train_preds = train_logits.argmax(dim=-1)
                train_acc = (train_preds == train_targets).float().mean().item()

                test_logits = model(test_data)
                test_loss = loss_fn(test_logits, test_targets).item()
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_targets).float().mean().item()

                res_norm = model.get_residual_norms(test_data[:16])

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['residual_norm'].append(res_norm)

            # Detect grokking: first epoch where test acc > 95%
            if grokking_epoch is None and test_acc > 0.95:
                grokking_epoch = epoch
                print(f"  *** GROKKING at epoch {epoch} (test_acc={test_acc:.3f}) ***")

            if epoch % (LOG_EVERY * 10) == 0:
                print(f"  Epoch {epoch:6d}: train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
                      f"loss={train_loss:.4f} ||h||={res_norm:.3f}")

        # ── Eigenvalue extraction at key epochs ──
        if epoch in eig_extract_epochs:
            eig_data, _ = extract_operator_eigenvalues(model, m)
            if eig_data:
                n_uc = count_unit_circle_eigenvalues(eig_data['eigenvalues'])
                history['n_unit_circle_eigs'].append((epoch, n_uc))
                eigenvalue_snapshots[epoch] = eig_data
                print(f"  Epoch {epoch}: {n_uc} eigenvalues on unit circle, "
                      f"accuracy={eig_data['accuracy']:.3f}, "
                      f"{eig_data['n_dominant_modes']} dominant Fourier modes")

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    if grokking_epoch:
        print(f"Grokking epoch: {grokking_epoch}")
    else:
        print("WARNING: No grokking detected (test_acc never exceeded 95%)")

    return history, eigenvalue_snapshots, grokking_epoch


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_training_curves(hist_std, hist_sph, grok_std, grok_sph):
    """Compare standard vs spherical training dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    ax = axes[0, 0]
    ax.plot(hist_std['epoch'], hist_std['test_acc'], 'b-', label='Standard test', alpha=0.8)
    ax.plot(hist_sph['epoch'], hist_sph['test_acc'], 'r-', label='Spherical test', alpha=0.8)
    ax.plot(hist_std['epoch'], hist_std['train_acc'], 'b--', label='Standard train', alpha=0.4)
    ax.plot(hist_sph['epoch'], hist_sph['train_acc'], 'r--', label='Spherical train', alpha=0.4)
    if grok_std:
        ax.axvline(grok_std, color='b', linestyle=':', alpha=0.5, label=f'Std grok @{grok_std}')
    if grok_sph:
        ax.axvline(grok_sph, color='r', linestyle=':', alpha=0.5, label=f'Sph grok @{grok_sph}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy: Standard vs Spherical')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[0, 1]
    ax.semilogy(hist_std['epoch'], hist_std['train_loss'], 'b-', label='Standard', alpha=0.8)
    ax.semilogy(hist_sph['epoch'], hist_sph['train_loss'], 'r-', label='Spherical', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (log)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual norms (Theorem NT indicator)
    ax = axes[1, 0]
    ax.plot(hist_std['epoch'], hist_std['residual_norm'], 'b-', label='Standard ||h||')
    ax.plot(hist_sph['epoch'], hist_sph['residual_norm'], 'r-', label='Spherical ||h||')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='Unit sphere')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual Stream Norm')
    ax.set_title('Residual Norms (Theorem NT: spherical should be ~1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Unit circle eigenvalue count over training
    ax = axes[1, 1]
    if hist_std['n_unit_circle_eigs']:
        epochs_s, counts_s = zip(*hist_std['n_unit_circle_eigs'])
        ax.plot(epochs_s, counts_s, 'bo-', label='Standard')
    if hist_sph['n_unit_circle_eigs']:
        epochs_p, counts_p = zip(*hist_sph['n_unit_circle_eigs'])
        ax.plot(epochs_p, counts_p, 'ro-', label='Spherical')
    ax.axhline(5, color='g', linestyle='--', alpha=0.5, label='QA prediction: 5 modes')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('# Eigenvalues on Unit Circle')
    ax.set_title('Eigenvalue Transition (Schiffman: snap to |λ|=1 at grokking)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'training_comparison.png'}")


def plot_eigenvalue_spectrum(eig_data, model_name, epoch):
    """Plot eigenvalues in complex plane — unit circle = QA modes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    eigenvalues = eig_data['eigenvalues']

    # Complex plane
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    ax.scatter(eigenvalues.real, eigenvalues.imag, c='red', s=60, zorder=5)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title(f'{model_name} @ epoch {epoch}\nEigenvalues in Complex Plane')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # DFT power spectrum
    ax = axes[1]
    dft = eig_data['dft_power']
    im = ax.imshow(dft, cmap='hot', origin='lower', aspect='equal')
    ax.set_xlabel('ky (frequency)')
    ax.set_ylabel('kx (frequency)')
    ax.set_title(f'{model_name} @ epoch {epoch}\n2D DFT Power (dominant modes = active frequencies)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fname = OUTPUT_DIR / f'eigenvalues_{model_name.lower().replace(" ", "_")}_epoch{epoch}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_qa_orbit_classification(classifications, model_name, epoch, m):
    """Visualize eigenvalue classification into QA orbit types."""
    fig, ax = plt.subplots(figsize=(8, 8))

    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

    colors = {'singularity': 'gold', 'satellite': 'blue',
              'cosmos': 'red', 'unclassified': 'gray', 'off-circle': 'lightgray'}
    markers = {'singularity': '*', 'satellite': 's',
               'cosmos': 'o', 'unclassified': 'x', 'off-circle': '.'}

    for c in classifications:
        mag, phase = c['mag'], c['phase']
        x, y = mag * np.cos(phase), mag * np.sin(phase)
        ax.scatter(x, y, c=colors[c['type']], marker=markers[c['type']],
                   s=100, zorder=5, edgecolors='k', linewidths=0.5)

    # Legend
    for orbit_type in ['singularity', 'satellite', 'cosmos', 'unclassified', 'off-circle']:
        ax.scatter([], [], c=colors[orbit_type], marker=markers[orbit_type],
                   s=100, label=orbit_type, edgecolors='k', linewidths=0.5)

    # Count
    type_counts = {}
    for c in classifications:
        t = c['type']
        type_counts[t] = type_counts.get(t, 0) + 1

    count_str = ', '.join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    ax.set_title(f'{model_name} @ epoch {epoch}\n'
                 f'QA Orbit Classification (m={m})\n{count_str}')
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = OUTPUT_DIR / f'qa_orbits_{model_name.lower().replace(" ", "_")}_epoch{epoch}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ─── QA Prediction Testing ──────────────────────────────────────────────────

def test_qa_predictions(eig_data_std, eig_data_sph, grok_std, grok_sph, m):
    """
    Test the combined QA prediction from [198]+[199]+[200]:

    1. Spherical should grok faster than standard (Yildirim [200])
    2. Eigenvalues should snap to unit circle at grokking (Schiffman [199])
    3. Exactly 5 modes should emerge for m=9 (Pudelko [198])
    4. Mode frequencies should match QA orbits:
       3 Cosmos (period 24) + 1 Satellite (period 8) + 1 Singularity (period 1)
    """
    results = {}
    print(f"\n{'='*60}")
    print("QA PREDICTION TESTS")
    print(f"{'='*60}")

    # Test 1: Grokking speedup
    if grok_std and grok_sph:
        speedup = grok_std / grok_sph
        results['speedup'] = speedup
        results['test1_pass'] = speedup > 2.0  # Expect >10x but accept >2x
        print(f"\n[Test 1] Grokking speedup (Yildirim [200]):")
        print(f"  Standard: epoch {grok_std}")
        print(f"  Spherical: epoch {grok_sph}")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  PASS: {results['test1_pass']} (need >2x, expect ~10-20x)")
    elif grok_sph and not grok_std:
        results['test1_pass'] = True  # Spherical grokkd, standard didn't
        print(f"\n[Test 1] Spherical grokked at {grok_sph}, standard did NOT grok — PASS (extreme)")
    else:
        results['test1_pass'] = False
        print(f"\n[Test 1] FAIL — could not compare grokking times")

    # Test 2: Eigenvalues on unit circle after grokking
    for name, eig_data in [('Standard', eig_data_std), ('Spherical', eig_data_sph)]:
        if eig_data:
            n_uc = count_unit_circle_eigenvalues(eig_data['eigenvalues'])
            results[f'n_unit_circle_{name.lower()}'] = n_uc
            print(f"\n[Test 2] Unit circle eigenvalues — {name} (Schiffman [199]):")
            print(f"  {n_uc} eigenvalues with |λ| ≈ 1 (of {len(eig_data['eigenvalues'])})")

    # Test 3: Mode count = 5 for m=9
    for name, eig_data in [('Standard', eig_data_std), ('Spherical', eig_data_sph)]:
        if eig_data:
            n_modes = eig_data['n_dominant_modes']
            results[f'n_modes_{name.lower()}'] = n_modes
            print(f"\n[Test 3] Dominant Fourier mode count — {name} (Pudelko [198]):")
            print(f"  {n_modes} dominant modes")
            print(f"  QA prediction: 5 modes (floor(9/2)+1 = 5)")
            print(f"  MATCH: {n_modes == 5}")

    # Test 4: Orbit classification
    for name, eig_data in [('Standard', eig_data_std), ('Spherical', eig_data_sph)]:
        if eig_data:
            classifications = classify_eigenvalue_periods(
                eig_data['eigenvalues'], m)
            on_circle = [c for c in classifications if c['type'] != 'off-circle']
            type_counts = {}
            for c in on_circle:
                t = c['type']
                type_counts[t] = type_counts.get(t, 0) + 1

            results[f'orbit_counts_{name.lower()}'] = type_counts
            print(f"\n[Test 4] QA orbit classification — {name}:")
            print(f"  Counts: {type_counts}")
            print(f"  QA prediction: cosmos=3, satellite=1, singularity=1")

            # Check prediction
            cosmos = type_counts.get('cosmos', 0)
            satellite = type_counts.get('satellite', 0)
            singularity = type_counts.get('singularity', 0)
            prediction_match = (cosmos == 3 and satellite == 1 and singularity == 1)
            results[f'orbit_match_{name.lower()}'] = prediction_match
            print(f"  EXACT MATCH: {prediction_match}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def run_single_modulus(m, num_epochs, out_dir):
    """Run the full experiment for one modulus."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    eig_extract_epochs = sorted(set(
        max(1, int(f * num_epochs)) for f in EIGENVALUE_EXTRACT_FRACTIONS
    ))

    print(f"\n{'#'*60}")
    print(f"  mod-{m} ({'prime' if is_prime(m) else 'COMPOSITE — QA target'})")
    print(f"{'#'*60}")

    data, targets = make_mod_addition_dataset(m)
    train_data, train_targets, test_data, test_targets = train_test_split(
        data, targets, TRAIN_FRACTION, seed=SEED)
    print(f"Dataset: {m*m} total, {len(train_data)} train, {len(test_data)} test")

    # Standard Transformer (needs weight decay for grokking)
    model_std = StandardTransformer(D_MODEL, NUM_HEADS, NUM_LAYERS, m, seq_len=2)
    hist_std, eigs_std, grok_std = train_model(
        model_std, train_data, train_targets, test_data, test_targets,
        f"Standard (m={m})", m, lr=LR, wd=1.0, num_epochs=num_epochs,
        eig_extract_epochs=eig_extract_epochs)

    # Spherical Transformer (no WD needed per Yildirim)
    model_sph = SphericalTransformer(D_MODEL, NUM_HEADS, NUM_LAYERS, m, seq_len=2)
    hist_sph, eigs_sph, grok_sph = train_model(
        model_sph, train_data, train_targets, test_data, test_targets,
        f"Spherical (m={m})", m, lr=LR, wd=0.0, num_epochs=num_epochs,
        eig_extract_epochs=eig_extract_epochs)

    # Extract final eigenvalues
    eig_std_final, _ = extract_operator_eigenvalues(model_std, m)
    eig_sph_final, _ = extract_operator_eigenvalues(model_sph, m)

    # Plots
    global OUTPUT_DIR
    saved_dir = OUTPUT_DIR
    OUTPUT_DIR = out_dir

    plot_training_curves(hist_std, hist_sph, grok_std, grok_sph)
    for tag, eig_final in [("Standard", eig_std_final), ("Spherical", eig_sph_final)]:
        if eig_final:
            plot_eigenvalue_spectrum(eig_final, f"{tag}_m{m}", num_epochs)
            cls = classify_eigenvalue_periods(eig_final['eigenvalues'], m)
            plot_qa_orbit_classification(cls, f"{tag}_m{m}", num_epochs, m)
    for tag, eigs in [("Standard", eigs_std), ("Spherical", eigs_sph)]:
        for ep, ed in sorted(eigs.items()):
            if ed['accuracy'] > 0.5:
                plot_eigenvalue_spectrum(ed, f"{tag}_m{m}", ep)

    OUTPUT_DIR = saved_dir

    results = test_qa_predictions(eig_std_final, eig_sph_final, grok_std, grok_sph, m)
    return {
        'modulus': m, 'is_prime': is_prime(m),
        'grok_std': grok_std, 'grok_sph': grok_sph,
        'acc_std': eig_std_final['accuracy'] if eig_std_final else None,
        'acc_sph': eig_sph_final['accuracy'] if eig_sph_final else None,
        'qa_results': results,
    }


def main():
    print(f"QA Spherical Grokking Experiment")
    print(f"Combined prediction: [198] Pudelko + [199] Schiffman + [200] Yildirim")
    print(f"Moduli: {MODULI} — prime control then QA composite target")
    print(f"Seed: {SEED}, d_model: {D_MODEL}, lr: {LR}, train_frac: {TRAIN_FRACTION}")

    all_results = {}
    for m in MODULI:
        # m=97 (prime): grokking expected ~10k-50k epochs, cap at 50k
        # m=9 (composite): harder, give full budget
        epochs = 50000 if m >= 50 else NUM_EPOCHS
        out = Path(f"results_spherical_grokking_m{m}")
        all_results[m] = run_single_modulus(m, epochs, out)

    # Save combined
    def jsonify(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, complex):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        if isinstance(obj, dict):
            return {str(k): jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [jsonify(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        return obj

    with open(OUTPUT_DIR / 'results_all_moduli.json', 'w') as f:
        json.dump(jsonify(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print("COMPARATIVE VERDICT")
    print(f"{'='*60}")
    for m, r in all_results.items():
        tag = "PRIME" if r['is_prime'] else "COMPOSITE"
        print(f"\n  m={m} ({tag}):")
        print(f"    Standard:  grok@{r['grok_std']}, acc={r['acc_std']}")
        print(f"    Spherical: grok@{r['grok_sph']}, acc={r['acc_sph']}")
        if r['grok_std'] and r['grok_sph']:
            print(f"    Speedup: {r['grok_std']/r['grok_sph']:.1f}x")
        elif r['grok_sph'] and not r['grok_std']:
            print(f"    Spherical grokked, standard did NOT")
        elif r['grok_std'] and not r['grok_sph']:
            print(f"    Standard grokked, spherical did NOT")
        else:
            print(f"    Neither grokked")


if __name__ == "__main__":
    main()
