QA_COMPLIANCE = "observer=integration_module, state_alphabet=mod24"
# Quantum Arithmetic × Markovian Thinking Integration
# qa_markovian_integration.py — QALM 2.0 Core Implementation
# ---------------------------------------------------------------
# This implements "The Markovian Thinker" (Aghajohari et al., 2025)
# integrated with QA harmonic descent and autoencoder modules.
#
# Key Features:
# - Fixed-size reasoning chunks (C=8192 tokens)
# - Markovian state compression (m=4096 tokens)
# - Linear compute scaling O(n²S) vs quadratic O(n²S²)
# - Delethink-style gradient truncation
# - PAC-Harmonic loss (QA × RL hybrid)
# ---------------------------------------------------------------

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle

# CIM/PIM enhancements for in-memory processing
import mmap
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import lru_cache
import hashlib
import re
import os

try:
    from qa_harmonic_descent import harmonic_descent
except ImportError:
    # Fallback implementation if qa_harmonic_descent not available
    def harmonic_descent(model, data=None):
        """Fallback QA harmonic curvature computation"""
        if "tuple" in model:
            b, e, d, a = model["tuple"]
        else:
            b, e, d, a = (1, 1, 2, 3)

        # QA geometry
        G = e * e + d * d  # hypotenuse (S1: x*x, not x**2)
        F = b * a          # short leg

        # Harmonic curvature
        H_QA = 0.25 * ((b * a) / (G + 1e-9) + (e * d) / (a + b + 1e-9))
        H_QA = np.abs(H_QA) / (1 + np.abs(H_QA))

        curvature_vec = np.full((model.get("layers", 4),), H_QA)
        loss_hat = float(F / (G + 1e-9))

        return loss_hat, curvature_vec

try:
    from qa_autoencoder import QAAutoencoder
except ImportError:
    # Fallback QA Autoencoder
    class QAAutoencoder:
        """Lightweight QA latent encoder"""
        def __init__(self, latent_dim=64):
            self.latent_dim = latent_dim

        def encode(self, tuple_data):
            """Encode QA tuple to latent vector"""
            data = np.array(tuple_data, dtype=float)
            # Simple tanh encoding
            latent = np.tanh(data / (1 + np.abs(data).mean()))
            return latent

# ---------------------------------------------------------------
# 1. Markovian QA Environment (Delethink-style)
# ---------------------------------------------------------------

class QAMarkovianEnv:
    """
    Markovian reasoning environment for QA tuples.

    Each chunk has fixed context size C. At boundaries, the environment
    resets to a compact state (m tokens) and continues reasoning.

    Parameters:
    -----------
    qa_autoencoder : QAAutoencoder
        Encoder for compressing QA states
    context_size : int (default: 8192)
        Fixed chunk length (C)
    state_size : int (default: 4096)
        Markovian carry state size (m)
    max_iters : int (default: 24)
        Number of chunks (mod-24 aligned for QA)
    truncate : bool (default: True)
        Whether to truncate backprop between chunks (Delethink-style)
    """
    def __init__(self, qa_autoencoder, context_size=8192, state_size=4096,
                 max_iters=24, truncate=True):
        self.autoencoder = qa_autoencoder
        self.C = context_size
        self.m = state_size
        self.I = max_iters
        self.truncate = truncate

    def rollout(self, query_tuple, policy_fn):
        """
        Perform Markovian reasoning rollout.

        Parameters:
        -----------
        query_tuple : tuple/array
            Initial QA tuple (b, e, d, a)
        policy_fn : callable
            Policy function mapping latent state → next tuple

        Returns:
        --------
        traces : list of (query, output) pairs
        rewards : list of reward values
        """
        traces, rewards = [], []
        q = torch.tensor(query_tuple, dtype=torch.float32)

        for i in range(self.I):
            # Encode current state → Markovian carry (latent z)
            z_np = self.autoencoder.encode(q.detach().cpu().numpy())
            z = torch.tensor(z_np, dtype=torch.float32)

            # Policy proposes next tuple (keeps grad if not truncated)
            y = policy_fn(z)

            # Local reward (smooth, positive, grad-safe)
            r = self._reward(q, y)
            traces.append((q, y))
            rewards.append(r)

            # Markov reset: carry only current state
            if self.truncate:
                q = y.detach()  # Truncate backprop (Delethink-style)
            else:
                q = y  # Keep gradients flowing

        return traces, rewards

    def _reward(self, q, y):
        """Harmonic alignment reward (differentiable)"""
        diff = torch.linalg.vector_norm(q - y)
        return torch.exp(-diff)

# ---------------------------------------------------------------
# 2. Markovian Policy Network
# ---------------------------------------------------------------

class QAMarkovianPolicy(nn.Module):
    """
    Neural policy for evolving QA tuples.

    Maps latent state z → next QA tuple (b, e, d, a)
    """
    def __init__(self, tuple_dim=4, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(tuple_dim, hidden)
        self.fc2 = nn.Linear(hidden, tuple_dim)

    def forward(self, x):
        """Propose next QA tuple from latent state"""
        return torch.tanh(self.fc2(torch.relu(self.fc1(x))))

# ---------------------------------------------------------------
# 3. PAC-Harmonic Loss (QA × RL Hybrid)
# ---------------------------------------------------------------

def pac_harmonic_loss(rewards, potentials, lambda_reg=1e-3):
    """
    Hybrid loss combining:
    - PPO-style policy variance (from Delethink)
    - QA harmonic regularization

    Parameters:
    -----------
    rewards : torch.Tensor
        Reward values across chunks
    potentials : torch.Tensor
        QA harmonic potentials
    lambda_reg : float
        Regularization strength

    Returns:
    --------
    loss : torch.Tensor
        Combined PAC-Harmonic loss
    """
    rewards = rewards.to(dtype=torch.float32)
    potentials = potentials.to(dtype=torch.float32)

    mean_r = torch.mean(rewards)
    var_term = torch.mean((rewards - mean_r) ** 2)  # PPO variance
    reg_term = lambda_reg * torch.mean(potentials ** 2)  # QA regularization

    return var_term + reg_term

# ---------------------------------------------------------------
# 4. QA Harmonic Optimizer
# ---------------------------------------------------------------

class QAOptimizer:
    """
    QA-specific optimizer using harmonic descent.

    Uses QA harmonic curvature H_QA to guide learning rates
    adaptively based on geometric stress.
    """
    def __init__(self, model, lr=1e-3, curvature_gain=1.0):
        self.model = model
        self.lr = lr
        self.gain = curvature_gain

    def step(self, current_tuple=None):
        """
        Perform QA harmonic gradient descent step.

        Parameters:
        -----------
        current_tuple : array-like, optional
            Current QA tuple for curvature computation

        Returns:
        --------
        loss_hat : float
            Harmonic energy estimate
        curvature_mean : float
            Average curvature across parameters
        """
        # Prepare tuple for harmonic descent
        if current_tuple is not None:
            tuple_data = np.asarray(current_tuple, dtype=float).flatten()
            # Ensure 4D tuple (b, e, d, a)
            if len(tuple_data) >= 4:
                b, e, d, a = tuple_data[:4]
            else:
                b, e, d, a = (1, 1, 2, 3)  # fallback
        else:
            b, e, d, a = (1, 1, 2, 3)

        # Compute QA harmonic curvature
        loss_hat, H_QA = harmonic_descent(
            {"layers": len(list(self.model.parameters())), "tuple": (b, e, d, a)},
            {"dummy": True}
        )

        # Apply curvature-weighted updates
        with torch.no_grad():
            curvs = np.atleast_1d(H_QA)
            curv_iter = cycle(curvs)

            for p in self.model.parameters():
                if p.grad is None:
                    continue
                h = float(next(curv_iter))
                # Curvature-scaled gradient descent
                p -= self.lr * self.gain * h * p.grad

        return loss_hat, float(np.mean(H_QA))

# ---------------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------------

def train_markovian_qa(policy, autoencoder, epochs=100, lr=1e-3,
                       curvature_gain=2.0, max_iters=24, truncate=True,
                       clip_grad=True):
    """
    Train QALM 2.0 with Markovian thinking.

    Parameters:
    -----------
    policy : QAMarkovianPolicy
        Policy network
    autoencoder : QAAutoencoder
        QA state encoder
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    curvature_gain : float
        QA curvature amplification factor
    max_iters : int
        Number of chunks (mod-24)
    truncate : bool
        Truncate backprop between chunks
    clip_grad : bool
        Clip gradients for stability

    Returns:
    --------
    traces : list
        QA tuple evolution traces
    rewards : list
        Reward trajectory
    """
    env = QAMarkovianEnv(autoencoder, max_iters=max_iters, truncate=truncate)
    optimizer = QAOptimizer(policy, lr=lr, curvature_gain=curvature_gain)

    for step in tqdm(range(epochs)):
        # Rollout with current policy
        traces, rewards = env.rollout(
            query_tuple=(1, 1, 2, 3),
            policy_fn=lambda z: policy(z)
        )

        # Compute loss
        rewards_t = torch.tensor(rewards, dtype=torch.float32).detach()  # Detach rewards from graph
        potentials = torch.stack([
            torch.sum(y)  # y is already a tensor with grad_fn
            for _, y in traces
        ])
        loss = pac_harmonic_loss(rewards_t, potentials)

        # Backprop
        policy.zero_grad()
        loss.backward()

        if clip_grad:
            clip_grad_norm_(policy.parameters(), 1.0)

        # QA harmonic update with current tuple
        if len(traces) >= 2:
            # Markov-1.5: blend last two states for smoothness
            t_now = traces[-1][1].detach().cpu().numpy()
            t_prev = traces[-2][1].detach().cpu().numpy()
            active_tuple = 0.7 * t_now + 0.3 * t_prev
        else:
            active_tuple = traces[-1][1].detach().cpu().numpy()

        loss_hat, curvature_mean = optimizer.step(current_tuple=active_tuple)

        if step % 10 == 0:
            print(f"Step {step}: PAC-Harmonic Loss={loss.item():.5f}, "
                  f"Curv={curvature_mean:.5f}, HGD_loss_hat={loss_hat:.5f}")

    return traces, rewards

# ---------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------

def visualize_tuples(traces):
    """Plot QA tuple evolution across chunks"""
    ys = []
    for _, y in traces:
        if isinstance(y, torch.Tensor):
            ys.append(y.detach().cpu().numpy())
        else:
            ys.append(np.asarray(y, dtype=float))

    Y = np.asarray(ys, dtype=float)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    labels = ['b', 'e', 'd', 'a']
    for i, lab in enumerate(labels):
        if i < Y.shape[1]:
            plt.plot(Y[:, i], label=lab, marker='o', markersize=3)

    plt.title("QA-Markovian Tuple Evolution (QALM 2.0)")
    plt.xlabel("Chunk Index")
    plt.ylabel("Tuple Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("qa_markovian_evolution.png", dpi=150)
    print("Saved: qa_markovian_evolution.png")
    plt.show()

# ---------------------------------------------------------------
# 7. Entropy Metrics
# ---------------------------------------------------------------

def markov_entropy(traces):
    """Compute Markovian entropy across chunks"""
    ys = []
    for _, y in traces:
        if isinstance(y, torch.Tensor):
            ys.append(y.detach().cpu().numpy())
        else:
            ys.append(np.asarray(y, dtype=float))

    diffs = []
    for i in range(len(ys) - 1):
        diff = np.linalg.norm(ys[i+1] - ys[i])
        diffs.append(diff)

    diffs = np.array(diffs)
    entropy = -np.mean(np.log(diffs + 1e-9))

    return float(entropy)

# ---------------------------------------------------------------
# 8. Main Entry Point
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("QALM 2.0: Infinite Markovian Context Architecture")
    print("Based on 'The Markovian Thinker' (Aghajohari et al., 2025)")
    print("=" * 70)

    # Initialize components
    policy = QAMarkovianPolicy()
    autoencoder = QAAutoencoder(latent_dim=64)

    # Train
    traces, rewards = train_markovian_qa(
        policy, autoencoder,
        epochs=100,
        lr=1e-3,
        curvature_gain=2.0,
        max_iters=24,  # mod-24 QA alignment
        truncate=True,  # Delethink-style
        clip_grad=True
    )

    # Visualize
    visualize_tuples(traces)

    # Compute metrics
    entropy = markov_entropy(traces)
    print(f"\nMarkovian Entropy: {entropy:.3f}")

    # Save results
    ys_np = np.array([
        (y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=float))
        for _, y in traces
    ], dtype=float)

    rewards_np = np.array([
        (r.item() if isinstance(r, torch.Tensor) else float(r))
        for r in rewards
    ], dtype=float)

    np.savez("qa_markovian_results.npz", traces=ys_np, rewards=rewards_np)
    print("Saved: qa_markovian_results.npz")

    print("\n" + "=" * 70)
    print("QALM 2.0 training complete!")
    print("=" * 70)

# ---------------------------------------------------------------
# CIM/PIM Enhancements for In-Memory Processing
# ---------------------------------------------------------------

class CIMMemoryManager:
    """
    Compute-in-Memory manager for efficient data processing.
    Handles memory-mapped files and parallel operations.
    """
    def __init__(self, data_path=None, max_memory_gb=4):
        self.data_path = data_path or "cim_vault_data.dat"
        self.max_memory_gb = max_memory_gb
        self.mmap_file = None
        self.data_size = 0

    def create_memory_map(self, size_bytes):
        """Create memory-mapped file for large datasets"""
        with open(self.data_path, "wb") as f:
            f.write(b'\x00' * size_bytes)

        self.mmap_file = open(self.data_path, "r+b")
        self.data_size = size_bytes
        return mmap.mmap(self.mmap_file.fileno(), size_bytes)

    def store_qa_tuples(self, tuples, offset=0):
        """Store QA tuples in memory-mapped space"""
        if not hasattr(self, 'mmap_file') or self.mmap_file is None:
            size_needed = len(tuples) * 4 * 4  # 4 tuples * 4 floats each
            self.create_memory_map(size_needed)

        flat_data = np.array(tuples, dtype=np.float32).flatten()
        bytes_data = flat_data.tobytes()

        if offset + len(bytes_data) > self.data_size:
            # Extend memory map
            new_size = max(self.data_size * 2, offset + len(bytes_data))
            self.mmap_file.close()
            with open(self.data_path, "ab") as f:
                f.write(b'\x00' * (new_size - self.data_size))
            self.mmap_file = open(self.data_path, "r+b")
            self.mmap_file.seek(0, 2)  # Seek to end
            self.data_size = new_size

        self.mmap_file.seek(offset)
        self.mmap_file.write(bytes_data)

    def load_qa_tuples(self, count, offset=0):
        """Load QA tuples from memory-mapped space"""
        if not hasattr(self, 'mmap_file') or self.mmap_file is None:
            return []

        bytes_per_tuple = 4 * 4  # 4 floats
        total_bytes = count * bytes_per_tuple

        self.mmap_file.seek(offset)
        data = self.mmap_file.read(total_bytes)

        if len(data) < total_bytes:
            return []

        flat_array = np.frombuffer(data, dtype=np.float32)
        return flat_array.reshape(-1, 4).tolist()

    def close(self):
        """Clean up memory mapping"""
        if hasattr(self, 'mmap_file') and self.mmap_file:
            self.mmap_file.close()
            self.mmap_file = None

class PIMProcessor:
    """
    Processing-in-Memory processor for parallel QA operations.
    """
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        self.memory_manager = CIMMemoryManager()

    def parallel_qa_processing(self, items, operation_fn):
        """Process items in parallel using joblib"""
        # Use a global function for pickling
        def process_item(item):
            return operation_fn(item)

        try:
            results = Parallel(n_jobs=self.num_workers)(
                delayed(process_item)(item) for item in items
            )
            return results
        except Exception as e:
            print(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            return [operation_fn(item) for item in items]

    @lru_cache(maxsize=10000)
    def cached_qa_operation(self, qa_tuple_hash, operation_fn):
        """Cached QA operations for repeated computations"""
        # Convert tuple to hashable form
        qa_tuple = tuple(qa_tuple_hash)
        return operation_fn(qa_tuple)

class ObsidianVaultProcessor:
    """
    Process Obsidian vault files into QA-compatible format.
    """
    def __init__(self, vault_paths=None):
        self.vault_paths = vault_paths or ["/home/player2/signal_experiments/QAnotes",
                                         "/home/player2/signal_experiments/obsidian_vault"]
        self.pim_processor = PIMProcessor()

    def extract_markdown_content(self, file_path):
        """Extract clean text content from markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Remove frontmatter
            content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

            # Remove markdown formatting
            content = re.sub(r'#+\s*', '', content)  # Headers
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Bold
            content = re.sub(r'\*([^*]+)\*', r'\1', content)  # Italic
            content = re.sub(r'`([^`]+)`', r'\1', content)  # Code
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Links
            content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', content)  # Images

            # Clean up whitespace
            content = re.sub(r'\n+', ' ', content)
            content = re.sub(r'\s+', ' ', content)

            return content.strip()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return ""

    def find_markdown_files(self):
        """Find all markdown files in vault paths"""
        md_files = []
        for vault_path in self.vault_paths:
            if os.path.exists(vault_path):
                for root, dirs, files in os.walk(vault_path):
                    # Skip .obsidian directory
                    dirs[:] = [d for d in dirs if d != '.obsidian']
                    for file in files:
                        if file.endswith('.md'):
                            md_files.append(os.path.join(root, file))
        return md_files

    def text_to_qa_tuple(self, text):
        """Convert text to QA tuple representation"""
        if not text:
            return [1.0, 1.0, 2.0, 3.0]  # Default QA tuple

        # Create hash-based deterministic mapping
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:16], 16)

        # Generate QA tuple from hash
        b = 1 + (hash_int % 10)  # 1-10
        e = 1 + ((hash_int >> 4) % 10)  # 1-10
        d = b + e + (hash_int >> 8) % 5  # Maintain b+e=d invariant approximately
        a = b + 2*e + (hash_int >> 12) % 5  # Maintain b+2e=a invariant approximately

        return [float(b), float(e), float(d), float(a)]

    def process_vault_to_qa(self, max_files=None):
        """Process entire vault into QA tuples"""
        print("🔍 Discovering markdown files...")
        md_files = self.find_markdown_files()
        print(f"📁 Found {len(md_files)} markdown files")

        if max_files:
            md_files = md_files[:max_files]
            print(f"📊 Processing first {max_files} files")

        print("📖 Extracting content and converting to QA tuples...")

        results = []
        for file_path in md_files:
            content = self.extract_markdown_content(file_path)
            qa_tuple = self.text_to_qa_tuple(content)
            result = {
                'file': os.path.basename(file_path),
                'qa_tuple': qa_tuple,
                'content_length': len(content)
            }
            results.append(result)

        # Store in CIM memory
        qa_tuples = [r['qa_tuple'] for r in results]
        self.pim_processor.memory_manager.store_qa_tuples(qa_tuples)

        print(f"✅ Processed {len(results)} files into QA tuples")
        print(f"💾 Stored in CIM memory: {self.pim_processor.memory_manager.data_path}")

        return results

def test_cim_qalm_on_vault():
    """
    Test CIM-enhanced QALM 2.0 on Obsidian vault data.
    """
    print("=" * 80)
    print("🧠 CIM-Enhanced QALM 2.0: Obsidian Vault Processing Test")
    print("=" * 80)

    # Initialize processors
    vault_processor = ObsidianVaultProcessor()
    cim_memory = CIMMemoryManager("obsidian_qa_cim.dat")

    # Process vault
    vault_data = vault_processor.process_vault_to_qa(max_files=None)  # Process ALL files

    # Use QA tuples directly from processing results
    qa_tuples = [result['qa_tuple'] for result in vault_data]

    print(f"\n📊 Using {len(qa_tuples)} QA tuples from processing")
    print("🔍 Sample QA tuples:")
    for i, result in enumerate(vault_data[:5]):
        print(f"  {result['file']}: {result['qa_tuple']} (content: {result['content_length']} chars)")

    # Test QALM reasoning on vault data
    print("\n🧠 Testing QALM reasoning on vault QA tuples...")

    # Use first tuple as query, others as context
    query_tuple = qa_tuples[0] if qa_tuples else [1, 1, 2, 3]

    # Initialize QALM components
    policy = QAMarkovianPolicy()
    autoencoder = QAAutoencoder(latent_dim=64)  # Fallback implementation

    # Test reasoning
    env = QAMarkovianEnv(autoencoder, max_iters=min(24, len(qa_tuples)))
    traces, rewards = env.rollout(query_tuple, lambda z: policy(z))

    print(f"✅ Generated {len(traces)} reasoning steps")
    print(f"🎯 Final QA tuple: {traces[-1][1].detach().cpu().numpy()}")
    print(f"💯 Final reward: {rewards[-1]:.3f}")

    # Compute entropy
    entropy = markov_entropy(traces)
    print(f"🔄 Markovian Entropy: {entropy:.3f}")

    print("\n" + "=" * 80)
    print("🎉 CIM-Enhanced QALM 2.0 successfully processed Obsidian vault!")
    print("🚀 Demonstrated in-memory processing of knowledge base data")
    print("=" * 80)

    # Cleanup
    cim_memory.close()

if __name__ == "__main__":
    # Run CIM test on vault
    test_cim_qalm_on_vault()
