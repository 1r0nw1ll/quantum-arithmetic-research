"""
qa_core.py

Shared QA engine utilities for signal experiments and hybrid training.

Entry points that import this module:
- dynamic_coprocessor_test.py (hybrid MNIST training)
- Other runners may adopt it to remove duplicated QA_Engine/QASystem code.

This module intentionally keeps the engine lightweight and NumPy-only so it
can be imported from training scripts without adding extra dependencies.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


class QAEngine:
    """QA co-processor measuring geometric stress.

    Parameters
    ----------
    nodes : int
        Number of nodes in the engine graph.
    coupling : float
        Coupling factor for neighbor pull.
    modulus : int
        Modulus for arithmetic of tuple fields.
    """

    def __init__(self, nodes: int = 24, coupling: float = 0.1, modulus: int = 24):
        self.N = nodes
        self.M = modulus
        self.alpha = coupling
        self.B = np.random.rand(self.N) * self.M
        self.E = np.random.rand(self.N) * self.M
        self.W = np.zeros((self.N, self.N))

    def _calculate_tuples(self, B: np.ndarray, E: np.ndarray) -> np.ndarray:
        D = B + E
        A = B + 2 * E
        return np.vstack([B, E, D, A]).T

    def step(self, signal: float | np.ndarray = 0.0, injection_strength: float = 0.2, noise: float = 0.1) -> None:
        """Advance the engine by one step with an optional external signal.

        If ``signal`` is an array (e.g., weights), its mean is used as the scalar nudge.
        """
        if isinstance(signal, np.ndarray):
            signal = float(np.mean(signal))

        thetas_mod = np.floor(self._calculate_tuples(self.B, self.E)) % self.M
        self.W = (np.einsum('ij,kj->ik', thetas_mod, thetas_mod)) % self.M
        row_sum = self.W.sum(axis=1)
        nz = row_sum != 0
        self.W[nz] /= row_sum[nz][:, np.newaxis]

        self.B = (self.B + self.alpha * (self.W @ self.B - self.B) + injection_strength * signal + np.random.randn(self.N) * noise) % self.M
        self.E = (self.E + self.alpha * (self.W @ self.E - self.E) + np.random.randn(self.N) * noise) % self.M

    def get_geometric_stress(self, use_chromo: bool = False) -> float:
        """Return a scalar stress; lower means more coherent.

        If ``use_chromo`` is True, augment with simple chromogeometric terms
        derived from circular means of B and E angles.
        """
        stress = float(np.var(self.W))

        if use_chromo:
            angles_b = 2 * np.pi * self.B / self.M
            angles_e = 2 * np.pi * self.E / self.M
            u = float(np.mean(np.cos(angles_b)))
            v = float(np.mean(np.sin(angles_e)))
            Qb = u ** 2 + v ** 2
            Qr = u ** 2 - v ** 2
            Qg = 2 * u * v
            chromo_stress = float(np.var([Qb, Qr, Qg]))
            stress = 0.7 * stress + 0.3 * chromo_stress

        return stress


# Backwards-compatible alias for existing scripts
QA_Engine = QAEngine

__all__ = ["QAEngine", "QA_Engine"]


class QASystem:
    """Self-organizing QA system for signal experiments.

    Supports multiple update modes to preserve behavior across experiment scripts:
    - "original": neighbor pull computed from current tuples; signal added to b in the
      neighbor input path (scaled by modulus) but not directly added to delta.
    - "corrected": neighbor pull from current tuples; signal added directly to b update.
    - "final": resonance computed from a signal-perturbed proposed state, and signal
      also added directly to b update.

    Parameters
    ----------
    num_nodes : int
    modulus : int
    coupling : float
    noise_base : float
    noise_annealing : float
    signal_injection_strength : float
    signal_mode : str
        One of {"original", "corrected", "final"}
    """

    def __init__(
        self,
        num_nodes: int,
        modulus: int,
        coupling: float,
        noise_base: float,
        noise_annealing: float = 0.995,
        signal_injection_strength: float = 0.1,
        signal_mode: str = "final",
    ) -> None:
        self.num_nodes = num_nodes
        self.modulus = modulus
        self.coupling = coupling
        self.noise_base = noise_base
        self.noise_annealing = noise_annealing
        self.signal_injection_strength = signal_injection_strength
        self.signal_mode = signal_mode

        self.b = np.random.rand(num_nodes) * modulus
        self.e = np.random.rand(num_nodes) * modulus
        self.adjacency_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        self.history = {"loss": [], "e8_alignment": [], "hi": []}

    def _get_qa_tuples(self, b: np.ndarray | None = None, e: np.ndarray | None = None) -> np.ndarray:
        if b is None:
            b = self.b
        if e is None:
            e = self.e
        d = (b + e) % self.modulus
        a = (b + 2 * e) % self.modulus
        return np.stack([b, e, d, a], axis=1)

    def _calculate_loss(self, tuples: np.ndarray) -> float:
        lhs = (tuples[:, 3] ** 2) % self.modulus
        rhs = (tuples[:, 2] ** 2 + 2 * tuples[:, 2] * tuples[:, 1] + tuples[:, 1] ** 2) % self.modulus
        diff = np.abs(lhs - rhs)
        loss = np.minimum(diff, self.modulus - diff)
        return float(np.mean(loss ** 2))

    def _calculate_e8_alignment(self, tuples: np.ndarray) -> float:
        qa_vectors_8d = np.zeros((self.num_nodes, 8))
        qa_vectors_8d[:, :4] = tuples
        norms = np.linalg.norm(qa_vectors_8d, axis=1)
        norms[norms == 0] = 1
        normalized_vectors = qa_vectors_8d / norms[:, np.newaxis]
        ideal_root = np.array([1, 1, 2, 3, 0, 0, 0, 0])
        ideal_root_norm = ideal_root / np.linalg.norm(ideal_root)
        return float(np.mean(np.abs(normalized_vectors @ ideal_root_norm)))

    def _noise(self, t: int) -> np.ndarray:
        noise_mag = self.noise_base * (self.noise_annealing ** t)
        return (np.random.rand(2, self.num_nodes) - 0.5) * noise_mag

    def step(self, t: int, signal_sample: float = 0.0) -> None:
        mode = self.signal_mode

        if mode == "original":
            tuples = self._get_qa_tuples()
            resonance = np.einsum("ik,jk->ij", tuples, tuples) % self.modulus
            signal_force = signal_sample * self.signal_injection_strength * self.modulus
            b_with_signal = (self.b + signal_force) % self.modulus
            weighted_adj = self.adjacency_matrix * (resonance / self.modulus) ** 2
            sum_w = np.sum(weighted_adj, axis=1)
            sum_w[sum_w == 0] = 1
            neighbor_pull_b = (weighted_adj @ b_with_signal) / sum_w
            neighbor_pull_e = (weighted_adj @ self.e) / sum_w
            noise = self._noise(t)
            delta_b = self.coupling * (neighbor_pull_b - self.b)
            delta_e = self.coupling * (neighbor_pull_e - self.e)
            self.b = (self.b + delta_b + noise[0]) % self.modulus
            self.e = (self.e + delta_e + noise[1]) % self.modulus

        elif mode == "corrected":
            tuples = self._get_qa_tuples()
            resonance = np.einsum("ik,jk->ij", tuples, tuples) % self.modulus
            signal_force = signal_sample * self.signal_injection_strength
            weighted_adj = self.adjacency_matrix * (resonance / self.modulus) ** 2
            sum_w = np.sum(weighted_adj, axis=1)
            sum_w[sum_w == 0] = 1
            neighbor_pull_b = (weighted_adj @ self.b) / sum_w
            neighbor_pull_e = (weighted_adj @ self.e) / sum_w
            noise = self._noise(t)
            delta_b = self.coupling * (neighbor_pull_b - self.b) + signal_force
            delta_e = self.coupling * (neighbor_pull_e - self.e)
            self.b = (self.b + delta_b + noise[0]) % self.modulus
            self.e = (self.e + delta_e + noise[1]) % self.modulus

        elif mode == "final":
            signal_force = signal_sample * self.signal_injection_strength
            b_proposed = (self.b + signal_force) % self.modulus
            proposed_tuples = self._get_qa_tuples(b_proposed, self.e)
            resonance = np.einsum("ik,jk->ij", proposed_tuples, proposed_tuples) % self.modulus
            weighted_adj = self.adjacency_matrix * (resonance / self.modulus) ** 2
            sum_w = np.sum(weighted_adj, axis=1)
            sum_w[sum_w == 0] = 1
            neighbor_pull_b = (weighted_adj @ self.b) / sum_w
            neighbor_pull_e = (weighted_adj @ self.e) / sum_w
            noise = self._noise(t)
            delta_b = self.coupling * (neighbor_pull_b - self.b) + signal_force
            delta_e = self.coupling * (neighbor_pull_e - self.e)
            self.b = (self.b + delta_b + noise[0]) % self.modulus
            self.e = (self.e + delta_e + noise[1]) % self.modulus

        else:
            raise ValueError(f"Unknown signal_mode: {mode}")

    def run_simulation(self, timesteps: int, signal_data: np.ndarray) -> None:
        for t in tqdm(range(timesteps), desc="Simulating"):
            self.step(t, float(signal_data[t]))
            current_tuples = self._get_qa_tuples()
            loss = self._calculate_loss(current_tuples)
            e8_align = self._calculate_e8_alignment(current_tuples)
            hi = e8_align * np.exp(-0.1 * loss)
            self.history["loss"].append(loss)
            self.history["e8_alignment"].append(e8_align)
            self.history["hi"].append(hi)

__all__.append("QASystem")
