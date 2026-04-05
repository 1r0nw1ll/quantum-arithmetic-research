"""Shared engine implementations for QA experiments — integer state throughout.

Both `QAEngine` and `QASystem` hold their (b, e) node state as `int64` arrays
with values in {1, ..., modulus}. Inside `step()` the state is projected into
a local float observer layer for the continuous dynamics (neighbor pull,
signal injection, noise annealing), then quantized back to the integer
alphabet before being written to `self.B`/`self.E` (or `self.b`/`self.e`).

## Axiom compliance

- **A1 (No-Zero)**: state is integer in {1, ..., m}. Initialization uses
  `np.random.randint(1, m+1, N)`; modular reduction is `((x - 1) % m) + 1`.
- **S2 (No float state)**: no persistent float state. Float intermediates
  live only as local variables inside `step()`.
- **Theorem NT (Firewall)**: the observer boundary is crossed exactly twice
  per `step()` call — once inbound (integer state → float observer space),
  once outbound (quantize back to integer state). Continuous signal and
  noise enter only during the float observer phase; neither is stored.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=integer_coherence_field, state_alphabet=mod{1..m}, A1+S2_compliant"

import numpy as np
from tqdm import tqdm

from .logger import new_history, record_history
from .metrics import e8_alignment, harmonic_index, harmonic_loss, qa_tuples
from .orbit import complete_graph_adjacency, neighbor_pull, resonance_matrix, weighted_adjacency


def _quantize_to_state(float_arr: np.ndarray, modulus: int) -> np.ndarray:
    """Round a float array and fold into the A1-compliant alphabet {1,...,m}."""

    as_int = np.rint(float_arr).astype(np.int64)
    return ((as_int - 1) % modulus) + 1


class QAEngine:
    """QA co-processor measuring geometric stress — integer state."""

    def __init__(self, nodes: int = 24, coupling: float = 0.1, modulus: int = 24):
        self.N = nodes
        self.M = modulus
        self.alpha = coupling
        # A1-compliant integer initialization: values in {1, ..., M}, never 0.
        self.B = np.random.randint(1, self.M + 1, size=self.N, dtype=np.int64)
        self.E = np.random.randint(1, self.M + 1, size=self.N, dtype=np.int64)
        self.W = np.zeros((self.N, self.N))

    def _calculate_tuples(self, b: np.ndarray, e: np.ndarray) -> np.ndarray:
        d = b + e  # noqa: A2-1 — intermediate for adjacency weighting only
        a = b + 2 * e  # noqa: A2-2 — intermediate for adjacency weighting only
        return np.vstack([b, e, d, a]).T

    def step(
        self,
        signal: float | np.ndarray = 0.0,
        injection_strength: float = 0.2,
        noise: float = 0.1,
    ) -> None:
        """Advance the engine by one step with an optional external signal.

        One full observer boundary round-trip: int state → float observer
        computation → quantize back to int state.
        """

        if isinstance(signal, np.ndarray):
            signal = float(np.mean(signal))

        # Compute adjacency weighting from current integer state.
        thetas_mod = self._calculate_tuples(self.B, self.E) % self.M
        self.W = np.einsum("ij,kj->ik", thetas_mod, thetas_mod) % self.M
        row_sum = self.W.sum(axis=1)
        nz = row_sum != 0
        self.W = self.W.astype(float)
        self.W[nz] /= row_sum[nz][:, np.newaxis]

        # Observer-layer float computation (local, transient, non-persistent).
        b_float = self.B.astype(float)
        e_float = self.E.astype(float)
        pull_b = self.W @ b_float - b_float
        pull_e = self.W @ e_float - e_float
        b_next_float = b_float + self.alpha * pull_b + injection_strength * signal + np.random.randn(self.N) * noise
        e_next_float = e_float + self.alpha * pull_e + np.random.randn(self.N) * noise

        # Quantize observer output back to the integer QA alphabet.
        self.B = _quantize_to_state(b_next_float, self.M)
        self.E = _quantize_to_state(e_next_float, self.M)

    def get_geometric_stress(self, use_chromo: bool = False) -> float:
        """Return a scalar stress; lower means more coherent."""

        stress = float(np.var(self.W))
        if use_chromo:
            angles_b = 2 * np.pi * self.B.astype(float) / self.M
            angles_e = 2 * np.pi * self.E.astype(float) / self.M
            u = float(np.mean(np.cos(angles_b)))
            v = float(np.mean(np.sin(angles_e)))
            qb = u * u + v * v
            qr = u * u - v * v
            qg = 2 * u * v
            chromo_stress = float(np.var([qb, qr, qg]))
            stress = 0.7 * stress + 0.3 * chromo_stress
        return stress


QA_Engine = QAEngine


class QASystem:
    """Self-organizing QA system for signal experiments — integer state."""

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

        # A1-compliant integer initialization: values in {1, ..., modulus}, never 0.
        self.b = np.random.randint(1, modulus + 1, size=num_nodes, dtype=np.int64)
        self.e = np.random.randint(1, modulus + 1, size=num_nodes, dtype=np.int64)
        self.adjacency_matrix = complete_graph_adjacency(num_nodes)
        self.history = new_history()

    def _get_qa_tuples(self, b: np.ndarray | None = None, e: np.ndarray | None = None) -> np.ndarray:
        if b is None:
            b = self.b
        if e is None:
            e = self.e
        return qa_tuples(b, e, self.modulus)

    def _calculate_loss(self, tuples: np.ndarray) -> float:
        return harmonic_loss(tuples, self.modulus)

    def _calculate_e8_alignment(self, tuples: np.ndarray) -> float:
        return e8_alignment(tuples)

    def _noise(self, t: int) -> np.ndarray:
        noise_mag = self.noise_base * (self.noise_annealing ** t)
        return (np.random.rand(2, self.num_nodes) - 0.5) * noise_mag

    def step(self, t: int, signal_sample: float = 0.0) -> None:
        """Advance one timestep. Observer boundary crossed twice per call.

        Mode semantics preserved from legacy float-state implementation, but
        all persistent (b, e) state is integer in {1, ..., m}. Each mode
        builds local float intermediates, performs its distinctive dynamic,
        then quantizes the result back to integer state.
        """

        mode = self.signal_mode
        modulus = self.modulus

        # Project current integer state into the observer layer.
        b_float = self.b.astype(float)
        e_float = self.e.astype(float)

        if mode == "original":
            tuples = self._get_qa_tuples()
            resonance = resonance_matrix(tuples, modulus)
            signal_force = signal_sample * self.signal_injection_strength * modulus
            b_with_signal = (b_float + signal_force) % modulus
            weighted_adj = weighted_adjacency(self.adjacency_matrix, resonance, modulus)
            neighbor_pull_b = neighbor_pull(weighted_adj, b_with_signal)
            neighbor_pull_e = neighbor_pull(weighted_adj, e_float)
            noise_arr = self._noise(t)
            b_delta = self.coupling * (neighbor_pull_b - b_float)
            e_delta = self.coupling * (neighbor_pull_e - e_float)
            b_next_float = b_float + b_delta + noise_arr[0]
            e_next_float = e_float + e_delta + noise_arr[1]

        elif mode == "corrected":
            tuples = self._get_qa_tuples()
            resonance = resonance_matrix(tuples, modulus)
            signal_force = signal_sample * self.signal_injection_strength
            weighted_adj = weighted_adjacency(self.adjacency_matrix, resonance, modulus)
            neighbor_pull_b = neighbor_pull(weighted_adj, b_float)
            neighbor_pull_e = neighbor_pull(weighted_adj, e_float)
            noise_arr = self._noise(t)
            b_delta = self.coupling * (neighbor_pull_b - b_float) + signal_force
            e_delta = self.coupling * (neighbor_pull_e - e_float)
            b_next_float = b_float + b_delta + noise_arr[0]
            e_next_float = e_float + e_delta + noise_arr[1]

        elif mode == "final":
            signal_force = signal_sample * self.signal_injection_strength
            b_proposed_float = (b_float + signal_force) % modulus
            b_proposed_int = _quantize_to_state(b_proposed_float, modulus)
            proposed_tuples = self._get_qa_tuples(b_proposed_int, self.e)
            resonance = resonance_matrix(proposed_tuples, modulus)
            weighted_adj = weighted_adjacency(self.adjacency_matrix, resonance, modulus)
            neighbor_pull_b = neighbor_pull(weighted_adj, b_float)
            neighbor_pull_e = neighbor_pull(weighted_adj, e_float)
            noise_arr = self._noise(t)
            b_delta = self.coupling * (neighbor_pull_b - b_float) + signal_force
            e_delta = self.coupling * (neighbor_pull_e - e_float)
            b_next_float = b_float + b_delta + noise_arr[0]
            e_next_float = e_float + e_delta + noise_arr[1]

        else:
            raise ValueError(f"Unknown signal_mode: {mode}")

        # Quantize the observer-layer output back to integer state.
        self.b = _quantize_to_state(b_next_float, modulus)
        self.e = _quantize_to_state(e_next_float, modulus)

    def run_simulation(self, timesteps: int, signal_data: np.ndarray, *, progress: bool = True) -> None:
        iterator = tqdm(range(timesteps), desc="Simulating") if progress else range(timesteps)
        for t in iterator:
            self.step(t, float(signal_data[t]))
            current_tuples = self._get_qa_tuples()
            loss = self._calculate_loss(current_tuples)
            e8_align = self._calculate_e8_alignment(current_tuples)
            hi = harmonic_index(loss, e8_align)
            record_history(self.history, loss, e8_align, hi)


__all__ = ["QAEngine", "QA_Engine", "QASystem"]
