"""Classical simulated annealing — Python port of Kochenderfer 2026 Opt Algorithm 8.5.

QA_COMPLIANCE = "classical-baseline port — stochastic + continuous Metropolis acceptance; firewall-rejected as causal QA dynamics per Theorem NT. Off-QA baseline only; not a QA-causal algorithm."

Source: (Kochenderfer 2026) Algorithms for Optimization, 2nd ed., MIT Press, §8.4 + Algorithm 8.5.
Transcribed from QA-MEM verbatim excerpt at
docs/theory/kochenderfer_optimization_excerpts.md#opt-8-4-simulated-annealing
"""

from __future__ import annotations
import math
import random
from typing import Callable


def simulated_annealing(
    f: Callable,
    x_init,
    sample_perturbation: Callable,
    temperature_schedule: Callable[[int], float],
    k_max: int,
    seed: int = 0,
) -> object:
    """Simulated annealing per Opt Algorithm 8.5.

    f                      — objective function to minimize
    x_init                 — starting design point
    sample_perturbation()  — returns a random perturbation drawn from T
    temperature_schedule(k) — temperature at iteration k (e.g., γ^k * t_0)
    k_max                  — number of iterations
    seed                   — RNG seed for reproducibility

    Per Opt Algorithm 8.5: at each iteration sample x' = x + perturbation,
    accept with probability 1 if Δy ≤ 0 else exp(-Δy/t).
    """
    rng = random.Random(seed)
    x = x_init
    y = f(x)
    best_x, best_y = x, y
    for k in range(1, k_max + 1):
        x_prime = x + sample_perturbation()
        y_prime = f(x_prime)
        dy = y_prime - y
        if dy <= 0 or rng.random() < math.exp(-dy / max(temperature_schedule(k), 1e-12)):
            x, y = x_prime, y_prime
        if y_prime < best_y:
            best_x, best_y = x_prime, y_prime
    return best_x


if __name__ == "__main__":
    # Minimize f(x) = (x - 3)² + 0.5 sin(5x) over scalar x ∈ [-5, 5]
    # Has many local minima from the sine term; SA should find x ≈ 3
    def f(x):
        return (x - 3.0) ** 2 + 0.5 * math.sin(5 * x)

    rng = random.Random(42)

    def sample():
        return rng.gauss(0.0, 0.5)

    def schedule(k):
        return 10.0 * (0.95 ** k)  # exponential annealing

    x_opt = simulated_annealing(f, x_init=-5.0, sample_perturbation=sample,
                                 temperature_schedule=schedule, k_max=1000, seed=42)
    print(f"Found x* ≈ {x_opt:.4f}, f(x*) = {f(x_opt):.4f} (true minimum near x=3.0)")
