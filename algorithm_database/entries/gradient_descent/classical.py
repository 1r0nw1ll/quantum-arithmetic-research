"""Classical gradient descent — Python port of Kochenderfer 2026 Opt Algorithm 5.1.

QA_COMPLIANCE = "classical-baseline port — continuous-domain steepest descent; firewall-rejected as causal QA dynamics per Theorem NT. Vocabulary alignment with cert [191] tier-1 generator selection on QA-discrete side; no QA-causal counterpart."

Source: (Kochenderfer 2026) Algorithms for Optimization, 2nd ed., MIT Press, §5.1 + Algorithm 5.1.
Transcribed from QA-MEM verbatim excerpt at
docs/theory/kochenderfer_optimization_excerpts.md#opt-5-1-gradient-descent-steepest-direction
"""

from __future__ import annotations
from typing import Callable


def gradient_descent(
    grad_f: Callable,
    x_init,
    alpha: float,
    k_max: int,
):
    """Plain gradient descent with fixed step factor α per Opt Algorithm 5.1.

    grad_f — gradient function ∇f(x); returns array-like or scalar
    x_init — starting design point (must support `x - alpha * g`)
    alpha  — step factor (learning rate)
    k_max  — number of iterations

    Per Opt §5.1: descent direction is `d = -g/‖g‖`; this implementation
    uses unnormalized step `x ← x - α * g` (Kochenderfer's note: "Some
    implementations of gradient descent do not normalize the descent
    direction. In that case, the step factor α does not correspond to
    the step length.").
    """
    x = x_init
    for _ in range(k_max):
        g = grad_f(x)
        # Subtract elementwise. Works for scalars, lists, numpy arrays.
        if hasattr(g, "__iter__"):
            x = [xi - alpha * gi for xi, gi in zip(x, g)]
        else:
            x = x - alpha * g
    return x


if __name__ == "__main__":
    # Minimize f(x) = (x - 3)² over scalar x; gradient = 2(x - 3); minimum at x = 3
    def grad_f(x):
        return 2 * (x - 3.0)

    x_opt = gradient_descent(grad_f, x_init=-5.0, alpha=0.1, k_max=100)
    print(f"Found x* = {x_opt:.6f} (true minimum at x = 3.0)")
