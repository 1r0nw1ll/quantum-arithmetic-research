#!/usr/bin/env python3
"""
QA Post-Calculus Engine
=======================

Recreates the vault-side "post-calculus prototype" in executable form.

The engine treats Quantum Arithmetic (QA) tuples `(b, e, d, a)` as a discrete
calculus primitive:

    d = b + e
    a = b + 2e
    Δ = a - d = e      (discrete derivative identity)

By feeding the harmonic increment `e` with samples of a target function, the
tuple evolution mirrors an Euler / Riemann integrator where the integral is a
structured sum of the `e` sequence.

Features
--------
1. Verifies the derivative identity Δ ≡ e across an arbitrary number of steps.
2. Performs QA-style integration for analytic test functions (sin, exp).
3. Supports rational quantisation of the harmonic increment (denominators 24,
   72, 144) to investigate stability vs. exact arithmetic.
4. Emits summary statistics and (optionally) plots to reproduce the vault
   narratives ("Δₙ overlap", convergence, quantisation drift).

Usage
-----
Basic derivative / integral check:

    python qa_post_calculus_engine.py --steps 64

Compare QA integral vs analytic truth on sin(x):

    python qa_post_calculus_engine.py --function sin --domain 0 6.283 --steps 128 --plot

Quantised harmonic increments (denominator = 144) on exp(x):

    python qa_post_calculus_engine.py --function exp --domain 0 2 --steps 96 --quantise 144

Outputs
-------
* Console summary with mean/std error and the Δ ≡ e check.
* Optional PNG plots saved alongside the script.

Reference documents:
* QA_UNIFIED_FRAMEWORK_SUMMARY.md
* HYPERSPECTRAL_RESEARCH_SUMMARY.md (shared terminology)
* vault chunk 837446ed178d011ec3d58c39f22a9adedc4b91b8e51dd3fa5221e8e8a10cfe8a
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

TAU = 2 * math.pi


# ============================================================================
# Core QA Tuple Mechanics
# ============================================================================

@dataclass
class QATupleState:
    """Container for a QA tuple at a single step."""

    b: float  # base value (e.g. integral so far)
    e: float  # harmonic increment
    d: float  # b + e
    a: float  # b + 2e

    def derivative_delta(self) -> float:
        """Return the discrete derivative Δ = a - d (should equal e)."""
        return self.a - self.d


def qa_step(previous_b: float, increment: float) -> QATupleState:
    """
    Produce the next QA tuple given the previous base value and new increment.

    Parameters
    ----------
    previous_b:
        Accumulated integral value `b_n`.
    increment:
        Harmonic increment `e_n` (e.g. f(x_n) * h).

    Returns
    -------
    QATupleState
        The populated tuple (b_{n+1}, e_n, d_n, a_n).
    """
    b_next = previous_b
    e_val = increment
    d_val = b_next + e_val
    a_val = b_next + 2.0 * e_val
    return QATupleState(b=b_next, e=e_val, d=d_val, a=a_val)


def rationalise(value: float, denominator: Optional[int]) -> float:
    """
    Quantise a float to multiples of 1/denominator, emulating QAℚ discussions.
    """
    if not denominator:
        return value
    return round(value * denominator) / denominator


# ============================================================================
# Function Catalogue
# ============================================================================

def reference_function(name: str) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Return (f, F) where f(x) is the integrand and F(x) its analytic integral.

    Supported functions:
    * sin : f(x) = sin(x), F(x) = 1 - cos(x) (shifted to zero at 0)
    * exp : f(x) = exp(x), F(x) = exp(x) - 1 (shifted to zero at 0)
    """
    name = name.lower()
    if name == "sin":
        return (
            math.sin,
            lambda x: 1.0 - math.cos(x),
        )
    if name == "exp":
        return (
            math.exp,
            lambda x: math.exp(x) - 1.0,
        )
    raise ValueError(f"Unsupported function '{name}'. Choose from 'sin', 'exp'.")


# ============================================================================
# QA Integration Engine
# ============================================================================

@dataclass
class QAIntegrationResult:
    x: np.ndarray
    qa_integral: np.ndarray
    qa_quantised: Optional[np.ndarray]
    analytic: np.ndarray
    tuples: np.ndarray  # structured array storing (b, e, d, a)


def qa_integrate(
    f: Callable[[float], float],
    F: Callable[[float], float],
    domain: Tuple[float, float],
    steps: int,
    quantise_den: Optional[int] = None,
) -> QAIntegrationResult:
    """
    Integrate function f over domain using QA tuple recursion.

    The update follows a forward Euler scheme:
        e_n     = f(x_n) * h
        b_{n+1} = b_n + e_n

    The tuple stored per step captures (b_n, e_n, d_n, a_n).
    """
    start, end = domain
    if steps <= 0:
        raise ValueError("steps must be positive")

    x = np.linspace(start, end, steps + 1)
    h = (end - start) / steps

    qa_values = np.zeros_like(x)
    qa_quantised = np.zeros_like(x) if quantise_den else None
    tuples = np.zeros((steps + 1, 4))

    b = 0.0
    q_b = 0.0

    for i, xi in enumerate(x):
        if i == 0:
            # Seed tuple at initial point (no increment yet).
            state = qa_step(b, 0.0)
            tuples[i] = (state.b, state.e, state.d, state.a)
            continue

        increment = f(xi) * h
        quantised_increment = rationalise(increment, quantise_den)

        b += increment
        state = qa_step(b - increment, increment)
        tuples[i] = (state.b, state.e, state.d, state.a)
        qa_values[i] = b

        if qa_quantised is not None:
            q_b += quantised_increment
            qa_quantised[i] = q_b

    analytic = np.array([F(xi) - F(start) for xi in x])

    return QAIntegrationResult(
        x=x,
        qa_integral=qa_values,
        qa_quantised=qa_quantised,
        analytic=analytic,
        tuples=tuples,
    )


# ============================================================================
# Reporting / Plotting
# ============================================================================

def summarise(result: QAIntegrationResult, name: str, quantise_den: Optional[int]) -> Dict[str, float]:
    """
    Compute summary statistics used both for console reporting and tests.
    """
    qa_error = result.qa_integral - result.analytic
    summary = {
        "function": name,
        "qa_mae": float(np.mean(np.abs(qa_error))),
        "qa_rmse": float(np.sqrt(np.mean(qa_error**2))),
        "qa_max_err": float(np.max(np.abs(qa_error))),
    }
    if result.qa_quantised is not None:
        q_error = result.qa_quantised - result.analytic
        summary.update(
            {
                "qa_quant_den": quantise_den or 0,
                "qa_quant_mae": float(np.mean(np.abs(q_error))),
                "qa_quant_rmse": float(np.sqrt(np.mean(q_error**2))),
                "qa_quant_max_err": float(np.max(np.abs(q_error))),
            }
        )
    return summary


def verify_derivative_identity(result: QAIntegrationResult) -> float:
    """
    Return the maximum absolute deviation between Δ and e across all tuples.
    """
    deltas = result.tuples[:, 3] - result.tuples[:, 2]  # a - d
    deviations = np.abs(deltas - result.tuples[:, 1])
    return float(np.max(deviations))


def plot_results(result: QAIntegrationResult, name: str, quantise_den: Optional[int]) -> None:
    """
    Save QA vs analytic integral comparison plots.
    """
    out_dir = Path(".")

    plt.figure(figsize=(10, 6))
    plt.plot(result.x, result.analytic, label="Analytic integral", linewidth=2)
    plt.plot(result.x, result.qa_integral, label="QA integral", linestyle="--")
    if result.qa_quantised is not None:
        plt.plot(result.x, result.qa_quantised, label=f"QA quantised (1/{quantise_den})", linestyle=":")
    plt.title(f"QA Post-Calculus Integration – {name}")
    plt.xlabel("x")
    plt.ylabel("Integral value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = out_dir / f"qa_post_calculus_{name}_integral.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {filename}")

    plt.figure(figsize=(10, 4))
    qa_error = result.qa_integral - result.analytic
    plt.plot(result.x, qa_error, label="QA error", linewidth=1.5)
    if result.qa_quantised is not None:
        plt.plot(result.x, result.qa_quantised - result.analytic, label="QA quantised error", linestyle="--")
    plt.title(f"QA Error Profile – {name}")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = out_dir / f"qa_post_calculus_{name}_error.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved plot: {filename}")


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QA post-calculus experiments inspired by the vault prototype.",
    )
    parser.add_argument(
        "--function",
        default="sin",
        choices=["sin", "exp"],
        help="Target function to integrate (default: sin).",
    )
    parser.add_argument(
        "--domain",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=(0.0, TAU),
        help="Integration domain [start end] (default: 0 2π).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of integration steps (default: 64).",
    )
    parser.add_argument(
        "--quantise",
        type=int,
        default=None,
        help="Optional denominator for harmonic increment quantisation (e.g., 24, 72, 144).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save comparison plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    f, F = reference_function(args.function)
    result = qa_integrate(
        f=f,
        F=F,
        domain=tuple(args.domain),
        steps=args.steps,
        quantise_den=args.quantise,
    )

    summary = summarise(result, args.function, args.quantise)
    delta_error = verify_derivative_identity(result)

    print("=" * 72)
    print(f"QA Post-Calculus Report – {args.function}")
    print("=" * 72)
    print(f"Domain:            [{args.domain[0]}, {args.domain[1]}]")
    print(f"Steps:             {args.steps}")
    if args.quantise:
        print(f"Quantisation:      1/{args.quantise}")
    else:
        print("Quantisation:      none (floating point)")
    print("-" * 72)
    print(f"QA mean abs error: {summary['qa_mae']:.6e}")
    print(f"QA RMSE:           {summary['qa_rmse']:.6e}")
    print(f"QA max abs error:  {summary['qa_max_err']:.6e}")
    if args.quantise:
        print(f"QAᴽ mean abs error: {summary['qa_quant_mae']:.6e}")
        print(f"QAᴽ RMSE:           {summary['qa_quant_rmse']:.6e}")
        print(f"QAᴽ max abs error:  {summary['qa_quant_max_err']:.6e}")
    print(f"Δ ≡ e max deviation: {delta_error:.6e}")
    print("=" * 72)

    if args.plot:
        plot_results(result, args.function, args.quantise)


if __name__ == "__main__":
    main()

