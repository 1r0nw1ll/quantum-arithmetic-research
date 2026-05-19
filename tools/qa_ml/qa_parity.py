"""QA-ML parity-benchmark harness.

Standardizes the schema and rendering convention for "QA-native primitive
matches continuous baseline" parity tests. Use this module when adding a
new GA→QA primitive (rotor, sandwich, motor, equivariant layer, …) so the
parity test produces comparable plots and JSON across the library.

The harness is intentionally small. It does NOT prescribe a task structure
— each primitive defines its own task (SE(3) compose, point-cloud register,
orbit classify, …). The harness only standardizes:

  1. The trial loop: run `n_trials` random tasks per (m, condition) cell.
  2. The result schema: `{cells: {condition_key: {n, m, *_median, *_p25, *_p75}}}`.
  3. The renderers: heatmap + scaling-plot, both with consistent styling.

Usage:
    from tools.qa_ml.qa_parity import run_parity_sweep, render_heatmap, render_scaling

    def task_factory(rng): ...                    # → (input, ground_truth)
    def continuous_method(input): ...             # → continuous prediction
    def qa_method_factory(m): ...                 # → (input) → quantized prediction
    def error_metrics(pred, gt): ...              # → {"rot": float, "trans": float}

    results = run_parity_sweep(
        task_factory, continuous_method, qa_method_factory, error_metrics,
        moduli=[24, 72, 144], conditions={"easy": 0.02, "hard": 0.1},
        n_trials=30, seed=0,
    )

QA_COMPLIANCE = "qa_ml_parity_harness — observer-projection wrapper; integer state lives inside qa_method"
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np


def run_parity_sweep(
    task_factory: Callable[[np.random.Generator, Any], Any],
    continuous_method: Callable[[Any], Any],
    qa_method_factory: Callable[[int], Callable[[Any], Any]],
    error_metrics: Callable[[Any, Any], dict[str, float]],
    moduli: Iterable[int],
    conditions: Mapping[str, Any] | None = None,
    n_trials: int = 30,
    seed: int = 0,
    include_continuous_baseline: bool = True,
) -> dict:
    """Run a parity sweep across (modulus, condition) cells.

    `task_factory(rng, condition_value)` returns either:
      - a 2-tuple (task_input, ground_truth), or
      - a single value used as both input and ground_truth.

    `error_metrics(prediction, ground_truth)` returns a flat dict of metric
    names → float. The same keys are aggregated across trials.

    Returns: {
      "moduli": [...], "conditions": {...}, "metric_names": [...],
      "cells": { "<cond_key>_m<m>": { "n": n_trials, "m": m, "condition": cond_key,
                                       "<metric>_median": ..., "<metric>_p25": ...,
                                       "<metric>_p75": ... } },
      "continuous_baseline": { "<cond_key>": {<metric>_median: ...} },  # if requested
    }"""
    conditions = conditions or {"default": None}
    moduli = list(moduli)
    rng = np.random.default_rng(seed)
    cells: dict[str, dict[str, Any]] = {}
    baseline: dict[str, dict[str, float]] = {}
    metric_names: list[str] = []

    for cond_key, cond_val in conditions.items():
        if include_continuous_baseline:
            bl_acc: dict[str, list[float]] = {}
            for _ in range(n_trials):
                task = task_factory(rng, cond_val)
                tin, gt = task if isinstance(task, tuple) and len(task) == 2 else (task, task)
                pred = continuous_method(tin)
                for k, v in error_metrics(pred, gt).items():
                    bl_acc.setdefault(k, []).append(float(v))
            baseline[cond_key] = {
                **{f"{k}_median": float(np.median(vs)) for k, vs in bl_acc.items()},
                **{f"{k}_p25": float(np.percentile(vs, 25)) for k, vs in bl_acc.items()},
                **{f"{k}_p75": float(np.percentile(vs, 75)) for k, vs in bl_acc.items()},
            }
            metric_names = list(bl_acc.keys())

        for m in moduli:
            qa_method = qa_method_factory(m)
            acc: dict[str, list[float]] = {}
            for _ in range(n_trials):
                task = task_factory(rng, cond_val)
                tin, gt = task if isinstance(task, tuple) and len(task) == 2 else (task, task)
                pred = qa_method(tin)
                for k, v in error_metrics(pred, gt).items():
                    acc.setdefault(k, []).append(float(v))
            cell_key = f"{cond_key}_m{m}"
            cells[cell_key] = {
                "n": n_trials,
                "m": m,
                "condition": cond_key,
                **{f"{k}_median": float(np.median(vs)) for k, vs in acc.items()},
                **{f"{k}_p25": float(np.percentile(vs, 25)) for k, vs in acc.items()},
                **{f"{k}_p75": float(np.percentile(vs, 75)) for k, vs in acc.items()},
            }
            if not metric_names:
                metric_names = list(acc.keys())

    return {
        "moduli": moduli,
        "conditions": list(conditions.keys()),
        "metric_names": metric_names,
        "cells": cells,
        "continuous_baseline": baseline if include_continuous_baseline else None,
    }


def render_heatmap(results: dict, metric: str, ax=None, fmt: str = "{:.2f}",
                   title: str | None = None):
    """Heatmap: condition (rows) × modulus (cols), one metric."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.0 + 1.2 * len(results["moduli"]),
                                        1.5 + 0.6 * len(results["conditions"])))
    conditions = results["conditions"]
    moduli = results["moduli"]
    grid = np.zeros((len(conditions), len(moduli)))
    for i, c in enumerate(conditions):
        for j, m in enumerate(moduli):
            grid[i, j] = results["cells"][f"{c}_m{m}"][f"{metric}_median"]
    im = ax.imshow(grid, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(moduli))); ax.set_xticklabels([f"m={m}" for m in moduli])
    ax.set_yticks(range(len(conditions))); ax.set_yticklabels(conditions)
    ax.set_xlabel("QA modulus")
    ax.set_title(title or f"{metric} (median)")
    vmax = grid.max() if grid.max() > 0 else 1.0
    for i in range(len(conditions)):
        for j in range(len(moduli)):
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                    color="white" if grid[i, j] > vmax * 0.5 else "black", fontsize=9)
    return ax


def render_scaling(results: dict, metric: str, x_key: str = "condition",
                   ax=None, title: str | None = None, logx: bool = False,
                   logy: bool = True):
    """Scaling plot: x = condition values (parsed from condition labels if numeric),
    one line per modulus. Use this when conditions encode a quantitative sweep
    (chain length, noise σ, …)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    conditions = results["conditions"]
    moduli = results["moduli"]
    # Try parse condition labels as numbers (e.g. "N5" → 5, "0.02" → 0.02)
    def _parse(c):
        import re
        m = re.search(r"-?\d+\.?\d*", str(c))
        return float(m.group()) if m else None
    xs = [_parse(c) for c in conditions]
    if any(v is None for v in xs):
        xs = list(range(len(conditions)))
        ax.set_xticks(xs); ax.set_xticklabels(conditions)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(moduli)))
    for j, m in enumerate(moduli):
        line = [results["cells"][f"{c}_m{m}"][f"{metric}_median"] for c in conditions]
        ax.plot(xs, line, "o-", color=colors[j], label=f"m={m}")
    if results.get("continuous_baseline"):
        bl_line = [results["continuous_baseline"][c][f"{metric}_median"] for c in conditions]
        ax.plot(xs, bl_line, "k--", linewidth=1.6, label="continuous")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(f"{metric} (median)")
    ax.set_title(title or f"{metric} vs {x_key}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    return ax


def parity_verdict(results: dict, metric: str, threshold: float,
                   condition: str | None = None) -> dict[int, bool]:
    """Returns {m: bool} indicating whether the median metric at this condition
    is below the parity threshold. Uses the first condition if none specified."""
    cond = condition or results["conditions"][0]
    return {
        m: results["cells"][f"{cond}_m{m}"][f"{metric}_median"] <= threshold
        for m in results["moduli"]
    }
