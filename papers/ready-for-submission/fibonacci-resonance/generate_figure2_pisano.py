"""Figure 2 — Pisano periods and Fibonacci-set divisibility.

Supports the mechanism section (§4) of the Fibonacci resonance paper by
making the 0.677-vs-0.491 divisibility claim visible.

Run:
    python papers/in-progress/fibonacci-resonance/generate_figure2_pisano.py
"""
QA_COMPLIANCE = {
    "observer": "statistical_figure_projection",
    "state_alphabet": "integer Pisano periods pi(m) for m in {2..500}; Fibonacci membership flags; per-k divisibility counts",
}

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path


def pisano(m: int) -> int:
    if m == 1:
        return 1
    a, b = 0, 1
    for i in range(1, 6 * m + 2):
        a, b = b, (a + b) % m
        if a == 0 and b == 1:
            return i
    raise RuntimeError(f"pisano({m}) did not converge")


def fibs_up_to(n: int) -> set[int]:
    out, a, b = set(), 1, 1
    while a <= n:
        out.add(a)
        a, b = b, a + b
    return out


def main():
    out_path = Path(__file__).resolve().parent / "figures" / "figure2_pisano_divisibility.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    M_MAX = 500
    pi = {m: pisano(m) for m in range(2, M_MAX + 1)}
    fib_all = fibs_up_to(M_MAX)

    mpl.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.6, 1.0]})

    # ------------- Panel (a): pi(m) for m = 2..60, Fib m highlighted -------------
    ax = axes[0]
    M_SHOW = 60
    ms = list(range(2, M_SHOW + 1))
    vals = [pi[m] for m in ms]
    is_fib = [m in fib_all for m in ms]
    colors = ["#d4a017" if f else "#6a7280" for f in is_fib]
    ax.bar(ms, vals, color=colors, edgecolor="none")
    ax.set_xlabel(r"modulus $m$")
    ax.set_ylabel(r"Pisano period $\pi(m)$")
    ax.set_title(r"(a) $\pi(m)$ for $m = 2..60$")
    ax.set_xlim(1.3, M_SHOW + 0.7)

    for m in ms:
        if m in fib_all:
            ax.annotate(str(m), (m, pi[m]), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#8a6b10")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d4a017", label=r"$m \in F$ (Fibonacci)"),
        Patch(facecolor="#6a7280", label=r"$m \notin F$"),
    ], loc="upper left", frameon=False, fontsize=8)

    # ------------- Panel (b): per-k divisibility rate over m = 2..500 -------------
    ax = axes[1]
    ks = list(range(2, 10))
    fib_set = {2, 3, 5, 8}
    rates = [sum(1 for m in pi if pi[m] % k == 0) / len(pi) for k in ks]
    bar_colors = ["#d4a017" if k in fib_set else "#6a7280" for k in ks]
    bars = ax.bar(ks, rates, color=bar_colors, edgecolor="none")

    for k, r in zip(ks, rates):
        ax.text(k, r + 0.012, f"{r:.2f}", ha="center", fontsize=7)

    fib_avg = np.mean([rates[ks.index(k)] for k in fib_set])
    nfib_avg = np.mean([rates[ks.index(k)] for k in {4, 6, 7, 9}])
    ax.axhline(fib_avg, color="#d4a017", linestyle="--", linewidth=1,
               label=f"Fib avg = {fib_avg:.3f}")
    ax.axhline(nfib_avg, color="#6a7280", linestyle="--", linewidth=1,
               label=f"non-Fib avg = {nfib_avg:.3f}")
    ax.set_xlabel(r"divisor $k$")
    ax.set_ylabel(r"fraction of $m \in [2,500]$ with $k \mid \pi(m)$")
    ax.set_title("(b) per-$k$ divisibility rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    for spine in ("top", "right"):
        for a in axes:
            a.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"wrote {out_path}")
    print(f"      {out_path.with_suffix('.png')}")
    print(f"Fib avg divisibility: {fib_avg:.3f}")
    print(f"Non-Fib avg divisibility: {nfib_avg:.3f}")


if __name__ == "__main__":
    main()
