"""Figure 3 — Primitive Pythagorean triples in Euclid generator (m,n) space.

Supports §6.1 of the Fibonacci resonance paper. The 16 primitive triples with
hypotenuse c ≤ 100 are plotted in (m,n). The three with all-Fibonacci QA
quantum numbers (3:4:5, 5:12:13, 39:80:89) are highlighted gold and connected
by the Fibonacci ladder through (m,n) ∈ {(F_k, F_{k-1})}.

Run:
    python papers/in-progress/fibonacci-resonance/generate_figure3_pythagorean.py
"""
QA_COMPLIANCE = {
    "observer": "statistical_figure_projection",
    "state_alphabet": "integer Euclid generators (m,n) with m > n > 0, gcd=1, m-n odd; triple (a,b,c) = (m^2-n^2, 2mn, m^2+n^2); Fibonacci membership via integer test",
}

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from math import gcd
from pathlib import Path


def fibs_up_to(n: int) -> set[int]:
    out, a, b = set(), 1, 1
    while a <= n:
        out.add(a)
        a, b = b, a + b
    return out


def enumerate_primitive_triples(c_max: int):
    triples = []
    m = 2
    while m * m <= c_max:
        for n in range(1, m):
            if (m - n) % 2 == 0:
                continue
            if gcd(m, n) != 1:
                continue
            a, b, c = m * m - n * n, 2 * m * n, m * m + n * n
            if c > c_max:
                continue
            if a > b:
                a, b = b, a
            triples.append({"m": m, "n": n, "a": a, "b": b, "c": c,
                            "qn": (m - n, n, m, m + n)})
        m += 1
    return triples


def main():
    out_path = Path(__file__).resolve().parent / "figures" / "figure3_pythagorean_fib.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c_max = 100
    triples = enumerate_primitive_triples(c_max)
    fib = fibs_up_to(c_max * 2)
    for t in triples:
        t["all_fib"] = all(x in fib for x in t["qn"])

    n_fib = sum(1 for t in triples if t["all_fib"])
    n_tot = len(triples)

    mpl.rcParams.update({"font.size": 9.5, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    ax.set_facecolor("#fafbfc")

    # subtle grid
    for x in range(1, 11):
        ax.axvline(x, color="#e4e7eb", linewidth=0.6, zorder=0)
        ax.axhline(x, color="#e4e7eb", linewidth=0.6, zorder=0)

    # m > n, m-n odd diagonal hatching (valid Euclid region)
    ax.fill_between([0, 10], [0, 10], [0, 0], color="#f3f5f8", zorder=0, alpha=0.6)

    # Fibonacci ladder line connecting the three all-Fibonacci generators
    fib_points = sorted(
        [(t["m"], t["n"]) for t in triples if t["all_fib"]],
        key=lambda p: p[0]
    )
    if len(fib_points) >= 2:
        xs = [p[0] for p in fib_points]
        ys = [p[1] for p in fib_points]
        ax.plot(xs, ys, color="#d4a017", linestyle="-", linewidth=2.2,
                alpha=0.55, zorder=1, label="Fibonacci ladder")

    # Plot non-Fibonacci triples
    for t in triples:
        if t["all_fib"]:
            continue
        label = f"{t['a']}:{t['b']}:{t['c']}"
        ax.scatter(t["m"], t["n"], s=90, c="#7a8190", edgecolors="#3a4050",
                   linewidths=1.0, zorder=2)
        # label placement: prefer upper-right, fall back based on congestion
        dx, dy = 0.18, 0.14
        if t["m"] >= 8:
            dx = -0.18
            ha = "right"
        else:
            ha = "left"
        ax.annotate(label, (t["m"], t["n"]), xytext=(t["m"] + dx, t["n"] + dy),
                    fontsize=7.8, color="#3a4050", ha=ha,
                    zorder=3)

    # Plot Fibonacci triples last, larger + gold
    for t in triples:
        if not t["all_fib"]:
            continue
        label = f"{t['a']}:{t['b']}:{t['c']}"
        ax.scatter(t["m"], t["n"], s=260, c="#e6b800", edgecolors="#8a6b10",
                   linewidths=1.8, zorder=4)
        ax.annotate(label, (t["m"], t["n"]),
                    xytext=(t["m"] + 0.22, t["n"] + 0.22),
                    fontsize=10.5, fontweight="bold", color="#5a4608",
                    ha="left", zorder=5)
        ax.annotate(f"QN=({t['qn'][0]},{t['qn'][1]},{t['qn'][2]},{t['qn'][3]})",
                    (t["m"], t["n"]),
                    xytext=(t["m"] + 0.22, t["n"] - 0.35),
                    fontsize=7.5, color="#6a5408", style="italic",
                    ha="left", zorder=5)

    ax.set_xlim(1.3, 9.8)
    ax.set_ylim(0.3, 8.8)
    ax.set_xlabel(r"Euclid generator $m$")
    ax.set_ylabel(r"Euclid generator $n$")
    ax.set_xticks(range(2, 10))
    ax.set_yticks(range(1, 9))
    ax.set_aspect("equal")

    title = (f"Primitive Pythagorean triples with $c \\leq {c_max}$:  "
             f"{n_fib}/{n_tot} have all-Fibonacci QA quantum numbers")
    ax.set_title(title)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="#e6b800",
               markeredgecolor="#8a6b10", markersize=13,
               label=f"All-Fibonacci QN ({n_fib})"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#7a8190",
               markeredgecolor="#3a4050", markersize=9,
               label=f"Non-Fibonacci QN ({n_tot - n_fib})"),
        Line2D([], [], color="#d4a017", linewidth=2.2, alpha=0.55,
               label="Fibonacci ladder: $(m,n) = (F_{k+1}, F_k)$"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8.5,
              framealpha=0.95)

    # Footer note
    fig.text(0.5, 0.015,
             "QN = (m−n, n, m, m+n).  Fibonacci triples: (2,1)→3:4:5,  "
             "(3,2)→5:12:13,  (8,5)→39:80:89.",
             ha="center", fontsize=8, color="#555")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout(rect=[0, 0.035, 1, 1])
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"wrote {out_path}")
    print(f"      {out_path.with_suffix('.png')}")
    print(f"Fibonacci triples: {n_fib}/{n_tot}")


if __name__ == "__main__":
    main()
