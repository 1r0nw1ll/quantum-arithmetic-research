#!/usr/bin/env python3
"""
generate_pi_pdi_quadrant.py

Produces Figure 1 for Family [26]: the (PI, PDI) structural regime map.
Output: pi_pdi_quadrant.png  (300 dpi, 8×7 inches)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── colour palette ─────────────────────────────────────────────────────────
C = {
    "frozen":    "#c8d8e8",   # muted blue-grey
    "explorer":  "#c8e8c8",   # muted green
    "thrashing": "#f4c8c8",   # muted red
    "planner":   "#f4ecc8",   # muted gold
    "axis":      "#333333",
    "divider":   "#888888",
    "label_bg":  "#ffffff",
    "text_dark": "#1a1a1a",
    "text_mid":  "#444444",
    "scatter":   "#1a6ea8",
    "scatter_edge": "#ffffff",
}

fig, ax = plt.subplots(figsize=(8, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

# ── quadrant fills ──────────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch((0, 0),   0.5, 0.5, boxstyle="square,pad=0",
                                      fc=C["frozen"],    ec="none", zorder=0))
ax.add_patch(mpatches.FancyBboxPatch((0.5, 0), 0.5, 0.5, boxstyle="square,pad=0",
                                      fc=C["explorer"],  ec="none", zorder=0))
ax.add_patch(mpatches.FancyBboxPatch((0, 0.5), 0.5, 0.5, boxstyle="square,pad=0",
                                      fc=C["thrashing"], ec="none", zorder=0))
ax.add_patch(mpatches.FancyBboxPatch((0.5, 0.5), 0.5, 0.5, boxstyle="square,pad=0",
                                      fc=C["planner"],   ec="none", zorder=0))

# ── dividers ────────────────────────────────────────────────────────────────
ax.axhline(0.5, color=C["divider"], lw=1.4, ls="--", zorder=1)
ax.axvline(0.5, color=C["divider"], lw=1.4, ls="--", zorder=1)

# ── quadrant titles ─────────────────────────────────────────────────────────
quad_kw = dict(ha="center", va="center", fontsize=13, fontweight="bold",
               color=C["text_dark"], zorder=3)

ax.text(0.25, 0.84, "STUCK-LOOP\nTHRASHING", **quad_kw)
ax.text(0.75, 0.84, "FLEXIBLE\nPLANNER", **quad_kw)
ax.text(0.25, 0.34, "FROZEN", **quad_kw)
ax.text(0.75, 0.34, "LINEAR\nEXPLORER", **quad_kw)

# ── quadrant sub-labels ─────────────────────────────────────────────────────
sub_kw = dict(ha="center", va="center", fontsize=8.5, color=C["text_mid"],
              style="italic", zorder=3)

ax.text(0.25, 0.74,
        "PI=lo  PDI=hi\n"
        "Redundant paths, no new territory\n"
        "cycles w/ multi-route access\n"
        "EA = low·high  →  moderate",
        **sub_kw)

ax.text(0.75, 0.74,
        "PI=hi  PDI=hi\n"
        "New territory + redundant paths\n"
        "rich reachability structure\n"
        "EA = high·high  →  maximum",
        **sub_kw)

ax.text(0.25, 0.24,
        "PI=lo  PDI=lo\n"
        "No new territory, no alternatives\n"
        "deterministic, tree-like expansion\n"
        "EA = low·low  →  minimum",
        **sub_kw)

ax.text(0.75, 0.24,
        "PI=hi  PDI=lo\n"
        "New territory, single route\n"
        "breadth without redundancy\n"
        "EA = high·low  →  moderate",
        **sub_kw)

# ── reference set scatter ───────────────────────────────────────────────────
# Actual (PI, PDI) from the 9 reference bundles (read from bundle JSON files)
REFS = [
    # label,          PI,      PDI,      domain
    ("multi-agent",   0.700,   0.4680,   "ai"),    # multi_agent_systems
    ("retrieval-RAG", 0.680,   0.4512,   "ai"),    # information_retrieval
    ("tool-debugger", 0.300,   0.4375,   "ai"),    # software_engineering  ← FROZEN
    ("organoid",      0.550,   0.4355,   "bio"),   # synthetic_biology
    ("planarian",     0.850,   0.4535,   "bio"),   # developmental_biology
    ("xenopus",       0.720,   0.4625,   "bio"),   # developmental_biology
    ("bioelec+LLM",   0.780,   0.4412,   "hybrid"),# bioelectric_computing
    ("human-loop",    0.820,   0.4747,   "hybrid"),# human_computer_interaction
    ("lab-robot",     0.350,   0.4255,   "hybrid"),# laboratory_automation  ← FROZEN
]

domain_colors = {"ai": "#1a6ea8", "bio": "#2a8a3a", "hybrid": "#8a3a8a"}
domain_labels = {"ai": "AI agents", "bio": "Biological", "hybrid": "Hybrid"}

label_offsets = {
    "multi-agent":   ( 0.03,  0.012),
    "retrieval-RAG": ( 0.03, -0.020),
    "tool-debugger": (-0.03,  0.018),
    "organoid":      ( 0.03,  0.012),
    "planarian":     ( 0.03,  0.012),
    "xenopus":       ( 0.03, -0.020),
    "bioelec+LLM":   ( 0.03,  0.012),
    "human-loop":    ( 0.03,  0.012),
    "lab-robot":     (-0.03, -0.020),
}
ha_map = {k: ("left" if v[0] > 0 else "right") for k, v in label_offsets.items()}

for label, pi, pdi, domain in REFS:
    ax.scatter(pi, pdi, s=90, color=domain_colors[domain],
               edgecolors=C["scatter_edge"], linewidths=0.8, zorder=5)
    dx, dy = label_offsets[label]
    ax.text(pi + dx, pdi + dy, label, fontsize=6.5, color="#333333",
            ha=ha_map[label], va="center", zorder=6)

# ── legend for domain colours ────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(fc=domain_colors[d], ec="#555555", lw=0.7, label=domain_labels[d])
    for d in ["ai", "bio", "hybrid"]
]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.085), ncol=3,
          fontsize=9, framealpha=0.9, edgecolor="#aaaaaa")

# ── axes labels ─────────────────────────────────────────────────────────────
ax.set_xlabel("Plasticity Index  (PI)", fontsize=12, labelpad=10,
              color=C["text_dark"])
ax.set_ylabel("Path Diversity Index  (PDI)", fontsize=12, labelpad=10,
              color=C["text_dark"])

ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "", "0.5\n(threshold)", "", "1.0"], fontsize=9)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "", "0.5\n(threshold)", "", "1.0"], fontsize=9)

ax.tick_params(length=3, color="#888888")
for spine in ax.spines.values():
    spine.set_edgecolor("#888888")
    spine.set_linewidth(0.8)

# ── title + subtitle ────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         "Structural Regime Map: Plasticity × Path Diversity",
         ha="center", va="top", fontsize=14, fontweight="bold",
         color=C["text_dark"])
fig.text(0.5, 0.935,
         r"$EA = AI \times PDI$     |     Family [26] reference set  ($n = 9$)",
         ha="center", va="top", fontsize=9.5, color=C["text_mid"])

# ── formula box ─────────────────────────────────────────────────────────────
formula_box = dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#aaaaaa", lw=0.8)
ax.text(0.5, 0.5,
        r"$PDI = \frac{|M|}{R}$",
        ha="center", va="center", fontsize=13,
        bbox=formula_box, zorder=6, color=C["text_dark"])

plt.tight_layout(rect=[0, 0.05, 1, 0.93])

out = "pi_pdi_quadrant.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
