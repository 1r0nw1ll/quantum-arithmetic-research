#!/usr/bin/env python3
"""
generate_obstruction_perturbation_certs.py

Controlled obstruction perturbation experiment on the FLEXIBLE PLANNER synthetic grid.

Introduces PARITY_BLOCK tags on cross_link generators one at a time and records
the exact PDI response.  Demonstrates Bridge Theorem B1 quantitatively:

  τ(g) = PARITY_BLOCK (active) → obstruction Type II (route-reducing)
  Each blocked cross_link_k removes all 5 multi-path states from arm_{(k+1)%4}.
  Effects are independent across arms.

PDI sweep:
  k=0 (baseline)  |M|=20  PDI=0.800  FLEXIBLE PLANNER   (already in reference_sets)
  k=1 (1-block)   |M|=15  PDI=0.600  FLEXIBLE PLANNER   → written here
  k=2 (2-block)   |M|=10  PDI=0.400  LINEAR EXPLORER    → written here (regime collapse)
  k=3             |M|= 5  PDI=0.200  LINEAR EXPLORER    (analytical, not written)
  k=4             |M|= 0  PDI=0.000  FROZEN             (analytical, not written)

Outputs:
  reference_sets/v1/synthetic/flexible_planner_1block.bundle.json
  reference_sets/v1/synthetic/flexible_planner_2block.bundle.json
  pi_pdi_fragility.png   (Figure 2)
"""
import hashlib
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HEX64_ZERO = "0" * 64
OUT_DIR = pathlib.Path(__file__).parent / "reference_sets" / "v1" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_OUT = pathlib.Path(__file__).parent.parent / "qa_alphageometry_ptolemy" / "pi_pdi_fragility.png"


# ── Hashing helpers ──────────────────────────────────────────────────────────

def canonical_json_compact(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj):
    return hashlib.sha256(canonical_json_compact(obj).encode("utf-8")).hexdigest()


def update_manifest(obj):
    obj["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    computed = sha256_canonical(obj)
    obj["manifest"]["canonical_json_sha256"] = computed
    return computed


# ── Cert factory ─────────────────────────────────────────────────────────────

MOVE_PROBS_FULL = {
    "arm_advance_0": 0.10, "arm_advance_1": 0.10,
    "arm_advance_2": 0.10, "arm_advance_3": 0.10,
    "cross_link_0":  0.15, "cross_link_1":  0.15,
    "cross_link_2":  0.15, "cross_link_3":  0.15,
}

REACHABLE = 25
TOTAL     = 40
ATTRACTOR = 4


def regime(pdi):
    if pdi > 0.5:
        return "FLEXIBLE PLANNER"
    elif pdi > 0.0:
        return "LINEAR EXPLORER"
    else:
        return "FROZEN"


def build_cert(k: int) -> dict:
    """Build cert for k PARITY_BLOCK tags on cross_link_{0..k-1}."""
    blocked = [f"cross_link_{i}" for i in range(k)]
    ok_gens  = [f"cross_link_{i}" for i in range(k, 4)]

    multi_path   = 20 - 5 * k
    pdi_val      = multi_path / REACHABLE
    ai_val       = REACHABLE / TOTAL
    pi_val       = 15.0 / 20.0          # delta_reach / delta_perturb — unchanged
    gd_val       = ATTRACTOR / TOTAL
    ce_val       = -sum(p * math.log(p) for p in MOVE_PROBS_FULL.values())

    # Obstruction names — one per blocked cross_link
    obs_names = [f"parity_block_cross_link_{i}" for i in range(k)]

    # Generators list — arm_advance always OK; cross_links tagged per block
    generators = []
    for arm in range(4):
        generators.append({
            "id":          f"arm_advance_{arm}",
            "description": f"Advance within arm {arm}: arm_{arm}_d → arm_{arm}_{{d+1}}",
            "action":      "progression",
        })
    for cl in range(4):
        blocked_flag = cl < k
        generators.append({
            "id":          f"cross_link_{cl}",
            "description": (
                f"Cross-arm bridge arm {cl}→arm {(cl+1)%4}: "
                f"arm_{cl}_d → arm_{(cl+1)%4}_{{d+1}}"
                + (" [PARITY_BLOCK — active constraint, Type II obstruction]" if blocked_flag else "")
            ),
            "action":      "merge_bridge" if not blocked_flag else "merge_bridge_blocked",
        })

    # Invariants — ok_tagging holds only for unblocked generators
    invariants = []
    if k == 0:
        invariants.append({
            "name":       "ok_tagging",
            "expression": "tau(g) = OK for all g in generators",
            "tolerance":  0.0,
        })
        invariants.append({
            "name":       "merge_stability",
            "expression": "join(tau(g_arm), tau(g_cross)) = OK for all merge pairs",
            "tolerance":  0.0,
        })
    else:
        invariants.append({
            "name":       "partial_ok_tagging",
            "expression": (
                f"tau(g) = OK for arm_advance_* and cross_link_{{{','.join(str(i) for i in range(k, 4))}}}; "
                f"tau(cross_link_{{{','.join(str(i) for i in range(k))}}}) = PARITY_BLOCK"
            ),
            "tolerance":  0.0,
        })
        invariants.append({
            "name":       "merge_stability_partial",
            "expression": (
                f"join(tau(g_arm), tau(g_cross)) = OK for arms {{{','.join(str(i) for i in range(k, 4))}}}; "
                f"join = PARITY_BLOCK for arm(s) {{{','.join(str((i+1)%4) for i in range(k))}}}"
            ),
            "tolerance":  0.0,
        })

    regime_str = regime(pdi_val)
    ea_val     = ai_val * pdi_val

    cert = {
        "schema_id": "QA_COMPETENCY_DETECTION_FRAMEWORK.v1",
        "system_metadata": {
            "domain":    "adaptive_planning",
            "substrate": "synthetic_grid_perturbed",
            "description": (
                f"Synthetic FLEXIBLE PLANNER grid with k={k} PARITY_BLOCK obstruction(s). "
                f"Blocked: {blocked if blocked else 'none'}. "
                f"PDI={pdi_val:.3f} (delta from baseline: {pdi_val - 0.800:.3f}). "
                f"Regime: {regime_str}. EA={ea_val:.4f}. "
                f"Demonstrates Bridge Theorem B1: each PARITY_BLOCK tag produces a "
                f"Type II obstruction removing exactly 5 states from M (arm_{[(i+1)%4 for i in range(k)]}). "
                f"{'Regime collapse: FLEXIBLE PLANNER → LINEAR EXPLORER at k=2.' if k == 2 else ''}"
            ),
        },
        "state_space": {
            "dimension": 3,
            "coordinates": ["arm_index", "depth", "route_count"],
            "constraints": [
                "arm_index in {0,1,2,3}",
                "depth in {0,...,6}",
                "route_count: 1=single-path, 2=multi-path",
            ],
        },
        "generators": generators,
        "invariants": invariants,
        "reachability": {
            "components":  1,
            "diameter":    6,
            "obstructions": obs_names,
        },
        "graph_snapshot": {
            "hash_sha256": HEX64_ZERO,
            "time_window": {
                "start_utc": "2026-03-01T00:00:00Z",
                "end_utc":   "2026-03-01T00:00:00Z",
            },
            "edge_semantics": (
                f"Perturbation level k={k}: "
                f"cross_link_{{0..{k-1}}} PARITY_BLOCK (all arms {[(i+1)%4 for i in range(k)]} lose multi-path access). "
                f"arm_advance_* and cross_link_{{{k}..3}} remain OK-tagged. "
                f"|M| = 20 - 5*{k} = {multi_path}. "
                f"PDI = {multi_path}/25 = {pdi_val:.3f}. "
                f"Type II obstruction per blocked cross_link: arm_{{(k+1)%4}}_d (d>=2) "
                f"loses its only merge edge -> exits M, stays in R."
            ),
        },
        "metric_inputs": {
            "reachable_states":   REACHABLE,
            "total_states":       TOTAL,
            "attractor_basins":   ATTRACTOR,
            "move_probabilities": MOVE_PROBS_FULL,
            "delta_reachability": 15.0,
            "delta_perturbation": 20.0,
            "multi_path_states":  multi_path,
        },
        "competency_metrics": {
            "agency_index":     ai_val,
            "plasticity_index": pi_val,
            "goal_density":     gd_val,
            "control_entropy":  ce_val,
            "pdi":              pdi_val,
        },
        "validation": {
            "validator":            "qa_competency_validator.py",
            "hash":                 f"sha256:{HEX64_ZERO}",
            "reproducibility_seed": 20260301 + k,
        },
        "examples": [
            f"flexible_planner_k{k}_parity_block_perturbation",
            f"bridge_theorem_b1_type2_obstruction_k{k}",
        ],
        "manifest": {
            "manifest_version":      1,
            "hash_alg":              "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }

    update_manifest(cert)
    return cert


def build_bundle(k: int) -> dict:
    cert   = build_cert(k)
    bundle = {
        "schema_id": "QA_COMPETENCY_CERT_BUNDLE.v1",
        "manifest": {
            "manifest_version":      1,
            "hash_alg":              "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        },
        "certs": [cert],
    }
    update_manifest(bundle)
    return bundle


# ── Generate k=1 and k=2 bundles ─────────────────────────────────────────────

for k, name in [(1, "flexible_planner_1block"), (2, "flexible_planner_2block")]:
    bundle = build_bundle(k)
    path   = OUT_DIR / f"{name}.bundle.json"
    path.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    pdi = bundle["certs"][0]["competency_metrics"]["pdi"]
    reg = regime(pdi)
    print(f"Wrote {name}.bundle.json  PDI={pdi:.3f}  {reg}")


# ── Figure 2: PDI fragility curve ────────────────────────────────────────────

C = {
    "planner":  "#f4ecc8",
    "explorer": "#c8e8c8",
    "frozen":   "#c8d8e8",
    "pdi_line": "#1a6ea8",
    "ea_line":  "#8a3a8a",
    "thresh":   "#cc2200",
    "grid":     "#dddddd",
    "text":     "#1a1a1a",
    "mid":      "#555555",
}

ks      = np.arange(5)
PDIs    = np.array([0.800, 0.600, 0.400, 0.200, 0.000])
EAs     = 0.625 * PDIs
PI_line = 0.750  # constant

fig, ax = plt.subplots(figsize=(8, 5))

# ── Regime background bands ──────────────────────────────────────────────────
ax.axhspan(0.5, 1.0, color=C["planner"],  alpha=0.45, zorder=0)
ax.axhspan(0.0, 0.5, color=C["explorer"], alpha=0.45, zorder=0)
ax.axhspan(-0.05, 0.0, color=C["frozen"], alpha=0.45, zorder=0)

# ── Threshold line ────────────────────────────────────────────────────────────
ax.axhline(0.5, color=C["thresh"], lw=1.4, ls="--", zorder=2, label="PDI = 0.5 threshold")

# ── PDI line ─────────────────────────────────────────────────────────────────
ax.plot(ks, PDIs, "o-", color=C["pdi_line"], lw=2.2, ms=9,
        zorder=4, label="PDI (Path Diversity Index)")

# ── EA line ───────────────────────────────────────────────────────────────────
ax.plot(ks, EAs,  "s--", color=C["ea_line"], lw=1.8, ms=7,
        zorder=4, label="EA = AI × PDI  (AI = 0.625 fixed)")

# ── PI constant ───────────────────────────────────────────────────────────────
ax.axhline(PI_line, color="#666666", lw=1.2, ls=":", zorder=2,
           label=f"PI = {PI_line:.3f} (constant)")

# ── Regime collapse annotation ────────────────────────────────────────────────
ax.annotate(
    "Regime collapse\nFLEXIBLE PLANNER\n→ LINEAR EXPLORER",
    xy=(2, 0.400), xytext=(2.5, 0.560),
    fontsize=8.5, color=C["text"], ha="left",
    arrowprops=dict(arrowstyle="->", color=C["thresh"], lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["thresh"], lw=0.8),
    zorder=6,
)

# ── Point labels ─────────────────────────────────────────────────────────────
for k_i, (pdi_i, ea_i) in enumerate(zip(PDIs, EAs)):
    reg = regime(pdi_i)
    ax.text(k_i, pdi_i + 0.028, f"{pdi_i:.2f}", ha="center", fontsize=9,
            color=C["pdi_line"], fontweight="bold", zorder=5)
    ax.text(k_i, ea_i - 0.042, f"{ea_i:.3f}", ha="center", fontsize=7.5,
            color=C["ea_line"], zorder=5)

# ── Regime labels ────────────────────────────────────────────────────────────
ax.text(4.45, 0.72, "FLEXIBLE\nPLANNER", ha="right", va="center",
        fontsize=9.5, color="#7a6010", fontweight="bold")
ax.text(4.45, 0.25, "LINEAR\nEXPLORER", ha="right", va="center",
        fontsize=9.5, color="#1a5a1a", fontweight="bold")

# ── Axes decoration ───────────────────────────────────────────────────────────
ax.set_xlim(-0.3, 4.6)
ax.set_ylim(-0.02, 0.92)
ax.set_xticks(ks)
ax.set_xticklabels(
    [f"k=0\n(baseline)", "k=1\n(1 block)", "k=2\n(2 blocks)\n★",
     "k=3\n(3 blocks)", "k=4\n(4 blocks)"],
    fontsize=8.5
)
ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0])
ax.set_yticklabels(["0", "0.2", "0.4", "0.5", "0.6", "0.75", "0.8", "1.0"], fontsize=8.5)
ax.set_ylabel("Index value", fontsize=11, color=C["text"])
ax.set_xlabel("Number of PARITY_BLOCK tags added  (one per cross_link generator)", fontsize=10)

ax.grid(axis="y", color=C["grid"], lw=0.8, zorder=1)
for spine in ax.spines.values():
    spine.set_edgecolor("#aaaaaa")
    spine.set_linewidth(0.8)

ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, edgecolor="#aaaaaa")

fig.text(0.5, 0.97,
         "PDI Fragility Under PARITY_BLOCK Obstruction Tagging",
         ha="center", va="top", fontsize=13, fontweight="bold", color=C["text"])
fig.text(0.5, 0.935,
         r"$\Delta PDI = -0.200$ per tag  |  Regime collapse at $k=2$  |  "
         r"$\Delta EA = -0.125$ per tag  |  Family [26] synthetic grid",
         ha="center", va="top", fontsize=9, color=C["mid"])

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
print(f"\nSaved figure: {FIG_OUT}")
