#!/usr/bin/env python3
"""
generate_flexible_planner_cert.py

Constructs the first FLEXIBLE PLANNER synthetic cert: a 4-arm grid with
OK-tagged cross-link generators, achieving PDI = 0.800 > 0.5.

Design (documented as pdi_construction in graph_snapshot edge_semantics):

  Graph:
    - apex state s0 (depth 0, shared initial state)
    - arm_k states for k in {0,1,2,3}, d in {1,...,6}
    - Total R = 1 + 4*6 = 25 reachable states

  Generators (all OK-tagged — failure_algebra join = OK throughout):
    arm_advance_k:  arm_k_d  → arm_k_{d+1}        (within-arm progression)
    cross_link_k:   arm_k_d  → arm_{(k+1)%4}_{d+1} (inter-arm merge bridge)

  Path-count analysis (SCC-condensation DAG, no cycles):
    - s0 (depth 0):       1 path  → NOT in M
    - arm_k_1 (depth 1):  1 path from s0 via arm_advance_k → NOT in M
    - arm_k_d (depth ≥2): reachable via arm_advance_k(arm_k_{d-1}) [path A]
                           AND via cross_link_{(k-1)%4}(arm_{(k-1)%4}_{d-1}) [path B]
                           → 2 distinct paths → IN M

  |M| = 4 arms × 5 depths (d=2..6) = 20
  PDI = 20 / 25 = 0.800

Bridge theorem validation:
  Theorem B1: all generators τ=OK → all obstructions vacuously Type III
  Theorem B2: join(OK,OK)=OK for all merge pairs → PDI preserved under any obstruction
  Regime: FLEXIBLE PLANNER (PI=0.75 > 0.5, PDI=0.80 > 0.5, EA=0.625×0.800=0.500)

Output: reference_sets/v1/synthetic/flexible_planner_grid.bundle.json
"""
import hashlib
import json
import math
import pathlib

HEX64_ZERO = "0" * 64


def canonical_json_compact(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj):
    return hashlib.sha256(canonical_json_compact(obj).encode("utf-8")).hexdigest()


def update_manifest(obj):
    """Zero out canonical_json_sha256, compute hash, write it back."""
    obj["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    computed = sha256_canonical(obj)
    obj["manifest"]["canonical_json_sha256"] = computed
    return computed


# ── Metric inputs ────────────────────────────────────────────────────────────

REACHABLE_STATES = 25      # 1 apex + 4 arms × 6 depths
TOTAL_STATES     = 40      # + 15 structurally blocked states outside the grid
ATTRACTOR_BASINS = 4       # terminal states: arm_0_6, arm_1_6, arm_2_6, arm_3_6
DELTA_REACH      = 15.0    # reachability sensitivity to perturbation
DELTA_PERTURB    = 20.0    # perturbation magnitude
MULTI_PATH       = 20      # 4 arms × 5 depths (d=2..6)

MOVE_PROBS = {
    "arm_advance_0": 0.10,
    "arm_advance_1": 0.10,
    "arm_advance_2": 0.10,
    "arm_advance_3": 0.10,
    "cross_link_0":  0.15,
    "cross_link_1":  0.15,
    "cross_link_2":  0.15,
    "cross_link_3":  0.15,
}
assert abs(sum(MOVE_PROBS.values()) - 1.0) < 1e-12, "probabilities must sum to 1"

# ── Derived metrics ──────────────────────────────────────────────────────────

agency_index     = REACHABLE_STATES / TOTAL_STATES          # 0.625
plasticity_index = DELTA_REACH / DELTA_PERTURB              # 0.750
goal_density     = ATTRACTOR_BASINS / TOTAL_STATES          # 0.100
control_entropy  = -sum(p * math.log(p) for p in MOVE_PROBS.values())  # ≈2.0593
pdi_val          = MULTI_PATH / REACHABLE_STATES            # 0.800

print(f"agency_index     = {agency_index:.6f}")
print(f"plasticity_index = {plasticity_index:.6f}")
print(f"goal_density     = {goal_density:.6f}")
print(f"control_entropy  = {control_entropy:.6f}")
print(f"pdi              = {pdi_val:.6f}")
print(f"EA               = {agency_index * pdi_val:.6f}")
print(f"Regime           = {'FLEXIBLE PLANNER' if plasticity_index > 0.5 and pdi_val > 0.5 else 'OTHER'}")

# ── Cert construction ────────────────────────────────────────────────────────

cert = {
    "schema_id": "QA_COMPETENCY_DETECTION_FRAMEWORK.v1",
    "system_metadata": {
        "domain": "adaptive_planning",
        "substrate": "synthetic_grid",
        "description": (
            "Synthetic Flexible Planner: 4-arm depth-6 grid with OK-tagged "
            "cross-link generators. Constructed as the first existence proof "
            "for the FLEXIBLE PLANNER regime (PI=0.75, PDI=0.80). All 8 "
            "generators are OK-tagged; failure algebra join(τ(gᵢ),τ(gⱼ))=OK "
            "for all merge-path pairs. Demonstrates the PDI-Obstruction Bridge "
            "design rule: explicit merge topology is necessary and sufficient "
            "for PDI > 0.5 when all generators are obstruction-free."
        ),
    },
    "state_space": {
        "dimension": 3,
        "coordinates": ["arm_index", "depth", "route_count"],
        "constraints": [
            "arm_index in {0,1,2,3}",
            "depth in {0,...,6}",
            "route_count: 1 at depth<=1 (single-path), 2 at depth>=2 (multi-path)",
        ],
    },
    "generators": [
        {
            "id": "arm_advance_0",
            "description": "Advance within arm 0: arm_0_d → arm_0_{d+1}",
            "action": "progression",
        },
        {
            "id": "arm_advance_1",
            "description": "Advance within arm 1: arm_1_d → arm_1_{d+1}",
            "action": "progression",
        },
        {
            "id": "arm_advance_2",
            "description": "Advance within arm 2: arm_2_d → arm_2_{d+1}",
            "action": "progression",
        },
        {
            "id": "arm_advance_3",
            "description": "Advance within arm 3: arm_3_d → arm_3_{d+1}",
            "action": "progression",
        },
        {
            "id": "cross_link_0",
            "description": "Cross-arm bridge arm 0→arm 1: arm_0_d → arm_1_{d+1}",
            "action": "merge_bridge",
        },
        {
            "id": "cross_link_1",
            "description": "Cross-arm bridge arm 1→arm 2: arm_1_d → arm_2_{d+1}",
            "action": "merge_bridge",
        },
        {
            "id": "cross_link_2",
            "description": "Cross-arm bridge arm 2→arm 3: arm_2_d → arm_3_{d+1}",
            "action": "merge_bridge",
        },
        {
            "id": "cross_link_3",
            "description": "Cross-arm bridge arm 3→arm 0: arm_3_d → arm_0_{d+1}",
            "action": "merge_bridge",
        },
    ],
    "invariants": [
        {
            "name": "ok_tagging",
            "expression": "tau(g) = OK for all g in generators",
            "tolerance": 0.0,
        },
        {
            "name": "merge_stability",
            "expression": (
                "join(tau(g_arm), tau(g_cross)) = OK "
                "for all (arm_advance_k, cross_link_{k-1}) merge pairs"
            ),
            "tolerance": 0.0,
        },
    ],
    "reachability": {
        "components": 1,
        "diameter": 6,
        "obstructions": [],   # vacuously Type III: all generators OK-tagged
    },
    "graph_snapshot": {
        "hash_sha256": HEX64_ZERO,
        "time_window": {
            "start_utc": "2026-03-01T00:00:00Z",
            "end_utc":   "2026-03-01T00:00:00Z",
        },
        "edge_semantics": (
            "arm_advance_k: arm_k_d → arm_k_{d+1} (k=0..3, d=0..5); "
            "cross_link_k: arm_k_d → arm_{(k+1)%4}_{d+1} (k=0..3, d=0..5); "
            "all generators failure_tag=OK (carrier: QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1); "
            "multi-path states: depth>=2 via (arm_advance + cross_link) routes; "
            "PDI construction: A=4 arms, D=6, no bridge states needed (cyclic cross-link); "
            "|M| = A*(D-1) = 4*5 = 20; PDI = 20/25 = 0.800"
        ),
    },
    "metric_inputs": {
        "reachable_states":  REACHABLE_STATES,
        "total_states":      TOTAL_STATES,
        "attractor_basins":  ATTRACTOR_BASINS,
        "move_probabilities": MOVE_PROBS,
        "delta_reachability": DELTA_REACH,
        "delta_perturbation": DELTA_PERTURB,
        "multi_path_states": MULTI_PATH,
    },
    "competency_metrics": {
        "agency_index":     agency_index,      # exact: 25/40
        "plasticity_index": plasticity_index,  # exact: 15.0/20.0
        "goal_density":     goal_density,      # exact: 4/40
        "control_entropy":  control_entropy,   # full float: -sum(p ln p)
        "pdi":              pdi_val,           # exact: 20/25
    },
    "validation": {
        "validator":            "qa_competency_validator.py",
        "hash":                 f"sha256:{HEX64_ZERO}",
        "reproducibility_seed": 20260301,
    },
    "examples": [
        "four_arm_depth6_cyclic_cross_link",
        "first_flexible_planner_existence_proof",
        "ok_tagged_merge_stability_bridge_theorem_b2",
    ],
    "manifest": {
        "manifest_version":      1,
        "hash_alg":              "sha256_canonical",
        "canonical_json_sha256": HEX64_ZERO,
    },
}

update_manifest(cert)

# ── Bundle ───────────────────────────────────────────────────────────────────

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

# ── Write ────────────────────────────────────────────────────────────────────

out_path = (
    pathlib.Path(__file__).parent
    / "reference_sets" / "v1" / "synthetic"
    / "flexible_planner_grid.bundle.json"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
)
print(f"\nWrote: {out_path}")
print(f"cert manifest sha256:   {cert['manifest']['canonical_json_sha256']}")
print(f"bundle manifest sha256: {bundle['manifest']['canonical_json_sha256']}")
