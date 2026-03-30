#!/usr/bin/env python3
"""
qa_cymatics_validate.py

Validator for QA Cymatics certificate families.

Cert families:
    QA_CYMATIC_MODE_CERT.v1         -- Chladni mode witness + QA (b,e) mapping
    QA_FARADAY_REACHABILITY_CERT.v1 -- Faraday pattern-basin reachability
    QA_CYMATIC_CONTROL_CERT.v1      -- Lawful generator sequence → target pattern (programmability tier)
    QA_CYMATIC_PLANNER_CERT.v1      -- Bounded-search plan synthesis: BFS/DFS witness or no-plan certificate

Replay contract:
    LOAD -> CANONICALIZE -> VERIFY_CHECKS -> VERIFY_FAIL_LEDGER -> EMIT_RESULT

Usage:
    python qa_cymatics_validate.py --mode     fixtures/mode_cert_pass.json
    python qa_cymatics_validate.py --faraday  fixtures/faraday_cert_pass.json
    python qa_cymatics_validate.py --control  fixtures/control_cert_pass_hexagon.json
    python qa_cymatics_validate.py --control  fixtures/control_cert_fail_illegal_transition.json
    python qa_cymatics_validate.py --planner  fixtures/planner_cert_pass_shortest_hexagon.json
    python qa_cymatics_validate.py --planner  fixtures/planner_cert_fail_no_plan_within_bound.json
    python qa_cymatics_validate.py --demo
    python qa_cymatics_validate.py --self-test
"""

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cert_hash(cert: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json(cert))


# ---------------------------------------------------------------------------
# Failure algebra
# ---------------------------------------------------------------------------

MODE_FAIL_TYPES = frozenset([
    "OFF_RESONANCE",
    "BOUNDARY_MISMATCH",
    "MODE_MIXING",
    "DAMPING_COLLAPSE",
    "MEASUREMENT_ALIAS",
    "TUPLE_FORMULA_VIOLATION",
    "ORBIT_CLASS_MISMATCH",
])

FARADAY_FAIL_TYPES = frozenset([
    "NONLINEAR_ESCAPE",
    "MODE_MIXING",
    "DAMPING_COLLAPSE",
    "BOUNDARY_MISMATCH",
    "ILLEGAL_TRANSITION",
    "RETURN_PATH_NOT_FOUND",
    "PATTERN_CLASS_UNRECOGNIZED",
])

CONTROL_FAIL_TYPES = frozenset([
    "OFF_RESONANCE",
    "BOUNDARY_MISMATCH",
    "MODE_MIXING",
    "DAMPING_COLLAPSE",
    "NONLINEAR_ESCAPE",
    "MEASUREMENT_ALIAS",
    "ILLEGAL_TRANSITION",
    "RETURN_PATH_NOT_FOUND",
    "GOAL_NOT_REACHED",
    "ORBIT_CLASS_MISMATCH",
    "PATH_LENGTH_EXCEEDED",
    "PATTERN_CLASS_UNRECOGNIZED",
])

# Canonical mapping from pattern class to QA orbit family
CONTROL_PATTERN_TO_ORBIT: Dict[str, str] = {
    "flat":          "singularity",
    "rings":         "singularity",
    "stripes":       "satellite",
    "squares":       "satellite",
    "oscillons":     "satellite",
    "hexagons":      "cosmos",
    "quasipattern":  "cosmos",
    "disordered":    "out_of_orbit",
}

PLANNER_FAIL_TYPES = frozenset([
    "NO_PLAN_WITHIN_BOUND",
    "GOAL_NOT_REACHABLE",
    "SEARCH_INCONSISTENCY",
    "NONMINIMAL_PLAN",
    "ILLEGAL_TRANSITION",
    "NONLINEAR_ESCAPE",
    "ORBIT_CLASS_MISMATCH",
    "RETURN_PATH_NOT_FOUND",
    "PLAN_CONTROL_MISMATCH",
    "REPLAY_INCONSISTENCY",
    "COMPILED_CERT_MISSING",
    "MINIMALITY_WITNESS_INCOMPLETE",
])

RECOGNIZED_ALGORITHMS = frozenset(["bfs", "dfs", "astar", "dijkstra", "iddfs"])

VALID_ORBIT_FAMILIES = frozenset(["cosmos", "satellite", "singularity"])
VALID_PATTERN_CLASSES = frozenset([
    "flat", "stripes", "squares", "hexagons", "oscillons", "quasipattern", "mixed", "disordered"
])
VALID_GEOMETRIES = frozenset([
    "circular_plate", "rectangular_plate", "square_plate", "membrane", "simulation"
])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class CymaticsValidationResult:
    def __init__(self, cert_id: str, cert_type: str) -> None:
        self.cert_id = cert_id
        self.cert_type = cert_type
        self.ok: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks_passed: int = 0
        self.checks_total: int = 0
        self.hash: str = ""

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def label(self) -> str:
        if not self.ok:
            return "FAIL"
        if self.warnings:
            return "PASS_WITH_WARNINGS"
        return "PASS"

    def report(self) -> str:
        lines = [
            f"  cert_id    : {self.cert_id}",
            f"  cert_type  : {self.cert_type}",
            f"  result     : {self.label}",
            f"  hash       : {self.hash[:16]}...",
            f"  checks     : {self.checks_passed}/{self.checks_total}",
        ]
        if self.errors:
            lines.append("  errors:")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    ~ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MODE CERT VALIDATOR
# ---------------------------------------------------------------------------

def validate_mode_cert(cert: Dict[str, Any]) -> CymaticsValidationResult:
    """Validate QA_CYMATIC_MODE_CERT.v1"""
    cid = cert.get("certificate_id", "unknown")
    out = CymaticsValidationResult(cid, "mode_witness")
    out.hash = cert_hash(cert)

    # --- Schema ---
    if cert.get("schema_version") != "QA_CYMATICS_CERT.v1":
        out.fail(f"Bad schema_version: {cert.get('schema_version')!r}")
        return out

    if cert.get("cert_type") != "mode_witness":
        out.fail(f"Expected cert_type='mode_witness', got {cert.get('cert_type')!r}")
        return out

    # --- Setup ---
    setup = cert.get("setup", {})
    if setup.get("geometry") not in VALID_GEOMETRIES:
        out.fail(f"Unknown geometry: {setup.get('geometry')!r}")

    # --- Drive conditions ---
    drive = cert.get("drive_conditions", {})
    freq = drive.get("frequency_hz")
    if freq is None or freq <= 0:
        out.fail(f"drive_conditions.frequency_hz must be > 0, got {freq!r}")

    # --- Mode witness ---
    mw = cert.get("mode_witness", {})
    m = mw.get("nodal_diameter_count")
    n = mw.get("nodal_circle_count")
    if m is None or m < 0:
        out.fail(f"nodal_diameter_count must be >= 0, got {m!r}")
    if n is None or n < 0:
        out.fail(f"nodal_circle_count must be >= 0, got {n!r}")

    obs_freq = mw.get("observed_frequency_hz")
    expected_eigen = mw.get("expected_eigenfrequency_hz")
    tol = mw.get("frequency_tolerance_hz")
    # Prefer expected_eigenfrequency_hz as reference; fall back to drive frequency.
    ref_freq = expected_eigen if expected_eigen is not None else freq
    detected_off_resonance = False
    if obs_freq is not None and ref_freq is not None and tol is not None:
        delta = abs(obs_freq - ref_freq)
        if delta > tol:
            detected_off_resonance = True

    # --- QA mapping ---
    qa = cert.get("qa_mapping", {})
    b = qa.get("b")
    e = qa.get("e")
    d_comp = qa.get("d_computed")
    a_comp = qa.get("a_computed")
    qa_norm = qa.get("qa_norm")
    orbit = qa.get("orbit_family")

    if b is not None and e is not None:
        # Recompute d and a
        expected_d = b + e
        expected_a = b + 2 * e
        if d_comp != expected_d:
            out.fail(f"TUPLE_FORMULA_VIOLATION: d_computed={d_comp} != b+e={expected_d}")
        if a_comp != expected_a:
            out.fail(f"TUPLE_FORMULA_VIOLATION: a_computed={a_comp} != b+2e={expected_a}")

        # Recompute Q(√5) norm
        expected_norm = b * b + b * e - e * e
        if qa_norm != expected_norm:
            out.fail(f"ORBIT_CLASS_MISMATCH: qa_norm={qa_norm} != b²+be-e²={expected_norm}")

        # Classify orbit from norm
        if expected_norm is not None:
            norm_mod3 = expected_norm % 3
            if norm_mod3 == 0 and b == 0 and e == 0:
                expected_orbit = "singularity"
            elif norm_mod3 == 0:
                expected_orbit = "satellite"
            else:
                expected_orbit = "cosmos"
            if orbit != expected_orbit:
                out.warn(
                    f"orbit_family={orbit!r} but norm mod 3 = {norm_mod3} suggests {expected_orbit!r}"
                )

        # Check Chladni formula echo: a = m + 2n
        if m is not None and n is not None:
            chladni_index_value = m + 2 * n
            chladni_match = qa.get("chladni_formula_qa_match")
            actual_match = (a_comp == chladni_index_value)
            if chladni_match != actual_match:
                out.fail(
                    f"chladni_formula_qa_match={chladni_match} but a_computed({a_comp}) == m+2n({chladni_index_value}) is {actual_match}"
                )

    if orbit not in VALID_ORBIT_FAMILIES:
        out.fail(f"orbit_family {orbit!r} not in {VALID_ORBIT_FAMILIES}")

    # --- Validation checks consistency ---
    checks = cert.get("validation_checks", [])
    out.checks_total = len(checks)
    out.checks_passed = sum(1 for c in checks if c.get("passed"))

    # --- Fail ledger + result consistency ---
    fail_ledger = cert.get("fail_ledger", [])
    declared_result = cert.get("result")
    ledger_fail_types = {e.get("fail_type") for e in fail_ledger}

    for entry in fail_ledger:
        ft = entry.get("fail_type")
        if ft not in MODE_FAIL_TYPES:
            out.fail(f"Unknown fail_type in fail_ledger: {ft!r}")

    ledger_has_fails = len(fail_ledger) > 0
    if declared_result == "PASS" and ledger_has_fails:
        out.fail("result=PASS but fail_ledger is non-empty")
    if declared_result == "FAIL" and not ledger_has_fails:
        out.warn("result=FAIL but fail_ledger is empty — consider documenting the failure")

    # Consistency: recomputed OFF_RESONANCE vs declared result
    if detected_off_resonance:
        if declared_result == "PASS":
            obs_freq = mw.get("observed_frequency_hz")
            ref_freq2 = mw.get("expected_eigenfrequency_hz") or drive.get("frequency_hz")
            tol2 = mw.get("frequency_tolerance_hz")
            out.fail(
                f"OFF_RESONANCE: |observed({obs_freq}) - expected({ref_freq2})| > tolerance({tol2}), "
                f"but result=PASS — inconsistency"
            )
        else:
            if "OFF_RESONANCE" not in ledger_fail_types:
                out.warn(
                    "Recomputed OFF_RESONANCE but fail_ledger does not contain OFF_RESONANCE entry"
                )

    # Warn if cert has failed validation_checks but no documented failures
    failed_checks = [c for c in checks if not c.get("passed")]
    if failed_checks and declared_result == "PASS" and not ledger_has_fails and out.ok:
        out.warn(
            f"{len(failed_checks)} validation_check(s) are passed=false but result=PASS and fail_ledger empty"
        )

    return out


# ---------------------------------------------------------------------------
# FARADAY REACHABILITY CERT VALIDATOR
# ---------------------------------------------------------------------------

def validate_faraday_cert(cert: Dict[str, Any]) -> CymaticsValidationResult:
    """Validate QA_FARADAY_REACHABILITY_CERT.v1"""
    cid = cert.get("certificate_id", "unknown")
    out = CymaticsValidationResult(cid, "faraday_reachability")
    out.hash = cert_hash(cert)

    # --- Schema ---
    if cert.get("schema_version") != "QA_CYMATICS_CERT.v1":
        out.fail(f"Bad schema_version: {cert.get('schema_version')!r}")
        return out

    if cert.get("cert_type") != "faraday_reachability":
        out.fail(f"Expected cert_type='faraday_reachability', got {cert.get('cert_type')!r}")
        return out

    # --- Pattern state ---
    ps = cert.get("pattern_state", {})
    current_class = ps.get("current_class")
    if current_class not in VALID_PATTERN_CLASSES:
        out.fail(f"pattern_state.current_class {current_class!r} not in known pattern classes")

    if ps.get("faraday_subharmonic_ratio") != "1/2":
        out.fail(
            f"faraday_subharmonic_ratio must be '1/2', got {ps.get('faraday_subharmonic_ratio')!r}"
        )

    # --- Pattern basin graph ---
    graph = cert.get("pattern_basin_graph", [])
    legal_moves: Dict[str, bool] = {}  # control_move -> legal
    for edge in graph:
        move = edge.get("control_move", "")
        legal_moves[move] = edge.get("legal", False)
        fp = edge.get("from_pattern")
        tp = edge.get("to_pattern")
        if fp not in VALID_PATTERN_CLASSES:
            out.warn(f"Unknown from_pattern in basin graph: {fp!r}")
        if tp not in VALID_PATTERN_CLASSES:
            out.warn(f"Unknown to_pattern in basin graph: {tp!r}")

    # --- Reachability ---
    reach = cert.get("reachability", {})
    path = reach.get("witnessed_path", [])
    path_k = reach.get("path_length_k")
    return_in_k = reach.get("return_in_k")
    return_path = reach.get("return_path")

    if len(path) != path_k:
        out.fail(
            f"witnessed_path length {len(path)} != path_length_k {path_k}"
        )

    # Each move in path must be legal
    for move in path:
        if move in legal_moves:
            if not legal_moves[move]:
                out.fail(f"Witnessed path contains illegal move: {move!r}")
        else:
            out.warn(f"Witnessed path move {move!r} not found in pattern_basin_graph edges")

    # Return consistency
    if return_in_k and return_path is None:
        out.fail("return_in_k=true but return_path is null")
    if not return_in_k and return_path is not None:
        out.warn("return_in_k=false but return_path is non-null — inconsistency")

    # --- Obstruction classes ---
    obstructions = cert.get("obstruction_classes", [])
    for obs in obstructions:
        ot = obs.get("obstruction_type")
        valid_obstruction_types = frozenset([
            "NONLINEAR_ESCAPE", "DAMPING_COLLAPSE", "BOUNDARY_MISMATCH",
            "SYMMETRY_FORBIDDEN", "AMPLITUDE_BELOW_THRESHOLD", "FREQUENCY_OUT_OF_BAND"
        ])
        if ot not in valid_obstruction_types:
            out.fail(f"Unknown obstruction_type: {ot!r}")

    # --- QA mapping ---
    qa = cert.get("qa_mapping", {})
    drive_maps_to = qa.get("drive_frequency_maps_to")
    if drive_maps_to not in VALID_ORBIT_FAMILIES:
        out.fail(f"drive_frequency_maps_to {drive_maps_to!r} not in {VALID_ORBIT_FAMILIES}")

    pattern_map = qa.get("pattern_class_to_orbit_family", {})
    if "flat" in pattern_map and pattern_map["flat"] != "singularity":
        out.warn(
            f"flat pattern maps to {pattern_map['flat']!r}, expected 'singularity' (zero-dimensional fixed point)"
        )

    three_regime = qa.get("three_regime_correspondence", {})
    for key in ["flat", "stripes_squares", "hexagons_quasipatterns"]:
        if key not in three_regime:
            out.warn(f"three_regime_correspondence missing key: {key!r}")

    # --- Validation checks ---
    checks = cert.get("validation_checks", [])
    out.checks_total = len(checks)
    out.checks_passed = sum(1 for c in checks if c.get("passed"))

    # --- Fail ledger ---
    fail_ledger = cert.get("fail_ledger", [])
    declared_result = cert.get("result")
    for entry in fail_ledger:
        ft = entry.get("fail_type")
        if ft not in FARADAY_FAIL_TYPES:
            out.fail(f"Unknown fail_type in fail_ledger: {ft!r}")

    ledger_has_fails = len(fail_ledger) > 0
    if declared_result == "PASS" and ledger_has_fails:
        out.fail("result=PASS but fail_ledger is non-empty")
    if declared_result == "FAIL" and not ledger_has_fails:
        out.warn("result=FAIL but fail_ledger is empty — consider documenting the failure")

    return out


# ---------------------------------------------------------------------------
# CONTROL CERT HELPERS
# ---------------------------------------------------------------------------

def _edge_exists(control_graph: Optional[Dict[str, Any]], from_node: str, move: str, to_node: str) -> bool:
    """Return True if (from_node, move, to_node) is a legal edge in control_graph, or if no graph provided."""
    if not control_graph:
        return True  # No graph = no strict edge checking
    for edge in control_graph.get("edges", []):
        if edge.get("from") == from_node and edge.get("move") == move and edge.get("to") == to_node:
            return True
    return False


def _parse_expected_to(step: Dict[str, Any]) -> Optional[str]:
    """Extract destination from expected_transition string 'from -> to'."""
    exp = step.get("expected_transition", "")
    if "->" in exp:
        return exp.split("->", 1)[1].strip()
    return None


def _parse_expected_from(step: Dict[str, Any]) -> Optional[str]:
    """Extract source from expected_transition string 'from -> to'."""
    exp = step.get("expected_transition", "")
    if "->" in exp:
        return exp.split("->", 1)[0].strip()
    return None


# ---------------------------------------------------------------------------
# CONTROL CERT VALIDATOR
# ---------------------------------------------------------------------------

def validate_control_cert(cert: Dict[str, Any]) -> CymaticsValidationResult:
    """Validate QA_CYMATIC_CONTROL_CERT.v1"""
    cid = cert.get("certificate_id", "unknown")
    out = CymaticsValidationResult(cid, "cymatic_control")
    out.hash = cert_hash(cert)

    # --- Schema ---
    if cert.get("schema_version") != "QA_CYMATIC_CONTROL_CERT.v1":
        out.fail(f"Bad schema_version: {cert.get('schema_version')!r}")
        return out
    if cert.get("cert_type") != "cymatic_control":
        out.fail(f"Expected cert_type='cymatic_control', got {cert.get('cert_type')!r}")
        return out

    initial  = cert.get("initial_state", {})
    target   = cert.get("target_spec", {})
    seq      = cert.get("generator_sequence", [])
    obs      = cert.get("observed_final_state", {})
    qa       = cert.get("qa_mapping", {})
    checks   = cert.get("validation_checks", [])
    fail_ledger = cert.get("fail_ledger", [])
    declared_result = cert.get("result")
    control_graph = cert.get("control_graph")  # optional

    ledger_fail_types = {e.get("fail_type") for e in fail_ledger}

    # Fail ledger type check
    for entry in fail_ledger:
        ft = entry.get("fail_type")
        if ft not in CONTROL_FAIL_TYPES:
            out.fail(f"Unknown fail_type in fail_ledger: {ft!r}")

    initial_class  = initial.get("pattern_class")
    target_class   = target.get("target_pattern_class")
    target_sym     = target.get("target_symmetry_group")
    final_class    = obs.get("pattern_class")
    final_sym      = obs.get("symmetry_group")
    path_length_k  = obs.get("path_length_k")
    max_k          = target.get("max_path_length_k")
    return_in_k    = obs.get("return_in_k")
    return_path    = obs.get("return_path") or []
    final_orbit    = qa.get("final_orbit_family")
    target_orbit   = qa.get("target_orbit_family")
    initial_orbit  = qa.get("initial_orbit_family")

    detected_fails: set = set()

    # C1: path_length_k == len(generator_sequence)
    if path_length_k != len(seq):
        detected_fails.add("PATH_LENGTH_EXCEEDED")
        out.warn(
            f"C1: observed_final_state.path_length_k={path_length_k} != "
            f"len(generator_sequence)={len(seq)}"
        )

    # C2: path_length_k <= max_path_length_k
    if max_k is not None and path_length_k is not None and path_length_k > max_k:
        detected_fails.add("PATH_LENGTH_EXCEEDED")

    # C3: final pattern == target pattern
    if final_class != target_class:
        detected_fails.add("GOAL_NOT_REACHED")

    # C4: final symmetry == target symmetry
    if final_sym != target_sym:
        detected_fails.add("GOAL_NOT_REACHED")

    # C5: final pattern class recognized
    if final_class not in CONTROL_PATTERN_TO_ORBIT:
        detected_fails.add("PATTERN_CLASS_UNRECOGNIZED")

    # C6: final orbit consistent with final pattern class
    expected_final_orbit = CONTROL_PATTERN_TO_ORBIT.get(final_class)
    if expected_final_orbit is not None and final_orbit != expected_final_orbit:
        detected_fails.add("ORBIT_CLASS_MISMATCH")

    # C7: target orbit consistent with target pattern class
    expected_target_orbit = CONTROL_PATTERN_TO_ORBIT.get(target_class)
    if expected_target_orbit is not None and target_orbit != expected_target_orbit:
        detected_fails.add("ORBIT_CLASS_MISMATCH")

    # C8: initial orbit consistent with initial pattern class
    expected_initial_orbit = CONTROL_PATTERN_TO_ORBIT.get(initial_class)
    if expected_initial_orbit is not None and initial_orbit != expected_initial_orbit:
        detected_fails.add("ORBIT_CLASS_MISMATCH")

    # C9: all transitions legal in control_graph (when provided)
    if control_graph:
        current = initial_class
        for step in seq:
            move = step.get("move")
            to_node = _parse_expected_to(step)
            if to_node is None:
                continue
            if not _edge_exists(control_graph, current, move, to_node):
                detected_fails.add("ILLEGAL_TRANSITION")
                break
            current = to_node

    # C10: return path consistency
    if return_in_k:
        if not return_path:
            detected_fails.add("RETURN_PATH_NOT_FOUND")
        elif return_path[-1].get("to") != initial_class:
            detected_fails.add("RETURN_PATH_NOT_FOUND")

    # C11: disordered final state = NONLINEAR_ESCAPE
    if final_class == "disordered":
        detected_fails.add("NONLINEAR_ESCAPE")

    # --- Validation checks accounting ---
    out.checks_total = len(checks)
    out.checks_passed = sum(1 for c in checks if c.get("passed"))

    # --- Result consistency model (same as mode/faraday certs) ---
    ledger_has_fails = len(fail_ledger) > 0

    if declared_result == "PASS" and ledger_has_fails:
        out.fail("result=PASS but fail_ledger is non-empty")

    if declared_result == "FAIL" and not ledger_has_fails:
        out.warn("result=FAIL but fail_ledger is empty — consider documenting the failure")

    if detected_fails:
        if declared_result == "PASS":
            out.fail(
                f"Recomputed control failures {sorted(detected_fails)} "
                f"but result=PASS — inconsistency"
            )
        else:
            # FAIL cert: check all detected failures are in fail_ledger
            missing = sorted(detected_fails - ledger_fail_types)
            if missing:
                out.warn(
                    "Recomputed control failures missing from fail_ledger: "
                    + ", ".join(missing)
                )
    else:
        if declared_result == "FAIL":
            out.warn("Declared FAIL but validator recomputed no control failure — verify cert")

    return out


# ---------------------------------------------------------------------------
# PLANNER CERT HELPERS
# ---------------------------------------------------------------------------

def _find_cert_by_id(cert_id: str, search_dir: Path) -> Optional[Dict[str, Any]]:
    """Search fixtures directory for a cert whose certificate_id matches cert_id.
    Returns the parsed cert dict, or None if not found.
    """
    try:
        for fp in sorted(search_dir.glob("*.json")):
            try:
                with open(fp) as f:
                    cert = json.load(f)
                if cert.get("certificate_id") == cert_id:
                    return cert
            except Exception:
                pass
    except Exception:
        pass
    return None


def _bfs_shortest(
    control_graph: Optional[Dict[str, Any]],
    start: str,
    goal: str,
    max_depth: int,
) -> Optional[int]:
    """BFS over control_graph edges.
    Returns the shortest path length from start to goal within max_depth,
    or None if the goal is not reachable within that bound.
    """
    if control_graph is None:
        return None
    if start == goal:
        return 0
    queue: deque = deque([(start, 0)])
    visited = {start}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in control_graph.get("edges", []):
            if edge.get("from") == node:
                neighbor = edge.get("to")
                new_depth = depth + 1
                if neighbor == goal:
                    return new_depth
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_depth))
    return None


# ---------------------------------------------------------------------------
# PLANNER CERT VALIDATOR
# ---------------------------------------------------------------------------

def validate_planner_cert(cert: Dict[str, Any]) -> CymaticsValidationResult:
    """Validate QA_CYMATIC_PLANNER_CERT.v1"""
    cid = cert.get("certificate_id", "unknown")
    out = CymaticsValidationResult(cid, "cymatic_planner")
    out.hash = cert_hash(cert)

    # --- Schema ---
    if cert.get("schema_version") != "QA_CYMATIC_PLANNER_CERT.v1":
        out.fail(f"Bad schema_version: {cert.get('schema_version')!r}")
        return out
    if cert.get("cert_type") != "cymatic_planner":
        out.fail(f"Expected cert_type='cymatic_planner', got {cert.get('cert_type')!r}")
        return out

    prob          = cert.get("planning_problem", {})
    control_graph = cert.get("control_graph")
    planner_run   = cert.get("planner_run", {})
    plan_witness  = cert.get("plan_witness", {})
    obs           = cert.get("observed_final_state", {})
    qa            = cert.get("qa_mapping", {})
    checks        = cert.get("validation_checks", [])
    fail_ledger   = cert.get("fail_ledger", [])
    declared_result = cert.get("result")

    initial_class = prob.get("initial_pattern_class")
    target_class  = prob.get("target_pattern_class")
    max_depth_k   = prob.get("max_depth_k", 0)
    opt_goal      = prob.get("optimization_goal")

    algorithm      = planner_run.get("algorithm", "")
    searched_nodes = planner_run.get("searched_nodes", 0)
    found_plan     = planner_run.get("found_plan", False)

    path_length_k = plan_witness.get("path_length_k", 0)
    steps         = plan_witness.get("steps", [])

    final_class  = obs.get("pattern_class")
    final_orbit  = obs.get("orbit_family")
    target_orbit = qa.get("target_orbit_family")

    ledger_fail_types = {e.get("fail_type") for e in fail_ledger}
    detected_fails: set = set()

    # Fail ledger type check
    for entry in fail_ledger:
        ft = entry.get("fail_type")
        if ft not in PLANNER_FAIL_TYPES:
            out.fail(f"Unknown fail_type in fail_ledger: {ft!r}")

    # P1: algorithm recognized
    if algorithm not in RECOGNIZED_ALGORITHMS:
        detected_fails.add("SEARCH_INCONSISTENCY")
        out.warn(f"P1: algorithm {algorithm!r} not in recognized set {sorted(RECOGNIZED_ALGORITHMS)}")

    # P2: path_length_k <= max_depth_k (only meaningful when plan was found)
    if found_plan and path_length_k > max_depth_k:
        detected_fails.add("NO_PLAN_WITHIN_BOUND")

    # P3: every plan step is a legal graph edge
    if control_graph and steps:
        for step in steps:
            f_node = step.get("from")
            move   = step.get("move")
            t_node = step.get("to")
            if not _edge_exists(control_graph, f_node, move, t_node):
                detected_fails.add("ILLEGAL_TRANSITION")
                break

    # P4: final class matches target
    if final_class != target_class:
        detected_fails.add("GOAL_NOT_REACHABLE")

    # P5: qa_mapping.final_orbit_family internally consistent with CONTROL_PATTERN_TO_ORBIT
    expected_final_orbit = CONTROL_PATTERN_TO_ORBIT.get(final_class)
    if expected_final_orbit is not None and final_orbit != expected_final_orbit:
        detected_fails.add("ORBIT_CLASS_MISMATCH")

    # P6: searched_nodes plausible (>= path_length_k+1 when plan found, else >= 1)
    min_nodes = (path_length_k + 1) if found_plan else 1
    if searched_nodes < min_nodes:
        detected_fails.add("SEARCH_INCONSISTENCY")

    # P7: shortest-path claim consistency (BFS recomputation)
    if control_graph and opt_goal == "shortest_path":
        recomputed = _bfs_shortest(control_graph, initial_class, target_class, max_depth_k)
        if found_plan:
            if recomputed is not None and path_length_k > recomputed:
                detected_fails.add("NONMINIMAL_PLAN")
        else:
            # Planner said no plan; BFS should also find none within bound
            if recomputed is not None:
                detected_fails.add("SEARCH_INCONSISTENCY")

    # P8: no disordered state reached in plan steps
    for step in steps:
        if step.get("to") == "disordered":
            detected_fails.add("NONLINEAR_ESCAPE")
            break

    # P9: if no plan found, that is NO_PLAN_WITHIN_BOUND
    if not found_plan:
        detected_fails.add("NO_PLAN_WITHIN_BOUND")

    # P11: minimality witness consistency (optional; only checked when present)
    is_shortest      = plan_witness.get("is_shortest", False)
    min_wit          = plan_witness.get("minimality_witness")

    if is_shortest and opt_goal == "shortest_path" and min_wit is not None:
        proved         = min_wit.get("proved_no_path_shorter_than")
        excluded       = min_wit.get("excluded_shorter_lengths", [])
        frontier_sizes = min_wit.get("frontier_sizes", [])
        # proved_no_path_shorter_than must equal path_length_k
        if proved != path_length_k:
            detected_fails.add("MINIMALITY_WITNESS_INCOMPLETE")
        # excluded must cover exactly [0, 1, ..., path_length_k-1]
        if sorted(excluded) != list(range(path_length_k)):
            detected_fails.add("MINIMALITY_WITNESS_INCOMPLETE")
        # frontier_sizes length must equal path_length_k (one entry per excluded depth)
        if len(frontier_sizes) != path_length_k:
            detected_fails.add("MINIMALITY_WITNESS_INCOMPLETE")

    # P10: replay consistency — optional; present only when compiled_control_certificate_id is given
    compiled_id   = cert.get("compiled_control_certificate_id")
    compiled_hash = cert.get("compiled_control_witness_hash")
    replay_consistent = cert.get("replay_consistent")

    if compiled_id is not None:
        here_dir   = Path(__file__).parent
        fixtures_d = here_dir / "fixtures"
        ref_cert   = _find_cert_by_id(compiled_id, fixtures_d)

        if ref_cert is None:
            detected_fails.add("COMPILED_CERT_MISSING")
        else:
            # Hash integrity check
            actual_hash = cert_hash(ref_cert)
            if compiled_hash is not None and compiled_hash != actual_hash:
                detected_fails.add("REPLAY_INCONSISTENCY")

            # Semantic consistency checks (use the found cert regardless of hash)
            ref_target  = ref_cert.get("target_spec",           {}).get("target_pattern_class")
            ref_initial = ref_cert.get("initial_state",         {}).get("pattern_class")
            ref_final   = ref_cert.get("observed_final_state",  {}).get("pattern_class")
            ref_seq     = ref_cert.get("generator_sequence",    [])

            if ref_target != target_class:
                detected_fails.add("PLAN_CONTROL_MISMATCH")
            if ref_initial != initial_class:
                detected_fails.add("PLAN_CONTROL_MISMATCH")
            if ref_final != target_class:
                detected_fails.add("PLAN_CONTROL_MISMATCH")
            if len(steps) != len(ref_seq):
                detected_fails.add("PLAN_CONTROL_MISMATCH")
            else:
                for ps, cs in zip(steps, ref_seq):
                    if ps.get("move") != cs.get("move"):
                        detected_fails.add("PLAN_CONTROL_MISMATCH")
                        break

            # replay_consistent flag: if declared, verify it matches recomputed state
            recomputed_consistent = not bool(
                detected_fails & {"REPLAY_INCONSISTENCY", "PLAN_CONTROL_MISMATCH", "COMPILED_CERT_MISSING"}
            )
            if replay_consistent is not None and replay_consistent != recomputed_consistent:
                detected_fails.add("REPLAY_INCONSISTENCY")

    # --- Validation checks accounting ---
    out.checks_total  = len(checks)
    out.checks_passed = sum(1 for c in checks if c.get("passed"))

    # --- Result consistency model (same as other cert families) ---
    ledger_has_fails = len(fail_ledger) > 0

    if declared_result == "PASS" and ledger_has_fails:
        out.fail("result=PASS but fail_ledger is non-empty")

    if declared_result == "FAIL" and not ledger_has_fails:
        out.warn("result=FAIL but fail_ledger is empty — consider documenting the failure")

    if detected_fails:
        if declared_result == "PASS":
            out.fail(
                f"Recomputed planner failures {sorted(detected_fails)} "
                f"but result=PASS — inconsistency"
            )
        else:
            missing = sorted(detected_fails - ledger_fail_types)
            if missing:
                out.warn(
                    "Recomputed planner failures missing from fail_ledger: "
                    + ", ".join(missing)
                )
    else:
        if declared_result == "FAIL":
            out.warn("Declared FAIL but validator recomputed no planner failure — verify cert")

    return out


# ---------------------------------------------------------------------------
# File dispatch
# ---------------------------------------------------------------------------

def validate_file(path: str, cert_type: Optional[str] = None) -> CymaticsValidationResult:
    with open(path) as f:
        cert = json.load(f)

    # Auto-detect type from cert_type field if not specified
    detected = cert.get("cert_type")
    chosen = cert_type or detected

    if chosen == "mode_witness":
        return validate_mode_cert(cert)
    elif chosen == "faraday_reachability":
        return validate_faraday_cert(cert)
    elif chosen == "cymatic_control":
        return validate_control_cert(cert)
    elif chosen == "cymatic_planner":
        return validate_planner_cert(cert)
    else:
        result = CymaticsValidationResult(cert.get("certificate_id", "unknown"), "unknown")
        result.fail(
            f"Cannot determine cert type: cert_type field = {detected!r}, "
            f"--mode/--faraday/--control/--planner not specified"
        )
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="QA Cymatics certificate validator")
    parser.add_argument("--mode",      metavar="FILE", help="Validate a mode_witness cert file")
    parser.add_argument("--faraday",   metavar="FILE", help="Validate a faraday_reachability cert file")
    parser.add_argument("--control",   metavar="FILE", help="Validate a cymatic_control cert file")
    parser.add_argument("--planner",   metavar="FILE", help="Validate a cymatic_planner cert file")
    parser.add_argument("--all",       action="store_true", help="Validate all fixtures in fixtures/")
    parser.add_argument("--demo",      action="store_true", help="Run all fixtures and print results")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="Run all fixtures; emit JSON {ok, passed, failed} to stdout")
    args = parser.parse_args()

    here = Path(__file__).parent
    fixtures_dir = here / "fixtures"

    results: List[Tuple[str, CymaticsValidationResult]] = []

    if args.mode:
        r = validate_file(args.mode, "mode_witness")
        results.append((args.mode, r))

    if args.faraday:
        r = validate_file(args.faraday, "faraday_reachability")
        results.append((args.faraday, r))

    if args.control:
        r = validate_file(args.control, "cymatic_control")
        results.append((args.control, r))

    if args.planner:
        r = validate_file(args.planner, "cymatic_planner")
        results.append((args.planner, r))

    if args.all or args.demo or args.self_test:
        for fp in sorted(fixtures_dir.glob("*.json")):
            r = validate_file(str(fp))
            results.append((fp.name, r))

    if not results:
        parser.print_help()
        return 0

    passed = sum(1 for _, r in results if r.ok)
    failed = sum(1 for _, r in results if not r.ok)

    if args.self_test:
        # JSON output for meta-validator consumption
        print(json.dumps({
            "ok": failed == 0,
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "details": [
                {"name": name, "result": r.label, "cert_type": r.cert_type}
                for name, r in results
            ],
        }))
        return 0 if failed == 0 else 1

    print()
    for name, r in results:
        print(f"[{r.label}] {name}")
        print(r.report())
        print()

    print(f"{'='*60}")
    print(f"Total: {passed+failed}  PASS: {passed}  FAIL: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
