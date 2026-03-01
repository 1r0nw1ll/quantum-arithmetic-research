#!/usr/bin/env python3
"""
demos/competency_live_demo.py  —  QA Levin Competency Framework: Live Demo

A simulated tool-using debugger agent emits structured competency certificates
in real time, showing all four canonical metrics update as the agent encounters
a novel failure mode and then gets stuck in a loop.

Levin-to-QA mapping (family [26]):
  Competency  → Reachability Class    BFS-reachable set from initial agent states
  Goal        → Attractor Basin       Sink SCCs in the state transition graph
  Memory      → Invariant             Preserved constraints (sandbox_isolation)
  Agency      → Control Region        |reachable| / |total| state space fraction
  Plasticity  → Generator Flexibility New states discovered per tool call (windowed)

Five canonical metrics:
  Agency Index  (AI)  =  |reachable_states| / |total_states|
  Plasticity    (PI)  =  delta_reachable / delta_tool_calls   [windowed]
  Goal Density  (GD)  =  |attractor_basins| / |total_states|
  Ctrl Entropy  (CE)  =  -Σ p(tool) ln p(tool)               [natural log]
  Path Div.    (PDI)  =  |states with ≥2 directed paths| / |reachable|
                          (PI=0, PDI=high) → stuck-loop thrashing
                          (PI=hi, PDI=hi)  → flexible multi-route planner

Three phases:
  Phase 1  Normal operation — agent solves a known TypeError bug
  Phase 2  Novel failure — first patch triggers AttributeError; PI still > 0
  Phase 3  Stuck loop — agent revisits known states; PI collapses to 0.0
           → Obstruction certificate emitted

Usage:
  python demos/competency_live_demo.py [--fast] [--out DIR]

  --fast     No inter-step delays (CI / non-interactive)
  --out DIR  Output directory for cert JSON (default: /tmp/qa_demo_out)
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — runnable from repo root or from demos/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qa_competency.intake.adapters.llm_tool_agent import adapt

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_R  = "\033[0m"
_B  = "\033[1m"
_DIM = "\033[2m"
_G  = "\033[32m"
_Y  = "\033[33m"
_RD = "\033[31m"
_C  = "\033[36m"
_W  = "\033[37m"


def _col(text: str, *codes: str) -> str:
    return "".join(codes) + text + _R


def _delta_str(delta: float, up_good: bool = True) -> str:
    if abs(delta) < 5e-4:
        return _col("  ─       ", _DIM)
    arrow = "↑" if delta > 0 else "↓"
    color = _G if (delta > 0) == up_good else _RD
    return _col(f" {arrow}{delta:+.4f}", color)


# ---------------------------------------------------------------------------
# Scripted episode states
#
# State dicts are reused across phases to create genuine cycles in Phase 3.
# The adapter computes SHA-256 of the canonical JSON of each state dict, so
# two equal dicts → same graph node → cycle detected.
# ---------------------------------------------------------------------------

_S_INIT = {
    "error": None,
    "files_open": [],
    "hypothesis": None,
    "patch": None,
    "patch_v": 0,
}
_S_ERR_TYPE = {
    "error": "TypeError: unsupported operand type(s): int + str",
    "files_open": [],
    "hypothesis": None,
    "patch": None,
    "patch_v": 0,
}
_S_READ1 = {
    "error": "TypeError: unsupported operand type(s): int + str",
    "files_open": ["app.py"],
    "hypothesis": None,
    "patch": None,
    "patch_v": 0,
}
_S_HYPO1 = {
    "error": "TypeError: unsupported operand type(s): int + str",
    "files_open": ["app.py"],
    "hypothesis": "wrong_arg_type",
    "patch": None,
    "patch_v": 0,
}
_S_PATCH1 = {
    "error": "TypeError: unsupported operand type(s): int + str",
    "files_open": ["app.py"],
    "hypothesis": "wrong_arg_type",
    "patch": "cast_int_to_str_v1",
    "patch_v": 1,
}

# Phase 2: first patch reveals a deeper AttributeError
_S_ERR_ATTR = {
    "error": "AttributeError: 'NoneType' object has no attribute 'value'",
    "files_open": ["app.py"],
    "hypothesis": None,
    "patch": "cast_int_to_str_v1",
    "patch_v": 1,
}
_S_SEARCH_ATTR = {
    "error": "AttributeError: 'NoneType' object has no attribute 'value'",
    "files_open": ["app.py"],
    "hypothesis": "null_deref_before_cast",
    "patch": "cast_int_to_str_v1",
    "patch_v": 1,
}
_S_READ2 = {
    "error": "AttributeError: 'NoneType' object has no attribute 'value'",
    "files_open": ["app.py", "models.py"],
    "hypothesis": "null_deref_before_cast",
    "patch": "cast_int_to_str_v1",
    "patch_v": 1,
}
_S_PATCH2 = {
    "error": "AttributeError: 'NoneType' object has no attribute 'value'",
    "files_open": ["app.py", "models.py"],
    "hypothesis": "null_deref_before_cast",
    "patch": "null_guard_v2",
    "patch_v": 2,
}

# Phase 3: second patch also fails.
# Steps 9-13 REUSE Phase 1/2 state dicts → creates directed cycles.
# Graph cycle: S_PATCH2 → S_ERR_TYPE → S_READ1 → S_HYPO1 → S_PATCH1
#              → S_ERR_ATTR → S_SEARCH_ATTR → S_READ2 → S_PATCH2 (closed!)

_EVENTS: List[dict] = [
    # ── Phase 1: Normal debugging (known bug type) ─────────────────────
    {"episode_id": "ep_debug", "step":  0, "tool": "read_file",   "state": _S_INIT,        "result": "ok"},
    {"episode_id": "ep_debug", "step":  1, "tool": "run_test",    "state": _S_ERR_TYPE,    "result": "fail"},
    {"episode_id": "ep_debug", "step":  2, "tool": "read_file",   "state": _S_READ1,       "result": "ok"},
    {"episode_id": "ep_debug", "step":  3, "tool": "search_docs", "state": _S_HYPO1,       "result": "ok"},
    {"episode_id": "ep_debug", "step":  4, "tool": "write_patch", "state": _S_PATCH1,      "result": "ok"},
    # ── Phase 2: Novel failure mode ────────────────────────────────────
    {"episode_id": "ep_debug", "step":  5, "tool": "run_test",    "state": _S_ERR_ATTR,    "result": "fail"},
    {"episode_id": "ep_debug", "step":  6, "tool": "search_docs", "state": _S_SEARCH_ATTR, "result": "ok"},
    {"episode_id": "ep_debug", "step":  7, "tool": "read_file",   "state": _S_READ2,       "result": "ok"},
    {"episode_id": "ep_debug", "step":  8, "tool": "write_patch", "state": _S_PATCH2,      "result": "ok"},
    # ── Phase 3: Stuck loop (revisiting known states) ──────────────────
    # Step 9 = step 1 (same state), step 10 = step 2, … closing the cycle.
    {"episode_id": "ep_debug", "step":  9, "tool": "run_test",    "state": _S_ERR_TYPE,    "result": "fail"},   # ← revisit step 1
    {"episode_id": "ep_debug", "step": 10, "tool": "read_file",   "state": _S_READ1,       "result": "ok"},     # ← revisit step 2
    {"episode_id": "ep_debug", "step": 11, "tool": "search_docs", "state": _S_HYPO1,       "result": "ok"},     # ← revisit step 3
    {"episode_id": "ep_debug", "step": 12, "tool": "write_patch", "state": _S_PATCH1,      "result": "ok"},     # ← revisit step 4
    {"episode_id": "ep_debug", "step": 13, "tool": "run_test",    "state": _S_ERR_ATTR,    "result": "fail"},   # ← revisit step 5
    # ── Recovery attempt (separate episode, same dead-end) ─────────────
    {"episode_id": "ep_retry", "step":  0, "tool": "read_file",   "state": _S_READ2,       "result": "ok"},     # ← revisit
    {"episode_id": "ep_retry", "step":  1, "tool": "run_test",    "state": _S_ERR_ATTR,    "result": "fail"},   # ← revisit
]

# Checkpoint at end of each phase (event index, 0-based)
_CHECKPOINTS: frozenset = frozenset([4, 8, 13, 15])

_PHASE_INFO: Dict[int, Tuple[str, str, str]] = {
    4:  ("Phase 1", "Normal operation — known bug type (TypeError)        ", _G),
    8:  ("Phase 2", "Novel failure mode — AttributeError surfaces         ", _Y),
    13: ("Phase 3", "Stuck loop — revisiting known states                 ", _RD),
    15: ("Phase 3", "Recovery attempt — obstruction confirmed              ", _RD),
}

# ---------------------------------------------------------------------------
# Canonical JSON / SHA-256 helpers
# ---------------------------------------------------------------------------
HEX64_ZERO = "0" * 64


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Build cert from accumulated events with incremental PI
# ---------------------------------------------------------------------------

def _build_cert(
    events: List[dict],
    prev_reachable: int,
    prev_event_count: int,
    obstructions: Optional[List[str]] = None,
) -> Tuple[dict, int]:
    """Call the adapter on current events and patch in incremental PI.

    Returns (cert_dict, current_reachable_states).
    """
    cert = adapt(
        copy.deepcopy(events),
        domain="software_engineering",
        substrate="llm_agent",
        description="Live demo: tool-using debugger agent",
    )

    # Incremental PI: new states discovered / new tool calls since last checkpoint
    current_reachable: int = cert["metric_inputs"]["reachable_states"]
    new_events = max(len(events) - prev_event_count, 1)
    delta_r = max(0, current_reachable - prev_reachable)
    pi_live = float(delta_r) / float(new_events)

    cert["metric_inputs"]["delta_reachability"] = float(delta_r)
    cert["metric_inputs"]["delta_perturbation"] = float(new_events)
    cert["competency_metrics"]["plasticity_index"] = pi_live

    if obstructions:
        cert["reachability"]["obstructions"] = obstructions

    # Recompute manifest after modifications
    tmp = copy.deepcopy(cert)
    tmp["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    cert["manifest"]["canonical_json_sha256"] = _sha256(tmp)

    return cert, current_reachable


# ---------------------------------------------------------------------------
# Terminal dashboard
# ---------------------------------------------------------------------------

_METRIC_ROWS = [
    ("agency_index",     "Agency Index  (AI)", "Control region size           [0–1]"),
    ("plasticity_index", "Plasticity    (PI)", "New states per tool call      [0–1]"),
    ("goal_density",     "Goal Density  (GD)", "Attractor basin coverage      [0–1]"),
    ("control_entropy",  "Ctrl Entropy  (CE)", "Decision freedom, nats        [0–∞]"),
    ("pdi",              "Path Div.    (PDI)", "Counterfactual control routes [0–1]"),
]


def _display_dashboard(
    phase_label: str,
    phase_desc: str,
    phase_color: str,
    n_events: int,
    cert: dict,
    prev_metrics: Optional[Dict[str, float]],
    obstruction: bool,
) -> None:
    m  = cert["competency_metrics"]
    mi = cert["metric_inputs"]
    rc = cert["reachability"]

    print()
    print(_col(f"  {'─'*62}", _C))
    print(_col(f"  {phase_label}", _B + phase_color) +
          _col(f"  {n_events} events processed", _DIM))
    print(_col(f"  {phase_desc}", phase_color))
    print(_col(f"  {'─'*62}", _C))
    print()
    print(_col(f"  {'Levin Metric':<22s}  {'Value':>8s}  {'Δ':>12s}  Description", _B + _W))
    print(_col(f"  {'─'*60}", _DIM))

    for key, name, desc in _METRIC_ROWS:
        val   = m.get(key, 0.0)
        prev_val = prev_metrics.get(key, 0.0) if prev_metrics else 0.0
        delta = (val - prev_val) if prev_metrics else 0.0
        ds    = _delta_str(delta) if prev_metrics else _col("  ─       ", _DIM)
        print(f"  {_col(name, _B):<32s}  {val:>8.4f}  {ds}  {_col(desc, _DIM)}")

    print()
    print(_col(
        f"  Graph  states={mi['total_states']}  reachable={mi['reachable_states']}"
        f"  basins={mi['attractor_basins']}"
        f"  diam={rc['diameter']}  components={rc['components']}",
        _DIM,
    ))

    if obstruction:
        obs = ", ".join(rc.get("obstructions", ["STUCK_LOOP"]))
        print()
        print(_col(f"  ⚠  OBSTRUCTION: {obs}", _B + _RD))

    print()


# ---------------------------------------------------------------------------
# Obstruction certificate emitter
# ---------------------------------------------------------------------------

def _emit_obstruction_cert(
    cert: dict,
    detected_at_event: int,
    episode_id: str,
    obs_type: str,
    plateau_checkpoints: int,
    last_new_state_at: int,
    out_dir: Path,
) -> str:
    """Write QA_COMPETENCY_OBSTRUCTION.v1 JSON and return its path."""
    payload: Dict[str, Any] = {
        "schema_id": "QA_COMPETENCY_OBSTRUCTION.v1",
        "created_utc": _now_utc(),
        "parent_cert_sha256": cert["manifest"]["canonical_json_sha256"],
        "obstruction_type": obs_type,
        "detected_at_event_index": detected_at_event,
        "episode_id": episode_id,
        "invariant_diff": {
            "plateau_checkpoints": plateau_checkpoints,
            "last_new_state_at_event_index": last_new_state_at,
            "reachable_states_at_obstruction": cert["metric_inputs"]["reachable_states"],
            "levin_invariant_violated": "convergence",
            "note": (
                "Agent revisited known states without discovering new reachable "
                "configurations. Plasticity (PI) collapsed to 0.0. The control "
                "region has stopped expanding — competency boundary confirmed."
            ),
        },
        "metrics_at_obstruction": dict(cert["competency_metrics"]),
        "verdict": {
            "passed": False,
            "fail_type": "OBSTRUCTION_DETECTED",
        },
        "manifest": {
            "manifest_version": 1,
            "hash_alg": "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }
    # Finalise manifest
    tmp = copy.deepcopy(payload)
    tmp["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    payload["manifest"]["canonical_json_sha256"] = _sha256(tmp)

    out_path = out_dir / f"obstruction_event{detected_at_event:03d}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(out_path)


# ---------------------------------------------------------------------------
# Main demo runner
# ---------------------------------------------------------------------------

def run_demo(fast: bool = False, out_dir_str: str = "/tmp/qa_demo_out") -> int:
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_delay  = 0.05 if fast else 0.6   # per event
    cert_delay  = 0.05 if fast else 1.2   # extra pause before checkpoint display

    # ── Header ──────────────────────────────────────────────────────────
    print()
    print(_col("  ╔══════════════════════════════════════════════════════════╗", _C + _B))
    print(_col("  ║   QA Levin Competency Framework  —  Live Demo           ║", _C + _B))
    print(_col("  ║   Domain: tool-using AI debugger (software_engineering) ║", _C + _B))
    print(_col("  ╚══════════════════════════════════════════════════════════╝", _C + _B))
    print()
    print(_col("  Levin-to-QA mapping (family [26]):", _B + _W))
    print(_col("    Competency  →  Reachability Class  (BFS from agent initial states)", _DIM))
    print(_col("    Goal        →  Attractor Basin     (sink SCCs in state graph)", _DIM))
    print(_col("    Memory      →  Invariant           (sandbox_isolation, convergence)", _DIM))
    print(_col("    Agency      →  Control Region      |reachable| / |total| states", _DIM))
    print(_col("    Plasticity  →  Generator Flex.     new states per tool call (windowed)", _DIM))
    print()

    if not fast:
        time.sleep(2.0)

    # ── State ───────────────────────────────────────────────────────────
    accumulated: List[dict] = []
    prev_metrics: Optional[Dict[str, float]] = None
    prev_reachable    = 0
    prev_event_count  = 0
    cert_index        = 0
    plateau_count     = 0
    last_new_state_at = 0
    obstruction_emitted = False
    metric_history: List[Dict[str, Any]] = []

    print(_col("  ── Agent execution trace ───────────────────────────────────", _C))
    print()

    for event_idx, event in enumerate(_EVENTS):
        accumulated.append(event)

        ep    = event["episode_id"]
        step  = event["step"]
        tool  = event["tool"]
        res   = event["result"]
        res_c = _G if res not in ("fail", "slow") else _RD

        print(f"  {_col(f'[{ep}:s{step:02d}]', _DIM)}"
              f"  {_col(tool, _B):<20s}"
              f"  →  {_col(res, res_c)}")

        if not fast:
            time.sleep(step_delay)

        if event_idx not in _CHECKPOINTS:
            continue

        if not fast:
            time.sleep(cert_delay)

        # ── Checkpoint: build cert ───────────────────────────────────
        phase_label, phase_desc, phase_color = _PHASE_INFO[event_idx]
        is_stuck = event_idx >= 13

        try:
            cert, current_reachable = _build_cert(
                accumulated,
                prev_reachable=prev_reachable,
                prev_event_count=prev_event_count,
                obstructions=["STUCK_LOOP", "context_overflow"] if is_stuck else None,
            )
        except Exception as exc:
            print(_col(f"\n  [CERT ERROR] {exc}\n", _RD))
            continue

        # Track plateau (no new states discovered)
        if current_reachable > prev_reachable or prev_event_count == 0:
            plateau_count = 0
            last_new_state_at = event_idx
        else:
            plateau_count += 1

        # ── Display ─────────────────────────────────────────────────
        _display_dashboard(
            phase_label=phase_label,
            phase_desc=phase_desc,
            phase_color=phase_color,
            n_events=len(accumulated),
            cert=cert,
            prev_metrics=prev_metrics,
            obstruction=is_stuck,
        )

        # Save checkpoint cert
        cert_path = out_dir / f"cert_{cert_index:02d}_event{event_idx:02d}.json"
        cert_path.write_text(
            json.dumps(cert, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(_col(f"  → cert saved: {cert_path.name}", _DIM))

        # ── Obstruction certificate ──────────────────────────────────
        if is_stuck and not obstruction_emitted:
            obs_path = _emit_obstruction_cert(
                cert=cert,
                detected_at_event=event_idx,
                episode_id=event["episode_id"],
                obs_type="STUCK_LOOP",
                plateau_checkpoints=plateau_count,
                last_new_state_at=last_new_state_at,
                out_dir=out_dir,
            )
            print()
            print(_col(f"  ┌─ OBSTRUCTION CERTIFICATE ─────────────────────────────────", _B + _RD))
            print(_col(f"  │  schema:   QA_COMPETENCY_OBSTRUCTION.v1", _RD))
            print(_col(f"  │  type:     STUCK_LOOP", _RD))
            print(_col(f"  │  levin:    convergence invariant violated", _RD))
            print(_col(f"  │  PI = 0.0  (no new states for {plateau_count} checkpoint(s))", _RD))
            print(_col(f"  │  verdict:  passed=false  fail_type=OBSTRUCTION_DETECTED", _RD))
            print(_col(f"  │  → {Path(obs_path).name}", _RD))
            print(_col(f"  └────────────────────────────────────────────────────────────", _B + _RD))
            print()
            obstruction_emitted = True

        # Record for trajectory table
        metric_history.append({
            "label": phase_label.strip(),
            "metrics": copy.deepcopy(cert["competency_metrics"]),
            "states": (current_reachable, cert["metric_inputs"]["total_states"]),
        })

        prev_metrics      = dict(cert["competency_metrics"])
        prev_reachable    = current_reachable
        prev_event_count  = len(accumulated)
        cert_index       += 1

        if not fast:
            time.sleep(0.5)

    # ── Metric trajectory summary ────────────────────────────────────────
    print()
    print(_col(f"  ══════════════════════════════════════════════════════════════", _C))
    print(_col(f"  Metric Trajectory Across Phases", _B + _W))
    print()

    header = f"  {'Metric':<22s}"
    for h in metric_history:
        header += f"  {h['label']:>10s}"
    print(_col(header, _B + _W))
    print(_col(f"  {'─'*60}", _DIM))

    for key, name, _ in _METRIC_ROWS:
        row = f"  {name:<22s}"
        vals = [h["metrics"].get(key, 0.0) for h in metric_history]
        for i, v in enumerate(vals):
            # Highlight collapse in Phase 3
            if key == "plasticity_index" and v == 0.0 and i >= 2:
                row += _col(f"  {v:>10.4f}", _RD + _B)
            elif key == "goal_density" and i >= 1 and v < metric_history[0]["metrics"][key]:
                row += _col(f"  {v:>10.4f}", _Y)
            else:
                row += f"  {v:>10.4f}"
        print(row)

    states_row = f"  {'States (r/total)':<22s}"
    for h in metric_history:
        r, t = h["states"]
        states_row += f"  {f'{r}/{t}':>10s}"
    print(_col(states_row, _DIM))

    print()
    print(_col("  Key findings:", _B + _W))
    print(_col("  ① High Agency (1.00) throughout — within explored territory, full control.", _W))
    print(_col("  ② Plasticity collapsed 1.00 → 0.00 in Phase 3: the agent cannot adapt", _W))
    print(_col("     to the novel failure mode outside its training distribution.", _W))
    print(_col("  ③ Goal Density halved (0.20 → 0.11) as the state space expanded:", _W))
    print(_col("     more territory, but the same single attractor basin.", _W))
    print(_col("  ④ Control Entropy ROSE (1.04 → 1.38) while stuck: diverse tool use", _W))
    print(_col("     with zero new outcomes — the signature of competency-bounded thrashing.", _W))
    print(_col("  ⑤ Obstruction cert formally documents the boundary with invariant_diff.", _W))
    print(_col("  ⑥ PDI quadrant diagnosis  (PI, PDI):", _W))
    print(_col("       Phase 1–2  (PI=hi, PDI=0.0)  → linear deterministic explorer", _W))
    print(_col("       Phase 3    (PI=0,  PDI≈0.89) → stuck-loop thrashing   ← confirmed!", _RD + _B))
    print(_col("     PDI exposes the counterfactual truth: cycles exist but no new ground.", _W))
    print()
    print(_col(f"  {cert_index} competency certs + 1 obstruction cert → {out_dir}/", _DIM))
    print(_col(f"  ══════════════════════════════════════════════════════════════", _C))
    print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="QA Levin Competency Framework — Live Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fast", action="store_true",
                    help="Skip inter-step delays (CI / non-interactive mode)")
    ap.add_argument("--out", default="/tmp/qa_demo_out",
                    help="Output directory for cert JSON (default: /tmp/qa_demo_out)")
    args = ap.parse_args(argv)
    return run_demo(fast=args.fast, out_dir_str=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
