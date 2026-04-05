#!/usr/bin/env python3
"""
cert_batch_route_through_kernel.py

Routes the 12 new [122] empirical observation certs (batch 2026-04-04)
through a live QALabKernel as CERTIFY tasks, one per artifact, ending with
a single IMPROVE cycle that sees the full batch as runtime history.

This is the execution counterpart of cert_batch_empirical_2026_04_04.py:
we already generated the artifacts; now we feed them through the hardened
pipeline so that v2's ρ-EWMA Lyapunov accumulates real heterogeneous
workload and the kernel gets exercised end-to-end on live certs.

Produces:
  qa_lab/kernel/cert_batch_kernel_routing_readout.json

Run:
    cd /home/player2/signal_experiments
    python tools/cert_batch_route_through_kernel.py
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_batch_kernel_router — routes validated [122] cert artifacts through CertAgent CERTIFY tasks; subprocess validator invocations are observer-layer operations per Theorem NT"

import datetime
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_QA_LAB_DIR = _REPO / "qa_lab"
if str(_QA_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_QA_LAB_DIR))

from kernel.loop import QALabKernel, Task, TaskType
from agents.cert_agent import CertAgent
from agents.self_improvement_agent_v2 import SelfImprovementAgentV2

RESULTS_DIR = _REPO / "qa_alphageometry_ptolemy" / "qa_empirical_observation_cert" / "results"
VALIDATOR_REL = "qa_alphageometry_ptolemy/qa_empirical_observation_cert/qa_empirical_observation_cert_validate.py"

# The 12 new artifacts (by filename stem) from the 2026-04-04 batch
BATCH_2026_04_04 = [
    "eoc_pass_audio_residual_control_consistent",
    "eoc_pass_climate_enso_teleconnection_consistent",
    "eoc_pass_era5_multilayer_observer_gap_partial",
    "eoc_pass_karate_hub_distance_partial",
    "eoc_pass_karate_spectral_fingerprint_consistent",
    "eoc_pass_finance_qci_robustness_consistent",
    "eoc_pass_curvature_loss_correlation_exp3_consistent",
    "eoc_pass_integration_bench_football_consistent",
    "eoc_pass_integration_bench_karate_contradicts",
    "eoc_pass_integration_bench_raman_qa21_inconclusive",
    "eoc_pass_eeg_chbmit_observer3_topographic_consistent",
    "eoc_pass_qa_reasoner_a1_compliance_consistent",
]


def main() -> int:
    print(f"[cert-batch-route] start {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    kernel = QALabKernel(
        repo_root=_REPO,
        modulus=9,
        verbose=False,
        dry_run=False,
        require_spawn_approval=True,
    )
    cert_agent = CertAgent(repo_root=_REPO)
    v2 = SelfImprovementAgentV2(
        modulus=9,
        lyapunov="rho_ewma",
        lyapunov_alpha=0.2,
        dry_run_probe_cycles=0,
    )
    kernel.register_agent(TaskType.CERTIFY, cert_agent)
    kernel.register_agent(TaskType.IMPROVE, v2)

    cycle_results = []
    for i, stem in enumerate(BATCH_2026_04_04, start=1):
        cert_file = RESULTS_DIR / f"{stem}.json"
        if not cert_file.exists():
            print(f"  [{i:02d}] MISSING {stem}")
            continue
        task = Task(
            task_type=TaskType.CERTIFY,
            description=f"batch route {i:02d}/12: {stem}",
            inputs={
                "validator_path": VALIDATOR_REL,
                "cert_path": str(cert_file),
                "family_name": "qa_empirical_observation_cert",
            },
        )
        result = kernel.run_cycle(task)
        cycle_results.append({
            "idx": i,
            "stem": stem,
            "verdict": result.output.get("verdict"),
            "ok": result.ok,
            "verified": result.verified,
            "orbit_family": result.orbit_family,
            "rho": result.rho,
            "exit_code": result.output.get("exit_code"),
        })
        print(f"  [{i:02d}] {stem[:60]:60s}  ok={result.ok}  verified={result.verified}  rho={result.rho:.4f}")

    # Final IMPROVE cycle — v2 now sees the full 12-cycle certification history
    print()
    print("[cert-batch-route] running terminal IMPROVE cycle (v2 with full runtime history)")
    improve = kernel.run_cycle(Task(
        task_type=TaskType.IMPROVE,
        description="post-batch IMPROVE cycle over 12 heterogeneous CERTIFY results",
    ))
    improve_out = improve.output
    if "verdict" not in improve_out:
        print(f"  ERROR: {improve_out}")
    else:
        lyap = improve_out["lyapunov"]
        fp = improve_out["fixed_point_candidates"]
        print(f"  verdict={improve_out['verdict']}")
        print(f"  lyap:  {lyap['name']}: {lyap['pre']:.6f} → {lyap['post']:.6f}")
        print(f"  fp: A={fp['A_trace_compression']:.4f}  B={fp['B_routing_barycenter']:.4f}  C={fp['C_pisano_periodicity']:.4f}")
        print(f"  all: {fp['all_lyapunovs']}")

    readout = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session": "cert-batch-empirical",
        "kernel_run_id": kernel._kernel_run_id,
        "batch_tag": "2026-04-04",
        "n_certs": len(BATCH_2026_04_04),
        "cycle_results": cycle_results,
        "certs_verified": sum(1 for r in cycle_results if r["verified"]),
        "certs_failed": sum(1 for r in cycle_results if not r["verified"]),
        "rho_distribution_mean": sum(r["rho"] for r in cycle_results) / len(cycle_results) if cycle_results else None,
        "terminal_improve_cycle": {
            "verdict": improve_out.get("verdict"),
            "summary": improve_out.get("summary"),
            "lyapunov": improve_out.get("lyapunov"),
            "fixed_point_candidates": improve_out.get("fixed_point_candidates"),
            "kernel_quality_vector": improve_out.get("kernel_quality_vector"),
            "cert_hash": (improve_out.get("cert") or {}).get("cert_hash"),
        },
        "kernel_state_final": {
            "orbit": list(kernel.orbit_state),
            "family": kernel.orbit_family,
            "cycle_count": kernel._cycle_count,
            "results_in_memory": len(kernel._results),
        },
    }

    out_path = _QA_LAB_DIR / "kernel" / "cert_batch_kernel_routing_readout.json"
    out_path.write_text(json.dumps(readout, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print()
    print(f"[cert-batch-route] wrote {out_path.relative_to(_REPO)}")
    print(f"[cert-batch-route] done {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    return 0 if readout["certs_verified"] == len(BATCH_2026_04_04) else 1


if __name__ == "__main__":
    raise SystemExit(main())
