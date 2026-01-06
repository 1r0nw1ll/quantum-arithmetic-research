
"""
qa_alphageometry/adapters/certificate_adapter.py

Adapter: AlphaGeometry SearchResult (beam.rs) -> ProofCertificate.

Aligned to the SearchResult format you described:
  {
    "solved": bool,
    "proof": { "steps": [...], "final_state_hash": int, "metadata": {...} } | null,
    "states_expanded": int,
    "successors_generated": int,
    "successors_kept": int,
    "depth_reached": int,
    "best_score": float,
    "beam_signatures": [[depth, hash], ...]
  }
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from hashlib import sha256
import json

from qa_certificate import (
    ProofCertificate, Generator, MoveWitness, StateRef,
    InvariantContract, SearchMetadata,
    ObstructionEvidence, FailType
)

def _stable_id(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode()
    return sha256(s).hexdigest()[:16]

def _mk_state_ref(theorem_id: str, idx: int, step: Dict[str, Any]) -> StateRef:
    payload = {
        "theorem_id": theorem_id,
        "idx": idx,
        "rule_id": step.get("rule_id", "UNKNOWN"),
        "step_id": step.get("id"),
    }
    sid = _stable_id(payload)
    return StateRef(state_id=sid, coords=tuple(), packet=None)

def wrap_searchresult_to_certificate(
    sr: Dict[str, Any],
    theorem_id: str,
    *,
    max_depth_limit: int = 50,
    tracked_invariants: Optional[List[str]] = None,
    repo_tag: Optional[str] = None,
    commit: Optional[str] = None,
) -> ProofCertificate:

    solved = bool(sr.get("solved", False))
    proof = sr.get("proof")
    steps = (proof.get("steps") if proof else None) or []

    states_expanded = int(sr.get("states_expanded", 0))
    successors_generated = int(sr.get("successors_generated", 0))
    successors_kept = int(sr.get("successors_kept", 0))
    depth_reached = int(sr.get("depth_reached", 0))
    best_score = float(sr.get("best_score", 0.0))
    beam_signatures = sr.get("beam_signatures", [])

    gen_names = sorted({f"AG:{st.get('rule_id', 'UNKNOWN')}" for st in steps})
    generator_set: Set[Generator] = (
        {Generator(name=g, params=tuple()) for g in gen_names}
        if gen_names else {Generator(name="AG:UNKNOWN", params=tuple())}
    )

    contracts = InvariantContract(
        tracked_invariants=tracked_invariants or [],
        non_reduction_enforced=False,  # AG uses its own algebra (not QA non-reduction axiom)
        fixed_q_mode=None
    )

    elapsed_ms = None
    if proof and "metadata" in proof:
        md = proof["metadata"]
        if "elapsed_ms" in md:
            elapsed_ms = int(md["elapsed_ms"])
        elif "timeout_ms" in md:
            elapsed_ms = int(md["timeout_ms"])

    search = SearchMetadata(
        max_depth=max_depth_limit,
        states_explored=states_expanded,
        frontier_policy="beam_search",
        time_elapsed_ms=elapsed_ms,
    )

    if solved and proof:
        path: List[MoveWitness] = []
        prev_ref = _mk_state_ref(theorem_id, -1, {"rule_id": "init", "id": -1})

        for idx, st in enumerate(steps):
            rule_id = st.get("rule_id", "UNKNOWN")
            gen = Generator(name=f"AG:{rule_id}", params=tuple())
            dst_ref = _mk_state_ref(theorem_id, idx, st)
            packet_delta = {k: 0 for k in contracts.tracked_invariants}

            path.append(MoveWitness(
                gen=gen, src=prev_ref, dst=dst_ref,
                packet_delta=packet_delta, legal=True
            ))
            prev_ref = dst_ref

        return ProofCertificate(
            theorem_id=theorem_id,
            generator_set=generator_set,
            contracts=contracts,
            witness_type="success",
            success_path=path,
            search=search,
            context={
                "source": "qa-alphageometry",
                "repo_tag": repo_tag,
                "commit": commit,
                "searchresult_hash": _stable_id(sr),
                "rule_ids": [st.get("rule_id", "UNKNOWN") for st in steps],
                "proof_length": len(steps),
                "final_state_hash": (proof.get("final_state_hash") if isinstance(proof, dict) else None),
                "successors_kept": successors_kept,
                "best_score": best_score,
            }
        )

    if depth_reached >= max_depth_limit:
        inferred_reason = "max_depth_reached"
    elif successors_generated == 0:
        inferred_reason = "no_successors_generated"
    else:
        inferred_reason = "search_exhausted"

    frontier_hash = _stable_id(beam_signatures) if beam_signatures else None

    obstruction = ObstructionEvidence(
        fail_type=FailType.DEPTH_EXHAUSTED,
        generator_set=generator_set,
        max_depth_reached=depth_reached,
        states_explored=states_expanded,
        reachable_frontier_hash=frontier_hash,
    )

    return ProofCertificate(
        theorem_id=theorem_id,
        generator_set=generator_set,
        contracts=contracts,
        witness_type="obstruction",
        obstruction=obstruction,
        search=search,
        context={
            "source": "qa-alphageometry",
            "repo_tag": repo_tag,
            "commit": commit,
            "searchresult_hash": _stable_id(sr),
            "inferred_stop_reason": inferred_reason,
            "depth_reached": depth_reached,
            "best_score": best_score,
            "successors_generated": successors_generated,
            "successors_kept": successors_kept,
            "partial_steps": len(steps),
            "partial_rule_ids": [st.get("rule_id", "UNKNOWN") for st in steps] if steps else [],
        }
    )
