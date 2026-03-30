#!/usr/bin/env python3
"""Generate deterministic harmonic obstruction episodes and batch plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ALPHAS: List[str] = [
    "1/2", "1/3", "1/4", "1/5", "1/6", "1/7", "1/8", "1/9", "1/10", "2/3",
    "2/5", "3/5", "3/7", "4/7", "4/9", "5/8", "5/12", "7/10", "7/12", "11/18",
]
WINDOWS: List[int] = [1024, 4096, 16384]
GENERATOR_SETS: Dict[str, Tuple[str, ...]] = {
    "gA": ("sigma", "mu"),
    "gB": ("sigma", "lambda"),
    "gC": ("sigma", "mu", "lambda", "nu"),
}

PLAN_SCHEMA_ID = "QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1"
EPISODE_SCHEMA_ID = "QA_CONJECTURE_PROVE_EPISODE_SCHEMA.v1"
PIPELINE_ID = "QA_DISCOVERY_PIPELINE.v1"
EPISODE_FAMILY = "QA_HARMONIC_OBSTRUCTION.v1"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(obj) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _alpha_slug(alpha: str) -> str:
    return alpha.replace("/", "-")


def _episode_name(alpha_index: int, alpha: str, window: int, gen_id: str) -> str:
    return f"EP_HO_a{alpha_index:02d}_{_alpha_slug(alpha)}_w{window}_{gen_id}.json"


def _run_id(alpha_index: int, window: int, gen_id: str, k: int, multi_k: bool) -> str:
    base = f"RUN-HO-a{alpha_index:02d}-w{window}-{gen_id}"
    if multi_k:
        return f"{base}-k{k}"
    return base


def _make_state_hash(label: str, payload: Dict[str, Any]) -> str:
    return _sha256(f"{label}:{_canonical(payload)}")


def _graph_profile(alpha_index: int, window: int, gen_id: str) -> Dict[str, Any]:
    """Deterministically parameterize a non-trivial reachability graph per case."""
    window_factor = window // 1024

    if gen_id == "gA":
        # Medium cycle lengths: some returns by k=64, all by k=256.
        core_len = 24 + ((alpha_index * 9 + window_factor * 5) % 96)   # [24, 119]
        has_return = True
    elif gen_id == "gB":
        # Acyclic baseline: no return regardless of k.
        core_len = 36 + ((alpha_index * 11 + window_factor * 7) % 140)  # [36, 175]
        has_return = False
    else:
        # Long cycles: mostly only visible at high k, some beyond k=256.
        core_len = 80 + ((alpha_index * 13 + window_factor * 17) % 260)  # [80, 339]
        has_return = True

    branch_stride = 4 + ((alpha_index + window_factor) % 5)  # [4, 8]
    return {
        "core_len": core_len,
        "has_return": has_return,
        "branch_stride": branch_stride,
    }


def _state_for(
    *,
    alpha: str,
    alpha_index: int,
    window: int,
    gen_id: str,
    node: str,
) -> str:
    return _make_state_hash(
        "HO_STATE",
        {
            "alpha": alpha,
            "alpha_index": alpha_index,
            "window": window,
            "generator_set": gen_id,
            "node": node,
        },
    )


def _append_step(
    *,
    steps: List[Dict[str, Any]],
    alpha: str,
    alpha_index: int,
    window: int,
    gen_id: str,
    generator: str,
    input_hash: str,
    output_hash: str,
    move_kind: str,
) -> None:
    idx = len(steps)
    step_payload = {
        "alpha": alpha,
        "alpha_index": alpha_index,
        "window": window,
        "generator_set": gen_id,
        "step_index": idx,
        "move_kind": move_kind,
        "input_hash": input_hash,
        "output_hash": output_hash,
    }
    steps.append(
        {
            "step_index": idx,
            "action": {
                "generator": generator,
                "params": {
                    "benchmark": "chowla_cosine_harmonic_obstruction",
                    "alpha": alpha,
                    "window": window,
                    "generator_set": gen_id,
                    "alpha_index": alpha_index,
                    "move_kind": move_kind,
                },
            },
            "input_hash": input_hash,
            "output_hash": output_hash,
            "trace_ref": {
                "family": "QA_MATH_COMPILER_STACK.v1",
                "trace_id": _make_state_hash("HO_TRACE", step_payload),
            },
            "result": {
                "status": "SUCCESS",
                "witness_hash": _make_state_hash("HO_WITNESS", step_payload),
            },
        }
    )


def _mutate_episode(
    template: Dict[str, Any],
    *,
    episode_id: str,
    created_utc: str,
    alpha: str,
    alpha_index: int,
    window: int,
    gen_id: str,
    generators: Tuple[str, ...],
) -> Dict[str, Any]:
    ep = copy.deepcopy(template)

    if ep.get("schema_id") != EPISODE_SCHEMA_ID:
        raise ValueError(
            f"Unexpected template schema_id={ep.get('schema_id')!r}; expected {EPISODE_SCHEMA_ID!r}."
        )

    ep["episode_id"] = episode_id
    ep["created_utc"] = created_utc
    ep["policy_id"] = "pi_harmonic_obstruction.v1"
    ep["generator_set_id"] = f"GENSET{{{','.join(generators)}}}.HO.v1"

    graph = _graph_profile(alpha_index, window, gen_id)
    core_len = int(graph["core_len"])
    has_return = bool(graph["has_return"])
    branch_stride = int(graph["branch_stride"])

    main_states = [
        _state_for(
            alpha=alpha,
            alpha_index=alpha_index,
            window=window,
            gen_id=gen_id,
            node=f"main_{i}",
        )
        for i in range(core_len + 1)
    ]

    ep["objective"]["type"] = "EXPLORE"
    ep["objective"]["target_hash"] = main_states[-1]
    ep["objective"]["budget"] = {"max_steps": 1000, "max_seconds": 60}
    ep["initial_state"] = {"layer": "formal", "state_hash": main_states[0]}

    steps: List[Dict[str, Any]] = []

    # Main path edges.
    for i in range(core_len):
        _append_step(
            steps=steps,
            alpha=alpha,
            alpha_index=alpha_index,
            window=window,
            gen_id=gen_id,
            generator=generators[i % len(generators)],
            input_hash=main_states[i],
            output_hash=main_states[i + 1],
            move_kind="main",
        )

        # Sparse branch edges increase reachable volume while keeping runtime bounded.
        if i > 0 and i % branch_stride == 0:
            branch_state = _state_for(
                alpha=alpha,
                alpha_index=alpha_index,
                window=window,
                gen_id=gen_id,
                node=f"branch_{i}",
            )
            _append_step(
                steps=steps,
                alpha=alpha,
                alpha_index=alpha_index,
                window=window,
                gen_id=gen_id,
                generator=generators[(i + 1) % len(generators)],
                input_hash=main_states[i],
                output_hash=branch_state,
                move_kind="branch",
            )

    # Cycle-closing edge for cycle-enabled generator sets.
    if has_return:
        _append_step(
            steps=steps,
            alpha=alpha,
            alpha_index=alpha_index,
            window=window,
            gen_id=gen_id,
            generator=generators[-1],
            input_hash=main_states[-1],
            output_hash=main_states[0],
            move_kind="return",
        )

    ep["steps"] = steps
    ep["final_status"] = {
        "status": "SUCCESS",
        "summary": (
            "Harmonic obstruction benchmark episode synthesized deterministically "
            f"(core_len={core_len}, has_return={has_return})."
        ),
    }
    ep["merkle_parent"] = _make_state_hash(
        "HO_MERKLE_PARENT",
        {
            "episode_id": episode_id,
            "alpha": alpha,
            "window": window,
            "generator_set": gen_id,
        },
    )
    ep["invariant_diff"] = {
        "steps_total": len(steps),
        "steps_success": len(steps),
        "steps_fail": 0,
        "final_status": "SUCCESS",
        "alpha": alpha,
        "window": window,
        "generator_set": gen_id,
        "core_len": core_len,
        "branch_stride": branch_stride,
        "has_return": has_return,
    }

    return ep


def _build_plan(
    *,
    plan_id: str,
    created_utc: str,
    run_queue: List[Dict[str, Any]],
    max_seconds_total: int,
) -> Dict[str, Any]:
    return {
        "schema_id": PLAN_SCHEMA_ID,
        "plan_id": plan_id,
        "created_utc": created_utc,
        "agent_id": "qa-agent-ctrl-1",
        "pipeline_id": PIPELINE_ID,
        "run_queue": run_queue,
        "determinism": {
            "queue_ordering": "lexicographic(run_id)",
            "seed_policy": "frontier[0]",
            "canonical_json": True,
        },
        "budget": {
            "max_runs": len(run_queue),
            "max_seconds_total": max_seconds_total,
        },
        "merkle_parent": "0" * 64,
    }


def _parse_k_values(k: int, k_values: str) -> List[int]:
    if not k_values.strip():
        return [k]

    out: List[int] = []
    for token in k_values.split(","):
        t = token.strip()
        if not t:
            continue
        v = int(t)
        if v < 1:
            raise ValueError("k values must be >= 1")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("k_values produced an empty list")
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    qa_root = repo_root / "qa_alphageometry_ptolemy"

    ap = argparse.ArgumentParser(
        description="Generate deterministic harmonic episodes and discovery plan."
    )
    ap.add_argument(
        "--episode_template",
        default=str(qa_root / "qa_conjecture_prove" / "fixtures" / "episode_valid.json"),
        help="Template episode JSON (must satisfy QA_CONJECTURE_PROVE_EPISODE_SCHEMA.v1).",
    )
    ap.add_argument(
        "--episodes_out_dir",
        default=str(qa_root / "qa_harmonic_obstruction" / "episodes"),
        help="Output directory for generated episodes.",
    )
    ap.add_argument(
        "--plan_out",
        default=str(qa_root / "qa_discovery_pipeline" / "plans" / "plan_harmonic_sweep_v1.json"),
        help="Output path for generated batch plan.",
    )
    ap.add_argument(
        "--created_utc",
        default="2026-02-13T17:00:00Z",
        help="created_utc timestamp written into episodes and plan.",
    )
    ap.add_argument("--k", type=int, default=16, help="k bound for each run in the batch plan.")
    ap.add_argument(
        "--k_values",
        default="",
        help="Optional comma-separated list of k values (e.g. 16,64,256). If set, overrides --k.",
    )
    ap.add_argument(
        "--plan_id",
        default="",
        help="Optional plan_id override. Defaults to PLAN-HARMONIC-SWEEP-V1 for single-k and PLAN-HARMONIC-KSWEEP-V1 for multi-k.",
    )
    ap.add_argument(
        "--max_seconds_total",
        type=int,
        default=7200,
        help="Batch budget max_seconds_total for plan emission.",
    )
    ap.add_argument(
        "--skip_episode_write",
        action="store_true",
        help="Emit only the plan (do not write episode files).",
    )
    args = ap.parse_args()

    template_path = Path(args.episode_template)
    episodes_out_dir = Path(args.episodes_out_dir)
    plan_out = Path(args.plan_out)

    template = _load_json(template_path)
    k_values = _parse_k_values(args.k, args.k_values)
    multi_k = len(k_values) > 1

    run_queue: List[Dict[str, Any]] = []
    episodes_written = 0
    written_episodes = set()

    for alpha_index, alpha in enumerate(ALPHAS, start=1):
        for window in WINDOWS:
            for gen_id, generators in GENERATOR_SETS.items():
                filename = _episode_name(alpha_index, alpha, window, gen_id)
                relative_episode_path = Path("qa_harmonic_obstruction") / "episodes" / filename
                absolute_episode_path = episodes_out_dir / filename

                episode = _mutate_episode(
                    template,
                    episode_id=filename[:-5],
                    created_utc=args.created_utc,
                    alpha=alpha,
                    alpha_index=alpha_index,
                    window=window,
                    gen_id=gen_id,
                    generators=generators,
                )

                if not args.skip_episode_write and filename not in written_episodes:
                    _write_json(absolute_episode_path, episode)
                    episodes_written += 1
                    written_episodes.add(filename)

                for k in k_values:
                    run_queue.append(
                        {
                            "run_id": _run_id(alpha_index, window, gen_id, k, multi_k),
                            "episode_ref": {
                                "family": EPISODE_FAMILY,
                                "path_or_hash": str(relative_episode_path),
                            },
                            "k": k,
                        }
                    )

    run_queue.sort(key=lambda item: str(item["run_id"]))
    plan_id = args.plan_id or ("PLAN-HARMONIC-KSWEEP-V1" if multi_k else "PLAN-HARMONIC-SWEEP-V1")

    plan = _build_plan(
        plan_id=plan_id,
        created_utc=args.created_utc,
        run_queue=run_queue,
        max_seconds_total=args.max_seconds_total,
    )
    _write_json(plan_out, plan)

    print(f"WROTE plan={plan_out}")
    print(f"K values={k_values}")
    print(f"RUNS total={len(run_queue)}")
    print(f"EPISODES written={episodes_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
