"""
llm_tool_agent.py

Adapter: JSONL agent tool-call events → QA_COMPETENCY_DETECTION_FRAMEWORK.v1 cert.

Pure function — no side effects, no learning, no heuristics.
Stdlib only (no networkx).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Canonical JSON helpers (self-contained copies — avoids coupling to validator)
# ---------------------------------------------------------------------------

def canonical_json_compact(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, full Unicode."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj: Any) -> str:
    """SHA256 of canonical JSON representation (64-hex)."""
    return hashlib.sha256(
        canonical_json_compact(obj).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Structured failure
# ---------------------------------------------------------------------------

HEX64_ZERO = "0" * 64

REQUIRED_EVENT_FIELDS = ("episode_id", "step", "state", "tool")


class IntakeError(Exception):
    """Structured intake failure carrying fail_type + invariant_diff."""

    def __init__(self, fail_type: str, invariant_diff: Dict[str, Any],
                 message: str = ""):
        self.fail_type = fail_type
        self.invariant_diff = invariant_diff
        super().__init__(message or f"{fail_type}: {invariant_diff}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fail_type": self.fail_type,
            "invariant_diff": self.invariant_diff,
        }


# ---------------------------------------------------------------------------
# Tarjan's SCC (stdlib only, ~40 lines)
# ---------------------------------------------------------------------------

def _tarjan_sccs(adj: Dict[str, List[str]]) -> List[List[str]]:
    """Tarjan's algorithm for strongly connected components.

    Args:
        adj: adjacency list {node: [successors]}

    Returns:
        List of SCCs (each a list of node ids), in reverse topological order.
    """
    index_counter = [0]
    stack: List[str] = []
    on_stack = set()
    index_map: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    result: List[List[str]] = []

    def strongconnect(v: str) -> None:
        index_map[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, []):
            if w not in index_map:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index_map[w])

        if lowlink[v] == index_map[v]:
            scc: List[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            result.append(scc)

    all_nodes = set(adj.keys())
    for targets in adj.values():
        all_nodes.update(targets)

    for node in sorted(all_nodes):
        if node not in index_map:
            strongconnect(node)

    return result


# ---------------------------------------------------------------------------
# BFS reachability
# ---------------------------------------------------------------------------

def _bfs_reachable(adj: Dict[str, List[str]], starts: List[str]) -> set:
    """BFS from start nodes, return set of reachable node ids."""
    visited = set()
    queue = list(starts)
    for s in queue:
        visited.add(s)
    i = 0
    while i < len(queue):
        node = queue[i]
        i += 1
        for nxt in adj.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return visited


# ---------------------------------------------------------------------------
# Graph diameter (BFS from every node in reachable set)
# ---------------------------------------------------------------------------

def _graph_diameter(adj: Dict[str, List[str]], nodes: set) -> int:
    """Compute diameter of subgraph induced by *nodes* (undirected BFS)."""
    if not nodes:
        return 0
    # Build undirected adjacency for diameter computation
    undirected: Dict[str, set] = defaultdict(set)
    for u in nodes:
        for v in adj.get(u, []):
            if v in nodes:
                undirected[u].add(v)
                undirected[v].add(u)

    diameter = 0
    for start in nodes:
        dist: Dict[str, int] = {start: 0}
        queue = [start]
        i = 0
        while i < len(queue):
            u = queue[i]
            i += 1
            for v in undirected.get(u, set()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if dist:
            diameter = max(diameter, max(dist.values()))
    return diameter


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def adapt(
    events: List[dict],
    *,
    domain: str,
    substrate: str,
    description: str,
) -> dict:
    """Pure: events list → QA_COMPETENCY_DETECTION_FRAMEWORK.v1 cert dict.

    The cert is returned with manifest placeholder (HEX64_ZERO).
    Caller is responsible for hashing manifests.

    Raises:
        IntakeError: on any validation or processing failure
    """
    # ---- 1. Validate required fields ----------------------------------
    for i, ev in enumerate(events):
        missing = [f for f in REQUIRED_EVENT_FIELDS if f not in ev]
        if missing:
            raise IntakeError(
                fail_type="INGEST_MISSING_REQUIRED_FIELD",
                invariant_diff={"event_index": i, "missing": missing},
            )

    # ---- 2. Sort by (episode_id, step); fail on duplicates ------------
    events = sorted(events, key=lambda e: (str(e["episode_id"]), int(e["step"])))
    seen_keys: set = set()
    for ev in events:
        key = (str(ev["episode_id"]), int(ev["step"]))
        if key in seen_keys:
            raise IntakeError(
                fail_type="INGEST_DUPLICATE_STEP",
                invariant_diff={"episode_id": key[0], "step": key[1]},
            )
        seen_keys.add(key)

    # ---- 3. Compute state_ids -----------------------------------------
    for ev in events:
        ev["_state_id"] = sha256_canonical(ev["state"])

    # ---- 4. Group by episode, build graph -----------------------------
    episodes: Dict[str, List[dict]] = defaultdict(list)
    for ev in events:
        episodes[str(ev["episode_id"])].append(ev)

    # adjacency list (directed) + edge labels
    adj: Dict[str, List[str]] = defaultdict(list)
    all_nodes: set = set()
    edge_set: set = set()  # (from, to, tool) for graph hash
    tool_edge_counts: Dict[str, int] = defaultdict(int)
    total_transitions = 0
    initial_states: List[str] = []

    for ep_id in sorted(episodes.keys()):
        ep_events = episodes[ep_id]
        if ep_events:
            initial_states.append(ep_events[0]["_state_id"])
        for ev in ep_events:
            all_nodes.add(ev["_state_id"])
        for i in range(len(ep_events) - 1):
            src = ep_events[i]["_state_id"]
            dst = ep_events[i + 1]["_state_id"]
            tool = str(ep_events[i]["tool"])
            adj[src].append(dst)
            edge_set.add((src, dst, tool))
            tool_edge_counts[tool] += 1
            total_transitions += 1

    if total_transitions == 0:
        raise IntakeError(
            fail_type="ADAPTER_EMPTY_TRACE",
            invariant_diff={"event_count": len(events), "episodes": len(episodes)},
        )

    # ---- 5. Generators from tool names --------------------------------
    all_tools: set = set()
    for ev in events:
        all_tools.add(str(ev["tool"]))
    generators = sorted(all_tools)

    # ---- 6. SCCs via Tarjan -------------------------------------------
    sccs = _tarjan_sccs(adj)

    # ---- 7. Sink SCCs (no outgoing edges to other SCCs) ---------------
    node_to_scc: Dict[str, int] = {}
    for idx, scc in enumerate(sccs):
        for n in scc:
            node_to_scc[n] = idx

    sink_sccs = []
    for idx, scc in enumerate(sccs):
        scc_set = set(scc)
        is_sink = True
        for n in scc:
            for succ in adj.get(n, []):
                if succ not in scc_set:
                    is_sink = False
                    break
            if not is_sink:
                break
        if is_sink:
            sink_sccs.append(idx)

    attractor_basins = len(sink_sccs)

    # ---- 8. Reachability via BFS from episode-initial states ----------
    reachable = _bfs_reachable(adj, initial_states)
    reachable_states = len(reachable)
    total_states = len(all_nodes)

    # ---- 9. Move probabilities ----------------------------------------
    move_probabilities: Dict[str, float] = {}
    if total_transitions > 0:
        for tool in generators:
            count = tool_edge_counts.get(tool, 0)
            move_probabilities[tool] = count / total_transitions

    # ---- 10. Graph hash -----------------------------------------------
    sorted_nodes = sorted(all_nodes)
    sorted_edges = sorted(edge_set)
    graph_hash = sha256_canonical({
        "nodes": sorted_nodes,
        "edges": [list(e) for e in sorted_edges],
    })

    # ---- 11. State space inference ------------------------------------
    all_coordinates: set = set()
    for ev in events:
        state = ev["state"]
        if isinstance(state, dict):
            all_coordinates.update(state.keys())
    coordinates = sorted(all_coordinates) if all_coordinates else ["opaque_state"]
    dimension = len(coordinates)

    # ---- 12. Connected components (weakly) ----------------------------
    # Build undirected adj for component counting
    undirected: Dict[str, set] = defaultdict(set)
    for u in all_nodes:
        for v in adj.get(u, []):
            undirected[u].add(v)
            undirected[v].add(u)

    visited: set = set()
    components = 0
    for n in sorted(all_nodes):
        if n not in visited:
            components += 1
            queue = [n]
            visited.add(n)
            qi = 0
            while qi < len(queue):
                cur = queue[qi]
                qi += 1
                for nb in undirected.get(cur, set()):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

    # ---- 13. Diameter -------------------------------------------------
    diameter = _graph_diameter(adj, reachable)

    # ---- 14. Compute metrics via qa_competency_metrics ----------------
    try:
        from qa_competency.qa_competency_metrics import compute_competency_metrics
    except ImportError:
        from qa_competency_metrics import compute_competency_metrics

    metrics = compute_competency_metrics(
        reachable_states=reachable_states,
        total_states=total_states,
        attractor_basins=attractor_basins,
        move_probabilities=move_probabilities,
        delta_reachability=0,
        delta_perturbation=1,
    )

    # ---- 15. Assemble cert dict ---------------------------------------
    cert: Dict[str, Any] = {
        "schema_id": "QA_COMPETENCY_DETECTION_FRAMEWORK.v1",
        "system_metadata": {
            "domain": domain,
            "substrate": substrate,
            "description": description,
        },
        "state_space": {
            "dimension": dimension,
            "coordinates": coordinates,
            "constraints": [],
        },
        "generators": [
            {
                "id": g,
                "description": f"Tool action: {g}",
                "action": g,
            }
            for g in generators
        ],
        "invariants": [],
        "reachability": {
            "components": components,
            "diameter": diameter,
            "obstructions": [],
        },
        "graph_snapshot": {
            "hash_sha256": graph_hash,
            "time_window": {
                "start_utc": "1970-01-01T00:00:00Z",
                "end_utc": "1970-01-01T00:00:00Z",
            },
            "edge_semantics": (
                "generator-edges: apply generator id to state yields next-state"
            ),
        },
        "metric_inputs": {
            "reachable_states": reachable_states,
            "total_states": total_states,
            "attractor_basins": attractor_basins,
            "move_probabilities": move_probabilities,
            "delta_reachability": 0,
            "delta_perturbation": 1,
        },
        "competency_metrics": metrics.as_dict(),
        "validation": {
            "validator": "qa_competency_validator.py",
            "hash": "sha256:" + HEX64_ZERO,
            "reproducibility_seed": 0,
        },
        "examples": sorted({str(ev["episode_id"]) for ev in events}),
        "manifest": {
            "manifest_version": 1,
            "hash_alg": "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }

    # Clean up temp keys
    for ev in events:
        ev.pop("_state_id", None)

    return cert
