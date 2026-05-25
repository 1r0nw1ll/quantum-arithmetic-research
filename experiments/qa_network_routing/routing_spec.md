# QA Substrate Ladder Level 4: Network/Routing Benchmark Spec

## Overview

Packets are QA-state pairs `(b, e)` traversing a directed generator graph on
`{1,...,N}²`. Routing is generator-legal path finding; delivery is reaching a
target orbit_9 class. Three routers are compared on four workload modes.

## Generator Graph

Nodes: `(b, e)` pairs in `{1,...,N}²`.
Edges (directed): legal generator moves from each node.
- **sigma**: `(b, e) → (b, e+1)` if `e+1 ≤ N`
- **mu**: `(b, e) → (e, b)` if `b ≠ e`
- **lambda2**: `(b, e) → (2b, 2e)` if `2b ≤ N` and `2e ≤ N`
- **nu**: `(b, e) → (b/2, e/2)` if `b` and `e` are both even

No self-loops. No duplicate edges. `(k, k)` cells have no mu edge.

## Orbit Classification

`orbit_9(b, e) = (b + e) % 9` — 9 classes, 0–8.

## Packet Model

Each `Packet` is an immutable descriptor: `(packet_id, src, dst_orbit, workload_type)`.
Delivery condition: the packet's current cell satisfies `orbit_9(cell) == dst_orbit`.
Runtime state (`pos`, `steps`, `delivered`) lives in `_PacketState` (created fresh
per simulation run). The same `Packet` list can be passed to multiple routers safely.

## Precomputed Orbit-Distance Table

`dist_table[b-1][e-1][t]` = minimum hops from `(b, e)` to any cell at `orbit_9 == t`.

Built via reverse-graph multi-source BFS: for each target orbit `t`, collect all
source nodes at orbit `t`, BFS backwards in the reverse generator graph. Total
complexity: `O(9 × (|V| + |E|))` where `|V| = N²`, `|E| ≤ 4N²`.

## Simulation Engine

One tick per step. All undelivered packets move simultaneously. Tick loop:
1. Count `orbit_load[o]` and `node_load[(b,e)]` for all active packets.
2. Record `peak_node_load` and `orbit_saturation = max(orbit_load) / n_active`.
3. Mark congestion for nodes with `node_load ≥ congestion_threshold` (default: 2).
4. Call `router_fn(pos, dst_orbit, orbit_load, dist_table, N, rng)` per packet.
5. Move each packet to the returned position; check for delivery.
6. Break if all packets delivered.

## Routers

| Router | Strategy |
|--------|----------|
| `random` | Uniform random choice among legal neighbors |
| `greedy` | Neighbor minimizing `dist_table[b-1][e-1][dst_orbit]` |
| `qa_router` | Neighbor minimizing `(orbit_distance, orbit_load_at_neighbor_orbit)` |

## Workload Modes

| Mode | Construction |
|------|-------------|
| `qa_lawful` | Random src; `dst_orbit = (src_orbit + 3) % 9` (always ≠ src_orbit) |
| `random_opaque` | Random src; `dst_orbit` uniform in 0–8 |
| `adversarial_congestion` | All packets: `src=(1,1)`, `dst_orbit=5` |
| `mixed` | 50% qa_lawful + 50% random_opaque, alternating |

## Hypotheses

**H1** (SUPPORTED): Greedy and QA take fewer mean hops than random on qa_lawful.
- Mechanism: orbit_distance minimization finds direct paths; random walk meanders.
- Benchmark: `greedy=2.9, qa=2.9` vs `random=9.6` mean steps.

**H2** (CONFIRMED NULL): QA orbit load tiebreaker is inert on qa_lawful.
- Mechanism: orbit_distance ties between legal neighbors are rare. The primary
  criterion (orbit_distance) resolves uniquely in almost all routing decisions,
  leaving no tiebreaker opportunities. QA routes identically to greedy.
- Observable: `abs(qa.mean_orbit_saturation - greedy.mean_orbit_saturation) < 0.01`.

**H3** (CONFIRMED): QA orbit load balancing has no systematic advantage vs greedy
on random_opaque.
- Mechanism: random dst_orbits provide no structured gradient for orbit_load to
  exploit. QA and greedy achieve similar orbit saturation.
- Observable: `abs(qa.peak_orbit_saturation - greedy.peak_orbit_saturation) < 0.05`.

**H4** (CONFIRMED): On adversarial_congestion, QA degenerates to greedy behavior;
random wanders and takes more steps.
- Mechanism: All packets share the same source cell → `orbit_load` is identical
  for every packet → QA's tiebreaker observes the same load signal as every other
  packet → QA makes the same choice as greedy. Random doesn't minimize orbit_distance
  → more steps.
- Observable: `abs(qa.mean_steps - greedy.mean_steps) < 0.5` AND
  `random.mean_steps > greedy.mean_steps + 1.0`.

## Key Empirical Results (N=20, 200 packets, seed=42)

| Mode | Router | del% | steps | pk_sat |
|------|--------|------|-------|--------|
| qa_lawful | random | 1.000 | 9.6 | 1.000 |
| qa_lawful | greedy | 1.000 | 2.9 | 1.000 |
| qa_lawful | qa_router | 1.000 | 2.9 | 1.000 |
| adversarial_congestion | random | 1.000 | 12.0 | 1.000 |
| adversarial_congestion | greedy | 1.000 | 2.0 | 1.000 |
| adversarial_congestion | qa_router | 1.000 | 2.0 | 1.000 |
| random_opaque | random | 1.000 | 10.0 | 1.000 |
| random_opaque | greedy | 1.000 | 3.1 | 0.500 |
| random_opaque | qa_router | 1.000 | 3.1 | 0.500 |

## Notes

- `peak_orbit_saturation` hits 1.0 for any router when the last 1–2 packets are
  undelivered (all active packets in one orbit). Not a useful discriminator.
- `mean_orbit_saturation` is also tail-dominated (end-game ticks with n_active=1
  contribute sat=1.0). The bulk-phase (ticks 0–2) has sat~0.2 for 50 packets / N=15.
- The orbit_load tiebreaker in QA is a latent capability: it would activate in
  denser networks where multiple neighbors have equal orbit_distance.
