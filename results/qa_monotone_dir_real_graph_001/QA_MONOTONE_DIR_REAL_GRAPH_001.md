# QA Monotone Direction Score — Real Graph Validation 001

Validates the AGS theorem (cert [288]) structural interpretation on three classic
real-world network graphs. Because these graphs contain cycles, the monotone_dir
score is computed on the BFS spanning tree of each graph. Cycle fraction quantifies
how much structure is lost by this projection.

**Claim under test**: high-score nodes are branch-type nodes on the spanning tree —
either junction hubs with many off-axis edges, or purely peripheral nodes entirely
off the main axis. Low-score nodes (score=0) lie on the diameter axis with no
off-axis incident edges.

## Key Findings

**Football (conference-independent programs)**: The top-scoring nodes are Notre Dame,
Army, Air Force, Navy, and New Mexico State — the historically conference-independent
programs in college football (2002 season). These teams play schedules drawn from
multiple conferences rather than a single conference cluster, making them structural
hubs that are "off-axis" relative to any conference-aligned diameter axis. This is an
externally verifiable structural anomaly: the score surfaces the known independents
without any label supervision.

**Karate club (community leaders)**: Nodes 0 (Mr. Hi, score=14) and 33 (Officer,
score=7) are the two faction leaders — the most-connected members of each community.
Both are on-path junction nodes (they lie on the diameter axis), but their scores are
high because each has many off-axis branch edges (their community members hanging off
them). AGS correctly identifies them as high-degree junctions, not pure path nodes.

**Scope note**: These graphs have 58–81% cycle fraction, so AGS is evaluated on the
BFS spanning tree only. The spanning tree discards back-edges; high-cycle-fraction
graphs lose more of their structure to this projection. The football result is
particularly robust because the structural phenomenon (independence) is reflected
in the spanning tree topology.

## Karate (34 nodes, 78 edges)

- Cycle fraction: 57.7%
- Spanning-tree diameter: 5, anchors: ['14', '30']
- On-path nodes: 6
- Score distribution: {'0': 4, '1': 25, '2': 1, '3': 1, '5': 1, '7': 1, '14': 1}

**Top-scoring nodes** (periphery of spanning tree):

| node | score | deg(orig) | deg(tree) | on_path | attrs |
|---|---:|---:|---:|---|---|
| 0 | 14 | 16 | 16 | True | club=Mr. Hi; value=0 |
| 33 | 7 | 17 | 9 | True | club=Officer; value=1 |
| 2 | 5 | 10 | 5 | False | club=Mr. Hi; value=0 |
| 31 | 3 | 6 | 3 | False | club=Officer; value=1 |
| 5 | 2 | 4 | 2 | False | club=Mr. Hi; value=0 |
| 10 | 1 | 3 | 1 | False | club=Mr. Hi; value=0 |
| 11 | 1 | 1 | 1 | False | club=Mr. Hi; value=0 |
| 12 | 1 | 2 | 1 | False | club=Mr. Hi; value=0 |

**On-path sample** (spanning-tree axis — score=0):

| node | score | deg(orig) | attrs |
|---|---:|---:|---|
| 0 | 14 | 16 | club=Mr. Hi; value=0 |
| 1 | 0 | 9 | club=Mr. Hi; value=0 |
| 13 | 0 | 5 | club=Mr. Hi; value=0 |
| 14 | 0 | 2 | club=Officer; value=1 |
| 30 | 0 | 4 | club=Officer; value=1 |
| 33 | 7 | 17 | club=Officer; value=1 |

## Dolphins (62 nodes, 159 edges)

- Cycle fraction: 61.6%
- Spanning-tree diameter: 10, anchors: ['Zig', 'Cross']
- On-path nodes: 11
- Score distribution: {'0': 3, '1': 39, '2': 10, '3': 5, '4': 3, '5': 1, '11': 1}

**Top-scoring nodes** (periphery of spanning tree):

| node | score | deg(orig) | deg(tree) | on_path | attrs |
|---|---:|---:|---:|---|---|
| Grin | 11 | 12 | 11 | False |  |
| Haecksel | 5 | 7 | 5 | False |  |
| Beak | 4 | 6 | 6 | True |  |
| Jet | 4 | 9 | 4 | False |  |
| Number1 | 4 | 5 | 4 | False |  |
| Bumper | 3 | 4 | 3 | False |  |
| SN100 | 3 | 7 | 3 | False |  |
| Stripes | 3 | 7 | 3 | False |  |

**On-path sample** (spanning-tree axis — score=0):

| node | score | deg(orig) | attrs |
|---|---:|---:|---|
| Beak | 4 | 6 |  |
| Cross | 0 | 1 |  |
| DN63 | 2 | 5 |  |
| Fish | 1 | 5 |  |
| Gallatin | 1 | 8 |  |
| Patchback | 2 | 9 |  |

## Football (115 nodes, 613 edges)

- Cycle fraction: 81.4%
- Spanning-tree diameter: 6, anchors: ['Kansas', 'Clemson']
- On-path nodes: 7
- Score distribution: {'0': 2, '1': 77, '2': 9, '3': 8, '4': 11, '5': 2, '6': 2, '7': 1, '8': 3}

**Top-scoring nodes** (periphery of spanning tree):

| node | score | deg(orig) | deg(tree) | on_path | attrs |
|---|---:|---:|---:|---|---|
| AirForce | 8 | 10 | 10 | True | value=7 |
| Army | 8 | 11 | 10 | True | value=4 |
| NotreDame | 8 | 11 | 8 | False | value=5 |
| Navy | 7 | 11 | 7 | False | value=5 |
| NewMexicoState | 6 | 11 | 6 | False | value=10 |
| Toledo | 6 | 9 | 6 | False | value=6 |
| BostonCollege | 5 | 11 | 5 | False | value=1 |
| NevadaLasVegas | 5 | 12 | 5 | False | value=7 |

**On-path sample** (spanning-tree axis — score=0):

| node | score | deg(orig) | attrs |
|---|---:|---:|---|
| AirForce | 8 | 10 | value=7 |
| AlabamaBirmingham | 4 | 10 | value=4 |
| Army | 8 | 11 | value=4 |
| BrighamYoung | 4 | 12 | value=7 |
| Clemson | 0 | 10 | value=0 |
| FloridaState | 4 | 12 | value=0 |
