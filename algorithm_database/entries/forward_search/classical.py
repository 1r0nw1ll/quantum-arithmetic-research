"""Classical forward search — Python port of Kochenderfer 2022 DM Algorithm 9.2.

QA_COMPLIANCE = "classical-baseline port — depth-d DFS on MDP; observer-projection at output if compared to QA-discrete BFS forward-reachability. Not a QA-causal algorithm."

Source: (Kochenderfer 2022) Algorithms for Decision Making, MIT Press, §9.3 + Algorithm 9.2.
Transcribed from QA-MEM verbatim excerpt at
docs/theory/kochenderfer_decision_making_excerpts.md#dm-9-3-forward-search
"""

from __future__ import annotations
from typing import Callable


def forward_search(
    states: list,
    actions: list,
    transition: Callable[[object, object, object], float],
    reward: Callable[[object, object], float],
    gamma: float,
    s,
    d: int,
    terminal_U: Callable[[object], float],
) -> tuple:
    """Depth-d forward search. Returns (best_action, best_utility).

    Worst-case complexity O((|S| · |A|)^d). Per DM §9.3, set
    terminal_U(s) = 0 for plain finite-horizon planning, or use an
    offline value-function approximation for hybrid planning.
    """
    if d <= 0:
        return (None, terminal_U(s))

    def utility_at_depth_minus_1(sp):
        return forward_search(
            states, actions, transition, reward, gamma, sp, d - 1, terminal_U
        )[1]

    best_a = None
    best_u = float("-inf")
    for a in actions:
        u = reward(s, a) + gamma * sum(
            transition(s, a, sp) * utility_at_depth_minus_1(sp) for sp in states
        )
        if u > best_u:
            best_a = a
            best_u = u
    return (best_a, best_u)


if __name__ == "__main__":
    # 3-state chain; forward-search to depth 5 should match infinite-horizon optimal
    # for γ=0.9 since the chain absorbs at C
    states = ["A", "B", "C"]
    actions = ["advance", "wait"]

    def transition(s, a, sp):
        if a == "advance":
            nxt = {"A": "B", "B": "C", "C": "C"}
            return 1.0 if sp == nxt[s] else 0.0
        return 1.0 if sp == s else 0.0

    def reward(s, a):
        if s == "C":
            return 1.0
        if a == "wait":
            return -0.1
        return 0.0

    a, u = forward_search(states, actions, transition, reward, 0.9, "A", 5, lambda s: 0.0)
    print(f"From A at depth 5: best_action={a}, utility={u:.4f}")
