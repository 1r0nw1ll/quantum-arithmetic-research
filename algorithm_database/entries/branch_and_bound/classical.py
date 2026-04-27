"""Classical branch-and-bound forward-search variant — Python port of Kochenderfer 2022 DM Algorithm 9.3.

QA_COMPLIANCE = "classical-baseline port — depth-d DFS with bound-pruning; firewall-rejected as causal QA on small orbit graphs (enumeration dominates); candidate at larger modulus. Not a QA-causal algorithm in v1."

Source: (Kochenderfer 2022) Algorithms for Decision Making, MIT Press, §9.4 + Algorithm 9.3.
Transcribed from QA-MEM verbatim excerpt at
docs/theory/kochenderfer_decision_making_excerpts.md#dm-9-4-branch-and-bound

The Opt §22.4 integer-programming variant requires an LP solver (e.g., GLPK / scipy.optimize.linprog) — not portable in <50 lines, so this file ports only the DM forward-search variant.
"""

from __future__ import annotations
from typing import Callable


def branch_and_bound(
    states: list,
    actions: list,
    transition: Callable[[object, object, object], float],
    reward: Callable[[object, object], float],
    gamma: float,
    s,
    d: int,
    U_lo: Callable[[object], float],
    Q_hi: Callable[[object, object], float],
) -> tuple:
    """Branch-and-bound depth-d forward search with pruning.

    Per DM Algorithm 9.3: requires lower-bound U_lo(s) on value function,
    upper-bound Q_hi(s, a) on action-value. Prunes when Q_hi(s, a) < best
    so far.
    """
    if d <= 0:
        return (None, U_lo(s))

    def utility_at_depth_minus_1(sp):
        return branch_and_bound(
            states, actions, transition, reward, gamma, sp, d - 1, U_lo, Q_hi
        )[1]

    best_a = None
    best_u = float("-inf")
    # Order actions by descending Q_hi for early-prune effectiveness
    for a in sorted(actions, key=lambda a: -Q_hi(s, a)):
        if Q_hi(s, a) < best_u:
            return (best_a, best_u)  # safe to prune; sorted order guarantees rest are worse
        u = reward(s, a) + gamma * sum(
            transition(s, a, sp) * utility_at_depth_minus_1(sp) for sp in states
        )
        if u > best_u:
            best_a = a
            best_u = u
    return (best_a, best_u)


if __name__ == "__main__":
    # 3-state chain with trivial bounds (worst case = 0, best case = 11.11)
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

    def U_lo(s):
        return 0.0

    def Q_hi(s, a):
        return 11.11  # 1/(1-0.9), loose upper bound

    a, u = branch_and_bound(states, actions, transition, reward, 0.9, "A", 5,
                             U_lo, Q_hi)
    print(f"From A at depth 5 with B&B: best_action={a}, utility={u:.4f}")
