"""Classical value iteration — Python port of Kochenderfer 2022 DM Algorithm 7.8.

QA_COMPLIANCE = "classical-baseline port — continuous-valued utility on discrete state-action space; observer-projection at output if used to compare to QA-discrete tier-classification (cert [191]). Not a QA-causal algorithm."

Source: (Kochenderfer 2022) Algorithms for Decision Making, MIT Press, §7.5 + Algorithm 7.8.
Transcribed from QA-MEM verbatim excerpt at
docs/theory/kochenderfer_decision_making_excerpts.md#dm-7-5-value-iteration-bellman-backup
"""

from __future__ import annotations
from typing import Callable


def lookahead(
    states: list,
    transition: Callable[[object, object, object], float],
    reward: Callable[[object, object], float],
    gamma: float,
    U: dict,
    s,
    a,
) -> float:
    """Bellman lookahead for one (state, action) pair.

    Returns: R(s, a) + γ Σ_{s'} T(s' | s, a) * U(s').
    """
    return reward(s, a) + gamma * sum(
        transition(s, a, sp) * U[sp] for sp in states
    )


def value_iteration(
    states: list,
    actions: list,
    transition: Callable[[object, object, object], float],
    reward: Callable[[object, object], float],
    gamma: float,
    k_max: int,
) -> dict:
    """Value iteration via Bellman backup. Computes optimal U* via
    `U_{k+1}(s) = max_a [R(s,a) + γ Σ T(s'|s,a) U_k(s')]`.

    Returns: {state: float utility} dict after k_max iterations.
    Optimal greedy policy: π*(s) = argmax_a [R(s,a) + γ Σ T(s'|s,a) U*(s')].
    """
    U = {s: 0.0 for s in states}
    for _ in range(k_max):
        U = {
            s: max(lookahead(states, transition, reward, gamma, U, s, a) for a in actions)
            for s in states
        }
    return U


if __name__ == "__main__":
    # Same 3-state chain, but now with two actions (advance / wait)
    states = ["A", "B", "C"]
    actions = ["advance", "wait"]

    def transition(s, a, sp):
        if a == "advance":
            nxt = {"A": "B", "B": "C", "C": "C"}
            return 1.0 if sp == nxt[s] else 0.0
        else:  # wait
            return 1.0 if sp == s else 0.0

    def reward(s, a):
        if s == "C":
            return 1.0
        if a == "wait":
            return -0.1  # waiting costs
        return 0.0

    U = value_iteration(states, actions, transition, reward, 0.9, 100)
    print(f"U(A)={U['A']:.4f}, U(B)={U['B']:.4f}, U(C)={U['C']:.4f}")
    # Optimal: always advance. Expected U(C) = 10, U(B) = 9, U(A) = 8.1.
